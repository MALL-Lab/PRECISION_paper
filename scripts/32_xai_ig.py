#!/usr/bin/env python3
"""
PRECISION: TF importance via Integrated Gradients (Captum).

Replaces the zero-out feature ablation baseline (Script 13 / src/xai/explainer_v2.py)
with Integrated Gradients (Sundararajan et al. 2017) on the trained SAGE GNN
(best_sage_v4.pt). IG is axiomatic (completeness, linearity, sensitivity), gives
SIGNED attributions, and aligns with standard XAI practice for deep-graph models.

Scope:
    - 12 TNBC cell lines (ModelSubtypeFeatures contains "TNBC")
    - 11 drugs: the 7 Table 5 candidates + 4 positive controls
      Candidates: osimertinib, saracatinib, erlotinib, brigatinib, pelitinib,
            entinostat, trametinib
      Controls: paclitaxel, docetaxel, epirubicin, olaparib
    - 3 seeds: 42, 123, 456
    - n_steps = 50 (Captum default)
    - Baseline: mean TF feature over 381 TRAIN cells (fixed, reproducible)

Total (cell, drug, seed) triplets: 12 * 11 * 3 = 396 IG attributions
(each triplet = 50 forward+backward passes = ~20k GNN calls).

Outputs in results/v4_xai/:
    tf_importance_ig_global.csv:   mean |attr| across seeds x cells x drugs
    tf_drug_ig_signed.csv:         top 20 TFs x 11 drugs, signed
    ig_full_attributions.csv:      full long-form table for reproducibility
    ig_gnn_vs_rf_comparison.csv:   IG vs RF permutation importance merge

Sign convention (IMPORTANT): target = predicted AUC. Positive attribution
    means "higher TF activity -> higher predicted AUC" -> resistance driver.
    Negative attribution means "higher TF activity -> lower AUC" -> sensitivity
    driver. Documented in the CSV headers.

Run from project root:
    python scripts/32_xai_ig.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import PAPER_RESULTS, DATA_DIR, RESULTS_DIR as RESULTS_ROOT
from src.data.drug_names import normalize_drug_columns
from src.data.load_depmap import load_model_info
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xai_ig")

RESULTS_DIR = RESULTS_ROOT / "v4_xai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_SEED = 42
SEEDS = [42, 123, 456]
N_STEPS = 50

# The seven candidates of Table 5 (main analysis, candidate set "table5").
CANDIDATE_DRUGS = [
    "osimertinib", "saracatinib", "erlotinib", "brigatinib", "pelitinib",
    "entinostat", "trametinib",
]
CONTROL_DRUGS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib"]
ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS
OUTPUT_SUFFIX = ""
SUPP_TABLE_S1 = PAPER_RESULTS / "supp_table_S1_candidate_selection.csv"


def resolve_candidate_set(name: str) -> list:
    """Candidate drugs for a named set: 'table5' (the seven drugs of Table 5)
    or 'extended' (every kinase or HDAC inhibitor passing the stability and
    development-stage filters of the selection rule, Supplementary Table S1,
    i.e. the same rule without the rank cutoff)."""
    if name == "table5":
        return list(CANDIDATE_DRUGS)
    if name == "extended":
        supp = pd.read_csv(SUPP_TABLE_S1)
        ext = supp[supp["pass_stability"] & supp["pass_class"] & supp["pass_phase"]]
        return ext.sort_values("rank")["drug"].tolist()
    raise ValueError(f"unknown candidate set: {name}")


def match_graph_drug_names(names, drug_map):
    """Map drug names to the keys of the graph drug map, case-insensitively.
    The graph keeps the original PRISM spelling (e.g. 'BVD-523') while the Cox
    tables and the selection rule use lowercase names."""
    lower = {k.lower(): k for k in drug_map}
    return [lower.get(str(n).lower(), n) for n in names]


def configure_candidate_set(name: str) -> None:
    """Select the candidate set for this run and the suffix of its outputs."""
    global CANDIDATE_DRUGS, ALL_DRUGS, OUTPUT_SUFFIX
    CANDIDATE_DRUGS = resolve_candidate_set(name)
    ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS
    OUTPUT_SUFFIX = "" if name == "table5" else f"_{name}"


# =============================================================================
# Captum wrapper
# =============================================================================
class GNNCellLineWrapper(nn.Module):
    """Wrap the HeteroGNN so Captum can attribute the scalar AUC prediction
    with respect to a single cell-line feature row.

    The wrapper takes one [1, D_cell] tensor (the cell row under attribution)
    and returns the predicted AUC for a fixed (cell_idx, drug_idx) pair. All
    other cell-line rows, gene/TF/drug features and message-passing edges stay
    frozen. This avoids the ambiguity of attributing over the whole
    [N_cells, D_cell] matrix (where Captum would assign baselines to cell rows
    that are NOT the target, producing noise).
    """

    def __init__(self, gnn, mp_edges, x_base, cell_idx: int, drug_idx: int,
                 device: torch.device):
        super().__init__()
        self.gnn = gnn
        self.mp_edges = mp_edges
        # Clone so we can overwrite in forward without mutating caller's tensors.
        self.x_base = {k: v.to(device).clone() for k, v in x_base.items()}
        self.cell_idx = int(cell_idx)
        self.drug_idx = int(drug_idx)
        self.device = device

        # Precompute the index tensors (single pair).
        self.cell_tensor = torch.tensor([self.cell_idx], device=device,
                                         dtype=torch.long)
        self.drug_tensor = torch.tensor([self.drug_idx], device=device,
                                         dtype=torch.long)

    def forward(self, cell_row: torch.Tensor) -> torch.Tensor:
        """cell_row: [B, D_cell], batch dim B=n_steps in IG.

        We substitute the target cell row with cell_row[b] for each step b and
        forward the GNN. Captum will compute gradient vs cell_row.
        """
        if cell_row.dim() == 1:
            cell_row = cell_row.unsqueeze(0)
        B = cell_row.shape[0]

        preds = []
        for b in range(B):
            x = {k: v for k, v in self.x_base.items()}
            # Replace ONLY the target cell-line row.
            cell_x = self.x_base["cell_line"].clone()
            cell_x[self.cell_idx] = cell_row[b]
            x["cell_line"] = cell_x
            out = self.gnn(x, self.mp_edges, self.cell_tensor, self.drug_tensor)
            preds.append(out)
        return torch.cat(preds, dim=0)  # [B]


# =============================================================================
# IG attribution (Captum with autograd fallback)
# =============================================================================
def integrated_gradients_manual(wrapper: GNNCellLineWrapper,
                                 input_x: torch.Tensor,
                                 baseline: torch.Tensor,
                                 n_steps: int = 50) -> torch.Tensor:
    """Pure torch.autograd fallback in case Captum struggles with HeteroGNN.

    Riemann mid-point approximation of IG:
        attr = (x - x') * mean_{alpha in steps} grad f(x' + alpha*(x-x'))

    Returns a 1-D tensor of same length as input_x.
    """
    alphas = torch.linspace(0.0, 1.0, n_steps, device=input_x.device)
    input_x = input_x.detach()
    baseline = baseline.detach()
    diff = input_x - baseline

    total_grad = torch.zeros_like(input_x)
    for alpha in alphas:
        interp = (baseline + alpha * diff).clone().requires_grad_(True)
        out = wrapper(interp.unsqueeze(0))  # [1]
        grad = torch.autograd.grad(out.sum(), interp)[0]
        total_grad = total_grad + grad / n_steps

    return diff * total_grad


def attribute_pair(wrapper, cell_row, baseline_row, n_steps, use_captum=True):
    """Return IG attribution for one (cell, drug) pair. Tries Captum first,
    falls back to manual autograd if Captum raises."""
    if use_captum:
        try:
            from captum.attr import IntegratedGradients
            ig = IntegratedGradients(wrapper)
            attr = ig.attribute(
                cell_row.unsqueeze(0),
                baselines=baseline_row.unsqueeze(0),
                n_steps=n_steps,
            )
            return attr.squeeze(0).detach()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Captum IG failed (%s), falling back to manual IG",
                           exc)
    return integrated_gradients_manual(
        wrapper, cell_row, baseline_row, n_steps=n_steps,
    )


# =============================================================================
# Model + data loading
# =============================================================================
def load_sage_and_data():
    """Rebuild SAGE architecture, load best_sage_v4.pt, return everything the
    wrapper needs (mp_edges, x_dict, node_maps, model_info, train cells, drug
    response)."""
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    model_info = load_model_info()
    data, node_maps = load_hetero_data()
    data = data.to(DEVICE)

    train_cells, test_cells = cell_line_holdout_split(
        drug_response, test_frac=0.2, seed=BASE_SEED,
    )

    rk = ("cell_line", "responds_to", "drug")
    mp_edges = {}
    for et in data.edge_types:
        if et == rk:
            continue
        mp_edges[et] = data[et].edge_index
        s, r, d = et
        mp_edges[(d, f"rev_{r}", s)] = data[et].edge_index.flip(0)

    x_dict = {nt: data[nt].x for nt in data.node_types}
    nfd = {nt: data[nt].x.shape[1] for nt in data.node_types}
    n_drugs = len(node_maps["drug"])

    sage = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3,
        dropout=0.33, conv_type="sage",
        edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
    ).to(DEVICE)

    state_dict = torch.load(
        RESULTS_DIR / "best_sage_v4.pt", map_location=DEVICE, weights_only=True,
    )
    sage.load_state_dict(state_dict)
    sage.eval()

    logger.info("Loaded SAGE from %s (n_layers=3, hidden=128, n_drugs=%d)",
                RESULTS_DIR / "best_sage_v4.pt", n_drugs)

    return sage, x_dict, mp_edges, node_maps, model_info, drug_response, train_cells


def identify_tnbc_cells(model_info: pd.DataFrame,
                         drug_response: pd.DataFrame,
                         cell_map: dict) -> list[str]:
    """Return TNBC depmap_ids present in both drug_response and the graph."""
    if "ModelSubtypeFeatures" in model_info.columns:
        mask = model_info["ModelSubtypeFeatures"].fillna("").str.contains(
            "TNBC", case=False,
        )
    elif "LegacySubSubtype" in model_info.columns:
        mask = model_info["LegacySubSubtype"].fillna("") == "ERneg_HER2neg"
    else:
        mask = model_info["OncotreeSubtype"].fillna("").str.contains(
            "TNBC|Triple Negative", case=False,
        )
    candidates = model_info.index[mask].tolist()
    return [c for c in candidates
            if c in drug_response.index and c in cell_map]


def load_tf_names() -> list[str]:
    return pd.read_csv(
        DATA_DIR / "precision_processed" / "tf_activities_all_prism.csv",
        index_col=0, nrows=0,
    ).columns.tolist()


# =============================================================================
# Main IG sweep
# =============================================================================
def run_ig_sweep(sage, x_dict, mp_edges, node_maps, tnbc_cell_ids,
                  drugs, train_cells, cell_map, drug_map) -> pd.DataFrame:
    """Run IG for all (seed, cell, drug) triplets. Returns the long-form
    DataFrame with one row per (seed, cell, drug, tf)."""
    tf_names = load_tf_names()
    n_tfs = len(tf_names)  # 771
    cell_feat = x_dict["cell_line"]  # [476, 1271]

    # Baseline = mean TF feature across TRAIN cells (only TF dims vary, rest 0).
    # We still need the full 1271-dim baseline for IG over the whole row;
    # the non-TF dims (gene expression, 500 dims) also get a baseline of
    # mean-train so they stay approximately equal to the cell's own values
    # and generate small residual attributions.
    train_cell_idx = torch.tensor(
        [cell_map[c] for c in train_cells if c in cell_map],
        device=DEVICE, dtype=torch.long,
    )
    baseline_row = cell_feat[train_cell_idx].mean(dim=0).detach()
    logger.info("Baseline row: shape=%s, |baseline_row| mean=%.4f",
                tuple(baseline_row.shape), baseline_row.abs().mean().item())

    records = []
    total = len(SEEDS) * len(tnbc_cell_ids) * len(drugs)
    idx = 0
    t0 = time.time()

    for seed in SEEDS:
        set_all_seeds(seed)

        for cell_id in tnbc_cell_ids:
            cell_idx = cell_map[cell_id]
            cell_row = cell_feat[cell_idx].detach()

            for drug_name in drugs:
                drug_idx = drug_map[drug_name]
                wrapper = GNNCellLineWrapper(
                    sage, mp_edges, x_dict, cell_idx, drug_idx, DEVICE,
                )

                attr = attribute_pair(
                    wrapper, cell_row, baseline_row, n_steps=N_STEPS,
                )
                attr_np = attr.cpu().numpy()[:n_tfs]  # keep only TF dims

                for k in range(n_tfs):
                    records.append({
                        "seed": seed,
                        "cell_depmap_id": cell_id,
                        "drug": drug_name,
                        "tf_index": k,
                        "tf": tf_names[k],
                        "attribution": float(attr_np[k]),
                    })

                idx += 1
                if idx % 20 == 0 or idx == total:
                    elapsed = time.time() - t0
                    rate = idx / max(elapsed, 1e-6)
                    eta = (total - idx) / max(rate, 1e-6)
                    logger.info(
                        "  IG %d/%d (seed=%d, cell=%s, drug=%s) - "
                        "elapsed %.0fs, eta %.0fs",
                        idx, total, seed, cell_id, drug_name, elapsed, eta,
                    )

    full = pd.DataFrame.from_records(records)
    logger.info("Full attributions: %d rows (expected %d = %d seeds * %d cells"
                " * %d drugs * %d TFs)",
                len(full), total * n_tfs, len(SEEDS), len(tnbc_cell_ids),
                len(drugs), n_tfs)
    return full


# =============================================================================
# Aggregations
# =============================================================================
def compute_global_importance(full: pd.DataFrame) -> pd.DataFrame:
    """Mean |attribution| across seeds x cells x drugs per TF, with bootstrap
    CI over all (seed, cell, drug) groups.

    importance_mean  = mean of |attr| over all rows per TF
    importance_std   = std of per-(seed, cell, drug) mean |attr|
    importance_ci_*  = 2.5 / 97.5 percentile of per-(seed, cell, drug) mean |attr|
    """
    work = full.copy()
    work["abs_attr"] = work["attribution"].abs()

    # Per (seed, cell, drug, tf) we already have a single attribution.
    # Per (seed, cell, drug) importance of a TF = |attr| directly.
    # Aggregate mean and distribution stats per TF.
    mean_abs = work.groupby("tf")["abs_attr"].agg(
        importance_mean="mean",
        importance_std="std",
        importance_ci_low=lambda s: float(np.percentile(s, 2.5)),
        importance_ci_high=lambda s: float(np.percentile(s, 97.5)),
    ).reset_index()

    mean_abs = mean_abs.sort_values(
        "importance_mean", ascending=False,
    ).reset_index(drop=True)
    mean_abs["rank"] = np.arange(1, len(mean_abs) + 1)
    return mean_abs


def compute_signed_matrix(full: pd.DataFrame, top_tfs: list[str],
                           drugs: list[str]) -> pd.DataFrame:
    """Matrix top_tfs x drugs with mean SIGNED attribution across (seed, cell)."""
    sub = full[full["tf"].isin(top_tfs)]
    mat = sub.groupby(["tf", "drug"])["attribution"].mean().unstack("drug")

    # Re-order and enforce presence of all expected columns.
    mat = mat.reindex(index=top_tfs, columns=drugs)
    mat.index.name = "tf"
    return mat.reset_index()


def compute_rf_overlap(global_imp: pd.DataFrame,
                        top_k: int = 20) -> pd.DataFrame:
    """Merge IG importance with RF permutation importance; flag top-k overlap."""
    rf_path = RESULTS_ROOT / "v6_strengthen" / "rf_permutation_importance.csv"
    if not rf_path.exists():
        logger.warning("RF importance file not found at %s, skipping overlap",
                       rf_path)
        return pd.DataFrame()

    rf = pd.read_csv(rf_path)
    rf = rf[["tf", "rf_importance"]].rename(
        columns={"rf_importance": "rf_importance_perm"},
    )

    merged = global_imp[["tf", "importance_mean", "rank"]].merge(
        rf, on="tf", how="outer",
    )
    merged = merged.rename(columns={"importance_mean": "importance_ig"})

    # Top-k flags
    ig_top = set(global_imp.nlargest(top_k, "importance_mean")["tf"])
    rf_top = set(rf.nlargest(top_k, "rf_importance_perm")["tf"])
    merged["in_ig_top20"] = merged["tf"].isin(ig_top)
    merged["in_rf_top20"] = merged["tf"].isin(rf_top)

    merged = merged.sort_values("importance_ig", ascending=False,
                                  na_position="last").reset_index(drop=True)

    overlap = len(ig_top & rf_top)
    logger.info("IG top-%d vs RF top-%d overlap: %d TFs", top_k, top_k, overlap)
    if overlap < top_k:
        ig_only = sorted(ig_top - rf_top)
        logger.info("  IG-only top: %s", ig_only[:10])
    return merged


# =============================================================================
# Main
# =============================================================================
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Integrated Gradients TF attributions")
    ap.add_argument("--candidate-set", choices=["table5", "extended"], default="table5",
                    help="table5: the seven candidates of Table 5 (default). extended: the 16 "
                         "kinase/HDAC inhibitors of Supplementary Table S1, outputs suffixed _extended")
    return ap.parse_args()


def main() -> int:
    global CANDIDATE_DRUGS, ALL_DRUGS
    args = _parse_args()
    configure_candidate_set(args.candidate_set)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(BASE_SEED)

    logger.info("DEVICE=%s, captum available=%s",
                DEVICE, "yes" if _captum_available() else "no")

    # ── Load everything ──────────────────────────────────────────────────
    sage, x_dict, mp_edges, node_maps, model_info, drug_response, train_cells \
        = load_sage_and_data()

    cell_map = node_maps["cell_line"]
    drug_map = node_maps["drug"]
    CANDIDATE_DRUGS = match_graph_drug_names(CANDIDATE_DRUGS, drug_map)
    ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS

    # ── Identify TNBC cells + validate drugs ────────────────────────────
    tnbc_cell_ids = identify_tnbc_cells(model_info, drug_response, cell_map)
    logger.info("TNBC cell lines for IG: %d", len(tnbc_cell_ids))

    missing = [d for d in ALL_DRUGS if d not in drug_map]
    if missing:
        raise RuntimeError(f"Drugs not in graph: {missing}")
    logger.info("Drugs for IG (%s set): %d (%d candidates + %d controls)",
                args.candidate_set, len(ALL_DRUGS), len(CANDIDATE_DRUGS), len(CONTROL_DRUGS))

    # ── Run IG sweep ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INTEGRATED GRADIENTS: %d seeds x %d cells x %d drugs",
                len(SEEDS), len(tnbc_cell_ids), len(ALL_DRUGS))
    logger.info("=" * 60)

    full = run_ig_sweep(
        sage, x_dict, mp_edges, node_maps,
        tnbc_cell_ids, ALL_DRUGS, train_cells, cell_map, drug_map,
    )

    # ── Sanity checks ───────────────────────────────────────────────────
    n_nan = full["attribution"].isna().sum()
    if n_nan > 0:
        raise RuntimeError(f"Found {n_nan} NaN attributions")
    logger.info("Sanity: %d rows, 0 NaN, attr range [%.4e, %.4e]",
                len(full), full["attribution"].min(), full["attribution"].max())

    # ── Save full attributions ──────────────────────────────────────────
    full_path = RESULTS_DIR / f"ig_full_attributions{OUTPUT_SUFFIX}.csv"
    full.to_csv(full_path, index=False)
    logger.info("Saved %s (%d rows)", full_path, len(full))

    # ── Global importance ───────────────────────────────────────────────
    global_imp = compute_global_importance(full)
    global_path = RESULTS_DIR / f"tf_importance_ig_global{OUTPUT_SUFFIX}.csv"
    _write_with_header(
        global_path, global_imp,
        header_comment=(
            "# IG global importance: mean |attribution| across seeds x "
            f"TNBC cells x drugs ({len(SEEDS)} x {len(tnbc_cell_ids)} x {len(ALL_DRUGS)} = "
            f"{len(SEEDS) * len(tnbc_cell_ids) * len(ALL_DRUGS)} samples per TF).\n"
            "# Non-signed (magnitude). For signed per-drug see "
            "tf_drug_ig_signed.csv.\n"
        ),
    )
    logger.info("Saved %s (top 5: %s)", global_path,
                ", ".join(global_imp.head(5)["tf"].tolist()))

    # ── Signed top 20 x drugs ───────────────────────────────────────────
    top20 = global_imp.head(20)["tf"].tolist()
    signed = compute_signed_matrix(full, top20, ALL_DRUGS)
    signed_path = RESULTS_DIR / f"tf_drug_ig_signed{OUTPUT_SUFFIX}.csv"
    _write_with_header(
        signed_path, signed,
        header_comment=(
            "# Signed IG attribution: mean across seeds x TNBC cells per "
            "(TF, drug). Rows = top 20 TFs by global IG importance.\n"
            "# Sign: target=predicted AUC. POSITIVE attribution -> higher TF "
            "activity raises predicted AUC -> resistance driver. NEGATIVE -> "
            "sensitivity driver.\n"
        ),
    )
    logger.info("Saved %s (%d rows x %d cols)",
                signed_path, signed.shape[0], signed.shape[1])

    # ── IG vs RF overlap ────────────────────────────────────────────────
    merged = compute_rf_overlap(global_imp)
    if not merged.empty:
        merged_path = RESULTS_DIR / f"ig_gnn_vs_rf_comparison{OUTPUT_SUFFIX}.csv"
        merged.to_csv(merged_path, index=False)
        logger.info("Saved %s (%d TFs)", merged_path, len(merged))

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TOP 10 TFs by IG global importance:")
    for _, r in global_imp.head(10).iterrows():
        logger.info("  #%d %-12s |attr|=%.4e  CI95=[%.4e, %.4e]",
                    r["rank"], r["tf"], r["importance_mean"],
                    r["importance_ci_low"], r["importance_ci_high"])
    logger.info("DONE. Results in %s/", RESULTS_DIR)
    return 0


def _captum_available() -> bool:
    try:
        import captum  # noqa: F401
        return True
    except ImportError:
        return False


def _write_with_header(path: Path, df: pd.DataFrame, header_comment: str):
    """Write CSV prepending a header comment. Pandas re-read with
    comment="#" keeps this compatible with normal loaders."""
    with open(path, "w", newline="") as f:
        f.write(header_comment)
        df.to_csv(f, index=False)


if __name__ == "__main__":
    sys.exit(main())
