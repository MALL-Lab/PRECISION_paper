#!/usr/bin/env python3
"""
PRECISION - IG baseline robustness for XAI.

Runs Integrated Gradients (Captum) against 3 different baselines to test
stability of the TF importance ranking:

  1. mean      - mean TF feature across 381 TRAIN cells (canonical, same as
                 script 32).
  2. zero      - zeros baseline (standard Captum default).
  3. random    - 3 independent random Gaussian baselines (seeds 1, 2, 3) with
                 per-dim mean/std matched to the train distribution; reported
                 as an average over the 3 random draws.

All baselines operate on the same trained SAGE GNN (best_sage_v4.pt, SEED=42
at training) and the same 12 TNBC cell lines x 11 drugs (7 candidates + 4 controls)
scope as script 32. We use a single random seed (42) for the bootstrap
sampling within each baseline to isolate the baseline effect.

Purpose: the internal methods review asked for IG stability across
baselines: without this, the IG top-10 (ZEB1/RELA/HOXB9/...) could be an
artefact of the mean-train baseline choice.

Outputs in results/v4_xai/:
  ig_baseline_mean.csv      - tf, importance_mean, rank
  ig_baseline_zero.csv      - tf, importance_mean, rank
  ig_baseline_random.csv    - tf, importance_mean, std (across 3 draws), rank
  ig_baseline_stability.csv - tf, rank_mean, rank_zero, rank_random,
                              in_top20_all, mean_across_baselines

Run from project root:
    python scripts/33_ig_baseline_robustness.py
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

from config import RESULTS_DIR as RESULTS_ROOT
from src.data.drug_names import normalize_drug_columns
from src.data.load_depmap import load_model_info
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

# Reuse the wrapper + attribute_pair from script 32.
sys.path.insert(0, str(Path(__file__).parent))
_script_32 = Path(__file__).parent / "32_xai_ig.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("script_32", _script_32)
_mod32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod32)

GNNCellLineWrapper = _mod32.GNNCellLineWrapper
attribute_pair = _mod32.attribute_pair
load_sage_and_data = _mod32.load_sage_and_data
identify_tnbc_cells = _mod32.identify_tnbc_cells
load_tf_names = _mod32.load_tf_names

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ig_robustness")

RESULTS_DIR = RESULTS_ROOT / "v4_xai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42
N_STEPS = 50

# Candidate set: the seven drugs of Table 5 by default, or the extended set of
# Supplementary Table S1 with --candidate-set extended (outputs suffixed).
CANDIDATE_DRUGS = list(_mod32.CANDIDATE_DRUGS)
CONTROL_DRUGS = list(_mod32.CONTROL_DRUGS)
ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS
OUTPUT_SUFFIX = ""


def build_baseline(name: str, cell_feat: torch.Tensor,
                   train_cell_idx: torch.Tensor,
                   random_seed: int = 0) -> torch.Tensor:
    """Construct the IG baseline vector of shape [D_cell]."""
    n_feat = cell_feat.shape[1]
    if name == "mean":
        return cell_feat[train_cell_idx].mean(dim=0).detach()
    if name == "zero":
        return torch.zeros(n_feat, device=cell_feat.device, dtype=cell_feat.dtype)
    if name == "random":
        train_cells = cell_feat[train_cell_idx]
        mu = train_cells.mean(dim=0)
        sigma = train_cells.std(dim=0).clamp_min(1e-6)
        g = torch.Generator(device="cpu").manual_seed(random_seed)
        noise = torch.randn(n_feat, generator=g).to(cell_feat.device).to(cell_feat.dtype)
        return (mu + noise * sigma).detach()
    raise ValueError(f"Unknown baseline: {name}")


def run_ig_with_baseline(sage, x_dict, mp_edges, cell_map, drug_map,
                          tnbc_cell_ids, drugs, tf_names, baseline_row,
                          tag: str) -> pd.DataFrame:
    """Sweep 12 TNBC x 11 drugs IG attributions with a given baseline.
    Returns long-form DataFrame (cell, drug, tf, attribution)."""
    cell_feat = x_dict["cell_line"]
    n_tfs = len(tf_names)
    records = []
    total = len(tnbc_cell_ids) * len(drugs)
    idx = 0
    t0 = time.time()
    set_all_seeds(BASE_SEED)

    for cell_id in tnbc_cell_ids:
        cell_idx = cell_map[cell_id]
        cell_row = cell_feat[cell_idx].detach()
        for drug in drugs:
            drug_idx = drug_map[drug]
            wrapper = GNNCellLineWrapper(
                sage, mp_edges, x_dict, cell_idx, drug_idx, DEVICE,
            )
            attr = attribute_pair(wrapper, cell_row, baseline_row,
                                  n_steps=N_STEPS)
            attr_np = attr.cpu().numpy()[:n_tfs]
            for k in range(n_tfs):
                records.append({
                    "baseline": tag,
                    "cell_depmap_id": cell_id,
                    "drug": drug,
                    "tf": tf_names[k],
                    "attribution": float(attr_np[k]),
                })
            idx += 1
            if idx % 20 == 0 or idx == total:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-6)
                eta = (total - idx) / max(rate, 1e-6)
                logger.info("  [%s] %d/%d elapsed %.0fs eta %.0fs",
                            tag, idx, total, elapsed, eta)
    return pd.DataFrame.from_records(records)


def aggregate_importance(full: pd.DataFrame) -> pd.DataFrame:
    """Global importance = mean |attribution| across cells & drugs per TF."""
    work = full.copy()
    work["abs_attr"] = work["attribution"].abs()
    imp = work.groupby("tf")["abs_attr"].mean().reset_index(
        name="importance_mean"
    )
    imp = imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="IG baseline robustness")
    ap.add_argument("--candidate-set", choices=["table5", "extended"], default="table5")
    return ap.parse_args()


def main() -> int:
    global CANDIDATE_DRUGS, ALL_DRUGS, OUTPUT_SUFFIX
    args = _parse_args()
    CANDIDATE_DRUGS = _mod32.resolve_candidate_set(args.candidate_set)
    ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS
    OUTPUT_SUFFIX = "" if args.candidate_set == "table5" else f"_{args.candidate_set}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(BASE_SEED)

    logger.info("Loading SAGE + graph...")
    (sage, x_dict, mp_edges, node_maps, model_info,
     drug_response, train_cells) = load_sage_and_data()

    cell_map = node_maps["cell_line"]
    drug_map = node_maps["drug"]
    CANDIDATE_DRUGS = _mod32.match_graph_drug_names(CANDIDATE_DRUGS, drug_map)
    ALL_DRUGS = CANDIDATE_DRUGS + CONTROL_DRUGS

    tnbc_cell_ids = identify_tnbc_cells(model_info, drug_response, cell_map)
    logger.info("TNBC cells: %d  drugs: %d", len(tnbc_cell_ids), len(ALL_DRUGS))

    tf_names = load_tf_names()
    cell_feat = x_dict["cell_line"]
    train_cell_idx = torch.tensor(
        [cell_map[c] for c in train_cells if c in cell_map],
        device=DEVICE, dtype=torch.long,
    )

    rankings = {}

    # 1. Mean baseline
    logger.info("=" * 60)
    logger.info("Baseline 1/3: mean-train")
    logger.info("=" * 60)
    bl = build_baseline("mean", cell_feat, train_cell_idx)
    full_mean = run_ig_with_baseline(
        sage, x_dict, mp_edges, cell_map, drug_map,
        tnbc_cell_ids, ALL_DRUGS, tf_names, bl, tag="mean",
    )
    imp_mean = aggregate_importance(full_mean)
    imp_mean.to_csv(RESULTS_DIR / f"ig_baseline_mean{OUTPUT_SUFFIX}.csv", index=False)
    logger.info("  wrote ig_baseline_mean.csv (top3: %s)",
                imp_mean.head(3)["tf"].tolist())
    rankings["mean"] = imp_mean

    # 2. Zero baseline
    logger.info("=" * 60)
    logger.info("Baseline 2/3: zero")
    logger.info("=" * 60)
    bl = build_baseline("zero", cell_feat, train_cell_idx)
    full_zero = run_ig_with_baseline(
        sage, x_dict, mp_edges, cell_map, drug_map,
        tnbc_cell_ids, ALL_DRUGS, tf_names, bl, tag="zero",
    )
    imp_zero = aggregate_importance(full_zero)
    imp_zero.to_csv(RESULTS_DIR / f"ig_baseline_zero{OUTPUT_SUFFIX}.csv", index=False)
    logger.info("  wrote ig_baseline_zero.csv (top3: %s)",
                imp_zero.head(3)["tf"].tolist())
    rankings["zero"] = imp_zero

    # 3. Random baseline (average over 3 draws, seeds 1, 2, 3)
    logger.info("=" * 60)
    logger.info("Baseline 3/3: random (3 draws)")
    logger.info("=" * 60)
    rand_aggregates = []
    for rseed in [1, 2, 3]:
        logger.info("  random draw seed=%d", rseed)
        bl = build_baseline("random", cell_feat, train_cell_idx,
                             random_seed=rseed)
        full_r = run_ig_with_baseline(
            sage, x_dict, mp_edges, cell_map, drug_map,
            tnbc_cell_ids, ALL_DRUGS, tf_names, bl,
            tag=f"random_{rseed}",
        )
        imp_r = aggregate_importance(full_r)
        imp_r["random_seed"] = rseed
        rand_aggregates.append(imp_r)
    rand_all = pd.concat(rand_aggregates, ignore_index=True)
    imp_random = rand_all.groupby("tf").agg(
        importance_mean=("importance_mean", "mean"),
        importance_std=("importance_mean", "std"),
    ).reset_index()
    imp_random = imp_random.sort_values(
        "importance_mean", ascending=False,
    ).reset_index(drop=True)
    imp_random["rank"] = np.arange(1, len(imp_random) + 1)
    imp_random.to_csv(RESULTS_DIR / f"ig_baseline_random{OUTPUT_SUFFIX}.csv", index=False)
    logger.info("  wrote ig_baseline_random.csv (top3: %s)",
                imp_random.head(3)["tf"].tolist())
    rankings["random"] = imp_random

    # Stability table
    logger.info("=" * 60)
    logger.info("STABILITY: top 20 overlap across baselines")
    logger.info("=" * 60)
    stability = imp_mean[["tf"]].merge(
        imp_mean[["tf", "rank"]].rename(columns={"rank": "rank_mean"}),
        on="tf",
    ).merge(
        imp_zero[["tf", "rank"]].rename(columns={"rank": "rank_zero"}),
        on="tf",
    ).merge(
        imp_random[["tf", "rank"]].rename(columns={"rank": "rank_random"}),
        on="tf",
    )
    stability["in_top20_mean"] = stability["rank_mean"] <= 20
    stability["in_top20_zero"] = stability["rank_zero"] <= 20
    stability["in_top20_random"] = stability["rank_random"] <= 20
    stability["in_top20_all"] = (
        stability["in_top20_mean"]
        & stability["in_top20_zero"]
        & stability["in_top20_random"]
    )
    stability["mean_rank_across_baselines"] = stability[
        ["rank_mean", "rank_zero", "rank_random"]
    ].mean(axis=1)
    stability = stability.sort_values(
        "mean_rank_across_baselines",
    ).reset_index(drop=True)
    stability.to_csv(RESULTS_DIR / f"ig_baseline_stability{OUTPUT_SUFFIX}.csv", index=False)

    top20_all = stability[stability["in_top20_all"]]
    logger.info("TFs in top-20 across ALL 3 baselines: %d", len(top20_all))
    logger.info("  %s", top20_all["tf"].tolist())

    # Pairwise top-20 overlap
    for a, b in [("mean", "zero"), ("mean", "random"), ("zero", "random")]:
        tops_a = set(rankings[a].head(20)["tf"])
        tops_b = set(rankings[b].head(20)["tf"])
        overlap = len(tops_a & tops_b)
        logger.info("  top-20 overlap %s-vs-%s: %d/20", a, b, overlap)

    # Spearman across rankings (over all 771 TFs)
    from scipy.stats import spearmanr
    for a, b in [("mean", "zero"), ("mean", "random"), ("zero", "random")]:
        merged = rankings[a][["tf", "rank"]].merge(
            rankings[b][["tf", "rank"]], on="tf", suffixes=(f"_{a}", f"_{b}"),
        )
        rho, pv = spearmanr(merged[f"rank_{a}"], merged[f"rank_{b}"])
        logger.info("  Spearman rank %s-vs-%s: rho=%.3f p=%.2e",
                    a, b, rho, pv)

    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
