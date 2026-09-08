#!/usr/bin/env python3
"""
PRECISION - IG multi-seed stability.

Runs Integrated Gradients with the canonical mean-train baseline on each
of the three GNN models trained by script 34 (best_sage_seed{42,123,456}.pt).
Compares TF rankings across seeds to determine whether IG is robust to
GNN training variance (unlike ablation, which is not, see the script 34 report).

Scope per seed: 12 TNBC cells x 11 drugs, n_steps=50 (same as script 32).
Time per seed: ~5 min on RTX 5060. Total ~15-20 min.

Outputs in results/v4_xai/:
  tf_importance_ig_seed{42,123,456}.csv
  tf_importance_ig_multi_seed_stability.csv

Run from project root:
    python scripts/35_ig_multi_seed.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import RESULTS_DIR as RESULTS_ROOT
from src.data.drug_names import normalize_drug_columns
from src.data.load_depmap import load_model_info
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

# Reuse helpers from script 32
_spec = importlib.util.spec_from_file_location(
    "script_32", Path(__file__).parent / "32_xai_ig.py",
)
_mod32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod32)
GNNCellLineWrapper = _mod32.GNNCellLineWrapper
attribute_pair = _mod32.attribute_pair
identify_tnbc_cells = _mod32.identify_tnbc_cells
load_tf_names = _mod32.load_tf_names

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ig_multi_seed")

RESULTS_DIR = RESULTS_ROOT / "v4_xai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 50
SEEDS = [42, 123, 456]

# Candidate set: the seven drugs of Table 5 by default, or the extended set of
# Supplementary Table S1 with --candidate-set extended (outputs suffixed).
CANDIDATES = list(_mod32.CANDIDATE_DRUGS)
CONTROLS = list(_mod32.CONTROL_DRUGS)
DRUGS = CANDIDATES + CONTROLS
OUTPUT_SUFFIX = ""


def load_model_for_seed(seed: int, data, node_maps):
    """Instantiate SAGE architecture and load weights from best_sage_seed{N}.pt."""
    x_dict = {nt: data[nt].x for nt in data.node_types}
    nfd = {nt: data[nt].x.shape[1] for nt in data.node_types}
    n_drugs = len(node_maps["drug"])

    rk = ("cell_line", "responds_to", "drug")
    mp_edges = {}
    for et in data.edge_types:
        if et == rk:
            continue
        mp_edges[et] = data[et].edge_index
        s, r, d = et
        mp_edges[(d, f"rev_{r}", s)] = data[et].edge_index.flip(0)

    sage = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3,
        dropout=0.33, conv_type="sage",
        edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
    ).to(DEVICE)

    state = torch.load(RESULTS_DIR / f"best_sage_seed{seed}.pt",
                        map_location=DEVICE, weights_only=True)
    sage.load_state_dict(state)
    sage.eval()
    return sage, x_dict, mp_edges


def run_ig_for_seed(seed: int, sage, x_dict, mp_edges, cell_map, drug_map,
                     tnbc_cell_ids, tf_names, train_cells):
    """IG sweep for this seed, mean-train baseline."""
    cell_feat = x_dict["cell_line"]
    train_cell_idx = torch.tensor(
        [cell_map[c] for c in train_cells if c in cell_map],
        device=DEVICE, dtype=torch.long,
    )
    baseline_row = cell_feat[train_cell_idx].mean(dim=0).detach()

    records = []
    n_tfs = len(tf_names)
    total = len(tnbc_cell_ids) * len(DRUGS)
    t0 = time.time()
    idx = 0

    for cell_id in tnbc_cell_ids:
        cell_idx = cell_map[cell_id]
        cell_row = cell_feat[cell_idx].detach()
        for drug in DRUGS:
            drug_idx = drug_map[drug]
            wrapper = GNNCellLineWrapper(
                sage, mp_edges, x_dict, cell_idx, drug_idx, DEVICE,
            )
            attr = attribute_pair(wrapper, cell_row, baseline_row,
                                  n_steps=N_STEPS)
            attr_np = attr.cpu().numpy()[:n_tfs]
            for k in range(n_tfs):
                records.append({
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
                logger.info("  [seed=%d] IG %d/%d elapsed %.0fs eta %.0fs",
                            seed, idx, total, elapsed, eta)
    return pd.DataFrame.from_records(records)


def aggregate_importance(full: pd.DataFrame) -> pd.DataFrame:
    work = full.copy()
    work["abs_attr"] = work["attribution"].abs()
    imp = work.groupby("tf")["abs_attr"].mean().reset_index(name="importance_mean")
    imp = imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="IG multi-seed stability")
    ap.add_argument("--candidate-set", choices=["table5", "extended"], default="table5")
    return ap.parse_args()


def main() -> int:
    global CANDIDATES, DRUGS, OUTPUT_SUFFIX
    args = _parse_args()
    CANDIDATES = _mod32.resolve_candidate_set(args.candidate_set)
    DRUGS = CANDIDATES + CONTROLS
    OUTPUT_SUFFIX = "" if args.candidate_set == "table5" else f"_{args.candidate_set}"
    logger.info("Candidate set %s: %d drugs", args.candidate_set, len(DRUGS))
    set_all_seeds(42)
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    model_info = load_model_info()
    data, node_maps = load_hetero_data()
    data = data.to(DEVICE)

    # Get TNBC cells (must match training split, but we use all TNBC here;
    # script 32 confirmed these 12 are in the graph)
    cell_map = node_maps["cell_line"]
    drug_map = node_maps["drug"]
    CANDIDATES = _mod32.match_graph_drug_names(CANDIDATES, drug_map)
    DRUGS = CANDIDATES + CONTROLS
    tnbc_cell_ids = identify_tnbc_cells(model_info, drug_response, cell_map)
    tf_names = load_tf_names()

    per_seed_imp = {}

    for seed in SEEDS:
        logger.info("=" * 70)
        logger.info("IG for SEED = %d", seed)
        logger.info("=" * 70)
        # Train/test split for this seed (just to use train_cells for baseline)
        train_cells, _ = cell_line_holdout_split(
            drug_response, test_frac=0.2, seed=seed,
        )
        sage, x_dict, mp_edges = load_model_for_seed(seed, data, node_maps)
        full = run_ig_for_seed(
            seed, sage, x_dict, mp_edges, cell_map, drug_map,
            tnbc_cell_ids, tf_names, train_cells,
        )
        imp = aggregate_importance(full)
        imp.to_csv(RESULTS_DIR / f"tf_importance_ig_seed{seed}{OUTPUT_SUFFIX}.csv", index=False)
        logger.info("  [seed=%d] wrote, top10: %s",
                    seed, imp.head(10)["tf"].tolist())
        per_seed_imp[seed] = imp

    # Stability
    logger.info("=" * 70)
    logger.info("STABILITY ACROSS SEEDS")
    logger.info("=" * 70)

    stab = per_seed_imp[SEEDS[0]][["tf"]].copy()
    for seed in SEEDS:
        col = per_seed_imp[seed][["tf", "rank", "importance_mean"]].rename(
            columns={"rank": f"rank_{seed}",
                     "importance_mean": f"importance_mean_{seed}"},
        )
        stab = stab.merge(col, on="tf", how="left")
    rank_cols = [f"rank_{s}" for s in SEEDS]
    stab["mean_rank"] = stab[rank_cols].mean(axis=1)
    stab["in_top20_all_seeds"] = (stab[rank_cols] <= 20).all(axis=1)
    stab = stab.sort_values("mean_rank").reset_index(drop=True)
    stab.to_csv(RESULTS_DIR / f"tf_importance_ig_multi_seed_stability{OUTPUT_SUFFIX}.csv",
                 index=False)

    n_all = int(stab["in_top20_all_seeds"].sum())
    logger.info("TFs in top-20 across ALL 3 seeds: %d", n_all)
    logger.info("  %s", stab[stab["in_top20_all_seeds"]]["tf"].tolist())

    for i, a in enumerate(SEEDS):
        for b in SEEDS[i+1:]:
            ta = set(per_seed_imp[a].head(20)["tf"])
            tb = set(per_seed_imp[b].head(20)["tf"])
            m = per_seed_imp[a][["tf", "rank"]].merge(
                per_seed_imp[b][["tf", "rank"]], on="tf",
                suffixes=(f"_{a}", f"_{b}"),
            )
            rho, _ = spearmanr(m[f"rank_{a}"], m[f"rank_{b}"])
            logger.info("  seeds %d vs %d: top20 overlap=%d/20  Spearman=%.3f",
                        a, b, len(ta & tb), rho)

    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
