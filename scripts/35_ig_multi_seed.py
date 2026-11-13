#!/usr/bin/env python3
"""
PRECISION - IG multi-seed stability.

Runs Integrated Gradients with the canonical mean-train baseline on the SAGE
models of seeds 42, 123 and 456 trained by scripts/57_train_gnns.py
(sage_split{S}_init{S}_full, loaded through the loader of script 32, which
also checks that every drug was trained). Each seed sets both the cell-line
split and the initialization, and the baseline of each model is the mean
feature row over the training cell lines of its own split. Seed 42 is the
canonical model of script 32, so tf_importance_ig_seed42.csv and
tf_importance_ig_global.csv of script 32 come from the same checkpoint and
the same attribution and must rank the TFs in the same order. Compares the TF
rankings across seeds to determine whether IG is robust to GNN training
variance.

Scope per seed: 12 TNBC cells x 11 drugs, n_steps=50 (same as script 32).
Time per seed: ~5 min on RTX 5060. Total ~15-20 min.

Outputs in results/v4_xai/:
  tf_importance_ig_seed{42,123,456}.csv       tf, importance_mean, rank
  tf_importance_ig_multi_seed_stability.csv   tf, rank_S and importance_mean_S
                                              for each seed S, mean_rank,
                                              in_top20_all_seeds
  tf_importance_ig_seed_pairs.csv             one row per pair of seeds:
                                              seed_a, seed_b, top20_overlap
                                              (TFs shared by the two top 20),
                                              spearman_rho (Spearman correlation
                                              of the ranks over all TFs), n_tfs
With --candidate-set extended the candidates are the 16 kinase or HDAC
inhibitors of Supplementary Data 1 and every output is suffixed _extended.

Run from project root:
    python scripts/35_ig_multi_seed.py [--candidate-set table5|extended]
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
from src.data.splits import set_all_seeds
from src.models import training as T

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
DEVICE = T.device()
N_STEPS = 50
# Split and initialization seed of each SAGE model. Seed 42 is the canonical model.
SEEDS = [42, 123, 456]
TOP_K = 20

# Candidate set: the seven drugs of Table 5 by default, or the extended set of
# Supplementary Data 1 with --candidate-set extended (outputs suffixed).
CANDIDATES = list(_mod32.CANDIDATE_DRUGS)
CONTROLS = list(_mod32.CONTROL_DRUGS)
DRUGS = CANDIDATES + CONTROLS
OUTPUT_SUFFIX = ""


def run_ig_for_seed(seed: int, sage, x_dict, mp_edges, cell_map, drug_map,
                     tnbc_cell_ids, tf_names, train_cells):
    """IG sweep for this seed, mean-train baseline."""
    cell_feat = x_dict["cell_line"]
    train_cell_idx = torch.tensor(
        [cell_map[c] for c in train_cells if c in cell_map],
        device=DEVICE, dtype=torch.long,
    )
    baseline_row = cell_feat[train_cell_idx].mean(dim=0).detach()
    logger.info("  [seed=%d] baseline: mean over %d training cell lines of split %d",
                seed, len(train_cell_idx), seed)

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


def seed_pair_table(per_seed_imp: dict, seeds: list, top_k: int = TOP_K) -> pd.DataFrame:
    """Agreement between the TF rankings of each pair of seeds: the number of
    TFs shared by the two top-k lists and the Spearman correlation of the
    ranks over every TF ranked by both."""
    rows = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            ta = set(per_seed_imp[a].head(top_k)["tf"])
            tb = set(per_seed_imp[b].head(top_k)["tf"])
            m = per_seed_imp[a][["tf", "rank"]].merge(
                per_seed_imp[b][["tf", "rank"]], on="tf",
                suffixes=(f"_{a}", f"_{b}"),
            )
            rho, _ = spearmanr(m[f"rank_{a}"], m[f"rank_{b}"])
            rows.append({"seed_a": a, "seed_b": b,
                         f"top{top_k}_overlap": len(ta & tb),
                         "spearman_rho": float(rho), "n_tfs": len(m)})
    return pd.DataFrame(rows)


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(42)
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    model_info = load_model_info()
    graph = T.load_graph(DEVICE)
    x_dict = graph.x_dict

    # The same 12 TNBC cell lines as script 32 for every seed, whether they
    # fall in the training or the test cell lines of that seed's split.
    cell_map = graph.node_maps["cell_line"]
    drug_map = graph.node_maps["drug"]
    CANDIDATES = _mod32.match_graph_drug_names(CANDIDATES, drug_map)
    DRUGS = CANDIDATES + _mod32.match_graph_drug_names(CONTROLS, drug_map)
    tnbc_cell_ids = identify_tnbc_cells(model_info, drug_response, cell_map)
    tf_names = load_tf_names()

    per_seed_imp = {}

    for seed in SEEDS:
        logger.info("=" * 70)
        logger.info("IG for SEED = %d", seed)
        logger.info("=" * 70)
        # Model of this split and init seed, and the training cell lines of
        # its split for the mean-train baseline.
        sage, mp_edges, train_cells = _mod32.load_ig_model(graph, seed, seed, drug_response)
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

    pairs = seed_pair_table(per_seed_imp, SEEDS)
    pairs.to_csv(RESULTS_DIR / f"tf_importance_ig_seed_pairs{OUTPUT_SUFFIX}.csv", index=False)
    for _, r in pairs.iterrows():
        logger.info("  seeds %d vs %d: top%d overlap=%d/%d  Spearman=%.3f (%d TFs)",
                    r["seed_a"], r["seed_b"], TOP_K, r[f"top{TOP_K}_overlap"], TOP_K,
                    r["spearman_rho"], r["n_tfs"])

    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
