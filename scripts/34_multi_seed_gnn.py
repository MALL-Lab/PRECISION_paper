#!/usr/bin/env python3
"""
PRECISION - Multi-seed SAGE training + ablation bootstrap for XAI stability.

Trains the SAGE GNN three times with seeds 42, 123, 456. For each seed:
  - cell-line hold-out split regenerated with the seed
  - 800 epochs Adam + cosine annealing, Huber loss
  - ablation-bootstrap TF importance (n=1000) on the test set
  - saves `results/v4_xai/best_sage_seed{N}.pt` +
    `results/v4_xai/tf_importance_bootstrap1000_seed{N}.csv`

At the end produces `results/v4_xai/tf_importance_multi_seed_stability.csv`
with columns: tf, rank_42, rank_123, rank_456, mean_rank,
in_top20_all_seeds, importance_mean_42, importance_mean_123,
importance_mean_456.

Purpose: the internal methods review asked for GNN training
variance across seeds: without this, the ablation top-5 (TP53, MYC, SPI1,
SRSF2, DNMT3A) could be an artefact of SEED=42.

Run from project root:
    python scripts/34_multi_seed_gnn.py
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
import torch.nn.functional as F
from scipy.stats import pearsonr

from config import RESULTS_DIR as RESULTS_ROOT
from src.data.drug_names import normalize_drug_columns
from src.data.load_depmap import load_model_info
from src.data.load_drug_response import load_prism_response
from src.data.splits import (cell_line_holdout_split, set_all_seeds,
                              split_response_edges)
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse
from src.xai.explainer_v2 import compute_tf_importance_bootstrap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_seed")

RESULTS_DIR = RESULTS_ROOT / "v4_xai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]
EPOCHS = 800


def train_and_ablate(seed: int, drug_response, model_info, data, node_maps):
    """Train SAGE with the given seed and compute ablation-bootstrap importance.

    Returns (model_state, tf_importance_df, global_pearson_test)."""
    set_all_seeds(seed)

    train_cells, test_cells = cell_line_holdout_split(
        drug_response, test_frac=0.2, seed=seed,
    )
    edges = split_response_edges(
        drug_response, node_maps, train_cells, test_cells, DEVICE,
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

    model = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3,
        dropout=0.33, conv_type="sage",
        edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.3e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5,
    )

    best_loss = float("inf")
    best_state = None
    for ep in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict, mp_edges, edges["train_cell"], edges["train_drug"])
        loss = F.huber_loss(pred, edges["train_auc"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                tp = model(x_dict, mp_edges,
                           edges["test_cell"], edges["test_drug"])
                tl = F.huber_loss(tp, edges["test_auc"]).item()
                pr, _ = pearsonr(edges["test_auc"].cpu().numpy(),
                                  tp.cpu().numpy())
            if tl < best_loss:
                best_loss = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info("  [seed=%d] ep %d | loss=%.4f | Pearson=%.4f",
                        seed, ep, tl, pr)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test Pearson for record
    model.eval()
    with torch.no_grad():
        tp = model(x_dict, mp_edges,
                   edges["test_cell"], edges["test_drug"])
        final_pr, _ = pearsonr(edges["test_auc"].cpu().numpy(),
                                tp.cpu().numpy())

    # Ablation bootstrap
    logger.info("  [seed=%d] running ablation bootstrap n=1000 on %d test edges",
                seed, edges["test_cell"].shape[0])
    tf_imp = compute_tf_importance_bootstrap(
        model, x_dict, mp_edges,
        edges["test_cell"], edges["test_drug"], edges["test_auc"],
        n_bootstrap=1000, seed=seed,
    )

    return model.state_dict(), tf_imp, final_pr


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    drug_response = normalize_drug_columns(
        load_prism_response(use_processed=True),
    )
    model_info = load_model_info()
    data, node_maps = load_hetero_data()
    data = data.to(DEVICE)

    per_seed_imp = {}
    per_seed_pearson = {}

    for seed in SEEDS:
        logger.info("=" * 70)
        logger.info("TRAINING SAGE with seed = %d", seed)
        logger.info("=" * 70)
        t0 = time.time()
        state, tf_imp, test_pr = train_and_ablate(
            seed, drug_response, model_info, data, node_maps,
        )

        torch.save(state, RESULTS_DIR / f"best_sage_seed{seed}.pt")
        out_csv = RESULTS_DIR / f"tf_importance_bootstrap1000_seed{seed}.csv"
        tf_imp.to_csv(out_csv, index=False)

        n_sig = int(tf_imp["significant"].sum())
        top1 = tf_imp.head(1)["tf"].iloc[0]
        top5 = tf_imp.head(5)["tf"].tolist()
        elapsed = time.time() - t0
        logger.info("  [seed=%d] test Pearson=%.4f  sig TFs=%d/771  "
                    "top1=%s  top5=%s  elapsed=%.0fs",
                    seed, test_pr, n_sig, top1, top5, elapsed)

        per_seed_imp[seed] = tf_imp
        per_seed_pearson[seed] = test_pr

    # Stability table
    logger.info("=" * 70)
    logger.info("BUILDING stability table across seeds")
    logger.info("=" * 70)

    # rank each TF by importance_mean per seed
    per_seed_ranked = {}
    for seed, imp in per_seed_imp.items():
        r = imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        per_seed_ranked[seed] = r

    stab = per_seed_ranked[SEEDS[0]][["tf"]].copy()
    for seed in SEEDS:
        col = per_seed_ranked[seed][["tf", "rank", "importance_mean"]].rename(
            columns={"rank": f"rank_{seed}",
                     "importance_mean": f"importance_mean_{seed}"},
        )
        stab = stab.merge(col, on="tf", how="left")

    rank_cols = [f"rank_{s}" for s in SEEDS]
    stab["mean_rank"] = stab[rank_cols].mean(axis=1)
    stab["max_rank"] = stab[rank_cols].max(axis=1)
    stab["in_top20_all_seeds"] = (stab[rank_cols] <= 20).all(axis=1)
    stab = stab.sort_values("mean_rank").reset_index(drop=True)
    stab.to_csv(RESULTS_DIR / "tf_importance_multi_seed_stability.csv",
                 index=False)

    n_top20_all = int(stab["in_top20_all_seeds"].sum())
    logger.info("TFs in top-20 across ALL 3 seeds: %d", n_top20_all)
    logger.info("  %s", stab[stab["in_top20_all_seeds"]]["tf"].tolist())

    # Pairwise top-20 overlap + Spearman
    from scipy.stats import spearmanr
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i+1:]:
            ta = set(per_seed_ranked[a].head(20)["tf"])
            tb = set(per_seed_ranked[b].head(20)["tf"])
            overlap = len(ta & tb)
            merged = per_seed_ranked[a][["tf", "rank"]].merge(
                per_seed_ranked[b][["tf", "rank"]], on="tf",
                suffixes=(f"_{a}", f"_{b}"),
            )
            rho, _ = spearmanr(merged[f"rank_{a}"], merged[f"rank_{b}"])
            logger.info("  seeds %d vs %d: top20 overlap=%d/20  "
                        "Spearman=%.3f",
                        a, b, overlap, rho)

    # Test Pearson summary
    prs = list(per_seed_pearson.values())
    logger.info("Test Pearson across seeds: mean=%.4f  std=%.4f  "
                "(values: %s)",
                float(np.mean(prs)), float(np.std(prs)),
                [f"{v:.4f}" for v in prs])

    logger.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
