#!/usr/bin/env python3
"""
PRECISION: GNN as feature extractor + per-drug predictor.

Strategy: use the trained GNN encoder to generate cell line and drug
embeddings that encode graph structure (PPI, TF-target, drug-target),
then train per-drug Ridge/RF models on these embeddings.

If Ridge(GNN_embeddings) > Ridge(expression), the graph adds value.

Comparison:
1. Ridge(expression):          19K raw gene features
2. RF(TF_activities):          771 TF features
3. Ridge(GNN_cell_emb):        128d graph-informed cell embeddings
4. Ridge(GNN_cell+drug_emb):   256d concatenated embeddings
5. Ridge(expression + GNN_emb): combined features

All with same cell-line hold-out split.

Run from project root:
    python scripts/12_gnn_embeddings_perdrug.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR as RESULTS_ROOT
from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gnn_emb")

RESULTS_DIR = RESULTS_ROOT / "v3_embeddings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def train_gnn_encoder(data, node_maps, drug_response, train_cells, epochs=800):
    """Train GNN and return the encoder (for embedding extraction)."""
    from src.data.splits import split_response_edges

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

    # Only use train cell lines for GNN training
    edges = split_response_edges(drug_response, node_maps, train_cells, [], DEVICE)

    model = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3,
        dropout=0.33, conv_type="sage",
        edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    logger.info("Training GNN encoder (%d params, %d epochs, train cells only)...",
                sum(p.numel() for p in model.parameters()), epochs)

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict, mp_edges, edges["train_cell"], edges["train_drug"])
        loss = F.huber_loss(pred, edges["train_auc"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 200 == 0:
            logger.info("  Epoch %d | loss=%.4f", ep, loss.item())

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        h_dict = model.encode(x_dict, mp_edges)

    cell_emb = h_dict["cell_line"].cpu().numpy()  # (n_cells, 128)
    drug_emb = h_dict["drug"].cpu().numpy()        # (n_drugs, 128)

    # Map to DataFrames with proper indices
    inv_cell = {v: k for k, v in node_maps["cell_line"].items()}
    inv_drug = {v: k for k, v in node_maps["drug"].items()}

    cell_emb_df = pd.DataFrame(
        cell_emb,
        index=[inv_cell[i] for i in range(len(inv_cell))],
        columns=[f"gnn_cell_{i}" for i in range(cell_emb.shape[1])],
    )
    drug_emb_df = pd.DataFrame(
        drug_emb,
        index=[inv_drug[i] for i in range(len(inv_drug))],
        columns=[f"gnn_drug_{i}" for i in range(drug_emb.shape[1])],
    )

    logger.info("Cell embeddings: %s, Drug embeddings: %s",
                cell_emb_df.shape, drug_emb_df.shape)
    return model, cell_emb_df, drug_emb_df, mp_edges, x_dict


def run_perdrug_comparison(features_dict, drug_response, train_cells, test_cells,
                           drug_emb_df=None, min_samples=30):
    """Run per-drug Ridge on multiple feature sets, optionally concatenating drug embeddings."""
    results = {}

    for feat_name, features in features_dict.items():
        shared = sorted(set(features.index) & set(drug_response.index))
        feat = features.loc[shared]
        resp = drug_response.loc[shared]

        train_mask = feat.index.isin(train_cells)
        test_mask = feat.index.isin(test_cells)

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            logger.warning("%s: not enough samples (train=%d, test=%d)",
                           feat_name, train_mask.sum(), test_mask.sum())
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(feat[train_mask])
        X_test = scaler.transform(feat[test_mask])

        per_drug_results = []
        all_yt, all_yp = [], []

        for drug in resp.columns:
            y_train = resp.loc[train_mask, drug].values
            y_test = resp.loc[test_mask, drug].values
            tr_ok = ~np.isnan(y_train)
            te_ok = ~np.isnan(y_test)

            if tr_ok.sum() < min_samples or te_ok.sum() < 5:
                continue

            # Optionally concat drug embedding to cell features
            if drug_emb_df is not None and drug in drug_emb_df.index:
                drug_vec = drug_emb_df.loc[drug].values.reshape(1, -1)
                drug_train = np.tile(drug_vec, (tr_ok.sum(), 1))
                drug_test = np.tile(drug_vec, (te_ok.sum(), 1))
                Xtr = np.hstack([X_train[tr_ok], drug_train])
                Xte = np.hstack([X_test[te_ok], drug_test])
            else:
                Xtr = X_train[tr_ok]
                Xte = X_test[te_ok]

            model = Ridge(alpha=100)
            model.fit(Xtr, y_train[tr_ok])
            pred = model.predict(Xte)

            pr, _ = pearsonr(y_test[te_ok], pred)
            sr, _ = spearmanr(y_test[te_ok], pred)
            per_drug_results.append({
                "drug": drug, "n_samples": te_ok.sum(),
                "pearson_r": pr, "spearman_r": sr,
                "rmse": np.sqrt(np.mean((y_test[te_ok] - pred) ** 2)),
            })
            all_yt.extend(y_test[te_ok])
            all_yp.extend(pred)

        pdf = pd.DataFrame(per_drug_results)
        valid = pdf.dropna(subset=["pearson_r"])
        global_pr, _ = pearsonr(all_yt, all_yp) if all_yt else (0, 0)

        results[feat_name] = {
            "per_drug": pdf,
            "global_pearson": global_pr,
            "median_pearson": valid["pearson_r"].median(),
            "mean_pearson": valid["pearson_r"].mean(),
            "n_drugs": len(valid),
            "pct_gt_03": (valid["pearson_r"] > 0.3).mean() * 100,
        }

        logger.info("%-35s | global=%.4f | median=%.4f | >0.3: %.0f%% | drugs=%d",
                    feat_name, global_pr, valid["pearson_r"].median(),
                    results[feat_name]["pct_gt_03"], len(valid))

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    expression = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    tf_activities = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0)

    train_cells, test_cells = cell_line_holdout_split(drug_response, test_frac=0.2, seed=SEED)

    # ── Train GNN and extract embeddings ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING GNN ENCODER")
    logger.info("=" * 60)

    data, node_maps = load_hetero_data()
    data = data.to(DEVICE)

    _, cell_emb, drug_emb, mp_edges, x_dict = train_gnn_encoder(
        data, node_maps, drug_response, train_cells
    )

    cell_emb.to_csv(RESULTS_DIR / "cell_line_embeddings.csv")
    drug_emb.to_csv(RESULTS_DIR / "drug_embeddings.csv")

    # ── Build feature sets ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PER-DRUG COMPARISON")
    logger.info("=" * 60)

    # Combined: expression + GNN embeddings
    shared_idx = sorted(set(expression.index) & set(cell_emb.index))
    combined = pd.concat([expression.loc[shared_idx], cell_emb.loc[shared_idx]], axis=1)

    features_dict = {
        "Ridge(expression_19K)": expression,
        "Ridge(TF_activities_771)": tf_activities,
        "Ridge(GNN_cell_emb_128)": cell_emb,
        "Ridge(expression+GNN_emb)": combined,
    }

    # Without drug embeddings first
    logger.info("--- Without drug embeddings ---")
    results_no_drug = run_perdrug_comparison(
        features_dict, drug_response, train_cells, test_cells
    )

    # With drug embeddings concatenated
    logger.info("\n--- With drug embeddings concatenated ---")
    results_with_drug = run_perdrug_comparison(
        {
            "Ridge(GNN_cell+drug_emb)": cell_emb,
            "Ridge(expr+GNN_cell+drug_emb)": combined,
        },
        drug_response, train_cells, test_cells, drug_emb_df=drug_emb,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)

    all_results = {**results_no_drug, **results_with_drug}
    summary = []
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["median_pearson"]):
        summary.append({
            "model": name,
            "global_pearson": r["global_pearson"],
            "median_pearson": r["median_pearson"],
            "mean_pearson": r["mean_pearson"],
            "pct_gt_03": r["pct_gt_03"],
            "n_drugs": r["n_drugs"],
        })
        r["per_drug"].to_csv(RESULTS_DIR / f"per_drug_{name.replace('/', '_')}.csv", index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "embedding_comparison.csv", index=False)

    for _, row in summary_df.iterrows():
        logger.info("%-35s | global=%.4f | median=%.4f | >0.3: %.0f%%",
                    row["model"], row["global_pearson"],
                    row["median_pearson"], row["pct_gt_03"])

    logger.info("DONE. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
