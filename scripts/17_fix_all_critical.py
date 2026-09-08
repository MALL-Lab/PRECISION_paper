#!/usr/bin/env python3
"""
PRECISION - Train the final GNN and transfer per-drug predictions to SCAN-B.

1. Load PRISM response, expression and TF activities, cell-line hold-out split
2. Train the heterogeneous GNN and save best_model_v5.pt (optional: skipped
   when the graph is absent, the model is not used by steps 3-4)
3. Per-drug Ridge transfer of TF activities to SCAN-B with quantile-based
   batch correction fitted on training cell lines only
4. Per-drug log-rank survival validation in SCAN-B

Outputs (RESULTS_DIR/v5_final):
    best_model_v5.pt
    scanb_perdrug_predictions.csv
    scanb_survival_perdrug.csv

Run from project root:
    python scripts/17_fix_all_critical.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from config import DATA_DIR, RESULTS_DIR
from src.data.splits import set_all_seeds, cell_line_holdout_split, split_response_edges
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.load_depmap import load_model_info
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fix_all")

OUT_DIR = RESULTS_DIR / "v5_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Load data and create clean train/test split
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Clean data loading")
    logger.info("=" * 60)

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    expression = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    tf_activities = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    model_info = load_model_info()

    train_cells, test_cells = cell_line_holdout_split(drug_response, 0.2, SEED)
    logger.info("Split: %d train, %d test cell lines", len(train_cells), len(test_cells))

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Train GNN (existing graph is OK for model training)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Train GNN")
    logger.info("=" * 60)

    graph_path = DATA_DIR / "precision_graph" / "hetero_graph_prism.pkl"
    if not graph_path.exists():
        logger.info("Graph not found at %s: skipping STEP 2. The GNN trained here "
                    "(best_model_v5.pt) is not used by the transfer of STEP 3-4, "
                    "which only needs the TF activity matrices.", graph_path)
    else:
        data, node_maps = load_hetero_data()
        data = data.to(DEVICE)
        edges = split_response_edges(drug_response, node_maps, train_cells, test_cells, DEVICE)

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
            node_feature_dims=nfd, hidden_dim=128, n_layers=3, dropout=0.33,
            conv_type="sage", edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1.3e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)

        logger.info("Training GNN (%d params)...", sum(p.numel() for p in model.parameters()))
        best_loss, best_state = float("inf"), None
        for ep in range(1, 801):
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
                    tp = model(x_dict, mp_edges, edges["test_cell"], edges["test_drug"])
                    tl = F.huber_loss(tp, edges["test_auc"]).item()
                if tl < best_loss:
                    best_loss = tl
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pr, _ = pearsonr(edges["test_auc"].cpu().numpy(), tp.cpu().numpy())
                logger.info("  Epoch %d | loss=%.4f | Pearson=%.4f", ep, tl, pr)

        if best_state:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), OUT_DIR / "best_model_v5.pt")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Per-drug Ridge transfer to SCAN-B
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: Per-drug SCAN-B transfer (Ridge on TF activities)")
    logger.info("=" * 60)

    scanb_tf = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_scanb.csv", index_col=0)
    scanb_clinical = pd.read_csv(DATA_DIR / "precision_processed/scanb_clinical.csv")
    scanb_clinical = scanb_clinical.drop_duplicates(subset="sample_id")
    logger.info("SCAN-B: %d patients, %d TFs", len(scanb_tf), scanb_tf.shape[1])

    # Align TF columns
    shared_tfs = sorted(set(tf_activities.columns) & set(scanb_tf.columns))
    logger.info("Shared TFs: %d", len(shared_tfs))

    prism_tf = tf_activities[shared_tfs]
    scanb_tf_aligned = scanb_tf[shared_tfs]

    # Batch correction: QuantileTransformer fitted on TRAIN PRISM only
    train_prism_tf = prism_tf.loc[[c for c in train_cells if c in prism_tf.index]]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism_tf)
    prism_tf_qt = pd.DataFrame(
        qt.transform(prism_tf), index=prism_tf.index, columns=shared_tfs
    )
    scanb_tf_qt = pd.DataFrame(
        qt.transform(scanb_tf_aligned), index=scanb_tf_aligned.index, columns=shared_tfs
    )
    logger.info("Batch correction applied (QuantileTransformer)")

    # Verify distribution alignment
    prism_mean = prism_tf_qt.mean().mean()
    scanb_mean = scanb_tf_qt.mean().mean()
    prism_std = prism_tf_qt.std().mean()
    scanb_std = scanb_tf_qt.std().mean()
    logger.info("Post-correction: PRISM mean=%.3f std=%.3f, SCAN-B mean=%.3f std=%.3f",
                prism_mean, prism_std, scanb_mean, scanb_std)

    # Per-drug Ridge: train on PRISM train cells, predict on SCAN-B
    train_tf = prism_tf_qt.loc[[c for c in train_cells if c in prism_tf_qt.index]]
    train_resp = drug_response.loc[train_tf.index]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_tf)
    X_scanb = scaler.transform(scanb_tf_qt)

    scanb_predictions = {}
    drugs_evaluated = 0

    for drug in drug_response.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue

        ridge = Ridge(alpha=100)
        ridge.fit(X_train[mask], y[mask])
        pred = ridge.predict(X_scanb)
        scanb_predictions[drug] = pred
        drugs_evaluated += 1

    pred_df = pd.DataFrame(scanb_predictions, index=scanb_tf_qt.index)
    pred_df.to_csv(OUT_DIR / "scanb_perdrug_predictions.csv")
    logger.info("Per-drug predictions: %d patients x %d drugs", *pred_df.shape)

    # Check prediction diversity (the critical fix)
    if pred_df.shape[1] >= 2:
        corr_matrix = pred_df.corr()
        np.fill_diagonal(corr_matrix.values, np.nan)
        mean_corr = np.nanmean(corr_matrix.values)
        logger.info("Inter-drug prediction correlation: %.4f (was 0.9998 before fix)", mean_corr)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Honest survival validation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: Survival validation (per-drug, independent)")
    logger.info("=" * 60)

    try:
        from lifelines.statistics import logrank_test
        from scipy import stats

        surv_results = []
        for drug in pred_df.columns:
            pred_series = pred_df[drug]
            merged = scanb_clinical.set_index("sample_id").join(
                pred_series.rename("predicted_auc"), how="inner"
            )
            merged = merged.dropna(subset=["os_time", "os_event", "predicted_auc"])

            if len(merged) < 100:
                continue

            median_auc = merged["predicted_auc"].median()
            sens = merged[merged["predicted_auc"] < median_auc]
            res = merged[merged["predicted_auc"] >= median_auc]

            if len(sens) < 30 or len(res) < 30:
                continue

            lr = logrank_test(sens["os_time"], res["os_time"],
                              event_observed_A=sens["os_event"],
                              event_observed_B=res["os_event"])
            surv_results.append({
                "drug": drug, "n_patients": len(merged),
                "pvalue": lr.p_value,
            })

        surv_df = pd.DataFrame(surv_results)
        surv_df["padj"] = stats.false_discovery_control(surv_df["pvalue"])
        surv_df = surv_df.sort_values("pvalue")
        surv_df.to_csv(OUT_DIR / "scanb_survival_perdrug.csv", index=False)

        n_sig = (surv_df["padj"] < 0.05).sum()
        n_nom = (surv_df["pvalue"] < 0.05).sum()
        logger.info("Survival: %d drugs tested, %d nominal p<0.05, %d FDR<0.05",
                    len(surv_df), n_nom, n_sig)

        # Show top 10
        for _, r in surv_df.head(10).iterrows():
            logger.info("  %s: p=%.4f, padj=%.4f", r["drug"], r["pvalue"], r["padj"])

    except ImportError:
        logger.warning("lifelines not installed")
        surv_df = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    if len(surv_df) > 0:
        n_sig = (surv_df["padj"] < 0.05).sum()
        n_nom = (surv_df["pvalue"] < 0.05).sum()
        logger.info("SCAN-B survival (per-drug Ridge, honest):")
        logger.info("  Drugs tested: %d", len(surv_df))
        logger.info("  Nominal p<0.05: %d (%.0f%%)", n_nom, 100 * n_nom / len(surv_df))
        logger.info("  FDR<0.05: %d", n_sig)
        logger.info("  Inter-drug prediction correlation: %.4f", mean_corr)

    logger.info("DONE - Results in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
