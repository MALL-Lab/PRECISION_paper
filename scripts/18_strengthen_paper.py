#!/usr/bin/env python3
"""
PRECISION: Three tests to strengthen the paper.

Test 1: GNN XAI TFs vs RF feature importance: are they different/better?
Test 2: Ridge per-drug with GNN embeddings in SCAN-B survival transfer
Test 3: Subtype-stratified analysis in SCAN-B (Basal vs LumA vs LumB)

Only Test 3 feeds the paper (test3_subtype_summary.csv, test3_survival_*.csv).
Test 1 depends on the legacy ablation importance file
(results/v4_xai/tf_importance_bootstrap1000.csv) and is skipped with a
warning when that file is absent.

Run from project root:
    python scripts/18_strengthen_paper.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.graph.hetero_data import load_hetero_data
from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("strengthen")

RESULTS = RESULTS_DIR / "v6_strengthen"
DATA = DATA_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def test1_xai_vs_shap():
    """Compare GNN XAI TF importance vs RF feature importance."""
    logger.info("=" * 60)
    logger.info("TEST 1: GNN XAI TFs vs RF feature importance")
    logger.info("=" * 60)

    # GNN XAI importance from the legacy ablation bootstrap (n=1000).
    # Not part of the paper: skip the whole test when the file is absent.
    gnn_imp_path = RESULTS_DIR / "v4_xai" / "tf_importance_bootstrap1000.csv"
    if not gnn_imp_path.exists():
        logger.warning("Test 1 skipped: %s not found (legacy ablation output)", gnn_imp_path)
        return None
    gnn_imp = pd.read_csv(gnn_imp_path)
    gnn_top50 = set(gnn_imp.head(50)["tf"])
    gnn_top20 = set(gnn_imp.head(20)["tf"])

    # RF feature importance
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_activities = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    train_cells, test_cells = cell_line_holdout_split(drug_response, 0.2, SEED)

    shared = sorted(set(tf_activities.index) & set(drug_response.index))
    tf = tf_activities.loc[shared]
    resp = drug_response.loc[shared]
    train_mask = tf.index.isin(train_cells)

    # Train RF on all drugs jointly (mean importance across drugs)
    all_importances = np.zeros(tf.shape[1])
    n_drugs = 0

    for drug in resp.columns:
        y = resp[drug].values
        mask = train_mask & ~np.isnan(y)
        if mask.sum() < 30:
            continue
        rf = RandomForestRegressor(n_estimators=50, max_depth=15,
                                   min_samples_leaf=5, n_jobs=-1, random_state=SEED)
        rf.fit(tf[mask].values, y[mask])
        all_importances += rf.feature_importances_
        n_drugs += 1

    all_importances /= n_drugs
    rf_imp = pd.DataFrame({"tf": tf.columns, "rf_importance": all_importances})
    rf_imp = rf_imp.sort_values("rf_importance", ascending=False)
    rf_top50 = set(rf_imp.head(50)["tf"])
    rf_top20 = set(rf_imp.head(20)["tf"])

    # Compare
    overlap_50 = len(gnn_top50 & rf_top50)
    overlap_20 = len(gnn_top20 & rf_top20)
    jaccard_50 = overlap_50 / len(gnn_top50 | rf_top50)
    jaccard_20 = overlap_20 / len(gnn_top20 | rf_top20)

    # Rank correlation
    merged = gnn_imp[["tf", "importance_mean"]].merge(rf_imp, on="tf")
    rank_corr, _ = spearmanr(merged["importance_mean"], merged["rf_importance"])

    gnn_unique_20 = gnn_top20 - rf_top20
    rf_unique_20 = rf_top20 - gnn_top20

    logger.info("Top 50 overlap: %d/50 (Jaccard=%.3f)", overlap_50, jaccard_50)
    logger.info("Top 20 overlap: %d/20 (Jaccard=%.3f)", overlap_20, jaccard_20)
    logger.info("Rank correlation (Spearman): %.4f", rank_corr)
    logger.info("GNN-unique top 20 TFs: %s", sorted(gnn_unique_20))
    logger.info("RF-unique top 20 TFs: %s", sorted(rf_unique_20))

    # Save
    comparison = merged.sort_values("importance_mean", ascending=False)
    comparison.to_csv(RESULTS / "test1_xai_vs_rf_importance.csv", index=False)
    rf_imp.to_csv(RESULTS / "test1_rf_importance.csv", index=False)

    return {
        "overlap_20": overlap_20, "overlap_50": overlap_50,
        "jaccard_20": jaccard_20, "jaccard_50": jaccard_50,
        "rank_corr": rank_corr,
        "gnn_unique_20": sorted(gnn_unique_20),
        "rf_unique_20": sorted(rf_unique_20),
    }


def test2_gnn_embeddings_transfer():
    """Test if GNN embeddings improve per-drug Ridge transfer to SCAN-B."""
    logger.info("=" * 60)
    logger.info("TEST 2: GNN embeddings in SCAN-B survival transfer")
    logger.info("=" * 60)

    from lifelines.statistics import logrank_test
    from scipy import stats

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_activities = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    cell_emb = pd.read_csv(RESULTS_DIR / "v3_embeddings" / "cell_line_embeddings.csv", index_col=0)
    scanb_tf = pd.read_csv(DATA / "precision_processed/tf_activities_scanb.csv", index_col=0)
    scanb_clinical = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    scanb_clinical = scanb_clinical.drop_duplicates(subset="sample_id")

    train_cells, test_cells = cell_line_holdout_split(drug_response, 0.2, SEED)

    shared_tfs = sorted(set(tf_activities.columns) & set(scanb_tf.columns))

    # Batch correction
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    prism_tf_qt = pd.DataFrame(qt.fit_transform(tf_activities[shared_tfs]),
                               index=tf_activities.index, columns=shared_tfs)
    scanb_tf_qt = pd.DataFrame(qt.transform(scanb_tf[shared_tfs]),
                               index=scanb_tf.index, columns=shared_tfs)

    # Combined features: TF + GNN embeddings for PRISM
    shared_cells = sorted(set(prism_tf_qt.index) & set(cell_emb.index))
    prism_combined = pd.concat([prism_tf_qt.loc[shared_cells], cell_emb.loc[shared_cells]], axis=1)

    # For SCAN-B: TF activities + zeros for GNN emb (patients don't have embeddings)
    scanb_combined = pd.DataFrame(
        np.hstack([scanb_tf_qt.values,
                   np.zeros((len(scanb_tf_qt), cell_emb.shape[1]))]),
        index=scanb_tf_qt.index,
        columns=list(shared_tfs) + list(cell_emb.columns),
    )

    # Per-drug Ridge: TF only vs TF+GNN emb
    configs = {
        "TF_only": (prism_tf_qt, scanb_tf_qt),
        "TF_plus_GNN_emb": (prism_combined, scanb_combined),
    }

    results = {}
    for config_name, (prism_feat, scanb_feat) in configs.items():
        train_feat = prism_feat.loc[[c for c in train_cells if c in prism_feat.index]]
        train_resp = drug_response.loc[train_feat.index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat)
        X_scanb = scaler.transform(scanb_feat)

        predictions = {}
        for drug in drug_response.columns:
            y = train_resp[drug].values
            mask = ~np.isnan(y)
            if mask.sum() < 30:
                continue
            ridge = Ridge(alpha=100)
            ridge.fit(X_train[mask], y[mask])
            predictions[drug] = ridge.predict(X_scanb)

        pred_df = pd.DataFrame(predictions, index=scanb_feat.index)

        # Survival test on shared drugs
        n_sig = 0
        n_tested = 0
        pvalues = []
        for drug in pred_df.columns:
            pred = pred_df[drug]
            merged = scanb_clinical.set_index("sample_id").join(
                pred.rename("auc"), how="inner"
            ).dropna(subset=["os_time", "os_event", "auc"])

            if len(merged) < 100:
                continue

            median_auc = merged["auc"].median()
            sens = merged[merged["auc"] < median_auc]
            res = merged[merged["auc"] >= median_auc]
            if len(sens) < 30 or len(res) < 30:
                continue

            lr = logrank_test(sens["os_time"], res["os_time"],
                              event_observed_A=sens["os_event"],
                              event_observed_B=res["os_event"])
            pvalues.append(lr.p_value)
            n_tested += 1

        padj = stats.false_discovery_control(pvalues) if pvalues else []
        n_sig = sum(p < 0.05 for p in padj)

        logger.info("%s: %d tested, %d FDR<0.05 (%.0f%%)",
                    config_name, n_tested, n_sig, 100 * n_sig / max(n_tested, 1))
        results[config_name] = {"n_tested": n_tested, "n_sig": n_sig,
                                "pct_sig": 100 * n_sig / max(n_tested, 1)}

    pd.DataFrame(results).T.to_csv(RESULTS / "test2_gnn_emb_transfer.csv")
    return results


def test3_subtype_analysis():
    """Subtype-stratified survival analysis in SCAN-B."""
    logger.info("=" * 60)
    logger.info("TEST 3: Subtype-stratified SCAN-B survival")
    logger.info("=" * 60)

    from lifelines.statistics import logrank_test
    from scipy import stats

    pred_df = pd.read_csv(RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv", index_col=0)
    clinical = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clinical = clinical.drop_duplicates(subset="sample_id")

    subtypes = {"Basal": "Basal", "LumA": "LumA", "LumB": "LumB", "All": None}
    results = {}

    for subtype_name, pam50_value in subtypes.items():
        if pam50_value:
            sub_clinical = clinical[clinical["pam50"] == pam50_value]
        else:
            sub_clinical = clinical

        n_sig = 0
        n_tested = 0
        pvalues = []
        top_drugs = []

        for drug in pred_df.columns:
            pred = pred_df[drug]
            merged = sub_clinical.set_index("sample_id").join(
                pred.rename("auc"), how="inner"
            ).dropna(subset=["os_time", "os_event", "auc"])

            if len(merged) < 50:
                continue

            median_auc = merged["auc"].median()
            sens = merged[merged["auc"] < median_auc]
            res = merged[merged["auc"] >= median_auc]
            if len(sens) < 15 or len(res) < 15:
                continue

            lr = logrank_test(sens["os_time"], res["os_time"],
                              event_observed_A=sens["os_event"],
                              event_observed_B=res["os_event"])
            pvalues.append(lr.p_value)
            top_drugs.append(drug)
            n_tested += 1

        if pvalues:
            padj = stats.false_discovery_control(pvalues)
            n_sig = sum(p < 0.05 for p in padj)

            # Top 5 drugs for this subtype
            drug_p = pd.DataFrame({"drug": top_drugs, "pvalue": pvalues,
                                   "padj": padj}).sort_values("pvalue")
        else:
            n_sig = 0
            drug_p = pd.DataFrame()

        logger.info("%s (n=%d): %d tested, %d FDR<0.05 (%.0f%%)",
                    subtype_name, len(sub_clinical), n_tested, n_sig,
                    100 * n_sig / max(n_tested, 1))
        if len(drug_p) > 0:
            for _, r in drug_p.head(5).iterrows():
                logger.info("  %s: p=%.2e, padj=%.2e", r["drug"], r["pvalue"], r["padj"])

        results[subtype_name] = {
            "n_patients": len(sub_clinical),
            "n_tested": n_tested, "n_sig": n_sig,
            "pct_sig": 100 * n_sig / max(n_tested, 1),
        }
        if len(drug_p) > 0:
            drug_p.to_csv(RESULTS / f"test3_survival_{subtype_name}.csv", index=False)

    pd.DataFrame(results).T.to_csv(RESULTS / "test3_subtype_summary.csv")

    # Key question: are the significant drugs DIFFERENT across subtypes?
    if all(Path(RESULTS / f"test3_survival_{s}.csv").exists() for s in ["Basal", "LumA"]):
        basal = pd.read_csv(RESULTS / "test3_survival_Basal.csv")
        luma = pd.read_csv(RESULTS / "test3_survival_LumA.csv")
        basal_sig = set(basal[basal["padj"] < 0.05]["drug"])
        luma_sig = set(luma[luma["padj"] < 0.05]["drug"])

        basal_only = basal_sig - luma_sig
        luma_only = luma_sig - basal_sig
        shared = basal_sig & luma_sig

        logger.info("\nSubtype-specific drugs:")
        logger.info("  Basal-only significant: %d", len(basal_only))
        logger.info("  LumA-only significant: %d", len(luma_only))
        logger.info("  Shared significant: %d", len(shared))

        if len(basal_only) > 0:
            logger.info("  Top Basal-only: %s", list(basal_only)[:5])
        if len(luma_only) > 0:
            logger.info("  Top LumA-only: %s", list(luma_only)[:5])

    return results


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    t1 = test1_xai_vs_shap()
    t2 = test2_gnn_embeddings_transfer()
    t3 = test3_subtype_analysis()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if t1 is None:
        logger.info("Test 1 (XAI vs RF): skipped (legacy bootstrap importance file not found)")
    else:
        logger.info("Test 1 (XAI vs RF): overlap=%d/20, rank_corr=%.3f",
                    t1["overlap_20"], t1["rank_corr"])
        logger.info("  GNN-unique TFs: %s", t1["gnn_unique_20"])
    logger.info("Test 2 (GNN emb transfer): TF_only=%d sig, TF+GNN=%d sig",
                t2["TF_only"]["n_sig"], t2["TF_plus_GNN_emb"]["n_sig"])
    logger.info("Test 3 (subtypes):")
    for k, v in t3.items():
        logger.info("  %s: %d/%d FDR<0.05 (%.0f%%)", k, v["n_sig"], v["n_tested"], v["pct_sig"])

    logger.info("DONE")


if __name__ == "__main__":
    main()
