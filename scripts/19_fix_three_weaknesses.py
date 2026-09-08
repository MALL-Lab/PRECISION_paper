#!/usr/bin/env python3
"""
PRECISION: Fix three remaining weaknesses in parallel.

D1-A: GNN-top-50 TFs vs RF-top-50 vs all in SCAN-B survival transfer
D2-B: METABRIC independent validation
D2-C: Basal nominal enrichment (permutation test)
D3-A: Penalized Cox (Ridge) for regularized HRs

Only D3-A feeds the paper (d3_penalized_cox.csv). D1-A depends on the legacy
ablation importance files (results/v4_xai/tf_importance_bootstrap1000.csv and
results/v6_strengthen/test1_rf_importance.csv) and is skipped with a warning
when either file is absent.

Run from project root:
    python scripts/19_fix_three_weaknesses.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest, fisher_exact
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fix_weak")

RESULTS = RESULTS_DIR / "v7_weaknesses"
DATA = DATA_DIR
SEED = 42


def weakness1_tf_subset_transfer():
    """D1-A: Does using GNN-selected TFs improve SCAN-B transfer?"""
    logger.info("=" * 60)
    logger.info("WEAKNESS 1: GNN TF selection vs RF TF selection in transfer")
    logger.info("=" * 60)

    from lifelines.statistics import logrank_test

    # TF subsets come from the legacy ablation bootstrap (n=1000) and from the
    # RF importance written by 18_strengthen_paper.py Test 1. Neither feeds the
    # paper: skip the whole test when either file is absent.
    gnn_imp_path = RESULTS_DIR / "v4_xai" / "tf_importance_bootstrap1000.csv"
    rf_imp_path = RESULTS_DIR / "v6_strengthen" / "test1_rf_importance.csv"
    for path in (gnn_imp_path, rf_imp_path):
        if not path.exists():
            logger.warning("Weakness 1 skipped: %s not found (legacy ablation output)", path)
            return None

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_activities = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    scanb_tf = pd.read_csv(DATA / "precision_processed/tf_activities_scanb.csv", index_col=0)
    scanb_clinical = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    scanb_clinical = scanb_clinical.drop_duplicates(subset="sample_id")

    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)

    gnn_imp = pd.read_csv(gnn_imp_path)
    rf_imp = pd.read_csv(rf_imp_path)

    shared_tfs = sorted(set(tf_activities.columns) & set(scanb_tf.columns))

    configs = {
        "All_769_TFs": shared_tfs,
        "GNN_top50": [t for t in gnn_imp.head(50)["tf"] if t in shared_tfs],
        "GNN_top100": [t for t in gnn_imp.head(100)["tf"] if t in shared_tfs],
        "RF_top50": [t for t in rf_imp.head(50)["tf"] if t in shared_tfs],
        "RF_top100": [t for t in rf_imp.head(100)["tf"] if t in shared_tfs],
        "GNN_significant_350": [t for t in gnn_imp[gnn_imp["significant"]]["tf"] if t in shared_tfs],
    }

    results = {}
    for config_name, tf_subset in configs.items():
        if len(tf_subset) < 5:
            logger.warning("%s: only %d TFs, skipping", config_name, len(tf_subset))
            continue

        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        prism_sub = tf_activities[[t for t in tf_subset if t in tf_activities.columns]]
        scanb_sub = scanb_tf[[t for t in tf_subset if t in scanb_tf.columns]]

        prism_qt = pd.DataFrame(qt.fit_transform(prism_sub), index=prism_sub.index, columns=prism_sub.columns)
        scanb_qt = pd.DataFrame(qt.transform(scanb_sub), index=scanb_sub.index, columns=scanb_sub.columns)

        train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
        train_resp = drug_response.loc[train_feat.index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat)
        X_scanb = scaler.transform(scanb_qt)

        pvalues = []
        for drug in drug_response.columns:
            y = train_resp[drug].values
            mask = ~np.isnan(y)
            if mask.sum() < 30:
                continue
            ridge = Ridge(alpha=100)
            ridge.fit(X_train[mask], y[mask])
            pred = ridge.predict(X_scanb)

            pred_s = pd.Series(pred, index=scanb_qt.index)
            merged = scanb_clinical.set_index("sample_id").join(
                pred_s.rename("auc"), how="inner"
            ).dropna(subset=["os_time", "os_event", "auc"])

            if len(merged) < 100:
                continue
            med = merged["auc"].median()
            s, r = merged[merged["auc"] < med], merged[merged["auc"] >= med]
            if len(s) < 30 or len(r) < 30:
                continue
            lr = logrank_test(s["os_time"], r["os_time"],
                              event_observed_A=s["os_event"], event_observed_B=r["os_event"])
            pvalues.append(lr.p_value)

        padj = stats.false_discovery_control(pvalues) if pvalues else []
        n_sig = sum(p < 0.05 for p in padj)

        results[config_name] = {
            "n_tfs": len(tf_subset), "n_tested": len(pvalues),
            "n_sig": n_sig, "pct_sig": 100 * n_sig / max(len(pvalues), 1),
        }
        logger.info("%s (%d TFs): %d/%d FDR<0.05 (%.0f%%)",
                    config_name, len(tf_subset), n_sig, len(pvalues),
                    results[config_name]["pct_sig"])

    pd.DataFrame(results).T.to_csv(RESULTS / "d1_tf_subset_comparison.csv")
    return results


def weakness2_metabric_and_basal():
    """D2-B: METABRIC validation. D2-C: Basal enrichment test."""
    logger.info("=" * 60)
    logger.info("WEAKNESS 2: METABRIC validation + Basal enrichment")
    logger.info("=" * 60)

    # ── D2-B: METABRIC ──
    # METABRIC has expression but we need TF activities
    # Check if we can compute them
    metabric_expr_path = DATA / "METABRIC/metabric_expression.rds"
    metabric_clinical_path = DATA / "METABRIC/metabric_clinical.rds"

    logger.info("METABRIC: checking data availability...")
    if not metabric_expr_path.exists():
        logger.warning("METABRIC expression not found, skipping")
        metabric_result = None
    else:
        logger.info("METABRIC expression found (RDS format, needs R for TF activities)")
        logger.info("METABRIC TF activities need to be computed via R script first")
        logger.info("Skipping METABRIC for now: requires separate R preprocessing")
        metabric_result = "needs_R_preprocessing"

    # ── D2-C: Basal enrichment ──
    logger.info("\n--- Basal nominal enrichment test ---")

    # Load Basal survival results
    basal_path = RESULTS_DIR / "v6_strengthen" / "test3_survival_Basal.csv"
    if basal_path.exists():
        basal = pd.read_csv(basal_path)
        n_total = len(basal)
        n_nominal = (basal["pvalue"] < 0.05).sum()
        expected = 0.05 * n_total

        # Binomial test: observed vs expected under null
        binom_p = binomtest(n_nominal, n_total, 0.05, alternative="greater").pvalue

        # Also check directionality: are the nominal hits enriched for HR > 1?
        # (i.e., predicted sensitive = better survival)

        logger.info("Basal enrichment:")
        logger.info("  Total drugs tested: %d", n_total)
        logger.info("  Nominal p<0.05: %d (%.1f%%, expected %.0f = 5%%)",
                    n_nominal, 100 * n_nominal / n_total, expected)
        logger.info("  Enrichment binomial test: p = %.2e", binom_p)
        logger.info("  Fold enrichment: %.1fx over expected",
                    n_nominal / max(expected, 1))

        # Permutation test: shuffle survival labels 1000 times
        logger.info("  Running permutation test (100 permutations)...")
        from lifelines.statistics import logrank_test

        pred = pd.read_csv(RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv", index_col=0)
        clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
        clin = clin.drop_duplicates(subset="sample_id")
        basal_clin = clin[clin["pam50"] == "Basal"]

        # Count significant in real data (first 200 drugs for speed)
        drugs_to_test = pred.columns[:200]
        real_sig = 0
        for drug in drugs_to_test:
            p = pred[drug]
            m = basal_clin.set_index("sample_id").join(p.rename("auc"), how="inner")
            m = m.dropna(subset=["os_time", "os_event", "auc"])
            if len(m) < 50:
                continue
            med = m["auc"].median()
            s, r = m[m["auc"] < med], m[m["auc"] >= med]
            if len(s) < 15 or len(r) < 15:
                continue
            lr = logrank_test(s["os_time"], r["os_time"],
                              event_observed_A=s["os_event"], event_observed_B=r["os_event"])
            if lr.p_value < 0.05:
                real_sig += 1

        # Permutation
        perm_sigs = []
        rng = np.random.RandomState(SEED)
        for perm_i in range(100):
            perm_sig = 0
            # Shuffle os_event labels
            shuffled_clin = basal_clin.copy()
            shuffled_clin["os_event"] = rng.permutation(shuffled_clin["os_event"].values)

            for drug in drugs_to_test:
                p = pred[drug]
                m = shuffled_clin.set_index("sample_id").join(p.rename("auc"), how="inner")
                m = m.dropna(subset=["os_time", "os_event", "auc"])
                if len(m) < 50:
                    continue
                med = m["auc"].median()
                s, r = m[m["auc"] < med], m[m["auc"] >= med]
                if len(s) < 15 or len(r) < 15:
                    continue
                lr = logrank_test(s["os_time"], r["os_time"],
                                  event_observed_A=s["os_event"],
                                  event_observed_B=r["os_event"])
                if lr.p_value < 0.05:
                    perm_sig += 1
            perm_sigs.append(perm_sig)

            if (perm_i + 1) % 20 == 0:
                logger.info("    Permutation %d/100: %d sig (real: %d)",
                            perm_i + 1, perm_sig, real_sig)

        perm_p = (sum(p >= real_sig for p in perm_sigs) + 1) / (len(perm_sigs) + 1)
        logger.info("  Permutation test: real=%d, perm mean=%.1f, p=%.4f",
                    real_sig, np.mean(perm_sigs), perm_p)

        basal_result = {
            "n_total": n_total, "n_nominal": n_nominal,
            "expected": expected, "binom_p": binom_p,
            "fold_enrichment": n_nominal / max(expected, 1),
            "perm_real": real_sig, "perm_mean": np.mean(perm_sigs), "perm_p": perm_p,
        }
    else:
        basal_result = None

    pd.DataFrame([basal_result]).to_csv(RESULTS / "d2_basal_enrichment.csv", index=False)
    return metabric_result, basal_result


def weakness3_penalized_cox():
    """D3-A: Penalized (Ridge) Cox regression for regularized HRs."""
    logger.info("=" * 60)
    logger.info("WEAKNESS 3: Penalized Cox (Ridge) for regularized HRs")
    logger.info("=" * 60)

    from lifelines import CoxPHFitter

    pred = pd.read_csv(RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv", index_col=0)
    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin.drop_duplicates(subset="sample_id")

    merged = clin.set_index("sample_id")[["os_time", "os_event", "pam50"]].join(pred, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "pam50"])

    pam_dummies = pd.get_dummies(merged["pam50"], prefix="pam50", drop_first=False)
    pam_dummies = pam_dummies.drop(columns=["pam50_LumA"], errors="ignore")
    merged = pd.concat([merged, pam_dummies], axis=1)
    pam_cols = list(pam_dummies.columns)

    results = []
    for i, drug in enumerate(pred.columns):
        if drug not in merged.columns:
            continue
        cox_data = merged[["os_time", "os_event", drug] + pam_cols].dropna()
        cox_data = cox_data.rename(columns={drug: "predicted_auc"})

        if len(cox_data) < 100 or cox_data["os_event"].sum() < 20:
            continue

        try:
            # Penalized Cox with higher regularization
            cph = CoxPHFitter(penalizer=0.1)  # 10x more than before
            cph.fit(cox_data, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR": hr, "pvalue": p})
        except Exception:
            continue

        if (i + 1) % 200 == 0:
            logger.info("  processed %d drugs...", i + 1)

    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    df = df.sort_values("pvalue")

    n_sig = (df["padj"] < 0.05).sum()
    max_hr = df[df["padj"] < 0.05]["HR"].max()
    min_hr = df[df["padj"] < 0.05]["HR"].min()

    logger.info("Penalized Cox (penalizer=0.1):")
    logger.info("  %d/%d FDR<0.05", n_sig, len(df))
    logger.info("  HR range (significant): %.2f - %.2f", min_hr, max_hr)
    logger.info("  Previous unpenalized: HR range up to 274")

    # Check BC drugs
    bc_drugs = ["paclitaxel", "olaparib", "talazoparib", "docetaxel",
                "epirubicin", "carboplatin", "mitoxantrone"]
    logger.info("\nBC drugs (penalized Cox):")
    for d in bc_drugs:
        row = df[df["drug"] == d]
        if len(row) > 0:
            r = row.iloc[0]
            sig = "*" if r["padj"] < 0.05 else ""
            logger.info("  %s: HR=%.3f p=%.2e padj=%.2e %s",
                        d, r["HR"], r["pvalue"], r["padj"], sig)

    logger.info("\nTop 10:")
    for _, r in df.head(10).iterrows():
        logger.info("  %s: HR=%.3f padj=%.2e", r["drug"], r["HR"], r["padj"])

    df.to_csv(RESULTS / "d3_penalized_cox.csv", index=False)
    return {"n_sig": n_sig, "max_hr": max_hr, "min_hr": min_hr}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    d1 = weakness1_tf_subset_transfer()
    d2_meta, d2_basal = weakness2_metabric_and_basal()
    d3 = weakness3_penalized_cox()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if d1 is None:
        logger.info("D1 (TF subset transfer): skipped (legacy ablation importance files not found)")
    else:
        logger.info("D1 (TF subset transfer):")
        for name, r in d1.items():
            logger.info("  %s: %d/%d FDR<0.05 (%.0f%%)", name, r["n_sig"], r["n_tested"], r["pct_sig"])

    if d2_basal:
        logger.info("D2 (Basal enrichment): fold=%.1fx, binom p=%.2e, perm p=%.4f",
                    d2_basal["fold_enrichment"], d2_basal["binom_p"], d2_basal["perm_p"])

    logger.info("D3 (Penalized Cox): %d FDR<0.05, HR range [%.2f, %.2f]",
                d3["n_sig"], d3["min_hr"], d3["max_hr"])

    logger.info("DONE")


if __name__ == "__main__":
    main()
