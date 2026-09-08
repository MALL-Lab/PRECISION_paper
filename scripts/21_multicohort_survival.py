#!/usr/bin/env python3
"""
PRECISION: Multi-cohort survival validation.

Runs per-drug Ridge transfer + penalized Cox survival on:
1. SCAN-B (n~8000, RNA-seq, already computed)
2. METABRIC (n~1904, microarray)
3. TCGA-BRCA (n~1100, RNA-seq)

Then: meta-analysis across cohorts (Fisher's method for combining p-values).

Run from project root (after 20_multicohort_validation.R):
    python scripts/21_multicohort_survival.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multicohort")

RESULTS = RESULTS_DIR / "v8_multicohort"
DATA = DATA_DIR
SEED = 42


def run_cohort_validation(cohort_name, tf_path, clinical_path,
                          os_time_col, os_event_col, subtype_col,
                          drug_response, tf_prism, train_cells):
    """Run per-drug Ridge transfer + penalized Cox for one cohort."""
    logger.info("=" * 60)
    logger.info("COHORT: %s", cohort_name)
    logger.info("=" * 60)

    cohort_tf = pd.read_csv(tf_path, index_col=0)
    cohort_clin = pd.read_csv(clinical_path, index_col=0)
    logger.info("  TF activities: %d x %d", *cohort_tf.shape)
    logger.info("  Clinical: %d patients", len(cohort_clin))

    # Cohort summary (rows of the clinical table, samples with TF activities,
    # patients after deduplication, patients with survival, events). Written to
    # cohort_summary.csv so that the manuscript numbers come from this script.
    summary = {
        "cohort": cohort_name,
        "n_clinical_rows": int(len(cohort_clin)),
        "n_tf_samples": int(len(cohort_tf)),
        "n_tfs_cohort": int(cohort_tf.shape[1]),
    }

    # Fix index alignment: clinical may use numeric index with patientId column
    if len(set(cohort_tf.index) & set(cohort_clin.index)) == 0:
        # Try patientId column
        if "patientId" in cohort_clin.columns:
            cohort_clin = cohort_clin.set_index("patientId")
            logger.info("  Reindexed clinical by patientId")
        elif "PATIENT_ID" in cohort_clin.columns:
            cohort_clin = cohort_clin.set_index("PATIENT_ID")
            logger.info("  Reindexed clinical by PATIENT_ID")
        elif "sample_id" in cohort_clin.columns:
            cohort_clin = cohort_clin.set_index("sample_id")
            logger.info("  Reindexed clinical by sample_id")

    overlap = len(set(cohort_tf.index) & set(cohort_clin.index))
    logger.info("  ID overlap: %d", overlap)

    # METABRIC: agreement between the genefu PAM50 call (script 20) and the
    # cBioPortal CLAUDIN_SUBTYPE label, excluding claudin-low and NC, which
    # have no PAM50 equivalent (Methods reports this concordance)
    if cohort_name == "METABRIC" and {"PAM50", "CLAUDIN_SUBTYPE"} <= set(cohort_clin.columns):
        both = cohort_clin.dropna(subset=["PAM50", "CLAUDIN_SUBTYPE"])
        comparable = both[both["CLAUDIN_SUBTYPE"].isin(["LumA", "LumB", "Her2", "Basal", "Normal"])]
        summary["pam50_genefu_n_compared"] = int(len(comparable))
        summary["pam50_genefu_concordance_pct"] = round(
            100 * float((comparable["PAM50"] == comparable["CLAUDIN_SUBTYPE"]).mean()), 1)
        summary["pam50_genefu_concordance_all_pct"] = round(
            100 * float((both["PAM50"] == both["CLAUDIN_SUBTYPE"]).mean()), 1)
        logger.info("  genefu PAM50 vs CLAUDIN_SUBTYPE: %.1f%% of %d (all labels %.1f%%)",
                    summary["pam50_genefu_concordance_pct"], len(comparable),
                    summary["pam50_genefu_concordance_all_pct"])

    # Align TFs
    shared_tfs = sorted(set(tf_prism.columns) & set(cohort_tf.columns))
    logger.info("  Shared TFs: %d", len(shared_tfs))
    summary["n_tfs_shared_with_prism"] = int(len(shared_tfs))

    if len(shared_tfs) < 50:
        logger.warning("  Too few shared TFs, skipping")
        return None

    # QuantileTransformer batch correction (fitted on TRAIN cells only)
    train_prism = tf_prism.loc[[c for c in train_cells if c in tf_prism.index]]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism[shared_tfs])
    prism_qt = pd.DataFrame(
        qt.transform(tf_prism[shared_tfs]),
        index=tf_prism.index, columns=shared_tfs
    )
    cohort_qt = pd.DataFrame(
        qt.transform(cohort_tf[shared_tfs]),
        index=cohort_tf.index, columns=shared_tfs
    )

    # Check distribution alignment
    logger.info("  Post-QT: PRISM mean=%.3f std=%.3f, %s mean=%.3f std=%.3f",
                prism_qt.mean().mean(), prism_qt.std().mean(),
                cohort_name, cohort_qt.mean().mean(), cohort_qt.std().mean())

    # Per-drug Ridge
    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_cohort = scaler.transform(cohort_qt)

    predictions = {}
    for drug in drug_response.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[mask], y[mask])
        predictions[drug] = ridge.predict(X_cohort)

    pred_df = pd.DataFrame(predictions, index=cohort_qt.index)
    logger.info("  Predictions: %d patients x %d drugs", *pred_df.shape)

    # Survival: penalized Cox adjusting for subtype
    # First identify available clinical columns
    logger.info("  Clinical columns: %s", cohort_clin.columns.tolist()[:20])

    # SCAN-B: TF index is long (S000001.l.r.m...), clinical is short (S000001)
    if cohort_name == "SCANB":
        cohort_tf["short_id"] = cohort_tf.index.str.replace(r"\..*", "", regex=True)
        cohort_tf = cohort_tf[~cohort_tf["short_id"].duplicated()]
        cohort_tf.index = cohort_tf["short_id"]
        cohort_tf = cohort_tf.drop(columns=["short_id"])
        overlap_new = len(set(cohort_tf.index) & set(cohort_clin.index))
        logger.info("  SCAN-B short_id mapping: %d matches", overlap_new)

        # CRITICAL: deduplicate by patient_id to enforce independence
        if "patient_id" in cohort_clin.columns:
            n_before = len(cohort_clin)
            # Keep first sample per patient (deterministic, sorted by index)
            cohort_clin = cohort_clin.sort_index()
            cohort_clin = cohort_clin[~cohort_clin["patient_id"].duplicated(keep="first")]
            logger.info("  SCAN-B patient dedup: %d -> %d (removed %d duplicate samples)",
                        n_before, len(cohort_clin), n_before - len(cohort_clin))

    # TCGA: filter to primary tumors only and dedup by submitter_id
    if cohort_name == "TCGA":
        n_before = len(cohort_clin)
        if "shortLetterCode" in cohort_clin.columns:
            cohort_clin = cohort_clin[cohort_clin["shortLetterCode"] == "TP"]
            logger.info("  TCGA primary tumor filter: %d -> %d (removed %d non-TP samples)",
                        n_before, len(cohort_clin), n_before - len(cohort_clin))
        if "submitter_id" in cohort_clin.columns:
            n_before2 = len(cohort_clin)
            cohort_clin = cohort_clin.sort_index()
            cohort_clin = cohort_clin[~cohort_clin["submitter_id"].duplicated(keep="first")]
            logger.info("  TCGA submitter dedup: %d -> %d (removed %d duplicates)",
                        n_before2, len(cohort_clin), n_before2 - len(cohort_clin))

    # Parse survival data (cohort-specific formats)
    if cohort_name == "SCANB":
        cohort_clin["os_time_parsed"] = pd.to_numeric(cohort_clin.get("os_time", pd.Series()), errors="coerce")
        cohort_clin["os_event_parsed"] = pd.to_numeric(cohort_clin.get("os_event", pd.Series()), errors="coerce")
    elif cohort_name == "METABRIC":
        cohort_clin["os_time_parsed"] = pd.to_numeric(cohort_clin.get("OS_MONTHS", pd.Series()), errors="coerce") * 30.44  # months to days
        # OS_STATUS is "0:LIVING" or "1:DECEASED"
        os_status = cohort_clin.get("OS_STATUS", pd.Series())
        cohort_clin["os_event_parsed"] = os_status.str.extract(r"^(\d)")[0].astype(float)
    elif cohort_name == "TCGA":
        # Combine days_to_death (dead) + days_to_last_follow_up (alive)
        dtd = pd.to_numeric(cohort_clin.get("days_to_death", pd.Series()), errors="coerce")
        dtf = pd.to_numeric(cohort_clin.get("days_to_last_follow_up", pd.Series()), errors="coerce")
        cohort_clin["os_time_parsed"] = dtd.fillna(dtf)
        vital = cohort_clin.get("vital_status", pd.Series())
        cohort_clin["os_event_parsed"] = (vital == "Dead").astype(float)

    # Merge predictions with clinical
    merged = cohort_clin[["os_time_parsed", "os_event_parsed"]].join(pred_df, how="inner")
    merged = merged.rename(columns={"os_time_parsed": "os_time", "os_event_parsed": "os_event"})
    merged = merged.dropna(subset=["os_time", "os_event"])
    merged = merged[merged["os_time"] > 0]

    logger.info("  Patients with survival: %d (events: %d)",
                len(merged), int(merged["os_event"].sum()))
    summary["n_patients_after_dedup"] = int(len(cohort_clin))
    summary["n_patients_with_survival"] = int(len(merged))
    summary["n_events"] = int(merged["os_event"].sum())
    # Follow-up of the analysed cohort: median observed time and the reverse
    # Kaplan-Meier median follow-up (censoring treated as the event)
    summary["median_os_time_years"] = round(float(merged["os_time"].median()) / 365.25, 2)
    try:
        from lifelines import KaplanMeierFitter
        rkm = KaplanMeierFitter().fit(merged["os_time"] / 365.25, 1 - merged["os_event"])
        summary["median_followup_years_reverse_km"] = round(float(rkm.median_survival_time_), 2)
    except Exception as exc:  # lifelines missing or degenerate censoring
        logger.warning("  reverse KM follow-up not computed: %s", exc)
        summary["median_followup_years_reverse_km"] = float("nan")
    logger.info("  Median observed time %.2f years, reverse-KM median follow-up %s years",
                summary["median_os_time_years"], summary["median_followup_years_reverse_km"])

    # Add subtype if available
    has_subtype = False
    if subtype_col and subtype_col in cohort_clin.columns:
        merged = merged.join(cohort_clin[[subtype_col]])
        merged = merged.dropna(subset=[subtype_col])
        pam_dummies = pd.get_dummies(merged[subtype_col], prefix="sub", drop_first=True)
        merged = pd.concat([merged, pam_dummies], axis=1)
        pam_cols = list(pam_dummies.columns)
        has_subtype = True
        logger.info("  Subtype distribution: %s", dict(merged[subtype_col].value_counts()))
    else:
        pam_cols = []

    # Penalized Cox per drug
    results = []
    for drug in pred_df.columns:
        if drug not in merged.columns:
            continue
        cox_data = merged[["os_time", "os_event", drug] + pam_cols].dropna()
        cox_data = cox_data.rename(columns={drug: "predicted_auc"})

        if len(cox_data) < 50 or cox_data["os_event"].sum() < 10:
            continue

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR": hr, "pvalue": p})
        except Exception:
            continue

    if not results:
        logger.warning("  No Cox results")
        return {"predictions": pred_df, "survival": None, "summary": summary}

    surv_df = pd.DataFrame(results)
    surv_df["padj"] = stats.false_discovery_control(surv_df["pvalue"])
    surv_df = surv_df.sort_values("pvalue")
    surv_df["cohort"] = cohort_name

    n_sig = (surv_df["padj"] < 0.05).sum()
    logger.info("  Results: %d tested, %d FDR<0.05 (%.0f%%)",
                len(surv_df), n_sig, 100 * n_sig / len(surv_df))

    # BC drugs check
    bc_drugs = ["paclitaxel", "olaparib", "talazoparib", "docetaxel",
                "epirubicin", "carboplatin", "mitoxantrone"]
    for d in bc_drugs:
        row = surv_df[surv_df["drug"] == d]
        if len(row) > 0:
            r = row.iloc[0]
            sig = "*" if r["padj"] < 0.05 else ""
            logger.info("  %s: HR=%.2f padj=%.2e %s", d, r["HR"], r["padj"], sig)

    surv_df.to_csv(RESULTS / f"survival_{cohort_name}.csv", index=False)
    summary["n_drugs_tested"] = int(len(surv_df))
    summary["n_drugs_fdr05"] = int(n_sig)
    return {"predictions": pred_df, "survival": surv_df, "summary": summary}


def meta_analysis(cohort_results):
    """Fisher's method to combine p-values across cohorts."""
    logger.info("=" * 60)
    logger.info("META-ANALYSIS (Fisher's method)")
    logger.info("=" * 60)

    all_survs = [r["survival"] for r in cohort_results.values()
                 if r and r.get("survival") is not None]

    if len(all_survs) < 2:
        logger.warning("Need at least 2 cohorts for meta-analysis")
        return None

    # Find drugs tested in all cohorts
    drug_sets = [set(s["drug"]) for s in all_survs]
    common_drugs = sorted(set.intersection(*drug_sets))
    logger.info("Drugs tested in all %d cohorts: %d", len(all_survs), len(common_drugs))

    results = []
    for drug in common_drugs:
        pvals = []
        hrs = []
        for surv in all_survs:
            row = surv[surv["drug"] == drug]
            if len(row) > 0:
                pvals.append(row.iloc[0]["pvalue"])
                hrs.append(row.iloc[0]["HR"])

        if len(pvals) < 2:
            continue

        # Fisher's method: -2 * sum(log(p)) ~ chi2(2k)
        stat = -2 * sum(np.log(p) for p in pvals)
        fisher_p = 1 - stats.chi2.cdf(stat, df=2 * len(pvals))
        mean_hr = np.exp(np.mean(np.log(hrs)))  # geometric mean of HRs

        results.append({
            "drug": drug,
            "fisher_p": fisher_p,
            "mean_HR": mean_hr,
            "n_cohorts": len(pvals),
            "pvals": str(pvals),
        })

    meta_df = pd.DataFrame(results)
    meta_df["fisher_padj"] = stats.false_discovery_control(meta_df["fisher_p"])
    meta_df = meta_df.sort_values("fisher_p")

    n_sig = (meta_df["fisher_padj"] < 0.05).sum()
    logger.info("Meta-analysis: %d drugs, %d FDR<0.05 (%.0f%%)",
                len(meta_df), n_sig, 100 * n_sig / len(meta_df))

    for _, r in meta_df.head(10).iterrows():
        logger.info("  %s: fisher_p=%.2e padj=%.2e mean_HR=%.2f",
                    r["drug"], r["fisher_p"], r["fisher_padj"], r["mean_HR"])

    # BC drugs
    bc_drugs = ["paclitaxel", "olaparib", "docetaxel", "epirubicin"]
    logger.info("\nBC drugs (meta-analysis):")
    for d in bc_drugs:
        row = meta_df[meta_df["drug"] == d]
        if len(row) > 0:
            r = row.iloc[0]
            sig = "*" if r["fisher_padj"] < 0.05 else ""
            logger.info("  %s: fisher_padj=%.2e mean_HR=%.2f %s",
                        d, r["fisher_padj"], r["mean_HR"], sig)

    meta_df.to_csv(RESULTS / "meta_analysis_fisher.csv", index=False)
    return meta_df


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)

    cohort_results = {}

    # SCAN-B (already have predictions, just run penalized Cox)
    scanb_tf = DATA / "precision_processed/tf_activities_scanb.csv"
    scanb_clin = DATA / "precision_processed/scanb_clinical.csv"
    if scanb_tf.exists():
        cohort_results["SCANB"] = run_cohort_validation(
            "SCANB", scanb_tf, scanb_clin,
            "os_time", "os_event", "pam50",
            drug_response, tf_prism, train_cells
        )

    # METABRIC
    metabric_tf = DATA / "precision_processed/tf_activities_metabric.csv"
    metabric_clin = DATA / "precision_processed/metabric_clinical.csv"
    if metabric_tf.exists():
        cohort_results["METABRIC"] = run_cohort_validation(
            "METABRIC", metabric_tf, metabric_clin,
            "OS_MONTHS", "OS_STATUS", "PAM50",
            drug_response, tf_prism, train_cells
        )
    else:
        logger.warning("METABRIC TF activities not found: run R script first")

    # TCGA-BRCA
    tcga_tf = DATA / "precision_processed/tf_activities_tcga.csv"
    tcga_clin = DATA / "precision_processed/tcga_clinical.csv"
    if tcga_tf.exists():
        cohort_results["TCGA"] = run_cohort_validation(
            "TCGA", tcga_tf, tcga_clin,
            "days_to_death", "vital_status", "paper_BRCA_Subtype_PAM50",
            drug_response, tf_prism, train_cells
        )
    else:
        logger.warning("TCGA TF activities not found: run R script first")

    # Meta-analysis
    meta = meta_analysis(cohort_results)

    # Cohort sizes cited in the manuscript (Table 2, Methods dedup chains)
    summaries = [r["summary"] for r in cohort_results.values() if r and "summary" in r]
    if summaries:
        pd.DataFrame(summaries).to_csv(RESULTS / "cohort_summary.csv", index=False)
        logger.info("Cohort summary written to %s", RESULTS / "cohort_summary.csv")

    # Summary
    logger.info("=" * 60)
    logger.info("MULTI-COHORT SUMMARY")
    logger.info("=" * 60)
    for name, r in cohort_results.items():
        if r and r.get("survival") is not None:
            s = r["survival"]
            n_sig = (s["padj"] < 0.05).sum()
            logger.info("  %s: %d/%d FDR<0.05 (%.0f%%)",
                        name, n_sig, len(s), 100 * n_sig / len(s))
    if meta is not None:
        n_meta = (meta["fisher_padj"] < 0.05).sum()
        logger.info("  Meta-analysis: %d/%d FDR<0.05", n_meta, len(meta))

    logger.info("DONE")


if __name__ == "__main__":
    main()
