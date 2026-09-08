#!/usr/bin/env python3
"""
Within-subtype survival screens in SCAN-B: Basal alone and the Basal + HER2
pool (Results, "within-subtype analysis", and Supplementary Figure on the
ER-negative pool).

Cohort definition (the one stated in the manuscript): samples with a
prediction in the per-drug matrix and overall-survival follow-up, one sample
per patient (the first one). Same test as the per-subtype screen of
scripts/18_strengthen_paper.py (test 3): for every drug, patients are split
at the median predicted AUC and the two arms are compared with a log-rank
test on overall survival, with Benjamini-Hochberg control over all tested
drugs. Script 18 keeps every sample (no patient-level deduplication), which
is why its counts differ from the ones reported in the manuscript.

Inputs:
  results/v5_final/scanb_perdrug_predictions.csv   (script 17)
  data/precision_processed/scanb_clinical.csv      (script 09)
Outputs (results/v7_weaknesses/):
  basal_only_survival.csv          columns drug, pvalue, padj (Basal alone)
  basal_her2_survival.csv          columns drug, pvalue, padj (Basal + HER2)
  subtype_pooling_summary.csv      one row per subset: n_patients, n_events,
                                   n_tested, n_fdr05, n_fdr10

Usage: python scripts/19b_basal_her2_pooled.py (paths from config.py)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from lifelines.statistics import logrank_test
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

OUT_DIR = RESULTS_DIR / "v7_weaknesses"

SUBSETS = {
    "Basal": ("Basal",),          # PAM50 label as stored in scanb_clinical.csv
    "Basal_HER2": ("Basal", "Her2"),
}
MIN_PATIENTS = 50
MIN_ARM = 15


def screen(subset: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-drug log-rank test between the two halves of predicted AUC."""
    drugs, pvalues = [], []
    for drug in pred_df.columns:
        merged = subset.join(pred_df[drug].rename("auc"), how="inner").dropna(
            subset=["os_time", "os_event", "auc"]
        )
        if len(merged) < MIN_PATIENTS:
            continue
        median_auc = merged["auc"].median()
        sens = merged[merged["auc"] < median_auc]
        res = merged[merged["auc"] >= median_auc]
        if len(sens) < MIN_ARM or len(res) < MIN_ARM:
            continue
        lr = logrank_test(sens["os_time"], res["os_time"],
                          event_observed_A=sens["os_event"],
                          event_observed_B=res["os_event"])
        drugs.append(drug)
        pvalues.append(lr.p_value)
    padj = stats.false_discovery_control(pvalues)
    return pd.DataFrame({"drug": drugs, "pvalue": pvalues, "padj": padj}).sort_values("pvalue")


def main() -> int:
    pred_df = pd.read_csv(RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv", index_col=0)
    clinical = pd.read_csv(DATA_DIR / "precision_processed" / "scanb_clinical.csv")
    clinical = clinical.drop_duplicates(subset="sample_id")
    cohort = clinical[clinical["sample_id"].isin(pred_df.index)]
    cohort = cohort.dropna(subset=["os_time", "os_event"]).drop_duplicates(subset="patient_id")
    cohort = cohort.set_index("sample_id")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, labels in SUBSETS.items():
        subset = cohort[cohort["pam50"].isin(labels)]
        n_events = int(subset["os_event"].sum())
        df = screen(subset, pred_df)
        n05, n10 = int((df["padj"] < 0.05).sum()), int((df["padj"] < 0.10).sum())
        out = OUT_DIR / ("basal_only_survival.csv" if name == "Basal" else "basal_her2_survival.csv")
        df.to_csv(out, index=False)
        rows.append({"subset": name, "pam50_labels": "+".join(labels), "n_patients": len(subset),
                     "n_events": n_events, "n_tested": len(df), "n_fdr05": n05, "n_fdr10": n10})
        logger.info("%s (%s): n = %d patients, %d events, %d drugs tested, %d at FDR < 0.05, %d at FDR < 0.10 -> %s",
                    name, "+".join(labels), len(subset), n_events, len(df), n05, n10, out)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "subtype_pooling_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
