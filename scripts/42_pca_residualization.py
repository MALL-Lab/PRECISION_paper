#!/usr/bin/env python3
"""
PRECISION - PCA + PC1 residualization of per-drug survival predictions.

The Discussion claimed (without a committed script) that:
  - PC1 of the per-drug prediction matrix explains 37% of variance
  - PC1 is not itself associated with survival (p = 0.059)
  - residualizing PC1 raises the count of drugs with nominal survival
    association from 110 to 138 (of 200 tested)
  - inter-drug prediction correlation r = 0.118

Those numbers had no reproducible source and the inter-drug value was wrong
(the real mean is ~0.072). This script recomputes the whole analysis from the
canonical SCAN-B per-drug predictions so we know what actually holds.

Design (matches the rest of the pipeline):
  - SCAN-B per-drug predictions (results/v5_final/scanb_perdrug_predictions.csv)
  - patient-level dedup + PAM50 + OS, same as scripts 17/38
  - PCA on the standardized prediction matrix; report PC1 explained variance
  - PC1 patient score vs OS via penalized Cox (penalizer=0.1, PAM50-adjusted)
  - per-drug nominal (p<0.05) survival association BEFORE and AFTER
    residualizing PC1, on the top-200-by-variance drugs (matches "200 tested")
    and, for completeness, on all drugs
  - empirical mean/median inter-drug correlation

Outputs in results/v8_multicohort/:
  pca_residualization_summary.csv
  pca_residualization_perdrug.csv

Run from project root:
    python scripts/42_pca_residualization.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR
from src.data.splits import set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pca_resid")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"
PRED_PATH = RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv"
SEED = 42
N_TOP = 200  # "200 tested" subset = top-200 by prediction variance


def load_matrix():
    """Patient-level prediction matrix + OS + PAM50 (same prep as script 38)."""
    preds = pd.read_csv(PRED_PATH, index_col=0)
    logger.info("Raw predictions: %d samples x %d drugs", *preds.shape)

    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")].set_index("sample_id")
    clin["os_time"] = pd.to_numeric(clin.get("os_time"), errors="coerce")
    clin["os_event"] = pd.to_numeric(clin.get("os_event"), errors="coerce")

    merged = clin[["os_time", "os_event", "pam50"]].join(preds, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "pam50"])
    merged = merged[merged["os_time"] > 0]

    drug_cols = [c for c in preds.columns if c in merged.columns]
    X = merged[drug_cols].copy()
    surv = merged[["os_time", "os_event", "pam50"]].copy()
    logger.info("Patient-level matrix: %d patients x %d drugs, %d events",
                X.shape[0], X.shape[1], int(surv["os_event"].sum()))
    return X, surv


def cox_pvalue(covariate: np.ndarray, surv: pd.DataFrame,
               pam_dummies: pd.DataFrame) -> float:
    """Penalized Cox (0.1) of OS on covariate + PAM50; return covariate p."""
    cd = pd.DataFrame({
        "os_time": surv["os_time"].values,
        "os_event": surv["os_event"].values,
        "x": covariate,
    }, index=surv.index)
    cd = pd.concat([cd, pam_dummies], axis=1).dropna()
    if cd["x"].nunique() < 3:
        return np.nan
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cd, duration_col="os_time", event_col="os_event")
        return float(cph.summary.loc["x", "p"])
    except Exception:
        return np.nan


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    X, surv = load_matrix()
    pam_dummies = pd.get_dummies(surv["pam50"], prefix="sub", drop_first=True,
                                 dtype=int)

    # ---- Inter-drug correlation (verify the r=0.118 claim) --------------
    corr = X.corr()
    np.fill_diagonal(corr.values, np.nan)
    mean_corr = float(np.nanmean(corr.values))
    med_corr = float(np.nanmedian(corr.values))
    logger.info("Inter-drug correlation: mean=%.4f, median=%.4f",
                mean_corr, med_corr)

    # ---- PCA on standardized prediction matrix --------------------------
    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=min(20, Xs.shape[1]), random_state=SEED)
    scores = pca.fit_transform(Xs)
    pc1_var = float(pca.explained_variance_ratio_[0])
    pc1_score = scores[:, 0]
    logger.info("PC1 explains %.1f%% of prediction variance", 100 * pc1_var)
    logger.info("Top-5 PC variance ratios: %s",
                [f"{v:.3f}" for v in pca.explained_variance_ratio_[:5]])

    # ---- PC1 vs survival ------------------------------------------------
    pc1_p = cox_pvalue(pc1_score, surv, pam_dummies)
    logger.info("PC1 ~ OS (PAM50-adjusted Cox): p=%.4f  -> %s",
                pc1_p, "assoc" if pc1_p < 0.05 else "NOT associated")

    # ---- Per-drug nominal association before/after PC1 residualization --
    # Select "200 tested" = top-200 drugs by prediction variance.
    variances = X.var().sort_values(ascending=False)
    top_drugs = variances.head(N_TOP).index.tolist()
    logger.info("Testing top-%d drugs by prediction variance", len(top_drugs))

    # PC1 residualization: for each drug, regress out the PC1 patient score.
    pc1 = pc1_score.reshape(-1, 1)
    pc1_design = np.hstack([np.ones_like(pc1), pc1])  # intercept + PC1
    # Least-squares projection matrix onto [1, PC1]
    beta_all, *_ = np.linalg.lstsq(pc1_design, X[top_drugs].values, rcond=None)
    fitted = pc1_design @ beta_all
    residuals = X[top_drugs].values - fitted  # PC1 removed, per drug

    rows = []
    n_raw_sig = 0
    n_res_sig = 0
    for j, drug in enumerate(top_drugs):
        p_raw = cox_pvalue(X[drug].values, surv, pam_dummies)
        p_res = cox_pvalue(residuals[:, j], surv, pam_dummies)
        raw_sig = (not np.isnan(p_raw)) and p_raw < 0.05
        res_sig = (not np.isnan(p_res)) and p_res < 0.05
        n_raw_sig += int(raw_sig)
        n_res_sig += int(res_sig)
        rows.append({"drug": drug, "p_raw": p_raw, "p_residualized": p_res,
                     "raw_sig": raw_sig, "res_sig": res_sig})
        if (j + 1) % 50 == 0:
            logger.info("  ...%d/%d drugs", j + 1, len(top_drugs))

    perdrug = pd.DataFrame(rows)
    perdrug.to_csv(RESULTS / "pca_residualization_perdrug.csv", index=False)

    logger.info("=" * 60)
    logger.info("RESULT (top-%d drugs):", N_TOP)
    logger.info("  nominal survival assoc BEFORE PC1 residualization: %d/%d",
                n_raw_sig, N_TOP)
    logger.info("  nominal survival assoc AFTER  PC1 residualization: %d/%d",
                n_res_sig, N_TOP)
    logger.info("  direction: %s",
                "INCREASE (drug-specific signal survives)"
                if n_res_sig >= n_raw_sig else
                "DECREASE (shared PC1 axis carried signal)")
    logger.info("=" * 60)

    summary = pd.DataFrame([
        {"metric": "n_patients", "value": X.shape[0]},
        {"metric": "n_drugs_total", "value": X.shape[1]},
        {"metric": "inter_drug_corr_mean", "value": round(mean_corr, 4)},
        {"metric": "inter_drug_corr_median", "value": round(med_corr, 4)},
        {"metric": "pc1_explained_var_pct", "value": round(100 * pc1_var, 1)},
        {"metric": "pc1_survival_pvalue", "value": round(pc1_p, 4)},
        {"metric": "n_tested", "value": N_TOP},
        {"metric": "n_nominal_sig_raw", "value": n_raw_sig},
        {"metric": "n_nominal_sig_residualized", "value": n_res_sig},
    ])
    summary.to_csv(RESULTS / "pca_residualization_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("DONE")


if __name__ == "__main__":
    main()
