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

Second output, results/v13_reaudit/pc1_orientation.csv (metric, value), which
says what PC1 is. The sign of a principal component is arbitrary, so the PC1
score is oriented to correlate positively with the predicted AUC of paclitaxel
(pc1_sign_flipped = 1 means the raw score was multiplied by -1). A higher
oriented score therefore means a higher predicted AUC, i.e. predicted more
resistant to paclitaxel. The file reports the Pearson correlation of the
oriented score with the SCAN-B TF activities of E2F1, MYC, TP53 and ESR1
(data/precision_processed/tf_activities_scanb.csv, same patients) and with the
predicted AUC of talazoparib and olaparib, the Cox Wald z of the oriented score
(same model as the PC1 survival test above), the variance explained by PC2, and
the mean signed and mean absolute inter-drug correlation of the predictions with
the percentage of drug pairs whose correlation is negative. Before writing it
the script reproduces published numbers and raises if they differ: the
PAM50-adjusted Cox cohort of script 21 (7,215 patients, 1,164 events), the
mean inter-drug correlation (0.073), the PC1 variance (12.0%) and the Cox
p-values of paclitaxel and erlotinib in results/v8_multicohort/survival_SCANB.csv.
This second output adds nothing to the two files above, which stay byte-identical.

Run from project root:
    python scripts/42_pca_residualization.py
"""

from __future__ import annotations

import logging
import os
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

# Second output (PC1 orientation), see the module docstring.
REAUDIT = RESULTS_DIR / "v13_reaudit"
TF_PATH = DATA / "precision_processed/tf_activities_scanb.csv"
SURV_REF_PATH = RESULTS / "survival_SCANB.csv"  # script 21
ORIENT_DRUG = "paclitaxel"
PARP_DRUGS = ["talazoparib", "olaparib"]
TFS_OF_INTEREST = ["E2F1", "MYC", "TP53", "ESR1"]
EXPECTED_N_PATIENTS = 7215      # PAM50-adjusted SCAN-B Cox cohort of script 21
EXPECTED_N_EVENTS = 1164
EXPECTED_CORR_MEAN = 0.073      # mean signed inter-drug correlation, 3 decimals
EXPECTED_PC1_VAR_PCT = 12.0     # PC1 explained variance, 1 decimal
REF_P_DRUGS = ["paclitaxel", "erlotinib"]
P_RTOL = 1e-6


def load_matrix():
    """Patient-level prediction matrix + OS + PAM50: one row per patient, PAM50 known, OS time > 0."""
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


def cox_stats(covariate: np.ndarray, surv: pd.DataFrame,
              pam_dummies: pd.DataFrame) -> tuple[float, float]:
    """Penalized Cox (0.1) of OS on covariate + PAM50; return covariate p and Wald z."""
    cd = pd.DataFrame({
        "os_time": surv["os_time"].values,
        "os_event": surv["os_event"].values,
        "x": covariate,
    }, index=surv.index)
    cd = pd.concat([cd, pam_dummies], axis=1).dropna()
    if cd["x"].nunique() < 3:
        return np.nan, np.nan
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cd, duration_col="os_time", event_col="os_event")
        return float(cph.summary.loc["x", "p"]), float(cph.summary.loc["x", "z"])
    except Exception:
        return np.nan, np.nan


def cox_pvalue(covariate: np.ndarray, surv: pd.DataFrame,
               pam_dummies: pd.DataFrame) -> float:
    """Penalized Cox (0.1) of OS on covariate + PAM50; return covariate p."""
    return cox_stats(covariate, surv, pam_dummies)[0]


def pc1_orientation(X: pd.DataFrame, surv: pd.DataFrame, pam_dummies: pd.DataFrame,
                    pca: PCA, pc1_score: np.ndarray, pc1_z: float,
                    corr: pd.DataFrame, mean_corr: float) -> pd.DataFrame:
    """Orient PC1 on paclitaxel and describe it (second output, see docstring).

    corr is the inter-drug correlation matrix of main() with its diagonal set to
    NaN. Raises if the reproduced reference numbers differ from the published ones.
    """
    # ---- Reproduction of published numbers ------------------------------
    n_pat, n_ev = X.shape[0], int(surv["os_event"].sum())
    if (n_pat, n_ev) != (EXPECTED_N_PATIENTS, EXPECTED_N_EVENTS):
        raise RuntimeError(f"Cox cohort is {n_pat} patients / {n_ev} events, expected "
                           f"{EXPECTED_N_PATIENTS} / {EXPECTED_N_EVENTS}")
    if round(mean_corr, 3) != EXPECTED_CORR_MEAN:
        raise RuntimeError(f"Mean inter-drug correlation {mean_corr:.4f} does not "
                           f"reproduce {EXPECTED_CORR_MEAN}")
    pc1_var_pct = round(100 * float(pca.explained_variance_ratio_[0]), 1)
    if pc1_var_pct != EXPECTED_PC1_VAR_PCT:
        raise RuntimeError(f"PC1 explains {pc1_var_pct}%, expected {EXPECTED_PC1_VAR_PCT}%")
    ref = pd.read_csv(SURV_REF_PATH).set_index("drug")
    for drug in REF_P_DRUGS:
        p_here = cox_pvalue(X[drug].values, surv, pam_dummies)
        p_ref = float(ref.loc[drug, "pvalue"])
        if not np.isclose(p_here, p_ref, rtol=P_RTOL, atol=0):
            raise RuntimeError(f"{drug} Cox p {p_here:.6g} does not reproduce "
                               f"{p_ref:.6g} of {SURV_REF_PATH}")
        logger.info("Reproduced %s Cox p = %.6g (survival_SCANB.csv %.6g)",
                    drug, p_here, p_ref)
    logger.info("Reproduced cohort %d / %d, mean inter-drug r = %.4f, PC1 = %.1f%%",
                n_pat, n_ev, mean_corr, pc1_var_pct)

    # ---- Orientation on paclitaxel --------------------------------------
    raw_r = float(np.corrcoef(pc1_score, X[ORIENT_DRUG].values)[0, 1])
    sign = 1.0 if raw_r >= 0 else -1.0
    pc1_oriented = pd.Series(sign * pc1_score, index=X.index, name="pc1_oriented")
    logger.info("PC1 raw r with %s AUC = %.4f, sign flipped: %s",
                ORIENT_DRUG, raw_r, sign < 0)

    rows = [
        {"metric": "n_patients", "value": n_pat},
        {"metric": "n_events", "value": n_ev},
        {"metric": "pc1_explained_var_pct", "value": pc1_var_pct},
        {"metric": "pc2_explained_var_pct",
         "value": round(100 * float(pca.explained_variance_ratio_[1]), 2)},
        {"metric": f"pc1_raw_corr_{ORIENT_DRUG}_auc", "value": round(raw_r, 4)},
        {"metric": "pc1_sign_flipped", "value": int(sign < 0)},
    ]
    for drug in [ORIENT_DRUG] + PARP_DRUGS:
        r = float(pc1_oriented.corr(X[drug]))
        rows.append({"metric": f"pc1_oriented_corr_{drug}_auc", "value": round(r, 4)})
        logger.info("Oriented PC1 vs %s predicted AUC: r = %.4f", drug, r)

    # ---- TF activities of the same patients ------------------------------
    tf = pd.read_csv(TF_PATH, index_col=0,
                     usecols=lambda c: c in ["", "Unnamed: 0"] + TFS_OF_INTEREST)
    missing = X.index.difference(tf.index)
    if len(missing):
        raise RuntimeError(f"{len(missing)} Cox patients have no SCAN-B TF activities")
    tf = tf.loc[X.index]
    for name in TFS_OF_INTEREST:
        r = float(pc1_oriented.corr(tf[name]))
        rows.append({"metric": f"pc1_oriented_corr_tf_{name}", "value": round(r, 4)})
        logger.info("Oriented PC1 vs %s TF activity: r = %.4f", name, r)

    # ---- Survival of the oriented score ----------------------------------
    p_or, z_or = cox_stats(pc1_oriented.values, surv, pam_dummies)
    if not np.isclose(abs(z_or), abs(pc1_z), rtol=1e-6):
        raise RuntimeError(f"Oriented PC1 |z| {abs(z_or):.4f} differs from the "
                           f"unoriented |z| {abs(pc1_z):.4f}")
    rows.append({"metric": "pc1_oriented_survival_z", "value": round(z_or, 3)})
    logger.info("Oriented PC1 ~ OS (PAM50-adjusted Cox): z = %.3f, p = %.3g", z_or, p_or)

    # ---- Inter-drug correlation: sign and magnitude over unique pairs ----
    iu = np.triu_indices(corr.shape[0], k=1)
    pairs = corr.values[iu]
    pairs = pairs[~np.isnan(pairs)]
    # The off-diagonal mean of main() must equal the mean over unique pairs.
    if not np.isclose(float(np.mean(pairs)), mean_corr, rtol=0, atol=1e-9):
        raise RuntimeError(f"Mean over unique pairs {np.mean(pairs):.6f} differs from "
                           f"the off-diagonal mean {mean_corr:.6f}")
    rows += [
        {"metric": "inter_drug_corr_mean", "value": round(mean_corr, 4)},
        {"metric": "inter_drug_abs_corr_mean", "value": round(float(np.mean(np.abs(pairs))), 4)},
        {"metric": "n_drug_pairs", "value": int(pairs.size)},
        {"metric": "pct_drug_pairs_negative_corr",
         "value": round(100 * float(np.mean(pairs < 0)), 2)},
    ]
    logger.info("Inter-drug pairs: %d, mean |r| = %.4f, %.2f%% negative",
                pairs.size, np.mean(np.abs(pairs)), 100 * np.mean(pairs < 0))

    out = pd.DataFrame(rows)
    os.makedirs(REAUDIT, exist_ok=True)
    out.to_csv(REAUDIT / "pc1_orientation.csv", index=False)
    logger.info("Wrote %s\n%s", REAUDIT / "pc1_orientation.csv", out.to_string(index=False))
    return out


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
    pc1_p, pc1_z = cox_stats(pc1_score, surv, pam_dummies)
    logger.info("PC1 ~ OS (PAM50-adjusted Cox): z=%.2f, p=%.3g  -> %s",
                pc1_z, pc1_p, "assoc" if pc1_p < 0.05 else "NOT associated")

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
        # Unrounded, so that the threshold quoted in the text can be checked.
        # The p-value underflows to 0 for very large |z|, hence the Wald z too.
        {"metric": "pc1_survival_pvalue", "value": pc1_p},
        {"metric": "pc1_survival_z", "value": round(pc1_z, 3)},
        {"metric": "n_tested", "value": N_TOP},
        {"metric": "n_nominal_sig_raw", "value": n_raw_sig},
        {"metric": "n_nominal_sig_residualized", "value": n_res_sig},
    ])
    summary.to_csv(RESULTS / "pca_residualization_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))

    # ---- Second output: what PC1 is (after the two files above) ---------
    pc1_orientation(X, surv, pam_dummies, pca, pc1_score, pc1_z, corr, mean_corr)
    logger.info("DONE")


if __name__ == "__main__":
    main()
