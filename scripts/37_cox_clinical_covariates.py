#!/usr/bin/env python3
"""
PRECISION - Cox multivariate with full clinical covariates (internal clinical review).

The internal clinical review asked: why only PAM50-adjusted?
Every oncologist reviewing this paper will ask about stage, tumor size,
nodal status, age at diagnosis, ER/PR, treatment exposure.

This script re-runs the Cox model for the 7 Table 5 candidates + 4 positive
controls adjusting for the full clinical panel available per cohort:
  - SCAN-B:   predicted_auc + PAM50 + age + t.stage + grade + ER + PR + LN+
  - METABRIC: predicted_auc + PAM50 + age + ER + HER2 + NPI + LN+
  - TCGA:     skipped (150 events, underpowered)

Outputs in paper/results/:
  cox_clinical_adjusted_scanb.csv
  cox_clinical_adjusted_metabric.csv
  cox_clinical_comparison.csv

For each drug reports the HR and padj from the clinical-adjusted model,
alongside the PAM50-only HR from the original analysis for comparison.

Run from project root:
    python scripts/37_cox_clinical_covariates.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from config import DATA_DIR, RESULTS_DIR

# Outputs go to RESULTS_DIR; generate_paper_results.py copies the comparison
# table cited in the manuscript to paper/results/.
OUT_DIR = RESULTS_DIR / "v8_multicohort"
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cox_clinical")

SEED = 42

TARGETS = [
    # Table 5 candidates
    "osimertinib", "saracatinib", "erlotinib", "brigatinib",
    "pelitinib", "entinostat", "trametinib",
    # Positive controls
    "paclitaxel", "docetaxel", "epirubicin", "olaparib",
]


def predict_scanb():
    """Train per-drug Ridge on PRISM train cells, predict on SCAN-B."""
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(
        DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0,
    )
    tf_scanb = pd.read_csv(
        DATA_DIR / "precision_processed/tf_activities_scanb.csv", index_col=0,
    )

    # Dedup SCAN-B TF by short_id
    tf_scanb["short_id"] = tf_scanb.index.str.replace(r"\..*", "", regex=True)
    tf_scanb = tf_scanb[~tf_scanb["short_id"].duplicated()]
    tf_scanb.index = tf_scanb["short_id"]
    tf_scanb = tf_scanb.drop(columns=["short_id"])

    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)
    shared = sorted(set(tf_prism.columns) & set(tf_scanb.columns))

    train_prism = tf_prism.loc[[c for c in train_cells if c in tf_prism.index],
                                 shared]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism)
    prism_qt = pd.DataFrame(qt.transform(tf_prism[shared]),
                             index=tf_prism.index, columns=shared)
    scanb_qt = pd.DataFrame(qt.transform(tf_scanb[shared]),
                             index=tf_scanb.index, columns=shared)

    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_scanb = scaler.transform(scanb_qt)

    preds = {}
    for d in TARGETS:
        if d not in train_resp.columns:
            continue
        y = train_resp[d].values
        m = ~np.isnan(y)
        if m.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[m], y[m])
        preds[d] = ridge.predict(X_scanb)

    return pd.DataFrame(preds, index=scanb_qt.index)


def predict_metabric():
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(
        DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0,
    )
    tf_met = pd.read_csv(
        DATA_DIR / "precision_processed/tf_activities_metabric.csv", index_col=0,
    )

    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)
    shared = sorted(set(tf_prism.columns) & set(tf_met.columns))

    train_prism = tf_prism.loc[[c for c in train_cells if c in tf_prism.index],
                                 shared]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism)
    prism_qt = pd.DataFrame(qt.transform(tf_prism[shared]),
                             index=tf_prism.index, columns=shared)
    met_qt = pd.DataFrame(qt.transform(tf_met[shared]),
                           index=tf_met.index, columns=shared)

    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_met = scaler.transform(met_qt)

    preds = {}
    for d in TARGETS:
        if d not in train_resp.columns:
            continue
        y = train_resp[d].values
        m = ~np.isnan(y)
        if m.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[m], y[m])
        preds[d] = ridge.predict(X_met)

    return pd.DataFrame(preds, index=met_qt.index)


def cox_scanb(preds_df):
    clin = pd.read_csv(DATA_DIR / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")]
    clin = clin.set_index("sample_id")
    clin["os_time"] = pd.to_numeric(clin.get("os_time"), errors="coerce")
    clin["os_event"] = pd.to_numeric(clin.get("os_event"), errors="coerce")
    clin["age"] = pd.to_numeric(clin.get("age"), errors="coerce")
    clin["grade"] = pd.to_numeric(clin.get("grade"), errors="coerce")
    clin["lymphNodePos"] = pd.to_numeric(clin.get("lymphNodePos"), errors="coerce")

    merged = clin[["os_time", "os_event", "pam50", "age", "t.stage", "grade",
                    "er", "pr", "lymphNodePos"]].join(preds_df, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "pam50", "age",
                                     "t.stage", "grade"])
    merged = merged[merged["os_time"] > 0]

    # Encode PAM50 and t.stage as dummies
    pam = pd.get_dummies(merged["pam50"], prefix="sub", drop_first=True,
                         dtype=int)
    stg = pd.get_dummies(merged["t.stage"], prefix="st", drop_first=True,
                         dtype=int)
    er_b = (merged["er"].astype(str).str.upper().str.startswith("P")).astype(int)
    pr_b = (merged["pr"].astype(str).str.upper().str.startswith("P")).astype(int)

    base = pd.concat([
        merged[["os_time", "os_event", "age", "grade", "lymphNodePos"]],
        pam, stg,
        pd.Series(er_b.values, index=merged.index, name="er_pos"),
        pd.Series(pr_b.values, index=merged.index, name="pr_pos"),
    ], axis=1)

    logger.info("SCAN-B clinical-adjusted Cox: %d patients, %d events",
                len(base), int(base["os_event"].sum()))

    results = []
    for drug in TARGETS:
        if drug not in preds_df.columns:
            continue
        common = merged.index.intersection(preds_df.index)
        cox_data = base.loc[common].copy()
        cox_data["predicted_auc"] = preds_df.loc[common, drug].values
        cox_data = cox_data.dropna()
        if len(cox_data) < 100:
            continue
        non_const = cox_data.columns[cox_data.nunique() > 1]
        cox_data = cox_data[non_const]
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR_clinical": hr, "pvalue": p})
        except Exception as e:
            logger.warning("  %s: %s", drug, e)
    if not results:
        logger.error("SCAN-B: no drug produced valid Cox fit")
        return pd.DataFrame(columns=["drug", "HR_clinical", "pvalue", "padj"])
    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    return df.sort_values("pvalue").reset_index(drop=True)


def cox_metabric(preds_df):
    clin = pd.read_csv(DATA_DIR / "precision_processed/metabric_clinical.csv")
    # Look up best keys
    clin = clin.set_index("patientId") if "patientId" in clin.columns else clin
    clin["os_time"] = pd.to_numeric(clin.get("OS_MONTHS"), errors="coerce") * 30.44
    clin["os_event"] = clin.get("OS_STATUS", pd.Series()).astype(str).str.extract(
        r"^(\d)")[0]
    clin["os_event"] = pd.to_numeric(clin["os_event"], errors="coerce")
    clin["age"] = pd.to_numeric(clin.get("AGE_AT_DIAGNOSIS"), errors="coerce")
    clin["ln_pos"] = pd.to_numeric(clin.get("LYMPH_NODES_EXAMINED_POSITIVE"),
                                     errors="coerce")
    clin["npi"] = pd.to_numeric(clin.get("NPI"), errors="coerce")

    merged = clin[["os_time", "os_event", "PAM50", "age", "ER_IHC", "HER2_SNP6",
                    "ln_pos", "npi"]].join(preds_df, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "PAM50", "age"])
    merged = merged[merged["os_time"] > 0]

    pam = pd.get_dummies(merged["PAM50"], prefix="sub", drop_first=True,
                         dtype=int)
    er_b = (merged["ER_IHC"].astype(str).str.lower() == "pos").astype(int)
    her2_b = (merged["HER2_SNP6"].astype(str).str.upper()
              .isin(["GAIN", "AMP", "POSITIVE"])).astype(int)

    base = pd.concat([
        merged[["os_time", "os_event", "age", "ln_pos", "npi"]],
        pam,
        pd.Series(er_b.values, index=merged.index, name="er_pos"),
        pd.Series(her2_b.values, index=merged.index, name="her2_pos"),
    ], axis=1)
    base = base.dropna()

    logger.info("METABRIC clinical-adjusted Cox: %d patients, %d events",
                len(base), int(base["os_event"].sum()))

    results = []
    for drug in TARGETS:
        if drug not in preds_df.columns:
            continue
        common = base.index.intersection(preds_df.index)
        cox_data = base.loc[common].copy()
        cox_data["predicted_auc"] = preds_df.loc[common, drug].values
        non_const = cox_data.columns[cox_data.nunique() > 1]
        cox_data = cox_data[non_const]
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR_clinical": hr, "pvalue": p})
        except Exception as e:
            logger.warning("  %s: %s", drug, e)
    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    return df.sort_values("pvalue").reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    logger.info("Predicting SCAN-B drug sensitivity...")
    preds_s = predict_scanb()
    logger.info("SCAN-B predictions: %s", preds_s.shape)

    logger.info("Predicting METABRIC drug sensitivity...")
    preds_m = predict_metabric()
    logger.info("METABRIC predictions: %s", preds_m.shape)

    logger.info("Cox (SCAN-B, clinical-adjusted)...")
    df_s = cox_scanb(preds_s)
    df_s.to_csv(OUT_DIR / "cox_clinical_adjusted_scanb.csv", index=False)
    logger.info("\n%s", df_s.to_string(index=False))

    logger.info("Cox (METABRIC, clinical-adjusted)...")
    df_m = cox_metabric(preds_m)
    df_m.to_csv(OUT_DIR / "cox_clinical_adjusted_metabric.csv", index=False)
    logger.info("\n%s", df_m.to_string(index=False))

    # Comparison table
    pam_only = pd.read_csv(
        RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv"
    )
    pam_tar = pam_only[pam_only["drug"].isin(TARGETS)].set_index("drug")

    cmp = pd.DataFrame({"drug": TARGETS})
    cmp["HR_pam50_scanb"] = cmp["drug"].map(pam_tar["HR"]).round(2)
    cmp["padj_pam50"] = cmp["drug"].map(pam_tar["padj"]).apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n.a.")
    ds = df_s.set_index("drug")
    cmp["HR_clinical_scanb"] = cmp["drug"].map(ds["HR_clinical"]).round(2)
    cmp["padj_clinical_scanb"] = cmp["drug"].map(ds["padj"]).apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n.a.")
    dm = df_m.set_index("drug")
    cmp["HR_clinical_metabric"] = cmp["drug"].map(dm["HR_clinical"]).round(2)
    cmp["padj_clinical_metabric"] = cmp["drug"].map(dm["padj"]).apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n.a.")

    cmp.to_csv(OUT_DIR / "cox_clinical_comparison.csv", index=False)
    logger.info("\nComparison table (PAM50 only vs full clinical):\n%s",
                cmp.to_string(index=False))


if __name__ == "__main__":
    main()
