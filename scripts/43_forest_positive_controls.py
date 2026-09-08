#!/usr/bin/env python3
"""
PRECISION - Forest-plot data for BC-relevant drugs across SCAN-B / METABRIC.

Internal review: the cross-cohort discordance for the clinically established
breast-cancer drugs deserves a clean HR figure. Five drugs are individually
FDR-significant in SCAN-B (paclitaxel, docetaxel, epirubicin, olaparib,
talazoparib) but only epirubicin is significant in METABRIC.

This script refits the PAM50-adjusted penalized Cox model (penalizer=0.1) for
those five drugs in both cohorts and extracts the hazard ratio with its 95%
confidence interval (lifelines reports these directly), so a forest plot can
show magnitude, direction, and the cross-cohort discordance at a glance.

Output: results/v8_multicohort/forest_positive_controls.csv
  columns: drug, cohort, HR, CI_low, CI_high, pvalue, padj_in_cohort

Run from project root:
    python scripts/43_forest_positive_controls.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from config import DATA_DIR, RESULTS_DIR
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("forest")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"
SEED = 42
DRUGS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]


def predict(cohort_tf_path, dedup_short_id=False):
    """Per-drug Ridge (QT batch correction, train-only) for the target drugs."""
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(
        DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    tf_c = pd.read_csv(cohort_tf_path, index_col=0)
    if dedup_short_id:
        tf_c["short_id"] = tf_c.index.str.replace(r"\..*", "", regex=True)
        tf_c = tf_c[~tf_c["short_id"].duplicated()]
        tf_c.index = tf_c["short_id"]
        tf_c = tf_c.drop(columns=["short_id"])

    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)
    shared = sorted(set(tf_prism.columns) & set(tf_c.columns))
    train_prism = tf_prism.loc[[c for c in train_cells if c in tf_prism.index], shared]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism)
    prism_qt = pd.DataFrame(qt.transform(tf_prism[shared]),
                            index=tf_prism.index, columns=shared)
    c_qt = pd.DataFrame(qt.transform(tf_c[shared]), index=tf_c.index, columns=shared)

    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_c = scaler.transform(c_qt)

    preds = {}
    for d in DRUGS:
        if d not in train_resp.columns:
            continue
        y = train_resp[d].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[mask], y[mask])
        preds[d] = ridge.predict(X_c)
    return pd.DataFrame(preds, index=c_qt.index)


def cox_with_ci(preds_df, clin, cohort):
    """PAM50-adjusted penalized Cox per drug; return HR + 95% CI + p."""
    rows = []
    pam = pd.get_dummies(clin["pam50"], prefix="sub", drop_first=True, dtype=int)
    base = pd.concat([clin[["os_time", "os_event"]], pam], axis=1)
    common = base.index.intersection(preds_df.index)
    base = base.loc[common]
    preds_df = preds_df.loc[common]
    for d in DRUGS:
        if d not in preds_df.columns:
            continue
        cd = base.copy()
        cd["predicted_auc"] = preds_df[d].values
        cd = cd.dropna()
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cd, duration_col="os_time", event_col="os_event")
            s = cph.summary.loc["predicted_auc"]
            rows.append({
                "drug": d, "cohort": cohort,
                "HR": float(s["exp(coef)"]),
                "CI_low": float(s["exp(coef) lower 95%"]),
                "CI_high": float(s["exp(coef) upper 95%"]),
                "pvalue": float(s["p"]),
            })
        except Exception as exc:
            logger.warning("  %s (%s): %s", d, cohort, exc)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # SCAN-B
    logger.info("SCAN-B predictions + Cox...")
    preds_s = predict(DATA / "precision_processed/tf_activities_scanb.csv",
                      dedup_short_id=True)
    clin_s = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin_s = clin_s[~clin_s["patient_id"].duplicated(keep="first")].set_index("sample_id")
    clin_s["os_time"] = pd.to_numeric(clin_s.get("os_time"), errors="coerce")
    clin_s["os_event"] = pd.to_numeric(clin_s.get("os_event"), errors="coerce")
    clin_s = clin_s.dropna(subset=["os_time", "os_event", "pam50"])
    clin_s = clin_s[clin_s["os_time"] > 0]
    df_s = cox_with_ci(preds_s, clin_s, "SCAN-B")

    # METABRIC
    logger.info("METABRIC predictions + Cox...")
    preds_m = predict(DATA / "precision_processed/tf_activities_metabric.csv")
    clin_m = pd.read_csv(DATA / "precision_processed/metabric_clinical.csv")
    idx_col = "patientId" if "patientId" in clin_m.columns else clin_m.columns[0]
    clin_m = clin_m.set_index(idx_col)
    clin_m["os_time"] = pd.to_numeric(clin_m.get("OS_MONTHS"), errors="coerce") * 30.44
    clin_m["os_event"] = clin_m.get("OS_STATUS", pd.Series()).astype(str).str.extract(r"^(\d)")[0]
    clin_m["os_event"] = pd.to_numeric(clin_m["os_event"], errors="coerce")
    clin_m["pam50"] = clin_m.get("PAM50")
    clin_m = clin_m.dropna(subset=["os_time", "os_event", "pam50"])
    clin_m = clin_m[clin_m["os_time"] > 0]
    df_m = cox_with_ci(preds_m, clin_m, "METABRIC")

    out = pd.concat([df_s, df_m], ignore_index=True)
    # FDR within cohort from the canonical survival CSVs (padj_in_cohort)
    surv = {
        "SCAN-B": pd.read_csv(RESULTS / "survival_SCANB.csv").set_index("drug"),
        "METABRIC": pd.read_csv(RESULTS / "survival_METABRIC.csv").set_index("drug"),
    }
    out["padj_in_cohort"] = out.apply(
        lambda r: surv[r["cohort"]].loc[r["drug"], "padj"]
        if r["drug"] in surv[r["cohort"]].index else np.nan, axis=1)
    out.to_csv(RESULTS / "forest_positive_controls.csv", index=False)
    logger.info("\n%s", out.to_string(index=False))
    logger.info("Wrote forest_positive_controls.csv")


if __name__ == "__main__":
    main()
