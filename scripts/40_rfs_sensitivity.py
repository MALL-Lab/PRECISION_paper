#!/usr/bin/env python3
"""
PRECISION - RFS (relapse-free survival) sensitivity analysis (internal clinical review).

Internal clinical review: SCAN-B has both OS and RFS. OS-only
analysis is a common simplification but clinical BC papers routinely
report RFS as sensitivity. If the TF-based drug sensitivity signal is
genuinely related to tumour biology (not just end-of-life outcomes), it
should replicate with RFS as outcome.

This script duplicates the Cox multivariate PAM50-adjusted analysis for
the 7 Table 5 candidates + 4 positive controls with RFS instead of OS.

Outputs in paper/results/:
  cox_rfs_scanb.csv         - RFS Cox, PAM50-adjusted, penalizer=0.1
  cox_rfs_comparison.csv    - side-by-side HR_OS vs HR_RFS

Run from project root:
    python scripts/40_rfs_sensitivity.py
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
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rfs_sensitivity")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"  # cox_rfs_comparison.csv is copied to paper/results/ by generate_paper_results.py
SEED = 42

TARGETS = [
    "osimertinib", "saracatinib", "erlotinib", "brigatinib",
    "pelitinib", "entinostat", "trametinib",
    "paclitaxel", "docetaxel", "epirubicin", "olaparib",
]


def predict_scanb():
    """Replica of the predictions from script 37, for the target drugs."""
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(
        DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0,
    )
    tf_scanb = pd.read_csv(
        DATA / "precision_processed/tf_activities_scanb.csv", index_col=0,
    )
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


def cox_rfs(preds_df, outcome="rfs"):
    """Cox PAM50-adjusted penalizer=0.1 with either OS (rfs=False) or RFS."""
    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")].set_index("sample_id")

    if outcome == "rfs":
        t = pd.to_numeric(clin.get("rfs_time"), errors="coerce")
        e = pd.to_numeric(clin.get("rfs_event"), errors="coerce")
    elif outcome == "os":
        t = pd.to_numeric(clin.get("os_time"), errors="coerce")
        e = pd.to_numeric(clin.get("os_event"), errors="coerce")
    else:
        raise ValueError(outcome)

    clin["time"] = t
    clin["event"] = e

    merged = clin[["time", "event", "pam50"]].join(preds_df, how="inner")
    merged = merged.dropna(subset=["time", "event", "pam50"])
    merged = merged[merged["time"] > 0]

    pam = pd.get_dummies(merged["pam50"], prefix="sub", drop_first=True,
                         dtype=int)
    base = pd.concat([merged[["time", "event"]], pam], axis=1)

    logger.info("Cox %s-adjusted: %d patients, %d events",
                outcome.upper(), len(base), int(base["event"].sum()))

    results = []
    for drug in TARGETS:
        if drug not in preds_df.columns:
            continue
        common = base.index.intersection(preds_df.index)
        cd = base.loc[common].copy()
        cd["predicted_auc"] = preds_df.loc[common, drug].values
        cd = cd.dropna()
        if len(cd) < 100:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cd, duration_col="time", event_col="event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR": hr, "pvalue": p})
        except Exception as exc:
            logger.warning("  %s: %s", drug, exc)

    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    return df.sort_values("pvalue").reset_index(drop=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    logger.info("Generating SCAN-B per-drug predictions (targets only)...")
    preds = predict_scanb()
    logger.info("Predictions: %s", preds.shape)

    logger.info("=" * 60)
    logger.info("OS Cox (for reference)")
    logger.info("=" * 60)
    df_os = cox_rfs(preds, outcome="os")
    logger.info("\n%s", df_os.to_string(index=False))

    logger.info("=" * 60)
    logger.info("RFS Cox (sensitivity)")
    logger.info("=" * 60)
    df_rfs = cox_rfs(preds, outcome="rfs")
    df_rfs.to_csv(RESULTS / "cox_rfs_scanb.csv", index=False)
    logger.info("\n%s", df_rfs.to_string(index=False))

    # Comparison
    cmp = pd.DataFrame({"drug": TARGETS})
    os_map = df_os.set_index("drug")
    rfs_map = df_rfs.set_index("drug")
    cmp["HR_OS"] = cmp["drug"].map(os_map["HR"]).round(2)
    cmp["padj_OS"] = cmp["drug"].map(os_map["padj"]).apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n.a.")
    cmp["HR_RFS"] = cmp["drug"].map(rfs_map["HR"]).round(2)
    cmp["padj_RFS"] = cmp["drug"].map(rfs_map["padj"]).apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n.a.")
    # Concordance: same direction + both sig
    def _concordant(row):
        if row["HR_OS"] is None or row["HR_RFS"] is None:
            return None
        same = ((row["HR_OS"] > 1) == (row["HR_RFS"] > 1))
        return "Yes" if same else "No"
    cmp["same_direction"] = cmp.apply(_concordant, axis=1)
    cmp.to_csv(RESULTS / "cox_rfs_comparison.csv", index=False)

    logger.info("\nSide-by-side OS vs RFS:\n%s", cmp.to_string(index=False))

    concordant = int((cmp["same_direction"] == "Yes").sum())
    logger.info("Direction-concordant OS vs RFS: %d/%d", concordant, len(cmp))


if __name__ == "__main__":
    main()
