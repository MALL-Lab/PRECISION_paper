#!/usr/bin/env python3
"""
PRECISION - ComBat vs QuantileTransformer sensitivity (internal methods review).

Internal methods review: QuantileTransformer as "batch correction" between PRISM
cell lines and SCAN-B patients is pragmatic but not the standard in the
cross-platform expression literature. A minimal sensitivity check is to
replicate the SCAN-B Cox analysis with ComBat empirical-Bayes adjustment
on the TF activities, and to show that the number of FDR-significant
drugs is consistent.

We operate at the TF-activity level (not raw gene expression): the model
consumes TF activities computed by ULM independently on PRISM and SCAN-B,
so the actionable batch variable is the cohort indicator on the
(PRISM + SCAN-B) joint TF matrix.

Outputs in paper/results/:
  combat_cox_scanb.csv            - Cox PAM50-adjusted with ComBat-corrected features
  combat_vs_qt_comparison.csv     - per-drug HR/padj under both strategies
  combat_vs_qt_summary.csv        - counts of FDR-sig drugs, direction overlap

Run from project root:
    python scripts/41_combat_sensitivity.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("combat_sens")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"  # combat_vs_qt_summary.csv is copied to paper/results/ by generate_paper_results.py
QT_BASELINE = RESULTS_DIR / "v8_multicohort" / "survival_SCANB.csv"  # multicohort stage output
SEED = 42


def run_pipeline_combat():
    """Ridge per-drug + Cox PAM50-adjusted, using ComBat-corrected TF matrix
    over PRISM+SCAN-B joint."""
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
    tf_prism = tf_prism[shared]
    tf_scanb = tf_scanb[shared]

    # Join PRISM (train subset) + SCAN-B, ComBat with batch indicator.
    train_prism_tf = tf_prism.loc[[c for c in train_cells
                                    if c in tf_prism.index]]
    joint = pd.concat([train_prism_tf, tf_scanb], axis=0)
    batch = np.array(["PRISM"] * len(train_prism_tf)
                     + ["SCANB"] * len(tf_scanb))
    logger.info("ComBat joint matrix: %d samples (PRISM=%d, SCAN-B=%d) x %d TFs",
                joint.shape[0], len(train_prism_tf), len(tf_scanb),
                joint.shape[1])

    # pycombat expects genes x samples format
    corrected = pycombat(joint.T, batch).T
    logger.info("ComBat correction applied")

    prism_c = corrected.iloc[:len(train_prism_tf)]
    scanb_c = corrected.iloc[len(train_prism_tf):]

    # Standard-scale features like original pipeline
    scaler = StandardScaler()
    X_train = scaler.fit_transform(prism_c)
    X_scanb = scaler.transform(scanb_c)

    train_resp = drug_response.loc[prism_c.index]
    drugs = train_resp.columns.tolist()
    logger.info("Fitting Ridge per-drug (%d drugs)...", len(drugs))
    preds = {}
    for d in drugs:
        y = train_resp[d].values
        m = ~np.isnan(y)
        if m.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[m], y[m])
        preds[d] = ridge.predict(X_scanb)
    preds_df = pd.DataFrame(preds, index=scanb_c.index)

    # Cox PAM50-adjusted per drug
    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")].set_index("sample_id")
    clin["os_time"] = pd.to_numeric(clin.get("os_time"), errors="coerce")
    clin["os_event"] = pd.to_numeric(clin.get("os_event"), errors="coerce")

    merged = clin[["os_time", "os_event", "pam50"]].join(preds_df, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "pam50"])
    merged = merged[merged["os_time"] > 0]
    pam = pd.get_dummies(merged["pam50"], prefix="sub", drop_first=True,
                         dtype=int)
    base = pd.concat([merged[["os_time", "os_event"]], pam], axis=1)

    logger.info("Cox: %d patients, %d events, %d drugs",
                len(base), int(base["os_event"].sum()), len(preds))

    results = []
    common = base.index.intersection(preds_df.index)
    base = base.loc[common]
    aligned_preds = preds_df.loc[common]
    for i, d in enumerate(preds.keys()):
        cd = base.copy()
        cd["predicted_auc"] = aligned_preds[d].values
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cd, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": d, "HR": hr, "pvalue": p})
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            logger.info("  processed %d/%d drugs", i + 1, len(preds))

    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    return df.sort_values("pvalue").reset_index(drop=True)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    logger.info("Running ComBat-corrected pipeline...")
    df_combat = run_pipeline_combat()
    df_combat.to_csv(RESULTS / "combat_cox_scanb.csv", index=False)
    n_combat = int((df_combat["padj"] < 0.05).sum())
    logger.info("ComBat: %d / %d drugs FDR<0.05", n_combat, len(df_combat))

    # Baseline QT from existing canonical CSV
    qt = pd.read_csv(QT_BASELINE)
    n_qt = int((qt["padj"] < 0.05).sum())
    logger.info("QT baseline: %d / %d drugs FDR<0.05", n_qt, len(qt))

    # Comparison
    merged = qt.merge(df_combat, on="drug", suffixes=("_qt", "_combat"))
    merged["sig_qt"] = merged["padj_qt"] < 0.05
    merged["sig_combat"] = merged["padj_combat"] < 0.05
    merged["both_sig"] = merged["sig_qt"] & merged["sig_combat"]
    merged["same_direction"] = (
        ((merged["HR_qt"] > 1) & (merged["HR_combat"] > 1))
        | ((merged["HR_qt"] < 1) & (merged["HR_combat"] < 1))
    )
    merged.to_csv(RESULTS / "combat_vs_qt_comparison.csv", index=False)

    n_both = int(merged["both_sig"].sum())
    n_only_qt = int((merged["sig_qt"] & ~merged["sig_combat"]).sum())
    n_only_combat = int((~merged["sig_qt"] & merged["sig_combat"]).sum())
    both_sig = merged[merged["both_sig"]]
    n_same_dir_both = int(both_sig["same_direction"].sum())
    total_same_dir = int(merged["same_direction"].sum())

    summary = pd.DataFrame([
        {"metric": "n_sig_qt", "value": n_qt},
        {"metric": "n_sig_combat", "value": n_combat},
        {"metric": "n_sig_both", "value": n_both},
        {"metric": "n_sig_only_qt", "value": n_only_qt},
        {"metric": "n_sig_only_combat", "value": n_only_combat},
        {"metric": "direction_concordant_overall", "value": total_same_dir},
        {"metric": "direction_concordant_among_both_sig",
         "value": n_same_dir_both},
        {"metric": "direction_concordant_pct_among_both_sig",
         "value": round(100 * n_same_dir_both / max(n_both, 1), 1)},
    ])
    summary.to_csv(RESULTS / "combat_vs_qt_summary.csv", index=False)

    logger.info("\n==================================================")
    logger.info("ComBat vs QuantileTransformer summary")
    logger.info("==================================================")
    for _, r in summary.iterrows():
        logger.info("  %-40s %s", r["metric"], r["value"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
