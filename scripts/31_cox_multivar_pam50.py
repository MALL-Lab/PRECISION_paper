#!/usr/bin/env python3
"""
PRECISION - Cox multivariate regression adjusting for PAM50 subtype.

Penalized (ridge) Cox per drug: an earlier unpenalized version produced
biologically implausible hazard ratios (up to 274 for rs-17053), which is
why the canonical analysis uses penalizer = 0.1 and reports 0.5 as a
sensitivity setting.

This script is reproducible and writes three output CSVs:
- cox_multivar_pam50_adjusted.csv  (canonical: penalizer=0.1, cited in paper)
- cox_multivar_pam50_pen05.csv     (sensitivity: penalizer=0.5)
- cox_multivar_pam50_pen001.csv    (near-unpenalized replica, internal check)

Dependencies:
- SCAN-B TF activities   (data/precision_processed/tf_activities_scanb.csv)
- SCAN-B clinical        (data/precision_processed/scanb_clinical.csv)
- PRISM TF activities    (data/precision_processed/tf_activities_all_prism.csv)
- PRISM drug response    (data/PRISM/processed/prism_response_matched.csv)

Run from project root:
    python scripts/31_cox_multivar_pam50.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from lifelines import CoxPHFitter

from config import DATA_DIR, RESULTS_DIR
from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cox_multivar_pam50")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v6_strengthen"
SEED = 42


def load_scanb():
    """Load SCAN-B TF activities, clinical, apply patient-level dedup."""
    tf = pd.read_csv(DATA / "precision_processed/tf_activities_scanb.csv", index_col=0)
    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin.drop_duplicates(subset="sample_id")

    tf["short_id"] = tf.index.str.replace(r"\..*", "", regex=True)
    tf = tf[~tf["short_id"].duplicated()]
    tf.index = tf["short_id"]
    tf = tf.drop(columns=["short_id"])

    if "patient_id" in clin.columns:
        n_before = len(clin)
        clin = clin.sort_values("sample_id")
        clin = clin[~clin["patient_id"].duplicated(keep="first")]
        logger.info("SCAN-B patient dedup: %d -> %d", n_before, len(clin))

    clin = clin.set_index("sample_id")
    logger.info("SCAN-B: %d samples x %d TFs, %d clinical rows",
                tf.shape[0], tf.shape[1], len(clin))
    return tf, clin


def fit_predictions(train_tf, train_resp, cohort_tf, shared_tfs, alpha=100):
    """QT batch correction (train-only) + per-drug Ridge predictions."""
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_tf[shared_tfs])
    prism_qt = pd.DataFrame(qt.transform(train_tf[shared_tfs].values),
                            index=train_tf.index, columns=shared_tfs)
    cohort_qt = pd.DataFrame(qt.transform(cohort_tf[shared_tfs].values),
                             index=cohort_tf.index, columns=shared_tfs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(prism_qt)
    X_cohort = scaler.transform(cohort_qt)

    predictions = {}
    for drug in train_resp.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train[mask], y[mask])
        predictions[drug] = ridge.predict(X_cohort)

    return pd.DataFrame(predictions, index=cohort_qt.index)


def run_cox_multivar(pred_df, clin, pam_col, penalizer):
    """Run penalized Cox per drug, adjusted for PAM50 dummies."""
    merged = clin[["os_time", "os_event", pam_col]].copy()
    merged = merged.dropna()
    merged["os_time"] = pd.to_numeric(merged["os_time"], errors="coerce")
    merged["os_event"] = pd.to_numeric(merged["os_event"], errors="coerce")
    merged = merged.dropna()
    merged = merged[merged["os_time"] > 0]

    pam_dummies = pd.get_dummies(merged[pam_col], prefix="sub", drop_first=True)
    base = pd.concat([merged[["os_time", "os_event"]], pam_dummies], axis=1)
    pam_cols = list(pam_dummies.columns)

    common = base.index.intersection(pred_df.index)
    base = base.loc[common]
    pred_df = pred_df.loc[common]

    logger.info("Multivariate Cox (penalizer=%.3f): %d patients, %d events, %d drugs",
                penalizer, len(base), int(base["os_event"].sum()), pred_df.shape[1])

    results = []
    for i, drug in enumerate(pred_df.columns):
        cox_data = base.copy()
        cox_data["predicted_auc"] = pred_df[drug].values
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(cox_data, duration_col="os_time", event_col="os_event")
            hr = cph.hazard_ratios_["predicted_auc"]
            p = cph.summary.loc["predicted_auc", "p"]
            results.append({"drug": drug, "HR": hr, "pvalue": p})
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            logger.info("  processed %d/%d drugs", i + 1, pred_df.shape[1])

    df = pd.DataFrame(results)
    df["padj"] = stats.false_discovery_control(df["pvalue"])
    df = df.sort_values("pvalue").reset_index(drop=True)

    n_sig = int((df["padj"] < 0.05).sum())
    sig = df[df["padj"] < 0.05]
    hr_min = sig["HR"].min() if len(sig) else np.nan
    hr_max = sig["HR"].max() if len(sig) else np.nan

    logger.info("  %d/%d FDR<0.05, HR range among sig: [%.3f, %.3f]",
                n_sig, len(df), hr_min, hr_max)
    return df


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # PRISM
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(drug_response, test_frac=0.2, seed=SEED)

    # SCAN-B
    tf_scanb, clin_scanb = load_scanb()

    # Parse survival
    clin_scanb["os_time"] = pd.to_numeric(clin_scanb.get("os_time"), errors="coerce")
    clin_scanb["os_event"] = pd.to_numeric(clin_scanb.get("os_event"), errors="coerce")

    # PAM50 column detection
    pam_col = None
    for cand in ["pam50", "PAM50", "PAM50_short", "NCN_PAM50"]:
        if cand in clin_scanb.columns:
            pam_col = cand
            break
    if pam_col is None:
        raise ValueError(f"PAM50 column not found in SCAN-B clinical: {list(clin_scanb.columns)}")
    logger.info("PAM50 column: %s", pam_col)

    # Align TFs
    shared_tfs = sorted(set(tf_prism.columns) & set(tf_scanb.columns))
    logger.info("Shared TFs PRISM-SCANB: %d", len(shared_tfs))

    # Train-only PRISM for QT + Ridge
    train_tf = tf_prism.loc[[c for c in train_cells if c in tf_prism.index]]
    train_resp = drug_response.loc[train_tf.index]

    # Per-drug predictions on SCAN-B
    pred_df = fit_predictions(train_tf, train_resp, tf_scanb, shared_tfs, alpha=100)
    logger.info("Predictions: %d samples x %d drugs", *pred_df.shape)

    # Run Cox at three penalizer values
    configs = [
        ("adjusted",  0.1),   # canonical (cited in paper)
        ("pen05",     0.5),   # conservative sensitivity
        ("pen001",    0.01),  # near-unpenalized replica of legacy v6
    ]
    for tag, pen in configs:
        df = run_cox_multivar(pred_df, clin_scanb, pam_col, pen)
        out = RESULTS / f"cox_multivar_pam50_{tag}.csv"
        df.to_csv(out, index=False)
        logger.info("  wrote %s", out)

    logger.info("DONE")


if __name__ == "__main__":
    main()
