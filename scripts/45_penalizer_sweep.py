#!/usr/bin/env python3
"""
PRECISION - Cox penalizer sensitivity sweep on SCAN-B per-drug survival.

Backs the penalizer-sensitivity claims that previously had no reproducing
script (hard-coded in the supplementary figure and quoted in the Discussion).
For each penalizer it refits the per-drug PAM50-adjusted Cox across all drugs
and reports, both for the full drug set (FDR, quoted in the Discussion) and for
the top-200-by-prediction-variance subset (nominal, shown in Supp Fig), the
number of significant drugs and the hazard-ratio range.

Output: results/v8_multicohort/penalizer_sweep.csv
  columns: penalizer, n_drugs, n_nominal_all, n_fdr_all, hr_min_all, hr_max_all,
           n_nominal_top200, hr_min_top200, hr_max_top200

Run from project root:
    python scripts/45_penalizer_sweep.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats as sstats

from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pensweep")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"
PRED_PATH = RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv"
PENALIZERS = [0.01, 0.1, 0.5, 1.0]
N_TOP = 200


def load_matrix():
    preds = pd.read_csv(PRED_PATH, index_col=0)
    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")].set_index("sample_id")
    clin["os_time"] = pd.to_numeric(clin.get("os_time"), errors="coerce")
    clin["os_event"] = pd.to_numeric(clin.get("os_event"), errors="coerce")
    merged = clin[["os_time", "os_event", "pam50"]].join(preds, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "pam50"])
    merged = merged[merged["os_time"] > 0]
    drug_cols = [c for c in preds.columns if c in merged.columns]
    return merged[drug_cols].copy(), merged[["os_time", "os_event", "pam50"]].copy()


def cox_hr_p(x, surv, pam, penalizer):
    cd = pd.DataFrame({"os_time": surv["os_time"].values,
                       "os_event": surv["os_event"].values, "x": x},
                      index=surv.index)
    cd = pd.concat([cd, pam], axis=1).dropna()
    if cd["x"].nunique() < 3:
        return np.nan, np.nan
    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cd, duration_col="os_time", event_col="os_event")
        return float(cph.summary.loc["x", "exp(coef)"]), float(cph.summary.loc["x", "p"])
    except Exception:
        return np.nan, np.nan


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    X, surv = load_matrix()
    pam = pd.get_dummies(surv["pam50"], prefix="sub", drop_first=True, dtype=int)
    logger.info("Matrix: %d patients x %d drugs, %d events",
                X.shape[0], X.shape[1], int(surv["os_event"].sum()))
    top200 = X.var().sort_values(ascending=False).head(N_TOP).index.tolist()

    rows = []
    for pen in PENALIZERS:
        hrs, ps = {}, {}
        for d in X.columns:
            hr, p = cox_hr_p(X[d].values, surv, pam, pen)
            hrs[d], ps[d] = hr, p
        pv = pd.Series(ps).dropna()
        hv = pd.Series(hrs).reindex(pv.index)
        # BH FDR over all drugs
        order = pv.sort_values()
        m = len(order)
        padj = (order.values * m / (np.arange(1, m + 1)))
        padj = np.minimum.accumulate(padj[::-1])[::-1]
        padj = pd.Series(np.clip(padj, 0, 1), index=order.index)
        n_fdr = int((padj < 0.05).sum())
        n_nom = int((pv < 0.05).sum())
        t2 = [d for d in top200 if d in pv.index]
        pv2 = pv[t2]
        hv2 = hv[t2]
        rows.append({
            "penalizer": pen, "n_drugs": int(m),
            "n_nominal_all": n_nom, "n_fdr_all": n_fdr,
            "hr_min_all": float(hv.min()), "hr_max_all": float(hv.max()),
            "n_nominal_top200": int((pv2 < 0.05).sum()),
            "hr_min_top200": float(hv2.min()), "hr_max_top200": float(hv2.max()),
        })
        logger.info("pen=%.2f: all FDR<0.05=%d, all nominal=%d, HR[%.2f,%.2f]; "
                    "top200 nominal=%d HR[%.2f,%.2f]", pen, n_fdr, n_nom,
                    hv.min(), hv.max(), int((pv2 < 0.05).sum()),
                    hv2.min(), hv2.max())

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "penalizer_sweep.csv", index=False)
    logger.info("Wrote penalizer_sweep.csv")


if __name__ == "__main__":
    main()
