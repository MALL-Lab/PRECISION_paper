#!/usr/bin/env python3
"""
PRECISION - TF-level univariate Cox on SCAN-B overall survival.

Backs the "TF-level validation confirmed biological coherence" claim in the
Results (the manuscript cited MYC/TP53/E2F1 as examples but had no script that
produced the numbers, and two of the three directions were biologically
reversed). This script fits a univariate penalized-free Cox model of overall
survival on each TF activity in SCAN-B (patient-deduplicated) and writes the
full table, so any TF quoted in the text is reproducible.

Convention: HR is per unit of TF activity. HR > 1 = higher TF activity
associated with worse overall survival; HR < 1 = better.

Output: results/v8_multicohort/tf_level_survival_scanb.csv
  columns: tf, HR, pvalue, coef, n

Run from project root:
    python scripts/44_tf_level_cox.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tf_cox")

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"
SEED = 42
HIGHLIGHT = ["MYC", "TP53", "E2F1"]  # examples quoted in the Results text


def load_scanb():
    tf = pd.read_csv(DATA / "precision_processed/tf_activities_scanb.csv", index_col=0)
    # SCAN-B TF matrix is per-sample; collapse to short sample id and dedup
    tf["short"] = tf.index.str.replace(r"\..*", "", regex=True)
    tf = tf[~tf["short"].duplicated()]
    tf.index = tf["short"]
    tf = tf.drop(columns="short")

    clin = pd.read_csv(DATA / "precision_processed/scanb_clinical.csv")
    clin = clin[~clin["patient_id"].duplicated(keep="first")].set_index("sample_id")
    clin["os_time"] = pd.to_numeric(clin.get("os_time"), errors="coerce")
    clin["os_event"] = pd.to_numeric(clin.get("os_event"), errors="coerce")
    clin = clin.dropna(subset=["os_time", "os_event"])
    clin = clin[clin["os_time"] > 0]

    common = clin.index.intersection(tf.index)
    return tf.loc[common], clin.loc[common]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tf, clin = load_scanb()
    n = len(tf)
    logger.info("SCAN-B TF-level Cox: %d patients, %d TFs", n, tf.shape[1])

    t = clin["os_time"].values
    e = clin["os_event"].values
    rows = []
    for g in tf.columns:
        x = pd.to_numeric(tf[g], errors="coerce").values.astype(float)
        d = pd.DataFrame({"t": t, "e": e, "x": x}).dropna()
        if len(d) < 100 or d["x"].std() == 0:
            continue
        try:
            cph = CoxPHFitter().fit(d, "t", "e")
            s = cph.summary.loc["x"]
            rows.append({"tf": g, "HR": float(s["exp(coef)"]),
                         "pvalue": float(s["p"]), "coef": float(s["coef"]),
                         "n": len(d)})
        except Exception:
            continue

    res = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    res.to_csv(RESULTS / "tf_level_survival_scanb.csv", index=False)
    logger.info("Wrote tf_level_survival_scanb.csv (%d TFs)", len(res))

    logger.info("Examples quoted in the Results text:")
    for g in HIGHLIGHT:
        r = res[res["tf"] == g]
        if len(r):
            r = r.iloc[0]
            direction = "worse" if r["HR"] > 1 else "better"
            logger.info("  %-6s HR=%.2f p=%.1e -> higher activity, %s survival",
                        g, r["HR"], r["pvalue"], direction)


if __name__ == "__main__":
    main()
