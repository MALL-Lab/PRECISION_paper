#!/usr/bin/env python3
"""Log-rank tests of the SCAN-B Kaplan-Meier panels (Figure 4A).

For each drug of the figure, the survival cohort (one sample per patient,
overall survival present and positive, src/data/scanb_cohort.py) is split at
the median predicted AUC, and overall survival is compared between the two
halves with a two-sided log-rank test. The figure used to compute these
p-values while drawing, so they had no statistics key and no sanity check.

Inputs:
  results/v5_final/scanb_perdrug_predictions.csv   (script 17)
  data/precision_processed/scanb_clinical.csv      (script 09)
Output:
  results/v5_final/scanb_km_logrank.csv with columns drug, n_patients,
  n_events, n_sensitive, n_resistant, chi2, p, median_os_sensitive_years,
  median_os_resistant_years, sensitive_worse (True if the predicted-sensitive
  half has the lower survival at the median follow-up time of the cohort)

Second output, results/v13_reaudit/scanb_km_extras.csv, one row per drug of
the figure, on the same cohort and the same median split:
  - os10_sensitive_pct, os10_resistant_pct, os10_diff_pp: Kaplan-Meier overall
    survival at 10 years (3,652.5 days) in the predicted-sensitive half (AUC
    below the median) and the predicted-resistant half, and their difference
    in percentage points (sensitive minus resistant), with the numbers still
    at risk at 10 years
  - trend_chi2, trend_p, trend_score_u: log-rank test for trend across
    predicted-AUC quintiles. It is implemented as the score test (at beta = 0,
    Breslow handling of tied event times, 1 df) of a Cox model with the
    quintile index 1 to 5 as a numeric covariate. A positive score U means
    that the hazard rises with the predicted AUC.
  - pam50_n_patients, pam50_n_events, pam50_strat_chi2, pam50_strat_p: the
    median split adjusted for PAM50, as the Cox score test of the split
    indicator with PAM50 as strata, among patients with a PAM50 call. The
    halves are those of the figure (median of the whole cohort).
  - median_split_score_chi2: the same score test without strata, which must
    match the log-rank chi2 of the first output (the check of the score test)
  - pam50_n_sensitive, pam50_n_resistant, pct_luma_sensitive,
    pct_basal_sensitive, pct_luma_resistant, pct_basal_resistant: size and
    PAM50 composition (percentage LumA and Basal) of each half among patients
    with a PAM50 call
Before writing it the script reproduces published numbers and raises if they
differ: the survival cohort (7,397 patients, 1,188 events), its PAM50 subset,
which is the Cox cohort of script 21 (7,215 patients, 1,164 events), and the
hazard ratio of each drug in results/v8_multicohort/survival_SCANB.csv
(penalized Cox of script 21, e.g. erlotinib 4.28 and paclitaxel 0.47). The
first output is unchanged by this addition.

Third output, results/v13_reaudit/scanb_km_at_risk.csv, the numbers at risk
printed under the Kaplan-Meier panels of Figure 4, in long format with columns
drug, split (median or quintile), group (sensitive and resistant, or Q1 to Q5),
time_years (the even years with a patient still at risk) and n_at_risk (the
patients with os_time >= time_years * 365.25, the definition of the 10-year
counts of the second output). It holds aggregate counts only, for each drug
and each group, on the same cohort and the same groups as the curves. Before
writing it the script raises unless the halves at 0 years are n_sensitive and
n_resistant of the first output and add up to the 7,397 patients, the halves
at 10 years are the counts of the second output, the total per time is the
same for all drugs and both splits, no count grows with time and the quintiles
at 0 years differ by at most 1. The first two outputs are unchanged by this
addition.

Run from project root:
    python scripts/58_scanb_km_logrank.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2 as chi2_dist

from src.data.scanb_cohort import scanb_patient_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("km_logrank")

PRED_PATH = RESULTS_DIR / "v5_final/scanb_perdrug_predictions.csv"
OUT_PATH = RESULTS_DIR / "v5_final/scanb_km_logrank.csv"
KM_DRUGS = ["paclitaxel", "epirubicin", "erlotinib", "entinostat"]

# Second output, see the module docstring.
EXTRAS_PATH = RESULTS_DIR / "v13_reaudit/scanb_km_extras.csv"
SURV_REF_PATH = RESULTS_DIR / "v8_multicohort/survival_SCANB.csv"  # script 21
TEN_YEARS = 10 * 365.25  # os_time is in days
N_QUINTILES = 5
EXPECTED_COHORT = (7397, 1188)  # survival cohort of Methods
EXPECTED_PAM50 = (7215, 1164)   # PAM50-adjusted Cox cohort of script 21
HR_RTOL = 1e-6
SCORE_VS_LOGRANK_RTOL = 1e-3    # Breslow score variance vs hypergeometric log-rank variance

# Third output, see the module docstring.
AT_RISK_PATH = RESULTS_DIR / "v13_reaudit/scanb_km_at_risk.csv"
AT_RISK_STEP_YEARS = 2


def main():
    pred = pd.read_csv(PRED_PATH, index_col=0, usecols=lambda c: c in (["Unnamed: 0"] + KM_DRUGS) or c == "")
    clin = scanb_patient_cohort()
    rows = []
    for drug in KM_DRUGS:
        m = clin.join(pred[drug].rename("auc"), how="inner").dropna(subset=["os_time", "os_event", "auc"])
        med = m["auc"].median()
        s, r = m[m["auc"] < med], m[m["auc"] >= med]
        lr = logrank_test(s["os_time"], r["os_time"], event_observed_A=s["os_event"],
                          event_observed_B=r["os_event"])
        t = m["os_time"].median()
        surv = {}
        for label, part in (("s", s), ("r", r)):
            kmf = KaplanMeierFitter().fit(part["os_time"], part["os_event"])
            surv[label] = float(kmf.predict(t))
            surv[label + "_med"] = kmf.median_survival_time_ / 365.25
        rows.append({"drug": drug, "n_patients": len(m), "n_events": int(m["os_event"].sum()),
                     "n_sensitive": len(s), "n_resistant": len(r),
                     "chi2": float(lr.test_statistic), "p": float(lr.p_value),
                     "median_os_sensitive_years": surv["s_med"],
                     "median_os_resistant_years": surv["r_med"],
                     "sensitive_worse": surv["s"] < surv["r"]})
        logger.info("%s: n = %d, chi2 = %.2f, p = %.2e", drug, len(m), lr.test_statistic, lr.p_value)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    logger.info("Wrote %s", OUT_PATH)

    first = pd.DataFrame(rows).set_index("drug")
    km_extras(clin, pred, first)
    at_risk_table(clin, pred, first)


def cox_score_test(time, event, x, strata=None) -> tuple[float, float, float]:
    """Score test of beta = 0 in a Cox model with one numeric covariate x.

    Tied event times use the Breslow form: at each distinct event time with d
    events, the score gains the sum of x over the events minus d times the mean
    of x in the risk set, and the variance gains d times the variance of x in
    the risk set. With strata, each stratum keeps its own risk sets and the
    score and variance are summed over strata. With a 0/1 indicator this is the
    two-group log-rank test (the variances differ only by the hypergeometric
    factor at tied times), with an ordered group index it is the log-rank test
    for trend. Returns chi2 (1 df), p and the score U.
    """
    df = pd.DataFrame({"t": np.asarray(time, float), "e": np.asarray(event, float),
                       "x": np.asarray(x, float)})
    df["s"] = 0 if strata is None else np.asarray(strata)
    df["ex"] = df["e"] * df["x"]
    df["x2"] = df["x"] ** 2
    score, var = 0.0, 0.0
    for _, g in df.groupby("s"):
        # One row per distinct time, from the latest to the earliest, so that
        # cumulative sums give the risk set (time >= t) of each time.
        by_t = g.groupby("t").agg(n=("x", "size"), sx=("x", "sum"), sx2=("x2", "sum"),
                                  d=("e", "sum"), dx=("ex", "sum")).sort_index(ascending=False)
        r_n, r_x, r_x2 = by_t["n"].cumsum(), by_t["sx"].cumsum(), by_t["sx2"].cumsum()
        ev = by_t["d"] > 0
        mean = r_x[ev] / r_n[ev]
        score += float((by_t["dx"][ev] - by_t["d"][ev] * mean).sum())
        var += float((by_t["d"][ev] * (r_x2[ev] / r_n[ev] - mean ** 2)).sum())
    stat = score ** 2 / var
    return stat, float(chi2_dist.sf(stat, 1)), score


def km_extras(clin: pd.DataFrame, pred: pd.DataFrame, first: pd.DataFrame):
    """Write the second output (see the module docstring).

    first is the first output indexed by drug, used to check the score test
    against the log-rank chi2 of the same median split.
    """
    ref = pd.read_csv(SURV_REF_PATH).set_index("drug")
    rows = []
    for drug in KM_DRUGS:
        m = clin.join(pred[drug].rename("auc"), how="inner").dropna(subset=["os_time", "os_event", "auc"])
        n_ev = int(m["os_event"].sum())
        if (len(m), n_ev) != EXPECTED_COHORT:
            raise RuntimeError(f"{drug}: cohort {len(m)} / {n_ev}, expected {EXPECTED_COHORT}")
        med = m["auc"].median()
        resistant = (m["auc"] >= med).astype(int)  # same halves as the first output
        s, r = m[resistant == 0], m[resistant == 1]

        # Kaplan-Meier overall survival at 10 years in each half
        os10 = {}
        for label, part in (("sensitive", s), ("resistant", r)):
            kmf = KaplanMeierFitter().fit(part["os_time"], part["os_event"])
            os10[label] = 100 * float(kmf.predict(TEN_YEARS))
            os10[label + "_at_risk"] = int((part["os_time"] >= TEN_YEARS).sum())

        # Score test of the median split, must match the log-rank of the first output
        split_chi2, _, _ = cox_score_test(m["os_time"], m["os_event"], resistant)
        lr_chi2 = float(first.loc[drug, "chi2"])
        if not np.isclose(split_chi2, lr_chi2, rtol=SCORE_VS_LOGRANK_RTOL, atol=0):
            raise RuntimeError(f"{drug}: score chi2 {split_chi2:.4f} vs log-rank {lr_chi2:.4f}")

        # Log-rank test for trend across predicted-AUC quintiles (index 1 to 5)
        quint = pd.qcut(m["auc"], N_QUINTILES, labels=False) + 1
        tr_chi2, tr_p, tr_u = cox_score_test(m["os_time"], m["os_event"], quint)

        # Patients with a PAM50 call: reproduce script 21, then the stratified split
        has_pam = m["pam50"].notna()
        mp, res_p = m[has_pam], resistant[has_pam]
        if (len(mp), int(mp["os_event"].sum())) != EXPECTED_PAM50:
            raise RuntimeError(f"{drug}: PAM50 cohort {len(mp)} / {int(mp['os_event'].sum())}, "
                               f"expected {EXPECTED_PAM50}")
        cox_data = pd.concat([mp[["os_time", "os_event", "auc"]].rename(columns={"auc": "predicted_auc"}),
                              pd.get_dummies(mp["pam50"], prefix="sub", drop_first=True)], axis=1)
        hr = float(CoxPHFitter(penalizer=0.1).fit(cox_data, duration_col="os_time", event_col="os_event")
                   .hazard_ratios_["predicted_auc"])
        hr_ref = float(ref.loc[drug, "HR"])
        if not np.isclose(hr, hr_ref, rtol=HR_RTOL, atol=0):
            raise RuntimeError(f"{drug}: Cox HR {hr:.6f} does not reproduce {hr_ref:.6f} of {SURV_REF_PATH}")
        st_chi2, st_p, _ = cox_score_test(mp["os_time"], mp["os_event"], res_p, strata=mp["pam50"])

        comp = {}
        for label, flag in (("sensitive", 0), ("resistant", 1)):
            pam = mp.loc[res_p == flag, "pam50"]
            comp[label] = (100 * float((pam == "LumA").mean()), 100 * float((pam == "Basal").mean()),
                           len(pam))

        rows.append({
            "drug": drug, "n_patients": len(m), "n_events": n_ev,
            "os10_sensitive_pct": os10["sensitive"], "os10_resistant_pct": os10["resistant"],
            "os10_diff_pp": os10["sensitive"] - os10["resistant"],
            "n_at_risk_10y_sensitive": os10["sensitive_at_risk"],
            "n_at_risk_10y_resistant": os10["resistant_at_risk"],
            "trend_chi2": tr_chi2, "trend_p": tr_p, "trend_score_u": tr_u,
            "median_split_score_chi2": split_chi2,
            "pam50_n_patients": len(mp), "pam50_n_events": int(mp["os_event"].sum()),
            "pam50_strat_chi2": st_chi2, "pam50_strat_p": st_p,
            "pam50_n_sensitive": comp["sensitive"][2], "pam50_n_resistant": comp["resistant"][2],
            "pct_luma_sensitive": comp["sensitive"][0], "pct_basal_sensitive": comp["sensitive"][1],
            "pct_luma_resistant": comp["resistant"][0], "pct_basal_resistant": comp["resistant"][1],
        })
        logger.info("%s: HR %.4f reproduced, OS10 %.1f%% vs %.1f%% (%+.1f pp), trend chi2 = %.2f "
                    "p = %.2e, PAM50-stratified split chi2 = %.2f p = %.2e (n = %d), "
                    "LumA %.1f%% vs %.1f%%, Basal %.1f%% vs %.1f%%",
                    drug, hr, os10["sensitive"], os10["resistant"],
                    os10["sensitive"] - os10["resistant"], tr_chi2, tr_p, st_chi2, st_p, len(mp),
                    comp["sensitive"][0], comp["resistant"][0], comp["sensitive"][1], comp["resistant"][1])
    os.makedirs(EXTRAS_PATH.parent, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXTRAS_PATH, index=False)
    logger.info("Wrote %s", EXTRAS_PATH)


def at_risk_table(clin: pd.DataFrame, pred: pd.DataFrame, first: pd.DataFrame):
    """Write the third output (see the module docstring).

    first is the first output indexed by drug, used to check the counts at 0
    years. The counts at 10 years are checked against the second output as
    written by km_extras.
    """
    extras = pd.read_csv(EXTRAS_PATH).set_index("drug")
    rows = []
    for drug in KM_DRUGS:
        m = clin.join(pred[drug].rename("auc"), how="inner").dropna(subset=["os_time", "os_event", "auc"])
        med = m["auc"].median()
        quint = pd.qcut(m["auc"], N_QUINTILES, labels=False)  # 0 = lowest predicted AUC (Q1)
        groups = [("median", "sensitive", m[m["auc"] < med]), ("median", "resistant", m[m["auc"] >= med])]
        groups += [("quintile", f"Q{k + 1}", m[quint == k]) for k in range(N_QUINTILES)]
        last_year = int(m["os_time"].max() / 365.25)
        times = [t for t in range(0, last_year + 1, AT_RISK_STEP_YEARS)
                 if (m["os_time"] >= t * 365.25).any()]
        rows += [{"drug": drug, "split": split, "group": group, "time_years": t,
                  "n_at_risk": int((part["os_time"] >= t * 365.25).sum())}
                 for split, group, part in groups for t in times]
    table = pd.DataFrame(rows)

    n_at = table.set_index(["drug", "group", "time_years"])["n_at_risk"]
    for drug in KM_DRUGS:
        start = (n_at[drug, "sensitive", 0], n_at[drug, "resistant", 0])
        if start != (first.loc[drug, "n_sensitive"], first.loc[drug, "n_resistant"]):
            raise RuntimeError(f"{drug}: at risk at 0 years {start}, first output "
                               f"{first.loc[drug, ['n_sensitive', 'n_resistant']].tolist()}")
        if sum(start) != EXPECTED_COHORT[0]:
            raise RuntimeError(f"{drug}: the halves at 0 years add up to {sum(start)}, not {EXPECTED_COHORT[0]}")
        ten = (n_at[drug, "sensitive", 10], n_at[drug, "resistant", 10])
        if ten != (extras.loc[drug, "n_at_risk_10y_sensitive"], extras.loc[drug, "n_at_risk_10y_resistant"]):
            raise RuntimeError(f"{drug}: at risk at 10 years {ten} differs from {EXTRAS_PATH}")
        q0 = [n_at[drug, f"Q{k + 1}", 0] for k in range(N_QUINTILES)]
        if max(q0) - min(q0) > 1:
            raise RuntimeError(f"{drug}: quintile sizes at 0 years {q0} differ by more than 1")
    totals = table.groupby(["drug", "split", "time_years"])["n_at_risk"].sum().unstack("time_years")
    if totals.nunique().max() != 1:
        raise RuntimeError(f"The total at risk per time differs between drugs or splits:\n{totals}")
    growing = table.groupby(["drug", "split", "group"])["n_at_risk"].apply(
        lambda n: not n.is_monotonic_decreasing)
    if growing.any():
        raise RuntimeError(f"Numbers at risk grow with time in {growing[growing].index.tolist()}")

    os.makedirs(AT_RISK_PATH.parent, exist_ok=True)
    table.to_csv(AT_RISK_PATH, index=False)
    logger.info("Numbers at risk per time, whole cohort: %s", totals.iloc[0].to_dict())
    logger.info("Wrote %s (%d rows)", AT_RISK_PATH, len(table))


if __name__ == "__main__":
    main()
