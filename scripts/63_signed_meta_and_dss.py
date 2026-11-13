#!/usr/bin/env python3
"""Direction-aware meta-analysis, directional agreement, positive-control base rate
and disease-specific survival in METABRIC (re-audit of the multi-cohort claims).

1. Signed meta-analysis. Fisher's method (script 21) combines two-sided p-values
   and ignores the direction of the hazard ratio, so a drug with HR > 1 in one
   cohort and HR < 1 in another can still reach significance. Here every cohort
   contributes a signed z = sign(log HR) * Phi^-1(1 - p/2). The z are combined
   with a Stouffer test whose weights are the square root of the events of the
   PAM50-adjusted Cox of each cohort, two-sided p, Benjamini-Hochberg over drugs.
2. Directional agreement of the SCAN-B and METABRIC hazard ratios over all
   drugs, over the SCAN-B FDR-significant drugs, over the drugs significant in
   both cohorts and over the Fisher-significant drugs. The agreement expected if
   the two cohorts chose their directions independently, given the share of
   HR > 1 in each, is reported next to the observed one.
3. Base rate of the positive controls. Fraction of all drugs that reach
   significance, binomial probability that at least as many of the five controls
   reach it at that rate, and the rank of each control among all drugs.
4. METABRIC disease-specific survival. The per-drug predictions of script 21 are
   rebuilt, its overall-survival hazard ratios reproduced for every drug, and the
   same model (penalizer 0.1, PAM50 dummies) is refitted with death from disease
   (VITAL_STATUS == "Died of Disease") as the event. Deaths from other causes are
   censored at their time (cause-specific hazard). The single death without a
   recorded cause is censored as well, so the patients are the same as for OS.

Reproduction checks (the script raises if any fails):
  - SCAN-B PAM50-adjusted Cox cohort: 7,215 patients and 1,164 events, rebuilt
    from src/data/scanb_cohort.py and the prediction index, equal to
    cohort_summary.csv
  - Fisher meta-analysis recomputed from the three survival tables: adjusted p
    of epirubicin and of every drug with Fisher p >= 1e-6 equal to
    meta_analysis_fisher.csv, 551 drugs at FDR < 0.05
  - 623 SCAN-B and 74 METABRIC drugs at FDR < 0.05, 54 significant in both
    cohorts of which 53 point in the same direction, 75 Fisher-significant drugs
    with opposite directions in SCAN-B and METABRIC
  - METABRIC overall survival: 1,979 patients and 1,143 events, hazard ratios of
    survival_METABRIC.csv reproduced for every drug (max |delta log HR| < 1e-6)

Inputs:
  results/v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv   (script 21)
  results/v8_multicohort/meta_analysis_fisher.csv             (script 21)
  results/v8_multicohort/cohort_summary.csv                   (script 21)
  results/v5_final/scanb_perdrug_predictions.csv              (script 17, index only)
  data/precision_processed/scanb_clinical.csv, metabric_clinical.csv,
  tf_activities_metabric.csv, tf_activities_all_prism.csv and the PRISM response
Outputs in results/v13_reaudit/:
  signed_meta_stouffer.csv        one row per drug, per-cohort HR, p and signed z,
                                  Stouffer z, p and BH-adjusted p, Fisher p and padj
  direction_agreement_summary.csv one row per drug subset, SCAN-B vs METABRIC
  positive_control_baserate.csv   one row per analysis and positive control
  metabric_dss_survival.csv       one row per drug, OS and DSS hazard ratios
  meta_dss_summary.csv            metric, value

Run from project root:
    python scripts/63_signed_meta_and_dss.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.scanb_cohort import scanb_patient_cohort
from src.data.splits import cell_line_holdout_split, set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signed_meta_dss")

DATA = DATA_DIR
PROCESSED = DATA / "precision_processed"
MULTI = RESULTS_DIR / "v8_multicohort"
PRED_PATH = RESULTS_DIR / "v5_final/scanb_perdrug_predictions.csv"
OUT = RESULTS_DIR / "v13_reaudit"

SEED = 42
PENALIZER = 0.1
FDR = 0.05
COHORTS = ["SCANB", "METABRIC", "TCGA"]   # order of script 21, kept for the Fisher sum
CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]

# Counts published by the pipeline (Results, Table 2 and Table 3). The script
# recomputes them from the result tables and stops if any differs.
PUBLISHED = {
    "fisher_n_sig": 551,
    "scanb_n_sig": 623,
    "metabric_n_sig": 74,
    "both_n_sig": 54,
    "both_n_same_direction": 53,
    "fisher_sig_opposite_direction": 75,
    "scanb_cox_n_patients": 7215,
    "scanb_cox_n_events": 1164,
    "metabric_cox_n_patients": 1979,
    "metabric_cox_n_events": 1143,
}
LOGHR_TOL = 1e-6
PADJ_RTOL = 1e-9
FISHER_P_FLOOR = 1e-6


def check(condition, message):
    """Raise with a clear message when a reproduction check fails."""
    if not condition:
        raise ValueError(f"Reproduction check failed: {message}")
    logger.info("Reproduced: %s", message)


def role(drug):
    if drug in CANDIDATES:
        return "candidate"
    if drug in CONTROLS:
        return "positive_control"
    return ""


# ---------------------------------------------------------------------------
# Reproduction of the published cohort sizes and of the Fisher meta-analysis
# ---------------------------------------------------------------------------

def reproduce_scanb_cohort(cohort_summary):
    """PAM50-adjusted SCAN-B Cox cohort from the shared cohort helper."""
    pred_index = pd.read_csv(PRED_PATH, usecols=[0], index_col=0).index
    clin = scanb_patient_cohort(DATA)
    cox = clin.loc[clin.index.intersection(pred_index)].dropna(subset=["pam50"])
    n, e = int(len(cox)), int(cox["os_event"].sum())
    ref = cohort_summary.loc["SCANB"]
    check(n == int(ref["n_patients_cox"]) == PUBLISHED["scanb_cox_n_patients"]
          and e == int(ref["n_events_cox"]) == PUBLISHED["scanb_cox_n_events"],
          f"SCAN-B PAM50-adjusted Cox cohort {n} patients, {e} events")
    return n, e


def load_survival():
    """Per-drug HR and p of the three cohorts, one table per cohort indexed by drug."""
    surv = {c: pd.read_csv(MULTI / f"survival_{c}.csv").set_index("drug") for c in COHORTS}
    for c, s in surv.items():
        if s[["HR", "pvalue"]].isna().any().any() or (s["HR"] <= 0).any():
            raise ValueError(f"survival_{c}.csv has missing or non-positive values")
    return surv


def reproduce_fisher(surv):
    """Fisher's method as meta_analysis() of script 21.

    The p-value comes from the chi-square survival function. Script 21 used
    1 - cdf until September 2026, which loses precision below about 1e-12, so
    the per-drug comparison is limited to Fisher p >= FISHER_P_FLOOR, where both
    formulas agree to better than PADJ_RTOL whichever version wrote the file.
    """
    common = sorted(set.intersection(*[set(s.index) for s in surv.values()]))
    stat = np.array([-2 * sum(np.log(surv[c].loc[d, "pvalue"]) for c in COHORTS) for d in common])
    fisher = pd.DataFrame({"fisher_p": stats.chi2.sf(stat, df=2 * len(COHORTS))}, index=common)
    fisher["fisher_padj"] = stats.false_discovery_control(fisher["fisher_p"])

    published = pd.read_csv(MULTI / "meta_analysis_fisher.csv").set_index("drug")
    check(set(published.index) == set(fisher.index), f"Fisher drug set ({len(fisher)} drugs)")
    epi_new, epi_pub = fisher.loc["epirubicin", "fisher_padj"], published.loc["epirubicin", "fisher_padj"]
    check(np.isclose(epi_new, epi_pub, rtol=PADJ_RTOL, atol=0),
          f"Fisher adjusted p of epirubicin {epi_new:.6e} (published {epi_pub:.6e})")
    comparable = fisher.index[fisher["fisher_p"] >= FISHER_P_FLOOR]
    check(np.allclose(fisher.loc[comparable, "fisher_padj"], published.loc[comparable, "fisher_padj"],
                      rtol=PADJ_RTOL, atol=0),
          f"Fisher adjusted p of the {len(comparable)} drugs with Fisher p >= {FISHER_P_FLOOR:g}")
    n_sig = int((fisher["fisher_padj"] < FDR).sum())
    check(n_sig == PUBLISHED["fisher_n_sig"], f"{n_sig} Fisher-significant drugs")
    return fisher, float(epi_new), float(epi_pub)


# ---------------------------------------------------------------------------
# 1. Signed meta-analysis
# ---------------------------------------------------------------------------

def signed_z(hr, p):
    """Two-sided p-value turned into a z with the sign of log HR."""
    return np.sign(np.log(hr)) * stats.norm.isf(p / 2)


def stouffer_meta(surv, fisher, events):
    """Weighted Stouffer combination of the signed z of the three cohorts."""
    drugs = fisher.index
    weights = {c: np.sqrt(events[c]) for c in COHORTS}
    table = pd.DataFrame(index=drugs)
    num = np.zeros(len(drugs))
    for c in COHORTS:
        s = surv[c].loc[drugs]
        table[f"HR_{c}"] = s["HR"].values
        table[f"p_{c}"] = s["pvalue"].values
        table[f"padj_{c}"] = s["padj"].values
        table[f"z_{c}"] = signed_z(s["HR"].values, s["pvalue"].values)
        num += weights[c] * table[f"z_{c}"].values
    table["stouffer_z"] = num / np.sqrt(sum(w ** 2 for w in weights.values()))
    table["stouffer_p"] = 2 * stats.norm.sf(np.abs(table["stouffer_z"]))
    table["stouffer_padj"] = stats.false_discovery_control(table["stouffer_p"])
    table["fisher_p"] = fisher["fisher_p"]
    table["fisher_padj"] = fisher["fisher_padj"]
    table["stouffer_sig"] = table["stouffer_padj"] < FDR
    table["fisher_sig"] = table["fisher_padj"] < FDR
    table["scanb_metabric_same_direction"] = (np.sign(np.log(table["HR_SCANB"]))
                                              == np.sign(np.log(table["HR_METABRIC"])))
    table["role"] = [role(d) for d in table.index]
    table.index.name = "drug"
    return table.sort_values("stouffer_p"), weights


# ---------------------------------------------------------------------------
# 2. Directional agreement between SCAN-B and METABRIC
# ---------------------------------------------------------------------------

def agreement(label, hr_s, hr_m):
    """Observed and independence-expected agreement of the HR directions."""
    up_s, up_m = np.log(hr_s) > 0, np.log(hr_m) > 0
    same = int((np.sign(np.log(hr_s)) == np.sign(np.log(hr_m))).sum())
    n = int(len(hr_s))
    ps, pm = float(up_s.mean()), float(up_m.mean())
    # 2 x 2 table of the directions, rows SCAN-B HR > 1 or < 1, columns METABRIC
    two_by_two = [[int((up_s & up_m).sum()), int((up_s & ~up_m).sum())],
                  [int((~up_s & up_m).sum()), int((~up_s & ~up_m).sum())]]
    odds, p_indep = stats.fisher_exact(two_by_two)
    return {
        "subset": label,
        "n_drugs": n,
        "n_same_direction": same,
        "n_opposite_direction": n - same,
        "pct_same_direction": round(100 * same / n, 1),
        "pct_hr_gt1_scanb": round(100 * ps, 1),
        "pct_hr_gt1_metabric": round(100 * pm, 1),
        "pct_same_expected_independent": round(100 * (ps * pm + (1 - ps) * (1 - pm)), 1),
        "binom_p_vs_half": float(stats.binomtest(same, n, 0.5).pvalue),
        "fisher_exact_odds_ratio": float(odds),
        "fisher_exact_p_independence": float(p_indep),
    }


def direction_agreement(table):
    s_sig = table["padj_SCANB"] < FDR
    m_sig = table["padj_METABRIC"] < FDR
    check(int(s_sig.sum()) == PUBLISHED["scanb_n_sig"], f"{int(s_sig.sum())} SCAN-B FDR-significant drugs")
    check(int(m_sig.sum()) == PUBLISHED["metabric_n_sig"], f"{int(m_sig.sum())} METABRIC FDR-significant drugs")
    subsets = [
        ("all drugs tested in both cohorts", pd.Series(True, index=table.index)),
        ("SCAN-B FDR < 0.05", s_sig),
        ("METABRIC FDR < 0.05", m_sig),
        ("FDR < 0.05 in both cohorts", s_sig & m_sig),
        ("Fisher FDR < 0.05", table["fisher_sig"]),
        ("Stouffer FDR < 0.05 (direction built in)", table["stouffer_sig"]),
    ]
    rows = [agreement(label, table.loc[mask, "HR_SCANB"].values, table.loc[mask, "HR_METABRIC"].values)
            for label, mask in subsets]
    out = pd.DataFrame(rows)
    both = out.set_index("subset").loc["FDR < 0.05 in both cohorts"]
    n_both, n_same = int(both["n_drugs"]), int(both["n_same_direction"])
    check(n_both == PUBLISHED["both_n_sig"] and n_same == PUBLISHED["both_n_same_direction"],
          f"{n_both} drugs significant in both cohorts, {n_same} in the same direction")
    n_opp = int(out.set_index("subset").loc["Fisher FDR < 0.05", "n_opposite_direction"])
    check(n_opp == PUBLISHED["fisher_sig_opposite_direction"],
          f"{n_opp} Fisher-significant drugs with opposite SCAN-B and METABRIC directions")
    return out


# ---------------------------------------------------------------------------
# 3. Positive-control base rate
# ---------------------------------------------------------------------------

def control_baserate(table):
    """Rank and binomial base rate of the five controls in three analyses."""
    analyses = [
        ("Fisher meta-analysis", "fisher_p", "fisher_padj", None),
        ("SCAN-B alone", "p_SCANB", "padj_SCANB", "HR_SCANB"),
        ("Signed Stouffer meta-analysis", "stouffer_p", "stouffer_padj", "stouffer_z"),
    ]
    rows = []
    for name, pcol, padjcol, dircol in analyses:
        n = len(table)
        frac = float((table[padjcol] < FDR).mean())
        rank = table[pcol].rank(method="min")
        k = int((table.loc[CONTROLS, padjcol] < FDR).sum())
        prob = float(stats.binom.sf(k - 1, len(CONTROLS), frac))
        for d in CONTROLS:
            if dircol is None:
                direction = ""
            elif dircol.startswith("HR"):
                direction = "HR>1" if table.loc[d, dircol] > 1 else "HR<1"
            else:
                direction = "HR>1" if table.loc[d, dircol] > 0 else "HR<1"
            rows.append({
                "analysis": name, "drug": d,
                "p": float(table.loc[d, pcol]), "padj": float(table.loc[d, padjcol]),
                "significant": bool(table.loc[d, padjcol] < FDR),
                "pooled_direction": direction,
                "rank_by_p": int(rank.loc[d]), "n_drugs": n,
                "rank_percentile": round(100 * rank.loc[d] / n, 1),
                "frac_drugs_significant": frac,
                "n_controls_significant": k,
                "prob_at_least_n_controls_significant": prob,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. METABRIC disease-specific survival
# ---------------------------------------------------------------------------

def metabric_predictions():
    """Per-drug Ridge predictions for METABRIC, step by step as run_cohort_validation of script 21."""
    set_all_seeds(SEED)
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(PROCESSED / "tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)
    cohort_tf = pd.read_csv(PROCESSED / "tf_activities_metabric.csv", index_col=0)

    shared = sorted(set(tf_prism.columns) & set(cohort_tf.columns))
    train_prism = tf_prism.loc[[c for c in train_cells if c in tf_prism.index]]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism[shared])
    prism_qt = pd.DataFrame(qt.transform(tf_prism[shared]), index=tf_prism.index, columns=shared)
    cohort_qt = pd.DataFrame(qt.transform(cohort_tf[shared]), index=cohort_tf.index, columns=shared)

    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_cohort = scaler.transform(cohort_qt)

    preds = {}
    for drug in drug_response.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[mask], y[mask])
        preds[drug] = ridge.predict(X_cohort)
    pred = pd.DataFrame(preds, index=cohort_qt.index)
    logger.info("METABRIC predictions: %d patients x %d drugs (%d shared TFs)", *pred.shape, len(shared))
    return pred


def metabric_design(pred):
    """Survival table of the PAM50-adjusted METABRIC Cox of script 21 plus the cause of death."""
    clin = pd.read_csv(PROCESSED / "metabric_clinical.csv", index_col=0).set_index("patientId")
    clin["os_time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce") * 30.44
    clin["os_event"] = clin["OS_STATUS"].str.extract(r"^(\d)")[0].astype(float)

    merged = clin[["os_time", "os_event"]].join(pred, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event"])
    merged = merged[merged["os_time"] > 0]
    merged = merged.join(clin[["PAM50"]]).dropna(subset=["PAM50"])
    pam = pd.get_dummies(merged["PAM50"], prefix="sub", drop_first=True)
    merged = pd.concat([merged, pam], axis=1)
    pam_cols = list(pam.columns)

    vital = clin.loc[merged.index, "VITAL_STATUS"]
    merged["dss_event"] = (vital == "Died of Disease").astype(float)
    cause = pd.Series("alive", index=merged.index)
    cause[merged["os_event"] == 1] = "unknown cause"
    cause[vital == "Died of Disease"] = "disease"
    cause[vital == "Died of Other Causes"] = "other causes"
    if ((merged["dss_event"] == 1) & (merged["os_event"] != 1)).any():
        raise ValueError("A disease-specific death is not an overall-survival event")
    if ((merged["os_event"] == 0) & vital.isin(["Died of Disease", "Died of Other Causes"])).any():
        raise ValueError("A censored patient has a recorded cause of death")
    check(len(merged) == PUBLISHED["metabric_cox_n_patients"]
          and int(merged["os_event"].sum()) == PUBLISHED["metabric_cox_n_events"],
          f"METABRIC PAM50-adjusted Cox cohort {len(merged)} patients, {int(merged['os_event'].sum())} events")
    return merged, pam_cols, cause


def fit_per_drug(merged, pam_cols, drugs, event_col):
    """Per-drug penalized Cox with PAM50 dummies, the loop of script 21."""
    rows = []
    for drug in drugs:
        cox_data = merged[["os_time", event_col, drug] + pam_cols].dropna()
        cox_data = cox_data.rename(columns={drug: "predicted_auc"})
        if len(cox_data) < 50 or cox_data[event_col].sum() < 10:
            continue
        try:
            cph = CoxPHFitter(penalizer=PENALIZER)
            cph.fit(cox_data, duration_col="os_time", event_col=event_col)
        except Exception as exc:  # same skip rule as script 21, but logged
            logger.warning("  %s (%s): Cox failed, %s", drug, event_col, exc)
            continue
        rows.append({"drug": drug, "HR": float(cph.hazard_ratios_["predicted_auc"]),
                     "pvalue": float(cph.summary.loc["predicted_auc", "p"])})
    out = pd.DataFrame(rows).set_index("drug")
    out["padj"] = stats.false_discovery_control(out["pvalue"])
    return out


def metabric_dss():
    pred = metabric_predictions()
    merged, pam_cols, cause = metabric_design(pred)
    drugs = list(pred.columns)

    logger.info("Refitting METABRIC overall survival for %d drugs", len(drugs))
    os_fit = fit_per_drug(merged, pam_cols, drugs, "os_event")
    published = pd.read_csv(MULTI / "survival_METABRIC.csv").set_index("drug")
    check(set(os_fit.index) == set(published.index), f"METABRIC OS drug set ({len(os_fit)} drugs)")
    dlog = float(np.max(np.abs(np.log(os_fit.loc[published.index, "HR"]) - np.log(published["HR"]))))
    check(dlog < LOGHR_TOL, f"METABRIC OS hazard ratios of every drug (max |delta log HR| = {dlog:.2e})")
    n_os_sig = int((os_fit["padj"] < FDR).sum())
    check(n_os_sig == PUBLISHED["metabric_n_sig"], f"{n_os_sig} METABRIC OS drugs at FDR < 0.05")

    logger.info("Fitting METABRIC disease-specific survival for %d drugs", len(drugs))
    dss_fit = fit_per_drug(merged, pam_cols, drugs, "dss_event")

    table = os_fit.add_suffix("_OS").join(dss_fit.add_suffix("_DSS"), how="outer")
    table["sig_OS"] = table["padj_OS"] < FDR
    table["sig_DSS"] = table["padj_DSS"] < FDR
    table["same_direction"] = np.sign(np.log(table["HR_OS"])) == np.sign(np.log(table["HR_DSS"]))
    table["role"] = [role(d) for d in table.index]
    table.index.name = "drug"
    return table.sort_values("pvalue_DSS"), cause, dlog


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []

    def put(metric, value):
        summary.append({"metric": metric, "value": value})

    cohort_summary = pd.read_csv(MULTI / "cohort_summary.csv").set_index("cohort")
    n_scanb, e_scanb = reproduce_scanb_cohort(cohort_summary)
    put("repro_scanb_cox_n_patients", n_scanb)
    put("repro_scanb_cox_n_events", e_scanb)

    surv = load_survival()
    fisher, epi_new, epi_pub = reproduce_fisher(surv)
    put("repro_fisher_epirubicin_padj_published", epi_pub)
    put("repro_fisher_epirubicin_padj_recomputed", epi_new)

    # 1. Signed meta-analysis
    events = {c: int(cohort_summary.loc[c, "n_events_cox"]) for c in COHORTS}
    table, weights = stouffer_meta(surv, fisher, events)
    table.to_csv(OUT / "signed_meta_stouffer.csv")
    for c in COHORTS:
        put(f"stouffer_events_{c}", events[c])
        put(f"stouffer_weight_{c}", round(weights[c] / np.sqrt(sum(w ** 2 for w in weights.values())), 4))
    st, fi = table["stouffer_sig"], table["fisher_sig"]
    opposite = ~table["scanb_metabric_same_direction"]
    put("n_drugs_meta", len(table))
    put("stouffer_n_sig", int(st.sum()))
    put("stouffer_pct_sig", round(100 * st.mean(), 1))
    put("stouffer_n_sig_hr_gt1", int((st & (table["stouffer_z"] > 0)).sum()))
    put("stouffer_n_sig_hr_lt1", int((st & (table["stouffer_z"] < 0)).sum()))
    put("fisher_n_sig", int(fi.sum()))
    put("stouffer_fisher_overlap", int((st & fi).sum()))
    put("fisher_sig_lost_under_stouffer", int((fi & ~st).sum()))
    put("stouffer_sig_not_fisher", int((st & ~fi).sum()))
    put("fisher_sig_opposite_direction", int((fi & opposite).sum()))
    put("fisher_sig_opposite_direction_lost_under_stouffer", int((fi & opposite & ~st).sum()))
    for d in CANDIDATES + CONTROLS:
        put(f"stouffer_{d}_z", round(float(table.loc[d, "stouffer_z"]), 3))
        put(f"stouffer_{d}_p", float(table.loc[d, "stouffer_p"]))
        put(f"stouffer_{d}_padj", float(table.loc[d, "stouffer_padj"]))
    logger.info("Stouffer: %d of %d drugs at FDR < 0.05 (Fisher %d, overlap %d, Fisher-only %d)",
                st.sum(), len(table), fi.sum(), (st & fi).sum(), (fi & ~st).sum())

    # 2. Directional agreement
    agree = direction_agreement(table)
    agree.to_csv(OUT / "direction_agreement_summary.csv", index=False)
    keys = ["all", "scanb_sig", "metabric_sig", "both_sig", "fisher_sig", "stouffer_sig"]
    for key, (_, r) in zip(keys, agree.iterrows()):
        put(f"direction_{key}_n", int(r["n_drugs"]))
        put(f"direction_{key}_n_same", int(r["n_same_direction"]))
        put(f"direction_{key}_pct_same", r["pct_same_direction"])
        put(f"direction_{key}_pct_same_expected_independent", r["pct_same_expected_independent"])
        put(f"direction_{key}_fisher_exact_p_independence", r["fisher_exact_p_independence"])
    logger.info("Direction agreement:\n%s", agree.to_string(index=False))

    # 3. Positive-control base rate
    base = control_baserate(table)
    base.to_csv(OUT / "positive_control_baserate.csv", index=False)
    for tag, name in (("fisher", "Fisher meta-analysis"), ("scanb", "SCAN-B alone"),
                      ("stouffer", "Signed Stouffer meta-analysis")):
        b = base[base["analysis"] == name].set_index("drug")
        frac = float(b["frac_drugs_significant"].iloc[0])
        put(f"baserate_{tag}_frac_drugs_sig", round(frac, 4))
        put(f"baserate_{tag}_n_controls_sig", int(b["n_controls_significant"].iloc[0]))
        put(f"baserate_{tag}_prob_at_least_observed", float(b["prob_at_least_n_controls_significant"].iloc[0]))
        put(f"baserate_{tag}_prob_at_least_4_of_5", float(stats.binom.sf(3, len(CONTROLS), frac)))
        put(f"baserate_{tag}_prob_5_of_5", float(frac ** len(CONTROLS)))
        for d in CONTROLS:
            put(f"baserate_{tag}_{d}_rank", int(b.loc[d, "rank_by_p"]))
    logger.info("Positive-control base rate:\n%s", base.to_string(index=False))

    # 4. METABRIC disease-specific survival
    dss, cause, dlog = metabric_dss()
    dss.to_csv(OUT / "metabric_dss_survival.csv")
    n_deaths = int((cause != "alive").sum())
    n_other = int((cause == "other causes").sum())
    put("metabric_n_patients", len(cause))
    put("metabric_n_deaths", n_deaths)
    put("metabric_n_dss_deaths", int((cause == "disease").sum()))
    put("metabric_n_other_cause_deaths", n_other)
    put("metabric_pct_deaths_other_causes", round(100 * n_other / n_deaths, 1))
    put("metabric_n_deaths_unknown_cause_censored", int((cause == "unknown cause").sum()))
    put("repro_metabric_os_max_abs_dloghr", dlog)
    so, sd = dss["sig_OS"], dss["sig_DSS"]
    put("metabric_os_n_sig", int(so.sum()))
    put("metabric_dss_n_sig", int(sd.sum()))
    put("metabric_os_dss_overlap", int((so & sd).sum()))
    put("metabric_os_only", int((so & ~sd).sum()))
    put("metabric_dss_only", int((sd & ~so).sum()))
    put("metabric_dss_n_sig_hr_gt1", int((sd & (dss["HR_DSS"] > 1)).sum()))
    put("metabric_dss_n_sig_hr_lt1", int((sd & (dss["HR_DSS"] < 1)).sum()))
    put("metabric_os_sig_same_direction_dss", int((so & dss["same_direction"]).sum()))
    put("metabric_os_dss_pct_same_direction", round(100 * dss["same_direction"].mean(), 1))
    put("metabric_os_dss_loghr_pearson",
        round(float(np.corrcoef(np.log(dss["HR_OS"]), np.log(dss["HR_DSS"]))[0, 1]), 3))
    for d in CANDIDATES + CONTROLS:
        for col in ("HR_OS", "pvalue_OS", "padj_OS", "HR_DSS", "pvalue_DSS", "padj_DSS"):
            put(f"metabric_{d}_{col}", float(dss.loc[d, col]))
    logger.info("METABRIC DSS: %d disease deaths, %d other-cause deaths, %d drugs FDR < 0.05 (OS %d, overlap %d)",
                (cause == "disease").sum(), n_other, sd.sum(), so.sum(), (so & sd).sum())
    logger.info("Candidates and controls:\n%s",
                dss.loc[CANDIDATES + CONTROLS, ["HR_OS", "padj_OS", "HR_DSS", "pvalue_DSS", "padj_DSS"]].to_string())

    out = pd.DataFrame(summary)
    out.to_csv(OUT / "meta_dss_summary.csv", index=False)
    logger.info("Wrote %s (%d metrics)", OUT / "meta_dss_summary.csv", len(out))


if __name__ == "__main__":
    main()
