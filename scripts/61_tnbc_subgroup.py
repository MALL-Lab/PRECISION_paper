#!/usr/bin/env python3
"""Survival signal of the candidates and positive controls inside TNBC and Basal tumors of SCAN-B.

The SCAN-B screen (scripts 21 and 31) adjusts for PAM50 but pools all subtypes,
so a hazard ratio can be carried by differences between subtypes rather than by
differences among triple-negative tumors, the population the paper targets. This
script asks how much of the signal of the seven Table 5 candidates and the five
positive controls survives inside two definitions of the target population, and
how strongly the predicted AUC of each drug separates the PAM50 subtypes.

Cohort: the PAM50-adjusted Cox cohort of scripts/21_multicohort_survival.py, that
is the survival cohort of src/data/scanb_cohort.py (7,397 patients) joined with
the per-drug predictions and restricted to a known PAM50 call (7,215 patients,
1,164 deaths). Before anything else the script reproduces these counts, the
canonical penalized PAM50-adjusted hazard ratios of the twelve drugs in
results/v8_multicohort/survival_SCANB.csv (erlotinib 4.28) and the Basal subset
of script 19b (686 patients, 167 deaths), and raises if any of them differs.

Analyses:
  1. Cox models per subset. Predicted AUC is standardized with the mean and the
     standard deviation (ddof 1) of the full 7,215-patient cohort, so hazard
     ratios are per standard deviation and comparable across subsets. In each
     subset an unpenalized univariate Cox model (lifelines, penalizer 0) is fitted
     per drug. Subsets are clinical TNBC (ER, PR and HER2 all known and negative,
     decoded with src/data/clinical_encoding.py) and PAM50 Basal. The same model
     stratified by PAM50 in the full cohort is reported for reference. Wald p
     values are adjusted with Benjamini-Hochberg within each family (the seven
     candidates, the five controls) and each subset.
  2. Predicted AUC by PAM50 subtype: mean per subtype, Basal minus LumA in units
     of the full-cohort standard deviation and a Kruskal-Wallis test across the
     five subtypes.
  3. PAM50 composition of each half of the cohort split at the median predicted
     AUC (sensitive half = AUC below the median, as in scripts 19b and 58).

Direction: low predicted AUC means predicted sensitive, so a hazard ratio above 1
per standard deviation means predicted-resistant patients die sooner, the
direction the candidates are selected for. The analysis has no random component.

Inputs:
  results/v5_final/scanb_perdrug_predictions.csv    (script 17)
  data/precision_processed/scanb_clinical.csv       (script 09)
  results/v8_multicohort/survival_SCANB.csv         (script 21, reproduction check)
  results/v7_weaknesses/subtype_pooling_summary.csv (script 19b, reproduction check)
Outputs (results/v13_reaudit/):
  tnbc_subgroup_cox.csv                 subset, drug, family, n, events, hr_per_sd,
                                        ci_low, ci_high, p, p_adj_family
  pam50_predicted_auc.csv               one row per drug with the subtype means,
                                        the Basal minus LumA shift and Kruskal-Wallis
  pam50_composition_sensitive_half.csv  one row per drug and half with the PAM50
                                        percentages
  tnbc_subgroup_summary.csv             metric, value

Run from project root:
    python scripts/61_tnbc_subgroup.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

from src.data.clinical_encoding import encode_binary_receptor
from src.data.drug_names import normalize_drug_columns
from src.data.scanb_cohort import scanb_patient_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tnbc_subgroup")

PRED_PATH = RESULTS_DIR / "v5_final/scanb_perdrug_predictions.csv"
SURV_SCANB = RESULTS_DIR / "v8_multicohort/survival_SCANB.csv"
POOLING_SUMMARY = RESULTS_DIR / "v7_weaknesses/subtype_pooling_summary.csv"
OUT_DIR = RESULTS_DIR / "v13_reaudit"

CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
FAMILY = {**{d: "candidate" for d in CANDIDATES}, **{d: "positive_control" for d in CONTROLS}}
DRUGS = CANDIDATES + CONTROLS
PAM50 = ["LumA", "LumB", "Her2", "Basal", "Normal"]

# Published values that the cohort and the canonical model must reproduce
EXPECTED_SURVIVAL_COHORT = 7397
EXPECTED_COX_N, EXPECTED_COX_EVENTS = 7215, 1164
EXPECTED_ERLOTINIB_HR = 4.28
CANONICAL_PENALIZER = 0.1
HR_RTOL = 1e-6


def load_cohort() -> pd.DataFrame:
    """PAM50-adjusted Cox cohort of script 21 with the predictions of the twelve drugs."""
    pred = pd.read_csv(PRED_PATH, index_col=0,
                       usecols=lambda c: c in (["Unnamed: 0", ""] + DRUGS))
    pred = normalize_drug_columns(pred)
    missing = [d for d in DRUGS if d not in pred.columns]
    if missing:
        raise ValueError(f"Drugs missing from {PRED_PATH}: {missing}")
    if not pred.index.is_unique:
        raise ValueError(f"{PRED_PATH} has duplicated sample ids")

    clin = scanb_patient_cohort()
    if len(clin) != EXPECTED_SURVIVAL_COHORT:
        raise ValueError(f"Survival cohort has {len(clin)} patients, expected {EXPECTED_SURVIVAL_COHORT}")
    merged = clin.join(pred[DRUGS], how="inner")
    merged = merged.dropna(subset=["pam50"])
    n, events = len(merged), int(merged["os_event"].sum())
    if (n, events) != (EXPECTED_COX_N, EXPECTED_COX_EVENTS):
        raise ValueError(f"PAM50-adjusted Cox cohort is {n} patients and {events} events, "
                         f"expected {EXPECTED_COX_N} and {EXPECTED_COX_EVENTS}")
    if merged[DRUGS].isna().any().any():
        raise ValueError("Missing predictions inside the Cox cohort")
    unexpected = sorted(set(merged["pam50"]) - set(PAM50))
    if unexpected:
        raise ValueError(f"Unexpected PAM50 labels: {unexpected}")
    logger.info("PAM50-adjusted Cox cohort: %d patients, %d events (reproduced)", n, events)
    return merged


def check_canonical_cox(cohort: pd.DataFrame) -> dict:
    """Refit the penalized PAM50-adjusted Cox of script 21 and compare with survival_SCANB.csv."""
    published = pd.read_csv(SURV_SCANB).set_index("drug")
    dummies = pd.get_dummies(cohort["pam50"], prefix="sub", drop_first=True)
    refit = {}
    for drug in DRUGS:
        data = pd.concat([cohort[["os_time", "os_event", drug]], dummies], axis=1)
        data = data.rename(columns={drug: "predicted_auc"})
        cph = CoxPHFitter(penalizer=CANONICAL_PENALIZER).fit(data, "os_time", "os_event")
        hr = float(cph.hazard_ratios_["predicted_auc"])
        pub = float(published.loc[drug, "HR"])
        if not np.isclose(hr, pub, rtol=HR_RTOL):
            raise ValueError(f"{drug}: canonical HR {hr:.6f} does not reproduce the published {pub:.6f}")
        refit[drug] = hr
    if round(refit["erlotinib"], 2) != EXPECTED_ERLOTINIB_HR:
        raise ValueError(f"Erlotinib canonical HR {refit['erlotinib']:.4f}, expected {EXPECTED_ERLOTINIB_HR}")
    logger.info("Canonical penalized Cox reproduced for the 12 drugs (erlotinib HR %.4f, paclitaxel HR %.4f)",
                refit["erlotinib"], refit["paclitaxel"])
    return refit


def decode_receptors(cohort: pd.DataFrame) -> pd.DataFrame:
    """Decode er, pr and her2 with the shared helper and flag clinical TNBC."""
    raw_her2 = sorted(cohort["her2"].dropna().unique().tolist())
    logger.info("Raw her2 values in scanb_clinical.csv: %s (dtype %s), %d missing",
                raw_her2, cohort["her2"].dtype, int(cohort["her2"].isna().sum()))
    out = cohort.copy()
    for col in ("er", "pr", "her2"):
        out[col + "_status"] = encode_binary_receptor(cohort[col])
    status = out[["er_status", "pr_status", "her2_status"]]
    out["tnbc"] = status.notna().all(axis=1) & (status == 0).all(axis=1)
    return out


def fit_standardized_cox(data: pd.DataFrame, strata: list[str] | None = None) -> dict:
    """Unpenalized Cox of survival on the standardized predicted AUC."""
    cols = ["os_time", "os_event", "z"] + (strata or [])
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(data[cols], duration_col="os_time", event_col="os_event", strata=strata)
    row = cph.summary.loc["z"]
    return {"hr_per_sd": float(row["exp(coef)"]), "ci_low": float(row["exp(coef) lower 95%"]),
            "ci_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"])}


def subset_cox(cohort: pd.DataFrame, mean_full: pd.Series, sd_full: pd.Series) -> pd.DataFrame:
    """Per-drug standardized Cox in each subset, with BH within family and subset."""
    subsets = {
        "TNBC_clinical": (cohort[cohort["tnbc"]], None),
        "PAM50_Basal": (cohort[cohort["pam50"] == "Basal"], None),
        "full_cohort_pam50_strata": (cohort, ["pam50"]),
    }
    rows = []
    for name, (sub, strata) in subsets.items():
        for drug in DRUGS:
            data = sub[["os_time", "os_event", "pam50"]].copy()
            data["z"] = (sub[drug] - mean_full[drug]) / sd_full[drug]
            res = fit_standardized_cox(data, strata)
            rows.append({"subset": name, "drug": drug, "family": FAMILY[drug], "n": len(sub),
                         "events": int(sub["os_event"].sum()), **res})
    df = pd.DataFrame(rows)
    df["p_adj_family"] = np.nan
    for _, idx in df.groupby(["subset", "family"]).groups.items():
        df.loc[idx, "p_adj_family"] = stats.false_discovery_control(df.loc[idx, "p"].values)
    cols = ["subset", "drug", "family", "n", "events", "hr_per_sd", "ci_low", "ci_high", "p", "p_adj_family"]
    return df[cols]


def pam50_differences(cohort: pd.DataFrame, sd_full: pd.Series) -> pd.DataFrame:
    """Mean predicted AUC per PAM50 subtype, Basal minus LumA shift and Kruskal-Wallis."""
    rows = []
    for drug in DRUGS:
        groups = {s: cohort.loc[cohort["pam50"] == s, drug] for s in PAM50}
        h, p = stats.kruskal(*groups.values())
        row = {"drug": drug, "family": FAMILY[drug]}
        row.update({f"n_{s}": len(g) for s, g in groups.items()})
        row.update({f"mean_auc_{s}": float(g.mean()) for s, g in groups.items()})
        row["sd_full_cohort"] = float(sd_full[drug])
        row["basal_minus_luma_sd"] = (row["mean_auc_Basal"] - row["mean_auc_LumA"]) / sd_full[drug]
        row["basal_less_sensitive_than_luma"] = bool(row["mean_auc_Basal"] > row["mean_auc_LumA"])
        row["kruskal_h"] = float(h)
        row["kruskal_p"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def composition_by_half(cohort: pd.DataFrame) -> pd.DataFrame:
    """PAM50 percentages in the halves of the cohort split at the median predicted AUC."""
    rows = []
    for drug in DRUGS:
        med = cohort[drug].median()
        halves = {"sensitive": cohort[cohort[drug] < med], "resistant": cohort[cohort[drug] >= med]}
        for half, part in halves.items():
            pct = part["pam50"].value_counts(normalize=True).reindex(PAM50, fill_value=0) * 100
            row = {"drug": drug, "family": FAMILY[drug], "half": half, "median_auc": float(med),
                   "n": len(part), "events": int(part["os_event"].sum())}
            row.update({f"pct_{s}": float(pct[s]) for s in PAM50})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cohort = load_cohort()
    canonical = check_canonical_cox(cohort)
    cohort = decode_receptors(cohort)

    # Basal subset of script 19b, whose cohort differs from this one only by the PAM50 filter
    pooling = pd.read_csv(POOLING_SUMMARY).set_index("subset")
    basal = cohort[cohort["pam50"] == "Basal"]
    n_basal, ev_basal = len(basal), int(basal["os_event"].sum())
    ref = (int(pooling.loc["Basal", "n_patients"]), int(pooling.loc["Basal", "n_events"]))
    if (n_basal, ev_basal) != ref:
        raise ValueError(f"Basal subset is {n_basal} patients and {ev_basal} events, script 19b reports {ref}")
    logger.info("PAM50 Basal subset: %d patients, %d events (reproduced from script 19b)", n_basal, ev_basal)

    tnbc = cohort[cohort["tnbc"]]
    n_tnbc, ev_tnbc = len(tnbc), int(tnbc["os_event"].sum())
    logger.info("Clinical TNBC (ER-, PR-, HER2-, all known): %d patients, %d events", n_tnbc, ev_tnbc)
    tnbc_pam50 = tnbc["pam50"].value_counts().reindex(PAM50, fill_value=0)
    logger.info("PAM50 of clinical TNBC: %s", tnbc_pam50.to_dict())

    mean_full = cohort[DRUGS].mean()
    sd_full = cohort[DRUGS].std()

    cox = subset_cox(cohort, mean_full, sd_full)
    cox.to_csv(OUT_DIR / "tnbc_subgroup_cox.csv", index=False)
    logger.info("Standardized unpenalized Cox per subset:\n%s", cox.to_string(index=False))

    diffs = pam50_differences(cohort, sd_full)
    diffs.to_csv(OUT_DIR / "pam50_predicted_auc.csv", index=False)
    logger.info("Predicted AUC by PAM50:\n%s", diffs[["drug", "basal_minus_luma_sd", "kruskal_p"]].to_string(index=False))

    comp = composition_by_half(cohort)
    comp.to_csv(OUT_DIR / "pam50_composition_sensitive_half.csv", index=False)
    sens = comp[comp["half"] == "sensitive"].set_index("drug")
    logger.info("PAM50 composition of the sensitive half:\n%s",
                sens[[f"pct_{s}" for s in PAM50]].round(1).to_string())

    # Summary
    summary = {
        "n_cox_cohort": EXPECTED_COX_N,
        "events_cox_cohort": EXPECTED_COX_EVENTS,
        "canonical_hr_erlotinib_pen01": round(canonical["erlotinib"], 4),
        "canonical_hr_paclitaxel_pen01": round(canonical["paclitaxel"], 4),
        "her2_encoding": "numeric 1.0 positive, 0.0 negative, NaN missing (decoded by encode_binary_receptor)",
        "her2_positive": int((cohort["her2_status"] == 1).sum()),
        "her2_negative": int((cohort["her2_status"] == 0).sum()),
        "her2_missing": int(cohort["her2_status"].isna().sum()),
        "er_missing": int(cohort["er_status"].isna().sum()),
        "pr_missing": int(cohort["pr_status"].isna().sum()),
        "n_receptors_all_known": int(cohort[["er_status", "pr_status", "her2_status"]].notna().all(axis=1).sum()),
        "n_tnbc_clinical": n_tnbc,
        "events_tnbc_clinical": ev_tnbc,
        "n_pam50_basal": n_basal,
        "events_pam50_basal": ev_basal,
        "n_tnbc_and_basal": int((cohort["tnbc"] & (cohort["pam50"] == "Basal")).sum()),
        "pct_tnbc_that_are_basal": round(100 * tnbc_pam50["Basal"] / n_tnbc, 1),
        "pct_basal_that_are_tnbc": round(100 * float(basal["tnbc"].mean()), 1),
    }
    for s in PAM50:
        summary[f"n_tnbc_pam50_{s}"] = int(tnbc_pam50[s])
    # Base rates of the PAM50 subtypes, the reference for the composition of each half
    cohort_pct = cohort["pam50"].value_counts(normalize=True).reindex(PAM50) * 100
    for s in PAM50:
        summary[f"cohort_pct_{s}"] = round(float(cohort_pct[s]), 1)
    # Spread of the predictions inside each subset relative to the full cohort, which
    # tells how many full-cohort standard deviations a subset actually spans
    for name, sub in (("tnbc", tnbc), ("basal", basal)):
        ratio = sub[DRUGS].std() / sd_full
        summary[f"median_sd_ratio_{name}_vs_full"] = round(float(ratio.median()), 3)
    for subset in cox["subset"].unique():
        for fam in ("candidate", "positive_control"):
            part = cox[(cox["subset"] == subset) & (cox["family"] == fam)]
            summary[f"{subset}_{fam}_n_padj05"] = int((part["p_adj_family"] < 0.05).sum())
            summary[f"{subset}_{fam}_n_p05"] = int((part["p"] < 0.05).sum())
            summary[f"{subset}_{fam}_n_hr_above1"] = int((part["hr_per_sd"] > 1).sum())
            summary[f"{subset}_{fam}_n_ci_excludes1"] = int(((part["ci_low"] > 1) | (part["ci_high"] < 1)).sum())
    cand = diffs[diffs["family"] == "candidate"]
    ctrl = diffs[diffs["family"] == "positive_control"]
    summary["n_candidates_less_sensitive_in_basal_than_luma"] = int(cand["basal_less_sensitive_than_luma"].sum())
    summary["n_controls_less_sensitive_in_basal_than_luma"] = int(ctrl["basal_less_sensitive_than_luma"].sum())
    summary["median_abs_basal_minus_luma_sd_candidates"] = round(float(cand["basal_minus_luma_sd"].abs().median()), 3)
    summary["n_drugs_kruskal_p_below_1e-10"] = int((diffs["kruskal_p"] < 1e-10).sum())
    for drug in DRUGS:
        summary[f"sensitive_half_pct_luma_{drug}"] = round(float(sens.loc[drug, "pct_LumA"]), 1)
        summary[f"sensitive_half_pct_basal_{drug}"] = round(float(sens.loc[drug, "pct_Basal"]), 1)
    pd.DataFrame({"metric": list(summary), "value": list(summary.values())}).to_csv(
        OUT_DIR / "tnbc_subgroup_summary.csv", index=False)
    logger.info("Summary:\n%s", pd.Series(summary).to_string())
    logger.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
