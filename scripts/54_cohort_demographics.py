#!/usr/bin/env python3
"""
PRECISION - Demographics and population descriptors of the analysed cohorts.

For each survival cohort (SCAN-B, METABRIC, TCGA-BRCA) this rebuilds the population with
the filters of scripts/21_multicohort_survival.py (index alignment, one sample per patient,
TCGA primary tumours, inner join with the TF-activity samples, non-missing overall
survival, os_time > 0), checks that its size and events equal what script 21 recorded in
cohort_summary.csv, both before and after the PAM50 filter of the Cox models, and only
then computes aggregate descriptors. Descriptors refer to the population of the
PAM50-adjusted Cox models, which is the N reported in Table 2. Sex is also reported on
the survival population, because TCGA-BRCA loses its male patients at the PAM50 filter.

For AURORA-US, sex, disease state, biopsy site, treatment timing and autopsy status are
read from the characteristics of the GEO series matrix.

Aggregates only. No patient-level row is printed or written.

Output: results/v10_reporting/cohort_demographics.csv  (cohort, population, variable, value)

Run from project root:
    python scripts/54_cohort_demographics.py
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

PROC = DATA_DIR / "precision_processed"
AURORA_MATRIX = DATA_DIR / "AURORA-US/GSE209998_series_matrix.txt.gz"
COHORT_SUMMARY = RESULTS_DIR / "v8_multicohort/cohort_summary.csv"
OUT = RESULTS_DIR / "v10_reporting"

COHORTS = {
    "SCANB": {"tf": "tf_activities_scanb.csv", "clin": "scanb_clinical.csv", "subtype": "pam50"},
    "METABRIC": {"tf": "tf_activities_metabric.csv", "clin": "metabric_clinical.csv", "subtype": "PAM50"},
    "TCGA": {"tf": "tf_activities_tcga.csv", "clin": "tcga_clinical.csv",
             "subtype": "paper_BRCA_Subtype_PAM50"},
}


def tf_index(name: str) -> pd.Index:
    return pd.Index(pd.read_csv(PROC / name, usecols=[0]).iloc[:, 0].astype(str))


def populations(cohort: str, summary: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Survival population and PAM50-adjusted Cox population, checked against script 21."""
    cfg = COHORTS[cohort]
    tf_idx = tf_index(cfg["tf"])
    clin = pd.read_csv(PROC / cfg["clin"], index_col=0, low_memory=False)
    clin.index = clin.index.astype(str)

    # script 21, index alignment
    if len(set(tf_idx) & set(clin.index)) == 0:
        for col in ("patientId", "PATIENT_ID", "sample_id"):
            if col in clin.columns:
                clin = clin.set_index(col)
                clin.index = clin.index.astype(str)
                break

    if cohort == "SCANB":
        clin = clin.sort_index()
        clin = clin[~clin["patient_id"].duplicated(keep="first")]
    if cohort == "TCGA":
        clin = clin[clin["shortLetterCode"] == "TP"].sort_index()
        clin = clin[~clin["submitter_id"].duplicated(keep="first")]
    n_after_dedup = len(clin)

    if cohort == "SCANB":
        os_time = pd.to_numeric(clin["os_time"], errors="coerce")
        os_event = pd.to_numeric(clin["os_event"], errors="coerce")
    elif cohort == "METABRIC":
        os_time = pd.to_numeric(clin["OS_MONTHS"], errors="coerce") * 30.44
        os_event = clin["OS_STATUS"].astype(str).str.extract(r"^(\d)")[0].astype(float)
    else:
        dtd = pd.to_numeric(clin["days_to_death"], errors="coerce")
        dtf = pd.to_numeric(clin["days_to_last_follow_up"], errors="coerce")
        os_time = dtd.fillna(dtf)
        os_event = (clin["vital_status"] == "Dead").astype(float)
    clin = clin.assign(_os_time=os_time, _os_event=os_event)

    surv = clin.join(pd.DataFrame(index=tf_idx), how="inner")
    surv = surv.dropna(subset=["_os_time", "_os_event"])
    surv = surv[surv["_os_time"] > 0]
    cox = surv[surv[cfg["subtype"]].notna()]

    checks = {
        "n_patients_after_dedup": (n_after_dedup, summary["n_patients_after_dedup"]),
        "n_patients_with_survival": (len(surv), summary["n_patients_with_survival"]),
        "n_events": (int(surv["_os_event"].sum()), summary["n_events"]),
        "n_patients_cox": (len(cox), summary["n_patients_cox"]),
        "n_events_cox": (int(cox["_os_event"].sum()), summary["n_events_cox"]),
    }
    for key, (got, expected) in checks.items():
        if int(got) != int(expected):
            raise AssertionError(f"{cohort} {key}: rebuilt {got}, script 21 recorded {expected}")
    return surv, cox


def numeric_rows(cohort, population, variable, s):
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    rows = [(cohort, population, f"{variable}_n_missing", int(s.isna().sum()))]
    if valid.empty:
        return rows
    q1, med, q3 = np.percentile(valid, [25, 50, 75], method="linear")
    for name, val in [("n", len(valid)), ("median", med), ("q1", q1), ("q3", q3),
                      ("min", valid.min()), ("max", valid.max())]:
        rows.append((cohort, population, f"{variable}_{name}",
                     int(val) if name == "n" else round(float(val), 1)))
    return rows


def count_rows(cohort, population, variable, s):
    labelled = s.astype(object).where(s.notna(), "MISSING")
    return [(cohort, population, f"{variable}_{k}", int(v))
            for k, v in labelled.value_counts().items()]


def aurora_rows():
    fields = {}
    with gzip.open(AURORA_MATRIX, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            values = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            key = next((v.split(":", 1)[0].strip() for v in values if ":" in v), None)
            if key:
                fields[key] = [v.split(":", 1)[1].strip() if ":" in v else "" for v in values]
    df = pd.DataFrame(fields)
    rows = [("AURORA", "samples", "n_samples", len(df))]
    for col, variable in [("genotype", "sex"), ("disease", "disease"),
                          ("treatment", "treatment_timing"), ("time", "autopsy")]:
        rows += [("AURORA", "samples", f"{variable}_{k}", int(v))
                 for k, v in Counter(df[col]).most_common()]
    meta = df[df["disease"] == "Metastatic tumor"]
    sites = Counter(meta["tissue"])
    rows.append(("AURORA", "metastatic_samples", "n_distinct_sites", len(sites)))
    rows += [("AURORA", "metastatic_samples", f"site_{k}", int(v)) for k, v in sites.most_common()]
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(COHORT_SUMMARY).set_index("cohort")
    rows = []
    for cohort, cfg in COHORTS.items():
        surv, cox = populations(cohort, summary.loc[cohort])
        rows += count_rows(cohort, "survival", "pam50", surv[cfg["subtype"]])
        rows += count_rows(cohort, "cox", "pam50", cox[cfg["subtype"]])
        if cohort == "SCANB":
            rows.append((cohort, "cox", "sex_variable_available", 0))
            rows += numeric_rows(cohort, "cox", "age_years_5y_bins", cox["age"])
        elif cohort == "METABRIC":
            rows += count_rows(cohort, "survival", "sex", surv["SEX"])
            rows += count_rows(cohort, "cox", "sex", cox["SEX"])
            rows += numeric_rows(cohort, "cox", "age_years", cox["AGE_AT_DIAGNOSIS"])
        else:
            rows += count_rows(cohort, "survival", "sex", surv["gender"])
            rows += count_rows(cohort, "cox", "sex", cox["gender"])
            rows += numeric_rows(cohort, "cox", "age_years_at_index", cox["age_at_index"])
            rows += count_rows(cohort, "cox", "age_obfuscated", cox["age_is_obfuscated"])
    rows += aurora_rows()

    out = pd.DataFrame(rows, columns=["cohort", "population", "variable", "value"])
    out.to_csv(OUT / "cohort_demographics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

