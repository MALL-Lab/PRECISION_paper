#!/usr/bin/env python3
"""Treatment received as a confounder and as an effect modifier of the survival associations.

The survival screen of scripts 21 and 31 adjusts only for PAM50. Patients who
received chemotherapy differ in prognosis from those who did not, and a predicted
sensitivity that tracks a treatment benefit (predictive) behaves differently from
one that only tracks baseline risk (prognostic). This script asks both questions
for the seven Table 5 candidates and the five positive controls, with the
predicted AUC standardized by its mean and SD in the full Cox cohort (HR per SD)
and PAM50 as strata of an unpenalized Cox model.

(A) Treatment-adjusted model. Three nested models on the same patients:
    pam50_strata (predicted AUC only), clinical (plus the covariates of
    scripts/37_cox_clinical_covariates.py) and clinical_treatment (plus
    chemotherapy and endocrine therapy). Missing receptor status and missing
    treatment are their own category through an indicator column, never coded
    as negative and never dropped. The other clinical covariates are complete
    case, as in script 37. For comparison only, the clinical model is also
    fitted as script 37 fits it (clinical_pen01_dummies: penalizer 0.1 and PAM50
    dummies), because the penalty shrinks the age coefficient and so
    under-adjusts for age.
(B) Interaction with chemotherapy. Patients with known chemotherapy only, model
    with the standardized predicted AUC, chemotherapy and their product
    (adjustment = pam50_strata). The HR per SD within treated and within
    untreated patients comes from the same model (linear contrast) and from
    separate fits in each group (suffix _sep). Chemotherapy is not randomized,
    so the same interaction is refitted with the clinical covariates of (A) and
    endocrine therapy as main effects (adjustment = clinical_endo, complete
    case). The drug-specific exposures (taxane for paclitaxel and docetaxel,
    anthracycline for epirubicin) are tested where the cohort records them.
    The SCAN-B columns anthra and taxan are empty for every sample of
    data/SCANB/clinical_formatted.rds, and METABRIC records only CHEMOTHERAPY,
    so these rows carry a status instead of estimates.

Before any new model, the script reproduces the published penalized PAM50 Cox
of scripts 21 and 31 (penalizer 0.1, PAM50 dummies, raw predicted AUC) for all
twelve drugs in both cohorts, and the cohort sizes of Table 2, and raises if
any of them differs.

Inputs:
  results/v5_final/scanb_perdrug_predictions.csv     (script 17, SCAN-B predictions)
  data/precision_processed/scanb_clinical.csv        (script 09)
  data/precision_processed/metabric_clinical.csv     (scripts 20 and 20b)
  data/precision_processed/tf_activities_metabric.csv and the PRISM inputs read by
  predict_metabric() of script 37, which recomputes the METABRIC predictions
  results/v8_multicohort/survival_SCANB.csv and survival_METABRIC.csv (reproduction only)
Outputs in results/v13_reaudit/:
  treatment_adjusted_cox.csv     one row per cohort, model and drug (HR per SD,
                                 95% CI, p, BH padj over the 12 drugs of the model)
  treatment_interaction_cox.csv  one row per cohort, drug, exposure and adjustment
                                 (n_cohort is the analysis set before the exposure
                                 filter, n_patients the patients with known exposure)
  treatment_summary.csv          metric, value

Run from project root:
    python scripts/62_treatment_interaction.py
"""

import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data.clinical_encoding import encode_binary_receptor
from src.data.scanb_cohort import scanb_patient_cohort
from src.data.splits import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("treatment_interaction")

SEED = 42
PRED_PATH = RESULTS_DIR / "v5_final/scanb_perdrug_predictions.csv"
SCANB_PUBLISHED = RESULTS_DIR / "v8_multicohort/survival_SCANB.csv"
METABRIC_PUBLISHED = RESULTS_DIR / "v8_multicohort/survival_METABRIC.csv"
METABRIC_CLINICAL = DATA_DIR / "precision_processed/metabric_clinical.csv"
OUT = RESULTS_DIR / "v13_reaudit"

CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
DRUGS = CANDIDATES + CONTROLS
# Exposure that matches the mechanism of a drug, tested on top of chemotherapy
DRUG_SPECIFIC_EXPOSURE = {"paclitaxel": "taxan", "docetaxel": "taxan", "epirubicin": "anthra"}

# Table 2 cohorts of the PAM50-adjusted Cox models (results/v8_multicohort/cohort_summary.csv)
EXPECTED_COHORT = {"SCANB": (7215, 1164), "METABRIC": (1979, 1143)}
# Drug whose published HR is quoted in the log and the summary for each cohort
HEADLINE_REPRO = {"SCANB": "erlotinib", "METABRIC": "epirubicin"}
PUBLISHED_PENALIZER = 0.1
REPRO_RTOL = 1e-6

MODELS = ["pam50_strata", "clinical", "clinical_treatment"]
# Clinical model with the fitting choices of script 37, reported for comparison only
REFERENCE_MODEL = "clinical_pen01_dummies"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S37 = load_module("s37_cox_clinical", ROOT / "scripts" / "37_cox_clinical_covariates.py")


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def encode_treatment(values: pd.Series, label: str) -> pd.Series:
    """Map a treatment column to 1.0 (received), 0.0 (not received) or NaN (missing).

    SCAN-B stores "chemo" and "no-chemo" (and the same pattern for endo), METABRIC
    stores "YES" and "NO". Any other value raises, so that a new encoding is never
    silently read as "not treated".
    """
    text = values.map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
    yes = {label, "yes", "1", "1.0"}
    no = {f"no-{label}", "no", "0", "0.0"}
    missing = {"", "nan", "na", "none"}
    unknown = sorted(set(text[~text.isin(yes | no | missing)]))
    if unknown:
        raise ValueError(f"Unrecognised values for {label}: {unknown}")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out[text.isin(yes)] = 1.0
    out[text.isin(no)] = 0.0
    return out


def encode_her2_snp6(values: pd.Series) -> pd.Series:
    """METABRIC HER2_SNP6 copy-number call: GAIN is positive as in script 37.

    NEUTRAL and LOSS are negative. UNDEF (undetermined call) and NaN are missing,
    which script 37 codes as negative and this script keeps as its own category.
    """
    text = values.map(lambda v: "" if pd.isna(v) else str(v).strip().upper())
    pos, neg, missing = {"GAIN", "AMP", "POSITIVE"}, {"NEUTRAL", "LOSS"}, {"", "UNDEF"}
    unknown = sorted(set(text[~text.isin(pos | neg | missing)]))
    if unknown:
        raise ValueError(f"Unrecognised HER2_SNP6 values: {unknown}")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out[text.isin(pos)] = 1.0
    out[text.isin(neg)] = 0.0
    return out


def with_missing_indicator(coded: pd.Series, name: str) -> pd.DataFrame:
    """Binary variable as two columns, name (1 if positive) and name_missing.

    Missing patients get name = 0 and name_missing = 1, so they form their own
    category with the negative patients as reference.
    """
    return pd.DataFrame({name: (coded == 1).astype(int),
                         f"{name}_missing": coded.isna().astype(int)}, index=coded.index)


def collapse_identical_columns(design: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Drop indicator columns identical to an earlier one, which would make the fit singular.

    In SCAN-B chemo, endo and immu are missing for exactly the same patients, so the
    two treatment missing indicators are one variable.
    """
    kept, dropped = [], []
    for col in candidates:
        # Constant columns are left to keep_covariates of script 37, which drops them
        if col not in design.columns or design[col].nunique() <= 1:
            continue
        twin = next((k for k in kept if design[k].equals(design[col])), None)
        if twin is None:
            kept.append(col)
        else:
            dropped.append(col)
            logger.info("  %s is identical to %s, one indicator kept", col, twin)
    return design.drop(columns=dropped)


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------

def scanb_cox_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """PAM50-adjusted SCAN-B Cox cohort of script 21 and its predicted AUCs."""
    pred = pd.read_csv(PRED_PATH, index_col=0, usecols=lambda c: c == "Unnamed: 0" or c in DRUGS)
    missing = [d for d in DRUGS if d not in pred.columns]
    if missing:
        raise ValueError(f"SCAN-B predictions lack {missing}")
    clin = scanb_patient_cohort(DATA_DIR)
    cohort = clin.join(pred[DRUGS], how="inner").dropna(subset=["pam50"])
    for col in ("age", "grade", "lymphNodePos", "t.stage"):
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
    return cohort, cohort[DRUGS]


def metabric_cox_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """OS cohort of script 21 for METABRIC, with predictions of predict_metabric() of script 37."""
    S37.TARGETS = DRUGS
    pred = S37.predict_metabric()
    missing = [d for d in DRUGS if d not in pred.columns]
    if missing:
        raise ValueError(f"METABRIC predictions lack {missing}")
    clin = pd.read_csv(METABRIC_CLINICAL, index_col=0).set_index("patientId")
    clin["os_time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce") * 30.44
    clin["os_event"] = clin["OS_STATUS"].str.extract(r"^(\d)")[0].astype(float)
    clin["age"] = pd.to_numeric(clin["AGE_AT_DIAGNOSIS"], errors="coerce")
    clin["ln_pos"] = pd.to_numeric(clin["LYMPH_NODES_EXAMINED_POSITIVE"], errors="coerce")
    clin["npi"] = pd.to_numeric(clin["NPI"], errors="coerce")
    cohort = clin.join(pred[DRUGS], how="inner").dropna(subset=["os_time", "os_event"])
    cohort = cohort[cohort["os_time"] > 0].dropna(subset=["PAM50"])
    cohort = cohort.rename(columns={"PAM50": "pam50"})
    return cohort, cohort[DRUGS]


def reproduce_published(cohort_name: str, cohort: pd.DataFrame, published: Path) -> dict:
    """Penalized PAM50-dummy Cox of scripts 21 and 31, compared drug by drug with the published HR."""
    n, events = len(cohort), int(cohort["os_event"].sum())
    if (n, events) != EXPECTED_COHORT[cohort_name]:
        raise ValueError(f"{cohort_name}: cohort is {n} patients and {events} events, "
                         f"expected {EXPECTED_COHORT[cohort_name]}")
    pub = pd.read_csv(published).set_index("drug")["HR"]
    pam = pd.get_dummies(cohort["pam50"], prefix="sub", drop_first=True)
    got = {}
    for drug in DRUGS:
        data = pd.concat([cohort[["os_time", "os_event"]],
                          cohort[drug].rename("predicted_auc"), pam], axis=1)
        cph = CoxPHFitter(penalizer=PUBLISHED_PENALIZER)
        cph.fit(data, duration_col="os_time", event_col="os_event")
        got[drug] = float(cph.hazard_ratios_["predicted_auc"])
        if not np.isclose(got[drug], pub[drug], rtol=REPRO_RTOL, atol=0):
            raise ValueError(f"{cohort_name} {drug}: HR {got[drug]:.6f} does not reproduce "
                             f"the published {pub[drug]:.6f}")
    head = HEADLINE_REPRO[cohort_name]
    logger.info("%s: %d patients, %d events, published HR reproduced for %d drugs "
                "(%s %.4f, published %.4f)", cohort_name, n, events, len(got), head,
                got[head], pub[head])
    return {"n": n, "events": events, "drug": head, "expected": float(pub[head]),
            "got": got[head], "n_drugs": len(got)}


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def scanb_design(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Script 37 covariates for SCAN-B plus chemo and endo, complete case on the non-receptor ones."""
    cc = cohort.dropna(subset=["age", "t.stage", "grade", "lymphNodePos"])
    stg = pd.get_dummies(cc["t.stage"], prefix="st", drop_first=True, dtype=int)
    er = with_missing_indicator(encode_binary_receptor(cc["er"]), "er_pos")
    pr = with_missing_indicator(encode_binary_receptor(cc["pr"]), "pr_pos")
    chemo = with_missing_indicator(encode_treatment(cc["chemo"], "chemo"), "chemo")
    endo = with_missing_indicator(encode_treatment(cc["endo"], "endo"), "endo")
    design = pd.concat([cc[["os_time", "os_event", "pam50", "age", "grade", "lymphNodePos"]],
                       stg, er, pr, chemo, endo], axis=1)
    design = collapse_identical_columns(
        design, ["er_pos_missing", "pr_pos_missing", "chemo_missing", "endo_missing"])
    clinical = ["age", "grade", "lymphNodePos", *stg.columns, "er_pos", "pr_pos"]
    clinical += [c for c in ("er_pos_missing", "pr_pos_missing") if c in design.columns]
    treatment = ["chemo", "endo"] + [c for c in ("chemo_missing", "endo_missing") if c in design.columns]
    required = [c for c in S37.SCANB_COVARIATES if c != "predicted_auc"]
    return design, {"clinical": clinical, "treatment": treatment, "required": required}


def metabric_design(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Script 37 covariates for METABRIC plus CHEMOTHERAPY and HORMONE_THERAPY."""
    cc = cohort.dropna(subset=["age", "ln_pos", "npi"])
    er = with_missing_indicator(encode_binary_receptor(cc["ER_IHC"]), "er_pos")
    her2 = with_missing_indicator(encode_her2_snp6(cc["HER2_SNP6"]), "her2_pos")
    chemo = with_missing_indicator(encode_treatment(cc["CHEMOTHERAPY"], "chemo"), "chemo")
    endo = with_missing_indicator(encode_treatment(cc["HORMONE_THERAPY"], "endo"), "endo")
    design = pd.concat([cc[["os_time", "os_event", "pam50", "age", "ln_pos", "npi"]],
                       er, her2, chemo, endo], axis=1)
    design = collapse_identical_columns(
        design, ["er_pos_missing", "her2_pos_missing", "chemo_missing", "endo_missing"])
    clinical = ["age", "ln_pos", "npi", "er_pos", "her2_pos"]
    clinical += [c for c in ("er_pos_missing", "her2_pos_missing") if c in design.columns]
    treatment = ["chemo", "endo"] + [c for c in ("chemo_missing", "endo_missing") if c in design.columns]
    required = [c for c in S37.METABRIC_COVARIATES if c != "predicted_auc"]
    return design, {"clinical": clinical, "treatment": treatment, "required": required}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_strata_cox(data: pd.DataFrame, label: str, penalizer: float = 0.0) -> CoxPHFitter:
    """Cox with PAM50 strata, unpenalized by default. Convergence warnings are logged, not hidden.

    If data has no pam50 column the model is fitted without strata, which is how the
    script 37 reference model uses PAM50 dummies instead.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    strata = ["pam50"] if "pam50" in data.columns else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph.fit(data, duration_col="os_time", event_col="os_event", strata=strata)
    for w in caught:
        logger.warning("  %s: %s", label, str(w.message).splitlines()[0])
    return cph


def term(cph: CoxPHFitter, name: str) -> dict:
    row = cph.summary.loc[name]
    return {"HR": float(row["exp(coef)"]), "CI_low": float(row["exp(coef) lower 95%"]),
            "CI_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"])}


def contrast(cph: CoxPHFitter, a: str, b: str) -> dict:
    """HR, 95% CI and Wald p of exp(beta_a + beta_b) from the model covariance."""
    beta = cph.params_[a] + cph.params_[b]
    v = cph.variance_matrix_
    se = float(np.sqrt(v.loc[a, a] + v.loc[b, b] + 2 * v.loc[a, b]))
    z = beta / se
    return {"HR": float(np.exp(beta)), "CI_low": float(np.exp(beta - 1.959964 * se)),
            "CI_high": float(np.exp(beta + 1.959964 * se)), "p": float(2 * stats.norm.sf(abs(z)))}


def adjusted_models(cohort_name: str, design: pd.DataFrame, cols: dict,
                    auc_z: pd.DataFrame) -> list[dict]:
    """(A) Three nested models on the same patients for every drug, plus the script 37 reference.

    The reference model clinical_pen01_dummies is the clinical model fitted as script 37
    fits it (penalizer 0.1, PAM50 dummies instead of strata). It is not part of (A),
    it shows how much of the difference from the published clinical-adjusted p-values
    comes from the penalty, which also shrinks the age coefficient.
    """
    n, events = len(design), int(design["os_event"].sum())
    logger.info("%s (A): %d patients, %d events after complete case on the clinical covariates",
                cohort_name, n, events)
    pam = pd.get_dummies(design["pam50"], prefix="sub", drop_first=True, dtype=int)
    spec = {"pam50_strata": [],
            "clinical": cols["clinical"],
            "clinical_treatment": cols["clinical"] + cols["treatment"],
            "clinical_pen01_dummies": cols["clinical"]}
    required = {"pam50_strata": ["auc_z"],
                "clinical": ["auc_z"] + cols["required"],
                "clinical_treatment": ["auc_z"] + cols["required"] + ["chemo", "endo"],
                "clinical_pen01_dummies": ["auc_z"] + cols["required"]}
    rows = []
    for model in MODELS + [REFERENCE_MODEL]:
        for drug in DRUGS:
            data = design[["os_time", "os_event", "pam50"] + spec[model]].copy()
            data["auc_z"] = auc_z.loc[design.index, drug]
            penalizer = 0.0
            if model == REFERENCE_MODEL:
                data = pd.concat([data.drop(columns="pam50"), pam], axis=1)
                penalizer = PUBLISHED_PENALIZER
            data = clean_design(data, required[model], cohort_name)
            cph = fit_strata_cox(data, f"{cohort_name} {model} {drug}", penalizer=penalizer)
            row = {"cohort": cohort_name, "model": model, "drug": drug,
                   "group": "candidate" if drug in CANDIDATES else "positive_control",
                   "n_patients": n, "n_events": events, "penalizer": penalizer,
                   "covariates": " + ".join(c for c in data.columns
                                            if c not in ("os_time", "os_event", "pam50"))}
            row.update(term(cph, "auc_z"))
            if "age" in data.columns:
                row["HR_age_per_year"] = float(cph.hazard_ratios_["age"])
            if model == "clinical_treatment":
                for t in ("chemo", "endo"):
                    tt = term(cph, t)
                    row[f"HR_{t}"], row[f"p_{t}"] = tt["HR"], tt["p"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out["padj"] = out.groupby("model")["p"].transform(lambda p: stats.false_discovery_control(p))
    return out.to_dict("records")


def clean_design(data: pd.DataFrame, required: list[str], cohort_name: str) -> pd.DataFrame:
    """Merge identical missing indicators, then drop constant columns with keep_covariates of script 37."""
    data = collapse_identical_columns(data, [c for c in data.columns if c.endswith("_missing")])
    return S37.keep_covariates(data, required, cohort_name)


def interaction_model(cohort_name: str, frame: pd.DataFrame, auc_z: pd.DataFrame,
                      drug: str, exposure_name: str, exposure: pd.Series,
                      covariates: list[str] | None = None, required: list[str] | None = None,
                      adjustment: str = "pam50_strata") -> dict:
    """(B) Predicted AUC by exposure interaction on the patients of frame with known exposure.

    With covariates, the same model is fitted with those clinical main effects added
    (sensitivity, complete case). Covariates listed in required must not be constant.
    """
    covariates, required = list(covariates or []), list(required or [])
    exposure = exposure.loc[frame.index]
    base = {"cohort": cohort_name, "drug": drug, "exposure": exposure_name,
            "adjustment": adjustment,
            "group": "candidate" if drug in CANDIDATES else "positive_control",
            "n_cohort": len(frame), "n_exposure_known": int(exposure.notna().sum())}
    known = exposure.dropna()
    if known.empty:
        return {**base, "status": "exposure_not_recorded"}
    if known.nunique() < 2:
        return {**base, "status": "exposure_constant"}
    data = frame.loc[known.index, ["os_time", "os_event", "pam50"] + covariates].copy()
    data["auc_z"] = auc_z.loc[known.index, drug]
    data["exposed"] = known.astype(int)
    data["auc_z_x_exposed"] = data["auc_z"] * data["exposed"]
    data = clean_design(data, ["auc_z", "exposed", "auc_z_x_exposed"] + required, cohort_name)
    label = f"{cohort_name} {drug} x {exposure_name} ({adjustment})"
    cph = fit_strata_cox(data, label)
    row = {**base, "status": "ok", "n_patients": len(data), "n_events": int(data["os_event"].sum()),
           "n_exposed": int(data["exposed"].sum()), "n_unexposed": int((data["exposed"] == 0).sum()),
           "events_exposed": int(data.loc[data["exposed"] == 1, "os_event"].sum()),
           "events_unexposed": int(data.loc[data["exposed"] == 0, "os_event"].sum())}
    for prefix, stat in (("interaction", term(cph, "auc_z_x_exposed")),
                         ("exposure_main", term(cph, "exposed")),
                         ("within_exposed", contrast(cph, "auc_z", "auc_z_x_exposed")),
                         ("within_unexposed", term(cph, "auc_z"))):
        row.update({f"{k}_{prefix}": v for k, v in stat.items()})
    # Separate fits in each group, which let the PAM50 baseline hazards differ by exposure
    kept = [c for c in covariates if c in data.columns]
    for flag, prefix in ((1, "within_exposed_sep"), (0, "within_unexposed_sep")):
        part = data.loc[data["exposed"] == flag, ["os_time", "os_event", "pam50", "auc_z"] + kept]
        part = clean_design(part, ["auc_z"], cohort_name)
        stat = term(fit_strata_cox(part, f"{label} group {flag}"), "auc_z")
        row.update({f"{k}_{prefix}": v for k, v in stat.items()})
    return row


def exposure_counts(values: pd.Series) -> dict:
    return {"yes": int((values == 1).sum()), "no": int((values == 0).sum()),
            "missing": int(values.isna().sum())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    set_all_seeds(SEED)
    summary = []

    def put(metric, value):
        summary.append({"metric": metric, "value": value})

    logger.info("Loading SCAN-B cohort and predictions...")
    scanb, scanb_pred = scanb_cox_cohort()
    logger.info("Recomputing METABRIC predictions with predict_metabric() of script 37...")
    metabric, metabric_pred = metabric_cox_cohort()

    cohorts = {
        "SCANB": {"cohort": scanb, "pred": scanb_pred, "published": SCANB_PUBLISHED,
                  "design": scanb_design,
                  "treatments": {"chemo": ("chemo", "chemo"), "endo": ("endo", "endo"),
                                 "taxan": ("taxan", "taxan"), "anthra": ("anthra", "anthra")}},
        "METABRIC": {"cohort": metabric, "pred": metabric_pred, "published": METABRIC_PUBLISHED,
                     "design": metabric_design,
                     "treatments": {"chemo": ("CHEMOTHERAPY", "chemo"),
                                    "endo": ("HORMONE_THERAPY", "endo")}},
    }

    adjusted, interaction = [], []
    for name, c in cohorts.items():
        cohort, pred = c["cohort"], c["pred"]
        rep = reproduce_published(name, cohort, c["published"])
        put(f"{name}_cox_n_patients", rep["n"])
        put(f"{name}_cox_n_events", rep["events"])
        put(f"{name}_repro_{rep['drug']}_HR_published", round(rep["expected"], 6))
        put(f"{name}_repro_{rep['drug']}_HR_reproduced", round(rep["got"], 6))
        put(f"{name}_repro_n_drugs_matched", rep["n_drugs"])

        # Standardized by the mean and SD of the full Cox cohort, so HR is per SD
        auc_z = (pred - pred.mean()) / pred.std()
        for drug in DRUGS:
            put(f"{name}_{drug}_auc_mean", float(pred[drug].mean()))
            put(f"{name}_{drug}_auc_sd", float(pred[drug].std()))

        exposures = {}
        for key, (col, label) in c["treatments"].items():
            exposures[key] = encode_treatment(cohort[col], label)
            for k, v in exposure_counts(exposures[key]).items():
                put(f"{name}_{key}_{k}", v)
            logger.info("%s %s: %s", name, key, exposure_counts(exposures[key]))

        design, cols = c["design"](cohort)
        for key in ("chemo", "endo"):
            for k, v in exposure_counts(exposures[key].loc[design.index]).items():
                put(f"{name}_A_{key}_{k}", v)
        rows_a = adjusted_models(name, design, cols, auc_z)
        adjusted.extend(rows_a)
        put(f"{name}_A_n_patients", len(design))
        put(f"{name}_A_n_events", int(design["os_event"].sum()))

        # Sensitivity of (B): chemotherapy is not randomized, and in SCAN-B untreated
        # patients are older, so the interaction is also fitted with the clinical
        # covariates of (A) and endocrine therapy as main effects
        adj_covs = cols["clinical"] + [c for c in cols["treatment"] if c != "chemo"]
        adj_required = cols["required"] + ["endo"]
        for drug in DRUGS:
            interaction.append(interaction_model(name, cohort, auc_z, drug, "chemo", exposures["chemo"]))
            interaction.append(interaction_model(name, design, auc_z, drug, "chemo", exposures["chemo"],
                                                 covariates=adj_covs, required=adj_required,
                                                 adjustment="clinical_endo"))
            specific = DRUG_SPECIFIC_EXPOSURE.get(drug)
            if specific is None:
                continue
            if specific in exposures:
                interaction.append(interaction_model(name, cohort, auc_z, drug, specific,
                                                     exposures[specific]))
            else:
                interaction.append({"cohort": name, "drug": drug, "exposure": specific,
                                    "adjustment": "pam50_strata",
                                    "group": "positive_control", "n_cohort": len(cohort),
                                    "n_exposure_known": 0, "status": "exposure_not_in_cohort"})

    adj = pd.DataFrame(adjusted)
    inter = pd.DataFrame(interaction)
    ok = inter["status"] == "ok"
    inter.loc[ok, "padj_interaction"] = inter[ok].groupby(["cohort", "exposure", "adjustment"]) \
        ["p_interaction"].transform(lambda p: stats.false_discovery_control(p))

    for name in cohorts:
        # (A): how far each block of covariates moves the log HR per SD
        wide = adj[adj["cohort"] == name].pivot(index="drug", columns="model", values="HR")
        p_tr = adj[(adj["cohort"] == name) & (adj["model"] == "clinical_treatment")].set_index("drug")
        log_hr = np.log(wide)
        same_dir = np.sign(log_hr["pam50_strata"]) == np.sign(log_hr["clinical_treatment"])
        by_clinical = (log_hr["clinical"] - log_hr["pam50_strata"]).abs()
        by_treatment = (log_hr["clinical_treatment"] - log_hr["clinical"]).abs()
        put(f"{name}_A_n_drugs_direction_kept_vs_pam50", int(same_dir.sum()))
        put(f"{name}_A_n_drugs_p005_treatment_adjusted", int((p_tr["p"] < 0.05).sum()))
        put(f"{name}_A_n_candidates_p005_treatment_adjusted",
            int((p_tr.loc[CANDIDATES, "p"] < 0.05).sum()))
        put(f"{name}_A_n_controls_p005_treatment_adjusted",
            int((p_tr.loc[CONTROLS, "p"] < 0.05).sum()))
        put(f"{name}_A_median_abs_change_logHR_pam50_to_clinical", round(float(by_clinical.median()), 4))
        put(f"{name}_A_median_abs_change_logHR_clinical_to_treatment",
            round(float(by_treatment.median()), 4))
        put(f"{name}_A_max_abs_change_logHR_clinical_to_treatment", round(float(by_treatment.max()), 4))
        put(f"{name}_A_HR_chemo_median_over_drugs", round(float(p_tr["HR_chemo"].median()), 3))
        put(f"{name}_A_HR_endo_median_over_drugs", round(float(p_tr["HR_endo"].median()), 3))
        # Script 37 fitting choices against the unpenalized clinical model, same patients
        ref = adj[adj["cohort"] == name].set_index(["model", "drug"])
        for model in ("clinical", REFERENCE_MODEL):
            put(f"{name}_A_n_drugs_p005_{model}", int((ref.loc[model, "p"] < 0.05).sum()))
            put(f"{name}_A_HR_age_per_year_{model}",
                round(float(ref.loc[model, "HR_age_per_year"].median()), 4))
        for drug in ("erlotinib", "paclitaxel", "epirubicin"):
            for model in MODELS + [REFERENCE_MODEL]:
                put(f"{name}_A_{drug}_{model}_HR_per_SD", round(float(wide.loc[drug, model]), 3))
                put(f"{name}_A_{drug}_{model}_p", float(ref.loc[(model, drug), "p"]))

        # (B): interaction with chemotherapy, as specified and with clinical adjustment
        for adjustment in ("pam50_strata", "clinical_endo"):
            ci = inter[(inter["cohort"] == name) & (inter["exposure"] == "chemo")
                       & (inter["adjustment"] == adjustment) & ok].set_index("drug")
            tag = f"{name}_B_chemo_{adjustment}"
            put(f"{tag}_n_patients", int(ci["n_patients"].iloc[0]))
            put(f"{tag}_n_events", int(ci["n_events"].iloc[0]))
            put(f"{tag}_n_interaction_p005", int((ci["p_interaction"] < 0.05).sum()))
            put(f"{tag}_n_interaction_padj005", int((ci["padj_interaction"] < 0.05).sum()))
            put(f"{tag}_n_candidates_interaction_p005",
                int((ci.loc[CANDIDATES, "p_interaction"] < 0.05).sum()))
            put(f"{tag}_n_controls_interaction_p005",
                int((ci.loc[CONTROLS, "p_interaction"] < 0.05).sum()))
            put(f"{tag}_n_drugs_same_direction_both_groups",
                int((np.sign(np.log(ci["HR_within_exposed"]))
                     == np.sign(np.log(ci["HR_within_unexposed"]))).sum()))
            for drug in ("erlotinib", "entinostat", "paclitaxel", "epirubicin", "talazoparib"):
                put(f"{tag}_{drug}_HR_interaction", round(float(ci.loc[drug, "HR_interaction"]), 3))
                put(f"{tag}_{drug}_p_interaction", float(ci.loc[drug, "p_interaction"]))
                put(f"{tag}_{drug}_HR_within_chemo", round(float(ci.loc[drug, "HR_within_exposed"]), 3))
                put(f"{tag}_{drug}_HR_within_no_chemo",
                    round(float(ci.loc[drug, "HR_within_unexposed"]), 3))
        n_specific = int(((inter["cohort"] == name) & (inter["exposure"] != "chemo") & ok).sum())
        put(f"{name}_B_n_drug_specific_tests_run", n_specific)

    adj.to_csv(OUT / "treatment_adjusted_cox.csv", index=False)
    inter.to_csv(OUT / "treatment_interaction_cox.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "treatment_summary.csv", index=False)

    show = ["cohort", "model", "drug", "n_patients", "HR", "CI_low", "CI_high", "p"]
    logger.info("\n%s", adj[adj["model"] == "clinical_treatment"][show].to_string(index=False))
    show = ["cohort", "drug", "exposure", "adjustment", "status", "n_patients", "HR_interaction",
            "p_interaction", "HR_within_exposed", "HR_within_unexposed"]
    logger.info("\n%s", inter.reindex(columns=show).to_string(index=False))
    logger.info("Wrote %s, %s and %s", OUT / "treatment_adjusted_cox.csv",
                OUT / "treatment_interaction_cox.csv", OUT / "treatment_summary.csv")


if __name__ == "__main__":
    main()
