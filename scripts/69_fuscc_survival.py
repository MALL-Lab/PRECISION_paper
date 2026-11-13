#!/usr/bin/env python3
"""Do the drug-survival associations of the paper hold in a cohort that is all TNBC?

The clinical screen of the paper runs in unselected breast cancer (SCAN-B 7,215
patients, METABRIC 1,979, TCGA-BRCA 1,060) while the disease the project targets
is triple-negative breast cancer. Inside SCAN-B the clinical TNBC subgroup is 676
patients with 183 events, and there only pelitinib and osimertinib survive. This
script asks whether the ordering of those associations appears again in FUSCC, a
cohort that is entirely TNBC, from another country, with another RNA-seq library
protocol and another endpoint. Both cohorts were aligned with HISAT and quantified
with StringTie on GRCh38. What changes is the library: total RNA with ribosomal
depletion in FUSCC against poly(A) selection in SCAN-B.

HONEST PROVENANCE. No model had been fitted on FUSCC when the design was written
(docs/plan_analisis_fuscc_tnbc.md). What was computed during the review of that
design is the SCAN-B side of the comparison: the reliabilities, the attenuation
ceiling, the power and the base rate of positive z. That is declared and this is
not a pre-registration.

POWER, and it is the headline
  The cohort has 49 relapses. A per-drug hazard ratio therefore needs to exceed
  about 1.49 per standard deviation to be detected at 80% power, while the
  reference effects, the two drugs that survive in the SCAN-B TNBC subgroup, are
  smaller. Individual drugs are consequently not the unit of this analysis.
  The reference effect of a drug is computed here, never typed in. Script 61
  reports its TNBC hazard ratio per standard deviation of the full SCAN-B Cox
  cohort (tnbc_subgroup_cox.csv), and the FUSCC fits standardize by the spread
  of the cohort itself, so the reference is rescaled to the spread inside the
  TNBC subgroup with the drug's own ratio of standard deviations:
      log HR_ref = log HR_61 * SD_tnbc / SD_full
  with both SDs of the predicted AUC (ddof 1) over the 676 TNBC patients and the
  7,215 patients of the Cox cohort. The unpenalized Cox is equivariant to a
  rescaling of its covariate, so the script checks the conversion against a
  direct refit of the TNBC subgroup standardized by its own SD.
  The ratio SD_tnbc / SD_full is each drug's own, which gives 1.465 for
  pelitinib and 1.24 for osimertinib. When the design was reviewed the ratio was
  the median over the 12 focal drugs (1.056), which gave 1.46 and 1.26.

PRIMARY, one statistic
  The Pearson correlation between the vector of per-drug Cox score z in FUSCC and
  the same vector in the SCAN-B clinical TNBC subgroup, over the 1,447 drugs of
  the screen. The score z is invariant to any rescaling of the predicted AUC, so
  the two cohorts are comparable without choosing whose standard deviation to
  standardize by, and at beta = 0 it has a closed form, which is what makes the
  permutation null affordable (src/analysis/score_test.py).
  Null: the (time, status) pairs of FUSCC are permuted across patients, 10,000
  times, seed 42, one-sided on the right, alpha 0.05. Permuting the outcome
  preserves the correlation between drugs, which a naive Pearson p-value would
  ignore: the predictions carry far fewer effective degrees of freedom than 1,445.
  Read with two numbers beside it: the attenuation ceiling sqrt(rel_ref x
  rel_fuscc), the value the correlation would reach if both cohorts estimated the
  same effects with their measured noise, and the disattenuated ratio.

SECONDARY, declared and single
  A Cox model of the composite of the seven candidates (mean of the within-drug z
  of their predicted AUC), hazard ratio per standard deviation with its interval,
  its Wald p and its permutation p, next to the percentile of that composite
  against 10,000 random sets of seven drugs matched on the spread of their
  predicted AUC. It is algebraically the equal-weight meta-analytic combination
  corrected for correlation, so only one of the two is reported.

EVERYTHING ELSE IS DESCRIPTIVE and says so in the column name: the per-drug
table with its three Benjamini-Hochberg corrections, the correlation restricted
to the 143 drugs of the statistical filter and to the 12 focal drugs, the
sensitivity arms and the sign agreement.

COHORT
  FUSCC-TNBC (Jiang et al., Cancer Cell 2019), 465 patients, 360 with RNA-seq,
  all of them recorded as female in Table S1 (n_female counts the analysed ones).
  Excluded before anything else and written out with their reason: the patients
  with a relapse-free time of zero, one microinvasive DCIS and one with an
  equivocal HER2 immunohistochemistry and no FISH result. Time is always in days,
  the months column never leaves script 68. The endpoint is relapse-free survival
  because it is the only one the cohort carries, while the reference is overall
  survival, and the two are as close as the data can show: over the SCAN-B TNBC
  patients the two z vectors correlate at 0.74 against a ceiling of 0.48 under
  independent noise.

COHORT GATE, before the candidates are read
  Positive nodes must carry the relapse risk they are known to carry in TNBC: the
  gate passes when their unpenalized hazard ratio falls inside the range the plan
  declared, 2.5 to 3.5 (GATE_NODES_HR_RANGE, written to the summary). Until
  30-oct-2026 the verdict was decided with 2.0 to 4.5 while the plan and the
  expected_range column said 2.5 to 3.5. The observed 3.11 passes with both.
  Tumor size and the immune subtype are reported without a verdict because they
  have too few events, and tumor stage is left out because 97% of the cohort is
  T1 or T2. The gate writes its verdict and never aborts. The patients whose
  node count is missing are counted as node-negative in the gate, the count is
  written out, and the gate is refitted on the complete cases as a sensitivity
  that does not decide.

PROPORTIONAL HAZARDS, for every Cox fit of the script
  The test of script 64 (SCAN-B), imported from it so both cohorts share one
  implementation: the exact score test of R survival 3.x
  cox.zph(transform="rank", terms=FALSE) for every term of every fit, with the
  km and event-rank transforms beside it, and the lifelines Grambsch-Therneau
  approximation on scaled Schoenfeld residuals kept for reference. Every fit
  here is unpenalized, so the ridge term of script 64 is zero. The test is
  descriptive and changes no estimate.

WHAT THIS CANNOT SAY
  No patient in FUSCC received any of the seven candidates and 94% received
  chemotherapy, so any association is prognostic within chemotherapy-treated
  disease and carries nothing about the benefit of a particular drug.

Outputs (results/v14_fuscc/), and the rows of every table are sorted so two runs
diff empty:
  fuscc_primary_correlation.csv   the primary statistic and its descriptive twins
  fuscc_cox_perdrug.csv           the 1,447 drugs in FUSCC and in the reference
  fuscc_composite.csv             the secondary
  fuscc_permutation_null.csv      the 10,000 replicates
  fuscc_sensitivity.csv           the declared arms
  fuscc_cohort_gate.csv           the gate
  fuscc_scanb_tnbc_reference_z.csv  the reference vector, which did not exist
  fuscc_excluded_patients.csv     every patient removed, with the reason. It
                                  carries patient identifiers and stays here
  fuscc_excluded_summary.csv      the same exclusions counted by reason, no identifiers
  fuscc_rnaseq_vs_not.csv         the 360 sequenced against the 105 that are not
  fuscc_power_mde.csv             minimum detectable effects and reference power
  fuscc_ph_tests.csv              proportional hazards test of every term of every fit
  fuscc_covariate_coding.csv      how every clinical level was decoded
  fuscc_qt_clipping.csv           how far FUSCC falls outside the training range
  fuscc_summary.csv               metric, value

Run from the project root (a few minutes, no training):
    python scripts/69_fuscc_survival.py
"""

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR, PAPER_RESULTS  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.analysis.score_test import martingale_residuals, score_p, score_test
from src.data.clinical_encoding import encode_binary_receptor
from src.data.drug_names import normalize_drug_columns, normalize_drug_name
from src.data.load_drug_response import load_prism_response
from src.data.scanb_cohort import scanb_patient_cohort
from src.data.splits import cell_line_holdout_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fuscc_survival")

DATA = DATA_DIR
PROCESSED = DATA / "precision_processed"
RESULTS = RESULTS_DIR
OUT_DIR = RESULTS / "v14_fuscc"
SCANB_PRED = RESULTS / "v5_final" / "scanb_perdrug_predictions.csv"
SURV_SCANB = RESULTS / "v8_multicohort" / "survival_SCANB.csv"
TNBC_SUBGROUP = RESULTS / "v13_reaudit" / "tnbc_subgroup_cox.csv"
UNIVERSE_PATH = RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv"
SUPP_S1 = PAPER_RESULTS / "supp_table_S1_candidate_selection.csv"
PLAN_PATH = Path("docs/plan_analisis_fuscc_tnbc.md")

CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
DRUGS = CANDIDATES + CONTROLS
FAMILY = {**{d: "candidate" for d in CANDIDATES}, **{d: "positive_control" for d in CONTROLS}}

SEED = 42
N_PERM = 10_000
N_MATCHED_SETS = 10_000
ALPHA = 0.05
RIDGE_ALPHA = 100
MIN_TRAIN_RESPONSES = 30
DAYS_PER_MONTH = 30.4375
# The QuantileTransformer of the transfer clips whatever falls outside the range it
# saw in PRISM training. The tolerable amount is not a number one invents: it is what
# the cohorts already in the paper show, so the run fails if FUSCC clips more than
# this multiple of SCAN-B, the cohort the reference comes from.
QT_CLIP_TOLERANCE_VS_SCANB = 1.5
SD_RATIO_RANGE = (0.7, 1.4)    # and outside this ratio of predicted spreads
# Hazard ratio of positive nodes the cohort gate requires, the range of the plan
GATE_NODES_HR_RANGE = (2.5, 3.5)
EXPECTED_SEX = "Female"

# Frozen values. Everything here was measured on the files this script reads and
# the run stops if any of them moves.
EXPECTED_FUSCC_SEQUENCED = 360
# Exclusions are defined by criterion and frozen as counts. Patient identifiers are
# never written in the code (they would reach git), nor in the error messages.
EXPECTED_N_ZERO_TIME = 2
EXPECTED_N_DCIS_MI = 1          # from the histology column
EXPECTED_N_HER2_EQUIVOCAL = 1   # IHC 2+ with no FISH result
EXPECTED_UNIVERSE = 1447
EXPECTED_SHARED_TFS = 769
EXPECTED_SCANB_TNBC = (676, 183)
EXPECTED_SCANB_COHORT = (7215, 1164)
EXPECTED_ERLOTINIB_HR = 4.28
EXPECTED_PELITINIB_HR_SD = 1.4315
EXPECTED_OSIMERTINIB_HR_SD = 1.2432
# The two drugs that survive in the SCAN-B TNBC subgroup give the reference effects of
# the power analysis, computed in reference_effects(). Pelitinib, the larger, also
# stands in as the upper bound of the composite.
REFERENCE_DRUGS = ("pelitinib", "osimertinib")
CONVERSION_RTOL = 1e-6   # rescaled script 61 HR against the direct refit of the subgroup


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The proportional hazards test of script 64, so SCAN-B and FUSCC share one implementation
PH = _load_script("s64_proportional_hazards", "64_proportional_hazards.py")
PH_ROWS = []   # one row per term of every Cox fit, filled by ph_test_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# transfer
# ----------------------------------------------------------------------------
def clipped_fraction(cohort_tf: pd.DataFrame) -> float:
    """Share of a cohort's TF activities outside the range the QT saw in training."""
    prism_tf = pd.read_csv(PROCESSED / "tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(normalize_drug_columns(load_prism_response()), 0.2, SEED)
    shared = sorted(set(prism_tf.columns) & set(cohort_tf.columns))
    train = prism_tf.loc[[c for c in train_cells if c in prism_tf.index], shared]
    sub = cohort_tf[shared]
    return float((sub < train.min()).to_numpy().mean() + (sub > train.max()).to_numpy().mean())


def transfer_models(cohort_tf: pd.DataFrame, label: str):
    """Per-drug Ridge of script 17, fitted on PRISM, applied to a clinical cohort.

    Returns the predictions, the shared TFs and the share of cohort activities
    that fall outside the range the QuantileTransformer saw in training, which is
    what would silently flatten a cohort whose scale differs from PRISM.
    """
    response = normalize_drug_columns(load_prism_response())
    prism_tf = pd.read_csv(PROCESSED / "tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(response, 0.2, SEED)

    shared = sorted(set(prism_tf.columns) & set(cohort_tf.columns))
    if len(shared) != EXPECTED_SHARED_TFS:
        raise ValueError(f"{label}: {len(shared)} TFs shared with PRISM, expected {EXPECTED_SHARED_TFS}")
    prism_tf = prism_tf[shared]
    cohort_aligned = cohort_tf[shared]

    train_tf = prism_tf.loc[[c for c in train_cells if c in prism_tf.index]]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_tf)
    train_min, train_max = train_tf.min(), train_tf.max()
    below = (cohort_aligned < train_min).mean()
    above = (cohort_aligned > train_max).mean()

    prism_qt = pd.DataFrame(qt.transform(prism_tf), index=prism_tf.index, columns=shared)
    cohort_qt = pd.DataFrame(qt.transform(cohort_aligned), index=cohort_aligned.index, columns=shared)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(prism_qt.loc[train_tf.index])
    x_cohort = scaler.transform(cohort_qt)
    train_resp = response.loc[train_tf.index]

    predictions, skipped = {}, []
    for drug in sorted(response.columns):
        y = train_resp[drug].to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() < MIN_TRAIN_RESPONSES:
            skipped.append(drug)
            continue
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(x_train[mask], y[mask])
        predictions[drug] = ridge.predict(x_cohort)
    pred = pd.DataFrame(predictions, index=cohort_qt.index)
    clip = pd.DataFrame({"tf": shared, "frac_below_train_min": below.to_numpy(),
                         "frac_above_train_max": above.to_numpy()})
    logger.info("%s: %d drugs predicted for %d samples (%d drugs without enough training data)",
                label, pred.shape[1], pred.shape[0], len(skipped))
    return pred, shared, clip


# ----------------------------------------------------------------------------
# cohorts
# ----------------------------------------------------------------------------
def fuscc_cohort():
    """The analysed FUSCC patients, with every exclusion written out."""
    clin = pd.read_csv(PROCESSED / "fuscc_clinical.csv")
    tf = pd.read_csv(PROCESSED / "tf_activities_fuscc.csv", index_col=0)
    if len(clin) != EXPECTED_FUSCC_SEQUENCED or len(tf) != EXPECTED_FUSCC_SEQUENCED:
        raise ValueError(f"FUSCC has {len(clin)} clinical rows and {len(tf)} TF rows, "
                         f"expected {EXPECTED_FUSCC_SEQUENCED}")
    if sorted(clin["sample_id"]) != sorted(tf.index):
        raise ValueError("The FUSCC clinical table and the TF activities cover different patients")
    if "rfs_time_months" in clin.columns or "followup_months" in clin.columns:
        raise ValueError("The months columns must not reach the analysis, days are the unit")
    if "sex" not in clin.columns or clin["sex"].isna().any():
        raise ValueError("The FUSCC clinical table has no sex for every patient, rerun 00c and 68")

    excluded = []
    zero = clin.loc[clin["rfs_time_days"] <= 0, "sample_id"].tolist()
    if len(zero) != EXPECTED_N_ZERO_TIME:
        raise ValueError(f"{len(zero)} zero-time patients, expected {EXPECTED_N_ZERO_TIME}")
    for s in sorted(zero):
        excluded.append({"patient": s, "reason": "relapse-free time of zero days"})
    dcis = clin.loc[clin["histology"].fillna("") == "DCIS-MI", "sample_id"].tolist()
    if len(dcis) != EXPECTED_N_DCIS_MI:
        raise ValueError(f"{len(dcis)} microinvasive DCIS patients, expected {EXPECTED_N_DCIS_MI}")
    for s in dcis:
        excluded.append({"patient": s, "reason": "microinvasive DCIS, not invasive disease"})
    equivocal = clin.loc[(clin["erbb2_ihc"] == 2) & clin["erbb2_fish"].isna(), "sample_id"].tolist()
    if len(equivocal) != EXPECTED_N_HER2_EQUIVOCAL:
        raise ValueError(f"{len(equivocal)} HER2-equivocal patients, expected {EXPECTED_N_HER2_EQUIVOCAL}")
    for s in equivocal:
        excluded.append({"patient": s, "reason": "HER2 immunohistochemistry 2+ with no FISH result"})

    drop = {row["patient"] for row in excluded}
    if len(drop) != EXPECTED_N_ZERO_TIME + EXPECTED_N_DCIS_MI + EXPECTED_N_HER2_EQUIVOCAL:
        raise ValueError("The exclusion criteria overlap, every excluded patient must have one reason")
    kept = clin[~clin["sample_id"].isin(drop)].sort_values("sample_id").reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded).merge(
        clin[["sample_id", "rfs_time_days", "rfs_event"]].rename(columns={"sample_id": "patient"}),
        on="patient", how="left").sort_values("patient")
    logger.info("FUSCC: %d patients analysed, %d events, %d excluded",
                len(kept), int(kept["rfs_event"].sum()), len(drop))
    return kept, tf.loc[kept["sample_id"]], excluded_df


def scanb_tnbc_cohort():
    """The clinical TNBC subgroup of SCAN-B, defined exactly as in script 61."""
    clin = scanb_patient_cohort()
    pred = pd.read_csv(SCANB_PRED, index_col=0)
    pred = normalize_drug_columns(pred)
    merged = clin.join(pred, how="inner").dropna(subset=["pam50"])
    if (len(merged), int(merged["os_event"].sum())) != EXPECTED_SCANB_COHORT:
        raise ValueError(f"The SCAN-B Cox cohort is {len(merged)} patients and "
                         f"{int(merged['os_event'].sum())} events, expected {EXPECTED_SCANB_COHORT}")
    status = pd.DataFrame({c: encode_binary_receptor(merged[c]) for c in ("er", "pr", "her2")})
    tnbc = merged[status.notna().all(axis=1) & (status == 0).all(axis=1)]
    if (len(tnbc), int(tnbc["os_event"].sum())) != EXPECTED_SCANB_TNBC:
        raise ValueError(f"The SCAN-B TNBC subgroup is {len(tnbc)} patients and "
                         f"{int(tnbc['os_event'].sum())} events, expected {EXPECTED_SCANB_TNBC}")
    logger.info("SCAN-B TNBC reference: %d patients, %d deaths, %d with relapse-free follow-up",
                len(tnbc), int(tnbc["os_event"].sum()), int(tnbc["rfs_event"].notna().sum()))
    return merged, tnbc


# ----------------------------------------------------------------------------
# checks that must pass before anything is written
# ----------------------------------------------------------------------------
def self_checks(fuscc_pred, scanb_all, tnbc, universe):
    """Reproduce published numbers, and the engine against an exact external test."""
    published = pd.read_csv(SURV_SCANB).set_index("drug")
    dummies = pd.get_dummies(scanb_all["pam50"], prefix="sub", drop_first=True)
    for drug in ("erlotinib", "paclitaxel"):
        data = pd.concat([scanb_all[["os_time", "os_event", drug]], dummies], axis=1)
        data = data.rename(columns={drug: "predicted_auc"})
        cph = CoxPHFitter(penalizer=0.1).fit(data, "os_time", "os_event")
        hr = float(cph.hazard_ratios_["predicted_auc"])
        if not np.isclose(hr, float(published.loc[drug, "HR"]), rtol=1e-6):
            raise ValueError(f"{drug}: canonical HR {hr:.6f} does not reproduce the published value")
    if round(float(published.loc["erlotinib", "HR"]), 2) != EXPECTED_ERLOTINIB_HR:
        raise ValueError("The published erlotinib HR is not 4.28")

    sub = pd.read_csv(TNBC_SUBGROUP)
    sub = sub[sub["subset"] == "TNBC_clinical"].set_index("drug")
    if (int(sub["n"].iloc[0]), int(sub["events"].iloc[0])) != EXPECTED_SCANB_TNBC:
        raise ValueError("Script 61 does not report the TNBC subgroup this script rebuilt")
    for drug, expected in (("pelitinib", EXPECTED_PELITINIB_HR_SD),
                           ("osimertinib", EXPECTED_OSIMERTINIB_HR_SD)):
        if not np.isclose(float(sub.at[drug, "hr_per_sd"]), expected, atol=5e-4):
            raise ValueError(f"{drug} HR per SD is {sub.at[drug, 'hr_per_sd']}, expected {expected}")

    # The score engine against an exact external reference: on a binary covariate
    # and untied times it equals the log-rank statistic of lifelines.
    rng = np.random.default_rng(SEED)
    t = rng.exponential(700, 300) + 1
    e = (rng.random(300) < 0.15).astype(float)
    x = (rng.random(300) < 0.5).astype(float)
    z = score_test(x[:, None], t, e)[2][0]
    lr = logrank_test(t[x == 1], t[x == 0], e[x == 1], e[x == 0])
    if not np.isclose(z ** 2, lr.test_statistic, rtol=1e-8):
        raise ValueError(f"The score engine gives {z ** 2:.10f} against a log-rank "
                         f"{lr.test_statistic:.10f}")

    if len(universe) != EXPECTED_UNIVERSE:
        raise ValueError(f"The universe holds {len(universe)} drugs, expected {EXPECTED_UNIVERSE}")
    missing = sorted(set(universe) - set(fuscc_pred.columns))
    if missing:
        raise ValueError(f"{len(missing)} universe drugs have no FUSCC prediction: {missing[:5]}")
    missing_ref = sorted(set(universe) - set(tnbc.columns))
    if missing_ref:
        raise ValueError(f"{len(missing_ref)} universe drugs have no SCAN-B prediction: {missing_ref[:5]}")
    a = np.random.default_rng(SEED).permutation(10)
    b = np.random.default_rng(SEED).permutation(10)
    if not np.array_equal(a, b):
        raise ValueError("The random generator is not deterministic")
    logger.info("Self-checks passed: canonical Cox, TNBC subgroup, score engine, universe, seed")


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def reliability(z):
    """Share of the variance of a z vector that is effect and not estimation noise.

    A z vector of pure noise has variance one, so the share is (var - 1) / var. When
    the variance sits at or below one the share is not estimable rather than zero,
    and the functions below refuse to build a ceiling out of it.
    """
    v = float(np.var(np.asarray(z, dtype=float), ddof=1))
    return ((v - 1.0) / v if v > 1.0 else np.nan), v


def attenuation_ceiling(rel_a, rel_b):
    """Highest correlation two noisy estimates of the same effects could reach."""
    if not np.isfinite(rel_a) or not np.isfinite(rel_b):
        return np.nan
    return float(np.sqrt(rel_a * rel_b))


def disattenuated(r, ceiling):
    """The correlation as a share of that ceiling, when the ceiling exists."""
    return float(r / ceiling) if np.isfinite(ceiling) and ceiling > 0 else np.nan


def effective_df(pred: pd.DataFrame) -> float:
    """Effective number of independent drugs, from the spectrum of their correlation."""
    corr = np.corrcoef(pred.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0, None)
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def ph_test_rows(model, label, cph, data, strata=None):
    """Proportional hazards test of every term of one unpenalized fit, as script 64 runs it.

    `data` is the frame the model was fitted on, with duration t and event e. The
    exact test (zph_exact of script 64, R survival 3.x cox.zph with the rank of time
    pooled across strata) is the primary one, and zph_exact raises unless its
    information matrix reproduces the variance of the lifelines fit. The km and
    event-rank transforms and the lifelines approximation sit beside it.
    """
    exact = PH.zph_exact(cph, data.rename(columns={"t": "os_time", "e": "os_event"}), 0.0,
                         strata=strata, transforms=PH.TRANSFORMS)
    approx = PH.ph_pvalues(cph, data)
    approx_r_rank = PH.ph_pvalues(cph, data, PH.r_rank)
    for term in cph.params_.index:
        PH_ROWS.append({
            "model": model, "label": label, "term": term, "strata": strata or "",
            "n": int(len(data)), "events": int(data["e"].sum()),
            "coef": float(cph.params_[term]), "hr": float(np.exp(cph.params_[term])),
            "ph_chisq_exact": float(exact.loc[term, "chisq_rank"]),
            "ph_p_exact": float(exact.loc[term, "p_rank"]),
            "ph_violation_p05_exact": bool(exact.loc[term, "p_rank"] < ALPHA),
            "ph_p_exact_km": float(exact.loc[term, "p_km"]),
            "ph_p_exact_event_rank": float(exact.loc[term, "p_event_rank"]),
            "ph_test_statistic_lifelines": float(approx.loc[term, "test_statistic"]),
            "ph_p_lifelines": float(approx.loc[term, "p"]),
            "ph_violation_p05_lifelines": bool(approx.loc[term, "p"] < ALPHA),
            "ph_p_approx_r_rank": float(approx_r_rank.loc[term, "p"])})


def cox_per_sd(time, event, x, strata=None, covariates=None, ph=None):
    """Unpenalized Cox of the standardized covariate, the specification of script 61.

    With ph = (model, label) the proportional hazards test of every term of the fit
    is added to PH_ROWS."""
    data = pd.DataFrame({"t": np.asarray(time, float), "e": np.asarray(event, float),
                         "z": (np.asarray(x, float) - np.mean(x)) / np.std(x, ddof=1)})
    if covariates is not None:
        data = pd.concat([data, covariates.reset_index(drop=True)], axis=1)
    if strata is not None:
        data["stratum"] = np.asarray(strata)
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(data, duration_col="t", event_col="e",
            strata=["stratum"] if strata is not None else None)
    if ph is not None:
        ph_test_rows(*ph, cph, data, strata="stratum" if strata is not None else None)
    row = cph.summary.loc["z"]
    return {"hr_per_sd": float(row["exp(coef)"]), "ci_low": float(row["exp(coef) lower 95%"]),
            "ci_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"]),
            "n": int(len(data)), "events": int(data["e"].sum())}


def composite_score(pred: pd.DataFrame, drugs) -> np.ndarray:
    """Mean over `drugs` of the within-drug z of the predicted AUC."""
    sub = pred[drugs].to_numpy(dtype=float)
    z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=1)
    return z.mean(axis=1)


def mde_hr_per_sd(events, alpha=ALPHA, power=0.80):
    """Smallest hazard ratio per standard deviation detectable with this many events."""
    return float(np.exp((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(events)))


def power_at(hr, events, alpha=ALPHA):
    """Power against a hazard ratio per standard deviation fixed in advance."""
    ncp = abs(np.log(hr)) * np.sqrt(events)
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha / 2) - ncp))


def reference_effects(scanb_all, tnbc) -> pd.DataFrame:
    """Reference effects of the power analysis, from their source (see POWER above).

    Script 61 standardizes the predicted AUC by the SD of the full SCAN-B Cox cohort,
    so its TNBC hazard ratio per SD is rescaled to the SD inside the subgroup,
    log HR_ref = log HR_61 * SD_tnbc / SD_full, with the drug's own two SDs. The
    result must equal a direct unpenalized refit of the subgroup standardized by its
    own SD, which is the same model under a rescaled covariate."""
    sub = pd.read_csv(TNBC_SUBGROUP)
    sub = sub[sub["subset"] == "TNBC_clinical"].set_index("drug")
    rows = []
    for drug in REFERENCE_DRUGS:
        hr_61 = float(sub.at[drug, "hr_per_sd"])
        sd_full = float(scanb_all[drug].std(ddof=1))
        sd_tnbc = float(tnbc[drug].std(ddof=1))
        hr_ref = float(np.exp(np.log(hr_61) * sd_tnbc / sd_full))
        refit = cox_per_sd(tnbc["os_time"], tnbc["os_event"], tnbc[drug])["hr_per_sd"]
        rel = abs(hr_ref - refit) / refit
        if not rel <= CONVERSION_RTOL:
            raise ValueError(f"{drug}: rescaled reference HR {hr_ref:.8f} against the refit "
                             f"{refit:.8f} (relative difference {rel:.2e})")
        rows.append({"drug": drug, "hr_per_sd_full_cohort_sd": hr_61, "sd_full": sd_full,
                     "sd_tnbc": sd_tnbc, "sd_ratio_tnbc_vs_full": sd_tnbc / sd_full,
                     "hr_per_sd_tnbc_sd": hr_ref, "hr_per_sd_tnbc_sd_refit": refit})
    out = pd.DataFrame(rows).set_index("drug")
    logger.info("Reference effects per SD inside the SCAN-B TNBC subgroup: %s",
                {d: round(float(out.at[d, "hr_per_sd_tnbc_sd"]), 4) for d in out.index})
    return out


# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = sorted(normalize_drug_name(d) for d in pd.read_csv(UNIVERSE_PATH)["drug"])
    clin, tf, excluded = fuscc_cohort()
    pred, shared_tfs, clip = transfer_models(tf, "FUSCC")
    pred = pred[universe]

    # ---- the transfer must not be silently flattened by a scale mismatch
    clipped = float((clip["frac_below_train_min"] + clip["frac_above_train_max"]).mean())
    scanb_clip = clipped_fraction(pd.read_csv(PROCESSED / "tf_activities_scanb.csv", index_col=0))
    scanb_all, tnbc = scanb_tnbc_cohort()
    sd_ratio = (pred.std(ddof=1) / tnbc[universe].std(ddof=1)).median()
    if clipped > QT_CLIP_TOLERANCE_VS_SCANB * scanb_clip:
        raise ValueError(f"{100 * clipped:.2f}% of the FUSCC TF activities fall outside the "
                         f"training range, more than {QT_CLIP_TOLERANCE_VS_SCANB} times the "
                         f"{100 * scanb_clip:.2f}% of SCAN-B")
    if not SD_RATIO_RANGE[0] <= sd_ratio <= SD_RATIO_RANGE[1]:
        raise ValueError(f"The median ratio of predicted spreads is {sd_ratio:.3f}, outside "
                         f"the declared {SD_RATIO_RANGE}")
    logger.info("Transfer: %.2f%% of activities clipped against %.2f%% in SCAN-B, "
                "median spread ratio %.3f", 100 * clipped, 100 * scanb_clip, sd_ratio)

    self_checks(pred, scanb_all, tnbc, universe)

    # ---- score statistics: FUSCC, and the reference that did not exist
    t_f = clin["rfs_time_days"].to_numpy(float)
    e_f = clin["rfs_event"].to_numpy(float)
    x_f = pred.loc[clin["sample_id"], universe].to_numpy(float)
    u_f, i_f, z_f = score_test(x_f, t_f, e_f)

    ref_rows = {}
    for endpoint, tcol, ecol in (("os", "os_time", "os_event"), ("rfs", "rfs_time", "rfs_event")):
        sub = tnbc.dropna(subset=[tcol, ecol])
        sub = sub[sub[tcol] > 0]
        x_r = sub[universe].to_numpy(float)
        _, _, z_r = score_test(x_r, sub[tcol].to_numpy(float), sub[ecol].to_numpy(float))
        ref_rows[endpoint] = {"z": z_r, "n": len(sub), "events": int(sub[ecol].sum())}
        logger.info("Reference %s: %d patients, %d events, %.1f%% of drugs with z > 0",
                    endpoint.upper(), len(sub), ref_rows[endpoint]["events"],
                    100 * float((z_r > 0).mean()))
    reference = pd.DataFrame({
        "drug": universe,
        "score_z_os": ref_rows["os"]["z"], "score_p_os": score_p(ref_rows["os"]["z"]),
        "score_z_rfs": ref_rows["rfs"]["z"], "score_p_rfs": score_p(ref_rows["rfs"]["z"]),
        "sd_pred_auc_tnbc": tnbc[universe].std(ddof=1).to_numpy(),
        "sd_pred_auc_full": scanb_all[universe].std(ddof=1).to_numpy(),
    }).sort_values("drug")

    # ---- primary correlation and its permutation null
    z_ref = ref_rows["os"]["z"]
    r_primary = float(np.corrcoef(z_f, z_ref)[0, 1])
    resid_perm = np.empty((N_PERM, len(universe)), dtype=np.float32)
    for b in range(N_PERM):
        perm = rng.permutation(len(t_f))
        resid_perm[b] = score_test(x_f[perm], t_f, e_f)[2]
    null_r = np.array([float(np.corrcoef(resid_perm[b], z_ref)[0, 1]) for b in range(N_PERM)])
    p_primary = (1 + int((null_r >= r_primary).sum())) / (1 + N_PERM)
    rel_ref, var_ref = reliability(z_ref)
    rel_fuscc, var_fuscc = reliability(z_f)
    ceiling = attenuation_ceiling(rel_ref, rel_fuscc)
    eff_df = effective_df(pred)
    logger.info("Primary: r = %.3f, permutation p = %.4f, effective df %.1f", r_primary,
                p_primary, eff_df)

    idx = {d: i for i, d in enumerate(universe)}
    s1_drugs = sorted(set(normalize_drug_name(d) for d in pd.read_csv(SUPP_S1)["drug"]) & set(universe))
    subsets = {"all1447": universe, "filter143_descriptive": s1_drugs,
               "focal12_descriptive": sorted(DRUGS)}
    corr_rows = []
    for name, drugs in subsets.items():
        cols = [idx[d] for d in drugs]
        r_obs = float(np.corrcoef(z_f[cols], z_ref[cols])[0, 1])
        nulls = np.array([float(np.corrcoef(resid_perm[b, cols], z_ref[cols])[0, 1])
                          for b in range(N_PERM)])
        p_obs = (1 + int((nulls >= r_obs).sum())) / (1 + N_PERM)
        rel_r, _ = reliability(z_ref[cols])
        rel_f, _ = reliability(z_f[cols])
        ceil_sub = attenuation_ceiling(rel_r, rel_f)
        corr_rows.append({
            "drug_set": name, "reference_endpoint": "OS", "arm": "primary",
            "n_drugs": len(drugs), "r_pearson": r_obs,
            "r_spearman_descriptive": float(stats.spearmanr(z_f[cols], z_ref[cols])[0]),
            "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
            "null_p95": float(np.percentile(nulls, 95)), "p_perm": p_obs,
            "mc_se": float(np.sqrt(p_obs * (1 - p_obs) / N_PERM)), "n_perm": N_PERM,
            "naive_pearson_p_descriptive": float(stats.pearsonr(z_f[cols], z_ref[cols])[1]),
            "effective_df": eff_df if name == "all1447" else np.nan,
            "rel_reference": rel_r, "rel_fuscc": rel_f, "ceiling_attenuation": ceil_sub,
            "r_disattenuated": disattenuated(r_obs, ceil_sub),
            "is_primary": name == "all1447"})
    # the same correlation against the relapse-free reference, a declared arm
    z_ref_rfs = ref_rows["rfs"]["z"]
    r_rfs = float(np.corrcoef(z_f, z_ref_rfs)[0, 1])
    nulls_rfs = np.array([float(np.corrcoef(resid_perm[b], z_ref_rfs)[0, 1]) for b in range(N_PERM)])
    p_rfs = (1 + int((nulls_rfs >= r_rfs).sum())) / (1 + N_PERM)
    rel_rfs, _ = reliability(z_ref_rfs)
    corr_rows.append({
        "drug_set": "all1447", "reference_endpoint": "RFS", "arm": "S-REF_rfs",
        "n_drugs": len(universe), "r_pearson": r_rfs,
        "r_spearman_descriptive": float(stats.spearmanr(z_f, z_ref_rfs)[0]),
        "null_mean": float(nulls_rfs.mean()), "null_sd": float(nulls_rfs.std(ddof=1)),
        "null_p95": float(np.percentile(nulls_rfs, 95)), "p_perm": p_rfs,
        "mc_se": float(np.sqrt(p_rfs * (1 - p_rfs) / N_PERM)), "n_perm": N_PERM,
        "naive_pearson_p_descriptive": float(stats.pearsonr(z_f, z_ref_rfs)[1]),
        "effective_df": eff_df, "rel_reference": rel_rfs, "rel_fuscc": rel_fuscc,
        "ceiling_attenuation": attenuation_ceiling(rel_rfs, rel_fuscc),
        "r_disattenuated": disattenuated(r_rfs, attenuation_ceiling(rel_rfs, rel_fuscc)),
        "is_primary": False})
    correlation = pd.DataFrame(corr_rows)

    # ---- per-drug table: FUSCC, the reference, and the descriptive corrections
    hr_rows = []
    for drug in sorted(DRUGS):
        hr_rows.append({"drug": drug, **cox_per_sd(t_f, e_f, pred[drug].to_numpy(float),
                                                    ph=("focal12", drug))})
    hr_focal = pd.DataFrame(hr_rows).set_index("drug")
    ref_idx = reference.set_index("drug")
    perdrug = pd.DataFrame({
        "drug": universe,
        "family": [FAMILY.get(d, "other") for d in universe],
        "in_focal12": [d in DRUGS for d in universe],
        "in_filter143": [d in set(s1_drugs) for d in universe],
        "n": len(clin), "events": int(e_f.sum()),
        "score_z": z_f, "score_p": score_p(z_f),
        "sd_pred_auc_fuscc": pred[universe].std(ddof=1).to_numpy(),
        "sd_pred_auc_scanb_tnbc": ref_idx.loc[universe, "sd_pred_auc_tnbc"].to_numpy(),
        "scanb_tnbc_score_z_os": ref_idx.loc[universe, "score_z_os"].to_numpy(),
        "scanb_tnbc_score_z_rfs": ref_idx.loc[universe, "score_z_rfs"].to_numpy(),
    })
    perdrug["sign_agrees_descriptive"] = (np.sign(perdrug["score_z"]) ==
                                          np.sign(perdrug["scanb_tnbc_score_z_os"]))
    for col, subset in (("padj_bh1447_descriptive", universe),
                        ("padj_bh143_descriptive", s1_drugs),
                        ("padj_bh12_descriptive", sorted(DRUGS))):
        perdrug[col] = np.nan
        mask = perdrug["drug"].isin(subset)
        perdrug.loc[mask, col] = stats.false_discovery_control(perdrug.loc[mask, "score_p"].to_numpy())
    for col in ("hr_per_sd_fuscc", "ci_low", "ci_high", "wald_p"):
        perdrug[col] = np.nan
    for drug in DRUGS:
        m = perdrug["drug"] == drug
        perdrug.loc[m, ["hr_per_sd_fuscc", "ci_low", "ci_high", "wald_p"]] = [
            hr_focal.at[drug, "hr_per_sd"], hr_focal.at[drug, "ci_low"],
            hr_focal.at[drug, "ci_high"], hr_focal.at[drug, "p"]]
    perdrug = perdrug.sort_values("drug").reset_index(drop=True)

    # ---- secondary: the composite of the seven candidates
    comp_rows = []
    for name, drug_set in (("candidates7", CANDIDATES), ("controls5", CONTROLS)):
        score = composite_score(pred, sorted(drug_set))
        fit = cox_per_sd(t_f, e_f, score, ph=("composite", name))
        _, _, z_comp = score_test(score[:, None], t_f, e_f)
        null_comp = np.array([score_test(score[rng.permutation(len(t_f))][:, None], t_f, e_f)[2][0]
                              for _ in range(N_PERM)])
        p_perm = (1 + int((np.abs(null_comp) >= abs(z_comp[0])).sum())) / (1 + N_PERM)
        # a background of random sets of the same size, matched on the spread of
        # the predicted AUC, so the composite is read against its own kind
        sd_all = pred[universe].std(ddof=1)
        target = sd_all[sorted(drug_set)].median()
        pool = sorted(sd_all.index[(sd_all > target * 0.8) & (sd_all < target * 1.25)])
        draws = np.empty(N_MATCHED_SETS)
        for b in range(N_MATCHED_SETS):
            pick = [pool[i] for i in rng.choice(len(pool), size=len(drug_set), replace=False)]
            draws[b] = score_test(composite_score(pred, pick)[:, None], t_f, e_f)[2][0]
        comp_rows.append({
            "set": name, "n_drugs": len(drug_set), **fit, "score_z": float(z_comp[0]),
            "p_perm": p_perm, "mc_se": float(np.sqrt(p_perm * (1 - p_perm) / N_PERM)),
            "pct_background_matched_descriptive": round(100 * float((draws <= z_comp[0]).mean()), 1),
            "n_background_pool": len(pool),
            "mde_hr_per_sd_80": mde_hr_per_sd(int(e_f.sum()))})
    composite = pd.DataFrame(comp_rows)

    # ---- cohort gate, before any candidate is read
    gate_rows = []
    # A missing node count is counted as node-negative in the gate. The count is written
    # out, and the complete cases are refitted as a sensitivity that does not decide.
    nodes_missing = clin["ln_positive"].isna().to_numpy()
    n_nodes_missing = int(nodes_missing.sum())
    node_pos = (clin["ln_positive"].fillna(0) > 0).astype(float).to_numpy()
    for term, keep, decides in (("positive lymph nodes", np.ones(len(clin), dtype=bool), True),
                                ("positive lymph nodes, complete cases", ~nodes_missing, False)):
        x_k, e_k = node_pos[keep], e_f[keep]
        gate_data = pd.DataFrame({"t": t_f[keep], "e": e_k, "x": x_k})
        # a binary term is reported as its own hazard ratio, not per standard deviation
        cph = CoxPHFitter().fit(gate_data, "t", "e")
        ph_test_rows("gate", term, cph, gate_data)
        row = cph.summary.loc["x"]
        gate_rows.append({
            "term": term, "hr": float(row["exp(coef)"]),
            "ci_low": float(row["exp(coef) lower 95%"]),
            "ci_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"]),
            "n_events_level": f"{int(e_k[x_k == 1].sum())} of {int((x_k == 1).sum())} "
                              f"against {int(e_k[x_k == 0].sum())} of {int((x_k == 0).sum())}",
            "expected_range": f"{GATE_NODES_HR_RANGE[0]} to {GATE_NODES_HR_RANGE[1]}",
            "decides": decides,
            "verdict": (bool(GATE_NODES_HR_RANGE[0] <= float(row["exp(coef)"]) <= GATE_NODES_HR_RANGE[1])
                        if decides else np.nan),
            "note": (f"the established prognostic signal of the cohort, the {n_nodes_missing} "
                     f"patients with no node count counted as node-negative" if decides else
                     f"sensitivity, the {n_nodes_missing} patients with no node count removed")})
    size = clin["size_cm"].astype(float)
    ok = size.notna().to_numpy()
    gate_data = pd.DataFrame({"t": t_f[ok], "e": e_f[ok], "x": size[ok].to_numpy()})
    cph = CoxPHFitter().fit(gate_data, "t", "e")
    ph_test_rows("gate", "tumour size (cm)", cph, gate_data)
    row = cph.summary.loc["x"]
    gate_rows.append({"term": "tumour size (cm)", "hr": float(row["exp(coef)"]),
                      "ci_low": float(row["exp(coef) lower 95%"]),
                      "ci_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"]),
                      "n_events_level": f"{int(e_f[ok].sum())} events in {int(ok.sum())} patients",
                      "expected_range": "1.25 to 1.45 per cm", "decides": False,
                      "verdict": np.nan, "note": "descriptive"})
    im = (clin["subtype_mrna"] == "IM").astype(float)
    gate_data = pd.DataFrame({"t": t_f, "e": e_f, "x": im.to_numpy()})
    cph = CoxPHFitter().fit(gate_data, "t", "e")
    ph_test_rows("gate", "immune subtype (IM) against the rest", cph, gate_data)
    row = cph.summary.loc["x"]
    gate_rows.append({"term": "immune subtype (IM) against the rest", "hr": float(row["exp(coef)"]),
                      "ci_low": float(row["exp(coef) lower 95%"]),
                      "ci_high": float(row["exp(coef) upper 95%"]), "p": float(row["p"]),
                      "n_events_level": f"{int(e_f[im == 1].sum())} of {int((im == 1).sum())}",
                      "expected_range": "below 1", "decides": False, "verdict": np.nan,
                      "note": "descriptive, too few events to decide"})
    gate = pd.DataFrame(gate_rows)
    gate_cc = gate.set_index("term").loc["positive lymph nodes, complete cases"]
    logger.info("Cohort gate: positive nodes HR %.2f [%.2f, %.2f], verdict %s, %d patients with no "
                "node count counted as negative, complete cases HR %.2f [%.2f, %.2f]",
                gate.at[0, "hr"], gate.at[0, "ci_low"], gate.at[0, "ci_high"], gate.at[0, "verdict"],
                n_nodes_missing, gate_cc["hr"], gate_cc["ci_low"], gate_cc["ci_high"])

    # ---- sensitivity arms, all of them declared before running
    sens_rows = []

    def add_arm(arm, note, mask=None, strata=None, covariates=None):
        keep = np.ones(len(clin), dtype=bool) if mask is None else np.asarray(mask)
        for drug in sorted(DRUGS) + ["composite_candidates7"]:
            x = (composite_score(pred, sorted(CANDIDATES)) if drug.startswith("composite")
                 else pred[drug].to_numpy(float))
            cov = None if covariates is None else covariates.loc[keep].reset_index(drop=True)
            fit = cox_per_sd(t_f[keep], e_f[keep], x[keep],
                             strata=None if strata is None else np.asarray(strata)[keep],
                             covariates=cov, ph=(arm, drug))
            sens_rows.append({"arm": arm, "drug": drug, "family": FAMILY.get(drug, "composite"),
                              **fit, "note": note})

    add_arm("S1_strata_mrna", "stratified by the mRNA subtype of the cohort",
            strata=clin["subtype_mrna"].to_numpy())
    add_arm("S1b_strata_intrinsic", "stratified by the intrinsic subtype",
            strata=clin["subtype_pam50"].fillna("Unknown").to_numpy())
    clinical_cov = pd.DataFrame({
        "age": clin["age"].astype(float),
        "size_cm": clin["size_cm"].astype(float),
        "ln_positive": clin["ln_positive"].astype(float)})
    complete = clinical_cov.notna().all(axis=1).to_numpy()
    add_arm("S2_clinical", "adjusted for age, tumour size and positive nodes, complete cases",
            mask=complete, covariates=clinical_cov)
    no_lar = (clin["subtype_mrna"] != "LAR").to_numpy()
    add_arm("S3_no_LAR", "the androgen-receptor subtype removed", mask=no_lar)
    add_arm("S4_surgery_year", "adjusted for the year of surgery",
            covariates=pd.DataFrame({"surgery_year": clin["surgery_year"].astype(float)}))

    # the correlation without LAR, with its own null, because the cohort changes
    x_nl = x_f[no_lar]
    _, _, z_nl = score_test(x_nl, t_f[no_lar], e_f[no_lar])
    r_nl = float(np.corrcoef(z_nl, z_ref)[0, 1])
    nulls_nl = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = rng.permutation(int(no_lar.sum()))
        nulls_nl[b] = float(np.corrcoef(score_test(x_nl[perm], t_f[no_lar], e_f[no_lar])[2], z_ref)[0, 1])
    p_nl = (1 + int((nulls_nl >= r_nl).sum())) / (1 + N_PERM)
    corr_rows.append({
        "drug_set": "all1447", "reference_endpoint": "OS", "arm": "S3_no_LAR",
        "n_drugs": len(universe), "r_pearson": r_nl,
        "r_spearman_descriptive": float(stats.spearmanr(z_nl, z_ref)[0]),
        "null_mean": float(nulls_nl.mean()), "null_sd": float(nulls_nl.std(ddof=1)),
        "null_p95": float(np.percentile(nulls_nl, 95)), "p_perm": p_nl,
        "mc_se": float(np.sqrt(p_nl * (1 - p_nl) / N_PERM)), "n_perm": N_PERM,
        "naive_pearson_p_descriptive": float(stats.pearsonr(z_nl, z_ref)[1]),
        "effective_df": np.nan, "rel_reference": rel_ref,
        "rel_fuscc": reliability(z_nl)[0],
        "ceiling_attenuation": attenuation_ceiling(rel_ref, reliability(z_nl)[0]),
        "r_disattenuated": disattenuated(r_nl, attenuation_ceiling(rel_ref, reliability(z_nl)[0])),
        "is_primary": False})
    correlation = pd.DataFrame(corr_rows)
    sensitivity = pd.DataFrame(sens_rows).sort_values(["arm", "drug"]).reset_index(drop=True)

    # ---- the sequenced patients against the rest of the cohort
    raw = pd.read_csv(DATA / "FUSCC" / "fuscc_clinical.csv")
    raw["sequenced"] = raw["RNA_Seqencing"].fillna("NO").str.upper() == "YES"
    comp_rows = []
    seq, other = raw[raw["sequenced"]], raw[~raw["sequenced"]]
    seq_data = pd.DataFrame({"t": raw["RFS_time_Days"].clip(lower=1), "e": raw["RFS_Status"],
                             "x": raw["sequenced"].astype(float)})
    cph = CoxPHFitter().fit(seq_data, "t", "e")
    ph_test_rows("rnaseq_vs_not_descriptive", "sequenced against not sequenced", cph, seq_data)
    row = cph.summary.loc["x"]
    comp_rows.append({"term": "relapse risk of the sequenced patients", "value_sequenced": float(row["exp(coef)"]),
                      "value_other": 1.0, "p": float(row["p"]),
                      "note": f"hazard ratio [{row['exp(coef) lower 95%']:.2f}, "
                              f"{row['exp(coef) upper 95%']:.2f}], the two groups are compared as they are"})
    for label, col in (("year of surgery", "Date_of_surgery"), ("age", "Age_at_surgery"),
                       ("tumour size (cm)", "Size_cm"), ("positive nodes", "LN_positive"),
                       ("relapse events", "RFS_Status")):
        a = (pd.to_datetime(seq[col]).dt.year if col == "Date_of_surgery"
             else pd.to_numeric(seq[col], errors="coerce"))
        b = (pd.to_datetime(other[col]).dt.year if col == "Date_of_surgery"
             else pd.to_numeric(other[col], errors="coerce"))
        p = float(stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided").pvalue)
        comp_rows.append({"term": label, "value_sequenced": float(a.median()),
                          "value_other": float(b.median()), "p": p, "note": "median, Mann-Whitney"})
    comp_rows.append({"term": "mRNA subtype annotated", "value_sequenced": float(seq["mRNA_Subtype"].notna().mean()),
                      "value_other": float(other["mRNA_Subtype"].notna().mean()), "p": np.nan,
                      "note": "the subtype is only defined where there is RNA-seq"})
    rnaseq_vs_not = pd.DataFrame(comp_rows)

    # ---- power, against the reference effects of the SCAN-B TNBC subgroup
    events = int(e_f.sum())
    ref_eff = reference_effects(scanb_all, tnbc)
    if ref_eff["hr_per_sd_tnbc_sd"].idxmax() != "pelitinib":
        raise ValueError("Pelitinib is no longer the largest reference effect, the composite "
                         "upper bound of the power table has to be revisited")
    hr_upper = float(ref_eff.at["pelitinib", "hr_per_sd_tnbc_sd"])
    power_rows = []
    for statistic, drug, source in (
            ("per drug, hazard ratio per standard deviation", "pelitinib", "pelitinib_rescaled"),
            ("per drug, hazard ratio per standard deviation", "osimertinib", "osimertinib_rescaled"),
            ("composite of the seven candidates", "pelitinib",
             "pelitinib rescaled, as the composite upper bound")):
        hr = float(ref_eff.at[drug, "hr_per_sd_tnbc_sd"])
        power_rows.append({"statistic": statistic, "n_events": events,
                           "mde_hr_per_sd_80": mde_hr_per_sd(events),
                           "reference_effect": hr, "reference_source": source,
                           "reference_hr_per_sd_full_cohort_sd": float(ref_eff.at[drug, "hr_per_sd_full_cohort_sd"]),
                           "reference_sd_ratio_tnbc_vs_full": float(ref_eff.at[drug, "sd_ratio_tnbc_vs_full"]),
                           "power_1.0x": power_at(hr, events),
                           "power_0.7x": power_at(np.exp(0.7 * np.log(hr)), events),
                           "power_0.5x": power_at(np.exp(0.5 * np.log(hr)), events)})
    power_rows.append({"statistic": "primary correlation", "n_events": events,
                       "mde_hr_per_sd_80": np.nan,
                       "reference_effect": ceiling, "reference_source": "attenuation ceiling",
                       "power_1.0x": float((null_r < ceiling).mean()),
                       "power_0.7x": float((null_r < 0.7 * ceiling).mean()),
                       "power_0.5x": float((null_r < 0.5 * ceiling).mean())})
    power = pd.DataFrame(power_rows)

    # ---- how every clinical level was decoded, so nothing is coerced in silence
    coding_rows = []
    for variable in ("chemo", "paclitaxel", "radiotherapy", "subtype_mrna", "subtype_pam50",
                     "n_stage", "t_stage", "grade", "brca12_mut"):
        counts = clin[variable].astype(object).where(clin[variable].notna(), "missing").value_counts()
        for raw_value, n in counts.items():
            if variable == "paclitaxel":
                coded = {True: "exposed", False: "not exposed"}.get(raw_value, "regimen unknown")
            elif variable in ("chemo", "radiotherapy"):
                coded = {True: "yes", False: "no"}.get(raw_value, str(raw_value))
            else:
                coded = str(raw_value)
            coding_rows.append({"variable": variable, "raw_value": str(raw_value),
                                "coded_value": coded, "n": int(n)})
    coding = pd.DataFrame(coding_rows).sort_values(["variable", "raw_value"]).reset_index(drop=True)
    allowed = {"exposed", "not exposed", "regimen unknown"}
    bad = coding[(coding["variable"] == "paclitaxel") & (~coding["coded_value"].isin(allowed))]
    if len(bad):
        raise ValueError(f"Unexpected treatment levels: {bad.to_dict(orient='records')}")

    # ---- median follow-up by reverse Kaplan-Meier, the definition frozen here
    order = np.argsort(t_f)
    t_sorted, censored_sorted = t_f[order], (1 - e_f)[order]
    surv, median_fu = 1.0, np.nan
    for i, (ti, ci) in enumerate(zip(t_sorted, censored_sorted)):
        if ci == 1:
            surv *= 1 - 1 / (len(t_f) - i)
            if surv <= 0.5 and np.isnan(median_fu):
                median_fu = ti
    median_fu_months = float(median_fu / DAYS_PER_MONTH)

    summary = {
        "plan_sha256": _sha256(PLAN_PATH)[:16],
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__, "pandas_version": pd.__version__,
        "seed": SEED, "n_perm": N_PERM, "n_matched_sets": N_MATCHED_SETS,
        "n_sequenced": EXPECTED_FUSCC_SEQUENCED, "n_patients": len(clin), "n_events": events,
        "n_female": int((clin["sex"] == EXPECTED_SEX).sum()),
        "n_excluded_total": len(excluded),
        "n_excluded_zero_time": int(excluded["reason"].str.contains("zero days").sum()),
        "n_excluded_predeclared": int((~excluded["reason"].str.contains("zero days")).sum()),
        "median_followup_months_reverse_km": round(median_fu_months, 1),
        "median_observed_months": round(float(np.median(t_f) / DAYS_PER_MONTH), 1),
        "max_observed_months": round(float(t_f.max() / DAYS_PER_MONTH), 1),
        "n_tfs_shared_prism": len(shared_tfs),
        "shared_tfs_sha256": _sha256_text(";".join(shared_tfs))[:16],
        "qt_clipped_fraction_pct": round(100 * clipped, 2),
        "qt_clipped_fraction_pct_scanb": round(100 * scanb_clip, 2),
        "sd_ratio_fuscc_over_scanb_tnbc": round(float(sd_ratio), 3),
        "scanb_tnbc_n": ref_rows["os"]["n"], "scanb_tnbc_events": ref_rows["os"]["events"],
        "scanb_tnbc_rfs_n": ref_rows["rfs"]["n"], "scanb_tnbc_rfs_events": ref_rows["rfs"]["events"],
        "scanb_tnbc_pct_z_positive_1447": round(100 * float((z_ref > 0).mean()), 1),
        "fuscc_pct_z_positive_1447": round(100 * float((z_f > 0).mean()), 1),
        "r_primary": round(r_primary, 4), "r_primary_p_perm": p_primary,
        "r_primary_mc_se": round(float(np.sqrt(p_primary * (1 - p_primary) / N_PERM)), 5),
        "r_primary_null_mean": round(float(null_r.mean()), 4),
        "r_primary_null_sd": round(float(null_r.std(ddof=1)), 4),
        "r_primary_null_p95": round(float(np.percentile(null_r, 95)), 4),
        "r_primary_naive_p": float(stats.pearsonr(z_f, z_ref)[1]),
        "effective_df": round(eff_df, 1),
        "rel_reference_os": round(rel_ref, 3) if np.isfinite(rel_ref) else "not estimable",
        "rel_fuscc": round(rel_fuscc, 3) if np.isfinite(rel_fuscc) else "not estimable",
        "var_z_reference_os": round(var_ref, 3), "var_z_fuscc": round(var_fuscc, 3),
        "ceiling_attenuation": round(ceiling, 3) if np.isfinite(ceiling) else "not estimable",
        "r_primary_disattenuated": (round(disattenuated(r_primary, ceiling), 3)
                                    if np.isfinite(ceiling) else "not estimable"),
        "r_rfs_reference": round(r_rfs, 4), "r_rfs_p_perm": p_rfs,
        "r_no_lar": round(r_nl, 4), "r_no_lar_p_perm": p_nl,
        "sign_agreement_pct_1447": round(100 * float(perdrug["sign_agrees_descriptive"].mean()), 1),
        "sign_agreement_pct_focal12": round(
            100 * float(perdrug.loc[perdrug["in_focal12"], "sign_agrees_descriptive"].mean()), 1),
        "n_drugs_fdr05_background": int((perdrug["padj_bh1447_descriptive"] < ALPHA).sum()),
        "mde_hr_per_sd_80": round(mde_hr_per_sd(events), 4),
        "reference_hr_pelitinib_rescaled": round(float(ref_eff.at["pelitinib", "hr_per_sd_tnbc_sd"]), 4),
        "reference_hr_osimertinib_rescaled": round(float(ref_eff.at["osimertinib", "hr_per_sd_tnbc_sd"]), 4),
        "power_at_pelitinib_rescaled": round(power_at(ref_eff.at["pelitinib", "hr_per_sd_tnbc_sd"], events), 2),
        "power_at_osimertinib_rescaled": round(power_at(ref_eff.at["osimertinib", "hr_per_sd_tnbc_sd"], events), 2),
        "gate_nodes_hr": round(float(gate.at[0, "hr"]), 2),
        "gate_nodes_verdict": bool(gate.at[0, "verdict"]),
        "gate_nodes_range_low": GATE_NODES_HR_RANGE[0],
        "gate_nodes_range_high": GATE_NODES_HR_RANGE[1],
        "gate_nodes_n_missing_counted_negative": n_nodes_missing,
        "gate_nodes_hr_complete_cases": round(float(gate_cc["hr"]), 2),
        "gate_nodes_ci_low_complete_cases": round(float(gate_cc["ci_low"]), 2),
        "gate_nodes_ci_high_complete_cases": round(float(gate_cc["ci_high"]), 2),
        "composite_hr_per_sd": round(float(composite.set_index("set").at["candidates7", "hr_per_sd"]), 3),
        "composite_p_perm": float(composite.set_index("set").at["candidates7", "p_perm"]),
        "composite_pct_background": float(
            composite.set_index("set").at["candidates7", "pct_background_matched_descriptive"]),
        "n_filter143_available": len(s1_drugs),
    }
    perdrug_idx = perdrug.set_index("drug")
    for drug in sorted(DRUGS):
        row = perdrug_idx.loc[drug]
        summary[f"{drug}_hr_per_sd"] = round(float(row["hr_per_sd_fuscc"]), 3)
        summary[f"{drug}_ci_low"] = round(float(row["ci_low"]), 3)
        summary[f"{drug}_ci_high"] = round(float(row["ci_high"]), 3)
        summary[f"{drug}_score_z"] = round(float(row["score_z"]), 3)
        summary[f"{drug}_padj_descriptive"] = float(row["padj_bh12_descriptive"])
        summary[f"{drug}_reference_z_os"] = round(float(row["scanb_tnbc_score_z_os"]), 3)

    # ---- proportional hazards, every term of every fit (exact test primary)
    ph = pd.DataFrame(PH_ROWS).sort_values(["model", "label", "term"]).reset_index(drop=True)
    summary["ph_n_fits"] = int(ph[["model", "label"]].drop_duplicates().shape[0])
    summary["ph_n_terms"] = len(ph)
    summary["ph_zph_exact_n_models"] = PH.ZPH_DIAG["n_models"]
    summary["ph_zph_exact_max_variance_rel_diff_vs_lifelines"] = PH.ZPH_DIAG["var_rel_diff"]
    summary["ph_zph_exact_max_newton_step_per_sd"] = PH.ZPH_DIAG["newton_step_per_sd"]
    for test in ("exact", "lifelines"):
        viol = ph[ph[f"ph_violation_p05_{test}"]]
        summary[f"ph_n_terms_violating_p05_{test}"] = len(viol)
        summary[f"ph_terms_violating_p05_{test}"] = ";".join(
            f"{m}|{lab}|{t}" for m, lab, t in zip(viol["model"], viol["label"], viol["term"]))
    for model, g in ph.groupby("model"):
        summary[f"ph_{model}_n_terms"] = len(g)
        summary[f"ph_{model}_n_terms_violating_p05_exact"] = int(g["ph_violation_p05_exact"].sum())
        summary[f"ph_{model}_n_terms_violating_p05_lifelines"] = int(g["ph_violation_p05_lifelines"].sum())
        summary[f"ph_{model}_p_min_exact"] = float(g["ph_p_exact"].min())
    main_term = ph.set_index(["model", "label", "term"])["ph_p_exact"]
    for drug in sorted(DRUGS):
        summary[f"{drug}_ph_p_exact"] = float(main_term[("focal12", drug, "z")])
    for name in ("candidates7", "controls5"):
        summary[f"composite_{name}_ph_p_exact"] = float(main_term[("composite", name, "z")])
    summary["gate_nodes_ph_p_exact"] = float(main_term[("gate", "positive lymph nodes", "x")])
    summary["gate_nodes_complete_cases_ph_p_exact"] = float(
        main_term[("gate", "positive lymph nodes, complete cases", "x")])
    logger.info("Proportional hazards: %d terms in %d fits, %d violate at p < 0.05 with the exact "
                "test (%s) and %d with the lifelines approximation", len(ph), summary["ph_n_fits"],
                summary["ph_n_terms_violating_p05_exact"], summary["ph_terms_violating_p05_exact"] or "none",
                summary["ph_n_terms_violating_p05_lifelines"])

    # ---- the exclusions counted by reason, the only form that leaves this folder
    excluded_summary = (excluded.groupby("reason").size().rename("n_patients")
                        .reset_index().sort_values("reason").reset_index(drop=True))

    # ---- write everything, with a fixed row order
    correlation.sort_values(["drug_set", "arm", "reference_endpoint"]).to_csv(
        OUT_DIR / "fuscc_primary_correlation.csv", index=False)
    perdrug.to_csv(OUT_DIR / "fuscc_cox_perdrug.csv", index=False)
    composite.to_csv(OUT_DIR / "fuscc_composite.csv", index=False)
    pd.DataFrame({"replicate": np.arange(N_PERM), "r_all1447": null_r,
                  "r_all1447_no_lar": nulls_nl, "r_rfs_reference": nulls_rfs}).to_csv(
        OUT_DIR / "fuscc_permutation_null.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "fuscc_sensitivity.csv", index=False)
    gate.to_csv(OUT_DIR / "fuscc_cohort_gate.csv", index=False)
    reference.to_csv(OUT_DIR / "fuscc_scanb_tnbc_reference_z.csv", index=False)
    excluded.to_csv(OUT_DIR / "fuscc_excluded_patients.csv", index=False)
    excluded_summary.to_csv(OUT_DIR / "fuscc_excluded_summary.csv", index=False)
    rnaseq_vs_not.to_csv(OUT_DIR / "fuscc_rnaseq_vs_not.csv", index=False)
    power.to_csv(OUT_DIR / "fuscc_power_mde.csv", index=False)
    ph.to_csv(OUT_DIR / "fuscc_ph_tests.csv", index=False)
    coding.to_csv(OUT_DIR / "fuscc_covariate_coding.csv", index=False)
    clip.sort_values("tf").to_csv(OUT_DIR / "fuscc_qt_clipping.csv", index=False)
    pd.DataFrame({"metric": list(summary), "value": list(summary.values())}).to_csv(
        OUT_DIR / "fuscc_summary.csv", index=False)

    logger.info("Primary: r = %.3f (permutation p = %.4f, null p95 %.3f), var(z) reference %.3f "
                "and FUSCC %.3f, ceiling %s", r_primary, p_primary,
                float(np.percentile(null_r, 95)), var_ref, var_fuscc,
                f"{ceiling:.3f}" if np.isfinite(ceiling) else "not estimable")
    logger.info("Composite of the candidates: HR per SD %.3f [%.3f, %.3f], permutation p %.3f",
                composite.at[0, "hr_per_sd"], composite.at[0, "ci_low"],
                composite.at[0, "ci_high"], composite.at[0, "p_perm"])
    logger.info("Sign agreement with the reference: %.1f%% over 1,447 drugs, %.1f%% over the twelve",
                summary["sign_agreement_pct_1447"], summary["sign_agreement_pct_focal12"])
    logger.info("Minimum detectable HR per SD %.2f with %d events, power %.2f against the largest "
                "reference effect (%.3f)", mde_hr_per_sd(events), events, power_at(hr_upper, events),
                hr_upper)
    logger.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
