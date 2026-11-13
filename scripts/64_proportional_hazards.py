#!/usr/bin/env python3
"""Proportional hazards checks, PAM50-stratified Cox and hazard ratios per SD (SCAN-B, METABRIC).

Re-audit of the canonical per-drug SCAN-B survival model of scripts 21 and 31
(lifelines CoxPHFitter with penalizer 0.1 on overall survival, raw predicted AUC
plus PAM50 dummies, Basal as reference) on the PAM50-adjusted Cox cohort
(7,215 patients, 1,164 events).

METABRIC (added 30-oct-2026, because Methods states the proportional hazards check
without exempting any cohort). Parts (1), (2) and (4) are repeated on the canonical
METABRIC model of script 21 (same specification, PAM50 from genefu, 1,979 patients
and 1,143 deaths), with the predictions rebuilt by metabric_predictions() of script
63, which reproduces survival_METABRIC.csv for every drug. The canonical hazard
ratio and p of the twelve drugs must reproduce survival_METABRIC.csv (relative
difference at most 1e-6). Part (3), the per-SD reading of the selection window, is
about the SCAN-B screen and is not repeated. The METABRIC outputs go to their own
files, so the SCAN-B outputs are the same as before the extension.

(1) For the seven Table 5 candidates and the five positive controls the canonical
    model is refitted, its hazard ratio must reproduce
    results/v8_multicohort/survival_SCANB.csv (relative difference at most 1e-6,
    the script raises otherwise), and the proportional hazards test of each term
    (drug and PAM50 dummies) is run in two versions, see "Proportional hazards
    tests" below. As a sensitivity both tests are also run on the unpenalized
    model with the same covariates.
(2) PAM50-stratified sensitivity for the same drugs. A penalized model
    (penalizer 0.1, strata pam50, raw predicted AUC) gives the HR per unit of
    predicted AUC, its 95% CI, the Wald p and the proportional hazards p of the
    drug term. An unpenalized stratified model with the predicted AUC
    standardized by its cohort SD gives the HR per SD, its 95% CI, p and the
    proportional hazards p. The unpenalized model with PAM50 dummies and
    standardized AUC is added to compare the canonical fit with a fit that has
    neither the penalty nor stratification.
(3) HR per SD for every drug of survival_SCANB.csv without refitting:
    HR_per_SD = exp(log(HR_per_unit) * SD), where SD is the sample SD (ddof 1) of
    the drug's predicted AUC in the 7,215-patient cohort. The 95% CI per SD is
    derived from the Wald test, since the Wald p gives |z| and hence the standard
    error of log HR.
    The per-unit HR window [1.5, 10] of the candidate rule is also read on the
    per-SD scale for the FDR-significant drugs with HR > 1.
(4) Counts of drug terms and PAM50 terms that violate proportional hazards at
    p < 0.05, under both tests.

Proportional hazards tests. The task asks for lifelines
proportional_hazard_test(cph, df, time_transform="rank"), and its p-values are
reported as specified (columns ph_p, ph_p_unpenalized, *_lifelines_rank). That
function implements the approximation that R survival used before version 3.0
(2019): the scaled Schoenfeld residuals take the variance of the residuals as
I / d, where I is the information matrix and d the number of events, and the
statistic uses the fitted variance of the coefficient. On this cohort the
approximation is far from the exact test (erlotinib in the unpenalized model
with PAM50 dummies, approximation p about 0.18, exact p about 7e-4), so every
conclusion of this script rests on the exact test, implemented in zph_exact as
in R survival 3.x cox.zph(fit, transform="rank", terms=FALSE). For each
covariate k it is the score test of adding x_k * g(t) to the fitted model, with
g(t) the rank of the follow-up time, evaluated at the fitted coefficients, and
it uses the full information matrix of (beta, gamma) accumulated over the Efron
risk sets. For a penalized fit the ridge second derivative is added to the beta
block, which makes it the score test of an unpenalized x_k * g(t) added to the
penalized model. R cox.zph also adds the ridge second derivative to the gamma
block, and that makes the statistic depend on the scale of g(t). With the rank
transform (g of the order of thousands) the difference is negligible (the
largest |log10 p| difference between the two conventions is reported in
ph_summary.csv), and the only other gap with the R statistic is the recycling of
the ridge over the beta block described below. With km (g between 0 and 1)
the R convention swamps the gamma information and pushes every p of a penalized
model towards 1, so the script leaves the gamma block unpenalized.
Transforms of time:
  - lifelines "rank" is np.cumsum(events) in the row order of the fit. Without
    strata that is the rank of the event times (ties broken by order). With
    strata the order is stratum first and time second, so it is not a rank of
    time at all, and its values for stratified models are kept only for
    reference (columns *_lifelines_rank).
  - rank, the primary transform of the exact test, is the R cox.zph "rank":
    rank(times) over every follow-up time of the fit, censored times included,
    ties averaged, pooled across strata, then centred on the mean over the
    events. The callable r_rank passes the same transform to the lifelines
    approximation (columns *_approx_r_rank, the pre-3.0 R cox.zph), so the
    comparison with ph_p_exact isolates the variance approximation at a fixed
    transform.
  - event_rank, the rank among the event times only, is the transform of
    lifelines "rank" without strata (up to ties), so ph_p_exact_event_rank
    against ph_p isolates the variance approximation at the transform the task
    specifies.
  - km, 1 - Kaplan-Meier, is the default transform of R cox.zph.
The script checks its own Efron information matrix against lifelines: the
inverse of (information + ridge second derivative) must equal
cph.variance_matrix_ of every refit (relative difference at most 1e-6). This
also confirms, through public attributes, how lifelines applies the penalizer.
The implementation was also compared with R survival 3.8.3 on the same data,
outside this script (coxph with ridge(..., theta = 0.1 * n, scale = TRUE) for
the penalized fits, cox.zph(fit, transform = "rank" or "km", terms = FALSE)),
and tests/test_zph_exact.py pins it to R on synthetic data for the four kinds of
model fitted here. For the unpenalized and the stratified models the chi-square
statistics agree: the largest |log10 p| difference at 1 df is 3e-5 for the
unpenalized model and 1e-8 for the stratified models in SCAN-B, and 1.1e-5 and
4e-6 in METABRIC. The canonical model, a ridge on five coefficients, differs
more. Its chi-square differs from R by up to 0.13% in SCAN-B (|log10 p| 9e-4)
and 0.30% in METABRIC (drug term of epirubicin, 15.340 here and 15.294 in R,
|log10 p| 1.0e-2). It is not convergence: the coefficients agree with R to
about 1e-6, and the Newton step left by lifelines (1.5e-5 per SD in SCAN-B,
1.7e-6 in METABRIC) moves nothing at that scale. The difference is in how
cox.zph adds the ridge to the information matrix. pmat[1:nvar, 1:nvar] <-
fit$coxlist2$second assigns the vector of diagonal second derivatives to the
nvar x nvar beta block, R recycles it column by column, and row i of the block
gets second[i] in every column, not only on the diagonal. Repeating that
recycling outside the script reproduces the R chi-square to a relative 6e-7 in
SCAN-B and 3e-6 in METABRIC. The diagonal is the second derivative of the ridge
penalty, the matrix whose inverse is cph.variance_matrix_, so the script keeps
it. With a single penalized coefficient, as in the stratified penalized model,
the recycling changes nothing. No verdict changes: in both cohorts no term of
the canonical model (12 drug terms, 48 PAM50 term tests) falls on the other
side of p = 0.05 under the two conventions. R prints a different p for
penalized fits because it refers the statistic to the effective df of the
penalized term: 0.48 to 0.61 in SCAN-B and 0.80 to 0.85 in METABRIC for the
ridge on predicted AUC of the stratified model, and with terms=FALSE on the
five-coefficient ridge of the canonical model the df of the whole term (2.8 in
SCAN-B, 3.9 in METABRIC) go to the drug row while the PAM50 rows get NA. The
score test of an unpenalized x_k * g(t) has 1 df, which is what this script
uses.

How lifelines applies the penalizer (checked in lifelines 0.30.3,
CoxPHFitter._fit_model and _newton_raphson_for_efron_model): before fitting,
every covariate is centred and divided by its sample SD over the fit rows
(pandas std, ddof 1). The ridge penalty 0.5 * penalizer * n * sum(beta**2), with
n the number of rows, acts on the coefficients of these standardized covariates,
and params_ is that coefficient divided by the SD. On the raw scale the second
derivative of the penalty is penalizer * n * SD**2, which is what zph_exact adds
and what the variance check confirms. The penalized fit is therefore invariant
to rescaling a covariate, and exp(log(HR_per_unit) * SD) is exactly the per-SD
hazard ratio that the penalized fit estimated, identical to refitting with the
predicted AUC divided by SD. The conversion is exact for the penalized fit. The
script checks it by refitting the twelve drugs with standardized AUC. A
consequence is that the penalty acts on every drug on the per-SD scale, whatever
the spread of its predictions, so a drug whose predictions vary little across
patients gets a large HR per unit for a modest HR per SD. The same penalty also
shrinks the PAM50 dummies, which the unpenalized refit makes visible.

Inputs:
  results/v5_final/scanb_perdrug_predictions.csv     (script 17, index sample_id)
  data/precision_processed/scanb_clinical.csv        (script 09, via src/data/scanb_cohort.py)
  results/v8_multicohort/survival_SCANB.csv          (script 21, reference HRs)
  results/v8_multicohort/penalizer_sweep_perdrug.csv (script 45, optional)
  results/v8_multicohort/survival_METABRIC.csv       (script 21, reference HRs of METABRIC)
  data/precision_processed/metabric_clinical.csv, tf_activities_metabric.csv,
  tf_activities_all_prism.csv and the PRISM response (METABRIC, through script 63)
Outputs in results/v13_reaudit/:
  ph_schoenfeld.csv          one row per drug and term of the canonical model
  cox_pam50_stratified.csv   one row per drug, stratified and per-SD fits
  hr_per_sd_all_drugs.csv    one row per drug of survival_SCANB.csv
  ph_summary.csv             metric, value
  ph_schoenfeld_metabric.csv, cox_pam50_stratified_metabric.csv and
  ph_summary_metabric.csv    the same three tables for METABRIC

Run from project root:
    python scripts/64_proportional_hazards.py
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats

from src.data.scanb_cohort import scanb_patient_cohort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("proportional_hazards")

PRED_PATH = RESULTS_DIR / "v5_final/scanb_perdrug_predictions.csv"
SURV_PATH = RESULTS_DIR / "v8_multicohort/survival_SCANB.csv"
SURV_METABRIC_PATH = RESULTS_DIR / "v8_multicohort/survival_METABRIC.csv"
SWEEP_PATH = RESULTS_DIR / "v8_multicohort/penalizer_sweep_perdrug.csv"
OUT = RESULTS_DIR / "v13_reaudit"

CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
PENALIZER = 0.1          # canonical penalizer of scripts 21 and 31
SWEEP_PENALIZER = 0.01   # penalizer at which the largest HR of script 45 is looked up
ALPHA = 0.05
REL_TOL = 1e-6
N_SURVIVAL, EVENTS_SURVIVAL = 7397, 1188   # survival cohort before the PAM50 filter
N_COX, EVENTS_COX = 7215, 1164             # PAM50-adjusted Cox cohort
N_COX_METABRIC, EVENTS_COX_METABRIC = 1979, 1143   # PAM50-adjusted METABRIC Cox cohort
Z_CRIT = stats.norm.ppf(1 - ALPHA / 2)
DRUG = "predicted_auc"
TRANSFORMS = ("rank", "km", "event_rank")   # time transforms of the exact PH test, rank is primary

# Over every model passed to zph_exact: largest relative difference between the
# inverse of its information matrix and lifelines variance_matrix_, largest Newton
# step left at the fitted coefficients (per-SD scale), and per transform the
# largest |log10 p| change if the ridge term is also added to the gamma block (R)
ZPH_DIAG = {"var_rel_diff": 0.0, "newton_step_per_sd": 0.0, "n_models": 0}


def check_close(label, got, expected, rtol=REL_TOL):
    """Raise unless got matches expected within the relative tolerance."""
    rel = abs(got - expected) / max(abs(expected), 1e-300)
    if not rel <= rtol:
        raise ValueError(f"{label}: got {got!r}, expected {expected!r} (relative difference {rel:.2e})")
    return rel


def load_cohort():
    """Survival cohort joined with the predictions, then the PAM50 filter of script 21."""
    pred = pd.read_csv(PRED_PATH, index_col=0)
    clin = scanb_patient_cohort()
    merged = clin[["os_time", "os_event", "pam50"]].join(pred, how="inner")
    n, e = len(merged), int(merged["os_event"].sum())
    if (n, e) != (N_SURVIVAL, EVENTS_SURVIVAL):
        raise ValueError(f"Survival cohort has {n} patients and {e} events, expected "
                         f"{N_SURVIVAL} and {EVENTS_SURVIVAL}")
    merged = merged.dropna(subset=["pam50"])
    n, e = len(merged), int(merged["os_event"].sum())
    if (n, e) != (N_COX, EVENTS_COX):
        raise ValueError(f"PAM50-adjusted Cox cohort has {n} patients and {e} events, expected "
                         f"{N_COX} and {EVENTS_COX}")
    logger.info("Cox cohort reproduced: %d patients, %d events, %d drugs", n, e, pred.shape[1])
    # Same dummies as script 21 (default bool dtype, Basal dropped as reference)
    pam = pd.get_dummies(merged["pam50"], prefix="sub", drop_first=True)
    return merged, pam


def load_metabric_cohort():
    """PAM50-adjusted METABRIC Cox cohort of script 21, with its per-drug predictions.

    The predictions and the survival table are those of metabric_predictions() and
    metabric_design() of script 63, which rebuild run_cohort_validation of script 21
    and reproduce survival_METABRIC.csv for every drug, so the two re-audit scripts
    share one implementation. Loaded here and not at import, because scripts 69 and
    71 import this module for its test only.
    """
    spec = importlib.util.spec_from_file_location(
        "s63_signed_meta_and_dss", Path(__file__).resolve().parent / "63_signed_meta_and_dss.py")
    s63 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s63)
    pred = s63.metabric_predictions()
    design, _, _ = s63.metabric_design(pred)
    merged = design[["os_time", "os_event", "PAM50"]].rename(columns={"PAM50": "pam50"}).join(pred)
    n, e = len(merged), int(merged["os_event"].sum())
    if (n, e) != (N_COX_METABRIC, EVENTS_COX_METABRIC):
        raise ValueError(f"METABRIC PAM50-adjusted Cox cohort has {n} patients and {e} events, expected "
                         f"{N_COX_METABRIC} and {EVENTS_COX_METABRIC}")
    logger.info("METABRIC Cox cohort reproduced: %d patients, %d events, %d drugs", n, e, pred.shape[1])
    pam = pd.get_dummies(merged["pam50"], prefix="sub", drop_first=True)
    return merged, pam


def fit(df, penalizer, strata=None):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="os_time", event_col="os_event", strata=strata)
    return cph


def r_rank(durations, events, weights):
    """R cox.zph "rank" transform as a lifelines time_transform callable.

    Rank of every follow-up time of the fit (censored included, ties averaged,
    pooled across strata). lifelines then keeps the event rows and centres them
    on their mean, which is the gtime of R cox.zph.
    """
    return pd.Series(stats.rankdata(np.asarray(durations, dtype=float)), index=durations.index)


def ph_pvalues(cph, df, time_transform="rank"):
    """lifelines Grambsch-Therneau approximation per term as a DataFrame (test_statistic, p)."""
    res = proportional_hazard_test(cph, df, time_transform=time_transform).summary
    return res[["test_statistic", "p"]]


def time_transform(time, event, kind):
    """Transform g(t) of the follow-up time, centred on the mean over the events as in R cox.zph.

    rank        R cox.zph "rank": rank of every follow-up time, censored included, ties averaged
    event_rank  rank among the event times only (ties averaged), the transform of lifelines "rank"
                in a fit without strata, up to ties
    km          R cox.zph default: 1 - left-continuous Kaplan-Meier estimate of the pooled cohort
    """
    if kind == "rank":
        g = stats.rankdata(time)
    elif kind == "event_rank":
        g = np.full(len(time), np.nan)
        g[event] = stats.rankdata(time[event])
    elif kind == "km":
        ut = np.unique(time)
        n_risk = len(time) - np.searchsorted(np.sort(time), ut, side="left")
        deaths = np.bincount(np.searchsorted(ut, time[event]), minlength=len(ut))
        surv = np.cumprod(1.0 - deaths / n_risk)
        g = 1.0 - np.concatenate([[1.0], surv])[np.searchsorted(ut, time, side="left")]
    else:
        raise ValueError(f"Unknown time transform {kind}")
    return g - g[event].mean()


def efron_event_terms(time, event, X, eta, strata):
    """Score and information of every event time of the Cox partial likelihood (Efron ties).

    For each distinct event time t within a stratum, with the d deaths D at t and
    the risk set of all rows of the stratum with time >= t, the Efron risk sets
    l = 0..d-1 give the deaths weight 1 - l/d. The function returns, per event
    time, the position of one of its deaths in the input, the score contribution
    sum_D x - sum_l xbar_l and the information contribution sum_l V_l, with
    xbar_l and V_l the weighted mean and covariance of x over the l-th Efron risk
    set. Weights are exp(eta).
    """
    p = X.shape[1]
    reps, scores, infos = [], [], []
    for st in pd.unique(strata):
        rows = np.flatnonzero(strata == st)
        order = np.argsort(-time[rows], kind="mergesort")
        rows = rows[order]
        t, e, x = time[rows], event[rows], X[rows]
        w = np.exp(eta[rows] - eta[rows].max())
        wx = w[:, None] * x
        wxx = wx[:, :, None] * x[:, None, :]
        c0, c1, c2 = np.cumsum(w), np.cumsum(wx, axis=0), np.cumsum(wxx, axis=0)
        neg_t = -t  # ascending, for searchsorted
        for dt in np.unique(t[e]):
            start = np.searchsorted(neg_t, -dt, side="left")
            stop = np.searchsorted(neg_t, -dt, side="right")   # rows [0, stop) are at risk
            dead = start + np.flatnonzero(e[start:stop])
            d = len(dead)
            d0, d1, d2 = w[dead].sum(), wx[dead].sum(axis=0), wxx[dead].sum(axis=0)
            score = x[dead].sum(axis=0)
            info = np.zeros((p, p))
            for l in range(d):
                f = l / d
                s0 = c0[stop - 1] - f * d0
                mean = (c1[stop - 1] - f * d1) / s0
                score = score - mean
                info += (c2[stop - 1] - f * d2) / s0 - np.outer(mean, mean)
            reps.append(rows[dead[0]])
            scores.append(score)
            infos.append(info)
    return np.array(reps), np.array(scores), np.array(infos)


def zph_exact(cph, df, penalizer, strata=None, transforms=("rank",)):
    """Exact-variance proportional hazards test per covariate (R survival 3.x cox.zph).

    Same algorithm as cox.zph(fit, transform=kind, terms=FALSE). g(t) is given by
    time_transform over all rows of the fit, and for each covariate k the
    statistic is u' A^-1 u with u = (0, U_gamma_k), where U_gamma_k is the score
    of x_k * g(t) at the fitted coefficients and A is the information matrix of
    (beta, gamma_k). For a penalized fit the ridge second derivative on the raw
    scale, penalizer * n * var(x), is added to the beta block, so the test is the
    score test of an unpenalized x_k * g(t) added to the penalized model. R also
    adds fit$coxlist2$second to the gamma block. That variant is computed too and
    its largest |log10 p| difference per transform goes to ZPH_DIAG. The other
    difference with cox.zph, the recycling of that vector over the beta block
    when the ridge has several coefficients, is not computed (see the module
    docstring). Returns a DataFrame indexed by covariate with the columns
    chisq_<kind> and p_<kind>.
    """
    covs = list(cph.params_.index)
    raw = df[covs].astype(float)
    X = raw.to_numpy() - raw.to_numpy().mean(axis=0)
    beta = cph.params_.to_numpy()
    time = df["os_time"].to_numpy(dtype=float)
    event = df["os_event"].to_numpy().astype(bool)
    grp = df[strata].to_numpy() if strata else np.zeros(len(df))
    rep, score, info = efron_event_terms(time, event, X, X @ beta, grp)
    sd = raw.std(ddof=1).to_numpy()
    pen = np.diag(penalizer * len(df) * sd ** 2)
    i_bb = info.sum(axis=0) + pen

    # The information matrix must reproduce the variance matrix of the lifelines fit
    var_ref = cph.variance_matrix_.loc[covs, covs].to_numpy()
    var_rel = float(np.max(np.abs(np.linalg.inv(i_bb) - var_ref) / np.abs(var_ref)))
    if not var_rel <= REL_TOL:
        raise ValueError(f"zph_exact information does not match lifelines variance (relative difference {var_rel:.2e})")
    # R sets the score of beta to zero, the Newton step left by lifelines measures how close it is
    step = float(np.max(np.abs(np.linalg.solve(i_bb, score.sum(axis=0) - pen @ beta)) * sd))
    ZPH_DIAG["var_rel_diff"] = max(ZPH_DIAG["var_rel_diff"], var_rel)
    ZPH_DIAG["newton_step_per_sd"] = max(ZPH_DIAG["newton_step_per_sd"], step)
    ZPH_DIAG["n_models"] += 1

    p = len(covs)
    out = pd.DataFrame(index=covs, dtype=float)
    for kind in transforms:
        gt = time_transform(time, event, kind)[rep]
        u_g = gt @ score
        i_bg = np.einsum("m,mij->ij", gt, info)
        i_gg = np.einsum("m,mij->ij", gt ** 2, info)
        for k, name in enumerate(covs):
            u = np.zeros(p + 1)
            u[-1] = u_g[k]
            pvals = []
            for gamma_pen in (0.0, pen[k, k]):   # this script, then the R convention
                a = np.block([[i_bb, i_bg[:, [k]]], [i_bg[[k], :], i_gg[[k]][:, [k]] + gamma_pen]])
                chisq = float(u @ np.linalg.solve(a, u))
                pvals.append(float(stats.chi2.sf(chisq, 1)))
                if gamma_pen == 0.0:
                    out.loc[name, f"chisq_{kind}"] = chisq
                    out.loc[name, f"p_{kind}"] = pvals[0]
            key = f"r_gamma_penalty_max_abs_dlog10p_{kind}"
            ZPH_DIAG[key] = max(ZPH_DIAG.get(key, 0.0), abs(np.log10(pvals[0]) - np.log10(pvals[1])))
    return out


def wald(cph, term, scale=1.0):
    """HR, 95% CI and p of one term, with the coefficient multiplied by scale."""
    row = cph.summary.loc[term]
    return (float(np.exp(row["coef"] * scale)),
            float(np.exp(row["coef lower 95%"] * scale)),
            float(np.exp(row["coef upper 95%"] * scale)),
            float(row["p"]))


def per_drug_models(merged, pam, reference):
    """Parts (1) and (2) for the twelve drugs. Returns the PH rows and the per-drug rows."""
    pam_cols = list(pam.columns)
    ph_rows, drug_rows = [], []
    for drug in CANDIDATES + CONTROLS:
        group = "candidate" if drug in CANDIDATES else "positive_control"
        x = merged[drug]
        sd = float(x.std())
        z = (x - x.mean()) / sd

        # (1) canonical model, same column order as script 21
        cox_data = pd.concat([merged[["os_time", "os_event", drug]], pam], axis=1)
        cox_data = cox_data[["os_time", "os_event", drug] + pam_cols].dropna().rename(columns={drug: DRUG})
        cph = fit(cox_data, PENALIZER)
        hr, lo, hi, p = wald(cph, DRUG)
        rel_hr = check_close(f"{drug} canonical HR", hr, reference.loc[drug, "HR"])
        rel_p = check_close(f"{drug} canonical p", p, reference.loc[drug, "pvalue"])

        # Conversion check: the penalized fit with standardized AUC gives the same HR per SD
        std_data = cox_data.assign(**{DRUG: z.loc[cox_data.index].values})
        cph_std = fit(std_data, PENALIZER)
        hr_sd_refit = float(cph_std.hazard_ratios_[DRUG])
        hr_sd_conv = float(np.exp(np.log(hr) * sd))
        rel_conv = check_close(f"{drug} HR per SD conversion", hr_sd_conv, hr_sd_refit)

        ph = ph_pvalues(cph, cox_data)
        ph_rr = ph_pvalues(cph, cox_data, r_rank)
        ph_ex = zph_exact(cph, cox_data, PENALIZER, transforms=TRANSFORMS)
        cph_unpen = fit(cox_data, 0.0)
        ph_unpen = ph_pvalues(cph_unpen, cox_data)
        ph_unpen_ex = zph_exact(cph_unpen, cox_data, 0.0)
        for term in [DRUG] + pam_cols:
            ph_rows.append({
                "drug": drug, "group": group, "term": term,
                "coef": float(cph.params_[term]), "HR": float(np.exp(cph.params_[term])),
                "ph_test_statistic": float(ph.loc[term, "test_statistic"]),
                "ph_p": float(ph.loc[term, "p"]),
                "ph_violation_p05": bool(ph.loc[term, "p"] < ALPHA),
                "ph_p_approx_r_rank": float(ph_rr.loc[term, "p"]),
                "ph_chisq_exact": float(ph_ex.loc[term, "chisq_rank"]),
                "ph_p_exact": float(ph_ex.loc[term, "p_rank"]),
                "ph_violation_p05_exact": bool(ph_ex.loc[term, "p_rank"] < ALPHA),
                "ph_p_exact_km": float(ph_ex.loc[term, "p_km"]),
                "ph_p_exact_event_rank": float(ph_ex.loc[term, "p_event_rank"]),
                "HR_unpenalized": float(np.exp(cph_unpen.params_[term])),
                "ph_test_statistic_unpenalized": float(ph_unpen.loc[term, "test_statistic"]),
                "ph_p_unpenalized": float(ph_unpen.loc[term, "p"]),
                "ph_violation_p05_unpenalized": bool(ph_unpen.loc[term, "p"] < ALPHA),
                "ph_chisq_exact_unpenalized": float(ph_unpen_ex.loc[term, "chisq_rank"]),
                "ph_p_exact_unpenalized": float(ph_unpen_ex.loc[term, "p_rank"]),
                "ph_violation_p05_exact_unpenalized": bool(ph_unpen_ex.loc[term, "p_rank"] < ALPHA),
            })

        # (2) PAM50-stratified, penalized, raw predicted AUC
        strat = merged[["os_time", "os_event", drug, "pam50"]].rename(columns={drug: DRUG})
        cph_s = fit(strat, PENALIZER, strata=["pam50"])
        s_hr, s_lo, s_hi, s_p = wald(cph_s, DRUG)
        s_ex = zph_exact(cph_s, strat, PENALIZER, strata="pam50", transforms=TRANSFORMS).loc[DRUG]
        s_ph_rr = ph_pvalues(cph_s, strat, r_rank).loc[DRUG, "p"]
        s_ph_lifelines = ph_pvalues(cph_s, strat).loc[DRUG, "p"]

        # (2) PAM50-stratified, unpenalized, predicted AUC standardized by its cohort SD
        strat_z = strat.assign(**{DRUG: z.values})
        cph_su = fit(strat_z, 0.0, strata=["pam50"])
        su_hr, su_lo, su_hi, su_p = wald(cph_su, DRUG)
        su_ex = zph_exact(cph_su, strat_z, 0.0, strata="pam50", transforms=TRANSFORMS).loc[DRUG]
        su_ph_rr = ph_pvalues(cph_su, strat_z, r_rank).loc[DRUG, "p"]
        su_ph_lifelines = ph_pvalues(cph_su, strat_z).loc[DRUG, "p"]

        # Unpenalized PAM50 dummies with standardized AUC (no penalty, no stratification)
        cph_du = fit(std_data, 0.0)
        du_hr, du_lo, du_hi, du_p = wald(cph_du, DRUG)

        drug_rows.append({
            "drug": drug, "group": group,
            "n_patients": len(cox_data), "n_events": int(cox_data["os_event"].sum()),
            "sd_predicted_auc": sd,
            "HR_reference": float(reference.loc[drug, "HR"]),
            "HR_canonical": hr, "CI_low_canonical": lo, "CI_high_canonical": hi,
            "p_canonical": p, "HR_rel_diff": rel_hr, "p_rel_diff": rel_p,
            "HR_per_sd_canonical": hr_sd_conv, "HR_per_sd_canonical_refit": hr_sd_refit,
            "HR_per_sd_conversion_rel_diff": rel_conv,
            "ph_p_canonical_lifelines": float(ph.loc[DRUG, "p"]),
            "ph_p_canonical_exact": float(ph_ex.loc[DRUG, "p_rank"]),
            "HR_strat_pen": s_hr, "CI_low_strat_pen": s_lo, "CI_high_strat_pen": s_hi,
            "p_strat_pen": s_p, "HR_per_sd_strat_pen": float(np.exp(np.log(s_hr) * sd)),
            "ph_p_strat_pen": float(s_ex["p_rank"]), "ph_p_strat_pen_exact_km": float(s_ex["p_km"]),
            "ph_p_strat_pen_exact_event_rank": float(s_ex["p_event_rank"]),
            "ph_p_strat_pen_approx_r_rank": float(s_ph_rr),
            "ph_p_strat_pen_lifelines_rank": float(s_ph_lifelines),
            "HR_per_sd_strat_unpen": su_hr, "CI_low_per_sd_strat_unpen": su_lo,
            "CI_high_per_sd_strat_unpen": su_hi, "p_strat_unpen": su_p,
            "ph_p_strat_unpen": float(su_ex["p_rank"]), "ph_p_strat_unpen_exact_km": float(su_ex["p_km"]),
            "ph_p_strat_unpen_exact_event_rank": float(su_ex["p_event_rank"]),
            "ph_p_strat_unpen_approx_r_rank": float(su_ph_rr),
            "ph_p_strat_unpen_lifelines_rank": float(su_ph_lifelines),
            "HR_per_sd_dummies_unpen": du_hr, "CI_low_per_sd_dummies_unpen": du_lo,
            "CI_high_per_sd_dummies_unpen": du_hi, "p_dummies_unpen": du_p,
            "ph_p_dummies_unpen_exact": float(ph_unpen_ex.loc[DRUG, "p_rank"]),
        })
        logger.info("%s: HR %.3f (ref %.3f), per SD %.3f, PH p exact %.3g (lifelines %.3g) | "
                    "stratified HR %.3f p %.2e PH p %.3g | unpenalized stratified HR/SD %.3f p %.2e PH p %.3g",
                    drug, hr, reference.loc[drug, "HR"], hr_sd_conv, ph_ex.loc[DRUG, "p_rank"], ph.loc[DRUG, "p"],
                    s_hr, s_p, s_ex["p_rank"], su_hr, su_p, su_ex["p_rank"])
    return pd.DataFrame(ph_rows), pd.DataFrame(drug_rows)


def hr_per_sd_all(merged, surv):
    """Part (3): HR per SD for every drug of survival_SCANB.csv, no refit."""
    missing = sorted(set(surv["drug"]) - set(merged.columns))
    if missing:
        raise ValueError(f"{len(missing)} drugs of survival_SCANB.csv have no predictions: {missing[:10]}")
    out = surv[["drug", "HR", "pvalue", "padj"]].copy().reset_index(drop=True)
    out["sd_predicted_auc"] = merged[out["drug"]].std().values
    out["sd_percentile"] = out["sd_predicted_auc"].rank(pct=True) * 100
    out["log_HR_per_unit"] = np.log(out["HR"])
    # |z| from the Wald p (lifelines p = chi2.sf(z**2, 1)), then SE of log HR per unit
    abs_z = stats.norm.isf(out["pvalue"] / 2)
    se = np.where(abs_z > 0, out["log_HR_per_unit"].abs() / abs_z, np.inf)
    out["log_HR_per_sd"] = out["log_HR_per_unit"] * out["sd_predicted_auc"]
    out["HR_per_sd"] = np.exp(out["log_HR_per_sd"])
    out["CI_low_per_sd"] = np.exp((out["log_HR_per_unit"] - Z_CRIT * se) * out["sd_predicted_auc"])
    out["CI_high_per_sd"] = np.exp((out["log_HR_per_unit"] + Z_CRIT * se) * out["sd_predicted_auc"])
    out["fdr_significant"] = out["padj"] < ALPHA
    out["group"] = np.select([out["drug"].isin(CANDIDATES), out["drug"].isin(CONTROLS)],
                             ["candidate", "positive_control"], default="other")
    if SWEEP_PATH.exists():
        sweep = pd.read_csv(SWEEP_PATH)
        s = sweep[np.isclose(sweep["penalizer"], SWEEP_PENALIZER)].set_index("drug")["HR"]
        out["HR_pen001"] = out["drug"].map(s)
        out["HR_per_sd_pen001"] = np.exp(np.log(out["HR_pen001"]) * out["sd_predicted_auc"])
    else:
        logger.warning("%s not found, the penalizer 0.01 columns are skipped", SWEEP_PATH)
    return out


def make_put(rows):
    """put(metric, value) appending one row of a metric/value summary to rows."""
    def put(metric, value):
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        rows.append({"metric": metric, "value": value})
    return put


def put_ph_summary(put, ph_df, strat_df, n_cox, events_cox):
    """Cohort size, reproduction, zph_exact diagnostics, parts (4) and (2), in that order.

    Shared by SCAN-B and METABRIC. For SCAN-B the rows are those, and in the order,
    that main() wrote before METABRIC was added, so ph_summary.csv does not change."""
    put("cox_n_patients", n_cox)
    put("cox_n_events", events_cox)
    put("reproduced_max_HR_rel_diff", float(strat_df["HR_rel_diff"].max()))
    put("reproduced_max_p_rel_diff", float(strat_df["p_rel_diff"].max()))
    put("hr_per_sd_conversion_max_rel_diff", float(strat_df["HR_per_sd_conversion_rel_diff"].max()))
    put("zph_exact_n_models", ZPH_DIAG["n_models"])
    put("zph_exact_max_variance_rel_diff_vs_lifelines", ZPH_DIAG["var_rel_diff"])
    put("zph_exact_max_newton_step_per_sd", ZPH_DIAG["newton_step_per_sd"])
    for kind in TRANSFORMS:
        put(f"zph_exact_r_gamma_penalty_max_abs_dlog10p_{kind}", ZPH_DIAG[f"r_gamma_penalty_max_abs_dlog10p_{kind}"])

    # (4) proportional hazards counts, canonical and unpenalized models, both tests
    drug_terms = ph_df[ph_df["term"] == DRUG]
    pam_terms = ph_df[ph_df["term"] != DRUG]
    put("ph_n_drug_terms", len(drug_terms))
    put("ph_n_pam50_term_tests", len(pam_terms))
    tests = {"": "ph_violation_p05", "_exact": "ph_violation_p05_exact",
             "_unpenalized": "ph_violation_p05_unpenalized",
             "_exact_unpenalized": "ph_violation_p05_exact_unpenalized"}
    for tag, col in tests.items():
        put(f"ph_n_drug_terms_violating_p05{tag}", int(drug_terms[col].sum()))
        put(f"ph_drug_terms_violating_p05{tag}", sorted(drug_terms.loc[drug_terms[col], "drug"]))
        put(f"ph_n_pam50_term_tests_violating_p05{tag}", int(pam_terms[col].sum()))
        for term, g in pam_terms.groupby("term"):
            put(f"ph_{term}_n_models_violating_p05{tag}", int(g[col].sum()))
    # Exact test with the other time transforms, canonical model
    for tag in ["_exact_km", "_exact_event_rank"]:
        viol = drug_terms[f"ph_p{tag}"] < ALPHA
        put(f"ph_n_drug_terms_violating_p05{tag}", int(viol.sum()))
        put(f"ph_drug_terms_violating_p05{tag}", sorted(drug_terms.loc[viol, "drug"]))
        put(f"ph_n_pam50_term_tests_violating_p05{tag}", int((pam_terms[f"ph_p{tag}"] < ALPHA).sum()))
        for term, g in pam_terms.groupby("term"):
            put(f"ph_{term}_n_models_violating_p05{tag}", int((g[f"ph_p{tag}"] < ALPHA).sum()))
    for tag, col in {"": "ph_p", "_exact": "ph_p_exact", "_exact_unpenalized": "ph_p_exact_unpenalized"}.items():
        put(f"ph_drug_term_p_min{tag}", float(drug_terms[col].min()))
        for term, g in pam_terms.groupby("term"):
            put(f"ph_{term}_p_min{tag}", float(g[col].min()))
            put(f"ph_{term}_p_max{tag}", float(g[col].max()))
    for term, g in pam_terms.groupby("term"):
        # The penalty also shrinks the PAM50 dummies, compare with the unpenalized fit
        put(f"{term}_HR_penalized_median", float(g["HR"].median()))
        put(f"{term}_HR_unpenalized_median", float(g["HR_unpenalized"].median()))
    for _, r in drug_terms.iterrows():
        put(f"{r['drug']}_ph_p_exact", r["ph_p_exact"])
        put(f"{r['drug']}_ph_p_exact_unpenalized", r["ph_p_exact_unpenalized"])
        put(f"{r['drug']}_ph_p_lifelines", r["ph_p"])

    # (2) stratified sensitivity
    same_dir = (np.sign(np.log(strat_df["HR_strat_pen"])) == np.sign(np.log(strat_df["HR_canonical"])))
    put("strat_pen_n_same_direction", int(same_dir.sum()))
    put("strat_pen_n_p05", int((strat_df["p_strat_pen"] < ALPHA).sum()))
    put("strat_unpen_n_p05", int((strat_df["p_strat_unpen"] < ALPHA).sum()))
    for model in ["strat_pen", "strat_unpen"]:
        for tag in ["", "_exact_km", "_exact_event_rank", "_approx_r_rank", "_lifelines_rank"]:
            col = f"ph_p_{model}{tag}"
            put(f"{model}_n_drug_terms_violating_ph_p05{tag}", int((strat_df[col] < ALPHA).sum()))
            put(f"{model}_drug_terms_violating_ph_p05{tag}", sorted(strat_df.loc[strat_df[col] < ALPHA, "drug"]))
        put(f"{model}_ph_p_min", float(strat_df[f"ph_p_{model}"].min()))
    # log HR per SD of the canonical fit over that of the unpenalized fit with the same
    # covariates. It mixes the ridge shrinkage of the drug term with the change in the
    # PAM50 adjustment, whose dummies the ridge also shrinks, so it can exceed 1.
    ratio = np.log(strat_df["HR_per_sd_canonical"]) / np.log(strat_df["HR_per_sd_dummies_unpen"])
    put("log_hr_per_sd_ratio_canonical_vs_unpenalized_median", float(ratio.median()))
    put("log_hr_per_sd_ratio_canonical_vs_unpenalized_min", float(ratio.min()))
    put("log_hr_per_sd_ratio_canonical_vs_unpenalized_max", float(ratio.max()))
    put("log_hr_per_sd_ratio_canonical_vs_unpenalized_n_above_1", int((ratio > 1).sum()))
    for _, r in strat_df.iterrows():
        put(f"{r['drug']}_HR_strat_pen", r["HR_strat_pen"])
        put(f"{r['drug']}_p_strat_pen", r["p_strat_pen"])
        put(f"{r['drug']}_HR_per_sd_strat_unpen", r["HR_per_sd_strat_unpen"])


def run_scanb():
    """Parts (1) to (4) on SCAN-B, written to the four files of the canonical cohort."""
    merged, pam = load_cohort()
    surv = pd.read_csv(SURV_PATH)
    reference = surv.set_index("drug")

    ph_df, strat_df = per_drug_models(merged, pam, reference)
    all_df = hr_per_sd_all(merged, surv)

    # Wald-derived CI per SD must match the fitted CI of the twelve refits
    chk = all_df.set_index("drug")
    for _, r in strat_df.iterrows():
        sd = r["sd_predicted_auc"]
        check_close(f"{r['drug']} derived CI low per SD", chk.loc[r["drug"], "CI_low_per_sd"],
                    float(np.exp(np.log(r["CI_low_canonical"]) * sd)))
        check_close(f"{r['drug']} derived CI high per SD", chk.loc[r["drug"], "CI_high_per_sd"],
                    float(np.exp(np.log(r["CI_high_canonical"]) * sd)))
    logger.info("Reproduction passed: 12 HRs and p-values, conversion and derived CIs. zph_exact "
                "information matches lifelines variance in %d models (max relative difference %.2e)",
                ZPH_DIAG["n_models"], ZPH_DIAG["var_rel_diff"])

    rows = []
    put = make_put(rows)
    put_ph_summary(put, ph_df, strat_df, N_COX, EVENTS_COX)

    # (3) HR per SD across drugs
    cand = all_df[all_df["group"] == "candidate"].set_index("drug")
    put("all_drugs_n", len(all_df))
    put("all_drugs_sd_median", float(all_df["sd_predicted_auc"].median()))
    put("all_drugs_sd_min", float(all_df["sd_predicted_auc"].min()))
    put("all_drugs_sd_max", float(all_df["sd_predicted_auc"].max()))
    put("candidates_sd_min", float(cand["sd_predicted_auc"].min()))
    put("candidates_sd_max", float(cand["sd_predicted_auc"].max()))
    put("candidates_sd_percentile_min", float(cand["sd_percentile"].min()))
    put("candidates_sd_percentile_max", float(cand["sd_percentile"].max()))
    put("candidates_HR_per_sd_min", float(cand["HR_per_sd"].min()))
    put("candidates_HR_per_sd_max", float(cand["HR_per_sd"].max()))
    for d in CANDIDATES:
        put(f"{d}_sd", float(cand.loc[d, "sd_predicted_auc"]))
        put(f"{d}_HR_per_sd", float(cand.loc[d, "HR_per_sd"]))
    for d in CONTROLS:
        put(f"{d}_HR_per_sd", float(all_df.set_index("drug").loc[d, "HR_per_sd"]))
    fdr = all_df[all_df["fdr_significant"]]
    put("fdr_n_drugs", len(fdr))
    put("fdr_HR_per_unit_min", float(fdr["HR"].min()))
    put("fdr_HR_per_unit_max", float(fdr["HR"].max()))
    put("fdr_HR_per_sd_min", float(fdr["HR_per_sd"].min()))
    put("fdr_HR_per_sd_max", float(fdr["HR_per_sd"].max()))
    # The per-unit HR window [1.5, 10] of the candidate rule, read on the per-SD scale
    up = fdr[fdr["HR"] > 1]
    parts = {"in": (up["HR"] >= 1.5) & (up["HR"] <= 10), "above10": up["HR"] > 10, "below1p5": up["HR"] < 1.5}
    put("window_n_fdr_hr_gt1", len(up))
    for tag, m in parts.items():
        put(f"window_{tag}_n", int(m.sum()))
        put(f"window_{tag}_sd_median", float(up.loc[m, "sd_predicted_auc"].median()))
        put(f"window_{tag}_HR_per_sd_min", float(up.loc[m, "HR_per_sd"].min()))
        put(f"window_{tag}_HR_per_sd_max", float(up.loc[m, "HR_per_sd"].max()))
    rho, p = stats.spearmanr(all_df["log_HR_per_unit"].abs(), all_df["sd_predicted_auc"])
    put("spearman_abs_log_hr_per_unit_vs_sd_rho", float(rho))
    put("spearman_abs_log_hr_per_unit_vs_sd_p", float(p))
    rho2, p2 = stats.spearmanr(all_df["log_HR_per_sd"].abs(), all_df["sd_predicted_auc"])
    put("spearman_abs_log_hr_per_sd_vs_sd_rho", float(rho2))
    put("spearman_abs_log_hr_per_sd_vs_sd_p", float(p2))
    logger.info("Spearman |log HR per unit| vs SD: rho %.3f (p %.2e), |log HR per SD| vs SD: rho %.3f",
                rho, p, rho2)
    if "HR_pen001" in all_df.columns:
        top = all_df.loc[all_df["HR_pen001"].idxmax()]
        put("pen001_max_HR_drug", top["drug"])
        put("pen001_max_HR", float(top["HR_pen001"]))
        put("pen001_max_HR_sd", float(top["sd_predicted_auc"]))
        put("pen001_max_HR_sd_percentile", float(top["sd_percentile"]))
        put("pen001_max_HR_per_sd_pen001", float(top["HR_per_sd_pen001"]))
        put("pen001_max_HR_canonical", float(top["HR"]))
        put("pen001_max_HR_per_sd_canonical", float(top["HR_per_sd"]))
        logger.info("Largest HR at penalizer 0.01: %s, HR %.1f, SD %.4f, HR per SD %.3f (canonical %.3f)",
                    top["drug"], top["HR_pen001"], top["sd_predicted_auc"], top["HR_per_sd_pen001"],
                    top["HR_per_sd"])

    ph_df.to_csv(OUT / "ph_schoenfeld.csv", index=False)
    strat_df.to_csv(OUT / "cox_pam50_stratified.csv", index=False)
    all_df.sort_values("pvalue").to_csv(OUT / "hr_per_sd_all_drugs.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ph_summary.csv", index=False)
    logger.info("Wrote %s, %s, %s, %s", OUT / "ph_schoenfeld.csv", OUT / "cox_pam50_stratified.csv",
                OUT / "hr_per_sd_all_drugs.csv", OUT / "ph_summary.csv")
    print(summary.to_string(index=False))


def run_metabric():
    """Parts (1), (2) and (4) on the canonical METABRIC model, in files of their own."""
    # zph_exact diagnostics are per cohort, so SCAN-B keeps the values it always had
    ZPH_DIAG.clear()
    ZPH_DIAG.update({"var_rel_diff": 0.0, "newton_step_per_sd": 0.0, "n_models": 0})
    merged, pam = load_metabric_cohort()
    reference = pd.read_csv(SURV_METABRIC_PATH).set_index("drug")
    ph_df, strat_df = per_drug_models(merged, pam, reference)
    logger.info("METABRIC reproduction passed: 12 HRs and p-values and the conversion. zph_exact "
                "information matches lifelines variance in %d models (max relative difference %.2e)",
                ZPH_DIAG["n_models"], ZPH_DIAG["var_rel_diff"])
    rows = []
    put = make_put(rows)
    put_ph_summary(put, ph_df, strat_df, N_COX_METABRIC, EVENTS_COX_METABRIC)
    ph_df.to_csv(OUT / "ph_schoenfeld_metabric.csv", index=False)
    strat_df.to_csv(OUT / "cox_pam50_stratified_metabric.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ph_summary_metabric.csv", index=False)
    logger.info("Wrote %s, %s, %s", OUT / "ph_schoenfeld_metabric.csv",
                OUT / "cox_pam50_stratified_metabric.csv", OUT / "ph_summary_metabric.csv")
    got = summary.set_index("metric")["value"]
    logger.info("METABRIC: PH violated at p < 0.05 (exact test) by %s of %s drug terms and %s of %s "
                "PAM50 term tests of the canonical model, stratified drug terms %s (penalized) and %s "
                "(unpenalized)", got["ph_n_drug_terms_violating_p05_exact"], got["ph_n_drug_terms"],
                got["ph_n_pam50_term_tests_violating_p05_exact"], got["ph_n_pam50_term_tests"],
                got["strat_pen_n_drug_terms_violating_ph_p05"], got["strat_unpen_n_drug_terms_violating_ph_p05"])
    print(summary.to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_scanb()
    run_metabric()


if __name__ == "__main__":
    main()
