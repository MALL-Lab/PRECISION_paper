"""The exact proportional hazards test of script 64 against R survival.

zph_exact (scripts/64_proportional_hazards.py) reimplements
cox.zph(fit, transform = "rank", terms = FALSE) of R survival 3.x, the test that
scripts 64, 69 and 71 report. Inside each run the only automatic guarantee is that
its information matrix reproduces the variance matrix of the lifelines fit, which
says nothing about the score of x_k * g(t) nor about the transform of time. This
test compares the statistic with R on a small synthetic data set with no patient
data: 40 rows, 31 events, integer times with tied event times and events tied with
censored times, one continuous and one binary covariate and three strata
(rep(1:3, length.out = 40)). It covers the four branches of zph_exact, with Efron
ties and the rank transform:

    plain         x1 + x2, unpenalized, no strata
    strata        x1 + x2 + strata(s), unpenalized (Efron risk sets per stratum,
                  rank of time pooled across strata)
    ridge         penalizer 0.1 on x1 and x2, in R ridge(x1, x2, theta = 0.1 * n,
                  scale = TRUE), as in the canonical model of script 64
    ridge_strata  penalizer 0.1 on x1 with strata(s), in R ridge(x1, theta = 0.1 * n,
                  scale = TRUE) + strata(s), as in the stratified penalized model of 64

It does not cover the km and event_rank transforms, nor the lifelines approximation
beyond showing that it is far from the exact test.

The reference values were computed with Rscript (tests/zph_exact_reference.R,
which holds the same data, checked below) and frozen here:

    R version 4.5.2 (2025-10-31 ucrt), survival 3.8.3
    coxph(..., ties = "efron",
          control = coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 100))
    cox.zph(fit, transform = "rank", terms = FALSE)$table

Unpenalized models: zph_exact reproduces chisq and p of cox.zph (relative tolerance
1e-9, the measured difference is below 4e-12).

Penalized models: cox.zph and zph_exact add the ridge second derivative to the
information matrix in different ways, so the comparison is made in two steps.
  - chisq_diag is the statistic of a copy of cox.zph in which only two lines
    change: the ridge second derivative goes on the diagonal of the beta block and
    not on the gamma block, the convention of zph_exact. Score and information
    still come from the C code of survival. zph_exact reproduces it at 1e-9.
  - Against cox.zph itself the agreement is only 1e-2 (measured 9.7e-3 for x1 in
    ridge and 1.4e-3 in ridge_strata). cox.zph assigns the vector
    fit$coxlist2$second to the nvar x nvar beta block, and R recycles it column by
    column, so row i gets second[i] in every column and not only on the diagonal
    (this only matters with more than one penalized coefficient). It also adds the
    ridge to the gamma block. Both choices are explained in the docstring of
    script 64. The p of cox.zph for a penalized fit uses the effective df of the
    ridge term, so only chisq is compared and p is taken at 1 df.

The lifelines fits use precision 1e-14 so both programs evaluate the test at the
same coefficients (relative difference below 2e-12). lifelines stops when the
Newton decrement falls below precision, so with the default 1e-7 it stops about
5e-7 per SD short of the optimum and the p-values move by up to 8e-7, and with
1e-12 the penalized fits still leave a step of 2e-7 per SD. The lifelines
approximation (pre-3.0 R cox.zph) gives p-values between 2.7% and 13% away from the
exact ones on the same data, so the tolerance separates the two tests.
"""

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest
from lifelines import CoxPHFitter
from scipy import stats

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "64_proportional_hazards.py"
R_SCRIPT = Path(__file__).resolve().parent / "zph_exact_reference.R"
REL_TOL = 1e-9
REL_TOL_COX_ZPH_PENALIZED = 1e-2

TIME = [7, 2, 25, 6, 2, 4, 6, 1, 8, 1, 25, 25, 25, 10, 8, 7, 25, 1, 4, 23,
        17, 12, 4, 12, 16, 25, 4, 10, 25, 1, 25, 1, 3, 8, 3, 12, 3, 10, 1, 25]
STATUS = [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
          1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
X1 = [0.5, 1.94, -0.93, 0.55, 1.09, -0.03, 0.44, 0.36, 0.3, 0.85,
      -1.03, -1.92, -0.49, -0.28, 1.56, 0.21, -0.13, 1.87, -0.66, 1.4,
      0.2, 1.53, -0.16, -1.43, 0.46, 0.57, 0.55, -0.48, 0.03, -0.39,
      -0.06, 1.18, -0.01, -0.61, 0.28, -0.64, -0.1, -0.51, 0.37, -1.38]
X2 = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
STRATUM = [i % 3 + 1 for i in range(len(TIME))]   # rep(1:3, length.out = 40) in R

# Per model: covariates, stratum column, lifelines penalizer
CASES = {
    "plain": (["x1", "x2"], None, 0.0),
    "strata": (["x1", "x2"], "s", 0.0),
    "ridge": (["x1", "x2"], None, 0.1),
    "ridge_strata": (["x1"], "s", 0.1),
}
PENALIZED = [c for c, (_, _, pen) in CASES.items() if pen > 0]

# R survival 3.8.3, printed by tests/zph_exact_reference.R. coef is coef(fit), chisq and p
# are cox.zph(fit, transform = "rank", terms = FALSE)$table (p only for unpenalized fits),
# chisq_diag is the same table with the ridge on the diagonal of the beta block only.
R_REF = {
    "plain": {
        "coef": {"x1": 0.541222578175, "x2": -0.275738709704},
        "chisq": {"x1": 0.639207115539514, "x2": 0.0192869937439489},
        "p": {"x1": 0.423998058003723, "x2": 0.889546855065781},
    },
    "strata": {
        "coef": {"x1": 0.609589737093, "x2": 0.156078927188},
        "chisq": {"x1": 0.0394596694067166, "x2": 0.0686844544862579},
        "p": {"x1": 0.842540778189372, "x2": 0.79326193064126},
    },
    "ridge": {
        "coef": {"x1": 0.455907663746, "x2": -0.191549261834},
        "chisq": {"x1": 0.761237069018749, "x2": 0.00757338514998872},
        "chisq_diag": {"x1": 0.753830724504622, "x2": 0.00756044964094755},
    },
    "ridge_strata": {
        "coef": {"x1": 0.537038167500},
        "chisq": {"x1": 0.0946509356413108},
        "chisq_diag": {"x1": 0.0947806882139619},
    },
}


def _load_script_64():
    spec = importlib.util.spec_from_file_location("s64_proportional_hazards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r_exact(case):
    """chisq and p at 1 df of the R statistic in the convention of zph_exact."""
    ref = R_REF[case]
    chisq = ref.get("chisq_diag", ref["chisq"])
    return {k: (v, ref["p"][k] if "p" in ref else float(stats.chi2.sf(v, 1))) for k, v in chisq.items()}


@pytest.fixture(scope="module")
def fitted():
    mod = _load_script_64()
    data = pd.DataFrame({"os_time": TIME, "os_event": STATUS, "x1": X1, "x2": X2,
                         "s": STRATUM}).astype(float)
    out = {}
    for case, (covs, strata, penalizer) in CASES.items():
        df = data[["os_time", "os_event"] + covs + ([strata] if strata else [])]
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df, duration_col="os_time", event_col="os_event",
                strata=[strata] if strata else None, fit_options={"precision": 1e-14})
        out[case] = (cph, df, strata, penalizer)
    return mod, out


def test_data_has_the_ties_the_test_is_about():
    d = pd.DataFrame({"time": TIME, "status": STATUS, "s": STRATUM})
    events = d.loc[d["status"] == 1, "time"]
    assert len(TIME) == len(STATUS) == len(X1) == len(X2) == 40
    assert int(sum(STATUS)) == 31
    assert int((events.value_counts() > 1).sum()) == 8      # event times shared by several events
    assert set(d.loc[d["status"] == 0, "time"]) & set(events)   # events tied with censored times
    for _, g in d.groupby("s"):                             # tied event times inside every stratum
        assert int((g.loc[g["status"] == 1, "time"].value_counts() > 1).sum()) >= 2


def test_reference_script_holds_the_same_data():
    text = R_SCRIPT.read_text(encoding="utf-8")
    vectors = {name: [float(v) for v in body.split(",")]
               for name, body in re.findall(r"(\w+) = c\(([^)]*)\)", text)}
    assert vectors == {"time": TIME, "status": STATUS, "x1": X1, "x2": X2}
    assert "d$s <- rep(1:3, length.out = nrow(d))" in text


@pytest.mark.parametrize("case", CASES)
def test_lifelines_fit_matches_r_coefficients(fitted, case):
    _, fits = fitted
    cph = fits[case][0]
    assert list(cph.params_.index) == list(R_REF[case]["coef"])
    for name, coef in R_REF[case]["coef"].items():
        assert float(cph.params_[name]) == pytest.approx(coef, rel=REL_TOL)


@pytest.mark.parametrize("case", CASES)
def test_zph_exact_reproduces_r_cox_zph_rank(fitted, case):
    mod, fits = fitted
    cph, df, strata, penalizer = fits[case]
    out = mod.zph_exact(cph, df, penalizer, strata=strata, transforms=("rank",))
    for name, (chisq, p) in _r_exact(case).items():
        assert float(out.loc[name, "chisq_rank"]) == pytest.approx(chisq, rel=REL_TOL)
        assert float(out.loc[name, "p_rank"]) == pytest.approx(p, rel=REL_TOL)


@pytest.mark.parametrize("case", PENALIZED)
def test_penalized_zph_exact_against_cox_zph_itself(fitted, case):
    """Only the way the ridge enters the information separates zph_exact from cox.zph."""
    mod, fits = fitted
    cph, df, strata, penalizer = fits[case]
    out = mod.zph_exact(cph, df, penalizer, strata=strata, transforms=("rank",))
    for name, chisq in R_REF[case]["chisq"].items():
        assert float(out.loc[name, "chisq_rank"]) == pytest.approx(chisq, rel=REL_TOL_COX_ZPH_PENALIZED)


@pytest.mark.parametrize("case", CASES)
def test_tolerance_separates_the_exact_test_from_the_approximation(fitted, case):
    mod, fits = fitted
    cph, df, _, _ = fits[case]
    approx = mod.ph_pvalues(cph, df, mod.r_rank)
    for name, (_, p) in _r_exact(case).items():
        assert float(approx.loc[name, "p"]) != pytest.approx(p, rel=1e-2)
