"""The vectorized Cox score test against the log-rank test of lifelines.

With a binary covariate the score at beta = 0 is the log-rank O - E, so the two
statistics share their numerator and differ only in the variance. Without tied
event times both variances are the same sum of p_j (1 - p_j) and z ** 2 equals the
log-rank chi-square to rounding. With ties the Breslow information of the module
keeps d_j p_j (1 - p_j) at each event time, whereas the log-rank test uses the
hypergeometric variance, smaller by the factor (n_j - d_j) / (n_j - 1). The ratio
z ** 2 / chi-square is then exactly V / I, the log-rank variance over the Breslow
information, a weighted mean of those factors that lies between their minimum over
the event times and 1: the score test is conservative. The tied case computes V and
I by hand and checks the ratio to rounding.
"""

import numpy as np
import pytest
from lifelines.statistics import logrank_test

from src.analysis.score_test import martingale_residuals, permutation_z, score_p, score_test


def _logrank(time, event, x):
    g = x == 1
    return logrank_test(time[g], time[~g], event[g], event[~g])


def _hypergeometric_factors(time, event):
    """(n_j - d_j) / (n_j - 1) at every event time, the log-rank to Breslow variance ratio."""
    factors = []
    for t in np.unique(time[event == 1]):
        n_risk = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        factors.append((n_risk - d) / (n_risk - 1))
    return np.array(factors)


def _breslow_and_logrank_variances(time, event, x):
    """By hand, for a binary x: the Breslow information I = sum_j d_j p_j (1 - p_j) and the
    hypergeometric log-rank variance V = sum_j d_j p_j (1 - p_j) (n_j - d_j) / (n_j - 1),
    with n_j at risk, d_j events and p_j the share of x = 1 at risk at event time t_j."""
    info, var = 0.0, 0.0
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n = int(at_risk.sum())
        d = int(((time == t) & (event == 1)).sum())
        p = float(x[at_risk].mean())
        info += d * p * (1 - p)
        var += d * p * (1 - p) * (n - d) / (n - 1)
    return info, var


def test_untied_binary_covariate_equals_logrank():
    rng = np.random.default_rng(42)
    n = 300
    time = rng.exponential(700, n) + 1
    event = (rng.random(n) < 0.3).astype(float)
    x = (rng.random(n) < 0.5).astype(float)
    assert np.unique(time).size == n

    _, _, z = score_test(x[:, None], time, event)
    lr = _logrank(time, event, x)
    assert z[0] ** 2 == pytest.approx(lr.test_statistic, rel=1e-10)
    assert score_p(z)[0] == pytest.approx(lr.p_value, rel=1e-8)


def test_heavy_ties_ratio_is_the_logrank_over_breslow_variance():
    rng = np.random.default_rng(7)
    n = 200
    # Six follow-up times for 200 patients: dozens of events share every time
    time = rng.integers(1, 7, n).astype(float)
    event = (rng.random(n) < 0.6).astype(float)
    x = (rng.random(n) < 0.4).astype(float)
    x[rng.random(n) < 0.3 * event] = 1.0   # some association, so the statistic is not near 0
    max_tied = max(int(((time == t) & (event == 1)).sum()) for t in np.unique(time))
    assert max_tied >= 15

    _, information, z = score_test(x[:, None], time, event)
    chi2 = _logrank(time, event, x).test_statistic
    info, var = _breslow_and_logrank_variances(time, event, x)
    # The module's information is the Breslow one, and the two statistics share O - E,
    # so z ** 2 / chi-square is exactly V / I
    assert information[0] == pytest.approx(info, rel=1e-10)
    assert z[0] ** 2 / chi2 == pytest.approx(var / info, rel=1e-10)
    # V / I is a weighted mean of the hypergeometric factors, and the regime is really tied
    lower = _hypergeometric_factors(time, event).min()
    assert lower - 1e-12 <= var / info < 0.99


def test_ties_of_two_events_stay_within_one_percent():
    """The regime of the cohorts the module serves: pairs of events on the same day in
    risk sets of hundreds of patients, where the two variances differ by under 1%."""
    rng = np.random.default_rng(11)
    n = 400
    time = rng.exponential(500, n) + 1
    event = (rng.random(n) < 0.3).astype(float)
    x = (rng.random(n) < 0.5).astype(float)
    # Tie the 40 earliest events in pairs, 20 event times with two events each
    early = np.flatnonzero(event == 1)[np.argsort(time[event == 1])][:40]
    time[early[1::2]] = time[early[0::2]]
    max_tied = max(int(((time == t) & (event == 1)).sum()) for t in np.unique(time))
    assert max_tied == 2

    _, _, z = score_test(x[:, None], time, event)
    chi2 = _logrank(time, event, x).test_statistic
    ratio = z[0] ** 2 / chi2
    lower = _hypergeometric_factors(time, event).min()
    assert lower > 0.99
    assert lower - 1e-12 <= ratio < 1


def test_invariant_to_linear_rescaling():
    rng = np.random.default_rng(3)
    n = 250
    time = rng.exponential(300, n) + 1
    event = (rng.random(n) < 0.4).astype(float)
    x = rng.normal(size=n)
    X = np.column_stack([x, 3.5 * x - 2.0, -0.2 * x + 7.0])
    _, _, z = score_test(X, time, event)
    assert z[1] == pytest.approx(z[0], rel=1e-10)
    assert z[2] == pytest.approx(-z[0], rel=1e-10)


def test_martingale_residuals_sum_to_zero_and_give_the_score():
    rng = np.random.default_rng(5)
    n = 150
    time = rng.integers(1, 40, n).astype(float)
    event = (rng.random(n) < 0.5).astype(float)
    X = rng.normal(size=(n, 4))
    resid = martingale_residuals(time, event)
    assert resid.sum() == pytest.approx(0.0, abs=1e-10)
    u, _, z = score_test(X, time, event)
    u_r, _, z_r = score_test(X, time, event, residuals=resid)
    assert np.allclose(u, X.T @ resid, rtol=0, atol=1e-12)
    assert np.array_equal(z, z_r)


def test_constant_covariate_gives_nan_and_permutation_shape():
    rng = np.random.default_rng(9)
    n = 60
    time = rng.exponential(100, n) + 1
    event = (rng.random(n) < 0.5).astype(float)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    _, information, z = score_test(X, time, event)
    assert information[0] == 0 and np.isnan(z[0]) and np.isfinite(z[1])
    null = permutation_z(X[:, 1:], time, event, np.random.default_rng(0), n_perm=25, block=10)
    assert null.shape == (25, 1) and np.all(np.isfinite(null))


@pytest.mark.parametrize("bad", ["time", "x"])
def test_invalid_inputs_raise(bad):
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    X = np.array([[0.0], [1.0], [0.5]])
    if bad == "time":
        time[0] = 0.0
    else:
        X[1, 0] = np.nan
    with pytest.raises(ValueError):
        score_test(X, time, event)
