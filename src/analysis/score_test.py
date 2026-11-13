"""Cox score test evaluated at beta = 0, vectorized over many covariates.

The score test is what makes a cohort-wide comparison of drug-survival
associations affordable. At beta = 0 the partial likelihood score and its
information have closed forms, so one call gives the statistic of every drug at
once, and a permutation null costs a matrix multiplication per replicate instead
of a Cox fit per drug and replicate.

Two properties matter for the comparison it serves:

* the statistic is invariant to any linear rescaling of the covariate, so two
  cohorts whose predicted values have different spreads can be compared without
  choosing whose standard deviation to standardize by, and
* with a binary covariate and no tied event times it equals the log-rank
  statistic exactly. tests/test_score_test.py checks it against
  lifelines.statistics.logrank_test, to rounding without ties, and with them
  checks that the ratio of the two statistics is the hypergeometric variance over
  the Breslow information, both computed by hand.

Ties are handled with the Breslow approximation. At an event time with d tied
events among n at risk, its information exceeds the hypergeometric variance of
the log-rank test by the factor (n - 1) / (n - d), so with ties the statistic is
smaller than the log-rank one, never larger, and Efron would differ as well. In
the cohorts of script 69 the factor stays below 1.004: FUSCC-TNBC has at most two
relapses on the same day (3 of 46 event times), and the SCAN-B TNBC reference at
most three deaths (one time, 7 more with two) and two relapses. The whole SCAN-B
cohort, which this module does not serve, has up to four deaths on the same day
(19 of its 994 death times with more than two).
"""

import numpy as np
from scipy import stats


def martingale_residuals(time, event):
    """Residuals at beta = 0: event minus the Nelson-Aalen cumulative hazard.

    The score of any covariate is its inner product with this vector, which is
    what turns a permutation null into one matrix multiplication.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    order = np.argsort(time, kind="stable")
    t, d = time[order], event[order]
    n = t.size
    n_risk = n - np.arange(n)
    # Increment of the Nelson-Aalen estimator at every event time, with ties
    # sharing the same risk set
    increments = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        d_j = d[i:j + 1].sum()
        if d_j > 0:
            increments[i] = d_j / n_risk[i]
        i = j + 1
    cum_hazard = np.cumsum(increments)
    resid = np.empty(n)
    resid[order] = d - cum_hazard
    return resid


def score_test(X, time, event, residuals=None):
    """Score statistic, information and z of every column of X, at beta = 0.

    X is (n patients, p covariates). Returns u, information and z, each of
    length p. A column with zero information gives z = nan rather than an error,
    and the caller decides what to do with it.
    """
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if X.shape[0] != time.size or time.size != event.size:
        raise ValueError(f"X has {X.shape[0]} rows for {time.size} times and {event.size} events")
    if np.any(time <= 0):
        raise ValueError("Non-positive follow-up times reached the score test")
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite covariate values reached the score test")

    resid = martingale_residuals(time, event) if residuals is None else np.asarray(residuals, float)
    u = X.T @ resid

    order = np.argsort(time, kind="stable")
    t, d = time[order], event[order]
    Xs = X[order]
    n = t.size
    # Reverse cumulative sums give the risk set of every time point
    s1 = np.cumsum(Xs[::-1], axis=0)[::-1]
    s2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
    n_risk = (n - np.arange(n)).astype(float)

    information = np.zeros(X.shape[1])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        d_j = d[i:j + 1].sum()
        if d_j > 0:
            mean = s1[i] / n_risk[i]
            var = s2[i] / n_risk[i] - mean ** 2
            information += d_j * var
        i = j + 1

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(information > 0, u / np.sqrt(information), np.nan)
    return u, information, z


def score_p(z):
    """Two-sided p-value of the score statistic."""
    return 2 * stats.norm.sf(np.abs(np.asarray(z, dtype=float)))


def permutation_z(X, time, event, rng, n_perm, block=200):
    """Null distribution of the z vector under permutation of the outcome.

    Permuting the (time, event) pairs across patients is the same as permuting
    the rows of X with the outcome held in place, which is what this does.
    Returns an (n_perm, p) array.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    out = np.empty((n_perm, X.shape[1]), dtype=np.float32)
    for start in range(0, n_perm, block):
        stop = min(start + block, n_perm)
        for r in range(start, stop):
            perm = rng.permutation(n)
            out[r] = score_test(X[perm], time, event)[2]
    return out
