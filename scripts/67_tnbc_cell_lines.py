#!/usr/bin/env python3
"""Activity of the candidates in the TNBC cell lines of the PRISM screen.

The seven candidates of Table 5 come from survival associations in unselected
breast cancer, while the disease target of the project is triple-negative breast
cancer (TNBC). This script asks a cell-line question: over the 12 TNBC lines of
PRISM, are the candidates more active than the rest of the screen is?

HONEST PROVENANCE, READ FIRST. The design was written down before coding
(docs/plan_analisis_tnbc_lineas.md) but the primary statistic was computed during
the review of that plan, on 23 September 2026, and it was null. The commitment to
report the analysis whatever it showed was therefore taken already knowing the
result. This is not a pre-registration and the text never calls it one.

Because the design can only detect very large effects (see POWER below), the
analysis is written as an estimate with a declared bound, not as a hypothesis
test with success criteria: one set-level test, effect sizes with confidence
intervals, a minimum detectable effect, and percentiles against explicit nulls.
Words that overstate it are avoided on purpose: the analysis is not independent
of the models (the same PRISM matrix trained them, and the training set of the
seed-42 cell-line split, on which the transfer models behind the candidates were
fitted, holds 8 of the 12 TNBC lines, with the other 4 in its test set), GDSC
is not an external cohort at the cell-line level (11 of the 12 TNBC lines are
shared), and nothing here validates or confirms the clinical associations.

DATA (versions and md5 recorded in the summary CSV)
  data/PRISM/processed/drug_response_prism.csv   476 lines x 1,448 drugs, AUC,
      1 = no effect. 10.87% missing. 2.48% of the observed values sit exactly at
      1.0, the ceiling of the assay.
  data/DepMap/Model.csv                          DepMap 25Q3, lineage and
      ModelSubtypeFeatures. The 12 TNBC lines are the ones the paper already uses
      for Integrated Gradients, and the script checks that the three routes in
      the repository (src/data/breast_subtypes.py, src/data/load_depmap.py and
      the IG attribution table) give the same 12.
  data/GDSC/processed/drug_response_sanger.csv   696 x 397, AUC in [0, 1], a
      second screen with 343 lines and 187 drugs in common with PRISM.
  data/PRISM/processed/counts_matched_prism.csv  log2 expression, proliferation
      metagene of the sensitivity analysis.
  Earlier outputs it checks itself against: results/v4_xai/ig_full_attributions.csv
      (the 12 lines of Integrated Gradients), results/v2_fair/per_drug_Ridge_expression.csv
      (the 1,447-drug universe of the paper), results/crossval_prism_to_gdsc.csv
      (187 shared drugs), results/v10_reporting/iclac_matches.csv (13 listed lines)
      and results/v8_multicohort/survival_SCANB.csv (the clinical axis).

STATISTIC
  Normalization. Raw AUC is not comparable across lines (different potency) nor
  across screens (different definition), and the drug panel measured per line
  differs. Every AUC is therefore converted into its empirical percentile within
  the distribution of its own cell line over a FIXED reference panel: the drugs
  measured in at least 95% of the lines of that screen. Ties take half weight,
  which is how the ceiling at AUC = 1 enters. Subtracting the line median over
  all measured drugs, the alternative in the plan, mixes potency with panel
  coverage and is kept only as a sensitivity arm.
  Primary test, one only: the composite of the seven candidates. Per line, that
  composite is the mean over the seven of the within-drug z of the percentile.
  The composite of the 12 TNBC lines is compared with the 464 remaining lines
  with Cliff's delta, a confidence interval built from the exact null standard
  error (the interval built from the estimated one covers 0.92 instead of 0.95
  at twelve values) and a two-sided permutation p over cell-line labels (10,000
  permutations, seed 42), which preserves the correlation between drugs.
  Everything else is descriptive and decides nothing: the same set statistic for
  the five positive controls and for the four EGFR-family candidates, the
  per-drug table of the 12 focal drugs, the background table of the 1,409
  evaluable drugs, the sensitivity grid and the GDSC arms. The columns that
  carry multiplicity corrections say "descriptive" in their name.
  Direction: a lower AUC means more active, so a NEGATIVE delta means the drug is
  more active in the TNBC lines, and background_percentile ranks the signed delta
  upwards, so a LOW percentile also means more active in TNBC.

NULLS (no claim without its base rate)
  1. Label permutation over the 476 lines, which is the p of the primary test.
  2. Mechanism-matched competitive null: 10,000 sets of seven kinase or HDAC
     inhibitors drawn from the drugs of the same class (the class filter of
     Table 5, imported from src/analysis/candidate_selection.py), matched on the
     number of TNBC lines measured and on the amount of ceiling censoring.
  3. Unmatched background over the 1,409 evaluable drugs. Storey's estimator of
     the null share comes out above 1 at every lambda tried, so the unclipped
     value is reported rather than a share of non-null drugs pinned at zero.
  The binomial enrichment test of the plan was dropped: drug response profiles
  are correlated, and under permutation the count of drugs beyond the asymptotic
  5% threshold has about ten times the variance a binomial assumes (measured in
  the run, asserted against EXPECTED_VARIANCE_INFLATION).

POWER, declared before the result
  With 12 TNBC lines against 464 the null standard error of Cliff's delta is
  sqrt((n1+n2+1)/(3 n1 n2)), so the minimum detectable effect at 80% power is
  about 0.47 in delta, and only a handful of drugs in the whole screen reach
  that magnitude. The design can exclude a large TNBC preference and says
  nothing about a small one. Power is also reported against two effects fixed in
  advance: the TNBC contrast of paclitaxel among breast lines in GDSC, and delta
  0.40. Observed power is never computed.

ASSAY GATE, run before looking at the candidates
  Before interpreting any null, the script checks whether each screen recovers
  effects that are established. Every chemotherapy control is reported three
  ways, because the contrast of the primary test mixes the TNBC effect with a
  lineage effect and the two screens differ in that lineage term: TNBC against
  the other breast lines (like for like), all breast lines against the rest (the
  lineage term), and TNBC against every other line (the contrast of the primary
  test). The PARP inhibitors are checked in the two BRCA1-mutant lines of the
  panel (HCC1937, MDA-MB-436) and the EGFR inhibitors in the two EGFR-driven
  lines (MDA-MB-468, HDQ-P1). A calibration control shows what a group of twelve
  lines can detect at all: BRAF and MEK inhibitors in random draws of twelve skin
  lines, with the detection rate over 200 draws, which must reach 0.8
  (CALIBRATION_MIN_DETECT_RATE). The verdict compares the exact rate, which is
  written in its own column (detection_rate), and the note carries it rounded to
  two decimals. A row passes only when its
  interval excludes zero on the expected side, never on the sign alone, and the
  sign is kept in its own column. The lineage rows have no expected side, so they
  carry no verdict (passed is left empty) and the three counts of the summary
  (gate_n_passed, gate_n_total and gate_n_sign_negative) leave them out. The gate
  writes its verdict, never aborts, and conditions how the result is worded.

SENSITIVITY (whole grid written out, the text quotes its range, never one cell)
  S1 comparison group: all 464 non-TNBC lines, the 452 non-breast lines, the 10
     other breast lines (descriptive only, its minimum detectable effect is
     above anything observable).
  S2 normalization: reference-panel percentile, line-median subtraction, raw AUC.
  S3 censoring: values exactly at 1.0 removed.
  S4 proliferation: rank model with a 15-gene proliferation metagene as covariate
     (DepMap does not distribute doubling times for these lines).
  S6 leave-one-line-out over the 12 TNBC lines, worst-case swing.
  S7 the 13 ICLAC-listed lines removed from the comparison group.
  S5 of the plan, a lineage-stratified van Elteren test, is degenerate here: all
     12 TNBC lines are breast, so the only within-stratum contrast is S1c, and
     the arm is dropped for that reason instead of being run empty.

OUTPUTS (results/v13_reaudit/)
  tnbc_selectivity_primary.csv     the 12 focal drugs, one row each
  tnbc_selectivity_background.csv  the 1,409 evaluable drugs
  tnbc_set_test.csv                set-level tests (candidates, controls, EGFR)
  tnbc_permutation_null.csv        the 10,000 label permutations
  tnbc_moa_matched_null.csv        the 10,000 mechanism-matched sets
  tnbc_sensitivity.csv             the sensitivity grid
  tnbc_power_mde.csv               minimum detectable effect and reference power
  line_potency_offset.csv          one row per cell line
  tnbc_excluded_drugs.csv          the 39 drugs that fail the inclusion rule
  tnbc_orthogonality.csv           correlation with the clinical axis and between
                                   normalizations
  tnbc_background_by_class.csv     background by mechanism class
  assay_sensitivity_gate.csv       the gate above
  gdsc_replication.csv             the two GDSC arms
  tnbc_cell_lines_summary.csv      metric, value

The script raises on any unexpected input and never skips a drug or a line
silently: the drugs that fail the inclusion rule are counted and written out with
their reason.

Run from the project root (no training, a few minutes):
    python scripts/67_tnbc_cell_lines.py
"""

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.candidate_selection import is_kinase_or_hdac_inhibitor
from src.data.breast_subtypes import breast_subtypes
from src.data.drug_names import normalize_drug_columns, normalize_drug_name
from src.data.load_depmap import filter_tnbc_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tnbc_cell_lines")

PRISM_PATH = DATA_DIR / "PRISM/processed/drug_response_prism.csv"
GDSC_PATH = DATA_DIR / "GDSC/processed/drug_response_sanger.csv"
MODEL_PATH = DATA_DIR / "DepMap/Model.csv"
EXPR_PATH = DATA_DIR / "PRISM/processed/counts_matched_prism.csv"
ANNOT_PATH = DATA_DIR / "precision_processed/prism_drug_annotations.csv"
IG_PATH = RESULTS_DIR / "v4_xai/ig_full_attributions.csv"
ICLAC_PATH = RESULTS_DIR / "v10_reporting/iclac_matches.csv"
# Canonical copies under results/, not the ones the paper stage copies into
# paper/results/, so this script never depends on a later stage of the pipeline
UNIVERSE_PATH = RESULTS_DIR / "v2_fair/per_drug_Ridge_expression.csv"
CROSSVAL_PATH = RESULTS_DIR / "crossval_prism_to_gdsc.csv"
SURV_SCANB = RESULTS_DIR / "v8_multicohort/survival_SCANB.csv"
PLAN_PATH = Path("docs/plan_analisis_tnbc_lineas.md")
OUT_DIR = RESULTS_DIR / "v13_reaudit"

CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
DRUGS = CANDIDATES + CONTROLS
FAMILY = {**{d: "candidate" for d in CANDIDATES}, **{d: "positive_control" for d in CONTROLS}}
# EGFR family among the candidates, a declared descriptive set contrast (design A.2)
EGFR_CANDIDATES = ["erlotinib", "osimertinib", "pelitinib", "brigatinib"]

SEED = 42
N_PERM = 10_000          # label permutations, no default anywhere
N_MOA_NULL = 10_000      # mechanism-matched sets
N_POWER_SIM = 10_000     # power simulation replicates
N_CALIBRATION_DRAWS = 200  # random twelve-line groups for the calibration control
CALIBRATION_MIN_DETECT_RATE = 0.8  # share of those draws whose interval must exclude zero
ALPHA = 0.05
REF_PANEL_MIN_FRAC = 0.95    # a drug joins the PRISM reference panel if measured in >= 95% of lines
# GDSC is far sparser (32.5% missing, 397 drugs): at 95% its panel would hold three
# drugs, so the second screen gets its own declared threshold and drops the lines
# that still fall short, counted in the summary.
GDSC_REF_PANEL_MIN_FRAC = 0.70
GDSC_MIN_REF_PER_LINE = 100
MIN_TNBC = 8                 # AUC needed in at least 8 of the 12 TNBC lines
MIN_REST = 30                # and in at least 30 comparison lines
MIN_REF_PANEL_PER_LINE = 200  # a line needs this many reference drugs for its percentiles
MIN_TNBC_WITHIN_BREAST = 8   # inclusion rule of the descriptive within-breast arm
MIN_TNBC_GDSC, MIN_REST_GDSC = 6, 30
CENSOR_VALUE = 1.0           # AUC ceiling of PRISM
NOT_RESOLVABLE_FRAC = 0.25   # censoring above this in either group flags the drug
REF_DELTA_MODERATE = 0.40    # reference effect for the power statement

# Values verified on 2026-09-23 against the files above. The script refuses to run
# if the inputs no longer reproduce them, which is what stops a silent data change.
EXPECTED_PRISM_SHAPE = (476, 1448)
EXPECTED_PRISM_MIN, EXPECTED_PRISM_MAX = 0.004174, 4.889162
EXPECTED_PRISM_MEDIAN, EXPECTED_PRISM_MISSING = 0.925535, 0.108703
EXPECTED_SUBTYPES = {"TNBC": 12, "HER2+": 6, "ER+": 4}
TNBC_LINES = ("ACH-000212", "ACH-000223", "ACH-000276", "ACH-000288", "ACH-000374",
              "ACH-000573", "ACH-000624", "ACH-000643", "ACH-000721", "ACH-000768",
              "ACH-000849", "ACH-000856")
# MDA-MB-435S (ACH-000884) is in PRISM and is often called a breast line. DepMap
# labels it Skin (melanoma M14, ICLAC-00565), so it is not a TNBC line here.
MISIDENTIFIED_NOT_TNBC = "ACH-000884"
EXPECTED_N_REST = 464
EXPECTED_BREAST_NON_TNBC = 10
EXPECTED_REF_PANEL = 402
EXPECTED_EVALUABLE = 1409
EXPECTED_EXCLUDED = 39
EXPECTED_UNIVERSE = 1447
EXPECTED_GDSC_SHAPE = (696, 397)
EXPECTED_GDSC_SHARED_LINES = 343
EXPECTED_GDSC_SHARED_DRUGS = 187
EXPECTED_ICLAC_LINES = 13
# Drug response profiles are correlated across lines, so the count of nominally
# extreme drugs per permutation varies far more than a binomial would. Measured
# here, which is why the enrichment test of the design was dropped.
EXPECTED_VARIANCE_INFLATION = 9.9
VARIANCE_INFLATION_TOL = 1.5
# BRCA1-mutant and EGFR-driven lines of the TNBC panel, for the assay gate
BRCA1_MUTANT_LINES = ("ACH-000223", "ACH-000573")      # HCC1937, MDA-MB-436
EGFR_DRIVEN_LINES = ("ACH-000849", "ACH-000643")       # MDA-MB-468, HDQ-P1
PROLIFERATION_GENES = ("MKI67", "CCNB1", "CCNA2", "TOP2A", "BUB1", "PLK1", "AURKA",
                       "CDK1", "RRM2", "TYMS", "PCNA", "MCM2", "TPX2", "UBE2C", "BIRC5")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(value, expected, atol, what):
    if not np.isclose(value, expected, atol=atol, rtol=0):
        raise ValueError(f"{what} is {value!r}, expected {expected!r}")


def reference_panel(resp: pd.DataFrame, min_frac=REF_PANEL_MIN_FRAC) -> list:
    """Drugs measured in at least `min_frac` of the lines of a screen."""
    coverage = resp.notna().mean(axis=0)
    return sorted(coverage.index[coverage >= min_frac])


def ecdf_percentiles(resp: pd.DataFrame, panel: list, min_ref=MIN_REF_PANEL_PER_LINE,
                     drop_short_lines=False):
    """AUC as its percentile inside the distribution of its own line over `panel`.

    Ties take half weight, so the values sitting at the ceiling of the assay do
    not all collapse onto the same extreme percentile. A drug that belongs to the
    panel is compared against itself as one tie, which costs 0.5 / n_panel.
    A line with fewer than `min_ref` reference drugs has no reliable distribution:
    in PRISM that raises, and in the sparser GDSC the line is dropped and counted.
    """
    values = resp.to_numpy(dtype=float)
    panel_idx = [resp.columns.get_loc(d) for d in panel]
    out = np.full(values.shape, np.nan)
    dropped = []
    for i in range(values.shape[0]):
        ref = values[i, panel_idx]
        ref = np.sort(ref[~np.isnan(ref)])
        if ref.size < min_ref:
            if not drop_short_lines:
                raise ValueError(f"Line {resp.index[i]} has {ref.size} reference drugs, "
                                 f"below the minimum of {min_ref}")
            dropped.append(resp.index[i])
            continue
        row = values[i]
        obs = ~np.isnan(row)
        left = np.searchsorted(ref, row[obs], side="left")
        right = np.searchsorted(ref, row[obs], side="right")
        out[i, obs] = (left + 0.5 * (right - left)) / ref.size
    frame = pd.DataFrame(out, index=resp.index, columns=resp.columns)
    return (frame, dropped) if drop_short_lines else frame


def midrank_matrix(values: pd.DataFrame):
    """Within-drug midranks of the observed values, plus the observation mask.

    Ranks do not depend on the group labels, so the same two matrices serve every
    permutation: with an indicator vector g, the rank sum of the group is g @ R
    and its size is g @ M.
    """
    arr = values.to_numpy(dtype=float)
    mask = ~np.isnan(arr)
    ranks = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        obs = mask[:, j]
        if obs.sum() == 0:
            continue
        ranks[obs, j] = stats.rankdata(col[obs], method="average")
    return ranks, mask.astype(float)


def delta_from_ranks(rank_sum, n1, n2):
    """Cliff's delta from the rank sum of group 1 (ties count as half)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = rank_sum - n1 * (n1 + 1) / 2.0
        delta = 2.0 * u1 / (n1 * n2) - 1.0
    delta = np.where((n1 > 0) & (n2 > 0), delta, np.nan)
    return delta


def cliff_delta_ci(x, y, alpha=ALPHA):
    """Cliff's delta with the consistent variance estimator (Cliff 1993).

    Returns delta, the two-sided confidence interval and the standard error. The
    interval is the usual symmetric normal one on the estimator's scale, which is
    what the effsize implementation reports, and it is preferred here over a
    bootstrap because 12 values make a bootstrap of a rank statistic granular.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan
    sign = np.sign(x[:, None] - y[None, :])
    delta = float(sign.mean())
    di = sign.mean(axis=1)
    dj = sign.mean(axis=0)
    s_di = float(((di - delta) ** 2).sum() / (n1 - 1))
    s_dj = float(((dj - delta) ** 2).sum() / (n2 - 1))
    s_dij = float(((sign - delta) ** 2).sum() / ((n1 - 1) * (n2 - 1)))
    var = ((n2 - 1) * s_di + (n1 - 1) * s_dj + s_dij) / (n1 * n2)
    se_est = float(np.sqrt(max(var, 0.0)))
    # The interval uses the exact null standard error, not the estimated one. With
    # twelve values the estimated variance is itself noisy and the interval built
    # from it covers 0.92 instead of 0.95 in simulation, which would understate the
    # bound of a null result. The null standard error restores 0.95 coverage. The
    # estimated one is still written to the CSV as delta_se.
    se_null = float(np.sqrt((n1 + n2 + 1) / (3.0 * n1 * n2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return (delta, float(np.clip(delta - z * se_null, -1, 1)),
            float(np.clip(delta + z * se_null, -1, 1)), se_est)


def hodges_lehmann(x, y, alpha=ALPHA):
    """Hodges-Lehmann shift (group 1 minus group 2) with a distribution-free CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    diffs = np.sort((x[:, None] - y[None, :]).ravel())
    hl = float(np.median(diffs))
    z = stats.norm.ppf(1 - alpha / 2)
    k = n1 * n2 / 2.0 - z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    k = int(np.floor(k))
    if k < 0:
        return hl, float(diffs[0]), float(diffs[-1])
    lo = diffs[min(k, diffs.size - 1)]
    hi = diffs[max(diffs.size - 1 - k, 0)]
    return hl, float(lo), float(hi)


def permutation_deltas(ranks, mask, group_pos, rng, n_perm):
    """Observed delta per column and its null distribution under label permutation.

    `group_pos` are the row positions of the group of interest. Every replicate
    draws a group of the same size from all rows. A column where a replicate
    leaves the group or the comparison empty gives NaN and is dropped from the p
    of that column only, with the count returned.
    """
    n_lines, n_cols = ranks.shape
    n_group = len(group_pos)
    indicator = np.zeros(n_lines)
    indicator[list(group_pos)] = 1.0
    n1_obs = indicator @ mask
    n_tot = mask.sum(axis=0)
    obs = delta_from_ranks(indicator @ ranks, n1_obs, n_tot - n1_obs)

    null = np.empty((n_perm, n_cols), dtype=np.float32)
    block = 500
    for start in range(0, n_perm, block):
        stop = min(start + block, n_perm)
        ind = np.zeros((stop - start, n_lines))
        for r in range(stop - start):
            ind[r, rng.choice(n_lines, size=n_group, replace=False)] = 1.0
        n1 = ind @ mask
        null[start:stop] = delta_from_ranks(ind @ ranks, n1, n_tot - n1).astype(np.float32)
    valid = ~np.isnan(null)
    extreme = (np.abs(null) >= np.abs(obs)[None, :] - 1e-12) & valid
    n_valid = valid.sum(axis=0)
    p = (1.0 + extreme.sum(axis=0)) / (1.0 + n_valid)
    p = np.where(n_valid > 0, p, np.nan)
    return obs, p, null, n_valid


def mc_se(p, n):
    """Monte Carlo standard error of a permutation p-value."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(p * (1 - p) / np.maximum(n, 1))


def group_stats(values: pd.DataFrame, group, rest, drugs):
    """Per-drug effect sizes of `group` against `rest`, without any permutation."""
    rows = []
    for drug in drugs:
        col = values[drug]
        x = col.loc[group].dropna().to_numpy()
        y = col.loc[rest].dropna().to_numpy()
        delta, lo, hi, se = cliff_delta_ci(x, y)
        hl, hl_lo, hl_hi = hodges_lehmann(x, y)
        if x.size >= 2 and y.size >= 2:
            u = stats.mannwhitneyu(x, y, alternative="two-sided",
                                   method="asymptotic", use_continuity=True)
            u_stat, p_asym = float(u.statistic), float(u.pvalue)
        else:
            u_stat, p_asym = np.nan, np.nan
        rows.append({"drug": drug, "n_group": x.size, "n_rest": y.size,
                     "median_group": float(np.median(x)) if x.size else np.nan,
                     "median_rest": float(np.median(y)) if y.size else np.nan,
                     "median_diff": (float(np.median(x) - np.median(y))
                                     if x.size and y.size else np.nan),
                     "hodges_lehmann": hl, "hl_ci_low": hl_lo, "hl_ci_high": hl_hi,
                     "cliffs_delta": delta, "delta_ci_low": lo, "delta_ci_high": hi,
                     "delta_se": se, "u_stat": u_stat, "p_asymptotic": p_asym})
    return pd.DataFrame(rows)


def mde_delta(n1, n2, alpha=ALPHA, power=0.80):
    """Minimum detectable Cliff's delta, from the null standard error of the statistic."""
    se = np.sqrt((n1 + n2 + 1) / (3.0 * n1 * n2))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se), float(se)


def delta_to_normal_shift(delta):
    """Location shift of two normals that produces a given Cliff's delta."""
    return float(np.sqrt(2) * stats.norm.ppf((delta + 1) / 2))


def simulated_power(n1, n2, delta, rng, n_sim=N_POWER_SIM, alpha=ALPHA):
    """Power of the two-sided rank test at a given delta, with the real group sizes."""
    if n1 < 2 or n2 < 2:
        return np.nan
    shift = delta_to_normal_shift(delta)
    total = n1 + n2
    sample = rng.standard_normal((n_sim, total))
    sample[:, :n1] += shift
    order = np.argsort(sample, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n_sim)[:, None]
    ranks[rows, order] = np.arange(1, total + 1)[None, :]
    r1 = ranks[:, :n1].sum(axis=1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (total + 1) / 12.0)
    z = (np.abs(u1 - mu) - 0.5) / sd
    p = 2 * stats.norm.sf(z)
    return float((p < alpha).mean())


# ----------------------------------------------------------------------------
# loading and reproduction checks
# ----------------------------------------------------------------------------
def load_inputs():
    """Load every input and raise unless it reproduces the values frozen above."""
    raw = pd.read_csv(PRISM_PATH, index_col=0)
    if raw.shape != EXPECTED_PRISM_SHAPE:
        raise ValueError(f"PRISM response is {raw.shape}, expected {EXPECTED_PRISM_SHAPE}")
    arr = raw.to_numpy(dtype=float)
    _close(np.nanmin(arr), EXPECTED_PRISM_MIN, 5e-5, "PRISM minimum AUC")
    _close(np.nanmax(arr), EXPECTED_PRISM_MAX, 5e-5, "PRISM maximum AUC")
    _close(np.nanmedian(arr), EXPECTED_PRISM_MEDIAN, 5e-5, "PRISM median AUC")
    _close(np.isnan(arr).mean(), EXPECTED_PRISM_MISSING, 5e-5, "PRISM missing fraction")
    prism = normalize_drug_columns(raw.copy())
    if prism.shape != EXPECTED_PRISM_SHAPE:
        raise ValueError("Normalizing the PRISM drug names collided two columns")

    model = pd.read_csv(MODEL_PATH, index_col=0)
    subtypes = breast_subtypes(model, prism.index)
    counts = subtypes.value_counts().to_dict()
    if {k: counts.get(k, 0) for k in EXPECTED_SUBTYPES} != EXPECTED_SUBTYPES:
        raise ValueError(f"Breast subtypes in PRISM are {counts}, expected {EXPECTED_SUBTYPES}")
    tnbc = tuple(sorted(subtypes[subtypes == "TNBC"].index))
    if tnbc != TNBC_LINES:
        raise ValueError(f"TNBC lines are {tnbc}, expected {TNBC_LINES}")
    route2 = tuple(sorted(set(filter_tnbc_lines(model)) & set(prism.index)))
    if route2 != TNBC_LINES:
        raise ValueError("src/data/load_depmap.filter_tnbc_lines and breast_subtypes disagree: "
                         f"{sorted(set(route2) ^ set(TNBC_LINES))}")
    ig_lines = tuple(sorted(pd.read_csv(IG_PATH, usecols=["cell_depmap_id"])
                            ["cell_depmap_id"].unique()))
    if ig_lines != TNBC_LINES:
        raise ValueError("The TNBC lines differ from the ones used for Integrated Gradients: "
                         f"{sorted(set(ig_lines) ^ set(TNBC_LINES))}")
    if MISIDENTIFIED_NOT_TNBC in TNBC_LINES:
        raise ValueError("MDA-MB-435S is a melanoma line and must not be a TNBC line")

    universe = pd.read_csv(UNIVERSE_PATH)
    universe_drugs = sorted(normalize_drug_name(d) for d in universe["drug"])
    if len(universe_drugs) != EXPECTED_UNIVERSE:
        raise ValueError(f"The paper universe has {len(universe_drugs)} drugs, "
                         f"expected {EXPECTED_UNIVERSE}")
    missing_focal = [d for d in DRUGS if d not in prism.columns]
    if missing_focal:
        raise ValueError(f"Focal drugs missing from PRISM: {missing_focal}")

    gdsc_raw = pd.read_csv(GDSC_PATH, index_col=0)
    if gdsc_raw.shape != EXPECTED_GDSC_SHAPE:
        raise ValueError(f"GDSC response is {gdsc_raw.shape}, expected {EXPECTED_GDSC_SHAPE}")
    gdsc = normalize_drug_columns(gdsc_raw.copy())
    gmin, gmax = float(np.nanmin(gdsc.to_numpy())), float(np.nanmax(gdsc.to_numpy()))
    if gmin < 0 or gmax > 1:
        raise ValueError(f"GDSC AUC out of [0, 1]: [{gmin}, {gmax}]")
    shared_lines = sorted(set(prism.index) & set(gdsc.index))
    shared_drugs = sorted(set(prism.columns) & set(gdsc.columns))
    if len(shared_lines) != EXPECTED_GDSC_SHARED_LINES:
        raise ValueError(f"{len(shared_lines)} shared cell lines, expected {EXPECTED_GDSC_SHARED_LINES}")
    if len(shared_drugs) != EXPECTED_GDSC_SHARED_DRUGS:
        raise ValueError(f"{len(shared_drugs)} shared drugs, expected {EXPECTED_GDSC_SHARED_DRUGS}")
    crossval = pd.read_csv(CROSSVAL_PATH)
    if len(crossval) != EXPECTED_GDSC_SHARED_DRUGS:
        raise ValueError(f"crossval_prism_to_gdsc.csv has {len(crossval)} rows, "
                         f"expected {EXPECTED_GDSC_SHARED_DRUGS}")

    annot = pd.read_csv(ANNOT_PATH)
    annot["drug"] = annot["name"].map(normalize_drug_name)
    annot = annot.drop_duplicates("drug").set_index("drug")
    unannotated = [d for d in DRUGS if d not in annot.index]
    if unannotated:
        raise ValueError(f"Focal drugs without a PRISM annotation: {unannotated}")

    iclac = pd.read_csv(ICLAC_PATH)
    if len(iclac) != EXPECTED_ICLAC_LINES:
        raise ValueError(f"iclac_matches.csv has {len(iclac)} rows, expected {EXPECTED_ICLAC_LINES}")

    expr = pd.read_csv(EXPR_PATH, index_col=0)
    missing_genes = [g for g in PROLIFERATION_GENES if g not in expr.columns]
    if missing_genes:
        raise ValueError(f"Proliferation genes missing from the expression matrix: {missing_genes}")
    if float(expr.to_numpy().max()) > 25:
        raise ValueError("The expression matrix does not look like log2")

    scanb = pd.read_csv(SURV_SCANB)
    scanb["drug"] = scanb["drug"].map(normalize_drug_name)
    scanb = scanb.drop_duplicates("drug").set_index("drug")

    logger.info("Inputs reproduce every frozen value")
    return prism, model, subtypes, gdsc, annot, iclac, expr, scanb, universe_drugs


# ----------------------------------------------------------------------------
# analysis blocks
# ----------------------------------------------------------------------------
def build_line_table(prism, model, subtypes, panel, iclac, expr):
    """One row per cell line: group, coverage, potency offset, proliferation, ICLAC."""
    prolif = expr[list(PROLIFERATION_GENES)]
    prolif_z = (prolif - prolif.mean()) / prolif.std(ddof=1)
    score = prolif_z.mean(axis=1)
    subtype_full = pd.Series("non_breast", index=prism.index, dtype=object)
    breast = model.loc[[c for c in prism.index if c in model.index], "OncotreeLineage"]
    subtype_full.loc[breast.index[breast.fillna("").str.lower() == "breast"]] = "breast_other"
    subtype_full.loc[subtypes.index] = subtypes.values
    features = model.reindex(prism.index)["ModelSubtypeFeatures"].fillna("")
    basal = features.str.extract(r"(basal_[AB])", expand=False).fillna("")
    rows = pd.DataFrame({
        "cell": prism.index,
        "cell_name": model.reindex(prism.index)["CellLineName"].to_numpy(),
        "lineage": model.reindex(prism.index)["OncotreeLineage"].to_numpy(),
        "subtype": subtype_full.reindex(prism.index).to_numpy(),
        "subtype_features": features.to_numpy(),
        "basal_subclass": basal.to_numpy(),
        "n_drugs_measured": prism.notna().sum(axis=1).to_numpy(),
        "n_ref_drugs": prism[panel].notna().sum(axis=1).to_numpy(),
        "median_all": prism.median(axis=1).to_numpy(),
        "median_ref": prism[panel].median(axis=1).to_numpy(),
        "prolif_score": score.reindex(prism.index).to_numpy(),
        "iclac_flag": prism.index.isin(iclac["ModelID"]),
    }).set_index("cell")
    rows["is_tnbc"] = rows.index.isin(TNBC_LINES)
    rows["median_all_minus_ref"] = rows["median_all"] - rows["median_ref"]
    if rows["prolif_score"].isna().any():
        raise ValueError("A cell line has no proliferation score")
    return rows


def evaluable_drugs(prism):
    """Drugs with AUC in at least MIN_TNBC TNBC lines and MIN_REST comparison lines."""
    rest = [c for c in prism.index if c not in TNBC_LINES]
    n_tnbc = prism.loc[list(TNBC_LINES)].notna().sum(axis=0)
    n_rest = prism.loc[rest].notna().sum(axis=0)
    keep = (n_tnbc >= MIN_TNBC) & (n_rest >= MIN_REST)
    excluded = pd.DataFrame({
        "drug": sorted(keep.index[~keep]),
        "n_tnbc": n_tnbc[~keep].reindex(sorted(keep.index[~keep])).to_numpy(),
        "n_rest": n_rest[~keep].reindex(sorted(keep.index[~keep])).to_numpy(),
    })
    excluded["reason"] = np.where(excluded["n_tnbc"] < MIN_TNBC,
                                  f"fewer than {MIN_TNBC} TNBC lines measured",
                                  f"fewer than {MIN_REST} comparison lines measured")
    return sorted(keep.index[keep]), excluded, rest


def censoring_fractions(prism, drugs, rest):
    """Share of the observed AUC sitting exactly at the ceiling, per group."""
    tn = prism.loc[list(TNBC_LINES), drugs]
    ot = prism.loc[rest, drugs]
    frac_t = (tn == CENSOR_VALUE).sum(axis=0) / tn.notna().sum(axis=0)
    frac_r = (ot == CENSOR_VALUE).sum(axis=0) / ot.notna().sum(axis=0)
    return frac_t, frac_r


def missingness_fisher(prism, drugs, rest):
    """Per-drug test of differential missingness between TNBC lines and the rest."""
    tn = prism.loc[list(TNBC_LINES), drugs].notna().sum(axis=0)
    ot = prism.loc[rest, drugs].notna().sum(axis=0)
    n_t, n_r = len(TNBC_LINES), len(rest)
    p = [float(stats.fisher_exact([[int(tn[d]), n_t - int(tn[d])],
                                   [int(ot[d]), n_r - int(ot[d])]])[1]) for d in drugs]
    p = pd.Series(p, index=drugs)
    return p, pd.Series(stats.false_discovery_control(p.to_numpy()), index=drugs)


def composite_scores(pct, drugs):
    """Per line, the mean over `drugs` of the within-drug z of the percentile."""
    sub = pct[drugs]
    z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=1)
    enough = z.notna().sum(axis=1) >= max(2, len(drugs) // 2)
    out = z.mean(axis=1, skipna=True)
    return out.where(enough)


def set_test(values: pd.Series, rng, n_perm=N_PERM):
    """Cliff's delta of a per-line score between TNBC lines and the rest, by permutation.

    The null is drawn over the midranks of the composite, which is the same
    statistic as the pairwise sign average and makes 10,000 permutations cheap.
    """
    obs_lines = values.dropna()
    group = [c for c in obs_lines.index if c in TNBC_LINES]
    rest = [c for c in obs_lines.index if c not in TNBC_LINES]
    x = obs_lines.loc[group].to_numpy()
    y = obs_lines.loc[rest].to_numpy()
    delta, lo, hi, se = cliff_delta_ci(x, y)
    pooled_ranks = stats.rankdata(obs_lines.to_numpy(), method="average")
    n_total, n1 = pooled_ranks.size, len(group)
    n2 = n_total - n1
    null = np.empty(n_perm)
    for b in range(n_perm):
        sel = rng.choice(n_total, size=n1, replace=False)
        null[b] = delta_from_ranks(pooled_ranks[sel].sum(), n1, n2)
    delta_rank = float(delta_from_ranks(
        pooled_ranks[[obs_lines.index.get_loc(c) for c in group]].sum(), n1, n2))
    if abs(delta_rank - delta) > 1e-9:
        raise ValueError(f"Set-level delta differs between routes: {delta_rank} and {delta}")
    p = (1.0 + int((np.abs(null) >= abs(delta) - 1e-12).sum())) / (1.0 + n_perm)
    return {"n_lines": int(n_total), "n_tnbc": n1, "cliffs_delta": delta,
            "ci_low": lo, "ci_high": hi, "delta_se": se, "p_perm": p,
            "mc_se": float(mc_se(p, n_perm)), "n_perm": n_perm,
            "null_sd": float(null.std(ddof=1)),
            "null_p95_abs": float(np.percentile(np.abs(null), 95))}, null


def moa_matched_null(background: pd.DataFrame, annot, rng, n_sets=N_MOA_NULL):
    """Sets of seven kinase or HDAC inhibitors matched on coverage and censoring.

    Uses the class filter of Table 5 (src/analysis/candidate_selection.py), never a
    new regular expression, so the class of the null is the class of the candidates.
    """
    bg = background.copy()
    bg["moa"] = annot.reindex(bg.index)["moa"].fillna("")
    bg["is_class"] = bg["moa"].map(is_kinase_or_hdac_inhibitor)
    pool = bg[bg["is_class"]].copy()
    not_class = [d for d in CANDIDATES if not bool(bg.loc[d, "is_class"])]
    if not_class:
        raise ValueError(f"Candidates outside the kinase or HDAC class: {not_class}")
    cens_edges = pool["frac_censored_tnbc"].quantile([1 / 3, 2 / 3]).to_numpy()
    def stratum(row):
        return (int(row["n_tnbc"]), int(np.searchsorted(cens_edges, row["frac_censored_tnbc"])))
    pool["stratum"] = [stratum(r) for _, r in pool.iterrows()]
    strata = {s: sorted(set(g.index) - set(CANDIDATES)) for s, g in pool.groupby("stratum")}
    coverage_only = {n: sorted(set(g.index) - set(CANDIDATES))
                     for n, g in pool.groupby(pool["n_tnbc"].astype(int))}
    target = [stratum(bg.loc[d]) for d in CANDIDATES]
    observed = float(bg.loc[CANDIDATES, "cliffs_delta"].mean())
    widened = 0
    draws = np.empty(n_sets)
    matched_on = []
    for b in range(n_sets):
        picked, how = [], "stratum"
        for st in target:
            options = strata.get(st, [])
            if len(options) < 5:
                options = coverage_only.get(st[0], [])
                how = "coverage_only"
            if len(options) < 5:
                options = sorted(set(pool.index) - set(CANDIDATES))
                how = "class_only"
            choice = options[int(rng.integers(len(options)))]
            tries = 0
            while choice in picked and tries < 50:
                choice = options[int(rng.integers(len(options)))]
                tries += 1
            picked.append(choice)
        widened += how != "stratum"
        draws[b] = float(bg.loc[picked, "cliffs_delta"].mean())
        matched_on.append(how)
    p = (1.0 + int((np.abs(draws - draws.mean()) >= abs(observed - draws.mean()) - 1e-12).sum())) / (1.0 + n_sets)
    null_df = pd.DataFrame({"replicate": np.arange(n_sets), "mean_signed_delta": draws,
                            "matched_on": matched_on})
    summary = {"pool_size": int(len(pool)), "observed_mean_delta": observed,
               "null_mean": float(draws.mean()), "null_sd": float(draws.std(ddof=1)),
               "p_two_sided": float(p), "mc_se": float(mc_se(p, n_sets)),
               "n_sets": n_sets, "n_sets_widened": int(widened)}
    return null_df, summary, pool


def _gate_calibration_rate(gate: pd.DataFrame) -> float:
    """Lowest detection rate of the calibration control, the twelve-line draws of skin
    lines, exact: read from its own column and not from the note, which rounds it."""
    rows = gate[gate["contrast"] == "skin12_vs_rest_draws"]
    if rows.empty:
        raise ValueError("The calibration control produced no rows")
    return float(rows["detection_rate"].min())


def _gate_row(values, group, others, drug, screen, contrast, criterion, note,
              expects_direction=True):
    """One gate row with its effect size, interval and the effect it could detect.

    A row without an expected direction (the lineage term) is context and gets no
    verdict: passed is left empty, and the counts of the gate leave it out.
    """
    if drug not in values.columns or len(group) < 2 or len(others) < 2:
        return {"control": drug, "screen": screen, "contrast": contrast,
                "criterion": criterion, "passed": False if expects_direction else np.nan,
                "sign_negative": False,
                "note": note or "drug or group absent from this screen"}
    st = group_stats(values, group, others, [drug]).iloc[0]
    n1, n2 = int(st["n_group"]), int(st["n_rest"])
    mde = mde_delta(n1, n2)[0] if n1 >= 2 and n2 >= 2 else np.nan
    negative = bool(st["cliffs_delta"] < 0)
    # A sign alone is a coin flip, so a row passes only when the interval excludes
    # zero on the expected side. The sign is kept in its own column.
    passed = bool(st["delta_ci_high"] < 0) if expects_direction else np.nan
    return {"control": drug, "screen": screen, "contrast": contrast,
            "n_group": n1, "n_rest": n2,
            "cliffs_delta": st["cliffs_delta"], "ci_low": st["delta_ci_low"],
            "ci_high": st["delta_ci_high"], "p": st["p_asymptotic"],
            "mde_delta_80": mde, "criterion": criterion,
            "sign_negative": negative, "passed": passed, "note": note}


def assay_gate(pct_prism, pct_gdsc, line_table, gdsc_subtypes, rest, rng):
    """Can each screen recover effects that are established, before reading candidates.

    The contrast of the primary test, TNBC against every other line, mixes the TNBC
    effect with a lineage effect, and the two screens differ in that lineage term:
    paclitaxel is broadly active in PRISM and broadly inactive in GDSC, so breast
    lines look resistant in one panel and sensitive in the other. Each control is
    therefore reported three ways, TNBC against the other breast lines (the
    like-for-like contrast), all breast lines against the rest (the lineage term),
    and TNBC against every other line (the contrast of the primary test), so the
    decomposition is visible instead of being folded into one number.
    """
    breast_prism = [c for c in line_table.index if line_table.loc[c, "subtype"] != "non_breast"]
    other_breast_prism = [c for c in breast_prism if c not in TNBC_LINES]
    non_breast_prism = [c for c in line_table.index if line_table.loc[c, "subtype"] == "non_breast"]
    tnbc_gdsc = sorted(gdsc_subtypes[gdsc_subtypes == "TNBC"].index)
    breast_gdsc = sorted(gdsc_subtypes.index)
    other_breast_gdsc = [c for c in breast_gdsc if c not in tnbc_gdsc]
    rest_gdsc = [c for c in pct_gdsc.index if c not in tnbc_gdsc]
    non_breast_gdsc = [c for c in pct_gdsc.index if c not in breast_gdsc]

    rows = []
    for drug in ("paclitaxel", "docetaxel", "epirubicin"):
        for screen, values, tnbc, other_breast, all_rest, breast, non_breast in (
                ("PRISM", pct_prism, list(TNBC_LINES), other_breast_prism, rest,
                 breast_prism, non_breast_prism),
                ("GDSC", pct_gdsc, tnbc_gdsc, other_breast_gdsc, rest_gdsc,
                 breast_gdsc, non_breast_gdsc)):
            rows.append(_gate_row(values, tnbc, other_breast, drug, screen,
                                  "tnbc_vs_other_breast",
                                  "delta < 0 with an interval excluding zero", ""))
            rows.append(_gate_row(values, breast, non_breast, drug, screen,
                                  "breast_vs_non_breast",
                                  "lineage term, no direction expected",
                                  "context for the contrast of the primary test",
                                  expects_direction=False))
            rows.append(_gate_row(values, tnbc, all_rest, drug, screen,
                                  "tnbc_vs_all_other_lines",
                                  "delta < 0 with an interval excluding zero",
                                  "contrast of the primary test, carries the lineage term"))
    # PARP inhibitors inside the panel: the two BRCA1-mutant lines against the other ten
    # TNBC lines of the panel, of which the comparison takes those measured for the
    # drug (n_rest of the row)
    others = [c for c in TNBC_LINES if c not in BRCA1_MUTANT_LINES]
    for drug in ("olaparib", "talazoparib"):
        rows.append(_gate_row(pct_prism, list(BRCA1_MUTANT_LINES), others, drug, "PRISM",
                              "brca1_mutant_vs_other_tnbc",
                              "delta < 0 in HCC1937 and MDA-MB-436 against the other TNBC lines measured",
                              "two lines against the other TNBC lines, descriptive"))
    # EGFR inhibitors in the two EGFR-driven lines against the other TNBC lines, again
    # those measured for the drug
    others = [c for c in TNBC_LINES if c not in EGFR_DRIVEN_LINES]
    for drug in ("erlotinib", "osimertinib", "pelitinib"):
        rows.append(_gate_row(pct_prism, list(EGFR_DRIVEN_LINES), others, drug, "PRISM",
                              "egfr_driven_vs_other_tnbc",
                              "delta < 0 in MDA-MB-468 and HDQ-P1 against the other TNBC lines measured",
                              "two lines against the other TNBC lines, descriptive"))
    # Calibration at the size of the analysis: can a group of twelve lines detect a
    # lineage effect that is beyond doubt? BRAF inhibitors in skin lines, with the
    # group drawn at random to match the twelve TNBC lines.
    skin = sorted(c for c in line_table.index if line_table.loc[c, "lineage"] == "Skin")
    non_skin = [c for c in line_table.index if c not in skin]
    for drug in ("dabrafenib", "vemurafenib", "trametinib"):
        if drug not in pct_prism.columns:
            continue
        rows.append(_gate_row(pct_prism, skin, non_skin, drug, "PRISM",
                              "skin_vs_rest_all",
                              "delta < 0, a lineage effect that is beyond doubt",
                              f"calibration, all {len(skin)} skin lines"))
        draws = []
        for _ in range(N_CALIBRATION_DRAWS):
            sample = [skin[i] for i in rng.choice(len(skin), size=len(TNBC_LINES), replace=False)]
            st = group_stats(pct_prism, sample, non_skin, [drug]).iloc[0]
            draws.append((st["cliffs_delta"], st["delta_ci_high"] < 0))
        deltas = np.array([d for d, _ in draws])
        detected = float(np.mean([bool(ok) for _, ok in draws]))
        rows.append({"control": drug, "screen": "PRISM", "contrast": "skin12_vs_rest_draws",
                     "n_group": len(TNBC_LINES), "n_rest": len(non_skin),
                     "cliffs_delta": float(np.median(deltas)),
                     "ci_low": float(np.percentile(deltas, 2.5)),
                     "ci_high": float(np.percentile(deltas, 97.5)),
                     "p": np.nan, "mde_delta_80": mde_delta(len(TNBC_LINES), len(non_skin))[0],
                     "criterion": f"detection rate over {N_CALIBRATION_DRAWS} random draws of twelve skin lines",
                     "sign_negative": bool(np.median(deltas) < 0),
                     "passed": detected >= CALIBRATION_MIN_DETECT_RATE,
                     "note": f"detection rate {detected:.2f}",
                     # The exact rate the verdict compares, the note rounds it
                     "detection_rate": detected})
    return pd.DataFrame(rows)


def sensitivity_grid(prism, pct, line_table, rng):
    """The pre-declared sensitivity arms, all of them, for the twelve focal drugs."""
    rest_all = [c for c in prism.index if c not in TNBC_LINES]
    non_breast = [c for c in rest_all if line_table.loc[c, "subtype"] == "non_breast"]
    breast_other = [c for c in rest_all if line_table.loc[c, "subtype"] != "non_breast"]
    iclac_free = [c for c in rest_all if not bool(line_table.loc[c, "iclac_flag"])]
    median_adj = prism.sub(prism.median(axis=1), axis=0)
    censored = prism.where(prism != CENSOR_VALUE)
    pct_censored = pct.where(prism != CENSOR_VALUE)

    arms = [
        ("S1a", "comparison group: all non-TNBC lines (primary)", pct, list(TNBC_LINES), rest_all),
        ("S1b", "comparison group: non-breast lines only", pct, list(TNBC_LINES), non_breast),
        ("S1c", "comparison group: the other breast lines (descriptive, underpowered)",
         pct, list(TNBC_LINES), breast_other),
        ("S2a", "normalization: reference-panel percentile (primary)", pct, list(TNBC_LINES), rest_all),
        ("S2b", "normalization: line median subtracted", median_adj, list(TNBC_LINES), rest_all),
        ("S2c", "normalization: raw AUC", prism, list(TNBC_LINES), rest_all),
        ("S3", "ceiling values (AUC = 1) removed", pct_censored, list(TNBC_LINES), rest_all),
        ("S7", "ICLAC-listed lines removed from the comparison group", pct, list(TNBC_LINES), iclac_free),
    ]
    rows = []
    for arm, note, values, group, others in arms:
        stats_df = group_stats(values, group, others, DRUGS)
        ranks, mask = midrank_matrix(values.loc[group + others, DRUGS])
        positions = list(range(len(group)))
        obs, p, _, n_valid = permutation_deltas(ranks, mask, positions, rng, N_PERM)
        for i, drug in enumerate(DRUGS):
            row = stats_df.iloc[i]
            if not np.isnan(row["cliffs_delta"]) and abs(row["cliffs_delta"] - obs[i]) > 1e-9:
                raise ValueError(f"{arm}/{drug}: two routes give different deltas "
                                 f"({row['cliffs_delta']} and {obs[i]})")
            rows.append({"arm": arm, "drug": drug, "family": FAMILY[drug],
                         "cliffs_delta": row["cliffs_delta"], "ci_low": row["delta_ci_low"],
                         "ci_high": row["delta_ci_high"], "p_perm": float(p[i]),
                         "mc_se": float(mc_se(p[i], n_valid[i])),
                         "n_tnbc": int(row["n_group"]), "n_rest": int(row["n_rest"]),
                         "note": note})

    # S4, proliferation as a covariate of a rank model
    prolif_rank = stats.rankdata(line_table["prolif_score"].to_numpy()) / len(line_table)
    prolif_rank = pd.Series(prolif_rank, index=line_table.index)
    for drug in DRUGS:
        col = pct[drug].dropna()
        y = stats.rankdata(col.to_numpy()) / col.size
        x_tnbc = col.index.isin(TNBC_LINES).astype(float)
        x_prol = prolif_rank.reindex(col.index).to_numpy()
        design = np.column_stack([np.ones(col.size), x_tnbc, x_prol])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = col.size - design.shape[1]
        cov = np.linalg.pinv(design.T @ design) * float(resid @ resid) / dof
        se = float(np.sqrt(cov[1, 1]))
        t = float(beta[1] / se)
        rows.append({"arm": "S4", "drug": drug, "family": FAMILY[drug],
                     "cliffs_delta": float(2 * beta[1]),
                     "ci_low": float(2 * (beta[1] - 1.96 * se)),
                     "ci_high": float(2 * (beta[1] + 1.96 * se)),
                     "p_perm": float(2 * stats.t.sf(abs(t), dof)), "mc_se": np.nan,
                     "n_tnbc": int(x_tnbc.sum()), "n_rest": int(col.size - x_tnbc.sum()),
                     "note": "rank model adjusted for a 15-gene proliferation metagene, "
                             "delta-equivalent = twice the coefficient, p from the t test"})

    # S6, leave one TNBC line out
    for drug in DRUGS:
        deltas = []
        for left_out in TNBC_LINES:
            group = [c for c in TNBC_LINES if c != left_out]
            st = group_stats(pct, group, rest_all, [drug]).iloc[0]
            deltas.append(st["cliffs_delta"])
            rows.append({"arm": "S6", "drug": drug, "family": FAMILY[drug],
                         "cliffs_delta": st["cliffs_delta"], "ci_low": st["delta_ci_low"],
                         "ci_high": st["delta_ci_high"], "p_perm": np.nan, "mc_se": np.nan,
                         "n_tnbc": int(st["n_group"]), "n_rest": int(st["n_rest"]),
                         "note": f"leaving out {left_out}"})
    return pd.DataFrame(rows)


def gdsc_replication(pct_prism, pct_gdsc, prism, gdsc, subtypes_gdsc, background):
    """The two GDSC arms: shared material, and the lines GDSC does not share."""
    tnbc_gdsc = sorted(subtypes_gdsc[subtypes_gdsc == "TNBC"].index)
    rest_gdsc = [c for c in gdsc.index if c not in tnbc_gdsc]
    disjoint_tnbc = [c for c in tnbc_gdsc if c not in prism.index]
    disjoint_rest = [c for c in rest_gdsc if c not in prism.index]
    shared_drugs = sorted(set(prism.columns) & set(gdsc.columns))
    rows = []
    for arm, group, others in (("technical_overlap", tnbc_gdsc, rest_gdsc),
                               ("disjoint", disjoint_tnbc, disjoint_rest)):
        if len(group) < MIN_TNBC_GDSC or len(others) < MIN_REST_GDSC:
            raise ValueError(f"GDSC arm {arm} has {len(group)} TNBC and {len(others)} other lines")
        st = group_stats(pct_gdsc, group, others, shared_drugs).set_index("drug")
        for drug in shared_drugs:
            prism_delta = (float(background.loc[drug, "cliffs_delta"])
                           if drug in background.index else np.nan)
            row = st.loc[drug]
            enough = row["n_group"] >= MIN_TNBC_GDSC and row["n_rest"] >= MIN_REST_GDSC
            rows.append({"drug": drug, "arm": arm, "family": FAMILY.get(drug, "other"),
                         "cliffs_delta_prism": prism_delta,
                         "cliffs_delta_gdsc": row["cliffs_delta"] if enough else np.nan,
                         "ci_low_gdsc": row["delta_ci_low"] if enough else np.nan,
                         "ci_high_gdsc": row["delta_ci_high"] if enough else np.nan,
                         "p_gdsc": row["p_asymptotic"] if enough else np.nan,
                         "n_tnbc_gdsc": int(row["n_group"]), "n_rest_gdsc": int(row["n_rest"]),
                         "sign_agree": (bool(np.sign(prism_delta) == np.sign(row["cliffs_delta"]))
                                        if enough and not np.isnan(prism_delta) else np.nan),
                         "reason": "" if enough else "too few lines with an AUC in this arm"})
    missing = [d for d in DRUGS if d not in shared_drugs]
    for drug in missing:
        rows.append({"drug": drug, "arm": "technical_overlap", "family": FAMILY[drug],
                     "cliffs_delta_prism": float(background.loc[drug, "cliffs_delta"]),
                     "cliffs_delta_gdsc": np.nan, "reason": "not screened in GDSC"})
    return pd.DataFrame(rows), tnbc_gdsc, rest_gdsc, disjoint_tnbc, disjoint_rest, missing


# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    prism, model, subtypes, gdsc, annot, iclac, expr, scanb, universe = load_inputs()

    panel = reference_panel(prism)
    if len(panel) != EXPECTED_REF_PANEL:
        raise ValueError(f"The reference panel has {len(panel)} drugs, expected {EXPECTED_REF_PANEL}")
    pct = ecdf_percentiles(prism, panel)
    line_table = build_line_table(prism, model, subtypes, panel, iclac, expr)
    if int(line_table["is_tnbc"].sum()) != len(TNBC_LINES):
        raise ValueError("The line table lost a TNBC line")

    drugs, excluded, rest = evaluable_drugs(prism)
    if len(rest) != EXPECTED_N_REST:
        raise ValueError(f"The comparison group has {len(rest)} lines, expected {EXPECTED_N_REST}")
    n_breast_non_tnbc = int((line_table.loc[rest, "subtype"] != "non_breast").sum())
    if n_breast_non_tnbc != EXPECTED_BREAST_NON_TNBC:
        raise ValueError(f"{n_breast_non_tnbc} non-TNBC breast lines, expected {EXPECTED_BREAST_NON_TNBC}")
    if (len(drugs), len(excluded)) != (EXPECTED_EVALUABLE, EXPECTED_EXCLUDED):
        raise ValueError(f"{len(drugs)} evaluable and {len(excluded)} excluded drugs, expected "
                         f"{EXPECTED_EVALUABLE} and {EXPECTED_EXCLUDED}")
    if len(drugs) + len(excluded) != prism.shape[1]:
        raise ValueError("Evaluable and excluded drugs do not add up to the screen")
    outside = sorted(set(drugs) - set(universe))
    if outside:
        raise ValueError(f"Evaluable drugs outside the paper universe: {outside[:5]}")
    missing_focal = [d for d in DRUGS if d not in drugs]
    if missing_focal:
        raise ValueError(f"Focal drugs excluded by the inclusion rule: {missing_focal}")
    logger.info("%d evaluable drugs, %d excluded, %d comparison lines",
                len(drugs), len(excluded), len(rest))

    # ---- background over the evaluable drugs, with its permutation null
    background = group_stats(pct, list(TNBC_LINES), rest, drugs).set_index("drug")
    ranks, mask = midrank_matrix(pct[drugs])
    positions = [prism.index.get_loc(c) for c in TNBC_LINES]
    obs, p_perm, null_matrix, n_valid = permutation_deltas(ranks, mask, positions, rng, N_PERM)
    if np.nanmax(np.abs(obs - background["cliffs_delta"].to_numpy())) > 1e-9:
        raise ValueError("The rank route and the pairwise route give different deltas")
    background["p_perm"] = p_perm
    background["mc_se"] = mc_se(p_perm, n_valid)
    background["padj_bh_background_descriptive"] = stats.false_discovery_control(
        background["p_perm"].to_numpy())
    background["padj_by_background_descriptive"] = stats.false_discovery_control(
        background["p_perm"].to_numpy(), method="by")
    frac_t, frac_r = censoring_fractions(prism, drugs, rest)
    background["frac_censored_tnbc"] = frac_t
    background["frac_censored_rest"] = frac_r
    background["not_resolvable"] = ((frac_t >= NOT_RESOLVABLE_FRAC) |
                                    (frac_r >= NOT_RESOLVABLE_FRAC))
    miss_p, miss_padj = missingness_fisher(prism, drugs, rest)
    background["missing_fisher_p"] = miss_p
    background["missing_fisher_padj"] = miss_padj
    background["n_tnbc"] = background["n_group"]
    background["background_percentile"] = (
        background["cliffs_delta"].rank(pct=True) * 100).round(2)
    background["moa"] = annot.reindex(background.index)["moa"].fillna("")
    background["moa_class"] = np.where(background["moa"].map(is_kinase_or_hdac_inhibitor),
                                       "kinase_or_hdac", "other")
    background["family"] = [FAMILY.get(d, "other") for d in background.index]
    background["mde_delta_80"] = [mde_delta(int(n1), int(n2))[0] for n1, n2
                                  in zip(background["n_group"], background["n_rest"])]

    # ---- the twelve focal drugs
    primary = background.loc[DRUGS].copy()
    primary["padj_bh12_descriptive"] = stats.false_discovery_control(primary["p_perm"].to_numpy())

    # ---- set-level test, the only inferential one
    set_rows, nulls = [], {}
    for name, drug_set in (("candidates7", CANDIDATES), ("controls5", CONTROLS),
                           ("egfr4", EGFR_CANDIDATES)):
        score = composite_scores(pct, drug_set)
        res, null = set_test(score, rng)
        inter = pct[drug_set].corr(method="spearman").to_numpy()
        res.update({"set_name": name, "n_drugs": len(drug_set),
                    "mean_inter_drug_spearman": float(inter[np.triu_indices(len(drug_set), 1)].mean())})
        set_rows.append(res)
        nulls[name] = null
    set_df = pd.DataFrame(set_rows)[
        ["set_name", "n_drugs", "n_lines", "n_tnbc", "cliffs_delta", "ci_low", "ci_high",
         "delta_se", "p_perm", "mc_se", "n_perm", "null_sd", "null_p95_abs",
         "mean_inter_drug_spearman"]]

    # ---- mechanism-matched null
    moa_null, moa_summary, pool = moa_matched_null(background, annot, rng)

    # ---- power and minimum detectable effect
    gdsc_panel = reference_panel(gdsc, GDSC_REF_PANEL_MIN_FRAC)
    pct_gdsc, gdsc_dropped = ecdf_percentiles(gdsc, gdsc_panel, GDSC_MIN_REF_PER_LINE,
                                              drop_short_lines=True)
    subtypes_gdsc = breast_subtypes(model, gdsc.index)
    gdsc_rep, tnbc_gdsc, rest_gdsc, disj_t, disj_r, not_in_gdsc = gdsc_replication(
        pct, pct_gdsc, prism, gdsc, subtypes_gdsc, background)
    # ---- assay gate, run before the candidates are read
    gate = assay_gate(pct, pct_gdsc, line_table, subtypes_gdsc, rest, rng)

    # ---- power and minimum detectable effect. The reference effect is the TNBC
    # contrast of paclitaxel inside breast lines in GDSC, not its contrast against
    # the whole GDSC panel: most of that one is a breast lineage effect and using it
    # would inflate the reference.
    _ref_row = gate[(gate["control"] == "paclitaxel") & (gate["screen"] == "GDSC") &
                    (gate["contrast"] == "tnbc_vs_other_breast")]
    ref_paclitaxel_gdsc = float(_ref_row["cliffs_delta"].iloc[0])
    power_rows = []
    for drug in DRUGS:
        n1, n2 = int(background.loc[drug, "n_group"]), int(background.loc[drug, "n_rest"])
        mde, se = mde_delta(n1, n2)
        power_rows.append({
            "drug": drug, "family": FAMILY[drug], "n_tnbc": n1, "n_rest": n2,
            "se_null": se, "mde_delta_80": mde,
            "power_at_mde": simulated_power(n1, n2, mde, rng),
            "power_at_ref_paclitaxel_gdsc_within_breast":
                simulated_power(n1, n2, abs(ref_paclitaxel_gdsc), rng),
            "power_at_ref_040": simulated_power(n1, n2, REF_DELTA_MODERATE, rng)})
    power = pd.DataFrame(power_rows)

    # ---- sensitivity grid
    sens = sensitivity_grid(prism, pct, line_table, rng)

    # ---- how much this axis overlaps the clinical one, measured and not assumed
    common = sorted(set(background.index) & set(scanb.index))
    rho_hr, p_hr = stats.spearmanr(background.loc[common, "cliffs_delta"],
                                   np.log(scanb.loc[common, "HR"]))
    rho_sig, p_sig = stats.spearmanr(background.loc[common, "cliffs_delta"].abs(),
                                     -np.log10(scanb.loc[common, "padj"]))
    ortho_rows = [
        {"key": "delta_vs_scanb_loghr", "pair": "cliffs delta against log HR in SCAN-B",
         "n": len(common), "spearman_rho": float(rho_hr), "p": float(p_hr)},
        {"key": "absdelta_vs_scanb_signif",
         "pair": "absolute cliffs delta against -log10 padj in SCAN-B",
         "n": len(common), "spearman_rho": float(rho_sig), "p": float(p_sig)},
    ]
    for a, b in (("S2a", "S2b"), ("S2a", "S2c"), ("S2b", "S2c")):
        wide = sens[sens["arm"].isin([a, b])].pivot(index="drug", columns="arm", values="cliffs_delta")
        rho, pv = stats.spearmanr(wide[a], wide[b])
        ortho_rows.append({"key": f"norm_{a}_vs_{b}",
                           "pair": f"delta under {a} against {b}, the twelve focal drugs",
                           "n": int(len(wide)), "spearman_rho": float(rho), "p": float(pv)})
    ortho = pd.DataFrame(ortho_rows)

    # ---- background by mechanism class
    by_class = background.groupby("moa_class").apply(
        lambda g: pd.Series({"n_drugs": len(g),
                             "n_p05": int((g["p_perm"] < ALPHA).sum()),
                             "pct_p05": round(100 * float((g["p_perm"] < ALPHA).mean()), 2),
                             "median_abs_delta": round(float(g["cliffs_delta"].abs().median()), 4),
                             "median_delta": round(float(g["cliffs_delta"].median()), 4)}),
        include_groups=False).reset_index().rename(columns={"moa_class": "class"})

    # ---- determinism check: the same seed must give the same permutation twice
    check_a = np.random.default_rng(SEED).choice(10, size=5, replace=False)
    check_b = np.random.default_rng(SEED).choice(10, size=5, replace=False)
    if not np.array_equal(check_a, check_b):
        raise ValueError("The random generator is not deterministic")

    # ---- how correlated the drugs are: the count of nominally extreme drugs per
    # permutation, against the variance a binomial would assume. This is why the
    # enrichment test of the plan was dropped.
    thresholds = (stats.norm.ppf(1 - ALPHA / 2) *
                  np.array([mde_delta(int(a), int(b))[1] for a, b
                            in zip(background["n_group"], background["n_rest"])]))
    hits_per_perm = (np.abs(null_matrix) >= thresholds[None, :]).sum(axis=1)
    hit_rate = float(hits_per_perm.mean()) / len(background)
    binom_sd = float(np.sqrt(len(background) * hit_rate * (1 - hit_rate)))
    variance_inflation = float((hits_per_perm.std(ddof=1) / binom_sd) ** 2) if binom_sd else np.nan
    if abs(variance_inflation - EXPECTED_VARIANCE_INFLATION) > VARIANCE_INFLATION_TOL:
        raise ValueError(f"Variance inflation is {variance_inflation:.2f}, expected "
                         f"{EXPECTED_VARIANCE_INFLATION} +- {VARIANCE_INFLATION_TOL}")

    # ---- summary
    nominal_hits = int((background["p_perm"] < ALPHA).sum())
    # Storey's estimator of the share of null drugs. It comes out above 1 at every
    # lambda, which is the honest statement (the p-values are, if anything, more
    # uniform than uniform), so the unclipped value is what gets published instead
    # of a clipped pi1 of exactly zero.
    pi0_raw = {lam: float((background["p_perm"] > lam).mean()) / (1 - lam)
               for lam in (0.3, 0.5, 0.7)}
    cand_percentile = float(background.loc[CANDIDATES, "background_percentile"].mean())
    expected_direction = int((background.loc[CANDIDATES, "cliffs_delta"] < 0).sum())
    # Base rate of the same count: how often a drug of the screen points that way
    background_negative_pct = 100 * float((background["cliffs_delta"] < 0).mean())
    mde_primary, se_primary = mde_delta(len(TNBC_LINES), len(rest))
    # The lineage rows carry no verdict, so the counts of the gate leave them out
    gate_scored = gate["passed"].notna()
    summary = {
        "prism_md5": _md5(PRISM_PATH), "gdsc_md5": _md5(GDSC_PATH), "model_md5": _md5(MODEL_PATH),
        "plan_sha256": _sha256(PLAN_PATH),
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__, "pandas_version": pd.__version__,
        "seed": SEED, "n_perm": N_PERM, "n_moa_null": N_MOA_NULL, "n_power_sim": N_POWER_SIM,
        "n_tnbc_lines": len(TNBC_LINES), "n_comparison_lines": len(rest),
        "n_breast_non_tnbc": n_breast_non_tnbc,
        "n_basal_a": int((line_table.loc[list(TNBC_LINES), "basal_subclass"] == "basal_A").sum()),
        "n_basal_b": int((line_table.loc[list(TNBC_LINES), "basal_subclass"] == "basal_B").sum()),
        "n_ref_panel": len(panel),
        "ref_cov_min": int(line_table["n_ref_drugs"].min()),
        "n_lines_ref_cov_below_300": int((line_table["n_ref_drugs"] < 300).sum()),
        "n_evaluable_drugs": len(drugs), "n_excluded_drugs": len(excluded),
        "pct_censored_observed": round(100 * float((prism == CENSOR_VALUE).sum().sum() /
                                                   prism.notna().sum().sum()), 2),
        "n_drugs_not_resolvable": int(background["not_resolvable"].sum()),
        "n_missing_differential_padj05": int((background["missing_fisher_padj"] < ALPHA).sum()),
        "n_missing_differential_p05": int((background["missing_fisher_p"] < ALPHA).sum()),
        "background_n_padj05": int((background["padj_bh_background_descriptive"] < ALPHA).sum()),
        "background_n_p05": nominal_hits,
        "background_pct_p05": round(100 * nominal_hits / len(background), 2),
        "perm_hits_mean": round(float(hits_per_perm.mean()), 2),
        "perm_hits_sd": round(float(hits_per_perm.std(ddof=1)), 2),
        "perm_hits_binomial_sd": round(binom_sd, 2),
        "variance_inflation": round(variance_inflation, 2),
        "storey_pi0_lambda03": round(pi0_raw[0.3], 3),
        "storey_pi0_lambda05": round(pi0_raw[0.5], 3),
        "storey_pi0_lambda07": round(pi0_raw[0.7], 3),
        "storey_pi0_above_one_at_every_lambda": all(v >= 1 for v in pi0_raw.values()),
        "background_pct_delta_negative": round(background_negative_pct, 2),
        "max_abs_delta_panel": round(float(background["cliffs_delta"].abs().max()), 4),
        "pct_panel_above_mde": round(100 * float((background["cliffs_delta"].abs() >=
                                                  mde_primary).mean()), 2),
        "mde_delta_80": round(mde_primary, 4), "se_null_delta": round(se_primary, 4),
        "mde_delta_80_bonf12": round((stats.norm.ppf(1 - ALPHA / 24) +
                                      stats.norm.ppf(0.8)) * se_primary, 4),
        "candidate_mean_percentile": round(cand_percentile, 2),
        "n_candidates_expected_direction": expected_direction,
        "moa_pool_size": moa_summary["pool_size"],
        "moa_observed_mean_delta": round(moa_summary["observed_mean_delta"], 4),
        "moa_null_mean": round(moa_summary["null_mean"], 4),
        "moa_null_sd": round(moa_summary["null_sd"], 4),
        "moa_p_two_sided": moa_summary["p_two_sided"],
        "moa_n_sets_widened": moa_summary["n_sets_widened"],
        "gdsc_ref_panel": len(gdsc_panel),
        "gdsc_lines_dropped_low_coverage": len(gdsc_dropped),
        "gdsc_shared_lines": len(set(prism.index) & set(gdsc.index)),
        "gdsc_shared_drugs": len(set(prism.columns) & set(gdsc.columns)),
        "gdsc_tnbc_lines": len(tnbc_gdsc),
        "gdsc_tnbc_shared_with_prism": len([c for c in tnbc_gdsc if c in prism.index]),
        "gdsc_tnbc_disjoint": len(disj_t), "gdsc_rest_disjoint": len(disj_r),
        "gdsc_focal_tested": len([d for d in DRUGS if d in gdsc.columns]),
        "gdsc_focal_missing": ";".join(not_in_gdsc) or "none",
        "gate_n_passed": int(gate.loc[gate_scored, "passed"].astype(bool).sum()),
        "gate_n_total": int(gate_scored.sum()),
        "gate_n_sign_negative": int(gate.loc[gate_scored, "sign_negative"].astype(bool).sum()),
        "gate_calibration_detect_rate": _gate_calibration_rate(gate),
        "power_ref_paclitaxel_gdsc_within_breast": round(ref_paclitaxel_gdsc, 4),
        "iclac_n_in_comparison": int(line_table.loc[rest, "iclac_flag"].sum()),
        "iclac_n_among_tnbc": int(line_table.loc[list(TNBC_LINES), "iclac_flag"].sum()),
    }
    # Every contrast of every control, so the decomposition of the gate is keyed and
    # not only drawn: tnbc_vs_other_breast is the like-for-like one, and
    # tnbc_vs_all_other_lines is the contrast of the primary test, which carries the
    # lineage term of each screen.
    for row in gate.itertuples():
        if not isinstance(getattr(row, "cliffs_delta", None), float) or np.isnan(row.cliffs_delta):
            continue
        tag = f"gate_{row.control}_{row.screen.lower()}_{row.contrast}"
        summary[f"{tag}_delta"] = round(float(row.cliffs_delta), 4)
        summary[f"{tag}_ci_low"] = round(float(row.ci_low), 4)
        summary[f"{tag}_ci_high"] = round(float(row.ci_high), 4)
    for name, row in set_df.set_index("set_name").iterrows():
        summary[f"set_{name}_delta"] = round(float(row["cliffs_delta"]), 4)
        summary[f"set_{name}_ci_low"] = round(float(row["ci_low"]), 4)
        summary[f"set_{name}_ci_high"] = round(float(row["ci_high"]), 4)
        summary[f"set_{name}_p_perm"] = float(row["p_perm"])
        summary[f"set_{name}_mc_se"] = round(float(row["mc_se"]), 5)
        summary[f"set_{name}_null_sd"] = round(float(row["null_sd"]), 4)
    for drug in DRUGS:
        row = primary.loc[drug]
        summary[f"{drug}_delta"] = round(float(row["cliffs_delta"]), 4)
        summary[f"{drug}_p_perm"] = float(row["p_perm"])
        summary[f"{drug}_percentile"] = float(row["background_percentile"])
        summary[f"{drug}_frac_censored_tnbc"] = round(float(row["frac_censored_tnbc"]), 4)
    grid = sens[sens["arm"].isin(["S1a", "S1b", "S1c", "S2a", "S2b", "S2c", "S3", "S4", "S7"])]
    for drug in DRUGS:
        part = grid[grid["drug"] == drug]["cliffs_delta"]
        summary[f"{drug}_sens_min_delta"] = round(float(part.min()), 4)
        summary[f"{drug}_sens_max_delta"] = round(float(part.max()), 4)
        loo = sens[(sens["arm"] == "S6") & (sens["drug"] == drug)]["cliffs_delta"]
        summary[f"{drug}_loo_swing"] = round(float(loo.max() - loo.min()), 4)
    for row in ortho.itertuples():
        summary[f"ortho_{row.key}_rho"] = round(row.spearman_rho, 4)
        summary[f"ortho_{row.key}_p"] = float(row.p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary.reset_index().rename(columns={"index": "drug"}).to_csv(
        OUT_DIR / "tnbc_selectivity_primary.csv", index=False)
    background.reset_index().rename(columns={"index": "drug"}).to_csv(
        OUT_DIR / "tnbc_selectivity_background.csv", index=False)
    set_df.to_csv(OUT_DIR / "tnbc_set_test.csv", index=False)
    pd.DataFrame({"replicate": np.arange(N_PERM),
                  "delta_candidates7": nulls["candidates7"],
                  "delta_controls5": nulls["controls5"],
                  "delta_egfr4": nulls["egfr4"]}).to_csv(
        OUT_DIR / "tnbc_permutation_null.csv", index=False)
    moa_null.to_csv(OUT_DIR / "tnbc_moa_matched_null.csv", index=False)
    sens.to_csv(OUT_DIR / "tnbc_sensitivity.csv", index=False)
    power.to_csv(OUT_DIR / "tnbc_power_mde.csv", index=False)
    line_table.reset_index().to_csv(OUT_DIR / "line_potency_offset.csv", index=False)
    excluded.to_csv(OUT_DIR / "tnbc_excluded_drugs.csv", index=False)
    ortho.to_csv(OUT_DIR / "tnbc_orthogonality.csv", index=False)
    by_class.to_csv(OUT_DIR / "tnbc_background_by_class.csv", index=False)
    gate.to_csv(OUT_DIR / "assay_sensitivity_gate.csv", index=False)
    gdsc_rep.to_csv(OUT_DIR / "gdsc_replication.csv", index=False)
    pd.DataFrame({"metric": list(summary), "value": list(summary.values())}).to_csv(
        OUT_DIR / "tnbc_cell_lines_summary.csv", index=False)

    logger.info("Set test (candidates): delta %.3f [%.3f, %.3f], permutation p %.3f",
                set_df.loc[0, "cliffs_delta"], set_df.loc[0, "ci_low"],
                set_df.loc[0, "ci_high"], set_df.loc[0, "p_perm"])
    logger.info("Background: %d of %d drugs at BH < 0.05, %d at nominal p < 0.05 (%.2f%%)",
                summary["background_n_padj05"], len(background), nominal_hits,
                summary["background_pct_p05"])
    logger.info("Candidates sit at mean percentile %.1f, %d of 7 in the expected direction",
                cand_percentile, expected_direction)
    logger.info("Minimum detectable delta at 80%% power: %.3f, reached by %.2f%% of the panel",
                mde_primary, summary["pct_panel_above_mde"])
    logger.info("Assay gate passed %d of %d criteria", summary["gate_n_passed"], summary["gate_n_total"])
    logger.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
