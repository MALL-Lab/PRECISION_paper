#!/usr/bin/env python3
"""Background rates for the AURORA-US analyses (re-audit).

The AURORA-US results of the paper test few hypotheses and report them without
the rate at which any TF or drug passes the same filter. This script puts each
of them next to its background, on exactly the same data and definitions.

1. TF background. The paired primary versus metastasis comparison of script 36
   (patients with both a primary and a metastatic sample, samples averaged per
   patient and tissue, two-sided paired Wilcoxon signed-rank test) is run for
   every TF of the AURORA activity matrix, not only for the ten IG core TFs of
   Table 4. The script reports how many TFs pass p < 0.005 (the Bonferroni
   threshold over ten TFs used in Table 4), p < 0.05, Benjamini-Hochberg and
   Bonferroni over all TFs, the number of core TFs expected below 0.005 for a
   random set of the same size, the hypergeometric probability of at least the
   observed number, and the rank by p of each core TF among all TFs.
2. Site background. The site-stratified comparison of script 39 (metastatic
   sites with at least three biopsies in the paired patients, per-site delta
   computed on patients with a biopsy at that site, at least three patients per
   site) is run for every TF. A site is concordant when the sign of its delta
   matches the sign of the pooled paired delta of step 1, which is the rule
   generate_paper_results.py applies to the core TFs.
3. Drugs at patient level. Script 22 compares predicted AUC between primary
   and metastatic samples with an unpaired Mann-Whitney test at sample level and
   counts nominal p < 0.05. Here the same predictions are compared at patient
   level with the pairing of the TFs (paired Wilcoxon on per-patient means) and
   Benjamini-Hochberg over drugs. Low predicted AUC means predicted sensitive, so
   a negative delta (metastasis minus primary) means more active in metastasis.

Every block first reproduces published numbers and raises if they do not match:
the p-values and deltas of results/v9_aurora/aurora_tf_ig_validation.csv
(to 1e-12), the per-site deltas and p-values of aurora_tf_by_site.csv and the
per-site concordance counts derived from aurora_tf_by_site_summary.csv, and the
p-values of aurora_drug_primary_vs_meta.csv with its 782, 633 and 149 counts.

Inputs:
  results/v9_aurora/aurora_tf_activities.csv            (script 22)
  results/v9_aurora/aurora_drug_predictions.csv         (script 22)
  results/v9_aurora/aurora_drug_primary_vs_meta.csv     (script 22, reference)
  results/v9_aurora/aurora_tf_ig_validation.csv         (script 36, reference)
  results/v9_aurora/aurora_tf_by_site.csv               (script 39, reference)
  results/v9_aurora/aurora_tf_by_site_summary.csv       (script 39, reference)
  results/v4_xai/tf_importance_ig_multi_seed_stability.csv  (script 35, core TFs)
  data/AURORA-US/GSE209998_series_matrix.txt.gz         (GEO sample metadata)
Outputs in results/v13_reaudit/:
  aurora_tf_background.csv        one row per TF (tf, pvalue, rank, is_core, ...)
  aurora_site_background.csv      one row per TF with per-site deltas and p-values
  aurora_drug_paired.csv          one row per drug, sample-level and patient-level
  aurora_background_summary.csv   metric, value

Run from project root:
    python scripts/60_aurora_background.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom, mannwhitneyu, wilcoxon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aurora_background")

RESULTS = RESULTS_DIR
V9 = RESULTS / "v9_aurora"
OUT = RESULTS / "v13_reaudit"
SERIES_MATRIX = DATA_DIR / "AURORA-US" / "GSE209998_series_matrix.txt.gz"
STABILITY = RESULTS / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv"

N_CORE = 10                        # script 36 takes the head(10) of the stability table
TABLE4_THRESHOLD = 0.05 / N_CORE   # Bonferroni over the ten core TFs (Table 4)
ALPHA = 0.05
MIN_PAIRS = 3                      # scripts 36 and 39
MIN_SITE_BIOPSIES = 3              # script 39
TOL = 1e-12
REPORT_DRUGS = ["entinostat", "paclitaxel", "panobinostat"]
PUBLISHED_DRUG_COUNTS = {"n_p005": 782, "n_more_active": 633, "n_less_active": 149}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Sample parsing is taken from script 39 so that the roles, patients and sites are
# those of the paper (script 36 uses the same patient and tissue rule inline).
S39 = load_module("s39_site", ROOT / "scripts" / "39_aurora_site_stratified.py")


def sample_info(index: pd.Index) -> pd.DataFrame:
    """Patient, role (primary or metastasis from the sample ID) and GEO fields per sample."""
    geo = S39.parse_geo_series_matrix(SERIES_MATRIX)
    ids = index.astype(str)
    if all(s.startswith("AUR-P") for s in ids):
        raise ValueError("The AURORA-US sample codes are pseudonymized (AUR-Pnn-<site>), so the GEO metadata "
                         "cannot be matched. Rerun scripts/22_aurora_complementary.py from the GEO files first "
                         "(see data/README.md).")
    parsed = [S39.parse_patient_tissue(s) for s in ids]
    return pd.DataFrame({
        "patient_id": [p for p, _ in parsed],
        "role": [r for _, r in parsed],
        "site": [geo["tissue_site"].get(s, "Unknown") for s in ids],
        "disease": [geo["disease"].get(s, "") for s in ids],
    }, index=index)


def patient_means(mat: pd.DataFrame, samples: pd.Series, pids: list[str]) -> pd.DataFrame:
    """Mean of each column over the samples of each patient (samples maps sample to patient)."""
    return pd.DataFrame([mat.loc[samples.index[samples == pid]].mean() for pid in pids], index=pids)


def paired_wilcoxon(prim: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Two-sided paired Wilcoxon per column, with the rules of script 36."""
    rows = []
    for col in prim.columns:
        p_vals, m_vals = prim[col].to_numpy(), meta[col].to_numpy()
        if len(p_vals) >= MIN_PAIRS and not np.allclose(p_vals, m_vals):
            pvalue = float(wilcoxon(p_vals, m_vals, alternative="two-sided")[1])
        else:
            pvalue = np.nan
        rows.append({"feature": col, "n_paired": len(p_vals),
                     "primary_mean": float(np.mean(p_vals)),
                     "metastasis_mean": float(np.mean(m_vals)),
                     "delta": float(np.mean(m_vals - p_vals)), "pvalue": pvalue})
    return pd.DataFrame(rows).set_index("feature")


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg over the non-missing p-values."""
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    out[ok] = stats.false_discovery_control(p[ok].to_numpy())
    return out


def check(name: str, ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(f"Reproduction failed: {name} ({detail})")
    logger.info("Reproduced: %s (%s)", name, detail)


def read_summary_sections(path: Path) -> list[pd.DataFrame]:
    """Pivot sections of aurora_tf_by_site_summary.csv, parsed as generate_paper_results.py does."""
    sections = []
    for chunk in path.read_text(encoding="utf-8").split("#"):
        body = "\n".join(line for line in chunk.strip().splitlines()[1:] if line.strip()) if chunk.strip() else ""
        if body.startswith("tf,"):
            sections.append(pd.read_csv(io.StringIO(body)).set_index("tf"))
    return sections


# ---------------------------------------------------------------------------
# 1. TF background
# ---------------------------------------------------------------------------
def tf_background(tf_acts: pd.DataFrame, info: pd.DataFrame, core: list[str],
                  summary: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    prim = info[info["role"] == "primary"]
    meta = info[info["role"] == "metastasis"]
    pids = sorted(set(prim["patient_id"]) & set(meta["patient_id"]))
    logger.info("TF block: %d primary, %d metastatic samples, %d paired patients",
                len(prim), len(meta), len(pids))
    p_mean = patient_means(tf_acts, prim["patient_id"], pids)
    m_mean = patient_means(tf_acts, meta["patient_id"], pids)
    res = paired_wilcoxon(p_mean, m_mean)

    # Reproduce script 36 (Table 4) to 1e-12.
    ref = pd.read_csv(V9 / "aurora_tf_ig_validation.csv").set_index("tf")
    check("core TF set of script 36", set(ref.index) == set(core),
          f"stability head(10) {sorted(core)} vs validation file {sorted(ref.index)}")
    dp = (res.loc[ref.index, "pvalue"] - ref["pvalue"]).abs().max()
    dd = (res.loc[ref.index, "delta"] - ref["delta"]).abs().max()
    check("aurora_tf_ig_validation.csv p-values and deltas", dp <= TOL and dd <= TOL,
          f"max |dp| = {dp:.1e}, max |ddelta| = {dd:.1e}, n_paired {sorted(set(res['n_paired']))}")
    check("paired patients", len(pids) == int(ref["n_paired"].iloc[0]), f"{len(pids)} patients")

    tested = res[res["pvalue"].notna()].copy()
    n = len(tested)
    tested["padj_bh"] = bh(tested["pvalue"])
    tested["bonferroni_p"] = (tested["pvalue"] * n).clip(upper=1.0)
    tested["rank"] = tested["pvalue"].rank(method="min").astype(int)
    tested["is_core"] = tested.index.isin(core)
    tested = tested.sort_values(["pvalue", "delta"]).rename_axis("tf").reset_index()

    k_all = int((tested["pvalue"] < TABLE4_THRESHOLD).sum())
    core_rows = tested[tested["is_core"]]
    k_core = int((core_rows["pvalue"] < TABLE4_THRESHOLD).sum())
    n_core = len(core_rows)
    expected = n_core * k_all / n
    p_hyper = float(hypergeom.sf(k_core - 1, n, k_all, n_core))

    put = summary.append
    put(("tf_n_paired_patients", len(pids)))
    put(("tf_n_in_matrix", tf_acts.shape[1]))
    put(("tf_n_tested", n))
    put(("tf_n_p_lt_0005", k_all))
    put(("tf_pct_p_lt_0005", round(100 * k_all / n, 1)))
    put(("tf_n_p_lt_005", int((tested["pvalue"] < ALPHA).sum())))
    put(("tf_pct_p_lt_005", round(100 * (tested["pvalue"] < ALPHA).mean(), 1)))
    put(("tf_n_bh_005", int((tested["padj_bh"] < ALPHA).sum())))
    put(("tf_n_bonferroni_all_005", int((tested["bonferroni_p"] < ALPHA).sum())))
    put(("core_n_tested", n_core))
    put(("core_n_p_lt_0005", k_core))
    put(("core_n_p_lt_005", int((core_rows["pvalue"] < ALPHA).sum())))
    put(("core_expected_p_lt_0005", round(expected, 3)))
    put(("core_hypergeom_p_ge_observed", p_hyper))
    put(("core_n_bh_005_all_tfs", int((core_rows["padj_bh"] < ALPHA).sum())))
    put(("core_n_bonferroni_005_all_tfs", int((core_rows["bonferroni_p"] < ALPHA).sum())))
    for _, r in core_rows.iterrows():
        put((f"core_rank_{r['tf']}", int(r["rank"])))
    logger.info("TFs tested %d, p < %.3f: %d (%.1f%%), p < 0.05: %d, BH: %d, Bonferroni: %d",
                n, TABLE4_THRESHOLD, k_all, 100 * k_all / n, (tested["pvalue"] < ALPHA).sum(),
                (tested["padj_bh"] < ALPHA).sum(), (tested["bonferroni_p"] < ALPHA).sum())
    logger.info("Core: %d of %d below %.3f, expected %.2f, hypergeometric P(X >= %d) = %.3g",
                k_core, n_core, TABLE4_THRESHOLD, expected, k_core, p_hyper)
    logger.info("Core ranks among %d TFs: %s", n,
                dict(zip(core_rows["tf"], core_rows["rank"].astype(int))))
    return tested, res


# ---------------------------------------------------------------------------
# 2. Site background
# ---------------------------------------------------------------------------
def site_background(tf_acts: pd.DataFrame, info: pd.DataFrame, pooled: pd.DataFrame,
                    core: list[str], summary: list) -> pd.DataFrame:
    prim = info[info["role"] == "primary"]
    met = info[info["role"] == "metastasis"]
    pids = sorted(set(prim["patient_id"]) & set(met["patient_id"]))
    site_counts = met[met["patient_id"].isin(pids)]["site"].value_counts()
    candidate_sites = [s for s, k in site_counts.items() if k >= MIN_SITE_BIOPSIES]

    p_mean = patient_means(tf_acts, prim["patient_id"], pids)
    deltas, pvals, n_by_site = {}, {}, {}
    for site in candidate_sites:
        m_site = met[met["site"] == site]
        site_pids = [p for p in pids if p in set(m_site["patient_id"])]
        n_by_site[site] = len(site_pids)
        if len(site_pids) < MIN_PAIRS:
            continue
        res = paired_wilcoxon(p_mean.loc[site_pids], patient_means(tf_acts, m_site["patient_id"], site_pids))
        deltas[site], pvals[site] = res["delta"], res["pvalue"]
    sites = sorted(deltas)
    logger.info("Site block: biopsies per site %s, sites kept %s, patients per site %s",
                dict(site_counts), sites, {s: n_by_site[s] for s in sites})
    delta = pd.DataFrame(deltas)[sites]
    pval = pd.DataFrame(pvals)[sites]

    # Reproduce script 39 at full precision and the concordance counts of the paper.
    ref = pd.read_csv(V9 / "aurora_tf_by_site.csv")
    ref = ref[ref["delta"].notna()]
    keys = list(zip(ref["tf"], ref["site"]))
    got_d = np.array([delta.at[t, s] for t, s in keys])
    got_p = np.array([pval.at[t, s] for t, s in keys])
    dd = np.nanmax(np.abs(got_d - ref["delta"].to_numpy()))
    dp = np.nanmax(np.abs(got_p - ref["pvalue"].to_numpy()))
    same_nan = np.array_equal(np.isnan(got_p), ref["pvalue"].isna().to_numpy())
    check("aurora_tf_by_site.csv deltas and p-values", dd <= TOL and dp <= TOL and same_nan,
          f"{len(ref)} TF-site rows, max |ddelta| = {dd:.1e}, max |dp| = {dp:.1e}")
    sec = read_summary_sections(V9 / "aurora_tf_by_site_summary.csv")
    pub_delta = sec[0]
    check("sites of aurora_tf_by_site_summary.csv", list(pub_delta.columns) == sites, f"{sites}")
    ig = pd.read_csv(V9 / "aurora_tf_ig_validation.csv").set_index("tf")["delta"]
    pub_conc = {tf: int((np.sign(pub_delta.loc[tf]) == np.sign(ig[tf])).sum()) for tf in pub_delta.index}

    sign_pooled = np.sign(pooled.loc[delta.index, "delta"])
    concordant = np.sign(delta).eq(sign_pooled, axis=0) & delta.notna()
    n_conc = concordant.sum(axis=1).astype(int)
    n_sites = delta.notna().sum(axis=1).astype(int)
    got_conc = {tf: int(n_conc[tf]) for tf in pub_delta.index}
    check("per-site concordance of the core TFs (aurora_tf_by_site_summary.csv)", got_conc == pub_conc,
          f"{got_conc}")

    tested = pooled["pvalue"].notna()
    out = pd.DataFrame({
        "tf": delta.index,
        "is_core": delta.index.isin(core),
        "pooled_delta": pooled.loc[delta.index, "delta"].to_numpy(),
        "pooled_pvalue": pooled.loc[delta.index, "pvalue"].to_numpy(),
        "n_sites": n_sites.to_numpy(),
        "n_concordant": n_conc.to_numpy(),
    })
    for s in sites:
        out[f"delta_{s}"] = delta[s].to_numpy()
    for s in sites:
        out[f"pvalue_{s}"] = pval[s].to_numpy()
    out = out[tested.loc[out["tf"]].to_numpy()].sort_values(["n_concordant", "pooled_pvalue"],
                                                              ascending=[False, True]).reset_index(drop=True)

    n_all_sites = len(sites)
    put = summary.append
    put(("site_n_sites", n_all_sites))
    put(("site_sites", json.dumps(sites)))
    put(("site_patients_per_site", json.dumps({s: n_by_site[s] for s in sites})))

    def block(label: str, df: pd.DataFrame) -> None:
        n = len(df)
        n_full = int((df["n_concordant"] == n_all_sites).sum())
        n_ge5 = int((df["n_concordant"] >= 5).sum())
        put((f"site_{label}_n_tfs", n))
        put((f"site_{label}_n_concordant_all", n_full))
        put((f"site_{label}_frac_concordant_all", round(n_full / n, 4) if n else np.nan))
        put((f"site_{label}_n_concordant_ge5", n_ge5))
        put((f"site_{label}_frac_concordant_ge5", round(n_ge5 / n, 4) if n else np.nan))
        logger.info("Site concordance, %s: %d TFs, %d in all %d sites (%.1f%%), %d in >= 5 (%.1f%%)",
                    label, n, n_full, n_all_sites, 100 * n_full / max(n, 1), n_ge5, 100 * n_ge5 / max(n, 1))

    block("all", out)
    block("pooled_p0005", out[out["pooled_pvalue"] < TABLE4_THRESHOLD])
    block("core", out[out["is_core"]])
    block("core_p0005", out[out["is_core"] & (out["pooled_pvalue"] < TABLE4_THRESHOLD)])
    for tf in core:
        put((f"site_n_concordant_{tf}", int(n_conc[tf])))
    # Probability that a random set of ten TFs holds at least one TF concordant in every site.
    n_tf = len(out)
    n_full_all = int((out["n_concordant"] == n_all_sites).sum())
    put(("site_hypergeom_p_core_ge1_all", float(hypergeom.sf(0, n_tf, n_full_all, N_CORE))))
    return out


# ---------------------------------------------------------------------------
# 3. Drugs at patient level
# ---------------------------------------------------------------------------
def drug_background(pred: pd.DataFrame, info: pd.DataFrame, summary: list) -> pd.DataFrame:
    # Sample level, as script 22: GEO "disease" field, unpaired Mann-Whitney.
    disease = info["disease"].str.lower()
    prim_s = pred.loc[disease.str.contains("primary")]
    meta_s = pred.loc[disease.str.contains("metast")]
    sample = []
    for drug in pred.columns:
        p_auc, m_auc = prim_s[drug].dropna(), meta_s[drug].dropna()
        if len(p_auc) > 3 and len(m_auc) > 3:
            sample.append({"drug": drug, "delta_auc_sample": m_auc.mean() - p_auc.mean(),
                           "pvalue_sample": float(mannwhitneyu(p_auc, m_auc, alternative="two-sided")[1])})
    sample = pd.DataFrame(sample).set_index("drug")
    sample["padj_bh_sample"] = bh(sample["pvalue_sample"])

    ref = pd.read_csv(V9 / "aurora_drug_primary_vs_meta.csv").set_index("drug")
    check("drug set of aurora_drug_primary_vs_meta.csv", set(ref.index) == set(sample.index),
          f"{len(sample)} drugs")
    ok = np.allclose(sample.loc[ref.index, "pvalue_sample"], ref["pvalue"], rtol=1e-9, atol=TOL)
    dp = (sample.loc[ref.index, "pvalue_sample"] - ref["pvalue"]).abs().max()
    check("aurora_drug_primary_vs_meta.csv p-values", ok, f"max |dp| = {dp:.1e}")
    sig = sample["pvalue_sample"] < ALPHA
    counts = {"n_p005": int(sig.sum()),
              "n_more_active": int((sig & (sample["delta_auc_sample"] < 0)).sum()),
              "n_less_active": int((sig & (sample["delta_auc_sample"] > 0)).sum())}
    check("sample-level counts (782 changed, 633 more active, 149 less active)",
          counts == PUBLISHED_DRUG_COUNTS, f"{counts}")
    agree = (disease.str.contains("primary") == (info["role"] == "primary")).all() and \
        (disease.str.contains("metast") == (info["role"] == "metastasis")).all()
    logger.info("GEO disease field and sample-ID roles agree for every sample: %s", agree)

    # Patient level, with the pairing of the TFs (script 36).
    prim = info[info["role"] == "primary"]
    meta = info[info["role"] == "metastasis"]
    pids = sorted(set(prim["patient_id"]) & set(meta["patient_id"]))
    paired = paired_wilcoxon(patient_means(pred, prim["patient_id"], pids),
                             patient_means(pred, meta["patient_id"], pids))
    paired["padj_bh"] = bh(paired["pvalue"])
    paired = paired.rename(columns={"primary_mean": "primary_mean_auc", "metastasis_mean": "metastasis_mean_auc",
                                    "delta": "delta_auc", "pvalue": "pvalue_paired",
                                    "padj_bh": "padj_bh_paired"})
    paired["more_active_in_metastasis"] = paired["delta_auc"] < 0
    out = paired.join(sample, how="outer").rename_axis("drug").reset_index()
    out = out.sort_values("pvalue_paired").reset_index(drop=True)

    tested = out[out["pvalue_paired"].notna()]
    sig_bh = tested["padj_bh_paired"] < ALPHA
    sig_nom = tested["pvalue_paired"] < ALPHA
    put = summary.append
    put(("drug_sample_n_tested", len(sample)))
    put(("drug_sample_n_p005", counts["n_p005"]))
    put(("drug_sample_n_more_active_p005", counts["n_more_active"]))
    put(("drug_sample_n_less_active_p005", counts["n_less_active"]))
    put(("drug_sample_n_bh005", int((sample["padj_bh_sample"] < ALPHA).sum())))
    put(("drug_patient_n_paired_patients", len(pids)))
    put(("drug_patient_n_tested", len(tested)))
    put(("drug_patient_n_p005", int(sig_nom.sum())))
    put(("drug_patient_n_bh005", int(sig_bh.sum())))
    put(("drug_patient_n_bh005_more_active", int((sig_bh & (tested["delta_auc"] < 0)).sum())))
    put(("drug_patient_n_bh005_less_active", int((sig_bh & (tested["delta_auc"] > 0)).sum())))
    put(("drug_patient_n_p005_more_active", int((sig_nom & (tested["delta_auc"] < 0)).sum())))
    put(("drug_patient_n_p005_less_active", int((sig_nom & (tested["delta_auc"] > 0)).sum())))
    idx = out.set_index("drug")
    for d in REPORT_DRUGS:
        put((f"drug_{d}_pvalue_paired", float(idx.at[d, "pvalue_paired"])))
        put((f"drug_{d}_padj_bh_paired", float(idx.at[d, "padj_bh_paired"])))
        put((f"drug_{d}_delta_auc_paired", float(idx.at[d, "delta_auc"])))
        put((f"drug_{d}_pvalue_sample", float(idx.at[d, "pvalue_sample"])))
        put((f"drug_{d}_padj_bh_sample", float(idx.at[d, "padj_bh_sample"])))
    logger.info("Drugs at patient level: %d tested on %d patients, %d with p < 0.05, %d with BH < 0.05 "
                "(%d more active in metastasis, %d less)", len(tested), len(pids), sig_nom.sum(), sig_bh.sum(),
                (sig_bh & (tested["delta_auc"] < 0)).sum(), (sig_bh & (tested["delta_auc"] > 0)).sum())
    for d in REPORT_DRUGS:
        logger.info("  %-13s paired p = %.3g, BH = %.3g, delta = %+.4f | sample p = %.3g",
                    d, idx.at[d, "pvalue_paired"], idx.at[d, "padj_bh_paired"], idx.at[d, "delta_auc"],
                    idx.at[d, "pvalue_sample"])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tf_acts = pd.read_csv(V9 / "aurora_tf_activities.csv", index_col=0)
    pred = pd.read_csv(V9 / "aurora_drug_predictions.csv", index_col=0)
    if not tf_acts.index.equals(pred.index):
        raise ValueError("TF activities and drug predictions do not share the same samples")
    logger.info("AURORA: %d samples, %d TFs, %d drugs", tf_acts.shape[0], tf_acts.shape[1], pred.shape[1])
    info = sample_info(tf_acts.index)
    core = pd.read_csv(STABILITY).head(N_CORE)["tf"].tolist()
    missing = [tf for tf in core if tf not in tf_acts.columns]
    if missing:
        raise ValueError(f"Core TFs missing from the AURORA activities: {missing}")
    logger.info("Core TFs (head of %s): %s", STABILITY.name, core)

    summary: list[tuple[str, object]] = [("core_tfs", json.dumps(core))]
    tf_tab, pooled = tf_background(tf_acts, info, core, summary)
    site_tab = site_background(tf_acts, info, pooled, core, summary)
    drug_tab = drug_background(pred, info, summary)

    tf_tab.to_csv(OUT / "aurora_tf_background.csv", index=False)
    site_tab.to_csv(OUT / "aurora_site_background.csv", index=False)
    drug_tab.to_csv(OUT / "aurora_drug_paired.csv", index=False)
    summ = pd.DataFrame(summary, columns=["metric", "value"])
    summ.to_csv(OUT / "aurora_background_summary.csv", index=False)
    logger.info("Wrote %s", ", ".join(f.name for f in (
        OUT / "aurora_tf_background.csv", OUT / "aurora_site_background.csv",
        OUT / "aurora_drug_paired.csv", OUT / "aurora_background_summary.csv")))
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

