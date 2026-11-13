#!/usr/bin/env python3
"""How much does the gene universe of a cohort change its clinical transfer?

The TF activities of a cohort come from a univariate linear model of its
expression profile on each regulon, and that statistic grows with the number of
genes in the matrix. The clinical transfer then applies a QuantileTransformer
fitted on the PRISM activities, which clips whatever falls outside the range it
saw in training. A cohort computed over a much larger gene set therefore arrives
off scale and its predictions are compressed, silently.

Measured here for the three clinical cohorts of the paper and for FUSCC. SCAN-B
and METABRIC sit near 13% outside the training range, FUSCC over its full
transcriptome (activities of script 70) reaches 71.4% and is computed over the
PRISM universe for that reason (script 68), and TCGA-BRCA, whose activities were
computed from the GDC SummarizedExperiment, sits at 80.3%.

The question this script answers is what that costs the published result. It
reruns the whole TCGA screen twice, with the published activities and with the
ones recomputed over the PRISM gene universe (script 70), using the cohort rules
and the model of scripts/21_multicohort_survival.py, and recomputes the Fisher
meta-analysis of the three cohorts both ways. The published numbers are left
untouched: this is a sensitivity analysis, not a replacement.

The recomputation is only possible for the shared data. The published activities
came from the 760 MB GDC SummarizedExperiment, which is not in the shared data
tree (data/TCGA/README.md), while the reference matrix that is there holds 17,351
genes, 17,170 of them shared with PRISM.

Proportional hazards are tested as script 64 tests them in SCAN-B, with its
implementation imported: for the seven candidates and five positive controls,
every term of the canonical model (drug and PAM50 dummies) under the exact score
test of R survival 3.x cox.zph(transform="rank", terms=FALSE), with the km and
event-rank transforms and the lifelines approximation beside it, and the same
tests on the unpenalized refit with the same covariates. Both TCGA versions are
tested. The test is descriptive and changes no estimate.

Inputs:
  data/precision_processed/tf_activities_{prism,scanb,metabric,tcga,fuscc}.csv
  data/precision_processed/tcga_clinical.csv
  results/v15_gene_universe/tf_activities_tcga_prism_universe.csv   (script 70)
  results/v15_gene_universe/tf_activities_fuscc_full_transcriptome.csv  (script 70)
  results/v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv         (script 21)
  results/v8_multicohort/meta_analysis_fisher.csv                   (script 21)

Outputs (results/v15_gene_universe/):
  cohort_clipping.csv          one row per cohort, share outside the training range
  tcga_screen_comparison.csv   the 1,447 drugs under both versions
  fisher_meta_comparison.csv   the meta-analysis under both versions
  tcga_ph_schoenfeld.csv       proportional hazards test per version, drug and term
  gene_universe_summary.csv    metric, value

Run from the project root (about ten minutes, two full Cox screens):
    python scripts/71_gene_universe_check.py
"""

import importlib.util
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gene_universe")

PROCESSED = DATA_DIR / "precision_processed"
RESULTS = RESULTS_DIR
MULTI = RESULTS / "v8_multicohort"
OUT_DIR = RESULTS / "v15_gene_universe"
RECOMPUTED_TCGA = OUT_DIR / "tf_activities_tcga_prism_universe.csv"
FUSCC_FULL_TRANSCRIPTOME = OUT_DIR / "tf_activities_fuscc_full_transcriptome.csv"

SEED = 42
RIDGE_ALPHA = 100
MIN_TRAIN_RESPONSES = 30
COX_PENALIZER = 0.1
ALPHA = 0.05

# Frozen values, all measured on the files this script reads
EXPECTED_TCGA_COX = (1060, 149)
EXPECTED_PUBLISHED_TCGA_FDR05 = 0
EXPECTED_PUBLISHED_META_FDR05 = 551
EXPECTED_DRUGS = 1447
CLIPPING_TOLERANCE = 0.01   # cohorts whose share differs from the frozen value by more
CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
DRUG = "predicted_auc"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The proportional hazards test of script 64, so SCAN-B and TCGA share one implementation
PH = _load_script("s64_proportional_hazards", "64_proportional_hazards.py")


def ph_rows(version, drug, cph, data):
    """Script 64 part (1) for one drug: every term of the canonical penalized model and
    of its unpenalized refit, exact test (rank primary, km, event rank) and lifelines."""
    ph = PH.ph_pvalues(cph, data)
    ph_rr = PH.ph_pvalues(cph, data, PH.r_rank)
    ph_ex = PH.zph_exact(cph, data, COX_PENALIZER, transforms=PH.TRANSFORMS)
    cph_unpen = CoxPHFitter(penalizer=0.0).fit(data, "os_time", "os_event")
    ph_unpen = PH.ph_pvalues(cph_unpen, data)
    ph_unpen_ex = PH.zph_exact(cph_unpen, data, 0.0)
    rows = []
    for term in cph.params_.index:
        rows.append({
            "version": version, "drug": drug,
            "group": "candidate" if drug in CANDIDATES else "positive_control", "term": term,
            "n": len(data), "events": int(data["os_event"].sum()),
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
            "ph_p_unpenalized": float(ph_unpen.loc[term, "p"]),
            "ph_violation_p05_unpenalized": bool(ph_unpen.loc[term, "p"] < ALPHA),
            "ph_chisq_exact_unpenalized": float(ph_unpen_ex.loc[term, "chisq_rank"]),
            "ph_p_exact_unpenalized": float(ph_unpen_ex.loc[term, "p_rank"]),
            "ph_violation_p05_exact_unpenalized": bool(ph_unpen_ex.loc[term, "p_rank"] < ALPHA)})
    return rows


def clipping_fraction(cohort_tf: pd.DataFrame, train: pd.DataFrame):
    """Share of a cohort's activities outside the range the QT saw in training."""
    shared = sorted(set(train.columns) & set(cohort_tf.columns))
    sub = cohort_tf[shared]
    lo, hi = train[shared].min(), train[shared].max()
    below = float((sub < lo).to_numpy().mean())
    above = float((sub > hi).to_numpy().mean())
    return shared, below, above


def tcga_cohort():
    """The TCGA survival cohort of script 21: primary tumours, one row per patient."""
    clin = pd.read_csv(PROCESSED / "tcga_clinical.csv", index_col=0)
    if "shortLetterCode" in clin.columns:
        clin = clin[clin["shortLetterCode"] == "TP"]
    clin = clin.sort_index()
    if "submitter_id" in clin.columns:
        clin = clin[~clin["submitter_id"].duplicated(keep="first")]
    days_to_death = pd.to_numeric(clin.get("days_to_death"), errors="coerce")
    days_to_last = pd.to_numeric(clin.get("days_to_last_follow_up"), errors="coerce")
    clin["os_time"] = days_to_death.fillna(days_to_last)
    clin["os_event"] = (clin["vital_status"] == "Dead").astype(float)
    return clin


def run_screen(cohort_tf, clin, prism_tf, response, train_cells, label):
    """The transfer and the penalized Cox screen of script 21, for one set of activities.

    Returns the screen, its lambda_GC, the median spread of the predictions and the
    proportional hazards rows of the twelve focal drugs."""
    shared = sorted(set(prism_tf.columns) & set(cohort_tf.columns))
    train = prism_tf.loc[[c for c in train_cells if c in prism_tf.index], shared]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train)
    prism_qt = pd.DataFrame(qt.transform(prism_tf[shared]), index=prism_tf.index, columns=shared)
    cohort_qt = pd.DataFrame(qt.transform(cohort_tf[shared]), index=cohort_tf.index, columns=shared)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(prism_qt.loc[train.index])
    x_cohort = scaler.transform(cohort_qt)
    train_resp = response.loc[train.index]

    predictions = {}
    for drug in sorted(response.columns):
        y = train_resp[drug].to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() < MIN_TRAIN_RESPONSES:
            continue
        predictions[drug] = Ridge(alpha=RIDGE_ALPHA).fit(x_train[mask], y[mask]).predict(x_cohort)
    pred = pd.DataFrame(predictions, index=cohort_qt.index)

    merged = clin[["os_time", "os_event", "paper_BRCA_Subtype_PAM50"]].join(pred, how="inner")
    merged = merged.dropna(subset=["os_time", "os_event", "paper_BRCA_Subtype_PAM50"])
    merged = merged[merged["os_time"] > 0]
    if (len(merged), int(merged["os_event"].sum())) != EXPECTED_TCGA_COX:
        raise ValueError(f"{label}: the Cox cohort is {len(merged)} patients and "
                         f"{int(merged['os_event'].sum())} events, expected {EXPECTED_TCGA_COX}")
    dummies = pd.get_dummies(merged["paper_BRCA_Subtype_PAM50"], prefix="sub", drop_first=True)

    rows, ph = [], []
    for drug in pred.columns:
        data = pd.concat([merged[["os_time", "os_event", drug]], dummies], axis=1).dropna()
        data = data.rename(columns={drug: DRUG})
        if len(data) < 50 or data["os_event"].sum() < 10:
            continue
        cph = CoxPHFitter(penalizer=COX_PENALIZER).fit(data, "os_time", "os_event")
        rows.append({"drug": drug, "HR": float(cph.hazard_ratios_[DRUG]),
                     "pvalue": float(cph.summary.loc[DRUG, "p"])})
        if drug in CANDIDATES + CONTROLS:
            ph.extend(ph_rows(label, drug, cph, data))
    if len({r["drug"] for r in ph}) != len(CANDIDATES + CONTROLS):
        raise ValueError(f"{label}: proportional hazards tested for {len({r['drug'] for r in ph})} "
                         f"of the {len(CANDIDATES + CONTROLS)} focal drugs")
    out = pd.DataFrame(rows).sort_values("drug").reset_index(drop=True)
    out["padj"] = stats.false_discovery_control(out["pvalue"].to_numpy())
    lam = float(np.median(stats.chi2.isf(out["pvalue"], 1)) / stats.chi2.ppf(0.5, 1))
    logger.info("%s: %d drugs, %d at FDR < 0.05, %d at nominal p < 0.05, lambda %.2f, "
                "median spread of the predicted AUC %.4f", label, len(out),
                int((out["padj"] < ALPHA).sum()), int((out["pvalue"] < ALPHA).sum()), lam,
                float(pred.std().median()))
    return out, lam, float(pred.std().median()), ph


def fisher_meta(scanb, metabric, tcga):
    """Fisher combination of the three cohorts, as in script 21."""
    drugs = sorted(set(scanb.index) & set(metabric.index) & set(tcga.index))
    chi2 = -2 * (np.log(scanb.loc[drugs, "pvalue"]) + np.log(metabric.loc[drugs, "pvalue"])
                 + np.log(tcga.loc[drugs, "pvalue"]))
    p = stats.chi2.sf(chi2, 2 * 3)
    return pd.DataFrame({"drug": drugs, "fisher_p": p,
                         "fisher_padj": stats.false_discovery_control(p)}).set_index("drug")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    response = normalize_drug_columns(load_prism_response())
    prism_tf = pd.read_csv(PROCESSED / "tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(response, 0.2, SEED)
    train = prism_tf.loc[[c for c in train_cells if c in prism_tf.index]]

    # ---- how far each cohort sits outside the training range
    rows = []
    cohorts = {"SCANB": PROCESSED / "tf_activities_scanb.csv",
               "METABRIC": PROCESSED / "tf_activities_metabric.csv",
               "TCGA_published": PROCESSED / "tf_activities_tcga.csv",
               "TCGA_prism_universe": RECOMPUTED_TCGA,
               "FUSCC": PROCESSED / "tf_activities_fuscc.csv",
               "FUSCC_full_transcriptome": FUSCC_FULL_TRANSCRIPTOME}
    for label, path in cohorts.items():
        tf = pd.read_csv(path, index_col=0)
        shared, below, above = clipping_fraction(tf, train)
        rows.append({"cohort": label, "n_samples": len(tf), "n_tfs": tf.shape[1],
                     "n_tfs_shared_with_prism": len(shared),
                     "activity_mean": round(float(tf[shared].to_numpy().mean()), 3),
                     "activity_sd": round(float(tf[shared].to_numpy().std()), 3),
                     "pct_below_train_min": round(100 * below, 2),
                     "pct_above_train_max": round(100 * above, 2),
                     "pct_outside_train_range": round(100 * (below + above), 2)})
    clipping = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    logger.info("Outside the training range: %s",
                dict(zip(clipping["cohort"], clipping["pct_outside_train_range"])))

    # ---- the TCGA screen, both ways
    clin = tcga_cohort()
    published_tf = pd.read_csv(PROCESSED / "tf_activities_tcga.csv", index_col=0)
    recomputed_tf = pd.read_csv(RECOMPUTED_TCGA, index_col=0)
    pub, lam_pub, sd_pub, ph_pub = run_screen(published_tf, clin, prism_tf, response, train_cells,
                                              "TCGA published")
    new, lam_new, sd_new, ph_new = run_screen(recomputed_tf, clin, prism_tf, response, train_cells,
                                              "TCGA over the PRISM gene universe")
    ph = pd.DataFrame(ph_pub + ph_new)
    ph["version"] = ph["version"].map({"TCGA published": "published",
                                       "TCGA over the PRISM gene universe": "prism_universe"})
    ph = ph.sort_values(["version", "drug", "term"]).reset_index(drop=True)
    logger.info("Proportional hazards, exact test: drug terms violating at p < 0.05 %s, PAM50 term "
                "tests violating %s", {v: sorted(g.loc[(g["term"] == DRUG) & g["ph_violation_p05_exact"], "drug"])
                                       for v, g in ph.groupby("version")},
                {v: int(((g["term"] != DRUG) & g["ph_violation_p05_exact"]).sum())
                 for v, g in ph.groupby("version")})

    stored = pd.read_csv(MULTI / "survival_TCGA.csv").set_index("drug")
    if len(stored) != EXPECTED_DRUGS or int((stored["padj"] < ALPHA).sum()) != EXPECTED_PUBLISHED_TCGA_FDR05:
        raise ValueError("The published TCGA screen is not the one this script expects")
    check = pub.set_index("drug").join(stored, rsuffix="_stored")
    if not np.allclose(check["pvalue"], check["pvalue_stored"], rtol=1e-6):
        raise ValueError("Rerunning the published activities does not reproduce survival_TCGA.csv")
    logger.info("The rerun with the published activities reproduces survival_TCGA.csv")

    comparison = pub.set_index("drug").join(new.set_index("drug"), lsuffix="_published",
                                            rsuffix="_prism_universe")
    comparison["sign_agrees"] = (np.sign(np.log(comparison["HR_published"]))
                                 == np.sign(np.log(comparison["HR_prism_universe"])))
    comparison = comparison.reset_index().sort_values("drug")

    # ---- the Fisher meta-analysis, both ways
    scanb = pd.read_csv(MULTI / "survival_SCANB.csv").set_index("drug")
    metabric = pd.read_csv(MULTI / "survival_METABRIC.csv").set_index("drug")
    meta_pub = fisher_meta(scanb, metabric, pub.set_index("drug"))
    meta_new = fisher_meta(scanb, metabric, new.set_index("drug"))
    stored_meta = pd.read_csv(MULTI / "meta_analysis_fisher.csv")
    n_stored = int((stored_meta["fisher_padj"] < ALPHA).sum())
    if n_stored != EXPECTED_PUBLISHED_META_FDR05:
        raise ValueError(f"The published meta-analysis has {n_stored} drugs at FDR < 0.05, "
                         f"expected {EXPECTED_PUBLISHED_META_FDR05}")
    if int((meta_pub["fisher_padj"] < ALPHA).sum()) != n_stored:
        raise ValueError("The recomputed meta-analysis does not reproduce the published count")
    meta = meta_pub.join(meta_new, lsuffix="_published", rsuffix="_prism_universe").reset_index()
    sig_pub = meta["fisher_padj_published"] < ALPHA
    sig_new = meta["fisher_padj_prism_universe"] < ALPHA
    logger.info("Fisher meta-analysis: %d drugs published, %d with the recomputed TCGA "
                "(%d lost, %d gained, %d shared)", int(sig_pub.sum()), int(sig_new.sum()),
                int((sig_pub & ~sig_new).sum()), int((~sig_pub & sig_new).sum()),
                int((sig_pub & sig_new).sum()))

    summary = {
        "n_drugs": len(comparison),
        "tcga_n_patients": EXPECTED_TCGA_COX[0], "tcga_n_events": EXPECTED_TCGA_COX[1],
        "pct_outside_scanb": float(clipping.set_index("cohort").at["SCANB", "pct_outside_train_range"]),
        "pct_outside_metabric": float(clipping.set_index("cohort").at["METABRIC", "pct_outside_train_range"]),
        "pct_outside_tcga_published": float(
            clipping.set_index("cohort").at["TCGA_published", "pct_outside_train_range"]),
        "pct_outside_tcga_prism_universe": float(
            clipping.set_index("cohort").at["TCGA_prism_universe", "pct_outside_train_range"]),
        "pct_outside_fuscc": float(clipping.set_index("cohort").at["FUSCC", "pct_outside_train_range"]),
        "pct_outside_fuscc_full_transcriptome": float(
            clipping.set_index("cohort").at["FUSCC_full_transcriptome", "pct_outside_train_range"]),
        "tcga_published_fdr05": int((pub["padj"] < ALPHA).sum()),
        "tcga_prism_universe_fdr05": int((new["padj"] < ALPHA).sum()),
        "tcga_published_nominal05": int((pub["pvalue"] < ALPHA).sum()),
        "tcga_prism_universe_nominal05": int((new["pvalue"] < ALPHA).sum()),
        "tcga_published_lambda_gc": round(lam_pub, 2),
        "tcga_prism_universe_lambda_gc": round(lam_new, 2),
        "tcga_published_pred_sd": round(sd_pub, 4),
        "tcga_prism_universe_pred_sd": round(sd_new, 4),
        "tcga_pvalue_spearman": round(float(stats.spearmanr(
            comparison["pvalue_published"], comparison["pvalue_prism_universe"])[0]), 3),
        "tcga_sign_agreement_pct": round(100 * float(comparison["sign_agrees"].mean()), 1),
        "meta_published_fdr05": int(sig_pub.sum()),
        "meta_prism_universe_fdr05": int(sig_new.sum()),
        "meta_lost": int((sig_pub & ~sig_new).sum()),
        "meta_gained": int((~sig_pub & sig_new).sum()),
        "meta_shared": int((sig_pub & sig_new).sum()),
    }
    for drug in ("paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"):
        row = meta[meta["drug"] == drug].iloc[0]
        summary[f"meta_{drug}_padj_published"] = float(row["fisher_padj_published"])
        summary[f"meta_{drug}_padj_prism_universe"] = float(row["fisher_padj_prism_universe"])
    # proportional hazards, the counts of script 64 for each TCGA version
    summary["ph_zph_exact_n_models"] = PH.ZPH_DIAG["n_models"]
    summary["ph_zph_exact_max_variance_rel_diff_vs_lifelines"] = PH.ZPH_DIAG["var_rel_diff"]
    summary["ph_zph_exact_max_newton_step_per_sd"] = PH.ZPH_DIAG["newton_step_per_sd"]
    for version, g in ph.groupby("version"):
        drug_terms, pam_terms = g[g["term"] == DRUG], g[g["term"] != DRUG]
        summary[f"ph_{version}_n_drug_terms"] = len(drug_terms)
        summary[f"ph_{version}_n_pam50_term_tests"] = len(pam_terms)
        for tag in ("_exact", "_exact_unpenalized", ""):
            col = f"ph_violation_p05{tag}"
            summary[f"ph_{version}_n_drug_terms_violating_p05{tag}"] = int(drug_terms[col].sum())
            summary[f"ph_{version}_drug_terms_violating_p05{tag}"] = ";".join(
                sorted(drug_terms.loc[drug_terms[col], "drug"]))
            summary[f"ph_{version}_n_pam50_term_tests_violating_p05{tag}"] = int(pam_terms[col].sum())
        summary[f"ph_{version}_drug_term_p_min_exact"] = float(drug_terms["ph_p_exact"].min())

    clipping.to_csv(OUT_DIR / "cohort_clipping.csv", index=False)
    comparison.to_csv(OUT_DIR / "tcga_screen_comparison.csv", index=False)
    meta.to_csv(OUT_DIR / "fisher_meta_comparison.csv", index=False)
    ph.to_csv(OUT_DIR / "tcga_ph_schoenfeld.csv", index=False)
    pd.DataFrame({"metric": list(summary), "value": list(summary.values())}).to_csv(
        OUT_DIR / "gene_universe_summary.csv", index=False)
    logger.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
