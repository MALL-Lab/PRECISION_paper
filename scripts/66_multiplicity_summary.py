#!/usr/bin/env python3
"""Multiplicity checks and descriptive counts requested by the re-audit of Paper 1.

The script trains nothing and fits no survival model. It reads the outputs of
earlier scripts, reproduces the published numbers it builds on and raises if
any of them differs, and then derives the counts below.

  1. Benjamini-Yekutieli counts at 0.05 (valid under any dependence between
     tests) for the per-drug Cox p-values of SCAN-B, METABRIC and TCGA and for
     the Fisher meta-analysis. The Fisher p-values are recomputed here from the
     three per-cohort p-values with stats.chi2.sf, so they do not depend on the
     1 - cdf underflow of the original script 21. The Benjamini-Hochberg counts
     (623, 74, 0 and 551) are reproduced first.
  2. Direction of the 623 SCAN-B drugs with BH FDR < 0.05 (HR below or above 1).
  3. Fisher-significant drugs (BH < 0.05) that are BH-significant in none of
     the three cohorts on its own.
  4. HR range of the 623 SCAN-B drugs at the canonical penalizer 0.1, and the
     number of BH-significant drugs at each penalizer of script 45.
  5. Graph ablation: mean and sample SD (ddof 1) over the five seeds of the
     median per-drug Pearson of each configuration, recomputed from the
     per-drug files of scripts/57_train_gnns.py, and the number of seeds in
     which the full graph is below the response-only configuration.
  6. METABRIC PAM50 coverage: how many of the 50 PAM50 genes are in the
     expression matrix that script 20 hands to genefu. The gene list comes from
     the genefu pam50 centroids through Rscript when R and genefu are available,
     otherwise from the hard-coded list below (genefu 2.42.0, Parker et al. 2009),
     which is also checked against the list of scripts/20b_metabric_pam50.R.
  7. PRISM to GDSC cross-screen of script 05: cell lines and drugs evaluated,
     and whether each evaluated cell line and drug has training edges under the
     edge-level split of that script (its own prepare_edges is reused).
  8. AURORA primary versus metastasis test of epirubicin.
  9. Clinical-adjusted SCAN-B padj of olaparib (script 37). Script 37 may be
     rerun by another process, so the file is read only when its modification
     time is at least a minute old and unchanged across the read, and the
     version read (time stamp, number of drugs, talazoparib present or not) is
     written to the summary.
 10. Hyperparameters as used by scripts 11, 12 and 21 (parsed from their
     source) and by src/models/training.py, and where each value comes from:
       - for the Ridge and random forest baselines, the search of
         scripts/72_baseline_hp_search.py with the design of
         docs/plan_hp_baselines.md (both versioned in 3d539a0 before it ran),
         whose CESGA outputs in results/v16_hp_search/cesga_11133811/ are
         compared with the constants of script 11,
       - kept as a record, the values in use before that search (script 11 and
         Table 1 at 3d539a0), compared with the best trials of
         scripts/06_optuna_tuning.py (089d315) and its result table
         results/optuna_tuned_comparison.csv (9622e25, later removed from git),
       - scripts/07_optuna_gnn_extended.py and results/optuna_extended_*.csv
         (47aaf50) for the SAGE architecture and the optimizer,
       - the defaults of script 05 before any search for HGT,
       - the commit that first hard-coded the values in script 11 (1f7836d).
     Each objective is described from code fragments checked in the source of
     its commit, and the edge-level split of the GNN search is rebuilt to count
     how many of the Table 1 test cell lines it trained on and scored on.

Inputs:
  results/v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv   (script 21)
  results/v8_multicohort/meta_analysis_fisher.csv             (script 21, check)
  results/v8_multicohort/penalizer_sweep.csv, penalizer_sweep_perdrug.csv (45)
  results/v6_strengthen/graph_ablation.csv, graph_ablation_per_seed.csv   (27)
  results/v12_gnn/per_drug/<run>.csv                          (script 57)
  data/METABRIC/metabric_expression.rds, metabric_gene_map.rds
  data/precision_graph/hetero_graph_prism.pkl, PRISM and GDSC response
  results/crossval_prism_to_gdsc.csv                          (script 05, check)
  results/v9_aurora/aurora_drug_primary_vs_meta.csv           (script 22)
  results/v8_multicohort/cox_clinical_comparison.csv          (script 37)
  scripts/11_rerun_fair_comparison.py, 12_gnn_embeddings_perdrug.py,
  21_multicohort_survival.py, 05_train_hgt_and_crossval.py, src/models/training.py
  results/v16_hp_search/cesga_11133811/ (best_params_*.json, trials_*.csv,
  script_sha256.txt) and results/v16_hp_search/inputs/ (manifest.json, folds.csv)
  git history: 3d539a0, 089d315, 9622e25, 47aaf50, 1f7836d and the parent of 089d315
Outputs (results/v13_reaudit/):
  multiplicity_summary.csv      metric, value
  ablation_perdrug_median.csv   one row per configuration with the per-seed
                                median per-drug Pearson, its mean and SD, the
                                mean global Pearson and the seed comparison
  metabric_pam50_coverage.csv   one row per PAM50 gene: genefu symbol, current
                                HGNC symbol, Entrez ID, presence by Entrez ID
                                (how genefu maps with do.mapping = TRUE) and by
                                symbol, and the source of the list

Run from project root:
    python scripts/66_multiplicity_summary.py
"""

import ast
import glob
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import numpy as np
import pandas as pd
import pyreadr
from scipy import stats

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_gdsc_response, load_prism_response
from src.data.splits import cell_line_holdout_split
from src.graph.hetero_data import load_hetero_data
from src.models import training as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiplicity_summary")

SURV_DIR = RESULTS_DIR / "v8_multicohort"
COHORTS = ("SCANB", "METABRIC", "TCGA")
FISHER_PATH = SURV_DIR / "meta_analysis_fisher.csv"
PENSWEEP_PATH = SURV_DIR / "penalizer_sweep.csv"
PENSWEEP_PERDRUG_PATH = SURV_DIR / "penalizer_sweep_perdrug.csv"
CLINICAL_PATH = SURV_DIR / "cox_clinical_comparison.csv"
ABLATION_PATH = RESULTS_DIR / "v6_strengthen/graph_ablation.csv"
ABLATION_SEED_PATH = RESULTS_DIR / "v6_strengthen/graph_ablation_per_seed.csv"
METABRIC_EXPR = DATA_DIR / "METABRIC/metabric_expression.rds"
METABRIC_MAP = DATA_DIR / "METABRIC/metabric_gene_map.rds"
SCRIPT_20B = Path("scripts/20b_metabric_pam50.R")
CROSSVAL_PATH = RESULTS_DIR / "crossval_prism_to_gdsc.csv"
SCRIPT_05 = Path("scripts/05_train_hgt_and_crossval.py")
AURORA_PATH = RESULTS_DIR / "v9_aurora/aurora_drug_primary_vs_meta.csv"
SCRIPT_11 = Path("scripts/11_rerun_fair_comparison.py")
SCRIPT_12 = Path("scripts/12_gnn_embeddings_perdrug.py")
SCRIPT_21 = Path("scripts/21_multicohort_survival.py")
GRAPH_PATH = DATA_DIR / "precision_graph/hetero_graph_prism.pkl"  # default of load_hetero_data
# Git history of the hyperparameters
OPTUNA_COMMIT = "089d315"          # first Optuna search, baselines and GNNs
OPTUNA_SCRIPT = "scripts/06_optuna_tuning.py"
OPTUNA_RESULT_COMMIT = "9622e25"   # result table of that search
OPTUNA_RESULT = "results/optuna_tuned_comparison.csv"
OPTUNA_EXT_COMMIT = "47aaf50"      # extended GNN search
OPTUNA_EXT_SCRIPT = "scripts/07_optuna_gnn_extended.py"
OPTUNA_EXT_RESULT = "results/optuna_extended_comparison.csv"
OPTUNA_EXT_TRIALS = "results/optuna_extended_{}_trials.csv"
HARDCODE_COMMIT = "1f7836d"        # first script 11 with the GNN values in use
# Search of the Ridge and random forest hyperparameters with internal validation
HP_PLAN_COMMIT = "3d539a0"         # plan and search script, versioned before the search ran
HP_PLAN = "docs/plan_hp_baselines.md"
HP_SEARCH_SCRIPT = Path("scripts/72_baseline_hp_search.py")
HP_SEARCH_DIR = RESULTS_DIR / "v16_hp_search/cesga_11133811"   # outputs of the search, CESGA job 11133811
HP_INPUT_DIR = RESULTS_DIR / "v16_hp_search/inputs"            # input bundle written by its prepare step
TABLE1_CSV = "paper/results/table1_model_comparison.csv"
OUT_DIR = RESULTS_DIR / "v13_reaudit"

ALPHA = 0.05
SEED = 42  # split seed of scripts 05, 11 and the Optuna search, nothing here is random

# Published values, copied once from paper/results/paper_statistics.json of
# 2026-09-18. The script never reads that file.
PUB = {
    "n_drugs_tested": 1447,
    "bh": {"SCANB": 623, "METABRIC": 74, "TCGA": 0, "Fisher": 551},
    "pensweep_n_fdr_pen05": 306,
    "pensweep_hr_sig": {0.01: (0.0017, 245.51), 0.5: (0.1314, 5.53), 1.0: (0.3053, 1.58)},
    "sage_median_perdrug": (0.032, 0.009),
    "ablation_pearson": {"full": 0.743, "no_ppi": 0.742, "no_collectri": 0.742,
                         "no_drug_target": 0.726, "no_ppi_no_drug_target": 0.731,
                         "response_only": 0.722,
                         # control with the drug-target edges rewired, added on 2026-11-07
                         "shuffled_drug_target": 0.741},
    "metabric_n_genes_matrix": 6549,
    "gdsc_n_cell_lines": 696,
    "gdsc_n_drugs": 397,
    "crossval_n_drugs": 187,
    "crossval_median_pearson": 0.052,
    "response_edges": 614325,
    "aurora_drugs_sig": 782,
    "aurora_p": {"entinostat": 0.0028, "paclitaxel": 0.56},
    "clinical_adj_scanb_hr_range": (2.26, 3.72),
    "holdout_n_test_cells": 95,
}

# PAM50 genes as named in the centroids of genefu 2.42.0 (data(pam50),
# pam50$centroids.map), the list of Parker et al. 2009, J Clin Oncol 27:1160,
# with the Entrez Gene IDs genefu maps on. Three symbols are outdated in genefu.
PAM50_GENEFU = {
    "ACTR3B": 57180, "ANLN": 54443, "BAG1": 573, "BCL2": 596, "BIRC5": 332, "BLVRA": 644,
    "CCNB1": 891, "CCNE1": 898, "CDC20": 991, "CDC6": 990, "CDCA1": 83540, "CDH3": 1001,
    "CENPF": 1063, "CEP55": 55165, "CXXC5": 51523, "EGFR": 1956, "ERBB2": 2064, "ESR1": 2099,
    "EXO1": 9156, "FGFR4": 2264, "FOXA1": 3169, "FOXC1": 2296, "GPR160": 26996, "GRB7": 2886,
    "KIF2C": 11004, "KNTC2": 10403, "KRT14": 3861, "KRT17": 3872, "KRT5": 3852, "MAPT": 4137,
    "MDM2": 4193, "MELK": 9833, "MIA": 8190, "MKI67": 4288, "MLPH": 79083, "MMP11": 4320,
    "MYBL2": 4605, "MYC": 4609, "NAT1": 9, "ORC6L": 23594, "PGR": 5241, "PHGDH": 26227,
    "PTTG1": 9232, "RRM2": 6241, "SFRP1": 6422, "SLC39A6": 25800, "TMEM45B": 120224,
    "TYMS": 7298, "UBE2C": 11065, "UBE2T": 29089,
}
PAM50_CURRENT_SYMBOL = {"CDCA1": "NUF2", "KNTC2": "NDC80", "ORC6L": "ORC6"}


class Summary:
    """Ordered metric, value pairs plus a log of the reproduction checks."""

    def __init__(self):
        self.rows = []
        self.checks = []

    def add(self, metric, value):
        self.rows.append({"metric": metric, "value": value})
        logger.info("  %s = %s", metric, value)

    def check(self, what, got, expected, tol=0.0):
        """Raise unless got matches expected (absolute tolerance tol for numbers)."""
        if isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(expected, bool):
            ok = abs(float(got) - float(expected)) <= tol
        else:
            ok = got == expected
        self.checks.append({"what": what, "expected": expected, "got": got, "ok": ok})
        if not ok:
            raise ValueError(f"Reproduction failed for {what}: expected {expected}, got {got}")
        logger.info("  reproduced %s: %s (expected %s)", what, got, expected)


def bh(p):
    return stats.false_discovery_control(np.asarray(p, dtype=float), method="bh")


def by(p):
    return stats.false_discovery_control(np.asarray(p, dtype=float), method="by")


# 1 to 4. Survival screens ------------------------------------------------------

def survival_multiplicity(S: Summary):
    logger.info("[1-4] Multiplicity of the per-drug Cox screens")
    surv = {c: pd.read_csv(SURV_DIR / f"survival_{c}.csv").set_index("drug") for c in COHORTS}
    for c, df in surv.items():
        if df.index.duplicated().any():
            raise ValueError(f"survival_{c}.csv has duplicated drugs")
        S.check(f"{c} drugs tested", len(df), PUB["n_drugs_tested"])
        S.check(f"{c} BH FDR<0.05 (padj column)", int((df["padj"] < ALPHA).sum()), PUB["bh"][c])
        S.check(f"{c} BH FDR<0.05 (recomputed)", int((bh(df["pvalue"]) < ALPHA).sum()), PUB["bh"][c])

    # Fisher meta-analysis over the drugs tested in all three cohorts, with the
    # survival function instead of 1 - cdf so that tiny p-values do not become 0.
    common = sorted(set.intersection(*(set(df.index) for df in surv.values())))
    pmat = pd.DataFrame({c: surv[c].loc[common, "pvalue"] for c in COHORTS})
    fisher_stat = -2.0 * np.log(pmat).sum(axis=1)
    fisher_p = pd.Series(stats.chi2.sf(fisher_stat, df=2 * len(COHORTS)), index=common)
    fisher_bh = pd.Series(bh(fisher_p), index=common)
    fisher_by = pd.Series(by(fisher_p), index=common)
    S.check("Fisher drugs (tested in all cohorts)", len(common), PUB["n_drugs_tested"])
    S.check("Fisher BH FDR<0.05 (chi2.sf)", int((fisher_bh < ALPHA).sum()), PUB["bh"]["Fisher"])

    # Agreement with the file of script 21, whichever of its two versions is on
    # disk (1 - cdf until 2026-09-18, chi2.sf afterwards).
    meta = pd.read_csv(FISHER_PATH).set_index("drug")
    S.check("Fisher BH FDR<0.05 (meta_analysis_fisher.csv)", int((meta["fisher_padj"] < ALPHA).sum()),
            PUB["bh"]["Fisher"])
    comparable = meta.loc[common, "fisher_p"] > 1e-12
    rel = ((meta.loc[common, "fisher_p"] - fisher_p).abs() / fisher_p)[comparable]
    S.check("Fisher p equal to script 21 where p > 1e-12 (max relative difference below 1e-9)",
            bool(rel.max() < 1e-9), True)
    n_zero = int((meta["fisher_p"] == 0).sum())
    S.add("fisher_script21_file_version", "1 - chi2.cdf (underflows to 0)" if n_zero else "chi2.sf")
    S.add("fisher_script21_n_p_equal_zero", n_zero)

    S.add("n_drugs_tested_per_cohort", len(common))
    for c in COHORTS:
        S.add(f"{c}_bh_fdr05", int((surv[c]["padj"] < ALPHA).sum()))
    S.add("fisher_bh_fdr05", int((fisher_bh < ALPHA).sum()))
    for c in COHORTS:
        S.add(f"{c}_by_fdr05", int((by(surv[c]["pvalue"]) < ALPHA).sum()))
    S.add("fisher_by_fdr05", int((fisher_by < ALPHA).sum()))
    S.add("fisher_min_p_chi2_sf", f"{fisher_p.min():.3e}")

    # 2. Direction among the SCAN-B BH-significant drugs
    scanb = surv["SCANB"]
    sig = scanb[scanb["padj"] < ALPHA]
    S.add("SCANB_bh_fdr05_hr_below_1", int((sig["HR"] < 1).sum()))
    S.add("SCANB_bh_fdr05_hr_above_1", int((sig["HR"] > 1).sum()))
    S.add("SCANB_bh_fdr05_hr_equal_1", int((sig["HR"] == 1).sum()))

    # 3. Fisher-significant drugs significant in no single cohort
    any_cohort = pd.concat([surv[c].loc[common, "padj"] < ALPHA for c in COHORTS], axis=1).any(axis=1)
    fisher_only = sorted(fisher_bh.index[(fisher_bh < ALPHA) & ~any_cohort])
    S.add("fisher_bh_fdr05_not_fdr05_in_any_cohort", len(fisher_only))
    S.add("fisher_bh_fdr05_not_fdr05_in_any_cohort_drugs", ", ".join(fisher_only))
    S.add("fisher_by_fdr05_not_fdr05_in_any_cohort",
          int(((fisher_by < ALPHA) & ~any_cohort).sum()))

    # 4. HR range at penalizer 0.1 and the penalizer sweep of script 45
    hr_min, hr_max = float(sig["HR"].min()), float(sig["HR"].max())
    sweep = pd.read_csv(PENSWEEP_PATH).set_index("penalizer")
    S.check("penalizer 0.5 BH FDR<0.05 (script 45)", int(sweep.loc[0.5, "n_fdr_all"]), PUB["pensweep_n_fdr_pen05"])
    for pen, (lo, hi) in PUB["pensweep_hr_sig"].items():
        S.check(f"penalizer {pen} HR min of significant drugs", round(sweep.loc[pen, "hr_min_fdr"], 4), lo, 1e-4)
        S.check(f"penalizer {pen} HR max of significant drugs", round(sweep.loc[pen, "hr_max_fdr"], 2), hi, 1e-2)
    S.check("penalizer 0.1 BH FDR<0.05 equals survival_SCANB.csv", int(sweep.loc[0.1, "n_fdr_all"]),
            PUB["bh"]["SCANB"])
    S.check("penalizer 0.1 HR min equals survival_SCANB.csv", round(sweep.loc[0.1, "hr_min_fdr"], 10),
            round(hr_min, 10), 1e-10)
    S.check("penalizer 0.1 HR max equals survival_SCANB.csv", round(sweep.loc[0.1, "hr_max_fdr"], 8),
            round(hr_max, 8), 1e-8)
    perdrug = pd.read_csv(PENSWEEP_PERDRUG_PATH)
    per_pen = perdrug.groupby("penalizer")["padj"].apply(lambda s: int((s < ALPHA).sum()))
    for pen in sweep.index:
        S.check(f"penalizer {pen} count from penalizer_sweep_perdrug.csv", int(per_pen.loc[pen]),
                int(sweep.loc[pen, "n_fdr_all"]))
    S.add("SCANB_bh_fdr05_hr_min_pen0.1", round(hr_min, 4))
    S.add("SCANB_bh_fdr05_hr_max_pen0.1", round(hr_max, 2))
    for pen in sweep.index:
        S.add(f"SCANB_bh_fdr05_pen{pen:g}", int(sweep.loc[pen, "n_fdr_all"]))


# 5. Graph ablation -----------------------------------------------------------

def ablation_medians(S: Summary) -> pd.DataFrame:
    logger.info("[5] Graph ablation, median per-drug Pearson over seeds")
    per_seed = pd.read_csv(ABLATION_SEED_PATH)
    summary = pd.read_csv(ABLATION_PATH).set_index("config")
    # Recompute every per-seed median from the per-drug file of the run.
    recomputed = []
    for run in per_seed["run"]:
        pdf = pd.read_csv(T.RUNS_DIR / "per_drug" / f"{run}.csv")
        recomputed.append(float(pdf["pearson_r"].dropna().median()))
    per_seed["median_recomputed"] = recomputed
    diff = (per_seed["median_recomputed"] - per_seed["median_pearson"]).abs().max()
    S.check("per-seed median per-drug Pearson equal to the per-drug files (max abs diff below 1e-6)",
            bool(diff < 1e-6), True)

    rows = []
    for tag in (*T.ABLATIONS, *T.SHUFFLED):
        config = T.config_label(tag)
        part = per_seed[per_seed["tag"] == tag].sort_values("init_seed")
        if len(part) != 5:
            raise ValueError(f"{tag}: {len(part)} seeds instead of 5")
        S.check(f"ablation {tag} mean global Pearson", round(part["test_pearson"].mean(), 3),
                PUB["ablation_pearson"][tag], 1e-9)
        S.check(f"ablation {tag} mean global Pearson equals graph_ablation.csv",
                round(part["test_pearson"].mean(), 10), round(summary.loc[config, "test_pearson"], 10), 1e-10)
        row = {"config": config, "tag": tag, "n_seeds": len(part)}
        for _, r in part.iterrows():
            row[f"median_pearson_seed{int(r['init_seed'])}"] = r["median_recomputed"]
        row["median_pearson_mean"] = part["median_recomputed"].mean()
        row["median_pearson_sd"] = part["median_recomputed"].std(ddof=1)
        row["global_pearson_mean"] = part["test_pearson"].mean()
        row["global_pearson_sd"] = part["test_pearson"].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)

    full = out.set_index("tag").loc["full"]
    S.check("SAGE median per-drug Pearson (Table 1)", round(full["median_pearson_mean"], 3),
            PUB["sage_median_perdrug"][0], 1e-9)
    S.check("SAGE median per-drug Pearson SD (Table 1)", round(full["median_pearson_sd"], 3),
            PUB["sage_median_perdrug"][1], 1e-9)

    # Paired by initialization seed, full graph against response edges only
    fs = per_seed[per_seed["tag"] == "full"].set_index("init_seed")
    ro = per_seed[per_seed["tag"] == "response_only"].set_index("init_seed")
    n_below_median = int((fs["median_recomputed"] < ro.loc[fs.index, "median_recomputed"]).sum())
    n_below_global = int((fs["test_pearson"] < ro.loc[fs.index, "test_pearson"]).sum())
    out["seeds_below_response_only_median"] = np.nan
    out.loc[out["tag"] == "full", "seeds_below_response_only_median"] = n_below_median
    out["seeds_below_response_only_global"] = np.nan
    out.loc[out["tag"] == "full", "seeds_below_response_only_global"] = n_below_global

    for _, r in out.iterrows():
        S.add(f"ablation_median_perdrug_{r['tag']}", f"{r['median_pearson_mean']:.4f} +/- {r['median_pearson_sd']:.4f}")
    S.add("ablation_seeds_full_below_response_only_median_perdrug", f"{n_below_median} of {len(fs)}")
    S.add("ablation_seeds_full_below_response_only_global", f"{n_below_global} of {len(fs)}")
    return out


# 6. METABRIC PAM50 coverage --------------------------------------------------

def find_rscript():
    exe = shutil.which("Rscript")
    if exe:
        return exe
    found = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"))
    return found[-1] if found else None


def genefu_pam50():
    """PAM50 symbols and Entrez IDs from genefu through Rscript, or None."""
    exe = find_rscript()
    if exe is None:
        return None, None
    code = ('suppressPackageStartupMessages(library(genefu)); data(pam50); '
            'cat("VERSION", as.character(packageVersion("genefu")), "\\n"); '
            'm <- pam50$centroids.map; '
            'write.csv(m[, c("probe.centroids", "EntrezGene.ID")], stdout(), row.names = FALSE, quote = FALSE)')
    try:
        res = subprocess.run([exe, "-e", code], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("  Rscript call failed: %s", exc)
        return None, None
    if res.returncode != 0:
        logger.warning("  Rscript returned %d: %s", res.returncode, res.stderr[-500:])
        return None, None
    lines = res.stdout.splitlines()
    version = next(ln.split()[1] for ln in lines if ln.startswith("VERSION"))
    csv = "\n".join(ln for ln in lines if not ln.startswith("VERSION"))
    df = pd.read_csv(io.StringIO(csv))
    return dict(zip(df["probe.centroids"], df["EntrezGene.ID"].astype(int))), version


def pam50_coverage(S: Summary) -> pd.DataFrame:
    logger.info("[6] METABRIC PAM50 coverage")
    from_r, version = genefu_pam50()
    if from_r is not None:
        S.check("genefu pam50 centroids (symbols and Entrez IDs) equal the hard-coded list",
                from_r == PAM50_GENEFU, True)
        genes, source = from_r, f"genefu {version} pam50$centroids.map via Rscript"
    else:
        genes, source = PAM50_GENEFU, "hard-coded list of genefu 2.42.0 (Parker et al. 2009)"
    # Cross-check with the list that scripts/20b_metabric_pam50.R prints
    text = SCRIPT_20B.read_text(encoding="utf-8")
    block = re.search(r"pam50_genes\s*<-\s*c\((.*?)\)", text, re.S).group(1)
    list_20b = set(re.findall(r'"([A-Z0-9]+)"', block))
    current = {PAM50_CURRENT_SYMBOL.get(g, g) for g in genes}
    S.check("PAM50 list of script 20b has 50 genes", len(list_20b), 50)
    S.check("PAM50 list of script 20b equals genefu after symbol updates (except ORC6L)",
            sorted(list_20b ^ current), ["ORC6", "ORC6L"])

    expr = next(iter(pyreadr.read_r(str(METABRIC_EXPR)).values()))
    gmap = next(iter(pyreadr.read_r(str(METABRIC_MAP)).values()))
    S.check("METABRIC matrix genes", expr.shape[0], PUB["metabric_n_genes_matrix"])
    symbols = set(expr.index)
    sym2ent = dict(zip(gmap["hugoGeneSymbol"], gmap["entrezGeneId"].astype(int)))
    entrez_in_matrix = {sym2ent[s] for s in symbols if s in sym2ent}

    rows = []
    for g, ent in sorted(genes.items()):
        cur = PAM50_CURRENT_SYMBOL.get(g, g)
        rows.append({"gene_genefu": g, "gene_current": cur, "entrez_id": ent,
                     "present_by_entrez": ent in entrez_in_matrix,
                     "present_by_genefu_symbol": g in symbols,
                     "present_by_current_symbol": cur in symbols,
                     "in_script_20b_list": (cur in list_20b) or (g in list_20b),
                     "source": source})
    cov = pd.DataFrame(rows)
    S.add("metabric_pam50_gene_list_source", source)
    S.add("metabric_pam50_present_by_entrez", int(cov["present_by_entrez"].sum()))
    S.add("metabric_pam50_present_by_current_symbol", int(cov["present_by_current_symbol"].sum()))
    S.add("metabric_pam50_present_by_genefu_symbol", int(cov["present_by_genefu_symbol"].sum()))
    S.add("metabric_pam50_missing_by_entrez",
          ", ".join(cov.loc[~cov["present_by_entrez"], "gene_current"]))
    return cov


# 7. PRISM to GDSC cross-screen -----------------------------------------------

def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gdsc_crossscreen(S: Summary, s05, data, node_maps):
    logger.info("[7] PRISM to GDSC cross-screen of script 05")
    gdsc = normalize_drug_columns(load_gdsc_response(use_processed=True))
    prism = normalize_drug_columns(load_prism_response(use_processed=True))
    S.check("GDSC cell lines", gdsc.shape[0], PUB["gdsc_n_cell_lines"])
    S.check("GDSC drugs", gdsc.shape[1], PUB["gdsc_n_drugs"])

    # Same selection as the cross-validation loop of script 05
    cell_map, drug_map = node_maps["cell_line"], node_maps["drug"]
    shared_cells = sorted(set(prism.index) & set(gdsc.index))
    shared_drugs = sorted(set(prism.columns) & set(gdsc.columns))
    n_samples, pairs = {}, []
    for drug in shared_drugs:
        if drug not in drug_map:
            continue
        cells = [c for c in shared_cells if c in cell_map and not np.isnan(gdsc.loc[c, drug])]
        if len(cells) < 5:
            continue
        n_samples[drug] = len(cells)
        pairs.extend((c, drug) for c in cells)
    cv = pd.read_csv(CROSSVAL_PATH).set_index("drug")
    S.check("cross-screen drugs evaluated", len(n_samples), PUB["crossval_n_drugs"])
    S.check("cross-screen n_samples per drug equal crossval_prism_to_gdsc.csv",
            bool(pd.Series(n_samples).sort_index().equals(cv["n_samples"].sort_index())), True)
    S.check("cross-screen median per-drug Pearson", round(cv["pearson_r"].median(), 3),
            PUB["crossval_median_pearson"], 1e-9)

    # Edge-level split of script 05, with its own function and constants
    train_cell, train_drug, _, test_cell, test_drug, _ = s05.prepare_edges(data)
    n_edges = int(data[T.RESPONSE].edge_index.shape[1])
    S.check("PRISM response edges", n_edges, PUB["response_edges"])
    n_drug_nodes = len(drug_map)
    train_pairs = set((train_cell * n_drug_nodes + train_drug).tolist())
    test_pairs = set((test_cell * n_drug_nodes + test_drug).tolist())
    train_cells_idx, train_drugs_idx = set(train_cell.tolist()), set(train_drug.tolist())
    eval_cells = sorted({c for c, _ in pairs})
    codes = [cell_map[c] * n_drug_nodes + drug_map[d] for c, d in pairs]
    n_pair_train = sum(k in train_pairs for k in codes)
    n_pair_test = sum(k in test_pairs for k in codes)

    S.add("gdsc_n_cell_lines_total", gdsc.shape[0])
    S.add("gdsc_prism_shared_cell_lines", len(shared_cells))
    S.add("gdsc_prism_shared_cell_lines_in_graph", sum(c in cell_map for c in shared_cells))
    S.add("gdsc_eval_n_cell_lines", len(eval_cells))
    S.add("gdsc_eval_n_drugs", len(n_samples))
    S.add("gdsc_eval_cell_lines_with_train_edges",
          f"{sum(cell_map[c] in train_cells_idx for c in eval_cells)} of {len(eval_cells)}")
    S.add("gdsc_eval_drugs_with_train_edges",
          f"{sum(drug_map[d] in train_drugs_idx for d in n_samples)} of {len(n_samples)}")
    S.add("gdsc_eval_pairs", len(codes))
    S.add("gdsc_eval_pairs_prism_train_edge", n_pair_train)
    S.add("gdsc_eval_pairs_prism_test_edge", n_pair_test)
    S.add("gdsc_eval_pairs_not_screened_in_prism", len(codes) - n_pair_train - n_pair_test)
    S.add("gdsc_eval_pairs_screened_in_prism_pct_train",
          round(100 * n_pair_train / max(1, n_pair_train + n_pair_test), 1))
    S.add("crossval_split", f"edge-level random permutation of the PRISM response edges "
                            f"(np.random.RandomState({s05.SEED})), {s05.TEST_FRAC:g} held out")
    S.add("crossval_epochs", s05.EPOCHS)
    S.add("crossval_lr", s05.LR)
    S.add("crossval_weight_decay", s05.WEIGHT_DECAY)
    S.add("crossval_seed", s05.SEED)
    S.add("crossval_hgt_hidden_layers_heads_dropout",
          f"{s05.HIDDEN_DIM}, {s05.N_LAYERS}, {s05.N_HEADS}, {s05.DROPOUT}")
    S.add("crossval_checkpoint", "final epoch (per script 05 docstring and train_model)")


# 8. AURORA epirubicin --------------------------------------------------------

def aurora_epirubicin(S: Summary):
    logger.info("[8] AURORA primary versus metastasis")
    df = pd.read_csv(AURORA_PATH).set_index("drug")
    S.check("AURORA drugs with p < 0.05", int((df["pvalue"] < ALPHA).sum()), PUB["aurora_drugs_sig"])
    for drug, p in PUB["aurora_p"].items():
        S.check(f"AURORA {drug} p", round(df.loc[drug, "pvalue"], 4 if p < 0.01 else 2), p, 1e-9)
    r = df.loc["epirubicin"]
    S.add("aurora_epirubicin_p", round(float(r["pvalue"]), 4))
    S.add("aurora_epirubicin_delta_auc", round(float(r["delta_auc"]), 4))
    S.add("aurora_epirubicin_primary_mean_auc", round(float(r["primary_mean_auc"]), 4))
    S.add("aurora_epirubicin_metastasis_mean_auc", round(float(r["metastasis_mean_auc"]), 4))
    S.add("aurora_drugs_nominal_p05", int((df["pvalue"] < ALPHA).sum()))
    S.add("aurora_drugs_bh_fdr05", int((bh(df["pvalue"]) < ALPHA).sum()))
    S.add("aurora_drugs_by_fdr05", int((by(df["pvalue"]) < ALPHA).sum()))


# 9. Clinical-adjusted olaparib -----------------------------------------------

def read_settled_csv(path: Path, min_age_s=60, tries=10, wait_s=15):
    """Read a CSV only once it is at least min_age_s old and unchanged across the read."""
    for _ in range(tries):
        st = path.stat()
        age = time.time() - st.st_mtime
        if age < min_age_s:
            logger.info("  %s modified %.0f s ago, waiting", path, age)
            time.sleep(wait_s)
            continue
        df = pd.read_csv(path)
        st2 = path.stat()
        if (st2.st_mtime, st2.st_size) == (st.st_mtime, st.st_size):
            return df, st2
        time.sleep(wait_s)
    raise RuntimeError(f"{path} kept changing, not read")


def clinical_olaparib(S: Summary):
    logger.info("[9] Clinical-adjusted SCAN-B Cox of script 37")
    df, st = read_settled_csv(CLINICAL_PATH)
    df = df.set_index("drug")
    scanb = pd.read_csv(SURV_DIR / "survival_SCANB.csv").set_index("drug")
    # The PAM50-only columns come from survival_SCANB.csv, whatever the version.
    S.check("olaparib padj_pam50 equals survival_SCANB.csv", round(df.loc["olaparib", "padj_pam50"], 4),
            round(float(scanb.loc["olaparib", "padj"]), 4), 1e-4)
    S.check("olaparib HR_pam50 equals survival_SCANB.csv", round(df.loc["olaparib", "HR_pam50_scanb"], 2),
            round(float(scanb.loc["olaparib", "HR"]), 2), 1e-9)
    cands = ["osimertinib", "saracatinib", "erlotinib", "brigatinib", "pelitinib", "entinostat", "trametinib"]
    lo, hi = PUB["clinical_adj_scanb_hr_range"]
    sig_c = df.loc[cands]
    sig_c = sig_c[sig_c["padj_clinical_scanb"] < ALPHA]
    S.check("clinical-adjusted SCAN-B HR min of significant candidates", sig_c["HR_clinical_scanb"].min(), lo, 1e-9)
    S.check("clinical-adjusted SCAN-B HR max of significant candidates", sig_c["HR_clinical_scanb"].max(), hi, 1e-9)
    S.add("clinical_file_mtime", datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))
    S.add("clinical_file_n_drugs", len(df))
    S.add("clinical_file_has_talazoparib", "talazoparib" in df.index)
    S.add("clinical_olaparib_hr_scanb", float(df.loc["olaparib", "HR_clinical_scanb"]))
    S.add("clinical_olaparib_padj_scanb", float(df.loc["olaparib", "padj_clinical_scanb"]))
    S.add("clinical_olaparib_padj_metabric", float(df.loc["olaparib", "padj_clinical_metabric"]))


# 10. Hyperparameters and their provenance ------------------------------------

def literal_dicts(src: str):
    """Every dict literal of a script source with constant string keys, values kept if constant."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and all(isinstance(k, ast.Constant) for k in node.keys):
            yield {k.value: (v.value if isinstance(v, ast.Constant) else None)
                   for k, v in zip(node.keys, node.values)}


def ridge_alphas(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    alphas = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Ridge":
            for kw in node.keywords:
                if kw.arg == "alpha" and isinstance(kw.value, ast.Constant):
                    alphas.add(kw.value.value)
    return alphas


def module_constants(src: str):
    """Top-level NAME = literal assignments of a script source, read with ast and never imported."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return out


def git(*args):
    res = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr}")
    return res.stdout


def git_bytes(*args):
    """Raw stdout of a git command, to hash a file exactly as versioned."""
    res = subprocess.run(["git", *args], capture_output=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.decode(errors='replace')}")
    return res.stdout


def history_available():
    """True when the development history the provenance checks read (OPTUNA_COMMIT) is present."""
    res = subprocess.run(["git", "cat-file", "-e", f"{OPTUNA_COMMIT}^{{commit}}"], capture_output=True, timeout=120)
    return res.returncode == 0


def git_csv(commit, path):
    """A CSV file as versioned in a commit."""
    return pd.read_csv(io.StringIO(git("show", f"{commit}:{path}")))


def commit_info(commit):
    """Short hash, Unix time and ISO date of a commit."""
    h, ts, date = git("show", "-s", "--format=%h|%at|%ad", "--date=iso", commit).strip().split("|")
    return h, int(ts), date


def removed_in(path):
    """Commit that deleted a path from git (hash and date), or 'not deleted'."""
    out = git("log", "--diff-filter=D", "--format=%h %ad", "--date=short", "--", path).strip()
    return out.splitlines()[0] if out else "not deleted"


def require(src, fragments, where):
    """Raise unless every code fragment (plain text, or a compiled regex) is in the source."""
    missing = [f if isinstance(f, str) else f.pattern for f in fragments
               if (f not in src if isinstance(f, str) else f.search(src) is None)]
    if missing:
        raise ValueError(f"Code fragments not found in {where}: {missing}")


def number(src, pattern, where):
    """First number captured by pattern in a source text, raise if absent."""
    m = re.search(pattern, src, re.M)
    if m is None:
        raise ValueError(f"Pattern {pattern!r} not found in {where}")
    return float(m.group(1))


def search_space(src):
    """Optuna search space of a script, one entry per parameter name and range."""
    found = re.findall(r'suggest_(?:float|int|categorical)\("(\w+)",\s*([^)]*)\)', src)
    return list(dict.fromkeys(f"{k}: {v.strip()}" for k, v in found))  # Ridge alpha appears twice in 06


def has_validation(src):
    return re.search(r"\bval_|validation", src, re.I) is not None


def hgt_key(hidden, layers, heads, dropout):
    """HGT architecture as a comparable tuple (hidden, layers, heads, dropout)."""
    return int(hidden), int(layers), int(heads), round(float(dropout), 3)


def fmt(values):
    return ", ".join(str(v) for v in values)


def hyperparameters_in_use(S: Summary):
    """Values in the current code of scripts 11, 12 and 21 and of src/models/training.py."""
    src11 = SCRIPT_11.read_text(encoding="utf-8")
    c11 = module_constants(src11)
    missing = {"RIDGE_ALPHA", "RF_PARAMS", "RF_SEEDS"} - set(c11)
    if missing:
        raise ValueError(f"{SCRIPT_11} lacks the constants {sorted(missing)}")
    ridge_alpha, rf, rf_seeds = c11["RIDGE_ALPHA"], c11["RF_PARAMS"], c11["RF_SEEDS"]
    alphas12, alphas21 = ridge_alphas(SCRIPT_12), ridge_alphas(SCRIPT_21)
    S.add("hp_script11_ridge_alpha", ridge_alpha)
    S.add("hp_script11_rf_n_estimators", rf["n_estimators"])
    S.add("hp_script11_rf_max_depth", rf["max_depth"])
    S.add("hp_script11_rf_min_samples_leaf", rf["min_samples_leaf"])
    S.add("hp_script11_rf_seeds", fmt(rf_seeds))
    S.add("hp_script12_ridge_alpha", fmt(sorted(alphas12)))
    S.add("hp_script21_ridge_alpha", fmt(sorted(alphas21)))
    for arch, hp in T.ARCHS.items():
        S.add(f"hp_gnn_{arch}", ", ".join(f"{k} {v}" for k, v in hp.items()))
    for k in ("lr", "weight_decay", "epochs", "eta_min"):
        S.add(f"hp_gnn_optim_{k}", T.OPTIM[k])
    # Until October 2026 script 11 described the baseline search as Bayesian with 50
    # trials, which it was not. It now cites the plan, the search script, the commit that
    # versioned both and the outputs of the search.
    S.check("script 11 no longer describes a Bayesian search of 50 trials",
            re.search(r"Bayesian|50 trials", src11) is None, True)
    require(src11, (HP_PLAN, HP_SEARCH_SCRIPT.as_posix(), HP_PLAN_COMMIT, HP_SEARCH_DIR.as_posix()),
            SCRIPT_11.as_posix())
    S.add("script11_hp_source_cited", f"{HP_PLAN} and {HP_SEARCH_SCRIPT.as_posix()} at {HP_PLAN_COMMIT}, "
                                      f"outputs in {HP_SEARCH_DIR.as_posix()}")
    return ridge_alpha, rf


def provenance_search(S: Summary, ridge_alpha, rf, t1_test):
    """Ridge and random forest of Table 1 against the search of scripts/72 run on CESGA."""
    best = {m: json.loads((HP_SEARCH_DIR / f"best_params_{m}.json").read_text()) for m in ("ridge", "rf")}
    trials = {m: pd.read_csv(HP_SEARCH_DIR / f"trials_{m}.csv") for m in ("ridge", "rf")}
    manifest = json.loads((HP_INPUT_DIR / "manifest.json").read_text())
    rf_keys = ("n_estimators", "max_depth", "min_samples_leaf")
    for m in ("ridge", "rf"):
        b, t, d = best[m], trials[m], best[m]["design"]
        S.check(f"{m} search: design of the plan (official, 10 trials, 5 folds, 200 drugs, seed 42)",
                (d["official"], d["n_trials"], d["n_folds"], d["n_objective_drugs"], d["seed"]),
                (True, 10, 5, 200, 42))
        S.check(f"{m} search: trials_{m}.csv holds trials 0 to 9", sorted(t["trial"].tolist()), list(range(10)))
        top = t.loc[t["cv_mean"].idxmax()]
        S.check(f"{m} search: best_trial of best_params_{m}.json is the best of trials_{m}.csv",
                int(top["trial"]), int(b["best_trial"]))
        S.check(f"{m} search: best_cv_mean equals the cv_mean of that trial", float(top["cv_mean"]),
                float(b["best_cv_mean"]), 1e-15)
        S.check(f"{m} search: input bundle (sha256) equals the manifest of {HP_INPUT_DIR.as_posix()}",
                b["bundle_sha256"], manifest["sha256"])
    rb, fb = best["ridge"], best["rf"]
    S.check("chosen Ridge alpha is the best alpha rounded to three significant figures",
            float(f"{rb['best_params']['alpha']:.3g}"), float(rb["chosen_params"]["alpha"]))
    S.check("script 11 Ridge alpha equals the chosen alpha of best_params_ridge.json",
            float(ridge_alpha), float(rb["chosen_params"]["alpha"]))
    S.check("chosen RF parameters equal the best RF trial",
            tuple(fb["chosen_params"][k] for k in rf_keys), tuple(fb["best_params"][k] for k in rf_keys))
    S.check("script 11 RF parameters equal the chosen parameters of best_params_rf.json",
            tuple(rf[k] for k in rf_keys), tuple(fb["chosen_params"][k] for k in rf_keys))
    S.check("test cell lines left out of the bundle are the Table 1 test lines",
            sorted(manifest["test_cell_ids"]), sorted(t1_test))
    # folds.csv carries the DepMap PatientID of each train line, it is only counted here
    folds = pd.read_csv(HP_INPUT_DIR / "folds.csv")
    n_test_in_bundle = len(set(folds["cell_line"]) & set(t1_test))
    S.check("Table 1 test lines in folds.csv", n_test_in_bundle, 0)
    S.check("patients split across folds", int((folds.groupby("PatientID")["fold"].nunique() > 1).sum()), 0)
    script_sha = (HP_SEARCH_DIR / "script_sha256.txt").read_text().split()[0]
    S.check(f"sha256 of the script run on CESGA equals {HP_SEARCH_SCRIPT.as_posix()}@{HP_PLAN_COMMIT}",
            script_sha, hashlib.sha256(git_bytes("show", f"{HP_PLAN_COMMIT}:{HP_SEARCH_SCRIPT.as_posix()}")).hexdigest())

    h, _, date = commit_info(HP_PLAN_COMMIT)
    S.add("hp_search_plan", f"{HP_PLAN} and {HP_SEARCH_SCRIPT.as_posix()}, versioned in {h} ({date})")
    S.add("hp_search_outputs", HP_SEARCH_DIR.as_posix())
    S.add("hp_search_design", f"{rb['design']['n_trials']} random trials per model ({rb['design']['sampler']}), "
                              f"{rb['design']['n_folds']}-fold cross-validation inside the {len(folds)} train lines, "
                              f"grouped by PatientID and stratified by OncotreeLineage, median per-drug Pearson "
                              f"over the {rb['design']['n_objective_drugs']} most complete drugs")
    S.add("hp_search_fold_sizes", fmt(folds["fold"].value_counts().sort_index().tolist()))
    S.add("hp_search_table1_test_lines_in_bundle", f"{n_test_in_bundle} of {len(t1_test)}")
    S.add("hp_search_best_ridge_expression",
          f"alpha {rb['best_params']['alpha']:.4f} (trial {rb['best_trial']}), rounded to "
          f"{rb['chosen_params']['alpha']:g}, CV {rb['chosen_cv_mean']:.4f} +/- {rb['chosen_cv_sd']:.4f}")
    S.add("hp_search_best_rf_tf", ", ".join(f"{k} {fb['chosen_params'][k]}" for k in rf_keys)
          + f" (trial {fb['best_trial']}), CV {fb['chosen_cv_mean']:.4f} +/- {fb['chosen_cv_sd']:.4f}")
    for m, tag in (("ridge", "ridge_expression"), ("rf", "rf_tf")):
        edge = best[m]["edge_diagnostic"]
        S.add(f"hp_search_edge_diagnostic_{tag}",
              "; ".join(f"{e['param']} {e['probe']:g}: CV {e['cv_mean']:.4f} +/- {e['cv_sd']:.4f}" for e in edge)
              or "none, the best trial is not in the outer 10% of the range")
    S.add("hp_source_ridge_rf", f"best trials of {HP_SEARCH_SCRIPT.as_posix()} ({h}), 5-fold cross-validation "
                                f"inside the train lines, outputs in {HP_SEARCH_DIR.as_posix()}, Ridge alpha "
                                f"{rb['best_params']['alpha']:.4f} rounded to {rb['chosen_params']['alpha']:g}")


def provenance_baselines(S: Summary, resp, t1_test):
    """Record of the Ridge and random forest used before the search of scripts/72.

    Script 11 and Table 1 as versioned in HP_PLAN_COMMIT, before the search ran, against
    scripts/06_optuna_tuning.py and its result table, which scored the trials on the
    Table 1 test lines."""
    prev11 = git("show", f"{HP_PLAN_COMMIT}:{SCRIPT_11.as_posix()}")
    d_prev = list(literal_dicts(prev11))
    ridge_prev = [d["alpha"] for d in d_prev if "alpha" in d]
    rf_prev = [d for d in d_prev if "n_estimators" in d]
    if len(ridge_prev) != 1 or len(rf_prev) != 1:
        raise ValueError(f"Expected one Ridge and one RF parameter dict in {SCRIPT_11.as_posix()}@{HP_PLAN_COMMIT}")
    ridge_alpha, rf = ridge_prev[0], rf_prev[0]
    S.add("hp_previous_script11_ridge_alpha", ridge_alpha)
    for k in ("n_estimators", "max_depth", "min_samples_leaf"):
        S.add(f"hp_previous_script11_rf_{k}", rf[k])
    t1_prev = git_csv(HP_PLAN_COMMIT, TABLE1_CSV).set_index("Model")
    for model in ("Ridge_expression", "RF_TF"):
        S.check(f"{model} of {TABLE1_CSV}@{HP_PLAN_COMMIT} is a single fit over 1,447 drugs",
                (int(t1_prev.at[model, "N_seeds"]), int(t1_prev.at[model, "N_drugs"])), (1, 1447))
        S.add(f"previous_table1_{model}_global_pearson", float(t1_prev.at[model, "Global_Pearson"]))
        S.add(f"previous_table1_{model}_median_perdrug", float(t1_prev.at[model, "Median_PerDrug_Pearson"]))
        S.add(f"previous_table1_{model}_pct_gt_03", float(t1_prev.at[model, "Pct_Drugs_GT_0.3"]))

    where = f"{OPTUNA_SCRIPT}@{OPTUNA_COMMIT}"
    stat = git("show", OPTUNA_COMMIT, "--stat", "--format=%h %ad %s", "--date=short")
    src = git("show", f"{OPTUNA_COMMIT}:{OPTUNA_SCRIPT}")
    S.add("optuna06_commit", stat.splitlines()[0])
    S.add("optuna06_commit_files", " ".join(ln.split("|")[0].strip() for ln in stat.splitlines() if "|" in ln))
    S.add("optuna06_script_deleted_in", removed_in(OPTUNA_SCRIPT))
    S.add("optuna06_n_trials_baseline", int(number(src, r"^N_TRIALS_BASELINE\s*=\s*(\d+)", where)))
    S.add("optuna06_n_trials_gnn", int(number(src, r"^N_TRIALS_GNN\s*=\s*(\d+)", where)))
    S.add("optuna06_search_space", " | ".join(search_space(src)))

    # Objective of the baselines, from the fragments that define it
    require(src, ('study = optuna.create_study(direction="maximize")', "return np.median(pearson_scores)",
                  "drug_counts.head(max_drugs)", "rng.choice(len(shared), n_test, replace=False)",
                  "pr, _ = pearsonr(y_test[test_mask], y_pred)"), where)
    seed = int(number(src, r"^SEED\s*=\s*(\d+)", where))
    frac = number(src, r"int\(len\(shared\) \* ([0-9.]+)\)", where)
    max_drugs = int(number(src, r"max_drugs=(\d+)", where))
    S.add("optuna06_baseline_objective",
          f"median per-drug Pearson, maximized, on the {frac:.0%} test cell lines of a random cell-line "
          f"split (RandomState({seed})), {max_drugs} most complete drugs, "
          + ("with a validation split" if has_validation(src) else "no validation split"))

    # Result table of that search, versioned in 9622e25 and later removed from git
    tuned = git_csv(OPTUNA_RESULT_COMMIT, OPTUNA_RESULT).set_index("model")
    h, _, date = commit_info(OPTUNA_RESULT_COMMIT)
    rf_keys = ("n_estimators", "max_depth", "min_samples_leaf")
    ridge_best = float(tuned.loc["Ridge_expression", "alpha"])
    rf_best = tuple(int(tuned.loc["RF_TF", k]) for k in rf_keys)
    S.check(f"script 11@{HP_PLAN_COMMIT} Ridge alpha equals the rounded Ridge_expression best of {OPTUNA_RESULT}@{h}",
            int(round(ridge_best)), ridge_alpha)
    S.check(f"script 11@{HP_PLAN_COMMIT} RF parameters equal the RF_TF best of {OPTUNA_RESULT}@{h}",
            rf_best, tuple(rf[k] for k in rf_keys))
    S.add("optuna06_result_table", f"{OPTUNA_RESULT} added in {h} ({date}), removed from git in "
                                   f"{removed_in(OPTUNA_RESULT)}")
    S.add("optuna06_best_ridge_expression", f"alpha {ridge_best:.4f}, objective "
                                            f"{tuned.loc['Ridge_expression', 'best_pearson']:.4f}")
    S.add("optuna06_best_rf_tf", ", ".join(f"{k} {v}" for k, v in zip(rf_keys, rf_best))
          + f", objective {tuned.loc['RF_TF', 'best_pearson']:.4f}")

    # Test cell lines of that split, rebuilt as in load_all_data of the script,
    # against the Table 1 hold-out split
    expr_ids = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0, usecols=[0]).index
    tf_ids = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0, usecols=[0]).index
    shared = sorted(set(expr_ids) & set(resp.index) & set(tf_ids))
    rng = np.random.RandomState(seed)
    opt_test = {shared[i] for i in rng.choice(len(shared), max(1, int(len(shared) * frac)), replace=False)}
    S.add("optuna06_baseline_test_cell_lines", len(opt_test))
    S.add("optuna06_baseline_test_cells_in_table1_test", f"{len(opt_test & set(t1_test))} of {len(t1_test)}")
    S.add("hp_previous_source_ridge_rf", f"best trials of {OPTUNA_SCRIPT} ({OPTUNA_COMMIT}), table {OPTUNA_RESULT} "
                                         f"({h}), Ridge alpha {ridge_best:.4f} rounded to {ridge_alpha}, in use in "
                                         f"script 11 until {HP_PLAN_COMMIT}")
    return tuned


def provenance_gnns(S: Summary, tuned, s05, data, node_maps, t1_test):
    """SAGE, HGT and the optimizer: scripts 06 and 07 and the defaults of script 05."""
    # First search (06) for the GNNs. Its bests are not the values in use.
    where06 = f"{OPTUNA_SCRIPT}@{OPTUNA_COMMIT}"
    src06 = git("show", f"{OPTUNA_COMMIT}:{OPTUNA_SCRIPT}")
    require(src06, ("perm = rng.permutation(n)", "n_test = int(n * 0.2)",
                    "train_p, test_p = perm[n_test:], perm[:n_test]",
                    "tl = F.huber_loss(tp, test_auc).item()", "trial.report(tl, epoch)",
                    "pr, _ = pearsonr(test_auc.cpu().numpy(), test_pred.cpu().numpy())",
                    re.compile(r'create_study\(\s*direction="maximize",\s*pruner=optuna\.pruners\.MedianPruner')),
            where06)
    S.add("optuna06_gnn_objective",
          "global Pearson at the last epoch on the 20% test edges of a random edge-level split of the PRISM "
          "response edges (RandomState(42)), "
          + ("with a validation split" if has_validation(src06) else "no validation split")
          + ". The MedianPruner is fed the test Huber loss while the study maximizes, so it treats a "
            "higher loss as better"
          + ("" if "lr_scheduler" in src06 else ". Adam without learning-rate scheduler"))
    wd06 = number(src06, r"weight_decay=([0-9.eE+-]+)", where06)
    searched06 = set(re.findall(r'suggest_\w+\("(\w+)"', src06))
    S.add("optuna06_gnn_weight_decay", f"{wd06:g}, " + ("searched" if "weight_decay" in searched06 else "fixed"))
    for name in ("SAGE_tuned", "HGT_tuned"):
        r = tuned.loc[name]
        S.add(f"optuna06_best_{name}", f"hidden {int(r['hidden_dim'])}, layers {int(r['n_layers'])}, "
              + (f"heads {int(r['n_heads'])}, " if not np.isnan(r["n_heads"]) else "")
              + f"dropout {r['dropout']:.3f}, lr {r['lr']:.5f}, epochs {int(r['epochs'])}, "
                f"objective {r['best_pearson']:.4f}")

    # Extended search (07): the SAGE architecture and the optimizer in use
    where07 = f"{OPTUNA_EXT_SCRIPT}@{OPTUNA_EXT_COMMIT}"
    src07 = git("show", f"{OPTUNA_EXT_COMMIT}:{OPTUNA_EXT_SCRIPT}")
    require(src07, ("rng = np.random.RandomState(SEED)", "perm = rng.permutation(ei.shape[1])",
                    '"train": (ei[0, perm[n_test:]]', '"test": (ei[0, perm[:n_test]]',
                    "tp = model(x, mp, test_c, test_d)",
                    "pr, _ = pearsonr(test_a.cpu().numpy(), tp.cpu().numpy())",
                    "best_pr = max(best_pr, pr)", "return best_pr", "CosineAnnealingLR(opt, T_max=epochs",
                    re.compile(r'create_study\(direction="maximize",\s*pruner=optuna\.pruners\.MedianPruner')),
            where07)
    seed07 = int(number(src07, r"^SEED\s*=\s*(\d+)", where07))
    frac07 = number(src07, r"n_test = int\(ei\.shape\[1\] \* ([0-9.]+)\)", where07)
    every07 = int(number(src07, r"if ep % (\d+) == 0", where07))
    wd07 = number(src07, r"weight_decay=([0-9.eE+-]+)", where07)
    eta07 = number(src07, r"eta_min=([0-9.eE+-]+)", where07)
    searched07 = set(re.findall(r'suggest_\w+\("(\w+)"', src07))
    S.check(f"weight decay of {where07} equals T.OPTIM", wd07, T.OPTIM["weight_decay"])
    S.check(f"cosine eta_min of {where07} equals T.OPTIM", eta07, T.OPTIM["eta_min"])
    S.check(f"weight decay and eta_min fixed (not searched) in {where07}",
            sorted(searched07 & {"weight_decay", "eta_min"}), [])
    h07, ts07, date07 = commit_info(OPTUNA_EXT_COMMIT)
    S.add("optuna07_commit", f"{h07} {date07}, results removed from git in {removed_in(OPTUNA_EXT_RESULT)}")
    S.add("optuna07_n_trials_per_model", int(number(src07, r"^N_TRIALS\s*=\s*(\d+)", where07)))
    S.add("optuna07_search_space", " | ".join(search_space(src07)))
    S.add("optuna07_fixed", f"Adam weight decay {wd07:g}, cosine annealing to {eta07:g} over the trial epochs")
    S.add("optuna07_gnn_objective",
          f"maximum, over the evaluations every {every07} epochs, of the global Pearson on the {frac07:.0%} "
          f"test edges of a random edge-level split of the PRISM response edges (RandomState({seed07})), "
          + ("with a validation split" if has_validation(src07) else
             "no validation split, so the test edges both score the trial and pick its best epoch"))
    S.add("optuna07_pruner", "MedianPruner fed with trial.report" if "trial.report(" in src07 else
          "MedianPruner configured but trial.report is never called, so no trial can be pruned")

    ext = git_csv(OPTUNA_EXT_COMMIT, OPTUNA_EXT_RESULT).set_index("model")
    for name in ("SAGE", "HGT"):
        trials = git_csv(OPTUNA_EXT_COMMIT, OPTUNA_EXT_TRIALS.format(name))
        S.check(f"best {name} trial of {h07} equals its comparison table",
                float(trials["value"].max()), float(ext.loc[f"{name}_ext", "best_pearson"]), 1e-12)
        S.add(f"optuna07_{name}_trial_states",
              ", ".join(f"{k} {v}" for k, v in trials["state"].value_counts().items()))
        r = ext.loc[f"{name}_ext"]
        S.add(f"optuna07_best_{name}_ext", f"hidden {int(r['hidden_dim'])}, layers {int(r['n_layers'])}, "
              + (f"heads {int(r['n_heads'])}, " if not np.isnan(r["n_heads"]) else "")
              + f"dropout {r['dropout']:.4f}, lr {r['lr']:.7f}, epochs {int(r['epochs'])}, "
                f"objective {r['best_pearson']:.4f}")
    sage, arch = ext.loc["SAGE_ext"], T.ARCHS["sage"]
    S.check("SAGE hidden_dim equals the SAGE_ext best", int(sage["hidden_dim"]), arch["hidden_dim"])
    S.check("SAGE n_layers equals the SAGE_ext best", int(sage["n_layers"]), arch["n_layers"])
    S.check("SAGE dropout equals the SAGE_ext best rounded to 2 decimals", round(float(sage["dropout"]), 2),
            arch["dropout"], 1e-12)
    S.check("GNN lr equals the SAGE_ext best rounded to 2 significant digits", float(f"{sage['lr']:.2g}"),
            T.OPTIM["lr"], 1e-12)
    S.check("GNN epochs equal the SAGE_ext best", int(sage["epochs"]), T.OPTIM["epochs"])

    # The commit that first hard-coded these values in script 11
    where11 = f"{SCRIPT_11.as_posix()}@{HARDCODE_COMMIT}"
    src11 = git("show", f"{HARDCODE_COMMIT}:{SCRIPT_11.as_posix()}")
    hgt = T.ARCHS["hgt"]
    require(src11, (f'"hidden_dim": {arch["hidden_dim"]}, "n_layers": {arch["n_layers"]}',
                    f'"dropout": {arch["dropout"]}, "conv_type": "sage"',
                    f'"hidden_dim": {hgt["hidden_dim"]}, "n_layers": {hgt["n_layers"]}',
                    f'"n_heads": {hgt["n_heads"]}, "dropout": {hgt["dropout"]:.2f}'), where11)
    S.check(f"GNN epochs of {where11} equal T.OPTIM", int(number(src11, r"^GNN_EPOCHS\s*=\s*(\d+)", where11)),
            T.OPTIM["epochs"])
    S.check(f"GNN lr of {where11} equals T.OPTIM", number(src11, r"^GNN_LR\s*=\s*([0-9.eE+-]+)", where11),
            T.OPTIM["lr"], 1e-12)
    S.check(f"weight decay of {where11} equals T.OPTIM", number(src11, r"weight_decay=([0-9.eE+-]+)", where11),
            T.OPTIM["weight_decay"])
    S.check(f"eta_min of {where11} equals T.OPTIM", number(src11, r"eta_min=([0-9.eE+-]+)", where11),
            T.OPTIM["eta_min"])
    h11, ts11, date11 = commit_info(HARDCODE_COMMIT)
    S.add("hp_first_hardcoded_in", f"{h11} {date11}, {round((ts11 - ts07) / 60)} min after {h07}")
    S.add("hp_source_sage_and_optimizer", f"best SAGE_ext trial of {OPTUNA_EXT_SCRIPT} ({h07}), lr and dropout "
                                          "rounded, weight decay and cosine floor fixed in that script")

    # HGT: defaults of script 05 before any search, trained with the optimizer tuned for SAGE
    parent = git("rev-parse", "--short", f"{OPTUNA_COMMIT}~1").strip()
    where05 = f"{SCRIPT_05.as_posix()}@{parent}"
    src05 = git("show", f"{parent}:{SCRIPT_05.as_posix()}")
    old05 = hgt_key(*(number(src05, rf"^{k}\s*=\s*([0-9.]+)", where05)
                      for k in ("HIDDEN_DIM", "N_LAYERS", "N_HEADS", "DROPOUT")))
    used = hgt_key(hgt["hidden_dim"], hgt["n_layers"], hgt["n_heads"], hgt["dropout"])
    k06 = hgt_key(*tuned.loc["HGT_tuned", ["hidden_dim", "n_layers", "n_heads", "dropout"]])
    k07 = hgt_key(*ext.loc["HGT_ext", ["hidden_dim", "n_layers", "n_heads", "dropout"]])
    S.check(f"HGT architecture in use equals the defaults of {where05} (before any Optuna search)", used, old05)
    S.check("HGT architecture in use equals the current defaults of script 05", used,
            hgt_key(s05.HIDDEN_DIM, s05.N_LAYERS, s05.N_HEADS, s05.DROPOUT))
    S.add("hgt_arch_in_use_hidden_layers_heads_dropout", fmt(used))
    S.add("hgt_equals_optuna06_best", f"{used == k06} (06 best {fmt(k06)})")
    S.add("hgt_equals_optuna07_best", f"{used == k07} (07 best {fmt(k07)})")
    S.add("hp_source_hgt", f"defaults of script 05 at {parent}, before any Optuna search, trained with the "
                           "optimizer of the SAGE_ext best")

    # Overlap of the edge-level split of the GNN searches with the Table 1
    # hold-out. prepare_edges of script 05 builds the same permutation as 06 and 07.
    S.check("script 05 split seed equals the seed of script 07", s05.SEED, seed07)
    S.check("script 05 test fraction equals the one of script 07", s05.TEST_FRAC, frac07, 1e-12)
    train_c, _, _, test_c, _, _ = s05.prepare_edges(data)
    train_c, test_c = train_c.cpu().numpy(), test_c.cpu().numpy()
    cell_map = node_maps["cell_line"]
    missing = [c for c in t1_test if c not in cell_map]
    if missing:
        raise ValueError(f"Table 1 test cell lines missing from the graph: {missing}")
    t1_idx = np.array([cell_map[c] for c in t1_test])
    S.add("gnn_search_split_table1_test_lines_with_training_edges",
          f"{int(np.isin(t1_idx, train_c).sum())} of {len(t1_idx)}")
    S.add("gnn_search_split_table1_test_lines_with_objective_edges",
          f"{int(np.isin(t1_idx, test_c).sum())} of {len(t1_idx)}")
    S.add("gnn_search_split_training_edges_from_table1_test_lines_pct",
          round(100 * float(np.isin(train_c, t1_idx).mean()), 1))
    S.add("gnn_search_split_objective_edges_from_table1_test_lines_pct",
          round(100 * float(np.isin(test_c, t1_idx).mean()), 1))
    mtime = GRAPH_PATH.stat().st_mtime
    S.add("gnn_search_split_graph_file",
          f"{GRAPH_PATH.as_posix()} modified {datetime.fromtimestamp(mtime).isoformat(timespec='seconds')}, "
          + (f"after {h07}, so the searched edge order may differ" if mtime > ts07 else f"before {h07}"))


def hyperparameters(S: Summary, s05, data, node_maps):
    logger.info("[10] Hyperparameters and their provenance")
    ridge_alpha, rf = hyperparameters_in_use(S)
    resp = normalize_drug_columns(load_prism_response(use_processed=True))
    _, t1_test = cell_line_holdout_split(resp, test_frac=T.TEST_FRAC, seed=SEED)
    S.check("Table 1 test cell lines", len(t1_test), PUB["holdout_n_test_cells"])
    provenance_search(S, ridge_alpha, rf, t1_test)
    tuned = provenance_baselines(S, resp, t1_test)
    provenance_gnns(S, tuned, s05, data, node_maps, t1_test)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    S = Summary()
    survival_multiplicity(S)
    ablation = ablation_medians(S)
    coverage = pam50_coverage(S)
    s05 = load_script(SCRIPT_05, "train_hgt_and_crossval")
    data, node_maps = load_hetero_data(GRAPH_PATH)
    gdsc_crossscreen(S, s05, data, node_maps)
    aurora_epirubicin(S)
    clinical_olaparib(S)
    if history_available():
        hyperparameters(S, s05, data, node_maps)
    else:
        # The public package does not carry the development history. Section [10] reads it,
        # so its rows are kept from the summary written where that history exists.
        prev = OUT_DIR / "multiplicity_summary.csv"
        if not prev.exists():
            raise FileNotFoundError(f"{prev} is needed to keep section [10] without the development history")
        carried = pd.read_csv(prev, dtype=str, keep_default_na=False).iloc[len(S.rows):]
        logger.warning("[10] skipped: commit %s of the development history is not in this repository, "
                       "%d rows kept from %s", OPTUNA_COMMIT, len(carried), prev)
        S.rows.extend(carried.to_dict("records"))

    pd.DataFrame(S.rows).to_csv(OUT_DIR / "multiplicity_summary.csv", index=False)
    ablation.to_csv(OUT_DIR / "ablation_perdrug_median.csv", index=False)
    coverage.to_csv(OUT_DIR / "metabric_pam50_coverage.csv", index=False)
    logger.info("%d reproduction checks passed", len(S.checks))
    logger.info("Wrote %s, %s and %s", OUT_DIR / "multiplicity_summary.csv",
                OUT_DIR / "ablation_perdrug_median.csv", OUT_DIR / "metabric_pam50_coverage.csv")


if __name__ == "__main__":
    main()
