#!/usr/bin/env python3
"""
PRECISION - Unified pipeline (Paper 1).

Runs the complete reproducible pipeline from the raw PRISM/DepMap/CollecTRI
inputs to the final paper CSVs, statistics and figures, replacing the ad-hoc
sequence of numbered scripts under scripts/ with a single orchestrator.

Stages (dependencies flow top -> bottom; the bracket names the script under
scripts/ or paper/scripts/ that the stage runs):

  data            [01]  data preprocessing + TF activities via ULM (PRISM, GDSC)
  data_r_scanb    [09]  SCAN-B TF activities + clinical table (R)
  data_r_cohorts  [20]  METABRIC + TCGA TF activities + clinical tables (R)
  graph           [02]  heterogeneous knowledge graph construction
  graph_summary   [51]  node and edge counts of the prebuilt graph (Methods, Figure 1)
  crossval        [05]  HGT variant + PRISM to GDSC cross-screen validation (SFig07)
  baselines       [11]  Ridge / RF / SAGE GNN / HGT on cell-line hold-out (Table 1)
  embeddings      [12]  Ridge(expression + GNN emb) hybrid row (Table 1)
  xai             [13]  trains the SAGE checkpoint used for Integrated Gradients
                        (its ablation-based TF ranking is illustrative only)
  training_history [50] SAGE training curves, per-epoch loss and test Pearson (SFig04)
  multi_seed      [34]  SAGE retrained with seeds 42, 123, 456 (checkpoints for 35)
  xai_ig          [32]  Integrated Gradients TF attributions (canonical XAI)
  ig_baselines    [33]  IG robustness to the baseline choice (mean, zero, random)
  ig_multi_seed   [35]  IG per seed + cross-seed stability (Figures 4, 5)
  transfer        [17]  per-drug Ridge SCAN-B transfer + survival Cox
  basal_her2      [19b] within-subtype survival screens, Basal and Basal + HER2 (SFig12)
  multicohort     [21]  multi-cohort survival + Fisher meta-analysis (Tables 2, 3)
  enrichment      [52]  MOA and target enrichment among Fisher-significant drugs (SFig06, SFig07)
  cox_pam50       [31]  Cox multivariate PAM50-adjusted, 3 penalizers (Table 5)
  pca             [42]  PCA + PC1 residualization: shared prognostic axis
  forest          [43]  positive-control HR + 95% CI, SCAN-B vs METABRIC (SFig10)
  tf_cox          [44]  TF-level univariate Cox on SCAN-B OS
  penalizer       [45]  Cox penalizer sweep (SFig11, Discussion)
  aurora          [22]  AURORA TF activities + drug sensitivity, primary vs metastasis
  aurora_ig       [36]  paired primary vs metastasis test of the 10 IG TFs (Table 4, Fig 9)
  strengthen      [18]  subtype-stratified survival (tests 1 and 2 need legacy inputs)
  weaknesses      [19]  D1 TF subset, D2 Basal enrichment, D3 penalized Cox
  ablation        [27]  graph component ablation (SFig17)
  sensitivity     [37, 39, 40, 41]  clinical-covariate Cox, AURORA by site,
                        RFS endpoint, ComBat vs QuantileTransformer
  paper           [generate_paper_results.py]  paper CSVs + paper_statistics.json
                        + sanity checks (Table 5 rule, Supplementary Table S1)
  xai_ig_extended, ig_multi_seed_extended, aurora_ig_extended, ig_extended_comparison
                  [32, 35, 36 with --candidate-set extended, 49]  IG repeated on
                        the extended candidate set (Supplementary Note 1)
  figures         [generate_paper_figures.py]  main + supplementary figures
                        (matplotlib)

Usage:
  python pipeline.py --stage all
  python pipeline.py --stage xai transfer aurora
  python pipeline.py --stage paper        # just regenerate tables
  python pipeline.py --list

Conventions:
  - All stages are idempotent: a stage is skipped when every declared output
    already exists (unless --force).
  - Seeds are fixed (SEED=42) in every stage.
  - Locations come from config.py: DATA_DIR and RESULTS_DIR default to data/
    and results/ under the package root and can be overridden with the
    PRECISION_DATA and PRECISION_RESULTS environment variables. Both are
    exported to every child process so that R scripts can read them too.
  - Intermediate CSVs live under RESULTS_DIR/v*/ (historical folder names are
    kept so that every result can be traced back to the manuscript).
  - Paper-facing CSVs live under paper/results/ and are the authoritative
    source for every number cited in the manuscript.
  - R stages need Rscript on PATH, or the RSCRIPT environment variable
    pointing to the binary.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Anchor imports to the package root regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, FIG_DIR, PAPER_RESULTS, RESULTS_DIR, ROOT, SFIG_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pipeline")

SCRIPTS = ROOT / "scripts"
PAPER_SCRIPTS = ROOT / "paper" / "scripts"
PROCESSED = DATA_DIR / "precision_processed"
GRAPH_DIR = DATA_DIR / "precision_graph"

# -----------------------------------------------------------------------------
# Stage registry: name -> (scripts to run in order, expected output files)
# -----------------------------------------------------------------------------
STAGES: dict[str, dict] = {
    "data": {
        "scripts": [SCRIPTS / "01_prepare_data.py"],
        "outputs": [PROCESSED / "tf_activities_all_prism.csv",
                    PROCESSED / "tf_activities_all_gdsc.csv",
                    PROCESSED / "drug_target_edges.csv"],
        "description": "Preprocess PRISM/GDSC/DepMap + TF activities via ULM (Python)",
    },
    "data_r_scanb": {
        "scripts": [SCRIPTS / "09_scanb_transfer.R"],
        "outputs": [PROCESSED / "tf_activities_scanb.csv",
                    PROCESSED / "scanb_clinical.csv"],
        "description": "SCAN-B TF activities + clinical table (R, decoupleR)",
    },
    "data_r_cohorts": {
        "scripts": [SCRIPTS / "20_multicohort_validation.R"],
        "outputs": [PROCESSED / "tf_activities_metabric.csv",
                    PROCESSED / "metabric_clinical.csv",
                    PROCESSED / "tf_activities_tcga.csv",
                    PROCESSED / "tcga_clinical.csv"],
        "description": "METABRIC + TCGA TF activities + clinical tables (R, decoupleR + genefu)",
    },
    "graph": {
        "scripts": [SCRIPTS / "02_build_graph.py"],
        "outputs": [GRAPH_DIR / "hetero_graph_prism.pkl"],
        "description": "Build heterogeneous knowledge graph (23,498 nodes)",
    },
    "graph_summary": {
        "scripts": [SCRIPTS / "51_graph_summary.py"],
        "outputs": [RESULTS_DIR / "graph_summary.csv"],
        "description": "Node and edge counts of the prebuilt graph (Methods, Figure 1)",
    },
    "data_summary": {
        "scripts": [SCRIPTS / "53_data_summary.py"],
        "outputs": [RESULTS_DIR / "data_summary.csv"],
        "description": "Data constants cited in Methods: PRISM/GDSC/expression sizes, MOA encoding, hold-out split",
    },
    "crossval": {
        "scripts": [SCRIPTS / "05_train_hgt_and_crossval.py"],
        "outputs": [RESULTS_DIR / "crossval_prism_to_gdsc.csv"],
        "description": "HGT variant + PRISM to GDSC cross-screen validation (SFig07)",
    },
    "baselines": {
        "scripts": [SCRIPTS / "11_rerun_fair_comparison.py"],
        "outputs": [RESULTS_DIR / "v2_fair" / "fair_comparison.csv",
                    RESULTS_DIR / "v2_fair" / "best_sage_v2.pt",
                    RESULTS_DIR / "v2_fair" / "per_drug_SAGE_learned_emb.csv"],
        "description": "Fair cell-line hold-out: Ridge + RF + SAGE + HGT (Table 1)",
    },
    "embeddings": {
        "scripts": [SCRIPTS / "12_gnn_embeddings_perdrug.py"],
        "outputs": [RESULTS_DIR / "v3_embeddings" / "embedding_comparison.csv",
                    RESULTS_DIR / "v3_embeddings" / "cell_line_embeddings.csv"],
        "description": "Ridge(expression+GNN emb) hybrid baseline (Table 1 row)",
    },
    "xai": {
        "scripts": [SCRIPTS / "13_xai_v2_and_scanb.py"],
        # best_sage_v4.pt is the checkpoint Integrated Gradients must run on: the
        # stage is considered fresh when it exists so that a rerun does not
        # silently retrain it. The script also writes the legacy ablation ranking
        # (tf_importance_bootstrap1000.csv), which is not needed downstream.
        "outputs": [RESULTS_DIR / "v4_xai" / "best_sage_v4.pt",
                    RESULTS_DIR / "v4_xai" / "drug_explanations_v2.csv",
                    RESULTS_DIR / "v4_xai" / "tnbc_drug_ranking_v2.csv"],
        "description": "Train the SAGE checkpoint for IG (best_sage_v4.pt) + illustrative ablation ranking",
    },
    "training_history": {
        "scripts": [SCRIPTS / "50_training_history.py"],
        "outputs": [RESULTS_DIR / "gnn_training_history.csv"],
        "description": "SAGE training curves with the protocol of script 13 (SFig04)",
    },
    "multi_seed": {
        "scripts": [SCRIPTS / "34_multi_seed_gnn.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "best_sage_seed42.pt",
                    RESULTS_DIR / "v4_xai" / "best_sage_seed123.pt",
                    RESULTS_DIR / "v4_xai" / "best_sage_seed456.pt"],
        "description": "SAGE retrained with seeds 42, 123, 456 (checkpoints for ig_multi_seed)",
    },
    "xai_ig": {
        "scripts": [SCRIPTS / "32_xai_ig.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_global.csv",
                    RESULTS_DIR / "v4_xai" / "tf_drug_ig_signed.csv",
                    RESULTS_DIR / "v4_xai" / "ig_full_attributions.csv"],
        "description": "TF importance via Integrated Gradients (Captum, 50 steps)",
    },
    "ig_baselines": {
        "scripts": [SCRIPTS / "33_ig_baseline_robustness.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "ig_baseline_mean.csv",
                    RESULTS_DIR / "v4_xai" / "ig_baseline_zero.csv",
                    RESULTS_DIR / "v4_xai" / "ig_baseline_random.csv",
                    RESULTS_DIR / "v4_xai" / "ig_baseline_stability.csv"],
        "description": "IG robustness to the baseline choice (mean, zero, random)",
    },
    "ig_multi_seed": {
        "scripts": [SCRIPTS / "35_ig_multi_seed.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed42.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed123.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed456.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv"],
        "description": "IG per seed + cross-seed stability (Figures 4, 5, Table 4 TF set)",
    },
    "transfer": {
        "scripts": [SCRIPTS / "17_fix_all_critical.py"],
        "outputs": [RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv",
                    RESULTS_DIR / "v5_final" / "scanb_survival_perdrug.csv"],
        "description": "SCAN-B per-drug Ridge + survival Cox (feeds multicohort)",
    },
    "basal_her2": {
        "scripts": [SCRIPTS / "19b_basal_her2_pooled.py"],
        "outputs": [RESULTS_DIR / "v7_weaknesses" / "basal_only_survival.csv",
                    RESULTS_DIR / "v7_weaknesses" / "basal_her2_survival.csv",
                    RESULTS_DIR / "v7_weaknesses" / "subtype_pooling_summary.csv"],
        "description": "Within-subtype survival screens: Basal alone and Basal + HER2 pool (SFig12)",
    },
    "multicohort": {
        "scripts": [SCRIPTS / "21_multicohort_survival.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "survival_SCANB.csv",
                    RESULTS_DIR / "v8_multicohort" / "survival_METABRIC.csv",
                    RESULTS_DIR / "v8_multicohort" / "survival_TCGA.csv",
                    RESULTS_DIR / "v8_multicohort" / "meta_analysis_fisher.csv",
                    RESULTS_DIR / "v8_multicohort" / "cohort_summary.csv"],
        "description": "Multi-cohort survival + Fisher meta + cohort sizes (Tables 2, 3)",
    },
    "enrichment": {
        "scripts": [SCRIPTS / "52_drug_enrichment.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "moa_enrichment.csv",
                    RESULTS_DIR / "v8_multicohort" / "target_enrichment.csv"],
        "description": "MOA and target enrichment among Fisher-significant drugs (SFig06, SFig07)",
    },
    "cox_pam50": {
        "scripts": [SCRIPTS / "31_cox_multivar_pam50.py"],
        "outputs": [RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv",
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_pen05.csv",
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_pen001.csv"],
        "description": "Cox multivariate PAM50-adjusted, 3 penalizers (Table 5)",
    },
    "pca": {
        "scripts": [SCRIPTS / "42_pca_residualization.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "pca_residualization_summary.csv",
                    RESULTS_DIR / "v8_multicohort" / "pca_residualization_perdrug.csv"],
        "description": "PCA + PC1 residualization: shared prognostic axis (Discussion)",
    },
    "forest": {
        "scripts": [SCRIPTS / "43_forest_positive_controls.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "forest_positive_controls.csv"],
        "description": "Positive-control HR + 95% CI, SCAN-B vs METABRIC (SFig10)",
    },
    "tf_cox": {
        "scripts": [SCRIPTS / "44_tf_level_cox.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "tf_level_survival_scanb.csv"],
        "description": "TF-level univariate Cox on SCAN-B OS (Results MYC/TP53/E2F1)",
    },
    "penalizer": {
        "scripts": [SCRIPTS / "45_penalizer_sweep.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "penalizer_sweep.csv"],
        "description": "Cox penalizer sweep: HR-range + FDR counts (SFig11, Discussion)",
    },
    "aurora": {
        "scripts": [SCRIPTS / "22_aurora_complementary.py"],
        "outputs": [RESULTS_DIR / "v9_aurora" / "aurora_tf_activities.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_primary_vs_meta_tfs.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_cohort_summary.csv"],
        "description": "AURORA TF activities + drug sensitivity, primary vs metastasis (SFig13)",
    },
    "aurora_ig": {
        "scripts": [SCRIPTS / "36_aurora_ig_validation.py"],
        "outputs": [RESULTS_DIR / "v9_aurora" / "aurora_tf_ig_validation.csv"],
        "description": "Paired primary vs metastasis test of the 10 IG TFs (Table 4, Fig 9)",
    },
    "xai_ig_extended": {
        "scripts": [(SCRIPTS / "32_xai_ig.py", ["--candidate-set", "extended"])],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_global_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_drug_ig_signed_extended.csv"],
        "description": "IG on the extended candidate set, seed-42 model (Supplementary Note 1)",
    },
    "ig_multi_seed_extended": {
        "scripts": [(SCRIPTS / "35_ig_multi_seed.py", ["--candidate-set", "extended"])],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed42_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed123_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed456_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_multi_seed_stability_extended.csv"],
        "description": "IG per seed on the extended candidate set (Supplementary Note 1)",
    },
    "aurora_ig_extended": {
        "scripts": [(SCRIPTS / "36_aurora_ig_validation.py", ["--candidate-set", "extended"])],
        "outputs": [RESULTS_DIR / "v9_aurora" / "aurora_tf_ig_validation_extended.csv"],
        "description": "AURORA validation of the extended core TFs (Supplementary Note 1)",
    },
    "ig_extended_comparison": {
        "scripts": [SCRIPTS / "49_ig_extended_comparison.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "ig_extended_comparison.csv",
                    RESULTS_DIR / "v4_xai" / "ig_extended_core_tfs.csv"],
        "description": "Main vs extended IG runs: top-20 overlap, core TFs, AURORA (Supplementary Note 1)",
    },
    "strengthen": {
        "scripts": [SCRIPTS / "18_strengthen_paper.py"],
        "outputs": [RESULTS_DIR / "v6_strengthen" / "test3_subtype_summary.csv"],
        "description": "Subtype-stratified SCAN-B survival (test 3); tests 1 and 2 need legacy inputs",
    },
    "weaknesses": {
        "scripts": [SCRIPTS / "19_fix_three_weaknesses.py"],
        "outputs": [RESULTS_DIR / "v7_weaknesses" / "d2_basal_enrichment.csv",
                    RESULTS_DIR / "v7_weaknesses" / "d3_penalized_cox.csv"],
        "description": "D2 Basal enrichment, D3 penalized Cox (D1 needs legacy inputs)",
    },
    "ablation": {
        "scripts": [SCRIPTS / "27_graph_ablation.py"],
        "outputs": [RESULTS_DIR / "v6_strengthen" / "graph_ablation.csv"],
        "description": "Graph component ablation (SFig17)",
    },
    "sensitivity": {
        "scripts": [SCRIPTS / "37_cox_clinical_covariates.py",
                    SCRIPTS / "39_aurora_site_stratified.py",
                    SCRIPTS / "40_rfs_sensitivity.py",
                    SCRIPTS / "41_combat_sensitivity.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "cox_clinical_comparison.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_tf_by_site.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_tf_by_site_summary.csv",
                    RESULTS_DIR / "v8_multicohort" / "cox_rfs_comparison.csv",
                    RESULTS_DIR / "v8_multicohort" / "combat_vs_qt_summary.csv"],
        "description": "Sensitivity analyses: clinical-covariate Cox, AURORA by site, "
                       "RFS endpoint, ComBat vs QuantileTransformer (37, 39, 40, 41)",
    },
    "paper": {
        "scripts": [PAPER_SCRIPTS / "generate_paper_results.py"],
        "outputs": [PAPER_RESULTS / "paper_statistics.json",
                    PAPER_RESULTS / "table1_model_comparison.csv",
                    PAPER_RESULTS / "table2_multicohort_summary.csv",
                    PAPER_RESULTS / "table3_positive_controls.csv",
                    PAPER_RESULTS / "table4_aurora_tfs.csv",
                    PAPER_RESULTS / "table5_candidates.csv",
                    PAPER_RESULTS / "tf_importance_ig_top20.csv",
                    PAPER_RESULTS / "shared_drugs.csv"],
        "description": "Generate paper CSVs + paper_statistics.json + sanity checks",
    },
    "figures": {
        "scripts": [PAPER_SCRIPTS / "generate_paper_figures.py"],
        "outputs": [FIG_DIR / "Figure1.png",
                    FIG_DIR / "Figure10.png",
                    SFIG_DIR / "FigureS11.png"],
        "description": "Generate paper figures with matplotlib (main + supplementary)",
    },
}

# Default order (respects dependencies)
DEFAULT_ORDER = [
    "data", "data_r_scanb", "data_r_cohorts", "graph", "graph_summary", "data_summary", "crossval",
    "baselines",
    "embeddings", "xai", "training_history", "multi_seed", "xai_ig", "ig_baselines", "ig_multi_seed",
    "transfer", "basal_her2", "multicohort", "enrichment", "cox_pam50", "pca", "forest",
    "tf_cox", "penalizer", "aurora", "aurora_ig", "strengthen", "weaknesses",
    "ablation", "sensitivity", "paper",
    "xai_ig_extended", "ig_multi_seed_extended", "aurora_ig_extended",
    "ig_extended_comparison", "figures",
]

# Stages that read raw inputs which are not redistributed with the package
# (see data/README.md). They are skipped whenever their outputs are present.
RAW_INPUT_STAGES = {
    "data": "the raw PRISM/GDSC/DepMap inputs",
    "data_r_scanb": "the SCAN-B expression and clinical sources",
    "data_r_cohorts": "the METABRIC and TCGA expression and clinical sources",
}

# A failure in one of these leaves nothing for the downstream stages to use.
CRITICAL_STAGES = {"data", "data_r_scanb", "data_r_cohorts", "graph",
                   "baselines", "xai", "transfer"}


def _outputs_fresh(stage: dict) -> bool:
    """Return True if all declared stage outputs exist."""
    return all(Path(p).exists() for p in stage["outputs"])


def _child_env() -> dict[str, str]:
    """Environment for child processes. The resolved data and results
    locations are exported so that R scripts can read them with Sys.getenv();
    values already set by the user are left untouched."""
    env = dict(os.environ)
    env.setdefault("PRECISION_DATA", str(DATA_DIR))
    env.setdefault("PRECISION_RESULTS", str(RESULTS_DIR))
    return env


def _find_rscript() -> str | None:
    """Locate the Rscript binary: the RSCRIPT environment variable wins,
    then PATH. Returns None when R is not available."""
    override = os.environ.get("RSCRIPT")
    if override:
        return override if Path(override).exists() else shutil.which(override)
    return shutil.which("Rscript")


def _run_python(script: Path, args: list[str]) -> int:
    """Run a Python script with the current interpreter from the package
    root. Stdout is inherited so the user sees progress. Returns exit code."""
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_child_env())
    return proc.returncode


def _run_rscript(script: Path, args: list[str]) -> int:
    """Run an R script with Rscript from the package root. Fails with a
    clear message when R is not installed or cannot be found."""
    rscript = _find_rscript()
    if rscript is None:
        logger.error(
            "Rscript not found: %s needs R (>= 4.5). Add the R 'bin' directory "
            "to PATH or set RSCRIPT to the full path of the Rscript binary "
            "(e.g. RSCRIPT=C:/Program Files/R/R-4.5.2/bin/Rscript.exe on "
            "Windows, RSCRIPT=/usr/bin/Rscript on Linux).",
            script.name)
        return 127
    cmd = [rscript, str(script), *args]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=_child_env())
    return proc.returncode


def _run_script(entry) -> int:
    """Dispatch on the script extension: .py -> Python, .R -> Rscript.
    A stage entry is either a Path or a (Path, [arguments]) tuple."""
    script, args = (entry, []) if isinstance(entry, Path) else (entry[0], list(entry[1]))
    if not script.exists():
        logger.error("Script not found: %s", script)
        return 127
    logger.info("running: %s %s", script.relative_to(ROOT), " ".join(args))
    if script.suffix.lower() == ".r":
        return _run_rscript(script, args)
    return _run_python(script, args)


def _run_stage(name: str, force: bool, skip_missing_data: bool) -> int:
    stage = STAGES[name]
    logger.info("=" * 60)
    logger.info("STAGE %s: %s", name, stage["description"])
    logger.info("=" * 60)

    if not force and _outputs_fresh(stage):
        logger.info("  outputs already fresh, skipping (use --force to re-run)")
        return 0

    if name in RAW_INPUT_STAGES and skip_missing_data:
        if _outputs_fresh(stage):
            logger.info("  preprocessed data present, skipping stage '%s'", name)
            return 0
        logger.warning("  preprocessed data not found; stage '%s' requires %s, "
                       "which are not redistributed with this package (see "
                       "data/README.md). Either run this stage on a machine "
                       "with access to those inputs, or obtain the "
                       "preprocessed CSVs.", name, RAW_INPUT_STAGES[name])

    for script in stage["scripts"]:
        rc = _run_script(script)
        if rc != 0:
            logger.error("  stage %s failed at %s with exit code %d",
                         name, script.name, rc)
            return rc
    logger.info("  stage %s OK", name)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PRECISION unified pipeline")
    ap.add_argument(
        "--stage", nargs="+", default=["all"],
        help="Stage(s) to run. Use 'all' for the default sequence, or one or "
             "more of: " + ", ".join(DEFAULT_ORDER),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if their outputs already exist.",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List available stages and exit.",
    )
    args = ap.parse_args()

    if args.list:
        print("\nAvailable stages (in default order):\n")
        for name in DEFAULT_ORDER:
            print(f"  {name:<15} {STAGES[name]['description']}")
        return 0

    if args.stage == ["all"]:
        order = DEFAULT_ORDER
    else:
        unknown = [s for s in args.stage if s not in STAGES]
        if unknown:
            logger.error("unknown stages: %s", unknown)
            logger.info("use --list to see available stages")
            return 1
        order = args.stage

    failures = []
    for name in order:
        rc = _run_stage(name, force=args.force, skip_missing_data=True)
        if rc != 0:
            failures.append(name)
            if name in CRITICAL_STAGES:
                logger.error("aborting: downstream stages depend on %s", name)
                break

    if failures:
        logger.error("FAILED stages: %s", failures)
        return 2

    logger.info("PIPELINE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
