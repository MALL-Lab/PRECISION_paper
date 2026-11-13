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
  gnn_train       [57]  trains every GNN of the paper once (results/v12_gnn/):
                        SAGE and HGT with 5 initialization seeds (Table 1), the
                        SAGE models of splits 123 and 456 (multi-seed IG) and the
                        SAGE graph ablation. Every later stage that needs a GNN
                        loads these checkpoints instead of training its own
  graph_summary   [51]  node and edge counts of the prebuilt graph (Methods, Figure 1)
  data_summary    [53]  data constants cited in Methods
  crossval        [05]  HGT variant + PRISM to GDSC cross-screen validation
                        (Supplementary Figure 14)
  baselines       [11]  Ridge / RF + the SAGE and HGT runs of gnn_train on the
                        cell-line hold-out (Table 1)
  embeddings      [12]  Ridge(expression + GNN emb) hybrid row (Table 1)
  xai_ig          [32]  Integrated Gradients TF attributions on the canonical
                        SAGE run (canonical XAI)
  ig_baselines    [33]  IG robustness to the baseline choice (mean, zero, random)
  ig_multi_seed   [35]  IG per seed + cross-seed stability (Figure 3)
  transfer        [17]  per-drug Ridge SCAN-B transfer + survival Cox
  basal_her2      [19b] within-subtype survival screens, Basal and Basal + HER2
                        (Supplementary Figure 13)
  multicohort     [21]  multi-cohort survival + Fisher meta-analysis (Tables 2, 3)
  km_logrank      [58]  log-rank tests of the SCAN-B Kaplan-Meier panels (Figure 4A)
                        and their extras (10-year survival, PAM50-stratified and
                        quintile trend tests, numbers at risk), which reproduce
                        survival_SCANB.csv and so run after multicohort
  enrichment      [52]  MOA and target enrichment among Fisher-significant drugs
                        (Supplementary Figures 8, 9)
  cox_pam50       [31]  Cox multivariate PAM50-adjusted, 3 penalizers (Table 5)
  pca             [42]  PCA + PC1 residualization: shared prognostic axis, and the
                        orientation of PC1 (results/v13_reaudit/pc1_orientation.csv)
  forest          [43]  positive-control HR + 95% CI, SCAN-B vs METABRIC
                        (Supplementary Figure 7)
  tf_cox          [44]  TF-level univariate Cox on SCAN-B OS
  penalizer       [45]  Cox penalizer sweep (Supplementary Figure 11, Discussion)
  aurora          [22]  AURORA TF activities + drug sensitivity, primary vs metastasis
                        (inputs of Figure 7 and Table 4, per-sample drug comparison kept as a check)
  aurora_ig       [36]  paired primary vs metastasis test of the 10 IG TFs (Table 4, Figure 7)
  strengthen      [18]  subtype-stratified survival (tests 1 and 2 need legacy inputs)
  weaknesses      [19]  D1 TF subset, D2 Basal enrichment, D3 penalized Cox
  ablation        [27]  graph component ablation from the ablation runs of
                        gnn_train (Supplementary Figure 12)
  sensitivity     [37, 39, 40, 41]  clinical-covariate Cox, AURORA by site,
                        RFS endpoint, ComBat vs QuantileTransformer
  reporting       [54, 55]  cohort demographics and ICLAC cross-check
  collectri_snapshot [56]  clinical results rerun on the CollecTRI snapshot
                        (Supplementary Note 2)
  data_r_fuscc    [68]  FUSCC-TNBC TF activities and clinical table (R, decoupleR)
  fuscc           [69]  FUSCC-TNBC: does the ordering of the drug-survival
                        associations hold in a cohort that is entirely TNBC
                        (results/v14_fuscc/)
  gene_universe   [70, 71]  what the gene universe of each cohort costs its
                        transfer, and the TCGA sensitivity of Supplementary Note 2
                        (results/v15_gene_universe/)
  reaudit         [59-67]  re-audit of September 2026 (results/v13_reaudit/):
                        reference predictors of Table 1, AURORA background rates,
                        TNBC and Basal subgroups, treatment, signed meta-analysis
                        and METABRIC DSS, proportional hazards and HR per SD, IG
                        diagnostics, multiplicity, and the activity of the
                        candidates in the TNBC cell lines of PRISM. Runs before
                        candidate_table, because generate_paper_results.py
                        requires its outputs
  candidate_table [generate_paper_results.py]  first pass of the paper stage, run
                        only for the 143 drugs of the statistical filter
                        (Supplementary Data 1, supp_table_S1_candidate_selection.csv),
                        from which the next four stages read the extended
                        candidate set
  xai_ig_extended, ig_multi_seed_extended, aurora_ig_extended, ig_extended_comparison
                  [32, 35, 36 with --candidate-set extended, 49]  IG repeated on
                        the extended candidate set (Supplementary Note 1)
  paper           [generate_paper_results.py]  paper CSVs + paper_statistics.json
                        + sanity checks (Table 5 rule, Supplementary Table S2
                        and Supplementary Data 1),
                        run after both Supplementary Notes so that their numbers
                        are included
  figures         [generate_paper_figures.py]  main + supplementary figures
                        (matplotlib)

Usage:
  python pipeline.py --stage all
  python pipeline.py --stage gnn_train xai_ig transfer aurora
  python pipeline.py --stage paper        # just regenerate tables
  python pipeline.py --list

Conventions:
  - All stages are idempotent: a stage is skipped when every declared output
    already exists (unless --force). scripts/57_train_gnns.py also keeps every
    run it finds, so an interrupted gnn_train resumes where it stopped.
  - --force applies to every listed stage, and "--stage all --force" also
    rebuilds the graph that the GNN runs were trained on. After retraining,
    name the stages that read results/v12_gnn/ and force only those.
  - Seeds are fixed in every stage: SEED=42, and for the GNNs the split and
    initialization seeds of scripts/57_train_gnns.py.
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
GNN_DIR = RESULTS_DIR / "v12_gnn"
REAUDIT_DIR = RESULTS_DIR / "v13_reaudit"
FUSCC_DIR = RESULTS_DIR / "v14_fuscc"
GENE_UNIVERSE_DIR = RESULTS_DIR / "v15_gene_universe"


def _gnn_runs() -> list[str]:
    """Run names that scripts/57_train_gnns.py --set all writes, as in its plan():
    SAGE and HGT on split 42 with the Table 1 seeds, SAGE on splits 123 and 456
    for the multi-seed IG, and SAGE on split 42 with the Table 1 seeds for each
    reduced graph of src/models/training.ABLATIONS and for the control of
    SHUFFLED, which rewires the drug-target edges.

    The list is written out instead of imported so that --list and --help do not
    load torch. It must follow TABLE1_SEEDS and IG_SEEDS of script 57 and the
    ABLATIONS and SHUFFLED tags of src/models/training.py."""
    seeds = (42, 123, 456, 789, 1011)
    ablations = ("no_ppi", "no_collectri", "no_drug_target", "no_ppi_no_drug_target",
                 "response_only", "shuffled_drug_target")
    runs = [f"{arch}_split42_init{s}_full" for arch in ("sage", "hgt") for s in seeds]
    runs += [f"sage_split{s}_init{s}_full" for s in (123, 456)]
    runs += [f"sage_split42_init{s}_{tag}" for tag in ablations for s in seeds]
    return runs


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
    # FUSCC-TNBC, a cohort that is entirely triple-negative. Its TF activities are
    # computed over the gene universe of PRISM, not the full transcriptome, because
    # the ULM statistic grows with the number of genes and the transfer clips what
    # falls outside the range it saw in training (see the script header).
    "data_r_fuscc": {
        "scripts": [SCRIPTS / "68_fuscc_tf_activities.R"],
        "outputs": [PROCESSED / "tf_activities_fuscc.csv",
                    PROCESSED / "fuscc_clinical.csv",
                    PROCESSED / "fuscc_r_summary.csv"],
        "description": "FUSCC-TNBC TF activities + clinical table (R, decoupleR)",
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
    "gnn_train": {
        "scripts": [SCRIPTS / "57_train_gnns.py"],
        # One checkpoint and one sidecar per run: the stage is fresh only when
        # every run is complete, and 57 skips the runs that already exist.
        "outputs": [GNN_DIR / "runs.csv"]
                   + [GNN_DIR / f"{run}{ext}" for run in _gnn_runs() for ext in (".pt", ".json")],
        "description": "Train every GNN once, final-epoch weights: SAGE/HGT x 5 seeds (Table 1), "
                       "SAGE splits 123/456 (IG), SAGE graph ablation and the rewired drug-target control "
                       "(about 50 min on the GPU recorded in the sidecars)",
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
        "description": "HGT variant + PRISM to GDSC cross-screen validation (Supplementary Figure 14)",
    },
    # The GNN rows of Table 1 and the GNN embeddings come from the runs of
    # gnn_train: these stages load checkpoints and never train a GNN. The
    # *_per_seed.csv outputs only exist since the five-seed protocol, so the
    # outputs of the earlier single-checkpoint versions do not count as fresh.
    # The Ridge and RF hyperparameters are constants of script 11, chosen by the search of
    # script 72 run on CESGA (results/v16_hp_search/cesga_11133811), which is not a stage.
    "baselines": {
        "scripts": [SCRIPTS / "11_rerun_fair_comparison.py"],
        "outputs": [RESULTS_DIR / "v2_fair" / "fair_comparison.csv",
                    RESULTS_DIR / "v2_fair" / "fair_comparison_per_seed.csv",
                    RESULTS_DIR / "v2_fair" / "per_drug_Ridge_expression.csv",
                    RESULTS_DIR / "v2_fair" / "per_drug_RF_TF.csv",
                    RESULTS_DIR / "v2_fair" / "per_drug_SAGE_learned_emb.csv"],
        "description": "Fair cell-line hold-out: Ridge, RF x 5 seeds (hyperparameters of script 72), "
                       "SAGE + HGT x 5 seeds from gnn_train (Table 1)",
    },
    "embeddings": {
        "scripts": [SCRIPTS / "12_gnn_embeddings_perdrug.py"],
        "outputs": [RESULTS_DIR / "v3_embeddings" / "embedding_comparison.csv",
                    RESULTS_DIR / "v3_embeddings" / "embedding_comparison_per_seed.csv",
                    RESULTS_DIR / "v3_embeddings" / "cell_line_embeddings.csv"],
        "description": "Ridge(expression+GNN emb) hybrid baseline on the 5 Table 1 SAGE runs (Table 1 row)",
    },
    "xai_ig": {
        "scripts": [SCRIPTS / "32_xai_ig.py"],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_global.csv",
                    RESULTS_DIR / "v4_xai" / "tf_drug_ig_signed.csv",
                    RESULTS_DIR / "v4_xai" / "ig_full_attributions.csv"],
        "description": "TF importance via Integrated Gradients on the canonical SAGE run (Captum, 50 steps)",
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
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed_pairs.csv"],
        "description": "IG per seed on the SAGE runs of splits 42, 123, 456 + cross-seed stability "
                       "(Figure 3, Table 4 TF set)",
    },
    "transfer": {
        "scripts": [SCRIPTS / "17_fix_all_critical.py"],
        "outputs": [RESULTS_DIR / "v5_final" / "scanb_perdrug_predictions.csv",
                    RESULTS_DIR / "v5_final" / "scanb_survival_perdrug.csv"],
        "description": "SCAN-B per-drug Ridge + survival Cox (feeds multicohort)",
    },
    "km_logrank": {
        "scripts": [SCRIPTS / "58_scanb_km_logrank.py"],
        # The extras reproduce the hazard ratios of survival_SCANB.csv, so the stage
        # follows multicohort in DEFAULT_ORDER.
        "outputs": [RESULTS_DIR / "v5_final" / "scanb_km_logrank.csv",
                    REAUDIT_DIR / "scanb_km_extras.csv",
                    REAUDIT_DIR / "scanb_km_at_risk.csv"],
        "description": "Log-rank tests of the SCAN-B Kaplan-Meier panels, median predicted AUC split (Figure 4A), "
                       "10-year survival, PAM50-stratified and quintile trend tests, numbers at risk",
    },
    "basal_her2": {
        "scripts": [SCRIPTS / "19b_basal_her2_pooled.py"],
        "outputs": [RESULTS_DIR / "v7_weaknesses" / "basal_only_survival.csv",
                    RESULTS_DIR / "v7_weaknesses" / "basal_her2_survival.csv",
                    RESULTS_DIR / "v7_weaknesses" / "subtype_pooling_summary.csv"],
        "description": "Within-subtype survival screens: Basal alone and Basal + HER2 pool "
                       "(Supplementary Figure 13)",
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
        "description": "MOA and target enrichment among Fisher-significant drugs "
                       "(Supplementary Figures 8, 9)",
    },
    "cox_pam50": {
        "scripts": [SCRIPTS / "31_cox_multivar_pam50.py"],
        "outputs": [RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv",
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_pen05.csv",
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_pen001.csv",
                    # Counts and list of failed fits, checked by generate_paper_results.py
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_fit_summary.csv",
                    RESULTS_DIR / "v6_strengthen" / "cox_multivar_pam50_failures.csv"],
        "description": "Cox multivariate PAM50-adjusted, 3 penalizers (Table 5)",
    },
    "pca": {
        "scripts": [SCRIPTS / "42_pca_residualization.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "pca_residualization_summary.csv",
                    RESULTS_DIR / "v8_multicohort" / "pca_residualization_perdrug.csv",
                    REAUDIT_DIR / "pc1_orientation.csv"],
        "description": "PCA + PC1 residualization: shared prognostic axis, orientation of PC1 (Discussion)",
    },
    "forest": {
        "scripts": [SCRIPTS / "43_forest_positive_controls.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "forest_positive_controls.csv"],
        "description": "Positive-control HR + 95% CI, SCAN-B vs METABRIC (Supplementary Figure 7)",
    },
    "tf_cox": {
        "scripts": [SCRIPTS / "44_tf_level_cox.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "tf_level_survival_scanb.csv"],
        "description": "TF-level univariate Cox on SCAN-B OS (Results MYC/TP53/E2F1)",
    },
    "penalizer": {
        "scripts": [SCRIPTS / "45_penalizer_sweep.py"],
        "outputs": [RESULTS_DIR / "v8_multicohort" / "penalizer_sweep.csv",
                    RESULTS_DIR / "v8_multicohort" / "penalizer_sweep_perdrug.csv"],
        "description": "Cox penalizer sweep: HR-range + FDR counts (Supplementary Figure 11, Discussion)",
    },
    "aurora": {
        "scripts": [SCRIPTS / "22_aurora_complementary.py"],
        "outputs": [RESULTS_DIR / "v9_aurora" / "aurora_tf_activities.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_primary_vs_meta_tfs.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",
                    RESULTS_DIR / "v9_aurora" / "aurora_cohort_summary.csv"],
        "description": "AURORA TF activities + drug sensitivity, primary vs metastasis "
                       "(inputs of Figure 7 and Table 4, per-sample drug comparison kept as a check)",
    },
    "aurora_ig": {
        "scripts": [SCRIPTS / "36_aurora_ig_validation.py"],
        "outputs": [RESULTS_DIR / "v9_aurora" / "aurora_tf_ig_validation.csv"],
        "description": "Paired primary vs metastasis test of the 10 IG TFs (Table 4, Figure 7)",
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
        "outputs": [RESULTS_DIR / "v6_strengthen" / "graph_ablation.csv",
                    RESULTS_DIR / "v6_strengthen" / "graph_ablation_per_seed.csv"],
        "description": "Graph component ablation from the ablation runs of gnn_train, 5 seeds per "
                       "configuration (Supplementary Figure 12)",
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
    "reporting": {
        "scripts": [SCRIPTS / "54_cohort_demographics.py",
                    SCRIPTS / "55_iclac_check.py"],
        "outputs": [RESULTS_DIR / "v10_reporting" / "cohort_demographics.csv",
                    RESULTS_DIR / "v10_reporting" / "iclac_summary.csv",
                    RESULTS_DIR / "v10_reporting" / "iclac_matches.csv"],
        "description": "Reporting Summary descriptors: cohort demographics (checked against script 21) and ICLAC cross-check",
    },
    "collectri_snapshot": {
        "scripts": [SCRIPTS / "56_collectri_snapshot_sensitivity.py"],
        "outputs": [RESULTS_DIR / "v11_collectri_snapshot" / "collectri_snapshot_sensitivity.csv"],
        "description": "Scripts 21 and 31 rerun on the CollecTRI snapshot TF activities, compared "
                       "with multicohort and cox_pam50 (Supplementary Note 2)",
    },
    # Re-audit of September 2026. Script 67 also reads the ICLAC table of the
    # reporting stage and the cross-screen table of script 05, both earlier in the
    # default order, and the canonical copies under results/, never the ones the
    # paper stage copies into paper/results/.
    # The scripts only read outputs of earlier stages
    # (split and GNN runs of Table 1, transfer, multicohort, penalizer, AURORA and
    # its site analysis, ablation, clinical covariates) and none reads another
    # re-audit output, so they run in numeric order. The stage precedes
    # candidate_table because generate_paper_results.py requires these outputs in
    # both of its passes. 65 also compares results/v4_xai/tf_drug_ig_signed.csv
    # with the copy in paper/results/ that the paper stage writes (tracked in git).
    "reaudit": {
        "scripts": [SCRIPTS / name for name in (
            "59_reference_baselines.py", "60_aurora_background.py", "61_tnbc_subgroup.py",
            "62_treatment_interaction.py", "63_signed_meta_and_dss.py", "64_proportional_hazards.py",
            "65_ig_diagnostics.py", "66_multiplicity_summary.py", "67_tnbc_cell_lines.py")],
        "outputs": [REAUDIT_DIR / name for name in (
            "reference_baselines.csv", "transfer_ridge_heldout_perdrug.csv",               # 59
            "aurora_tf_background.csv", "aurora_site_background.csv",                      # 60
            "aurora_drug_paired.csv", "aurora_background_summary.csv",
            "tnbc_subgroup_cox.csv", "pam50_predicted_auc.csv",                            # 61
            "pam50_composition_sensitive_half.csv", "tnbc_subgroup_summary.csv",
            "treatment_adjusted_cox.csv", "treatment_interaction_cox.csv",                 # 62
            "treatment_summary.csv",
            "signed_meta_stouffer.csv", "direction_agreement_summary.csv",                 # 63
            "positive_control_baserate.csv", "metabric_dss_survival.csv", "meta_dss_summary.csv",
            "ph_schoenfeld.csv", "cox_pam50_stratified.csv", "hr_per_sd_all_drugs.csv",    # 64
            "ph_summary.csv", "ph_schoenfeld_metabric.csv", "cox_pam50_stratified_metabric.csv",
            "ph_summary_metabric.csv",
            "ig_gradient_sign.csv", "ig_gradient_sign_by_tf.csv", "ig_gradient_check.csv", # 65
            "ig_diagnostics_summary.csv",
            "multiplicity_summary.csv", "ablation_perdrug_median.csv",                     # 66
            "metabric_pam50_coverage.csv",
            "tnbc_selectivity_primary.csv", "tnbc_selectivity_background.csv",              # 67
            "tnbc_set_test.csv", "tnbc_sensitivity.csv", "tnbc_power_mde.csv",
            "tnbc_excluded_drugs.csv", "tnbc_orthogonality.csv", "tnbc_permutation_null.csv",
            "tnbc_moa_matched_null.csv", "tnbc_background_by_class.csv",
            "line_potency_offset.csv", "assay_sensitivity_gate.csv", "gdsc_replication.csv",
            "tnbc_cell_lines_summary.csv")],
        "description": "Re-audit (September 2026, 59 to 67): Table 1 references, AURORA background, TNBC and "
                       "Basal subgroups, treatment, signed meta-analysis and DSS, proportional hazards and HR per "
                       "SD, IG diagnostics, multiplicity, TNBC cell lines of PRISM",
    },
    # Supplementary Note 1 repeats the IG analysis on the extended candidate set
    # (the 16 kinase/HDAC inhibitors of Supplementary Data 1 plus the 4
    # controls). Scripts 32 and 35 read that set from Supplementary Data 1
    # (supp_table_S1_candidate_selection.csv), which generate_paper_results.py
    # writes, so the paper stage runs twice: once
    # here for the table and once at the end, when Supplementary Notes 1 and 2
    # exist. A missing Supplementary Note 1 is only a warning in the first pass.
    # What the gene universe of a cohort costs its transfer. Diagnostic only: it
    # recomputes the TCGA activities over the PRISM gene universe and reruns that
    # cohort and the Fisher meta-analysis both ways, and computes the FUSCC
    # activities over the full transcriptome to measure their clipping, without
    # replacing anything.
    "gene_universe": {
        "scripts": [SCRIPTS / "70_gene_universe_tcga.R", SCRIPTS / "71_gene_universe_check.py"],
        "outputs": [GENE_UNIVERSE_DIR / name for name in (
            "tf_activities_tcga_prism_universe.csv", "tf_activities_fuscc_full_transcriptome.csv",
            "gene_universe_r_summary.csv",
            "cohort_clipping.csv", "tcga_screen_comparison.csv",
            "fisher_meta_comparison.csv", "tcga_ph_schoenfeld.csv", "gene_universe_summary.csv")],
        "description": "Gene universe of each cohort: share of TF activities clipped by the "
                       "transfer, and what recomputing TCGA over the PRISM genes changes "
                       "(Supplementary Note 2)",
    },
    # FUSCC survival. It reads the TNBC subgroup of the re-audit and the canonical
    # transfer, and generate_paper_results.py reads its outputs, so it sits between
    # reaudit and candidate_table.
    "fuscc": {
        "scripts": [SCRIPTS / "69_fuscc_survival.py"],
        "outputs": [FUSCC_DIR / name for name in (
            "fuscc_primary_correlation.csv", "fuscc_cox_perdrug.csv", "fuscc_composite.csv",
            "fuscc_permutation_null.csv", "fuscc_sensitivity.csv", "fuscc_cohort_gate.csv",
            "fuscc_scanb_tnbc_reference_z.csv", "fuscc_excluded_patients.csv",
            "fuscc_excluded_summary.csv", "fuscc_rnaseq_vs_not.csv", "fuscc_power_mde.csv",
            "fuscc_ph_tests.csv", "fuscc_covariate_coding.csv", "fuscc_qt_clipping.csv",
            "fuscc_summary.csv")],
        "description": "FUSCC-TNBC: transfer, per-drug Cox score statistics, correlation with "
                       "the SCAN-B TNBC ordering and its permutation null",
    },
    "candidate_table": {
        "scripts": [PAPER_SCRIPTS / "generate_paper_results.py"],
        "outputs": [PAPER_RESULTS / "supp_table_S1_candidate_selection.csv"],
        "description": "First pass of generate_paper_results.py for Supplementary Data 1 "
                       "(supp_table_S1_candidate_selection.csv, the extended candidate set)",
    },
    "xai_ig_extended": {
        "scripts": [(SCRIPTS / "32_xai_ig.py", ["--candidate-set", "extended"])],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_global_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_drug_ig_signed_extended.csv"],
        "description": "IG on the extended candidate set, canonical SAGE run (Supplementary Note 1)",
    },
    "ig_multi_seed_extended": {
        "scripts": [(SCRIPTS / "35_ig_multi_seed.py", ["--candidate-set", "extended"])],
        "outputs": [RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed42_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed123_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed456_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_multi_seed_stability_extended.csv",
                    RESULTS_DIR / "v4_xai" / "tf_importance_ig_seed_pairs_extended.csv"],
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
    "paper": {
        "scripts": [PAPER_SCRIPTS / "generate_paper_results.py"],
        "outputs": [PAPER_RESULTS / "paper_statistics.json",
                    PAPER_RESULTS / "table1_model_comparison.csv",
                    PAPER_RESULTS / "table2_multicohort_summary.csv",
                    PAPER_RESULTS / "table3_positive_controls.csv",
                    PAPER_RESULTS / "table4_aurora_tfs.csv",
                    PAPER_RESULTS / "table5_candidates.csv",
                    PAPER_RESULTS / "tf_importance_ig_top20.csv",
                    PAPER_RESULTS / "tf_importance_ig_top20_per_line.csv",
                    PAPER_RESULTS / "shared_drugs.csv",
                    # Supplementary Tables S3 (robustness) and S1 (METABRIC replication
                    # and chemotherapy interaction), built from the re-audit keys. The
                    # file names keep the numbering of the re-audit
                    PAPER_RESULTS / "supp_table_S3_robustness.csv",
                    PAPER_RESULTS / "supp_table_S4_replication.csv",
                    # trials of the Ridge and RF search of script 72
                    PAPER_RESULTS / "supp_table_hp_search.csv",
                    # Copied only when Supplementary Note 1 exists, so that the
                    # candidate_table pass alone, run before the Note, does not
                    # make this stage fresh when the copy is absent.
                    PAPER_RESULTS / "ig_extended_comparison.csv"],
        "description": "Generate paper CSVs + paper_statistics.json + sanity checks, "
                       "with Supplementary Notes 1 and 2",
    },
    "figures": {
        "scripts": [PAPER_SCRIPTS / "generate_paper_figures.py"],
        "outputs": [FIG_DIR / "Figure1.png",
                    FIG_DIR / "Figure8.png",
                    SFIG_DIR / "FigureS12.png"],
        "description": "Generate paper figures with matplotlib (main + supplementary)",
    },
}

# Default order (respects dependencies): gnn_train precedes every stage that
# loads a GNN (baselines, embeddings, the IG stages and ablation), km_logrank
# follows multicohort, reaudit follows every stage it reads and precedes
# candidate_table, and paper runs after both Supplementary Notes.
DEFAULT_ORDER = [
    "data", "data_r_scanb", "data_r_cohorts", "data_r_fuscc", "graph", "gnn_train",
    "graph_summary", "data_summary",
    "crossval", "baselines",
    "embeddings", "xai_ig", "ig_baselines", "ig_multi_seed",
    "transfer", "basal_her2", "multicohort", "km_logrank", "enrichment", "cox_pam50", "pca", "forest",
    "tf_cox", "penalizer", "aurora", "aurora_ig", "strengthen", "weaknesses",
    "ablation", "sensitivity", "reporting", "collectri_snapshot", "reaudit", "fuscc",
    "gene_universe", "candidate_table",
    "xai_ig_extended", "ig_multi_seed_extended", "aurora_ig_extended",
    "ig_extended_comparison", "paper", "figures",
]

# Stages that read raw inputs which are not redistributed with the package
# (see data/README.md). They are skipped whenever their outputs are present.
RAW_INPUT_STAGES = {
    "data": "the raw PRISM/GDSC/DepMap inputs",
    "data_r_scanb": "the SCAN-B expression and clinical sources",
    "data_r_cohorts": "the METABRIC and TCGA expression and clinical sources",
    "data_r_fuscc": "the processed FUSCC-TNBC files of data/FUSCC/, built by "
                    "scripts/00c_prepare_fuscc.py from the source files (see data/README.md)",
}

# A failure in one of these leaves nothing for the downstream stages to use.
CRITICAL_STAGES = {"data", "data_r_scanb", "data_r_cohorts", "graph",
                   "gnn_train", "baselines", "transfer"}


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
        width = max(len(name) for name in DEFAULT_ORDER)
        for name in DEFAULT_ORDER:
            print(f"  {name:<{width}} {STAGES[name]['description']}")
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
