# PRECISION: interpretable drug repurposing in triple-negative breast cancer with heterogeneous graph neural networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22657695.svg)](https://doi.org/10.5281/zenodo.22657695)
![Paper](https://img.shields.io/badge/Paper-in%20preparation-lightgrey.svg)

This repository is the reproducibility package for the manuscript *Heterogeneous graph neural networks with biological prior knowledge for interpretable drug repurposing in triple-negative breast cancer*. The pipeline builds a heterogeneous knowledge graph (23,498 nodes: 20,389 genes, 1,185 transcription factors, 1,448 drugs and 476 cell lines, linked by CollecTRI regulatory edges, OmniPath protein-protein interactions, drug-target annotations and PRISM drug-response edges), trains a GraphSAGE model to predict drug-response AUC on a cell-line hold-out split against Ridge, random forest and HGT baselines, explains the model with Integrated Gradients over transcription-factor activities, transfers per-drug predictors to three breast-cancer cohorts (SCAN-B, METABRIC, TCGA-BRCA) for survival validation with a Fisher meta-analysis, and tests the resulting transcription-factor signature in paired primary and metastatic samples from AURORA-US. Every number, table and figure in the manuscript can be traced to a CSV file in `paper/results/` and to the script that wrote it.

## What this repository contains

| Directory or file | Content |
|---|---|
| `paper/results/` | 49 CSV/JSON files. The canonical source of every number in the manuscript (5 tables, Supplementary Table S1, the intermediate tables behind every figure and supplementary note, and `paper_statistics.json`). 3.9 MB |
| `paper/figures/`, `paper/supplementary/` | The 25 figure assets the manuscript includes (11 files in `figures/` composing the 7 main figures, 14 files in `supplementary/`, one per supplementary figure), as PNG and, for some, PDF. Supplementary names match the manuscript, main-figure names do not, see *Figure file names* |
| `paper/scripts/` | `generate_paper_results.py` (tables, statistics, sanity checks) and `generate_paper_figures.py` (all figures, matplotlib) |
| `paper/build_paper.sh` | Runs the two paper stages in order and, when LaTeX sources are present, compiles the PDF |
| `pipeline.py` | Single orchestrator for the whole analysis, from raw inputs to figures (`python pipeline.py --list`) |
| `scripts/` | The 34 numbered analysis scripts (32 Python, 2 R) behind the results, plus the two download helpers `00_download_tcga_brca.R` and `00b_download_metabric.R` |
| `src/` | Library code: data loaders, graph construction, the GNN models and the explainer |
| `results/` | The 57 intermediate CSVs that `generate_paper_results.py` and `49_ig_extended_comparison.py` read (under `results/` and its subfolders `v2_fair/`, `v3_embeddings/`, `v4_xai/`, `v6_strengthen/`, `v7_weaknesses/`, `v8_multicohort/`, `v9_aurora/`) and the four trained GNN checkpoints needed for Integrated Gradients (16 MB) |
| `data/DepMap/Model.csv` | DepMap cell-line metadata (CC BY 4.0), the exact file used to select the TNBC cell lines in the explainability scripts |
| `data/precision_processed/prism_drug_annotations.csv` | One row per PRISM compound (MOA, target, clinical phase, disease area, indication) from the PRISM Repurposing 19Q4 treatment metadata (CC BY 4.0), the annotation the candidate selection rule reads |
| `data/prior_knowledge/collectri_edges.csv` | Snapshot of the CollecTRI regulon (composite licence, see *License and citation*) from which the graph and the cell-line TF-activity matrices were computed. 42,990 edges |
| `config.py`, `R/config.R` | Path configuration shared by every Python and R script |
| `environment.yml`, `requirements.txt`, `R/install.R`, `R/sessionInfo.txt` | Exact software versions used to produce the results |
| `CITATION.cff`, `LICENSE` | Citation metadata and the GPL-3.0 license text |

## What it does not contain

* **The development history.** The analysis was developed in a private repository. This package is the audited subset that produces the manuscript: scripts that belong to other projects, abandoned analyses and internal notes are not included.
* **The heterogeneous graph and the OmniPath cache** (`data/precision_graph/hetero_graph_prism.pkl`, `data/prior_knowledge/omnipath_ppi.csv`). OmniPath aggregates 76 resources with mixed licences and its API does not filter by licence, so the merged interaction set cannot be redistributed. The script that builds it (`scripts/02_build_graph.py`) is included together with the counts obtained for the paper, so any deviation of a rebuilt graph is measurable. See `data/README.md`.
* **Raw inputs.** DepMap expression, PRISM and GDSC screens, SCAN-B, METABRIC, TCGA-BRCA and AURORA-US expression and clinical data must be downloaded from their sources. `data/README.md` lists every file the code reads, where to obtain it, its licence and the exact path it must occupy under `data/`.
* **Large regenerable outputs**, in particular the 220 MB SCAN-B per-drug prediction matrix (`results/v5_final/scanb_perdrug_predictions.csv`), which the `transfer` stage recomputes on CPU in minutes from the TF-activity matrices of PRISM and SCAN-B (`data/precision_processed/`, produced by the `data` and `data_r_scanb` stages).
* **The manuscript sources.** The LaTeX and the journal template are not part of the package.

## Three levels of reproduction

### Level 1: verify every number in the paper

No computation is needed. `paper/results/` is the authoritative source of every figure, table and number in the manuscript:

* `table1_model_comparison.csv` to `table5_candidates.csv` are the five tables as printed.
* `paper_statistics.json` holds every number quoted in the text (cohort counts, FDR-significant drug counts, direction concordance, IG rankings, AURORA tests, sensitivity analyses) under a descriptive key.
* The remaining CSVs are the exact data behind each figure (the table in *Provenance of every paper output* maps each file to the figure that reads it and to the script that produced it).

Open the files with any spreadsheet or `pandas.read_csv`. Nothing in the manuscript is typed by hand: any figure or number that is not in these files is a defect.

### Level 2: regenerate tables and figures

Requires only the Python environment. Everything is computed from the shipped `results/` directory, so no raw data, GPU or model training is involved:

```bash
python paper/scripts/generate_paper_results.py   # tables 1-5, Supplementary Table S1, paper_statistics.json, 41 copies, sanity checks
python paper/scripts/generate_paper_figures.py   # the 25 figure assets (matplotlib)
```

or equivalently `bash paper/build_paper.sh`, or `python pipeline.py --force --stage paper figures`.

`generate_paper_results.py` first checks that the 21 critical CSVs are present, writes the tables (Table 5 is computed by the candidate selection rule of Methods from the Cox results and the PRISM annotations, and Supplementary Table S1 lists every drug the rule considers), then runs a block of regression asserts (sanity checks) that compare the regenerated numbers with what the manuscript states (graph size, FDR counts per cohort, direction concordance, IG stability, AURORA counts, penalizer sweep, candidate rule counts, Supplementary Note 1). It exits with code 2 if any assert fails. The 49 files it leaves in `paper/results/` must be identical to the shipped ones.

`generate_paper_figures.py` never fails on a missing input: it skips the figure with a warning and prints the list of skipped figures at the end. With only the shipped files it regenerates 22 of the 25 figure assets. The other three need inputs that are not included in the package (`Figure7`, `Figure8`, `FigureS4`, see *Known limitations*).

### Level 3: full rerun

1. Download the raw inputs listed in `data/README.md` into `data/` (or into any directory, exported as `PRECISION_DATA`). TCGA-BRCA can be fetched with `Rscript scripts/00_download_tcga_brca.R` and METABRIC with `Rscript scripts/00b_download_metabric.R`.
2. Install both environments (see *Environment*). A CUDA-capable GPU is strongly recommended: the GNN is trained several times (baselines, IG checkpoint, multi-seed, ablation) with 800 epochs each, and Integrated Gradients performs about 20,000 forward and backward passes per seed. The clinical transfer and every survival analysis downstream of it run on CPU.
3. Run the pipeline:

```bash
python config.py                 # prints every resolved path and whether it exists
python pipeline.py --list        # stage names in dependency order
python pipeline.py --stage all   # every stage, skipping those whose outputs already exist
```

Stages are idempotent: a stage is skipped when all its declared outputs exist, `--force` re-runs it. Two stages are written in R (`data_r_scanb`, `data_r_cohorts`): `pipeline.py` launches them with `Rscript` from `PATH`, or from the `RSCRIPT` environment variable, and exports `PRECISION_DATA` and `PRECISION_RESULTS` so that `R/config.R` resolves the same directories as `config.py`. Every script that produces a paper input is a registered stage. The pipeline stops at the first failure of a critical stage (`data`, `data_r_scanb`, `data_r_cohorts`, `graph`, `baselines`, `xai`, `transfer`).

## Pipeline stages

Stages in the default order of `pipeline.py`. Intermediate outputs live under `results/` (`RESULTS_DIR`) or `data/precision_processed/` (`DATA_DIR`). Figure names refer to the asset files in `paper/figures/` and `paper/supplementary/`, whose numbering is stable across manuscript revisions and does not necessarily match the number printed in the rendered manuscript.

| Stage | Script | Main outputs | Feeds |
|---|---|---|---|
| `data` | `scripts/01_prepare_data.py` | `precision_processed/tf_activities_all_prism.csv`, `tf_activities_all_gdsc.csv`, `drug_target_edges.csv`, the CollecTRI and OmniPath caches | Every downstream stage |
| `data_r_scanb` (R) | `scripts/09_scanb_transfer.R` | `precision_processed/tf_activities_scanb.csv`, `scanb_clinical.csv` (plus `results/scanb_tf_cox_results.csv`, a sanity check not used by the paper) | SCAN-B survival (Tables 2, 3, 5, Figure7, Figure8, Figure10) |
| `data_r_cohorts` (R) | `scripts/20_multicohort_validation.R` | `precision_processed/tf_activities_metabric.csv`, `metabric_clinical.csv` (PAM50 via genefu), `tf_activities_tcga.csv`, `tcga_clinical.csv` | METABRIC and TCGA survival (Tables 2, 3, Figure10) |
| `graph` | `scripts/02_build_graph.py` | `precision_graph/hetero_graph_prism.pkl` (23,498 nodes) | All GNN stages |
| `graph_summary` | `scripts/51_graph_summary.py` | `results/graph_summary.csv` (node and edge counts of the prebuilt graph) | Methods and Figure1 counts in `paper_statistics.json` (the graph itself is not shipped, the summary is) |
| `data_summary` | `scripts/53_data_summary.py` | `results/data_summary.csv` (sizes of the PRISM, expression and GDSC matrices, MOA encoding, hold-out split) | Methods data constants in `paper_statistics.json` (the matrices are not shipped, the summary is) |
| `crossval` | `scripts/05_train_hgt_and_crossval.py` | `results/crossval_prism_to_gdsc.csv`, `best_hgt_model.pt` | FigureS14 |
| `baselines` | `scripts/11_rerun_fair_comparison.py` | `v2_fair/fair_comparison.csv`, `per_drug_{RF_TF,Ridge_expression,SAGE_learned_emb}.csv`, `best_sage_v2.pt` | Table 1, Figure3, FigureS2 |
| `embeddings` | `scripts/12_gnn_embeddings_perdrug.py` | `v3_embeddings/embedding_comparison.csv`, `cell_line_embeddings.csv` | Table 1 (hybrid row), FigureS3 |
| `xai` | `scripts/13_xai_v2_and_scanb.py` | `v4_xai/best_sage_v4.pt`, `tf_importance_bootstrap1000.csv`, `drug_explanations_v2.csv`, `tnbc_drug_ranking_v2.csv` | Trained model for Integrated Gradients (scripts 32, 33). The ablation outputs are illustrative only |
| `training_history` | `scripts/50_training_history.py` | `results/gnn_training_history.csv` (per-epoch loss and test Pearson, same protocol as `xai`, no checkpoint written) | FigureS1, training keys in `paper_statistics.json` |
| `multi_seed` | `scripts/34_multi_seed_gnn.py` | `v4_xai/best_sage_seed{42,123,456}.pt` | Models for `ig_multi_seed` |
| `xai_ig` | `scripts/32_xai_ig.py` | `v4_xai/tf_importance_ig_global.csv`, `tf_drug_ig_signed.csv`, `ig_full_attributions.csv` | Figure11, IG helpers reused by scripts 33 and 35 |
| `ig_baselines` | `scripts/33_ig_baseline_robustness.py` | `v4_xai/ig_baseline_{mean,zero,random,stability}.csv` | IG baseline robustness in `paper_statistics.json` |
| `ig_multi_seed` | `scripts/35_ig_multi_seed.py` | `v4_xai/tf_importance_ig_seed{42,123,456}.csv`, `tf_importance_ig_multi_seed_stability.csv` | Figure4, Figure5, FigureS4, TF selection for Table 4 and Figure9 |
| `transfer` | `scripts/17_fix_all_critical.py` | `v5_final/scanb_perdrug_predictions.csv` (220 MB), `scanb_survival_perdrug.csv` (plus `best_model_v5.pt` when the graph is present: a GNN trained for reference only, not used by the transfer) | Figure7, Figure8, stages `basal_her2`, `pca`, `penalizer`, `strengthen`, `weaknesses` |
| `basal_her2` | `scripts/19b_basal_her2_pooled.py` | `v7_weaknesses/basal_only_survival.csv`, `basal_her2_survival.csv`, `subtype_pooling_summary.csv` | FigureS12, Results (within-subtype counts) |
| `multicohort` | `scripts/21_multicohort_survival.py` | `v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv`, `meta_analysis_fisher.csv`, `cohort_summary.csv` (patients, events, deduplication chain, TFs and genefu PAM50 concordance per cohort) | Tables 2, 3, `shared_drugs.csv`, Methods cohort sizes, Figure6, Figure10, FigureS5, FigureS13, stage `enrichment` |
| `enrichment` | `scripts/52_drug_enrichment.py` | `v8_multicohort/moa_enrichment.csv`, `target_enrichment.csv` (one-sided Fisher tests of the PRISM MOA classes and molecular targets over-represented among the Fisher-significant drugs, Benjamini-Hochberg within each family) | FigureS7, FigureS6, enrichment keys in `paper_statistics.json` |
| `cox_pam50` | `scripts/31_cox_multivar_pam50.py` | `v6_strengthen/cox_multivar_pam50_{adjusted,pen05,pen001}.csv` | Table 5 |
| `pca` | `scripts/42_pca_residualization.py` | `v8_multicohort/pca_residualization_{summary,perdrug}.csv` | Discussion (shared prognostic axis), `paper_statistics.json` |
| `forest` | `scripts/43_forest_positive_controls.py` | `v8_multicohort/forest_positive_controls.csv` | FigureS8 |
| `tf_cox` | `scripts/44_tf_level_cox.py` | `v8_multicohort/tf_level_survival_scanb.csv` | Results text (MYC, TP53, E2F1), `paper_statistics.json` |
| `penalizer` | `scripts/45_penalizer_sweep.py` | `v8_multicohort/penalizer_sweep.csv` | FigureS10, Discussion |
| `aurora` | `scripts/22_aurora_complementary.py` | `v9_aurora/aurora_tf_activities.csv`, `aurora_primary_vs_meta_tfs.csv`, `aurora_drug_primary_vs_meta.csv`, `aurora_cohort_summary.csv` (samples, patients, primary/metastatic/normal, paired patients) | FigureS9, Methods cohort sizes, stages `aurora_ig` and `sensitivity` (script 39) |
| `aurora_ig` | `scripts/36_aurora_ig_validation.py` | `v9_aurora/aurora_tf_ig_validation.csv` | Table 4, Figure9 |
| `strengthen` | `scripts/18_strengthen_paper.py` | `v6_strengthen/test3_subtype_summary.csv`, `test3_survival_*.csv` (tests 1 and 2 run only when legacy inputs exist) | Subtype counts in `paper_statistics.json` |
| `weaknesses` | `scripts/19_fix_three_weaknesses.py` | `v7_weaknesses/d2_basal_enrichment.csv`, `d3_penalized_cox.csv` | Discussion (penalized Cox) |
| `ablation` | `scripts/27_graph_ablation.py` | `v6_strengthen/graph_ablation.csv` | FigureS11, `paper_statistics.json` |
| `sensitivity` | `scripts/37_cox_clinical_covariates.py`, `39_aurora_site_stratified.py`, `40_rfs_sensitivity.py`, `41_combat_sensitivity.py` | `v8_multicohort/cox_clinical_comparison.csv`, `cox_rfs_comparison.csv`, `combat_vs_qt_summary.csv`, `v9_aurora/aurora_tf_by_site.csv`, `aurora_tf_by_site_summary.csv` | Sensitivity statements in Results and Discussion (copied to `paper/results/` by the `paper` stage) |
| `paper` | `paper/scripts/generate_paper_results.py` | Tables 1 to 5, Supplementary Table S1, `shared_drugs.csv`, `paper_statistics.json`, 41 copies from `results/`, sanity checks | Level 1 and the figure generator |
| `xai_ig_extended` | `scripts/32_xai_ig.py --candidate-set extended` | `v4_xai/tf_importance_ig_global_extended.csv`, `tf_drug_ig_signed_extended.csv` | Supplementary Note 1 |
| `ig_multi_seed_extended` | `scripts/35_ig_multi_seed.py --candidate-set extended` | `v4_xai/tf_importance_ig_seed{42,123,456}_extended.csv`, `tf_importance_ig_multi_seed_stability_extended.csv` | Supplementary Note 1 |
| `aurora_ig_extended` | `scripts/36_aurora_ig_validation.py --candidate-set extended` | `v9_aurora/aurora_tf_ig_validation_extended.csv` | Supplementary Note 1 |
| `ig_extended_comparison` | `scripts/49_ig_extended_comparison.py` | `v4_xai/ig_extended_comparison.csv`, `ig_extended_core_tfs.csv` | Supplementary Note 1, `paper_statistics.json` |
| `figures` | `paper/scripts/generate_paper_figures.py` | the 25 figure assets (`Figure1` to `Figure10`, `FigureS4` to `Figure11` without S5 and S8, matplotlib) | Manuscript |

Two download helpers are not stages because they only need to run once and need network access: `scripts/00_download_tcga_brca.R` fetches the TCGA-BRCA input of `data_r_cohorts` from GDC (TCGAbiolinks required) and `scripts/00b_download_metabric.R` fetches the three METABRIC inputs from the cBioPortal API (httr and jsonlite required).

## Figure file names versus manuscript numbering

Supplementary assets match the manuscript one to one: `FigureSN.png` **is** Supplementary Figure N, for N = 1 to 14.

Main-figure assets keep their historical identifiers, which do not match the rendered numbers, because several rendered figures are composed of more than one asset. This table is the correspondence:

| Manuscript | File | Content |
|---|---|---|
| Figure 1 | `Figure1.png` (panel A), `Figure2.png` (panel B) | graph schema and pipeline |
| Figure 2 | `Figure3.png` | model comparison |
| Figure 3 | `Figure4.png` (A), `Figure5.png` (B), `Figure11.png` (C) | IG top-20 TFs, stability across seeds, signed TF-drug attributions |
| Figure 4 | `Figure10.png` | multi-cohort survival overlap |
| Figure 5 | `Figure6.png` | meta-analysis drug ranking |
| Figure 6 | `Figure7.png` (A), `Figure8.png` (B) | SCAN-B Kaplan-Meier and dose-response quintiles |
| Figure 7 | `Figure9.png` | AURORA primary versus metastasis |

## Provenance of every paper output

The 49 files in `paper/results/` fall into two groups.

**Computed by `paper/scripts/generate_paper_results.py` (8 files)**

| File | Computed from | Upstream script |
|---|---|---|
| `table1_model_comparison.csv` | `results/v2_fair/fair_comparison.csv`, `results/v3_embeddings/embedding_comparison.csv` | 11, 12 |
| `table2_multicohort_summary.csv` | `results/v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv`, `meta_analysis_fisher.csv` | 21 |
| `table3_positive_controls.csv` | same as Table 2 | 21 |
| `table4_aurora_tfs.csv` | `results/v9_aurora/aurora_tf_ig_validation.csv` | 36 |
| `table5_candidates.csv` | the candidate selection rule of Methods applied to `results/v6_strengthen/cox_multivar_pam50_adjusted.csv`, `cox_multivar_pam50_pen05.csv` and `data/precision_processed/prism_drug_annotations.csv` | 31, 01 |
| `supp_table_S1_candidate_selection.csv` | same inputs: the 143 drugs passing the statistical filter, with one indicator per filter of the rule | 31, 01 |
| `shared_drugs.csv` | `results/v8_multicohort/survival_SCANB.csv`, `survival_METABRIC.csv` | 21 |
| `paper_statistics.json` | all of the above plus `test3_subtype_summary.csv` (18), `graph_ablation.csv` (27), `tf_importance_ig_seed42.csv` and `tf_importance_ig_multi_seed_stability.csv` (35), `ig_baseline_{mean,random,stability}.csv` (33), `pca_residualization_summary.csv` (42), `tf_level_survival_scanb.csv` (44), `penalizer_sweep.csv` (45), `aurora_drug_primary_vs_meta.csv` (22), `cohort_summary.csv` (21), `aurora_cohort_summary.csv` (22), `graph_summary.csv` (51), `moa_enrichment.csv` and `target_enrichment.csv` (52), `crossval_prism_to_gdsc.csv` (05), `ig_full_attributions.csv` (32), `data_summary.csv` (53) | see list |

**Copied verbatim from `results/` by `generate_paper_results.py` (41 files)**

| File | Copied from | Producing script | Read by |
|---|---|---|---|
| `meta_analysis_fisher.csv` | `results/v8_multicohort/` | 21 | Figure6, Figure7, script 52 (stage `enrichment`) |
| `survival_SCANB.csv` | `results/v8_multicohort/` | 21 | Figure6, Figure10, FigureS5, FigureS13 |
| `survival_METABRIC.csv` | `results/v8_multicohort/` | 21 | Figure6, Figure10, FigureS5, FigureS13 |
| `survival_TCGA.csv` | `results/v8_multicohort/` | 21 | Figure6, Figure10, FigureS5 |
| `aurora_tf_activities.csv` | `results/v9_aurora/` | 22 | Figure9 |
| `aurora_drug_primary_vs_meta.csv` | `results/v9_aurora/` | 22 | FigureS9, AURORA drug counts in `paper_statistics.json` |
| `drug_explanations_v2.csv` | `results/v4_xai/` | 13 | Per-drug TF explanations from the ablation-era explainer, kept for completeness (the canonical attributions are the IG files) |
| `tnbc_drug_ranking_v2.csv` | `results/v4_xai/` | 13 | Drugs ranked by mean predicted AUC of the GNN across the 22 breast cancer cell lines (Methods, Spearman against the TNBC-only ranking) |
| `tf_importance_ig_top20.csv` | `results/v4_xai/tf_importance_ig_seed42.csv` (full ranking, renamed) | 35 | Figure4, IG top-10 in `paper_statistics.json` |
| `tf_importance_ig_stability.csv` | `results/v4_xai/tf_importance_ig_multi_seed_stability.csv` (renamed) | 35 | Figure4, Figure5, FigureS4 |
| `ig_baseline_stability.csv` | `results/v4_xai/` | 33 | IG baseline robustness (Results) |
| `tf_drug_ig_signed.csv` | `results/v4_xai/` | 32 | Figure11 |
| `cell_line_embeddings.csv` | `results/v3_embeddings/` | 12 | FigureS3 |
| `gnn_training_history.csv` | `results/` | 50 | FigureS1, training keys in `paper_statistics.json` |
| `per_drug_RF_TF.csv` | `results/v2_fair/` | 11 | FigureS2 |
| `per_drug_Ridge_expression.csv` | `results/v2_fair/` | 11 | FigureS2 |
| `per_drug_SAGE_learned_emb.csv` | `results/v2_fair/` | 11 | FigureS2 |
| `crossval_prism_to_gdsc.csv` | `results/` | 05 | FigureS14 |
| `basal_her2_survival.csv` | `results/v7_weaknesses/` | 19b | FigureS12, within-subtype counts in `paper_statistics.json` |
| `basal_only_survival.csv` | `results/v7_weaknesses/` | 19b | Results (Basal alone), `paper_statistics.json` |
| `subtype_pooling_summary.csv` | `results/v7_weaknesses/` | 19b | Results paragraph and FigureS12 caption (patients, events, FDR counts per subset), keys `subtype_*` in `paper_statistics.json` |
| `graph_ablation.csv` | `results/v6_strengthen/` | 27 | FigureS11 |
| `pca_residualization_summary.csv` | `results/v8_multicohort/` | 42 | Discussion, `paper_statistics.json` |
| `pca_residualization_perdrug.csv` | `results/v8_multicohort/` | 42 | Discussion |
| `forest_positive_controls.csv` | `results/v8_multicohort/` | 43 | FigureS8 |
| `tf_level_survival_scanb.csv` | `results/v8_multicohort/` | 44 | Results text, `paper_statistics.json` |
| `penalizer_sweep.csv` | `results/v8_multicohort/` | 45 | FigureS10, `paper_statistics.json` |
| `cox_clinical_comparison.csv` | `results/v8_multicohort/` | 37 | Results: Table 5 candidates and positive controls under full clinical adjustment (SCAN-B and METABRIC) versus PAM50-only |
| `aurora_tf_by_site.csv` | `results/v9_aurora/` | 39 | Results: primary versus metastasis delta and Wilcoxon p per TF and metastatic tissue site |
| `aurora_tf_by_site_summary.csv` | `results/v9_aurora/` | 39 | Results: pivot summary of the previous file |
| `cox_rfs_comparison.csv` | `results/v8_multicohort/` | 40 | Discussion: hazard ratios with relapse-free survival versus overall survival as endpoint |
| `combat_vs_qt_summary.csv` | `results/v8_multicohort/` | 41 | Discussion: FDR-significant drug counts and direction overlap under ComBat versus QuantileTransformer batch correction |
| `ig_extended_comparison.csv` | `results/v4_xai/` | 49 | Supplementary Note 1: main versus extended IG runs (top-20 overlap, core TFs, Spearman, AURORA counts), keys `ig_extended_*` in `paper_statistics.json` |
| `ig_extended_core_tfs.csv` | `results/v4_xai/` | 49 | Supplementary Note 1: the ten seed-stable core TFs of each run |
| `aurora_tf_ig_validation_extended.csv` | `results/v9_aurora/` | 36 (`--candidate-set extended`) | Supplementary Note 1: AURORA test of the extended core TFs |
| `cohort_summary.csv` | `results/v8_multicohort/` | 21 | Methods cohort sizes (patients, events, deduplication chains, TFs per cohort, genefu PAM50 concordance, reverse Kaplan-Meier follow-up), Table 2 checks in `paper_statistics.json` |
| `aurora_cohort_summary.csv` | `results/v9_aurora/` | 22 | Methods AURORA-US cohort composition (samples, patients, primary, metastatic, normal, paired) in `paper_statistics.json` |
| `graph_summary.csv` | `results/` | 51 | Node and edge counts of the knowledge graph (Methods, Figure 1) in `paper_statistics.json` |
| `data_summary.csv` | `results/` | 53 | Methods data constants (PRISM, expression and GDSC matrix sizes, MOA encoding, hold-out split) in `paper_statistics.json` |
| `moa_enrichment.csv` | `results/v8_multicohort/` | 52 | FigureS7, MOA enrichment keys in `paper_statistics.json` |
| `target_enrichment.csv` | `results/v8_multicohort/` | 52 | FigureS6, target enrichment keys in `paper_statistics.json` |

Scripts 37, 40 and 41 also write per-drug detail tables next to their outputs under `results/` (`cox_clinical_adjusted_{scanb,metabric}.csv`, `cox_rfs_scanb.csv`, `combat_cox_scanb.csv`, `combat_vs_qt_comparison.csv`). They are not cited by the manuscript and are not shipped.

Figures with no data input: `Figure1` (graph schema, counts hard-coded from the graph metadata) and `Figure2` (pipeline diagram).

## Trained models

Four GraphSAGE checkpoints (PyTorch `state_dict`, 3.9 MB each, 16 MB in total) are shipped under `results/v4_xai/` because Integrated Gradients has to be run on the exact weights that produced the paper:

| File | Trained by | Used by |
|---|---|---|
| `best_sage_v4.pt` | `scripts/13_xai_v2_and_scanb.py` (seed 42, 800 epochs, cell-line hold-out) | `scripts/32_xai_ig.py` (canonical IG attributions), `scripts/33_ig_baseline_robustness.py` |
| `best_sage_seed42.pt`, `best_sage_seed123.pt`, `best_sage_seed456.pt` | `scripts/34_multi_seed_gnn.py` (one training per seed, split regenerated with the seed) | `scripts/35_ig_multi_seed.py` (multi-seed IG stability) |

Loading a checkpoint requires the heterogeneous graph (`data/precision_graph/hetero_graph_prism.pkl`, not redistributed) because the architecture is instantiated from the graph metadata. The other checkpoints produced by a full rerun (`v2_fair/best_sage_v2.pt`, `v5_final/best_model_v5.pt`, `best_hgt_model.pt`) are not needed downstream and are not shipped.

## Environment

**Python 3.12.10.** Install with `pip install -r requirements.txt` or `conda env create -f environment.yml` (environment name `precision-paper`). Exact versions used for the paper:

| Package | Version | Package | Version |
|---|---|---|---|
| torch | 2.11.0 (CUDA 12.8 build) | scipy | 1.17.1 |
| torch-geometric | 2.7.0 | matplotlib | 3.10.8 |
| captum | 0.9.0 | seaborn | 0.13.2 |
| lifelines | 0.30.3 | umap-learn | 0.5.11 |
| decoupler | 2.1.4 | networkx | 3.6.1 |
| omnipath | 1.0.12 | matplotlib-venn | 1.1.2 (Figure10 only) |
| pandas | 2.2.3 | scikit-learn | 1.8.0 |
| numpy | 2.4.3 | combat (pycombat) | 0.3.3, used only by `scripts/41_combat_sensitivity.py` |
| requests | 2.33.0 | | |

The pinned `torch` resolves to the default PyPI wheel of your platform, which on Linux is the CUDA build with its `nvidia-*` dependencies (several GB). For a CPU-only machine install `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` first and then the rest of the requirements. For a GPU build install `torch==2.11.0` from the PyTorch index for your CUDA version first. Level 2 (tables and figures) does not import torch at all. Only the direct dependencies are pinned: transitive packages such as numba and llvmlite (used by umap-learn for `FigureS3`) resolve to whatever pip selects.

**R 4.5.2 with Bioconductor 3.22.** Install with `Rscript R/install.R`, which pins the Bioconductor release, installs the CRAN packages from the archive at their exact versions and prints a comparison table at the end. `R/sessionInfo.txt` records the versions and which script loads which package.

| Package | Version | Package | Version |
|---|---|---|---|
| decoupleR | 2.16.0 | SummarizedExperiment | 1.40.0 |
| genefu | 2.42.0 | dplyr | 1.2.0 |
| survival | 3.8.3 | tidyr | 1.3.2 |
| survminer | 0.5.2 | tibble | 3.3.1 |
| OmnipathR | 3.18.4 (live CollecTRI fallback only) | | |

`scripts/00_download_tcga_brca.R` additionally needs TCGAbiolinks (Bioconductor) and `scripts/00b_download_metabric.R` needs httr and jsonlite (CRAN). Neither is part of `R/install.R` because they are only used to fetch inputs.

## Configuration

All paths are resolved by `config.py` (Python) and `R/config.R` (R), both anchored to the package root, so every script can be launched from any working directory.

| Setting | Default | How to override |
|---|---|---|
| `DATA_DIR` (raw and processed inputs) | `<package>/data` | environment variable `PRECISION_DATA` |
| `RESULTS_DIR` (intermediate outputs and models) | `<package>/results` | environment variable `PRECISION_RESULTS` |
| `PAPER_RESULTS`, `FIG_DIR`, `SFIG_DIR` | `<package>/paper/{results,figures,supplementary}` | not overridable (part of the package) |
| `Rscript` binary for the R stages | first `Rscript` on `PATH` | environment variable `RSCRIPT` |
| `DEPMAP_VERSION` | `24Q4` | first line of `data/DepMap/ACTIVE_VERSION` |

`pipeline.py` exports `PRECISION_DATA` and `PRECISION_RESULTS` to every child process, so the R scripts see the same directories as the Python ones. `python config.py` prints every resolved path, whether it exists, and the DepMap files expected for the active release.

`config.depmap_file(kind)` returns the path of each DepMap file for the active release. DepMap renamed the protein-coding expression matrix between releases (`OmicsExpressionProteinCodingGenesTPMLogp1.csv` up to 24Q4, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` from 25Q3): whichever of the two exists on disk is used, and when neither exists the error names the file expected for `DEPMAP_VERSION`. `Model.csv`, `CRISPRGeneEffect.csv` and the other files kept their names across releases.

## Known limitations

* **Figures that need inputs not shipped with the package.** `generate_paper_figures.py` skips them with a warning and lists them at the end of its log. With the shipped files it regenerates 22 of the 25 assets (`FigureS3`, the UMAP of cell-line embeddings, is among them: it needs `data/DepMap/Model.csv`, included, and `umap-learn`. `FigureS6` and `FigureS7`, the target and mechanism-of-action enrichment, read the tables of script 52 shipped in `paper/results/`).
  * `Figure7` and `Figure8` (SCAN-B Kaplan-Meier and tertile curves) need `results/v5_final/scanb_perdrug_predictions.csv` (220 MB, regenerated by the `transfer` stage) and `data/precision_processed/scanb_clinical.csv` (from the `data_r_scanb` stage).
  * `FigureS4` (per-drug Ridge coefficients on TF activities) refits the Ridge models, so it needs the PRISM TF-activity matrix and response matrix produced by the `data` stage.
* **CollecTRI snapshot versus live network.** The clinical TF-activity matrices used in the paper (SCAN-B, METABRIC, TCGA-BRCA) were computed in April 2026 with the CollecTRI network downloaded live through decoupleR. The package makes the R scripts read the same snapshot as the graph (`data/prior_knowledge/collectri_edges.csv`, 42,990 edges, 1,185 TFs). The live network of September 2026 differs from the snapshot in 394 added edges, 225 removed edges and 62 sign changes (plus one TF, PLSCR1). Measured on 7 September 2026: rerunning `data_r_scanb` and `data_r_cohorts` with the snapshot reproduces the shipped matrices for most TFs but changes 125 (SCAN-B), 119 (METABRIC) and 137 (TCGA-BRCA) of them, 26 to 39 per cohort with Pearson r below 0.99 against the shipped values, because the regulons of those TFs differ between the two network versions. Downstream, that rerun gives 604 instead of 623 SCAN-B drugs at FDR<0.05, 87 instead of 74 in METABRIC, 558 instead of 551 in the Fisher meta-analysis, and six of the seven Table 5 candidates (indirubin replaces trametinib). The AURORA-US stage (script 22, decoupler-py) reproduces its shipped outputs byte for byte with the snapshot. The shipped clinical matrices are the ones the manuscript reports.
* **OmniPath drift.** The PPI edges are downloaded live from OmniPath by the `data` stage and cached locally. The paper's graph was built from 84,587 protein-protein interactions (169,174 directed edges) and has 23,498 nodes with 20,389 genes. The same query at packaging time (September 2026) already returns 86,189 interactions. Because the merged interaction set cannot be redistributed (mixed per-resource licences, no licence filtering in the API), a full rerun produces a slightly different graph and slightly different GNN results. The counts above, shipped in `results/graph_summary.csv` (written by `scripts/51_graph_summary.py` from the canonical graph) and checked by `generate_paper_results.py`, are the reference against which a rebuilt graph should be compared.
* **Sanity checks are calibrated to the published run.** The asserts at the end of `generate_paper_results.py` encode the numbers stated in the manuscript (for instance the penalizer sweep must yield exactly 306 FDR-significant drugs at penalizer 0.5). After a Level 3 rerun with a drifted OmniPath snapshot, or on different hardware, some asserts may fail even when the analysis is correct: the exit code then flags that the manuscript numbers would have to be updated.
* **Training is seeded but not bit-reproducible across hardware.** Every stage fixes `SEED=42` (and 123, 456 for the multi-seed analysis), but CUDA kernels and driver versions can change low-order digits of the GNN results.
* **Some inputs have ambiguous redistribution terms.** METABRIC (cBioPortal, no explicit licence) and GDSC (non-commercial) are therefore not shipped, and neither are the TF-activity matrices derived from them. `data/README.md` explains how to obtain each file.

## Adapting the pipeline to another setting

The scripts are written for the analysis of the paper and take no command-line arguments. To reuse the method on another cancer type or drug panel, the places to edit are:

* the cell-line filter (TNBC, breast lineage): `src/data/load_depmap.py` (`filter_tnbc_lines`) and `src/xai/explainer_v2.py`;
* the candidate selection rule (`SELECTION_RULE`, `KINASE_HDAC_MOA`, `PHASE_RANK` in `paper/scripts/generate_paper_results.py`) and the candidate and control drug lists of the downstream scripts: `scripts/32_xai_ig.py` (`CANDIDATE_DRUGS`, `resolve_candidate_set`), `33_ig_baseline_robustness.py`, `35_ig_multi_seed.py`, `37_cox_clinical_covariates.py`, `40_rfs_sensitivity.py`, `43_forest_positive_controls.py`, `21_multicohort_survival.py`;
* the TF set of the site-stratified analysis: `scripts/39_aurora_site_stratified.py`;
* cohort sizes and the candidate list of Table 5 written into `paper_statistics.json`: `paper/scripts/generate_paper_results.py`;
* the GNN hyperparameters (128 hidden units, 3 layers, dropout 0.33, lr 1.3e-3, 800 epochs) repeated in `scripts/11`, `12`, `13`, `17`, `27` and `34`, and the Ridge alpha (730) in `scripts/11`, selected with a Bayesian search during development.

`src/graph/build_hetero_graph.py` names its PPI argument `string_ppi` for historical reasons: the paper feeds it the OmniPath interactions (`scripts/02_build_graph.py`), the STRING loader is only a fallback in `scripts/01_prepare_data.py`.

## License and citation

The code in this repository is released under the **GNU General Public License v3.0** (see `LICENSE`), Copyright (C) 2026 Carlos Fernandez-Lozano, David Ferreiro and Patricia V.-del-Rio.

Data files under `data/` keep the licence of their source. DepMap `Model.csv` is CC BY 4.0 (it contains the public DepMap model annotations, including patient age, sex and treatment fields as distributed by DepMap). `collectri_edges.csv` is a snapshot of CollecTRI as served by OmniPath: a composite resource whose constituent databases carry their own licences (CC BY 4.0 for most, CC BY-SA 4.0 for TRRUST, LGPL-3.0 for HTRI, AFL 3.0 for GEREDB), all of which permit redistribution with attribution. The CollecTRI code repository itself is GPL-3.0.

Results under `results/` and `paper/results/` are per-drug or per-TF summary statistics derived from: DepMap, CCLE and PRISM (CC BY 4.0); SCAN-B (CC BY 4.0, Mendeley Data `yzxtxn4nmd` v3); TCGA-BRCA (GDC open access); GDSC (`crossval_prism_to_gdsc.csv` only, per-drug correlations, under the GDSC terms of use, which are non-commercial); METABRIC (`survival_METABRIC.csv`, `shared_drugs.csv`, `forest_positive_controls.csv`, `cox_clinical_comparison.csv`, computed from cBioPortal data that has no explicit licence); and AURORA-US (`aurora_tf_activities.csv`, TF activity scores per sample identified by GEO sample id, derived from the public series GSE209998; no clinical or genotype data are included). The trained checkpoints, `cell_line_embeddings.csv` and the Integrated Gradients tables were computed on a graph that contains OmniPath edges, but they hold model weights, embeddings and attribution scores, not the edges themselves, which is why they are redistributed while the graph and the OmniPath cache are not.

If you use this code or these results, please cite the archived version of this repository (metadata in `CITATION.cff`): DOI 10.5281/zenodo.22657695. That DOI always resolves to the latest release; the snapshot released with this version is 10.5281/zenodo.22657696.

The accompanying manuscript is in preparation:

> Fernandez-Lozano C, Ferreiro D, V.-del-Rio P. Heterogeneous graph neural networks with biological prior knowledge for interpretable drug repurposing in triple-negative breast cancer. Manuscript in preparation.

This section and `CITATION.cff` will be updated with the preprint DOI and the journal DOI as soon as each one exists.

The Zenodo record is an archive of this repository, code and paper tables, with no bulk data: the large intermediate outputs are regenerated by the pipeline from the sources listed in `data/README.md`.

## Funding and acknowledgements

Grant PID2024-162441OA-I00 funded by MICIU/AEI/10.13039/501100011033 and by "ERDF/EU".

RePo-SUDOE, with project reference S1/1.1/P0033, a project co-financed by the Interreg Sudoe Programme through the European Regional Development Fund (ERDF).

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its supercomputing infrastructure.
