# PRECISION: Transcription factor activities as a shared representation for drug repurposing hypotheses in breast cancer

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22657695.svg)](https://doi.org/10.5281/zenodo.22657695)
[![Preprint](https://img.shields.io/badge/Preprint-bioRxiv-lightgrey.svg)](https://doi.org/10.64898/2026.09.08.750045)

This repository is the reproducibility package for the manuscript *Transcription factor activities as a shared representation for drug repurposing hypotheses in breast cancer*. The pipeline builds a heterogeneous knowledge graph (23,498 nodes: 20,389 genes, 1,185 transcription factors (TFs), 1,448 drugs and 476 cell lines, linked by CollecTRI regulatory edges, OmniPath protein-protein interactions, drug-target annotations and PRISM drug-response edges) and trains GraphSAGE and HGT models to predict drug-response AUC on a cell-line hold-out split. It compares them with a per-drug mean reference and with Ridge and Random Forest baselines whose hyperparameters were chosen by cross-validation inside the training cell lines, and explains the GraphSAGE model with Integrated Gradients over TF activities. Per-drug Ridge models on TF activities transfer the predictions to three unselected breast cancer cohorts (SCAN-B, METABRIC and TCGA-BRCA), whose survival associations are combined with Fisher and signed Stouffer meta-analyses, and an explicit rule selects the repurposing candidates. The TF signature and the drug predictions are further examined in the paired primary and metastatic tumors of AURORA-US. Triple-negative breast cancer (TNBC) motivates the work and is examined directly in the TNBC cell lines of PRISM and in FUSCC-TNBC, a cohort made up entirely of TNBC. The manuscript presents an interpretable framework for generating repurposing hypotheses, not a claim of predictive superiority. Every number, table and figure in it can be traced to a file in `paper/results/` and to the script that wrote it.

## What this repository contains

| Directory or file | Content |
|---|---|
| `paper/results/` | 132 files (131 CSV and `paper_statistics.json`), 9.2 MB. The canonical source of every number in the manuscript: the five tables, Supplementary Tables S1, S3, S4 and S5, Supplementary Data 1, the tables behind every figure and supplementary note, and `paper_statistics.json` (2,415 keys). See *Provenance of every paper output* |
| `paper/figures/` | 8 PNG files, 2.5 MB. `FigureN.png` is Figure N |
| `paper/supplementary/` | 14 PNG files, 1.7 MB. `FigureSN.png` is Supplementary Figure N |
| `paper/scripts/` | `generate_paper_results.py` (tables, statistics, sanity checks) and `generate_paper_figures.py` (all figures, matplotlib) |
| `paper/build_paper.sh` | Runs the two paper stages in order and, when LaTeX sources are present, compiles the PDF |
| `pipeline.py` | Single orchestrator for the whole analysis, 42 stages from raw inputs to figures (`python pipeline.py --list`) |
| `scripts/` | 54 scripts (47 Python, 7 R). 49 of them are run by the stages of `pipeline.py`. The other five are the download helpers `00_download_tcga_brca.R` and `00b_download_metabric.R`, the FUSCC-TNBC preparation `00c_prepare_fuscc.py`, the hyperparameter search `72_baseline_hp_search.py` (run on a supercomputer, outputs shipped) and `20b_metabric_pam50.R` (superseded by script 20, which computes the METABRIC PAM50 calls itself, and kept because script 66 checks the PAM50 gene list it holds) |
| `src/` | Library code, 20 files: data loaders and cohort helpers, clinical covariate encoding, graph construction, the GNN models and the trainer of every GNN run (`src/models/training.py`), the candidate selection rule (`src/analysis/candidate_selection.py`) and the vectorized Cox score test (`src/analysis/score_test.py`) |
| `tests/` | 6 test files and the R script that produced the reference values of one of them (`python -m pytest tests`) |
| `results/` | 280 files, 89.2 MB: the intermediate outputs that the two generators read (all of them except the 220 MB SCAN-B prediction matrix), the 42 GNN runs with 7 checkpoints, and the inputs and outputs of the hyperparameter search. Subfolders in the next table |
| `data/DepMap/Model.csv` | DepMap 25Q3 cell-line annotations, 2,132 models and 8 columns (`ModelID`, `PatientID`, `CellLineName`, `StrippedCellLineName`, `OncotreeLineage`, `OncotreeSubtype`, `RRID`, `ModelSubtypeFeatures`). `PatientID` is pseudonymized as `D0001` to `D1932`, the donors numbered in the sorted order of their DepMap identifiers. No sex, age or ethnicity of the donor. Under the depmap.org terms |
| `data/PRISM/processed/drug_response_prism.csv` | PRISM Repurposing 19Q4 secondary screen, AUC of 476 cell lines by 1,448 drugs (614,325 measured pairs), 10.7 MB, CC BY 4.0 |
| `data/precision_processed/prism_drug_annotations.csv` | One row per PRISM compound (MOA, target, clinical phase, disease area, indication) from the PRISM Repurposing 19Q4 treatment metadata (CC BY 4.0, curated by the Broad Drug Repurposing Hub), the annotation the candidate selection rule reads |
| `data/precision_processed/fuscc_r_summary.csv` | Versions, checksums and gene and sample counts of the FUSCC-TNBC TF-activity step (script 68). No patient-level values |
| `data/prior_knowledge/collectri_edges.csv` | Snapshot of the CollecTRI regulon (composite license, see *License and citation*) from which the graph and every TF-activity matrix except those of SCAN-B, METABRIC and TCGA-BRCA were computed. 42,990 edges |
| `data/README.md` | Every input file the code reads, where to obtain it, its license and the path it must occupy under `data/` |
| `config.py`, `R/config.R` | Path configuration shared by every Python and R script |
| `environment.yml`, `requirements.txt`, `R/install.R`, `R/sessionInfo.txt` | Exact software versions used to produce the results |
| `CITATION.cff`, `LICENSE`, `DATA_LICENSES.md` | Citation metadata, the GPL-3.0 license text and the licenses of the bundled data files |

The `results/` subfolders keep the historical names of the pipeline versions, so that every result can be traced back to the stage that wrote it.

| Folder | Files | Size | Content (script) |
|---|---|---|---|
| `results/` (root) | 3 | 16 kB | Cross-screen evaluation (05), graph counts (51), data constants (53) |
| `v2_fair/` | 6 | 0.4 MB | Table 1 rows of Ridge, Random Forest, GraphSAGE and HGT, per seed and per drug (11) |
| `v3_embeddings/` | 4 | 0.5 MB | Hybrid row of Table 1 and the cell-line embeddings (12) |
| `v4_xai/` | 12 | 6.2 MB | Integrated Gradients: canonical attributions (32), baseline robustness (33), seeds (35), Supplementary Note 1 (49) |
| `v5_final/` | 1 | 0.5 kB | Log-rank tests of Figure 4 (58). The prediction matrix of script 17 is not shipped |
| `v6_strengthen/` | 8 | 0.6 MB | PAM50-adjusted Cox at three penalizers (31), graph ablation (27), subtype counts (18) |
| `v7_weaknesses/` | 3 | 0.2 MB | Within-subtype survival screens (19b) |
| `v8_multicohort/` | 21 | 1.5 MB | Survival per cohort and Fisher meta-analysis (21), sensitivity analyses (37, 40, 41), PCA (42), forest plot (43), TF-level Cox (44), penalizer sweep (45), enrichment (52) |
| `v9_aurora/` | 7 | 2.0 MB | AURORA-US TF activities and drug comparison (22), core TF test (36), metastatic sites (39) |
| `v10_reporting/` | 3 | 5 kB | Cohort demographics (54) and ICLAC cross-check (55) |
| `v11_collectri_snapshot/` | 1 | 1 kB | Supplementary Note 2, CollecTRI version (56) |
| `v12_gnn/` | 134 | 32.6 MB | The 42 GNN runs (57), see *Trained models* |
| `v13_reaudit/` | 48 | 3.7 MB | Re-audit of September 2026 (59 to 67), PC1 orientation (42), Kaplan-Meier extras (58) and the summary of a one-off check |
| `v14_fuscc/` | 13 | 1.2 MB | FUSCC-TNBC per-drug and aggregate tables (69) |
| `v15_gene_universe/` | 6 | 0.4 MB | Supplementary Note 2, gene universe (70, 71) |
| `v16_hp_search/` | 10 | 40.2 MB | Ridge and Random Forest hyperparameter search (72): the input bundle of the 381 training cell lines in `inputs/` (expression, response, TF activities, folds and manifest) and the outputs of the supercomputer job in `cesga_11133811/` |

## What it does not contain

* **The development history.** The analysis was developed in a private repository. This package is the audited subset that produces the manuscript: scripts that belong to other projects, abandoned analyses and internal notes are not included.
* **The heterogeneous graph and the OmniPath cache** (`data/precision_graph/hetero_graph_prism.pkl`, `data/prior_knowledge/omnipath_ppi.csv`). OmniPath aggregates many resources with mixed licenses and its API does not filter by license, so the merged interaction set cannot be redistributed. The script that builds the graph (`scripts/02_build_graph.py`) is included together with the counts obtained for the paper (`results/graph_summary.csv`), so any deviation of a rebuilt graph is measurable. See `data/README.md`.
* **Raw inputs.** DepMap expression, the PRISM and GDSC screens, and the expression and clinical data of SCAN-B, METABRIC, TCGA-BRCA, AURORA-US and FUSCC-TNBC must be downloaded from their sources (see *Level 3*).
* **Patient-level clinical data and per-patient matrices.** No clinical table and no per-patient TF-activity or prediction matrix of SCAN-B, METABRIC, TCGA-BRCA or FUSCC-TNBC is shipped. Their results travel as per-drug, per-TF or aggregate tables. The one per-sample matrix in the package is the AURORA-US TF activities, computed from the public GEO series GSE209998, with sample codes pseudonymized as `AUR-Pnn-<site>` (patients numbered P01 to P53 in the sorted order of their AURORA codes, the site code TTP, TTM or NT kept, aliquot and plate fields dropped).
* **The 220 MB SCAN-B per-patient prediction matrix** (`results/v5_final/scanb_perdrug_predictions.csv`). The `transfer` stage recomputes it on CPU from the TF-activity matrices of PRISM and SCAN-B, which the `data` and `data_r_scanb` stages produce from the raw inputs.
* **The FUSCC-TNBC clinical table and excluded-patient list.** The clinical annotations are Table S1 of Jiang et al. 2019 (Cancer Cell), published under the Elsevier user license, which does not permit redistribution. `results/v14_fuscc/fuscc_excluded_patients.csv` carries patient identifiers and is not shipped: the exclusions travel as counts by reason in `fuscc_excluded_summary.csv`. The FPKM matrix is CC BY 4.0 on figshare (doi 10.6084/m9.figshare.19783498.v5) and is downloaded from there.
* **The internal analysis plans** (`docs/plan_hp_baselines.md`, `docs/plan_analisis_tnbc_lineas.md`, `docs/plan_analisis_fuscc_tnbc.md`). Several scripts cite them, but they are internal documents of the private repository. Rerun without them, scripts 67 and 69 record `absent` as the hash of their plan, and the shipped summaries keep the hash recorded when the analysis ran.
* **Outputs that neither generator reads.** Among them the checkpoints of the 35 GNN runs that no later script loads (their metrics are shipped, see *Trained models*), the HGT of the cross-screen evaluation, the per-seed rankings of seeds 123 and 456 and a few tables of later stages. Besides the raw-input stages, `graph` (the graph is not redistributed) and `transfer`, the stages `gnn_train`, `ig_multi_seed`, `aurora`, `weaknesses`, `reaudit`, `fuscc`, `gene_universe`, `xai_ig_extended` and `ig_multi_seed_extended` are therefore not fresh in the package and rerun when called.
* **The manuscript sources.** The LaTeX and the journal template are not part of the package.

## Three levels of reproduction

### Level 1: verify every number in the paper

No computation is needed. `paper/results/` is the authoritative source of every figure, table and number in the manuscript:

* `table1_model_comparison.csv` to `table5_candidates.csv` are the five tables as printed, and the supplementary tables are listed in *Figure and table file names*.
* `paper_statistics.json` holds every number quoted in the text (cohort counts, FDR-significant drug counts, direction concordance, Integrated Gradients rankings, AURORA-US tests, cell-line and FUSCC-TNBC analyses, sensitivity analyses) under a descriptive key.
* The remaining CSVs are the exact data behind each figure, table and note (the tables in *Provenance of every paper output* map each file to what it backs and to the script that produced it).

Open the files with any spreadsheet or `pandas.read_csv`. Nothing in the manuscript is typed by hand: any figure or number that is not in these files is a defect.

### Level 2: regenerate tables and figures

Requires only the Python environment and the Arial font. Everything is computed from the shipped `results/` and `data/` directories, so no raw data, GPU or model training is involved:

```bash
python paper/scripts/generate_paper_results.py   # tables, supplementary tables, paper_statistics.json, 120 copies, sanity checks
python paper/scripts/generate_paper_figures.py   # 8 main and 14 supplementary figures (matplotlib)
```

or equivalently `bash paper/build_paper.sh`, or `python pipeline.py --force --stage paper figures`.

`generate_paper_results.py` first checks that its 111 required inputs are present. It then writes the tables (Table 5 is computed by the candidate selection rule of Methods, `src/analysis/candidate_selection.py`, from the Cox results and the PRISM annotations, and Supplementary Data 1 lists the 143 drugs of the rule's statistical filter), registers every number quoted in the text under a key of `paper_statistics.json`, copies 120 intermediate tables from `results/` and runs 214 sanity checks that compare the regenerated numbers with what the manuscript states and with each other. It exits with code 2 if any check fails. One further check, which compares `scripts/72_baseline_hp_search.py` with the version committed in the private development repository before the search ran (commit `3d539a0`), is skipped with a warning because that commit is not in the public history, and the check of the previous Table 1 rows runs without its comparison with that commit. The SHA-256 of the script is still checked against the one recorded on the supercomputer. The 132 files it writes in `paper/results/` (12 computed, 120 copied) are identical to the shipped ones.

`generate_paper_figures.py` stops if Arial is missing, because matplotlib would otherwise fall back to another typeface without warning. It skips a figure whose inputs are missing, with a warning, and lists the skipped figures at the end of its log. With the shipped files it regenerates 20 of the 22 figure files, identical to the shipped PNGs on the machine that produced them (other platforms may render text slightly differently). The two it skips need data that are not in the package, and their shipped PNGs are the ones in the manuscript:

* **Figure 4** (SCAN-B Kaplan-Meier curves) needs the per-patient predictions `results/v5_final/scanb_perdrug_predictions.csv` (stage `transfer`) and the SCAN-B clinical table `data/precision_processed/scanb_clinical.csv` (stage `data_r_scanb`).
* **Supplementary Figure 4** (coefficients of the per-drug transfer Ridge models) refits those models, so it needs the PRISM TF-activity matrix `data/precision_processed/tf_activities_all_prism.csv` (stage `data`) and the TFs of the SCAN-B matrix `tf_activities_scanb.csv` (stage `data_r_scanb`).

### Level 3: full rerun

1. Download the raw inputs listed in `data/README.md` into `data/` (or into any directory, exported as `PRECISION_DATA`). TCGA-BRCA can be fetched with `Rscript scripts/00_download_tcga_brca.R` and METABRIC with `Rscript scripts/00b_download_metabric.R`. For FUSCC-TNBC, download the FPKM matrix `FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv` from figshare (doi 10.6084/m9.figshare.19783498.v5) and Table S1 of Jiang et al. 2019 (Cancer Cell, doi 10.1016/j.ccell.2019.02.001, file `mmc2.xlsx`) into `data/FUSCC/raw/` and run `python scripts/00c_prepare_fuscc.py` (or pass the folder that holds them with `--raw-dir`). The script checks the md5 that figshare publishes for the matrix and a SHA-256 of the clinical columns it uses before writing `data/FUSCC/`. Script 55 also reads the ICLAC register of misidentified cell lines, version 14, at `data/reference/iclac_register_v14.xlsx`.
2. Install both environments (see *Environment*). A CUDA-capable GPU is recommended for `scripts/57_train_gnns.py`, which trains the 42 GNN runs (800 epochs each, final-epoch weights). The training times recorded in `results/v12_gnn/runs.csv` add up to about 50 minutes on the laptop GPU recorded in the sidecars (NVIDIA GeForce RTX 5060 Laptop GPU). The scripts that load the checkpoints (12, 32, 33, 35, 65) and the cross-screen HGT of script 05 also use the GPU when there is one. The clinical transfer and every survival analysis run on CPU.
3. Run the pipeline:

```bash
python config.py                 # prints every resolved path and whether it exists
python pipeline.py --list        # stage names in dependency order
python pipeline.py --stage all   # every stage, skipping those whose outputs already exist
```

Stages are idempotent: a stage is skipped when all its declared outputs exist, and `--force` re-runs it. `scripts/57_train_gnns.py` also keeps every run it finds, so an interrupted `gnn_train` resumes where it stopped. After retraining, name the stages that read `results/v12_gnn/` and force only those, because `--stage all --force` would also rebuild the graph the runs were trained on. Three stages are written in R (`data_r_scanb`, `data_r_cohorts`, `data_r_fuscc`) and `gene_universe` runs one R script (70): `pipeline.py` launches them with `Rscript` from `PATH`, or from the `RSCRIPT` environment variable, and exports `PRECISION_DATA` and `PRECISION_RESULTS` so that `R/config.R` resolves the same directories as `config.py`. Every script that produces a paper input is a registered stage, except the hyperparameter search of step 4, the download and preparation helpers of step 1 and the one-off check behind `talazoparib_family12_summary.csv`. The pipeline stops at the first failure of a critical stage (`data`, `data_r_scanb`, `data_r_cohorts`, `graph`, `gnn_train`, `baselines`, `transfer`), so on the package as shipped `--stage all` stops at `data` until the raw inputs are in place. Name the stages to run instead.

4. Optional: the hyperparameter search of the Ridge and Random Forest baselines (Supplementary Table S5). Its outputs are shipped in `results/v16_hp_search/cesga_11133811/` and the values it chose are written as constants in `scripts/11_rerun_fair_comparison.py` (`RIDGE_ALPHA = 506`, the best trial rounded to three significant figures, and `RF_PARAMS`, 150 trees, depth 8, minimum leaf 13), so no stage depends on rerunning it. `python scripts/72_baseline_hp_search.py prepare` writes the input bundle of the 381 training cell lines (shipped in `results/v16_hp_search/inputs/`), and `python scripts/72_baseline_hp_search.py search --model ridge --n-jobs 64` and `search --model rf --n-jobs 64` run the search, ten trials per model scored by five-fold cross-validation grouped by donor and stratified by lineage. The search ran on the CESGA supercomputer (job 11133811). The script is shipped byte for byte as it ran there, and `generate_paper_results.py` checks its SHA-256 against the one the job recorded. It therefore keeps paths relative to the working directory: run it from the package root with the default `data/` and `results/` layout, because it does not read `PRECISION_DATA` or `PRECISION_RESULTS`.

Three details of a Level 3 rerun:

* `scripts/66_multiplicity_summary.py` needs the development history only for its section [10], the provenance of the hyperparameters. Without that history it warns and keeps the rows of that section from the shipped `results/v13_reaudit/multiplicity_summary.csv`.
* Scripts 39 and 60 match each AURORA-US sample with the GEO series matrix by its sample code, so they stop with an explicit error while the codes are pseudonymized: rerun script 22 from the GEO files first. That rerun overwrites the tracked `results/v9_aurora/aurora_tf_activities.csv` with the GEO sample codes. Restore the shipped copy with `git checkout -- results/v9_aurora/aurora_tf_activities.csv` before committing.
* Besides the raw inputs, `.gitignore` ignores `data/FUSCC/`, `data/reference/`, `data/prior_knowledge/string_ppi_*.csv` and the per-patient or per-sample outputs of a Level 3 rerun (`results/v14_fuscc/fuscc_excluded_patients.csv`, `results/v15_gene_universe/tf_activities_*.csv`, `results/v9_aurora/aurora_drug_predictions.csv`), so that none of them is committed by accident.

## Pipeline stages

Stages in the default order of `pipeline.py` (`python pipeline.py --list`). Script numbers refer to `scripts/`. Intermediate outputs live under `results/` (`RESULTS_DIR`) or `data/precision_processed/` (`DATA_DIR`).

| Stage | Script | Main outputs | Feeds |
|---|---|---|---|
| `data` | `01_prepare_data.py` | `precision_processed/tf_activities_all_prism.csv`, `tf_activities_all_gdsc.csv`, `drug_target_edges.csv`, `prism_drug_annotations.csv`, the CollecTRI and OmniPath caches | Every downstream stage |
| `data_r_scanb` (R) | `09_scanb_transfer.R` | `precision_processed/tf_activities_scanb.csv`, `scanb_clinical.csv` | SCAN-B survival |
| `data_r_cohorts` (R) | `20_multicohort_validation.R` | `precision_processed/tf_activities_metabric.csv`, `metabric_clinical.csv` (PAM50 with genefu), `tf_activities_tcga.csv`, `tcga_clinical.csv` | METABRIC and TCGA-BRCA survival |
| `data_r_fuscc` (R) | `68_fuscc_tf_activities.R` | `precision_processed/tf_activities_fuscc.csv` (over the gene universe of PRISM), `fuscc_clinical.csv`, `fuscc_r_summary.csv` | Stage `fuscc` |
| `graph` | `02_build_graph.py` | `precision_graph/hetero_graph_prism.pkl` (23,498 nodes) | `gnn_train`, `graph_summary` and every script that loads a GNN |
| `gnn_train` | `57_train_gnns.py` | `v12_gnn/`: 42 runs, each with checkpoint, sidecar, per-drug and history tables, and `runs.csv` | Table 1, Figure 2, Integrated Gradients, graph ablation, Supplementary Figure 1 |
| `graph_summary` | `51_graph_summary.py` | `graph_summary.csv` | Methods and Figure 1a counts (the graph is not shipped, the summary is) |
| `data_summary` | `53_data_summary.py` | `data_summary.csv` | Methods data constants (matrix sizes, MOA encoding, hold-out split) |
| `crossval` | `05_train_hgt_and_crossval.py` | `crossval_prism_to_gdsc.csv` (its own HGT, not a run of `gnn_train`) | Supplementary Figure 14 |
| `baselines` | `11_rerun_fair_comparison.py` | `v2_fair/fair_comparison.csv`, `fair_comparison_per_seed.csv`, `per_drug_*.csv` | Table 1, Figure 2, Supplementary Figures 2 and 12 |
| `embeddings` | `12_gnn_embeddings_perdrug.py` | `v3_embeddings/embedding_comparison.csv`, `embedding_comparison_per_seed.csv`, `cell_line_embeddings.csv`, per-drug table | Table 1 (hybrid row), Figure 2, Supplementary Figure 3 |
| `xai_ig` | `32_xai_ig.py` | `v4_xai/tf_importance_ig_global.csv`, `tf_drug_ig_signed.csv`, `ig_full_attributions.csv` | Figure 3a (per-line points) and 3c |
| `ig_baselines` | `33_ig_baseline_robustness.py` | `v4_xai/ig_baseline_{mean,zero,random,stability}.csv` | Robustness of Integrated Gradients to the baseline (Results) |
| `ig_multi_seed` | `35_ig_multi_seed.py` | `v4_xai/tf_importance_ig_seed{42,123,456}.csv`, `tf_importance_ig_multi_seed_stability.csv`, `tf_importance_ig_seed_pairs.csv` | Figure 3a and 3b, the ten core TFs of Table 4, Figure 7 and Supplementary Figure 4 |
| `transfer` | `17_fix_all_critical.py` | `v5_final/scanb_perdrug_predictions.csv` (220 MB, not shipped), `scanb_survival_perdrug.csv` | Figure 4 and every SCAN-B survival stage |
| `basal_her2` | `19b_basal_her2_pooled.py` | `v7_weaknesses/basal_only_survival.csv`, `basal_her2_survival.csv`, `subtype_pooling_summary.csv` | Supplementary Figure 13, Results |
| `multicohort` | `21_multicohort_survival.py` | `v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv`, `meta_analysis_fisher.csv`, `cohort_summary.csv` | Tables 2 and 3, Figures 5 and 6, Supplementary Figures 5 and 6 |
| `km_logrank` | `58_scanb_km_logrank.py` | `v5_final/scanb_km_logrank.csv`, `v13_reaudit/scanb_km_extras.csv`, `scanb_km_at_risk.csv` | Figure 4 |
| `enrichment` | `52_drug_enrichment.py` | `v8_multicohort/moa_enrichment.csv`, `target_enrichment.csv` | Supplementary Figures 8 and 9 |
| `cox_pam50` | `31_cox_multivar_pam50.py` | `v6_strengthen/cox_multivar_pam50_{adjusted,pen05,pen001,fit_summary,failures}.csv` | Table 5, Supplementary Table S2, Supplementary Data 1 |
| `pca` | `42_pca_residualization.py` | `v8_multicohort/pca_residualization_{summary,perdrug}.csv`, `v13_reaudit/pc1_orientation.csv` | Discussion (shared prognostic axis) |
| `forest` | `43_forest_positive_controls.py` | `v8_multicohort/forest_positive_controls.csv` | Table 3 (confidence intervals), Supplementary Figure 7 |
| `tf_cox` | `44_tf_level_cox.py` | `v8_multicohort/tf_level_survival_scanb.csv` | Results (TF-level Cox) |
| `penalizer` | `45_penalizer_sweep.py` | `v8_multicohort/penalizer_sweep.csv`, `penalizer_sweep_perdrug.csv` | Supplementary Figure 11, Discussion |
| `aurora` | `22_aurora_complementary.py` | `v9_aurora/aurora_tf_activities.csv`, `aurora_primary_vs_meta_tfs.csv`, `aurora_drug_primary_vs_meta.csv`, `aurora_cohort_summary.csv` | Figure 7 and Table 4 (through `aurora_ig`), Methods cohort composition. The per-sample drug comparison is kept as a check |
| `aurora_ig` | `36_aurora_ig_validation.py` | `v9_aurora/aurora_tf_ig_validation.csv` | Table 4, Figure 7 |
| `strengthen` | `18_strengthen_paper.py` | `v6_strengthen/test3_subtype_summary.csv` (tests 1 and 2 need outputs of a retired script and are skipped) | Subtype counts in `paper_statistics.json` |
| `weaknesses` | `19_fix_three_weaknesses.py` | `v7_weaknesses/d2_basal_enrichment.csv`, `d3_penalized_cox.csv` (not shipped) | Neither generator reads them |
| `ablation` | `27_graph_ablation.py` | `v6_strengthen/graph_ablation.csv`, `graph_ablation_per_seed.csv`, summarized from the ablation runs of `gnn_train` | Supplementary Figure 12 |
| `sensitivity` | `37_cox_clinical_covariates.py`, `39_aurora_site_stratified.py`, `40_rfs_sensitivity.py`, `41_combat_sensitivity.py` | `v8_multicohort/cox_clinical_*.csv`, `cox_rfs_*.csv`, `combat_*.csv`, `v9_aurora/aurora_tf_by_site*.csv` | Sensitivity statements in Results and Discussion |
| `reporting` | `54_cohort_demographics.py`, `55_iclac_check.py` | `v10_reporting/cohort_demographics.csv`, `iclac_summary.csv`, `iclac_matches.csv` | Reporting Summary, Supplementary Table S4 |
| `collectri_snapshot` | `56_collectri_snapshot_sensitivity.py` | `v11_collectri_snapshot/collectri_snapshot_sensitivity.csv` | Supplementary Note 2 |
| `reaudit` | `59` to `67` | `v13_reaudit/` (46 declared outputs, 44 shipped) | Reference rows of Table 1, Stouffer column of Table 3, HR per SD and test correlation columns of Table 5, Figure 7b, Figure 8a and 8b, Supplementary Figure 10, Supplementary Tables S1 and S3 |
| `fuscc` | `69_fuscc_survival.py` | `v14_fuscc/` (15 declared outputs, 13 shipped) | Figure 8c and 8d, Results |
| `gene_universe` | `70_gene_universe_tcga.R`, `71_gene_universe_check.py` | `v15_gene_universe/` (8 declared outputs, the 6 tables shipped) | Supplementary Note 2 |
| `candidate_table` | `generate_paper_results.py` (first pass) | `paper/results/supp_table_S1_candidate_selection.csv` | Supplementary Data 1, and the extended candidate set the next four stages read |
| `xai_ig_extended` | `32_xai_ig.py --candidate-set extended` | `v4_xai/tf_importance_ig_global_extended.csv`, `tf_drug_ig_signed_extended.csv` | Supplementary Note 1 |
| `ig_multi_seed_extended` | `35_ig_multi_seed.py --candidate-set extended` | `v4_xai/tf_importance_ig_seed{42,123,456}_extended.csv` and the stability and seed-pair tables | Supplementary Note 1 |
| `aurora_ig_extended` | `36_aurora_ig_validation.py --candidate-set extended` | `v9_aurora/aurora_tf_ig_validation_extended.csv` | Supplementary Note 1 |
| `ig_extended_comparison` | `49_ig_extended_comparison.py` | `v4_xai/ig_extended_comparison.csv`, `ig_extended_core_tfs.csv` | Supplementary Note 1 |
| `paper` | `generate_paper_results.py` | Tables, supplementary tables, `paper_statistics.json`, 120 copies, sanity checks | Level 1 and the figure generator |
| `figures` | `generate_paper_figures.py` | `paper/figures/Figure1.png` to `Figure8.png`, `paper/supplementary/FigureS1.png` to `FigureS14.png` | Manuscript |

The paper stage runs twice because the four extended stages of Supplementary Note 1 read Supplementary Data 1, which the generator writes: `candidate_table` runs it first for that table, and `paper` repeats it at the end, when Supplementary Notes 1 and 2 exist.

## Figure and table file names

File names match the manuscript one to one: `paper/figures/FigureN.png` **is** Figure N (N = 1 to 8) and `paper/supplementary/FigureSN.png` **is** Supplementary Figure N (N = 1 to 14). Every figure is a single PNG, RGB at 300 dpi, and every main figure is 170 mm wide, the text width. The panels of Figures 1, 3 and 4 are drawn by separate functions and composed into one PNG by the generator, which also draws every panel letter.

| File | Content |
|---|---|
| `Figure1.png` | Knowledge graph schema (a) and pipeline (b) |
| `Figure2.png` | Model comparison in the cell-line hold-out |
| `Figure3.png` | Integrated Gradients: top 20 TFs (a), stability across seeds (b), signed TF-drug attributions (c) |
| `Figure4.png` | SCAN-B Kaplan-Meier curves at the median predicted AUC (a) and by quintile (b) |
| `Figure5.png` | Survival associations in the three cohorts |
| `Figure6.png` | The 30 most significant drugs of the Fisher meta-analysis |
| `Figure7.png` | AURORA-US: core TFs in paired primary and metastatic tumors (a), against all TFs (b) |
| `Figure8.png` | The candidates in the TNBC cell lines of PRISM (a, b) and in FUSCC-TNBC (c, d) |

The supplementary table files keep historical names that do not match the numbering of the manuscript:

| Manuscript | File in `paper/results/` |
|---|---|
| Supplementary Table S1 (METABRIC replication, signed meta-analysis, chemotherapy interaction) | `supp_table_S4_replication.csv` |
| Supplementary Table S2 (candidate selection rule) | summarizes the rule, whose counts are in `paper_statistics.json` |
| Supplementary Data 1 (the 143 drugs of the statistical filter) | `supp_table_S1_candidate_selection.csv` |
| Supplementary Table S3 (robustness of the SCAN-B associations) | `supp_table_S3_robustness.csv` |
| Supplementary Table S4 (cell lines in the ICLAC register) | `iclac_matches.csv` |
| Supplementary Table S5 (hyperparameter search) | `supp_table_hp_search.csv` |

## Provenance of every paper output

The 132 files in `paper/results/` fall into two groups. Script numbers refer to `scripts/`, and folders to `results/`.

**Computed by `paper/scripts/generate_paper_results.py` (12 files)**

| File | Backs | Computed from (script) |
|---|---|---|
| `table1_model_comparison.csv` | Table 1, Figure 2, Supplementary Figure 12 | `v2_fair/fair_comparison.csv` (11), `v3_embeddings/embedding_comparison.csv` (12), `v13_reaudit/reference_baselines.csv` (59) |
| `table2_multicohort_summary.csv` | Table 2 | `v8_multicohort/survival_{SCANB,METABRIC,TCGA}.csv`, `meta_analysis_fisher.csv`, `cohort_summary.csv` (21) |
| `table3_positive_controls.csv` | Table 3, Figures 3c and 8, Supplementary Figure 4 | The survival tables and Fisher meta-analysis (21), `forest_positive_controls.csv` (43), `v13_reaudit/signed_meta_stouffer.csv` (63) |
| `table4_aurora_tfs.csv` | Table 4, Figure 7 | `v9_aurora/aurora_tf_ig_validation.csv` (36) |
| `table5_candidates.csv` | Table 5, Figures 3c and 8, Supplementary Figure 4 | The selection rule applied to `v6_strengthen/cox_multivar_pam50_adjusted.csv` and `cox_multivar_pam50_pen05.csv` (31) and `data/precision_processed/prism_drug_annotations.csv` (01), plus `hr_per_sd_all_drugs.csv` (64) and `transfer_ridge_heldout_perdrug.csv` (59) |
| `supp_table_S1_candidate_selection.csv` | Supplementary Data 1, with one indicator per filter of the rule | Same inputs as Table 5 |
| `supp_table_S3_robustness.csv` | Supplementary Table S3 | Keys of scripts 61, 62 and 64 |
| `supp_table_S4_replication.csv` | Supplementary Table S1 | METABRIC survival (21) and keys of scripts 62 and 63 |
| `supp_table_hp_search.csv` | Supplementary Table S5 | `v16_hp_search/cesga_11133811/` (72) and the values of Table 1 (11) |
| `shared_drugs.csv` | Results and Supplementary Figure 6 (drugs significant in both SCAN-B and METABRIC and their direction) | `survival_SCANB.csv`, `survival_METABRIC.csv` (21) |
| `tf_importance_ig_top20_per_line.csv` | Figure 3a (one point per TNBC cell line) | `v4_xai/ig_full_attributions.csv` (32) |
| `paper_statistics.json` | Every number quoted in the text | All of the above and the tables of the next group |

**Copied from `results/` by `generate_paper_results.py` (120 files)**

All are byte-for-byte copies except `aurora_tf_activities.csv`, whose sample codes are pseudonymized on the way if the source still carries the original AURORA-US codes (the shipped source already carries the pseudonyms, so here it is copied as it is). Where the copy has another name, the source file is given.

*Model comparison, GNN runs and graph*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `gnn_runs.csv` | `v12_gnn/runs.csv` | 57 | Every GNN run with its test metrics, behind the GNN rows of Table 1 and Supplementary Figure 12 |
| `gnn_training_history.csv` | `v12_gnn/history/sage_split42_init42_full.csv` | 57 | Supplementary Figure 1 |
| `fair_comparison_per_seed.csv` | `v2_fair/` | 11 | Figure 2, Supplementary Figure 12 |
| `embedding_comparison_per_seed.csv` | `v3_embeddings/` | 12 | Figure 2 |
| `per_drug_RF_TF.csv` | `v2_fair/` | 11 | Figure 2, Supplementary Figure 2 |
| `per_drug_Ridge_expression.csv` | `v2_fair/` | 11 | Figure 2, Supplementary Figure 2 |
| `per_drug_SAGE_learned_emb.csv` | `v2_fair/` | 11 | Figure 2, Supplementary Figure 2 |
| `per_drug_HGT_learned_emb.csv` | `v2_fair/` | 11 | Figure 2 |
| `per_drug_Ridge_expr_GNN_emb.csv` | `v3_embeddings/per_drug_Ridge(expression+GNN_emb).csv` | 12 | Figure 2 |
| `cell_line_embeddings.csv` | `v3_embeddings/` | 12 | Supplementary Figure 3 |
| `reference_baselines.csv` | `v13_reaudit/` | 59 | The two reference rows of Table 1 (per-drug mean, transfer Ridge on TF activities) |
| `transfer_ridge_heldout_perdrug.csv` | `v13_reaudit/` | 59 | Figure 2, test correlation column of Table 5 |
| `graph_ablation.csv` | `v6_strengthen/` | 27 | Supplementary Figure 12 |
| `graph_ablation_per_seed.csv` | `v6_strengthen/` | 27 | Supplementary Figure 12 (one point per seed) |
| `ablation_perdrug_median.csv` | `v13_reaudit/` | 66 | Median per-drug correlation of each ablation configuration (Supplementary Figure 12 caption) |
| `graph_summary.csv` | `results/` | 51 | Node and edge counts of the graph (Methods, Figure 1a) |
| `data_summary.csv` | `results/` | 53 | Methods data constants (matrix sizes, MOA encoding, hold-out split) |
| `crossval_prism_to_gdsc.csv` | `results/` | 05 | Supplementary Figure 14 |

*Integrated Gradients*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `tf_importance_ig_top20.csv` | `v4_xai/tf_importance_ig_seed42.csv` (full ranking of the 771 TFs) | 35 | Figure 3a |
| `tf_importance_ig_stability.csv` | `v4_xai/tf_importance_ig_multi_seed_stability.csv` | 35 | Figure 3a and 3b, the ten core TFs (Table 4, Figure 7, Supplementary Figure 4) |
| `tf_importance_ig_seed_pairs.csv` | `v4_xai/` | 35 | Results (top-20 overlap and rank correlation between seeds) |
| `tf_drug_ig_signed.csv` | `v4_xai/` | 32 | Figure 3c |
| `ig_baseline_stability.csv` | `v4_xai/` | 33 | Results (robustness to the baseline) |
| `ig_diagnostics_summary.csv` | `v13_reaudit/` | 65 | Results and Methods (diagnostics of the attributions) |
| `ig_gradient_sign.csv` | `v13_reaudit/` | 65 | Sign of each attribution of Figure 3c against the sign of its gradient |
| `ig_gradient_sign_by_tf.csv` | `v13_reaudit/` | 65 | The same, per TF |
| `ig_extended_comparison.csv` | `v4_xai/` | 49 | Supplementary Note 1 |
| `ig_extended_core_tfs.csv` | `v4_xai/` | 49 | Supplementary Note 1 (core TFs of each run) |
| `aurora_tf_ig_validation_extended.csv` | `v9_aurora/` | 36 (`--candidate-set extended`) | Supplementary Note 1 (AURORA-US test of the extended core TFs) |

*Multi-cohort survival*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `survival_SCANB.csv` | `v8_multicohort/` | 21 | Tables 2 and 3, Figures 5 and 6, Supplementary Figures 5 and 6 |
| `survival_METABRIC.csv` | `v8_multicohort/` | 21 | Tables 2 and 3, Figures 5 and 6, Supplementary Figures 5 and 6, Supplementary Table S1 |
| `survival_TCGA.csv` | `v8_multicohort/` | 21 | Table 2, Figure 5, Supplementary Figure 5 |
| `meta_analysis_fisher.csv` | `v8_multicohort/` | 21 | Tables 2 and 3, Figure 6 |
| `cohort_summary.csv` | `v8_multicohort/` | 21 | Table 2 and Methods (patients, events, deduplication, follow-up, genefu PAM50 concordance) |
| `signed_meta_stouffer.csv` | `v13_reaudit/` | 63 | Stouffer column of Table 3, Supplementary Table S1 |
| `direction_agreement_summary.csv` | `v13_reaudit/` | 63 | Results (direction agreement between SCAN-B and METABRIC against independence) |
| `positive_control_baserate.csv` | `v13_reaudit/` | 63 | Results (positive controls against the base rate of significant drugs) |
| `metabric_dss_survival.csv` | `v13_reaudit/` | 63 | Supplementary Table S1 (METABRIC disease-specific survival) |
| `meta_dss_summary.csv` | `v13_reaudit/` | 63 | Supplementary Table S1, Results |
| `forest_positive_controls.csv` | `v8_multicohort/` | 43 | Table 3 (confidence intervals), Supplementary Figure 7 |
| `moa_enrichment.csv` | `v8_multicohort/` | 52 | Supplementary Figure 9 |
| `target_enrichment.csv` | `v8_multicohort/` | 52 | Supplementary Figure 8 |
| `scanb_km_logrank.csv` | `v5_final/` | 58 | Figure 4a (log-rank tests) |
| `scanb_km_extras.csv` | `v13_reaudit/` | 58 | Figure 4 (trend tests, ten-year survival, PAM50-stratified tests) |
| `scanb_km_at_risk.csv` | `v13_reaudit/` | 58 | Figure 4 (numbers at risk) |
| `pam50_composition_sensitive_half.csv` | `v13_reaudit/` | 61 | Figure 4 caption (PAM50 composition of each half of the median split) |
| `penalizer_sweep.csv` | `v8_multicohort/` | 45 | Supplementary Figure 11, Discussion |
| `hr_per_sd_all_drugs.csv` | `v13_reaudit/` | 64 | HR per SD column of Table 5, hazard ratios per SD in the Supplementary Figure 11 caption |
| `pca_residualization_summary.csv` | `v8_multicohort/` | 42 | Discussion (shared prognostic axis) |
| `pca_residualization_perdrug.csv` | `v8_multicohort/` | 42 | Discussion |
| `pc1_orientation.csv` | `v13_reaudit/` | 42 | Discussion (orientation of PC1) |
| `tf_level_survival_scanb.csv` | `v8_multicohort/` | 44 | Results (TF-level Cox) |
| `basal_her2_survival.csv` | `v7_weaknesses/` | 19b | Supplementary Figure 13 |
| `basal_only_survival.csv` | `v7_weaknesses/` | 19b | Results (basal-like alone) |
| `subtype_pooling_summary.csv` | `v7_weaknesses/` | 19b | Supplementary Figure 13 caption, Results |
| `multiplicity_summary.csv` | `v13_reaudit/` | 66 | Methods (Benjamini-Yekutieli correction, direction of the SCAN-B associations, hazard-ratio range, provenance of the hyperparameters) |
| `metabric_pam50_coverage.csv` | `v13_reaudit/` | 66 | Methods (PAM50 genes present in the METABRIC matrix) |

*Sensitivity analyses*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `cox_clinical_comparison.csv` | `v8_multicohort/` | 37 | Results (full clinical adjustment in SCAN-B and METABRIC) |
| `cox_clinical_adjusted_scanb.csv` | `v8_multicohort/` | 37 | Unrounded fits behind the previous file |
| `cox_clinical_adjusted_metabric.csv` | `v8_multicohort/` | 37 | Unrounded fits behind the previous file |
| `cox_rfs_comparison.csv` | `v8_multicohort/` | 40 | Discussion (relapse-free survival as endpoint) |
| `cox_rfs_scanb.csv` | `v8_multicohort/` | 40 | Unrounded fits behind the previous file |
| `combat_vs_qt_summary.csv` | `v8_multicohort/` | 41 | Discussion (ComBat against QuantileTransformer) |
| `combat_vs_qt_comparison.csv` | `v8_multicohort/` | 41 | The same comparison per drug |
| `talazoparib_family12_summary.csv` | `v13_reaudit/` | One-off check (its script is not a stage and is not shipped) | Check that adding talazoparib to scripts 37 and 40 (twelve drugs instead of eleven) leaves the other results unchanged. The generator compares it when present |
| `tnbc_subgroup_cox.csv` | `v13_reaudit/` | 61 | Supplementary Table S3 (TNBC and Basal columns) |
| `tnbc_subgroup_summary.csv` | `v13_reaudit/` | 61 | Supplementary Table S3 caption, Results |
| `pam50_predicted_auc.csv` | `v13_reaudit/` | 61 | Results (predicted AUC by PAM50 subtype) |
| `treatment_adjusted_cox.csv` | `v13_reaudit/` | 62 | Supplementary Tables S1 and S3 (treatment-adjusted columns) |
| `treatment_interaction_cox.csv` | `v13_reaudit/` | 62 | Supplementary Table S1 (chemotherapy interaction) |
| `treatment_summary.csv` | `v13_reaudit/` | 62 | Patients and events of those models (Supplementary Tables S1 and S3 captions) |
| `ph_summary.csv` | `v13_reaudit/` | 64 | Supplementary Table S3 (proportional hazards), Methods |
| `ph_schoenfeld.csv` | `v13_reaudit/` | 64 | Supplementary Table S3 (proportional hazards) |
| `cox_pam50_stratified.csv` | `v13_reaudit/` | 64 | Supplementary Table S3 (PAM50 strata column) |
| `ph_summary_metabric.csv` | `v13_reaudit/` | 64 | Results (proportional hazards in METABRIC) |
| `ph_schoenfeld_metabric.csv` | `v13_reaudit/` | 64 | Results (proportional hazards in METABRIC) |
| `cox_pam50_stratified_metabric.csv` | `v13_reaudit/` | 64 | Results (PAM50 strata in METABRIC) |
| `collectri_snapshot_sensitivity.csv` | `v11_collectri_snapshot/` | 56 | Supplementary Note 2 (CollecTRI version) |
| `cohort_clipping.csv` | `v15_gene_universe/` | 71 | Supplementary Note 2 (TF activities outside the training range of the transfer) |
| `tcga_screen_comparison.csv` | `v15_gene_universe/` | 71 | Supplementary Note 2 |
| `fisher_meta_comparison.csv` | `v15_gene_universe/` | 71 | Supplementary Note 2 |
| `tcga_ph_schoenfeld.csv` | `v15_gene_universe/` | 71 | Supplementary Note 2 |
| `gene_universe_summary.csv` | `v15_gene_universe/` | 71 | Supplementary Note 2 |
| `gene_universe_r_summary.csv` | `v15_gene_universe/` | 70 | Supplementary Note 2 (genes shared by TCGA-BRCA and PRISM) |

*AURORA-US*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `aurora_tf_activities.csv` | `v9_aurora/` | 22 | Figure 7 (per-sample TF activities, pseudonymized sample codes) |
| `aurora_cohort_summary.csv` | `v9_aurora/` | 22 | Methods (samples, patients, primary, metastatic, paired) |
| `aurora_tf_background.csv` | `v13_reaudit/` | 60 | Figure 7b (the paired test over every TF) |
| `aurora_background_summary.csv` | `v13_reaudit/` | 60 | Results (background rate of the paired test) |
| `aurora_site_background.csv` | `v13_reaudit/` | 60 | Results (site concordance for every TF) |
| `aurora_drug_paired.csv` | `v13_reaudit/` | 60 | Supplementary Figure 10, Results (paired comparison of the 1,447 drugs) |
| `aurora_drug_primary_vs_meta.csv` | `v9_aurora/` | 22 | Per-sample comparison of the drugs (`aurora_drugs_*` keys), kept as a check. The text cites the paired comparison |
| `aurora_tf_by_site.csv` | `v9_aurora/` | 39 | Results (metastatic sites) |
| `aurora_tf_by_site_summary.csv` | `v9_aurora/` | 39 | Results (metastatic sites) |

*TNBC cell lines of PRISM and FUSCC-TNBC*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `tnbc_selectivity_primary.csv` | `v13_reaudit/` | 67 | Figure 8a |
| `tnbc_selectivity_background.csv` | `v13_reaudit/` | 67 | Figure 8, Results (the evaluable drugs) |
| `tnbc_set_test.csv` | `v13_reaudit/` | 67 | Figure 8a and 8b, Results (composite of the seven candidates) |
| `tnbc_permutation_null.csv` | `v13_reaudit/` | 67 | Figure 8b |
| `tnbc_power_mde.csv` | `v13_reaudit/` | 67 | Figure 8a, Results (minimum detectable effect) |
| `assay_sensitivity_gate.csv` | `v13_reaudit/` | 67 | Figure 8a (assay checks and calibration) |
| `tnbc_cell_lines_summary.csv` | `v13_reaudit/` | 67 | Figure 8, Results |
| `tnbc_sensitivity.csv` | `v13_reaudit/` | 67 | Results (sensitivity arms) |
| `tnbc_excluded_drugs.csv` | `v13_reaudit/` | 67 | Drugs left out because too few TNBC lines were measured |
| `tnbc_orthogonality.csv` | `v13_reaudit/` | 67 | Results (TNBC selectivity against the SCAN-B associations) |
| `gdsc_replication.csv` | `v13_reaudit/` | 67 | Results (replication in GDSC) |
| `fuscc_primary_correlation.csv` | `v14_fuscc/` | 69 | Figure 8d, Results (primary test) |
| `fuscc_permutation_null.csv` | `v14_fuscc/` | 69 | Figure 8d |
| `fuscc_cox_perdrug.csv` | `v14_fuscc/` | 69 | Figure 8c and 8d |
| `fuscc_composite.csv` | `v14_fuscc/` | 69 | Figure 8c, Results |
| `fuscc_cohort_gate.csv` | `v14_fuscc/` | 69 | Figure 8c, Results (prognostic signal of the cohort) |
| `fuscc_scanb_tnbc_reference_z.csv` | `v14_fuscc/` | 69 | The SCAN-B TNBC reference vector of the primary test |
| `fuscc_sensitivity.csv` | `v14_fuscc/` | 69 | Results (sensitivity analyses) |
| `fuscc_power_mde.csv` | `v14_fuscc/` | 69 | Results (minimum detectable effect) |
| `fuscc_ph_tests.csv` | `v14_fuscc/` | 69 | Methods (proportional hazards) |
| `fuscc_excluded_summary.csv` | `v14_fuscc/` | 69 | Methods (exclusions counted by reason) |
| `fuscc_rnaseq_vs_not.csv` | `v14_fuscc/` | 69 | Methods (sequenced against non-sequenced patients) |
| `fuscc_covariate_coding.csv` | `v14_fuscc/` | 69 | Coding of the clinical covariates |
| `fuscc_summary.csv` | `v14_fuscc/` | 69 | Results and Methods (patients, events) |

*Reporting*

| File | Copied from | Script | Backs |
|---|---|---|---|
| `cohort_demographics.csv` | `v10_reporting/` | 54 | Reporting Summary (aggregate descriptors of each cohort) |
| `iclac_summary.csv` | `v10_reporting/` | 55 | Methods, Supplementary Table S4 caption |
| `iclac_matches.csv` | `v10_reporting/` | 55 | Supplementary Table S4 |

Figure 1 has no data input other than `paper_statistics.json`, from which it takes the node and edge counts of the graph.

## Trained models

`scripts/57_train_gnns.py` trains every GNN of the paper once with the protocol of `src/models/training.py` (800 epochs, the weights of the last epoch, all 1,448 drugs in training, test data never used to select a checkpoint). It writes 42 runs to `results/v12_gnn/`: GraphSAGE and HGT on the seed-42 cell-line split with five initialization seeds (42, 123, 456, 789, 1011) for Table 1, GraphSAGE on splits 123 and 456 for the multi-seed Integrated Gradients, and GraphSAGE with the five seeds on each of the five reduced graphs of the ablation and on the control that rewires the drug-target edges with degree-preserving swaps (tag `shuffled_drug_target`).

For every run the package ships a JSON sidecar (hyperparameters, edge counts, test metrics, versions, device and the SHA-256 of the checkpoint, plus the SHA-256 of the rewired edges for the control), its per-drug test correlations in `per_drug/` and its per-epoch history in `history/`, and `runs.csv` has one row per run. The checkpoints (PyTorch `state_dict`, 4.0 MB each, 28.3 MB in total) are shipped only for the seven runs that later scripts load:

| Checkpoint | Used by |
|---|---|
| `sage_split42_init42_full.pt` (canonical model) | `12_gnn_embeddings_perdrug.py`, `32_xai_ig.py`, `33_ig_baseline_robustness.py`, `35_ig_multi_seed.py`, `65_ig_diagnostics.py` |
| `sage_split42_init{123,456,789,1011}_full.pt` | `12_gnn_embeddings_perdrug.py` (hybrid row of Table 1, five seeds) |
| `sage_split123_init123_full.pt`, `sage_split456_init456_full.pt` | `35_ig_multi_seed.py` |

`src/models/training.load_run` refuses a checkpoint whose SHA-256 differs from its sidecar. Loading one requires the heterogeneous graph (`data/precision_graph/hetero_graph_prism.pkl`, not redistributed) because the architecture is instantiated from the graph metadata. The Table 1 rows of HGT and the graph ablation are computed from the metrics of the runs, so their checkpoints are not needed downstream and are not shipped. These runs replace the checkpoints of v1.1.0 (`results/v4_xai/best_sage_*.pt`), which came from separate training scripts in which 501 of the 1,448 drugs never reached training, because of the case of their names, and the checkpoint was chosen on the test lines.

## Environment

**Python 3.12.10.** Install with `pip install -r requirements.txt` or `conda env create -f environment.yml` (environment name `precision-paper`). Exact versions used for the paper:

| Package | Version | Package | Version |
|---|---|---|---|
| torch | 2.11.0 (CUDA 12.8 build) | scipy | 1.17.1 |
| torch-geometric | 2.7.0 | matplotlib | 3.10.8 |
| captum | 0.9.0 | seaborn | 0.13.2 |
| lifelines | 0.30.3 | umap-learn | 0.5.11 |
| decoupler | 2.1.4 | networkx | 3.6.1 |
| omnipath | 1.0.12 | matplotlib-venn | 1.1.2 (Figure 5 only) |
| pandas | 2.2.3 | scikit-learn | 1.8.0 |
| numpy | 2.4.3 | combat (pycombat) | 0.3.3, used only by `scripts/41_combat_sensitivity.py` |
| requests | 2.33.0 | openpyxl | 3.1.5, for `scripts/00c_prepare_fuscc.py` and `55_iclac_check.py` |
| optuna | 4.8.0, for `scripts/72_baseline_hp_search.py` | pyreadr | 0.5.4, for `scripts/53_data_summary.py` and `66_multiplicity_summary.py` |
| pytest | 9.1.1, for `tests/` | | |

The pinned `torch` resolves to the default PyPI wheel of your platform, which on Linux is the CUDA build with its `nvidia-*` dependencies (several GB). For a CPU-only machine install `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` first and then the rest of the requirements. For a GPU build install `torch==2.11.0` from the PyTorch index for your CUDA version first. Level 2 (tables and figures) needs no GPU: it imports torch only through `src/data/splits.py`, so the CPU-only wheel is enough. Only the direct dependencies are pinned: transitive packages such as numba and llvmlite (used by umap-learn for Supplementary Figure 3) resolve to whatever pip selects. The figures also need the Arial font (`arial.ttf` and `arialbd.ttf`).

From the package root, `python -m pytest tests` runs the tests: 55 pass and 4 are skipped without the graph, because `tests/test_training.py` trains a few epochs on `data/precision_graph/hetero_graph_prism.pkl`. `tests/test_zph_exact.py` compares the exact proportional hazards test of script 64 with R survival through reference values that `tests/zph_exact_reference.R` printed.

**R 4.5.2 with Bioconductor 3.22.** Install with `Rscript R/install.R`, which pins the Bioconductor release, installs the CRAN packages from the archive at their exact versions and prints a comparison table at the end. `R/sessionInfo.txt` records the versions and which of the original R scripts loads which package.

| Package | Version | Package | Version |
|---|---|---|---|
| decoupleR | 2.16.0 | SummarizedExperiment | 1.40.0 |
| genefu | 2.42.0 | dplyr | 1.2.0 |
| survival | 3.8.3 | tidyr | 1.3.2 |
| survminer | 0.5.2 | tibble | 3.3.1 |
| OmnipathR | 3.18.4 (live CollecTRI fallback only) | data.table | 1.18.2.1 (scripts 68 and 70) |

The data.table version is also recorded in `data/precision_processed/fuscc_r_summary.csv`. `scripts/00_download_tcga_brca.R` additionally needs TCGAbiolinks (Bioconductor) and `scripts/00b_download_metabric.R` needs httr and jsonlite (CRAN). Neither is part of `R/install.R` because they are only used to fetch inputs.

## Configuration

All paths are resolved by `config.py` (Python) and `R/config.R` (R), both anchored to the package root, so every script can be launched from any working directory. The one exception is `scripts/72_baseline_hp_search.py`, shipped exactly as it ran, which uses `data/` and `results/` relative to the working directory and must be run from the package root.

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

* **Two figures need data that are not shipped.** Figure 4 and Supplementary Figure 4 are skipped by `generate_paper_figures.py` with the shipped files (see *Level 2*). Their shipped PNGs are the published ones. Every other figure, including Supplementary Figure 3 (the UMAP of the cell-line embeddings, which needs `data/DepMap/Model.csv`, included, and umap-learn), is regenerated.
* **Some stages need more than the package.** Besides the stages that read raw inputs, `scripts/66_multiplicity_summary.py` (stage `reaudit`) reads the provenance of the earlier hyperparameters, its section [10], from the git history of the private development repository. Without that history it warns and keeps the rows of that section from the shipped `multiplicity_summary.csv`. Scripts 39 and 60 need the GEO sample codes of AURORA-US (see *Level 3*). Without the internal analysis plans, scripts 67 and 69 record `absent` as their hash. `generate_paper_results.py` skips the one check that needs commit `3d539a0` and runs the check of the previous Table 1 rows without its comparison with that commit.
* **CollecTRI snapshot versus live network.** The graph and the TF activities of PRISM, GDSC, AURORA-US and FUSCC-TNBC were computed with the CollecTRI snapshot shipped in `data/prior_knowledge/collectri_edges.csv` (42,990 edges, 1,185 TFs). The SCAN-B, METABRIC and TCGA-BRCA matrices used in the paper were computed in April 2026 with CollecTRI retrieved through decoupleR, whose regulons differ for some TFs, and the R scripts now read the snapshot. Rebuilding those matrices with the snapshot changes 125 (SCAN-B), 119 (METABRIC) and 137 (TCGA-BRCA) TFs, gives 604 instead of 623 SCAN-B drugs at FDR < 0.05, 87 instead of 74 in METABRIC and 558 instead of 551 in the Fisher meta-analysis, and keeps six of the seven Table 5 candidates (indirubin replaces trametinib). Supplementary Note 2 reports this comparison (`collectri_snapshot_sensitivity.csv`, script 56). The shipped results come from the April 2026 matrices, which the manuscript reports.
* **Gene universe of each cohort.** The TF activity of the univariate linear model grows with the number of genes, and the quantile transformation of the transfer, fitted on PRISM, clips what falls outside its training range: 12.7% of the SCAN-B activities, 12.3% of the METABRIC ones and 80.3% of the published TCGA-BRCA ones. Recomputed over the genes shared with PRISM, TCGA-BRCA still has no drug at FDR < 0.05 and the Fisher meta-analysis moves from 551 to 559 drugs (scripts 70 and 71, Supplementary Note 2). The published TCGA-BRCA results are kept. FUSCC-TNBC is computed over the PRISM gene universe from the start.
* **OmniPath drift.** The `data` stage downloads the protein-protein interactions live from OmniPath and caches them. The paper's graph was built from 84,587 interactions (169,174 directed edges). OmniPath is updated continuously and the merged interaction set cannot be redistributed, so a full rerun produces a slightly different graph and slightly different GNN results. The counts in `results/graph_summary.csv`, written by script 51 from the canonical graph and checked by `generate_paper_results.py`, are the reference against which a rebuilt graph should be compared.
* **Sanity checks are calibrated to the published run.** The checks at the end of `generate_paper_results.py` encode the numbers stated in the manuscript (for instance, the penalizer sweep must yield exactly 306 FDR-significant drugs at penalizer 0.5). After a Level 3 rerun with a drifted OmniPath set, or on different hardware, some checks may fail even when the analysis is correct: the exit code then flags that the manuscript numbers would have to be updated.
* **Training is seeded but not bit-reproducible across hardware.** Every stage fixes `SEED=42`, and the GNN runs use the split and initialization seeds of `scripts/57_train_gnns.py`, but CUDA kernels and driver versions can change low-order digits of the GNN results. The seven shipped checkpoints are pinned by the SHA-256 of their sidecars, so Integrated Gradients reruns use the exact weights of the paper.
* **Pseudonymized folds of the hyperparameter search.** `results/v16_hp_search/inputs/manifest.json` and the outputs of the supercomputer job record the SHA-256 of the original `folds.csv`, which carried DepMap `PatientID` values. The shipped `folds.csv` carries the same `D0001` to `D1932` codes as `Model.csv`, and `generate_paper_results.py` checks it against the key `hp_folds_pseudonymized_sha256` instead.
* **Some inputs have restrictive or unclear redistribution terms.** METABRIC (cBioPortal, no explicit license), GDSC (non-commercial terms of use) and the FUSCC-TNBC clinical table (Elsevier user license) are therefore not shipped, and neither are the TF-activity matrices derived from them. `data/README.md` explains how to obtain each file.

## Adapting the pipeline to another setting

The scripts are written for the analysis of the paper and take few command-line arguments (`--candidate-set` of scripts 32, 33, 35 and 36, `--set` and `--force` of script 57, `--raw-dir` of script 00c and the subcommands of script 72). To reuse the method on another cancer type or drug panel, the places to edit are:

* the cell-line filters (TNBC, breast lineage): `src/data/load_depmap.py` (`filter_tnbc_lines`), `src/data/breast_subtypes.py` and `identify_tnbc_cells` in `scripts/32_xai_ig.py`
* the candidate selection rule (`SELECTION_RULE`, `KINASE_HDAC_MOA`, `PHASE_RANK` in `src/analysis/candidate_selection.py`) and the candidate and control drugs named in the downstream scripts: `scripts/32_xai_ig.py` (`CANDIDATE_DRUGS`, `resolve_candidate_set`, which scripts 33 and 35 reuse), `21`, `37`, `40`, `42`, `43`, `56` and `58` to `71` except `65`, `68` and `70`
* the GNN architectures, optimizer and ablation sets (`ARCHS`, `OPTIM`, `ABLATIONS`, `SHUFFLED` in `src/models/training.py`) and the seeds of `scripts/57_train_gnns.py`
* the Ridge and Random Forest hyperparameters (`RIDGE_ALPHA`, `RF_PARAMS` in `scripts/11_rerun_fair_comparison.py`), which come from the search of script 72
* the TF set of the site-stratified analysis: `scripts/39_aurora_site_stratified.py`

`src/graph/build_hetero_graph.py` names its PPI argument `string_ppi` for historical reasons: the paper feeds it the OmniPath interactions (`scripts/02_build_graph.py`), and the STRING loader is only a fallback in `scripts/01_prepare_data.py`.

## License and citation

The code in this repository is released under the **GNU General Public License v3.0** (see `LICENSE`), Copyright (C) 2026 Carlos Fernandez-Lozano, David Ferreiro and Patricia V.-del-Rio.

**Data files under `data/` are not covered by that license.** Each keeps the license of its source, and `DATA_LICENSES.md` records those licenses, the attribution each one requires and the changes made to each file. In short: `prism_drug_annotations.csv` and `drug_response_prism.csv` are CC BY 4.0 from the PRISM Repurposing 19Q4 figshare deposit, whose annotations are curated by the Broad Drug Repurposing Hub. `Model.csv` is a subset of DepMap release 25Q3 with the donor identifiers pseudonymized. That release has no figshare deposit and is governed by the depmap.org Terms and Conditions rather than by CC BY 4.0, which do not permit commercial use. `collectri_edges.csv` is a snapshot of CollecTRI as served by OmniPath: a composite resource whose constituent databases carry their own licenses (CC BY 4.0 for most, CC BY-SA 4.0 for TRRUST, LGPL-3.0 for HTRI, AFL 3.0 for GEREDB). All of them permit redistribution. CC BY 4.0 and AFL 3.0 require attribution, CC BY-SA 4.0 also requires that adaptations of the TRRUST content be shared under CC BY-SA 4.0, and LGPL-3.0 requires that its license text accompany the HTRI content. Anyone who redistributes this snapshot or a modified version of it takes on those conditions. The CollecTRI code repository itself is GPL-3.0. `fuscc_r_summary.csv` holds only versions, checksums and counts of the FUSCC-TNBC TF-activity step.

Results under `results/` and `paper/results/` are per-drug, per-TF or aggregate summary statistics derived from: DepMap and CCLE expression, release 24Q4, and PRISM 19Q4, both from figshare deposits under CC BY 4.0, plus the cell-line annotations of DepMap 25Q3 under the depmap.org terms. The hyperparameter search bundle in `results/v16_hp_search/inputs/` holds the expression, response and TF activities of the 381 training cell lines from those same sources, and its folds carry the pseudonymized donor codes of `Model.csv`. The clinical sources are SCAN-B (CC BY 4.0, Mendeley Data `yzxtxn4nmd` v3), TCGA-BRCA (GDC open access), METABRIC (computed from cBioPortal data that has no explicit license), FUSCC-TNBC (aggregate per-drug tables computed from the CC BY 4.0 FPKM matrix and the clinical annotations of Jiang et al. 2019, which are not redistributed) and AURORA-US (`aurora_tf_activities.csv`, TF activity scores per sample with pseudonymized sample codes, derived from the public series GSE209998, with no clinical or genotype data). GDSC enters through per-drug statistics only (`crossval_prism_to_gdsc.csv`, `gdsc_replication.csv`, `assay_sensitivity_gate.csv`), under the GDSC terms of use, which are non-commercial. The trained checkpoints, `cell_line_embeddings.csv` and the Integrated Gradients tables were computed on a graph that contains OmniPath edges, but they hold model weights, embeddings and attribution scores, not the edges themselves, which is why they are redistributed while the graph and the OmniPath cache are not.

If you use this code or these results, please cite the archived version of this repository (metadata in `CITATION.cff`): DOI [10.5281/zenodo.22657695](https://doi.org/10.5281/zenodo.22657695). That concept DOI always resolves to the latest release. Every release also gets its own version DOI, listed on the Zenodo record, if you need to point at one exact snapshot: 10.5281/zenodo.22657696 for v1.0.0 and 10.5281/zenodo.22660922 for v1.1.0. The version DOI of v1.2.0 is assigned by Zenodo when the release is archived.

Version 1 of the accompanying preprint was posted on bioRxiv on 14 September 2026 under an earlier title:

> Fernandez-Lozano C, Ferreiro D, V.-del-Rio P. Heterogeneous graph neural networks with biological prior knowledge for interpretable drug repurposing in triple-negative breast cancer. bioRxiv 2026. doi: [10.64898/2026.09.08.750045](https://doi.org/10.64898/2026.09.08.750045)

This release reproduces the revised manuscript, *Transcription factor activities as a shared representation for drug repurposing hypotheses in breast cancer*. This section and `CITATION.cff` will be updated with the journal DOI once the article is published.

The Zenodo record archives this repository as released: code, result tables, figures and the seven checkpoints. Raw inputs are not included and are obtained from the sources listed in `data/README.md`.

## Funding and acknowledgments

Grant PID2024-162441OA-I00 funded by MICIU/AEI/10.13039/501100011033 and by "ERDF/EU".

RePo-SUDOE, with project reference S1/1.1/P0033, a project co-financed by the Interreg Sudoe Programme through the European Regional Development Fund (ERDF).

This research project was made possible through the access granted by the Galician Supercomputing Center (CESGA) to its supercomputing infrastructure.
