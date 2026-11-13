# Data for the PRECISION paper package

This directory holds every input the pipeline reads. Five data files are shipped with
the package:

| File | Content | Size | License |
|---|---|---|---|
| `DepMap/Model.csv` | DepMap 25Q3 cell-line metadata, 2,132 models, 8 of the 49 published columns, donor identifier pseudonymized | 168 KB | depmap.org Terms and Conditions |
| `PRISM/processed/drug_response_prism.csv` | PRISM Repurposing 19Q4 secondary screen, AUC, 476 cell lines x 1,448 compounds | 10.7 MB | CC BY 4.0 |
| `precision_processed/prism_drug_annotations.csv` | PRISM 19Q4 compound annotations, one row per compound, 1,448 rows | 126 KB | CC BY 4.0 |
| `precision_processed/fuscc_r_summary.csv` | software versions, checksums and counts of the FUSCC-TNBC TF-activity step (script 68), 17 metrics, no patient data | 519 bytes | see `DATA_LICENSES.md` |
| `prior_knowledge/collectri_edges.csv` | CollecTRI snapshot, 42,990 signed TF-target edges | 699 KB | composite, see `DATA_LICENSES.md` |

`DATA_LICENSES.md` records the license of each of them, the attribution it requires and
the changes made to it. Everything else under `data/` is downloaded from its provider or
rebuilt by a script, for the reasons given in section 4.

Levels 1 and 2 of the top-level `README.md` need nothing under `data/` beyond these
files. `paper/scripts/generate_paper_results.py` reads four of them (`Model.csv`,
`drug_response_prism.csv`, `prism_drug_annotations.csv` and `fuscc_r_summary.csv`), and
`paper/scripts/generate_paper_figures.py` reads `Model.csv` for Supplementary Figure 3.
The figure generator skips Figure 4, which needs `precision_processed/scanb_clinical.csv`
and the SCAN-B per-patient prediction matrix, and Supplementary Figure 4, which needs
`precision_processed/tf_activities_all_prism.csv` and
`precision_processed/tf_activities_scanb.csv`. None of those is shipped (section 4).
Level 3, the full rerun, needs the raw inputs of section 1 in the layout of section 6.

All paths are resolved through `config.py` (Python) and `R/config.R` (R). `DATA_DIR`
defaults to this directory and can be pointed elsewhere with the environment variable
`PRECISION_DATA`. Running `python config.py` prints every resolved path and marks the
files that are missing, which is the quickest way to check an installation before
launching `pipeline.py`. The one exception is `scripts/72_baseline_hp_search.py`, which
is shipped exactly as it ran on the CESGA supercomputer and reads `data/` relative to the
package root, so run it from the root with the default layout.

## 1. Data sources

| Source | Provider and license | Files under `data/` | Status |
|---|---|---|---|
| DepMap / CCLE | [depmap.org](https://depmap.org/portal). Release 25Q3 for `Model.csv`, under the depmap.org Terms and Conditions (**not** CC BY 4.0). Release 24Q4 for expression, figshare [10.25452/figshare.plus.27993248.v1](https://doi.org/10.25452/figshare.plus.27993248.v1), CC BY 4.0. See `DATA_LICENSES.md` | `DepMap/Model.csv` (8-column subset, donor pseudonymized, 168 KB) | **Included, reduced** |
| | | `DepMap/OmicsExpressionProteinCodingGenesTPMLogp1.csv` (24Q4, 518 MB, the name changed in 25Q3, see section 2) | Download |
| PRISM Repurposing secondary screen | figshare [10.6084/m9.figshare.9393293.v4](https://doi.org/10.6084/m9.figshare.9393293.v4) (PRISM Repurposing 19Q4), CC BY 4.0 | `PRISM/processed/drug_response_prism.csv` (476 x 1,448 AUC, 10.7 MB) | **Included** |
| | | `PRISM/raw/secondary-screen-replicate-treatment-info.csv` (15 MB) | Download |
| | | `PRISM/raw/secondary-screen-dose-response-curve-parameters.csv` | Download, only to rebuild the response matrix |
| | | `PRISM/processed/counts_matched_prism.csv`, `PRISM/processed/drug_net_prism.csv` | Derived, not included (section 1.2) |
| GDSC (Sanger) | [cancerrxgene.org](https://www.cancerrxgene.org/), non-exclusive, non-commercial terms of use | `GDSC/processed/drug_response_sanger.csv`, `counts_matched_sanger.csv`, `drug_net_sanger.csv` | Derived, not included (section 1.3) |
| | | `GDSC/raw/sanger-dose-response.csv` | Download, only to rebuild the response matrix |
| SCAN-B | [Mendeley Data yzxtxn4nmd v3](https://data.mendeley.com/datasets/yzxtxn4nmd/3), CC BY 4.0 | `SCANB/SCANB.9206.genematrix_noNeg.Rdata` (1.2 GB) | Download |
| | | `SCANB/clinical_formatted.rds`, `SCANB/Gene.ID.ann.Rdata` | Download / derive |
| METABRIC | [cBioPortal `brca_metabric`](https://www.cbioportal.org/study/summary?id=brca_metabric), no explicit license | `METABRIC/metabric_expression.rds`, `METABRIC/metabric_gene_map.rds`, `METABRIC/metabric_clinical.rds` | Built by `scripts/00b_download_metabric.R` |
| TCGA-BRCA | [GDC](https://portal.gdc.cancer.gov/projects/TCGA-BRCA), open access | `TCGA/tcga_brca_se.rds` | Built by `scripts/00_download_tcga_brca.R` |
| | | `TCGA/tcga_brca_log2tpm_reference.rds` (152 MB, read by script 70 only) | Not built by the package (section 1.6) |
| AURORA-US | [GEO GSE209998](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209998), GEO public data | `AURORA-US/GSE209998_AUR_129_UQN.txt.gz`, `AURORA-US/GSE209998_series_matrix.txt.gz` | Download |
| FUSCC-TNBC | Expression: figshare [10.6084/m9.figshare.19783498.v5](https://doi.org/10.6084/m9.figshare.19783498.v5), CC BY 4.0. Clinical: Table S1 of Jiang et al. 2019 ([10.1016/j.ccell.2019.02.001](https://doi.org/10.1016/j.ccell.2019.02.001)), Elsevier user license, not redistributable | `FUSCC/raw/FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv`, `FUSCC/raw/mmc2.xlsx` | Download |
| | | `FUSCC/fuscc_expression_fpkm.csv.gz`, `FUSCC/fuscc_clinical.csv`, `FUSCC/fuscc_sample_ids.txt`, `FUSCC/README.md` | Built by `scripts/00c_prepare_fuscc.py` |
| ICLAC register | [iclac.org](https://iclac.org), Register of Misidentified Cell Lines, version 14 (15 February 2026) | `reference/iclac_register_v14.xlsx` | Download |
| CollecTRI | [saezlab/CollecTRI](https://github.com/saezlab/CollecTRI) (code GPL-3.0). The network is a composite of sources under CC BY 4.0, CC BY-SA 4.0 (TRRUST), LGPL-3.0 (HTRI) and AFL 3.0 (GEREDB), as declared in the OmniPath resource metadata | `prior_knowledge/collectri_edges.csv` (699 KB) | **Included** (snapshot) |
| OmniPath | [omnipathdb.org](https://omnipathdb.org/), no license of its own, each resource keeps its own | `prior_knowledge/omnipath_ppi.csv` (cache, 1.9 MB) | Downloaded by `01`/`02`, **not redistributed** (section 4.1) |

### 1.1 DepMap

- `Model.csv`: cell-line metadata, read by scripts 01, 17, 32, 33, 35, 55, 65, 67 and 72
  (33 and 65 through the loader of 32) and by both generators. **The included copy is
  a column subset with a pseudonymized donor field, not the file as published by
  DepMap.** It carries the 2,132 rows of release 25Q3 and only the eight fields the
  package reads:

  | Column | Use |
  |---|---|
  | `ModelID` | row identifier (`ACH-XXXXXX`), the index of every reader |
  | `PatientID` | donor of the line, pseudonymized as `D0001` to `D1932` (section 3.1). Script 72 groups its cross-validation folds by donor, and `generate_paper_results.py` counts the Table 1 test lines that share a donor with a training line |
  | `CellLineName` | readable name in the ICLAC check (55) and in the TNBC cell-line tables (67) |
  | `StrippedCellLineName` | readable name in the IG diagnostics (65) |
  | `OncotreeLineage` | breast lines, lineage of Supplementary Figure 3, and stratification of the folds of script 72 |
  | `OncotreeSubtype` | fallback for the TNBC flag and the breast subtype when `ModelSubtypeFeatures` is absent |
  | `RRID` | Cellosaurus accession matched against the ICLAC register by script 55 (147 of the 2,132 models have none) |
  | `ModelSubtypeFeatures` | TNBC flag (substring `TNBC`) and breast subtype (`src/data/breast_subtypes.py`) |

  The loaders also accept `LegacySubSubtype` as a TNBC fallback for releases older
  than those used here. The other 41 columns of the published file, among them donor
  age, sex, race, treatment and stage, are read by no script and are not
  redistributed. To work with the full file, download release 25Q3 from
  [depmap.org/portal](https://depmap.org/portal) and drop it in place: every reader
  selects columns by name, and `generate_paper_results.py` accepts either the DepMap
  donor codes or the pseudonyms, so the complete file is a drop-in replacement. The
  published original has md5 `af4472ab734ea3aec974d992b504c7e5`. Release 25Q3, unlike
  the releases up to 24Q4, has no figshare deposit and is therefore **not under CC BY
  4.0** but under the depmap.org Terms and Conditions. See `DATA_LICENSES.md`.
- `OmicsExpression*.csv`: protein-coding expression, already in log2(TPM+1), one row
  per `ModelID`, columns named `SYMBOL (EntrezID)`. `load_expression()` in
  `src/data/load_depmap.py` keeps only the symbol. Script 01 reads it, and so does the
  rebuild of the response matrices from the raw screens (sections 1.2 and 1.3).
- `CRISPRGeneEffect.csv` is **not needed**. `src/data/load_depmap.py` has a loader for
  it (`load_crispr_effect`), but no script calls it.

### 1.2 PRISM

`PRISM/processed/drug_response_prism.csv` is **included**: the AUC matrix of the PRISM
Repurposing 19Q4 secondary screen, 476 cell lines (rows, DepMap `ModelID`, index column
`depmap_id`) x 1,448 compound names (columns), with 614,325 non-missing values, which are
the response edges of the graph. Almost every script reads it through
`load_prism_response()` in `src/data/load_drug_response.py`, and script 67 and
`generate_paper_results.py` read it directly. 501 of the 1,448 names carry upper-case
letters (for example `1-naphthyl-PP1`), and the scripts lowercase every name on loading
with `normalize_drug_columns()` of `src/data/drug_names.py`. Compare drug names only
after that step.

The matrix was produced by an earlier pipeline of the laboratory, not by a script of
this package. `load_prism_response(use_processed=False)` builds a matrix of this form
from the raw dose-response file: rows restricted to the cell lines of the DepMap
expression matrix, each compound taken from one screen in the priority order
`MTS010 > MTS006 > MTS005 > HTS002`, and the maximum AUC per (cell line, compound name).

Raw files from the PRISM Repurposing 19Q4 secondary screen:

- `secondary-screen-replicate-treatment-info.csv`: columns used are `name`, `moa`,
  `target`, `phase`, `disease.area` and `indication`. Script 01 reads it to write the
  PRISM drug-target edges (`build_drug_target_edges` in
  `src/data/load_prior_knowledge.py`) and the per-compound annotation table
  `precision_processed/prism_drug_annotations.csv`, and `src/data/drug_features.py`
  reads it to build the multi-hot MOA features of the drug nodes. The annotation table
  (one row per compound name, 1,448 rows) is **included** because the candidate
  selection rule of Table 5 (`generate_paper_results.py` and script 56, Supplementary
  Note 2), the MOA and target enrichment of script 52 (Supplementary Figures 8 and 9)
  and script 67 read its MOA, target and phase columns. Without the raw file, script 01 keeps the shipped table.
- `secondary-screen-dose-response-curve-parameters.csv`: columns used are
  `depmap_id`, `screen_id`, `broad_id`, `name`, `auc`. Needed only to rebuild
  `drug_response_prism.csv`.

Two more processed files come from the same earlier pipeline and are not shipped, so
they have to be rebuilt or obtained from the authors:

- `counts_matched_prism.csv`: 476 x 19,193, the DepMap 24Q4 log2(TPM+1) rows of the 476
  cell lines with columns reduced to the gene symbol (147 MB). Read by scripts 01, 02,
  11, 12, 17, 53, 59, 66, 67, 68, 70 and 72.
- `drug_net_prism.csv`: compound annotations (`broad_id`, `name`, `moa`) from the
  treatment-info file, one row per drug-MOA pair. Read by script 01.

The earlier pipeline also wrote `metadata_ccle_prism.csv` (the `Model.csv` rows of the
476 lines). `src/data/load_drug_response.py` still names it, but no script reads it.

### 1.3 GDSC

GDSC enters the cross-screen validation of script 05 (Supplementary Figure 14), the
assay control of script 67, the ICLAC check of script 55 and the counts of scripts 53
and 66. The graph itself is built from PRISM only. The processed files mirror those of
PRISM and come from the same earlier pipeline:

- `drug_response_sanger.csv`: 696 cell lines x 397 drugs, AUC. Read by scripts 01, 05,
  53, 55, 66 and 67.
- `counts_matched_sanger.csv`: 696 x 19,193, DepMap log2(TPM+1) (214 MB). Read by
  script 01 to write `tf_activities_all_gdsc.csv`.
- `drug_net_sanger.csv`: drug annotations (`DRUG_ID`, `SCREENING_SITE`, `name`,
  `SYNONYMS`, `TARGET`, `moa`). Script 01 reads it and passes it to
  `build_drug_target_edges`, which takes GDSC targets only from a lower-case `target`
  column, so it adds no edge: the 3,442 drug-target edges of the paper all come from
  the PRISM treatment-info file.

The raw file `sanger-dose-response.csv` holds GDSC1 and GDSC2 dose-response in the
format distributed through the DepMap portal (columns `ARXSPAN_ID`, `DATASET`,
`DRUG_NAME`, `BROAD_ID`, `auc`). It is needed only to rebuild `drug_response_sanger.csv`
with `load_gdsc_response(use_processed=False)`, where GDSC2 takes precedence over GDSC1
for drugs screened in both. `screened_compounds_rel_8.5.csv` and
`metadata_ccle_sanger.csv` are still named in `src/data/load_drug_response.py`, but no
script reads them.

### 1.4 SCAN-B

`scripts/09_scanb_transfer.R` reads three objects from `SCANB/`:

- `SCANB.9206.genematrix_noNeg.Rdata`: R object `SCANB.9206.genematrix_noNeg`,
  genes (Ensembl IDs) x 9,206 samples, **FPKM already batch-adjusted by the
  SCAN-B consortium**, linear scale. Negative values produced by the
  adjustment were clipped to zero by the consortium (hence `noNeg`). The
  script applies `log2(x + 1)` (range after transformation 0 to 17.3). Do
  not apply any further batch correction, and do not expect the columns to
  sum to one million, FPKM does not.
- `clinical_formatted.rds`: clinical table, one row per sample, derived from
  the clinical annotation distributed with the Mendeley dataset. The
  pipeline expects at least `sample_id`, `include` (`"yes"` keeps the
  sample), `os_time`, `os_event`, `rfs_time`, `rfs_event`, `pam50`, `er`,
  `pr`, `her2`, `age`, `grade`, `size`, `lymphNodePos`, `chemo`, `endo`
  (the full column list is the one of `precision_processed/scanb_clinical.csv`,
  section 5).
- `Gene.ID.ann.Rdata`: R object `Gene.ID.ann` with columns `Gene.ID`
  (Ensembl) and `Gene.Name` (HGNC symbol), used to map the expression rows
  to symbols.

The expression matrix is the one the deposit files under *StringTie FPKM Gene
Data LibProtocol adjusted*: the adjustment brings every sample to a TruSeq-like
library-preparation baseline and leaves the already-TruSeq samples untouched, so
it is not a batch correction in the usual sense and no further correction should
be applied. The `noNeg` suffix marks that the adjustment can produce negative
values, which the consortium clips to zero.

Version 4 of the same DOI (January 2025) adds a folder of raw prepDE counts
(`SCANB.9142.matrixprepDEgenecount`), which the deposit itself states were not
used in the analysis of the single-sample-predictor paper. Those counts do not
work with this pipeline either, because raw counts saturate the quantile
transformer fitted on PRISM. Version 3 is cited because it is the earliest
version that contains the three files this pipeline reads.

### 1.5 METABRIC

The three files are produced by `scripts/00b_download_metabric.R` from the
cBioPortal web API (study `brca_metabric`), using the CollecTRI snapshot of the
package to select the genes. The script needs network access and the R
packages httr and jsonlite.

- `metabric_expression.rds`: `data.frame`, **6,549 genes x 1,980 samples**,
  genes in rows (HGNC symbols), samples in columns (`MB-XXXX` style). This is
  **not the full METABRIC transcriptome**: it is the subset of CollecTRI
  target genes obtained by querying the cBioPortal API for study
  `brca_metabric` restricted to the regulon genes. Illumina HT-12
  log-intensities, range 4.60 to 14.88. Do not log-transform again. It is
  sufficient for TF activities and insufficient as a gene feature matrix.
- `metabric_gene_map.rds`: `data.frame` with `hugoGeneSymbol` and
  `entrezGeneId`, needed by genefu for PAM50 calls.
- `metabric_clinical.rds`: cBioPortal clinical attributes of the 2,509
  patients (`patientId`, `OS_MONTHS`, `OS_STATUS`, `RFS_MONTHS`,
  `RFS_STATUS`, `INTCLUST`, `CLAUDIN_SUBTYPE`, `ER_IHC`, `HER2_SNP6`, ...).

`scripts/20_multicohort_validation.R` reads the three files, adds a `PAM50` column
computed with genefu and writes `precision_processed/tf_activities_metabric.csv` and
`precision_processed/metabric_clinical.csv`. Scripts 19, 53 and 66 also read the `.rds`
files directly. The standalone `scripts/20b_metabric_pam50.R` recomputes the PAM50
calls from the expression and the gene map and writes them back into
`metabric_clinical.csv`, together with `precision_processed/metabric_pam50_proba.csv`.

### 1.6 TCGA-BRCA

`TCGA/tcga_brca_se.rds` is a `SummarizedExperiment` produced by
`scripts/00_download_tcga_brca.R` with TCGAbiolinks (project `TCGA-BRCA`,
Transcriptome Profiling, Gene Expression Quantification, workflow
`STAR - Counts`). `scripts/20_multicohort_validation.R` takes the
`tpm_unstrand` assay (linear TPM), maps Ensembl IDs to symbols through
`rowData(se)$gene_name` and applies `log2(TPM + 1)` so that the scale matches
DepMap. The clinical table is `colData(se)`. The download is about 1.5 GB and
the cached RDS about 760 MB.

`TCGA/tcga_brca_log2tpm_reference.rds` is read only by
`scripts/70_gene_universe_tcga.R`, the diagnostic of Supplementary Note 2 that
recomputes the TCGA-BRCA TF activities over the genes shared with PRISM (script 71
reads its output). It is a matrix of 17,351 gene symbols x 1,231 samples in log2 TPM
(152 MB). **No script of this package writes it**: it was prepared outside the
package. Script 70 checks its dimensions and its scale (maximum below 25) and records
its md5 (`46388f502a095a436931520d158e76b3`) in
`results/v15_gene_universe/gene_universe_r_summary.csv`. Nothing else depends on it,
and the canonical TCGA-BRCA activities (`tf_activities_tcga.csv`) come from
`tcga_brca_se.rds`.

### 1.7 AURORA-US

Two supplementary files of GEO series GSE209998, downloaded as is:

- `GSE209998_AUR_129_UQN.txt.gz`: upper-quartile normalized expression, 58,387 genes
  x 129 samples, linear scale, md5 `12eda41a1eb43ad477443502373a3cea`.
  `scripts/22_aurora_complementary.py` applies `log2(x + 1)` and computes the TF
  activities with ULM against the CollecTRI snapshot.
- `GSE209998_series_matrix.txt.gz`: GEO series matrix, md5
  `5ab10bdd86c57c46e7ecc43136dde903`. Scripts 22, 39 and 60 parse its sample
  characteristics (disease, tissue site, treatment, time), and script 54 counts sex,
  disease, treatment timing, autopsy and metastatic site from them for the Reporting
  Summary.

Script 22 writes `results/v9_aurora/aurora_tf_activities.csv` (129 samples x 770 TFs)
with the GEO sample codes. The copy shipped with the package carries pseudonymized
codes instead (section 3.2). Script 36 reads the patient and the primary or metastasis
role from the code itself, so it runs on the shipped copy. Scripts 39 and 60 look up
the tissue site of each sample in the series matrix by its GEO code, so they stop with
an explicit error while the codes are pseudonymized, and script 60 also reads
`results/v9_aurora/aurora_drug_predictions.csv`, which is not shipped (section 4.2). To
rerun them, start from script 22 with the two GEO files, which writes both files with
the GEO codes. That rerun overwrites the tracked `results/v9_aurora/aurora_tf_activities.csv`:
restore the shipped copy with `git checkout -- results/v9_aurora/aurora_tf_activities.csv`
before committing. `generate_paper_results.py` pseudonymizes the codes again when it
copies the file to `paper/results/`, and `aurora_drug_predictions.csv` is in
`.gitignore`.

`scripts/36_aurora_ig_validation.py` looks for an optional
`AURORA-US/AURORA_clinical.csv` in a fallback branch. The file is not needed:
primary/metastasis pairing is parsed from the sample identifiers.

### 1.8 FUSCC-TNBC

The triple-negative breast cancer cohort of the Fudan University Shanghai Cancer
Center, 465 patients, 360 of them with RNA-seq (Jiang et al., *Cancer Cell* 2019,
doi 10.1016/j.ccell.2019.02.001). Two source files go into `FUSCC/raw/`, the default
input directory of `scripts/00c_prepare_fuscc.py` (any other directory can be passed
with `--raw-dir`):

| File | Source | License | Checked by 00c |
|---|---|---|---|
| `FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv` | figshare deposit of the data descriptor of the cohort (Chen et al., *Scientific Data* 2022, doi 10.1038/s41597-022-01681-z), doi [10.6084/m9.figshare.19783498.v5](https://doi.org/10.6084/m9.figshare.19783498.v5). 56,858 genes x 360 samples, FPKM, linear scale | CC BY 4.0 | md5 `44e173a7b74fe71738d245a9ffe09da0`, the checksum figshare publishes |
| `mmc2.xlsx` | Supplementary Table S1 of Jiang et al. 2019 (sheet `Table S1`, 465 patients x 50 columns), from the supplementary material of the article | Elsevier user license, which does not permit redistribution | SHA-256 of the 31 columns the analysis uses, as read and serialized by the script: `96ab7091b1bbdcc7ae10d9926ff4f2f4c998a434e3372e2b459493a30ef19646` |

The clinical checksum is a hash of the parsed content, not of the file, because cloud
sync clients rewrite the internal parts of Office files without changing any cell. It
was frozen with pandas 2.2.3 and openpyxl 3.1.5, and another version that reads the
sheet differently can fail the check without any cell having changed.

`python scripts/00c_prepare_fuscc.py` checks both sources, repairs the header of the
expression matrix (its first two fields are separated by a space instead of a tab) and
writes to `FUSCC/`:

- `fuscc_expression_fpkm.csv.gz`: 56,858 gene symbols x 360 samples, FPKM, md5
  `329f9adae644472e09cb3b848049f300`, the md5 that scripts 68 and 70 check. No gene
  symbol is duplicated in this release of the matrix.
- `fuscc_clinical.csv`: the 465 patients of Table S1, `Project_ID` plus the 30 other
  columns the analysis uses. It may not be redistributed (section 4.3).
- `fuscc_sample_ids.txt`: the 360 sample identifiers in file order.
- `README.md`: provenance, licenses, checksums and the quirks of the raw files.

The pipeline stage `data_r_fuscc` (`scripts/68_fuscc_tf_activities.R`) then writes
`precision_processed/tf_activities_fuscc.csv` (360 samples x 769 TFs),
`precision_processed/fuscc_clinical.csv` (the 360 sequenced patients, column names
normalized) and `precision_processed/fuscc_r_summary.csv`. The activities are computed
over the 18,822 genes the FPKM matrix shares with `counts_matched_prism.csv` (18,769 of
them with nonzero variance), not over the full transcriptome, because the ULM statistic
grows with the number of genes and the clinical transfer clips activities that fall
outside the range it was fitted on (Supplementary Note 2). Script 69 reads the two
processed files and, for the comparison of sequenced and non-sequenced patients,
`FUSCC/fuscc_clinical.csv`. Script 70 reads the FPKM matrix to measure the clipping
over the full transcriptome. The whole `FUSCC/` directory is in `.gitignore`. So are
the list of excluded patients that script 69 writes
(`results/v14_fuscc/fuscc_excluded_patients.csv`, section 4.2) and every file of
`precision_processed/` except the two included ones.

`fuscc_r_summary.csv` is **included** because `generate_paper_results.py` reads it. It
holds 17 metrics: the R, decoupleR and data.table versions, the md5 of the two
processed inputs, the gene, sample and TF counts of the step, the CollecTRI counts, the
ULM `minsize`, and the maximum FPKM of the full matrix and the maximum log2 value over
the genes that reach ULM. It holds no patient record.

Scripts 68 and 70 load the R package data.table besides decoupleR, dplyr, tidyr and
tibble. `R/install.R` pins it at version 1.18.2.1, the one recorded in
`fuscc_r_summary.csv`.

### 1.9 ICLAC register

`reference/iclac_register_v14.xlsx` is the ICLAC Register of Misidentified Cell Lines,
version 14 (15 February 2026), downloaded from [iclac.org](https://iclac.org). It is read
only by `scripts/55_iclac_check.py`, which matches the `RRID` (Cellosaurus accession) of
each cell line of `Model.csv` against the accessions listed in Tables 1 and 2 of the
register, and writes `results/v10_reporting/iclac_matches.csv` (Supplementary Table S4)
and `iclac_summary.csv`. Script 55 also reads `DepMap/Model.csv`, the graph pickle,
`GDSC/processed/drug_response_sanger.csv` and `results/v4_xai/ig_full_attributions.csv`.
The register is pinned by a SHA-256 of its parsed cell content,
`25c0051a8a86769d2e2f60a4bca8ffbf9939b65f09a11274af951e9512fe66a5`, not by a hash of the
file, for the same reason as the FUSCC-TNBC clinical table. Any later version of the
register fails that check by design.

### 1.10 CollecTRI (included)

`prior_knowledge/collectri_edges.csv` is the exact snapshot used to build the
graph and to compute every TF activity matrix except the published SCAN-B, METABRIC
and TCGA-BRCA ones (provenance note below): **42,990 signed edges, 1,185 TFs, 6,675
target genes**, columns `source`, `target`, `weight` (+1 or -1).
Both the Python loader (`src/data/load_prior_knowledge.py`) and the R loader
(`R/config.R::load_collectri_snapshot`) read this file first. When it is
absent they fall back to a live download through decoupler/OmniPath, which
drifts over time and would change the TF set. Keep the snapshot.

Provenance note: the snapshot was cached by the Python side on 28 March 2026.
The clinical TF-activity matrices used in the paper (SCAN-B, METABRIC,
TCGA-BRCA) were computed in April 2026 by the R scripts with the network
downloaded live at that time, which is why they are not byte-identical to what
the snapshot produces. Measured against the snapshot, the live network of
September 2026 has 43,159 edges and 1,186 TFs: 42,765 edges are shared with the
same sign, 62 TF-target pairs changed sign, and, counting each of those pairs on
both sides, 225 signed edges exist only in the snapshot and 394 only in the live
network (the extra TF is PLSCR1). Downstream scripts intersect the TF columns of
every cohort with the 771 TFs of the PRISM matrix, which bounds the effect of this
drift.

Scripts 00b, 09, 20, 68 and 70 of this package read the snapshot (00b to select the
METABRIC genes), so a rerun of 09 and 20 writes snapshot-based matrices in place of
the published ones. Script 56 (Supplementary Note 2) compares the two versions: next to
the published matrices it reads
`precision_processed/collectri_snapshot/tf_activities_{scanb,metabric,tcga}.csv`, the
output of scripts 09 and 20 on the snapshot, first produced on 7 September 2026.

## 2. DepMap release

The package pins the DepMap release in `data/DepMap/ACTIVE_VERSION`, a one
line text file with the release name (for example `24Q4`). When the file is
absent `config.DEPMAP_VERSION` defaults to `24Q4`. The only file whose name
changed between releases is the expression matrix:

| Release | Expression file name |
|---|---|
| up to 24Q4 | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` |
| 25Q3 onwards | `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` |

`config.depmap_file("expression")` looks for both names on disk and returns
the one that exists, so a single `DepMap/` directory can hold either. Every
other file (`Model.csv`, `CRISPRGeneEffect.csv`) kept its name.

A note on provenance, now settled by evidence rather than inference. The
processed expression used in the paper (`counts_matched_prism.csv`,
`counts_matched_sanger.csv`) has 19,193 genes, and that is the column count of
the **24Q4** protein-coding matrix exactly. The 25Q3 matrix has 19,220, so the
expression cannot come from 25Q3. The `Model.csv` shipped here comes from the 25Q3
file, verified by md5 against the DepMap release manifest, because the
cell-line metadata (TNBC flags, lineage of Supplementary Figure 3) were taken
from it when the Integrated Gradients analyses were run. Expression and
metadata therefore come from two releases, and the licenses differ: 24Q4 is on
figshare under CC BY 4.0, whereas 25Q3 has no deposit and falls under the
depmap.org terms. See `DATA_LICENSES.md`.
Model identifiers (`ACH-XXXXXX`) are stable across releases, so the mismatch
does not affect the matching, but a strict reproduction should download the
24Q4 expression matrix and may use either `Model.csv`.

## 3. Pseudonymized identifiers

### 3.1 DepMap donors

DepMap records the donor of each model in `PatientID` (format `PT-XXXXXX`), a code
shared by every line derived from the same person. The package needs only which lines
share a donor: script 72 keeps all the lines of a donor in the same cross-validation
fold, and `generate_paper_results.py` counts the Table 1 test lines that share a donor
with a training line. The shipped `Model.csv` therefore replaces each DepMap donor code
with `D0001` to `D1932`, numbered in the sorted order of the DepMap codes over all
2,132 models. There are 1,932 donors, 164 of them with more than one model (up to six).
Each pseudonym stands for exactly one DepMap donor, so every grouping by donor is
unchanged: with seed 42, `StratifiedGroupKFold` puts each training line in the same
fold with either coding. The other donor fields of the published file (age, sex, race,
treatment, stage) are not shipped at all (section 1.1).

The same pseudonyms replace the donor codes in
`results/v16_hp_search/inputs/folds.csv`, the fold assignment of the hyperparameter
search of script 72. The `manifest.json` next to it records the SHA-256 of the file as
it was created, with the DepMap codes, so the shipped file is pinned instead by the key
`hp_folds_pseudonymized_sha256` of `paper/results/paper_statistics.json`.

### 3.2 AURORA-US samples

The GEO sample codes of GSE209998 have the form `AUR-XXXX-TTP1-A-1-0-R-XXXX-XX`: the
second field is the AURORA patient code, the third the site within the patient (`TTP`
primary tumor, `TTM` metastasis, `NT` normal tissue, followed by a number) and the rest
are aliquot and plate fields. The analysis needs only which samples belong to the same
patient and the site of each sample. `results/v9_aurora/aurora_tf_activities.csv` and
its byte-identical copy `paper/results/aurora_tf_activities.csv` are therefore shipped
with each sample renamed `AUR-Pnn-<site>`: the 53 patients are numbered `P01` to `P53`
in the sorted order of their AURORA codes, the site field is kept and the aliquot and
plate fields are dropped. Only the first column changed, and every value keeps its
bytes. `generate_paper_results.py` checks the copy (same values, one pseudonym per
patient, site kept), and Figure 7 reads patient and site from these fields. A rerun of
script 22 from the GEO files writes the GEO codes again (section 1.7).

## 4. What is NOT redistributed and why

### 4.1 The graph and the OmniPath cache (licensing)

`precision_graph/hetero_graph_prism.pkl` and `prior_knowledge/omnipath_ppi.csv`
are **not included**. The gene-gene edges of the graph come from OmniPath,
which aggregates resources with mixed licenses. Checked against the OmniPath
resource metadata for our build: of the 76 resources that contribute
interactions to our edge set, **35 do not allow commercial use**, including
the two largest contributors, PhosphoSitePlus (18,060 edges) and HPRD
(16,151 edges), both with an explicit restriction on redistribution. Only
69.6% of the interactions have every contributing source under a commercial
license, and the `license=` parameter of the OmniPath web API does not filter
this endpoint (the three license levels return the same table). Redistributing
the cache or any file that contains its interactions (the pickle) would
redistribute those resources. The trained checkpoints and embeddings shipped under
`results/` hold model weights and contain no interaction.

What is provided instead is the code that builds both files:
`scripts/01_prepare_data.py` downloads and caches the OmniPath interactions
(`OmniPath.get(genesymbols=True, organisms="human")` through the `omnipath`
Python client, self-loops and duplicates removed, one row per interaction)
and `scripts/02_build_graph.py` assembles the graph. Gene nodes are the union
of CollecTRI targets, OmniPath proteins, drug-target genes and the 19,193
expression genes, so the gene count and the PPI edge count depend on the
OmniPath download date. The exact counts of the build used in the paper are
listed below so that the deviation of a rebuild can be measured:

| Component | Count in the paper build |
|---|---|
| Nodes | **23,498** = 20,389 genes + 1,185 TFs + 1,448 drugs + 476 cell lines |
| TF -> gene edges (CollecTRI, signed) | 42,990 |
| Gene - gene edges (OmniPath PPI, stored in both directions) | 169,174 (84,587 interactions x 2) |
| Drug -> gene edges (PRISM `target` column, section 1.3 for GDSC) | 3,442 |
| Cell line - drug response edges (non-missing PRISM AUC) | 614,325 |

OmniPath grows over time. A download made in September 2026 returns 86,189
interactions instead of 84,587, which shifts the gene node count and the PPI
edge count but leaves the CollecTRI, drug-target and response edges unchanged
because those come from the included snapshot and from the PRISM files.

### 4.2 Patient-level data

The package ships no patient-level record of the clinical cohorts. The following are
not distributed. Each is rebuilt from the sources of section 1 by the script that wrote
it, with the exceptions given after the list:

- the clinical tables and TF-activity matrices of SCAN-B, METABRIC, TCGA-BRCA and
  FUSCC-TNBC (`precision_processed/*_clinical.csv`,
  `precision_processed/tf_activities_{scanb,metabric,tcga,fuscc}.csv`,
  `precision_processed/metabric_pam50_proba.csv` and
  `precision_processed/collectri_snapshot/`), and the per-sample expression of every
  cohort,
- the SCAN-B per-patient prediction matrix, `results/v5_final/scanb_perdrug_predictions.csv`
  (220 MB), read by Figure 4,
- the per-sample TF activities written by script 70 and read by script 71
  (`results/v15_gene_universe/tf_activities_tcga_prism_universe.csv` and
  `tf_activities_fuscc_full_transcriptome.csv`),
- the AURORA-US per-sample drug predictions of script 22,
  `results/v9_aurora/aurora_drug_predictions.csv`, read by script 60,
- the list of FUSCC-TNBC patients excluded by script 69 with their reasons
  (`results/v14_fuscc/fuscc_excluded_patients.csv`). Only the counts by reason are
  shipped (`fuscc_excluded_summary.csv`).

Exceptions. A rerun of scripts 09 and 20 writes the snapshot-based matrices, not the
published ones, which were computed with the CollecTRI network downloaded live in April
2026 (section 1.10). No script writes `precision_processed/collectri_snapshot/`: it
keeps the output of scripts 09 and 20 on the snapshot from the run of 7 September 2026,
which script 56 reads. `results/v15_gene_universe/tf_activities_tcga_prism_universe.csv`
needs `TCGA/tcga_brca_log2tpm_reference.rds`, which no script of the package builds
(section 1.6).

This is a rule of the package and not a license question: SCAN-B is under CC BY 4.0 and
TCGA-BRCA is open access. Every record of a person stays in the deposit that published
it, under that deposit's terms, and the package carries statistics computed over
cohorts. The tables under `results/` and `paper/results/` are per-drug, per-TF and
per-model statistics and counts. The one per-sample table shipped is the AURORA-US
TF-activity matrix, with pseudonymized sample codes (section 3.2), because Table 4 and
Figure 7 are computed from it.

Every file of the list above is in `.gitignore` or under an ignored directory
(section 6), so a plain `git add` after a Level 3 rerun leaves them out.

### 4.3 The FUSCC-TNBC clinical table (licensing)

Table S1 of Jiang et al. 2019 is under the Elsevier user license, which does not permit
redistributing it. Neither `mmc2.xlsx` nor the files built from it with one row per
patient (`FUSCC/fuscc_clinical.csv`, `precision_processed/fuscc_clinical.csv`) are
shipped. The FPKM matrix is under CC BY 4.0 but is patient-level data (section 4.2) and
is downloaded from figshare, where its md5 can be checked. Of FUSCC-TNBC the package
carries only aggregated results: the 13 tables of `results/v14_fuscc/` (per-drug Cox
statistics, the permutation null, the cohort gate, the sensitivity arms, the
proportional-hazards tests, the power analysis and counts by covariate level and by
exclusion reason), their byte-identical copies in `paper/results/`, and
`precision_processed/fuscc_r_summary.csv`.

### 4.4 Large third-party files

The DepMap expression matrix (518 MB), the processed expression matrices
`counts_matched_prism.csv` (147 MB) and `counts_matched_sanger.csv` (214 MB), the PRISM
and GDSC raw files, the SCAN-B expression object (1.2 GB), the TCGA
`SummarizedExperiment` (760 MB) and the AURORA-US files are available from their
providers under the licenses in section 1. They must be downloaded or rebuilt into the
layout of section 6. The TCGA reference matrix of script 70 (152 MB) has no public
source of its own (section 1.6).

### 4.5 Files without a clear redistribution license

`METABRIC/*.rds` (cBioPortal has no explicit data license) and the GDSC raw
and processed files (non-commercial terms) are excluded, together with the TF-activity
matrices derived from them. They are rebuilt from the providers as described in
sections 1.3 and 1.5 (`scripts/00b_download_metabric.R` for METABRIC). The ICLAC
register is a third-party document maintained and versioned by ICLAC. The package
pins the version it used by its content hash and ships only the matches it produces
(Supplementary Table S4), not the register.

### 4.6 Regenerable outputs

`precision_processed/` and `precision_graph/` are written by the pipeline stages `data`
(script 01, with the MOA cache of `src/data/drug_features.py`), `data_r_scanb` (09),
`data_r_fuscc` (68), `data_r_cohorts` (20) and `graph` (02). They are not inputs, and
of `precision_processed/` the package ships only `prism_drug_annotations.csv` and
`fuscc_r_summary.csv`, because the generators read them and neither holds patient data
or content under restrictive terms.

## 5. Expected shapes of the processed files

Shapes as rows x columns excluding the index column, as produced for the paper.
TF activity matrices have samples in rows (index column unnamed) and TFs in
columns, ULM scores (decoupler, `minsize = 5`). The TF set differs slightly
between cohorts because the `minsize` filter depends on which target genes
are measured in each cohort. Downstream scripts always intersect the TF
columns with those of `tf_activities_all_prism.csv`.

| File | Shape | Row identifier | Producer |
|---|---|---|---|
| `precision_processed/tf_activities_all_prism.csv` | 476 x 771 | DepMap `ModelID` | `01_prepare_data.py` |
| `precision_processed/tf_activities_all_gdsc.csv` | 696 x 771 | DepMap `ModelID` | `01_prepare_data.py` |
| `precision_processed/drug_target_edges.csv` | 3,442 x 3 (`drug`, `target_gene`, `source`) | none | `01_prepare_data.py` |
| `precision_processed/prism_drug_annotations.csv` (INCLUDED) | 1,448 x 6 (`name`, `moa`, `target`, `disease_area`, `indication`, `phase`) | none | `01_prepare_data.py` |
| `precision_processed/drug_moa_features.npy`, `drug_moa_names.csv` (1,079 drugs), `drug_moa_categories.csv` (155 MOA classes) | 1,079 x 155 | drug name | `src/data/drug_features.py` (cache) |
| `precision_processed/tf_activities_scanb.csv` | 8,438 x 769 | SCAN-B sample id | `09_scanb_transfer.R` |
| `precision_processed/scanb_clinical.csv` | 8,269 rows x 28 named columns | `sample_id` | `09_scanb_transfer.R` |
| `precision_processed/tf_activities_metabric.csv` | 1,980 x 767 | `MB-XXXX` | `20_multicohort_validation.R` |
| `precision_processed/metabric_clinical.csv` | 2,509 rows x 26 columns plus the R row-name column (includes `PAM50`) | `patientId` | `20_multicohort_validation.R` |
| `precision_processed/metabric_pam50_proba.csv` | 1,980 rows x 6 (`Basal`, `Her2`, `LumA`, `LumB`, `Normal`, `sample`) | `sample` | `20b_metabric_pam50.R` |
| `precision_processed/tf_activities_tcga.csv` | 1,231 x 772 | TCGA barcode | `20_multicohort_validation.R` |
| `precision_processed/tcga_clinical.csv` | 1,231 rows x 95 columns plus the R row-name column | `barcode` | `20_multicohort_validation.R` |
| `precision_processed/tf_activities_fuscc.csv` | 360 x 769 | FUSCC-TNBC sample id | `68_fuscc_tf_activities.R` |
| `precision_processed/fuscc_clinical.csv` | 360 rows x 23 named columns | `sample_id` | `68_fuscc_tf_activities.R` |
| `precision_processed/fuscc_r_summary.csv` (INCLUDED) | 17 rows x 2 (`metric`, `value`) | `metric` | `68_fuscc_tf_activities.R` |
| `precision_processed/collectri_snapshot/tf_activities_{scanb,metabric,tcga}.csv` | 8,438 x 769, 1,980 x 767 and 1,231 x 772 | as the published matrices | `09` and `20` run on the snapshot |
| `PRISM/processed/drug_response_prism.csv` (INCLUDED) | 476 x 1,448 | `depmap_id` | inherited (section 1.2) |
| `PRISM/processed/counts_matched_prism.csv` | 476 x 19,193 | DepMap `ModelID` | inherited (section 1.2) |
| `GDSC/processed/drug_response_sanger.csv` | 696 x 397 | `depmap_id` | inherited (section 1.3) |
| `GDSC/processed/counts_matched_sanger.csv` | 696 x 19,193 | DepMap `ModelID` | inherited (section 1.3) |
| `FUSCC/fuscc_expression_fpkm.csv.gz` | 56,858 x 360 | gene symbol | `00c_prepare_fuscc.py` |
| `FUSCC/fuscc_clinical.csv` | 465 rows x 30 columns plus `Project_ID` | `Project_ID` | `00c_prepare_fuscc.py` |
| `TCGA/tcga_brca_log2tpm_reference.rds` | 17,351 x 1,231 (genes in rows) | gene symbol | outside the package (section 1.6) |

Clinical tables and TF matrices do not have the same rows: SCAN-B has 8,269
clinical rows after the `include == "yes"` filter against 8,438 TF rows (one per
sample code), and METABRIC has 2,509 clinical patients against 1,980 expression
samples. The scripts join on the identifier and never assume the two tables are
aligned.

Columns of `scanb_clinical.csv`: `id`, `sample_id`, `patient_id`, `include`,
`frac_dups`, `cohort`, `age`, `os_time`, `os_event`, `rfs_time`, `rfs_event`,
`drfs_time`, `drfs_event`, `chemo`, `anthra`, `endo`, `immu`, `taxan`, `er`,
`pr`, `her2`, `lymphNodePos`, `LN.spec`, `t.stage`, `size`, `grade`, `ic10`,
`pam50`.

Columns of `precision_processed/fuscc_clinical.csv`: `sample_id`, `rfs_event`,
`rfs_time_days`, `subtype_mrna`, `subtype_pam50`, `age`, `t_stage`, `n_stage`, `grade`,
`size_cm`, `ln_positive`, `ki67`, `stils`, `chemo`, `paclitaxel`, `radiotherapy`,
`brca12_mut`, `histology`, `histology_ext`, `erbb2_ihc`, `erbb2_fish`, `surgery_year`,
`sex`.

## 6. Expected layout of `data/`

```
data/
|-- README.md                                   this file
|-- DepMap/
|   |-- ACTIVE_VERSION                          one line, e.g. 24Q4 (optional, default 24Q4)
|   |-- Model.csv                               INCLUDED (25Q3, 2,132 models, 8 columns, donors pseudonymized)
|   `-- OmicsExpressionProteinCodingGenesTPMLogp1.csv        download (24Q4 name)
|       or OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv (name from 25Q3)
|-- PRISM/
|   |-- raw/
|   |   |-- secondary-screen-replicate-treatment-info.csv         download
|   |   `-- secondary-screen-dose-response-curve-parameters.csv   download, only to rebuild the AUC matrix
|   `-- processed/
|       |-- drug_response_prism.csv             INCLUDED (476 x 1,448 AUC)
|       |-- counts_matched_prism.csv            476 x 19,193 log2(TPM+1), rebuild (section 1.2)
|       `-- drug_net_prism.csv                  compound annotations, rebuild (section 1.2)
|-- GDSC/
|   |-- raw/
|   |   `-- sanger-dose-response.csv            download, only to rebuild the AUC matrix
|   `-- processed/
|       |-- drug_response_sanger.csv            696 x 397 AUC
|       |-- counts_matched_sanger.csv           696 x 19,193 log2(TPM+1)
|       `-- drug_net_sanger.csv                 drug annotations with targets
|-- SCANB/
|   |-- SCANB.9206.genematrix_noNeg.Rdata       download (Mendeley), batch-adjusted FPKM
|   |-- clinical_formatted.rds                  clinical table (section 1.4)
|   `-- Gene.ID.ann.Rdata                       Ensembl to symbol annotation
|-- METABRIC/                                   built by scripts/00b_download_metabric.R
|   |-- metabric_expression.rds                 6,549 CollecTRI targets x 1,980 samples
|   |-- metabric_gene_map.rds                   hugoGeneSymbol to entrezGeneId
|   `-- metabric_clinical.rds                   cBioPortal clinical attributes
|-- TCGA/
|   |-- tcga_brca_se.rds                        built by scripts/00_download_tcga_brca.R
|   `-- tcga_brca_log2tpm_reference.rds         read by script 70 only, not built here (section 1.6)
|-- AURORA-US/
|   |-- GSE209998_AUR_129_UQN.txt.gz            download (GEO)
|   `-- GSE209998_series_matrix.txt.gz          download (GEO)
|-- FUSCC/                                      built by scripts/00c_prepare_fuscc.py, NEVER commit
|   |-- raw/
|   |   |-- FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv   download (figshare, CC BY 4.0)
|   |   `-- mmc2.xlsx                           download (Table S1 of Jiang et al. 2019, not redistributable)
|   |-- fuscc_expression_fpkm.csv.gz            56,858 genes x 360 samples, FPKM
|   |-- fuscc_clinical.csv                      465 patients, not redistributable
|   |-- fuscc_sample_ids.txt                    the 360 sample identifiers
|   `-- README.md                               provenance and checksums
|-- reference/
|   `-- iclac_register_v14.xlsx                 download (iclac.org), read by script 55
|-- prior_knowledge/
|   |-- collectri_edges.csv                     INCLUDED (42,990 edges snapshot)
|   |-- omnipath_ppi.csv                        cache written by 01/02, NOT redistributed
|   `-- string_ppi_700.csv                      optional cache of a STRING fallback in 01, unused by the graph
|-- precision_processed/                        written by 01, 09, 20, 68 and drug_features.py
|   |-- prism_drug_annotations.csv              INCLUDED (PRISM 19Q4 MOA/target/phase per compound)
|   |-- fuscc_r_summary.csv                     INCLUDED (versions, checksums and counts of script 68)
|   |-- tf_activities_all_prism.csv
|   |-- tf_activities_all_gdsc.csv
|   |-- drug_target_edges.csv
|   |-- drug_moa_features.npy
|   |-- drug_moa_names.csv
|   |-- drug_moa_categories.csv
|   |-- tf_activities_scanb.csv
|   |-- scanb_clinical.csv
|   |-- tf_activities_metabric.csv
|   |-- metabric_clinical.csv
|   |-- metabric_pam50_proba.csv                written by scripts/20b_metabric_pam50.R
|   |-- tf_activities_tcga.csv
|   |-- tcga_clinical.csv
|   |-- tf_activities_fuscc.csv
|   |-- fuscc_clinical.csv
|   `-- collectri_snapshot/                     read by script 56
|       |-- tf_activities_scanb.csv
|       |-- tf_activities_metabric.csv
|       `-- tf_activities_tcga.csv
`-- precision_graph/
    `-- hetero_graph_prism.pkl                  written by 02, NOT redistributed
```

Only the five files marked INCLUDED are tracked by git. Everything else in this layout
must stay out of commits, `FUSCC/` above all, because its clinical table may not be
redistributed. `.gitignore` covers the downloaded and rebuilt inputs of this layout,
`FUSCC/`, `reference/` and `prior_knowledge/string_ppi_*.csv` among them, and the
per-patient or per-sample outputs of a Level 3 rerun under `results/`
(`v14_fuscc/fuscc_excluded_patients.csv`, `v15_gene_universe/tf_activities_*.csv` and
`v9_aurora/aurora_drug_predictions.csv`). Check `git status --ignored data` before
committing anything under `data/`.
