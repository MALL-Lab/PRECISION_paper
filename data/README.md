# Data for the PRECISION paper package

This directory holds every input the pipeline reads. Only three small files are
shipped with the package (`DepMap/Model.csv`, `prior_knowledge/collectri_edges.csv`
and `precision_processed/prism_drug_annotations.csv`). Everything else is either
downloaded from its provider or rebuilt by a script, for the reasons given in
section 3.

All paths are resolved through `config.py` (Python) and `R/config.R` (R).
`DATA_DIR` defaults to this directory and can be pointed elsewhere with the
environment variable `PRECISION_DATA`. Running `python config.py` prints every
resolved path and marks the files that are missing, which is the quickest way
to check an installation before launching `pipeline.py`.

## 1. Data sources

| Source | Provider and license | Files under `data/` | Status |
|---|---|---|---|
| DepMap / CCLE | [depmap.org](https://depmap.org/portal/data_page/?tab=allData), CC BY 4.0 | `DepMap/Model.csv` (684 KB) | **Included** |
| | | `DepMap/OmicsExpression*.csv` (518 MB, name depends on release, see section 2) | Download |
| | | `DepMap/CRISPRGeneEffect.csv` | Download |
| PRISM Repurposing secondary screen | [depmap.org](https://depmap.org/portal/data_page/?tab=allData) (PRISM Repurposing 19Q4), CC BY 4.0 | `PRISM/raw/secondary-screen-replicate-treatment-info.csv` (15 MB) | Download |
| | | `PRISM/raw/secondary-screen-dose-response-curve-parameters.csv` | Download |
| | | `PRISM/processed/*.csv` (4 files, see section 1.2) | Derived, not included |
| GDSC (Sanger) | [cancerrxgene.org](https://www.cancerrxgene.org/), non-exclusive, non-commercial terms of use | `GDSC/raw/sanger-dose-response.csv` | Download |
| | | `GDSC/raw/screened_compounds_rel_8.5.csv` | Download |
| | | `GDSC/processed/*.csv` (4 files, see section 1.3) | Derived, not included |
| SCAN-B | [Mendeley Data yzxtxn4nmd v3](https://data.mendeley.com/datasets/yzxtxn4nmd/3), CC BY 4.0 | `SCANB/SCANB.9206.genematrix_noNeg.Rdata` (1.2 GB) | Download |
| | | `SCANB/clinical_formatted.rds`, `SCANB/Gene.ID.ann.Rdata` | Download / derive |
| METABRIC | [cBioPortal `brca_metabric`](https://www.cbioportal.org/study/summary?id=brca_metabric), no explicit license | `METABRIC/metabric_expression.rds`, `METABRIC/metabric_gene_map.rds`, `METABRIC/metabric_clinical.rds` | Built by `scripts/00b_download_metabric.R` |
| TCGA-BRCA | [GDC](https://portal.gdc.cancer.gov/projects/TCGA-BRCA), open access | `TCGA/tcga_brca_se.rds` | Built by `scripts/00_download_tcga_brca.R` |
| AURORA-US | [GEO GSE209998](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209998), GEO public data | `AURORA-US/GSE209998_AUR_129_UQN.txt.gz`, `AURORA-US/GSE209998_series_matrix.txt.gz` | Download |
| CollecTRI | [saezlab/CollecTRI](https://github.com/saezlab/CollecTRI) (code GPL-3.0); the network is a composite of sources under CC BY 4.0, CC BY-SA 4.0 (TRRUST), LGPL-3.0 (HTRI) and AFL 3.0 (GEREDB), as declared in the OmniPath resource metadata | `prior_knowledge/collectri_edges.csv` (684 KB) | **Included** (snapshot) |
| OmniPath | [omnipathdb.org](https://omnipathdb.org/), no license of its own, each resource keeps its own | `prior_knowledge/omnipath_ppi.csv` (cache, 1.9 MB) | Downloaded by `01`/`02`, **not redistributed** (section 3) |

### 1.1 DepMap

Three files, all from the same release (see section 2 for the release policy):

- `Model.csv`: cell line metadata. The package uses `OncotreeLineage`
  (breast lines) and `ModelSubtypeFeatures` (TNBC flag, substring `TNBC`),
  with fallbacks to `LegacySubSubtype` and `OncotreeSubtype` for older
  releases. The included copy is the 25Q3 file (2,132 models, 49 columns
  including `ModelID`). It is redistributed as published by DepMap under
  CC BY 4.0 and contains the public model annotations, including donor age,
  sex, race and treatment fields.
- `OmicsExpression*.csv`: protein-coding expression, already in
  log2(TPM+1), one row per `ModelID`, columns named `SYMBOL (EntrezID)`.
  `src/data/load_depmap.py` keeps only the symbol.
- `CRISPRGeneEffect.csv`: Chronos gene effect. Loaded by
  `scripts/01_prepare_data.py` and reported in its summary log only. It does
  not enter the graph or any model.

### 1.2 PRISM

Raw files from the PRISM Repurposing 19Q4 secondary screen:

- `secondary-screen-dose-response-curve-parameters.csv`: columns used are
  `depmap_id`, `screen_id`, `broad_id`, `name`, `auc`. Screens are
  deduplicated by priority `MTS010 > MTS006 > MTS005 > HTS002` and the AUC
  matrix is the maximum per (cell line, compound name).
- `secondary-screen-replicate-treatment-info.csv`: columns used are `name`,
  `moa`, `target`, `phase`, `disease.area` and `indication`. Provides the
  drug-target edges, the multi-hot MOA drug features
  (`src/data/drug_features.py`), the MOA annotations of supplementary figures
  S14 and S16, and the per-compound annotation table
  `precision_processed/prism_drug_annotations.csv` (written by
  `scripts/01_prepare_data.py`, one row per compound name, 1,448 rows,
  **included** in the package because the candidate selection rule of Table 5
  reads its MOA and phase columns). Without the raw file, figures S14 and S16
  are skipped with a warning and the shipped annotation table is kept.

`PRISM/processed/` holds four derived files inherited from an earlier lab
pipeline. No script in this package writes them, so they have to be obtained
from the authors or rebuilt:

- `drug_response_prism.csv`: 476 cell lines x 1,448 drugs, AUC. Rebuild with
  `load_prism_response(use_processed=False)` in
  `src/data/load_drug_response.py` (rows restricted to cell lines present in
  the DepMap expression matrix).
- `counts_matched_prism.csv`: 476 x 19,193, the DepMap log2(TPM+1) rows of
  those 476 cell lines with columns reduced to the gene symbol.
- `drug_net_prism.csv`: compound annotations (`broad_id`, `name`, `moa`)
  from the treatment-info file, one row per drug-MOA pair.
- `metadata_ccle_prism.csv`: the `Model.csv` rows of the 476 cell lines.

### 1.3 GDSC

Raw files:

- `sanger-dose-response.csv`: GDSC1/GDSC2 dose-response in the format
  distributed through the DepMap portal (columns `ARXSPAN_ID`, `DATASET`,
  `DRUG_NAME`, `BROAD_ID`, `auc`). GDSC2 takes precedence over GDSC1 for
  drugs screened in both.
- `screened_compounds_rel_8.5.csv`: compound list of GDSC release 8.5 with
  `DRUG_ID`, `DRUG_NAME`, `TARGET`, `TARGET_PATHWAY`.

`GDSC/processed/` mirrors `PRISM/processed/` (`drug_response_sanger.csv`
696 x 397, `counts_matched_sanger.csv` 696 x 19,193, `drug_net_sanger.csv`,
`metadata_ccle_sanger.csv`). GDSC is only used for the cross-screen
validation, the graph itself is built from PRISM.

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
  section 4).
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
  genes in rows (HGNC symbols), samples in columns (`MB-0000` style). This is
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
  `scripts/20_multicohort_validation.R` adds a `PAM50` column computed with
  genefu before writing `precision_processed/metabric_clinical.csv`.

### 1.6 TCGA-BRCA

`TCGA/tcga_brca_se.rds` is a `SummarizedExperiment` produced by
`scripts/00_download_tcga_brca.R` with TCGAbiolinks (project `TCGA-BRCA`,
Transcriptome Profiling, Gene Expression Quantification, workflow
`STAR - Counts`). `scripts/20_multicohort_validation.R` takes the
`tpm_unstrand` assay (linear TPM), maps Ensembl IDs to symbols through
`rowData(se)$gene_name` and applies `log2(TPM + 1)` so that the scale matches
DepMap. The clinical table is `colData(se)`. The download is about 1.5 GB and
the cached RDS about 760 MB.

### 1.7 AURORA-US

Two supplementary files of GEO series GSE209998, downloaded as is:

- `GSE209998_AUR_129_UQN.txt.gz`: upper-quartile normalised expression, genes
  x 129 samples, linear scale. `scripts/22_aurora_complementary.py` applies
  `log2(x + 1)`.
- `GSE209998_series_matrix.txt.gz`: GEO series matrix, parsed for the sample
  characteristics (patient, tissue site, primary versus metastasis).

`scripts/36_aurora_ig_validation.py` looks for an optional
`AURORA-US/AURORA_clinical.csv` in a fallback branch. The file is not needed:
primary/metastasis pairing is parsed from the sample identifiers.

### 1.8 CollecTRI (included)

`prior_knowledge/collectri_edges.csv` is the exact snapshot used to build the
graph and to compute every TF activity matrix: **42,990 signed edges, 1,185
TFs, 6,675 target genes**, columns `source`, `target`, `weight` (+1 or -1).
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
same sign, 62 changed sign, 225 exist only in the snapshot and 394 only in the
live network (the extra TF is PLSCR1). Downstream scripts intersect the TF
columns of every cohort with the 771 TFs of the PRISM matrix, which bounds the
effect of this drift.

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

An honest note on provenance. The processed expression used in the paper
(`counts_matched_prism.csv`, `counts_matched_sanger.csv`) has 19,193 genes,
which matches the 24Q4 protein-coding matrix. The `Model.csv` shipped here is
the 25Q3 file, because the cell-line metadata (TNBC flags, lineage of
supplementary figure S3) were taken from it when the Integrated Gradients
analyses were run. Expression and metadata therefore come from two releases.
Model identifiers (`ACH-......`) are stable across releases, so the mismatch
does not affect the matching, but a strict reproduction should download the
24Q4 expression matrix and may use either `Model.csv`.

## 3. What is NOT redistributed and why

### 3.1 The graph and the OmniPath cache (licensing)

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
the cache or any file derived from it (the pickle) would redistribute those
resources.

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
| Drug -> gene edges (PRISM `target` column plus GDSC annotations) | 3,442 |
| Cell line - drug response edges (non-missing PRISM AUC) | 614,325 |

OmniPath grows over time. A download made in September 2026 returns 86,189
interactions instead of 84,587, which shifts the gene node count and the PPI
edge count but leaves the CollecTRI, drug-target and response edges unchanged
because those come from the included snapshot and from the PRISM files.

### 3.2 Large third-party files (size)

The DepMap expression matrix (518 MB), the PRISM and GDSC raw files, the
SCAN-B expression object (1.2 GB), the TCGA `SummarizedExperiment` (760 MB)
and the AURORA files are available from their providers under the licenses in
section 1. They are excluded by `.gitignore` and must be downloaded into the
layout of section 5.

### 3.3 Files without a clear redistribution license

`METABRIC/*.rds` (cBioPortal has no explicit data license) and the GDSC raw
and processed files (non-commercial terms) are excluded. They are rebuilt
from the providers as described in sections 1.3 and 1.5
(`scripts/00b_download_metabric.R` for METABRIC).

### 3.4 Regenerable outputs

`precision_processed/` and `precision_graph/` are written by the pipeline
stages `data` (script 01), `data_r_scanb` (09), `data_r_cohorts` (20) and
`graph` (02). They are not inputs and the package does not ship them.

## 4. Expected shapes of the processed files

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
| `precision_processed/drug_moa_features.npy`, `drug_moa_names.csv` (1,079 drugs), `drug_moa_categories.csv` (155 MOA classes) | 1,079 x 155 | drug name | `src/data/drug_features.py` (cache) |
| `precision_processed/tf_activities_scanb.csv` | 8,438 x 769 | SCAN-B sample id | `09_scanb_transfer.R` |
| `precision_processed/scanb_clinical.csv` | 8,269 rows x 28 named columns | `sample_id` | `09_scanb_transfer.R` |
| `precision_processed/tf_activities_metabric.csv` | 1,980 x 767 | `MB-....` | `20_multicohort_validation.R` |
| `precision_processed/metabric_clinical.csv` | 2,509 rows x 26 columns plus the R row-name column (includes `PAM50`) | `patientId` | `20_multicohort_validation.R` |
| `precision_processed/tf_activities_tcga.csv` | 1,231 x 772 | TCGA barcode | `20_multicohort_validation.R` |
| `precision_processed/tcga_clinical.csv` | 1,231 rows x 95 columns plus the R row-name column | `barcode` | `20_multicohort_validation.R` |
| `PRISM/processed/counts_matched_prism.csv` | 476 x 19,193 | DepMap `ModelID` | inherited (section 1.2) |
| `PRISM/processed/drug_response_prism.csv` | 476 x 1,448 | `depmap_id` | inherited (section 1.2) |
| `GDSC/processed/counts_matched_sanger.csv` | 696 x 19,193 | DepMap `ModelID` | inherited (section 1.3) |
| `GDSC/processed/drug_response_sanger.csv` | 696 x 397 | `depmap_id` | inherited (section 1.3) |

Clinical tables have more rows than the matching TF matrices (SCAN-B 8,269
versus 8,438 samples after the `include == "yes"` filter, METABRIC 2,509
patients versus 1,980 expression samples). The scripts join on the identifier
and never assume the two tables are aligned.

Columns of `scanb_clinical.csv`: `id`, `sample_id`, `patient_id`, `include`,
`frac_dups`, `cohort`, `age`, `os_time`, `os_event`, `rfs_time`, `rfs_event`,
`drfs_time`, `drfs_event`, `chemo`, `anthra`, `endo`, `immu`, `taxan`, `er`,
`pr`, `her2`, `lymphNodePos`, `LN.spec`, `t.stage`, `size`, `grade`, `ic10`,
`pam50`.

## 5. Expected layout of `data/`

```
data/
|-- README.md                                   this file
|-- DepMap/
|   |-- ACTIVE_VERSION                          one line, e.g. 24Q4 (optional, default 24Q4)
|   |-- Model.csv                               INCLUDED (25Q3, 2,132 models)
|   |-- OmicsExpressionProteinCodingGenesTPMLogp1.csv        download (name up to 24Q4)
|   |   or OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv (name from 25Q3)
|   `-- CRISPRGeneEffect.csv                    download
|-- PRISM/
|   |-- raw/
|   |   |-- secondary-screen-dose-response-curve-parameters.csv   download
|   |   `-- secondary-screen-replicate-treatment-info.csv         download
|   `-- processed/
|       |-- drug_response_prism.csv             476 x 1,448 AUC
|       |-- counts_matched_prism.csv            476 x 19,193 log2(TPM+1)
|       |-- drug_net_prism.csv                  compound annotations
|       `-- metadata_ccle_prism.csv             Model.csv rows of the 476 lines
|-- GDSC/
|   |-- raw/
|   |   |-- sanger-dose-response.csv            download
|   |   `-- screened_compounds_rel_8.5.csv      download
|   `-- processed/
|       |-- drug_response_sanger.csv            696 x 397 AUC
|       |-- counts_matched_sanger.csv           696 x 19,193 log2(TPM+1)
|       |-- drug_net_sanger.csv                 compound annotations
|       `-- metadata_ccle_sanger.csv            Model.csv rows of the 696 lines
|-- SCANB/
|   |-- SCANB.9206.genematrix_noNeg.Rdata       download (Mendeley), batch-adjusted FPKM
|   |-- clinical_formatted.rds                  clinical table (section 1.4)
|   `-- Gene.ID.ann.Rdata                       Ensembl to symbol annotation
|-- METABRIC/                                   built by scripts/00b_download_metabric.R
|   |-- metabric_expression.rds                 6,549 CollecTRI targets x 1,980 samples
|   |-- metabric_gene_map.rds                   hugoGeneSymbol to entrezGeneId
|   `-- metabric_clinical.rds                   cBioPortal clinical attributes
|-- TCGA/
|   `-- tcga_brca_se.rds                        built by scripts/00_download_tcga_brca.R
|-- AURORA-US/
|   |-- GSE209998_AUR_129_UQN.txt.gz            download (GEO)
|   `-- GSE209998_series_matrix.txt.gz          download (GEO)
|-- prior_knowledge/
|   |-- collectri_edges.csv                     INCLUDED (42,990 edges snapshot)
|   |-- omnipath_ppi.csv                        cache written by 01/02, NOT redistributed
|   `-- string_ppi_700.csv                      optional cache of a STRING fallback in 01, unused by the graph
|-- precision_processed/                        written by 01, 09, 20 and drug_features.py
|   |-- prism_drug_annotations.csv              INCLUDED (PRISM 19Q4 MOA/target/phase per compound)
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
|   |-- tf_activities_tcga.csv
|   `-- tcga_clinical.csv
`-- precision_graph/
    `-- hetero_graph_prism.pkl                  written by 02, NOT redistributed
```

The repository `.gitignore` excludes the third-party downloads
(`DepMap/OmicsExpression*.csv`, `DepMap/CRISPR*.csv`, `PRISM/raw/`,
`GDSC/raw/`, `SCANB/`, `METABRIC/*.rds`, `TCGA/`, `AURORA-US/`), the OmniPath
cache and `precision_graph/`. The two files marked INCLUDED are the only data
files shipped with the package. `PRISM/processed/`, `GDSC/processed/` and
`precision_processed/` are not listed in `.gitignore`, so anything placed
there in a clone would be picked up by `git add`: keep them out of commits
unless the license of the underlying source allows redistribution.
