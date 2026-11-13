# Licenses of the data files distributed with this repository

The **code** in this repository is under the GNU General Public License v3.0, in
`LICENSE`. The **data files** are not: each keeps the license of its source, and this
file records those licenses, the required attribution and the changes made to each
file. Five data files are distributed under `data/` (sections 1 to 5). Sections 6 to 8
cover the files under `results/` and `paper/results/` that carry records of a source
dataset or are derived from a source with terms of its own. Everything else under
`data/` is downloaded from its provider or rebuilt by a script, as `data/README.md`
explains.

---

## 1. `data/precision_processed/prism_drug_annotations.csv`

**License: CC BY 4.0.** <https://creativecommons.org/licenses/by/4.0/legalcode.en>

- **Title of the original work**: PRISM Repurposing 19Q4 Dataset
- **Creators**: Broad DepMap, Steven Corsello, Mustafa Kocak, Todd Golub
- **Source**: figshare deposit, <https://doi.org/10.6084/m9.figshare.9393293.v4>
- **Curation of the annotation fields**: Broad Drug Repurposing Hub. Corsello, S. M.
  et al. The Drug Repurposing Hub: a next-generation drug library and information
  resource. *Nature Medicine* **23**, 405-408 (2017).
  <https://doi.org/10.1038/nm.4306>

**Changes made to the original.** This file is a modified version. It was derived from
`secondary-screen-replicate-treatment-info.csv` of that deposit by keeping the six
columns `name`, `moa`, `target`, `disease.area`, `indication` and `phase`, keeping the
first row of each compound name, sorting by name and renaming `disease.area` to
`disease_area`. No value was altered.

The original work is provided by the licensor on an as-is basis, without warranties of
any kind. See sections 5 and 6 of the license for the full disclaimer and limitation of
liability.

---

## 2. `data/PRISM/processed/drug_response_prism.csv`

**License: CC BY 4.0.** <https://creativecommons.org/licenses/by/4.0/legalcode.en>

- **Title of the original work**: PRISM Repurposing 19Q4 Dataset
- **Creators**: Broad DepMap, Steven Corsello, Mustafa Kocak, Todd Golub
- **Source**: figshare deposit, <https://doi.org/10.6084/m9.figshare.9393293.v4>
- **Reference**: Corsello, S. M. et al. Discovering the anticancer potential of
  non-oncology drugs by systematic viability profiling. *Nature Cancer* **1**, 235-248
  (2020). <https://doi.org/10.1038/s43018-019-0018-6>

**Changes made to the original.** This file is a modified version. It reshapes the
`auc` field of `secondary-screen-dose-response-curve-parameters.csv` of that deposit
into a matrix of 476 cell lines (rows, DepMap `ModelID`) by 1,448 compound names
(columns), with 614,325 non-missing values. Rows are restricted to the cell lines that
also have DepMap expression, each compound is taken from one screen in the priority
order MTS010, MTS006, MTS005, HTS002, and where a cell line and a compound name still
have more than one curve the maximum AUC is kept. These are the steps of
`load_prism_response(use_processed=False)` in `src/data/load_drug_response.py`. No AUC
value was otherwise altered.

The original work is provided by the licensor on an as-is basis, without warranties of
any kind. See sections 5 and 6 of the license for the full disclaimer and limitation of
liability.

---

## 3. `data/DepMap/Model.csv`

**License: the depmap.org Terms and Conditions.** <https://depmap.org/portal/terms>

Not CC BY 4.0. DepMap releases up to 24Q4 are deposited on figshare under CC BY 4.0,
but this file comes from release **25Q3**, which has no figshare deposit and is
distributed only through the portal. The portal terms are a separate instrument that
grants no Creative Commons license and that does not permit Commercial Use of the data.

- **Title**: DepMap Public 25Q3
- **Creator**: Broad DepMap
- **Citation requested by DepMap**: DepMap, Broad (2025). DepMap Public 25Q3. Dataset.
  depmap.org
- **Program reference**: Arafeh, R., Shibue, T., Dempster, J. M., Hahn, W. C. &
  Vazquez, F. The present and future of the Cancer Dependency Map. *Nature Reviews
  Cancer* **25**, 59-73 (2025). <https://doi.org/10.1038/s41568-024-00763-x>

**Changes made to the original.** This file is a **column subset with a pseudonymized
field**, not the file as published. It keeps the 2,132 rows of release 25Q3 and only the
eight fields the pipeline reads: `ModelID`, `PatientID`, `CellLineName`,
`StrippedCellLineName`, `OncotreeLineage`, `OncotreeSubtype`, `RRID` and
`ModelSubtypeFeatures`. The remaining 41 published columns, which include donor `Age`,
`Sex`, `PatientRace`, treatment and stage fields, are read by no script in this package
and are deliberately not redistributed. In `PatientID` each DepMap donor code is
replaced by a pseudonym, `D0001` to `D1932`, numbered in the sorted order of the DepMap
codes, so that the package keeps which lines share a donor and nothing else about the
donor (`data/README.md`, section 3.1). Every other value is as published. The published
file has md5 `af4472ab734ea3aec974d992b504c7e5` and can be downloaded from
<https://depmap.org/portal>. It is a drop-in replacement, because every reader selects
columns by name and the code accepts either form of `PatientID`.

Anyone using this file is bound by the depmap.org terms, in particular the prohibition
on Commercial Use and the obligation to preserve the confidentiality of data relating
to identifiable subjects. Machine learning models are permitted to be used with or on
the data for internal use or shared for non-profit research purposes, which is the use
made here.

---

## 4. `data/prior_knowledge/collectri_edges.csv`

**License: composite.** A snapshot of the CollecTRI regulon as served by OmniPath.
CollecTRI is built from constituent databases that carry their own licenses: CC BY 4.0
for most of them, CC BY-SA 4.0 for TRRUST, LGPL-3.0 for HTRI and AFL 3.0 for GEREDB, as
declared in the OmniPath resource metadata. All of them permit redistribution. CC BY
4.0 and AFL 3.0 require attribution, CC BY-SA 4.0 also requires that adaptations of
the TRRUST content be shared under CC BY-SA 4.0, and LGPL-3.0 requires that its license
text accompany the HTRI content. Anyone who redistributes this snapshot or a modified
version of it takes on those conditions. The CollecTRI code repository itself is
GPL-3.0.

- **Source**: <https://github.com/saezlab/CollecTRI> and <https://omnipathdb.org>
- **Reference**: Muller-Dott, S. et al. Expanding the coverage of regulons from
  high-confidence prior knowledge for accurate estimation of transcription factor
  activities. *Nucleic Acids Research* **51**, 10934-10949 (2023).
  <https://doi.org/10.1093/nar/gkad841>
- **Changes made**: none to the edge values. The file is a snapshot of 42,990 edges
  taken with the Python package decoupler on 28 March 2026 (`load_collectri` in
  `src/data/load_prior_knowledge.py`), keeping only the `source`, `target` and `weight`
  columns and dropping duplicate rows, and stored so that results do not drift with
  the live resource.

The OmniPath protein-protein interaction cache is **not** redistributed here, because
its constituent resources have heterogeneous licenses that do not all permit
redistribution. The package rebuilds it instead.

---

## 5. `data/precision_processed/fuscc_r_summary.csv`

**Content: processing metadata written by script 68, no data record.** The file holds
17 metrics of the step that computes the FUSCC-TNBC TF activities: the R, decoupleR and
data.table versions, two md5 checksums, gene, sample and TF counts, the CollecTRI counts,
the ULM `minsize`, the maximum FPKM of the full matrix (56,858 genes) and the maximum
log2 value over the 18,769 genes that reach ULM.

- **Expression source** (the counts and the two maxima describe it): FUSCC-TNBC RNA-seq
  FPKM matrix, `FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv`, figshare deposit
  <https://doi.org/10.6084/m9.figshare.19783498.v5>, **CC BY 4.0**. Data descriptor:
  Chen, Q. et al. A comprehensive genomic and transcriptomic dataset of triple-negative
  breast cancers. *Scientific Data* **9**, 587 (2022).
  <https://doi.org/10.1038/s41597-022-01681-z>
- **Cohort reference**: Jiang, Y.-Z. et al. Genomic and transcriptomic landscape of
  triple-negative breast cancers: subtypes and treatment strategies. *Cancer Cell*
  **35**, 428-440.e5 (2019). <https://doi.org/10.1016/j.ccell.2019.02.001>
- **Clinical source**: the value `clinical_md5` is the md5 checksum of the processed
  clinical table built from Supplementary Table S1 of Jiang et al. 2019, which is under
  the Elsevier user license and is not distributed. A checksum carries none of the
  content of the table.

---

## 6. `results/v16_hp_search/inputs/`, the input bundle of the hyperparameter search

The input bundle of the Ridge and random forest hyperparameter search of
`scripts/72_baseline_hp_search.py`, written by its `prepare` step and read by the search
that ran on the CESGA supercomputer (job 11133811). It holds the 381 training cell lines
of Table 1 only, about 40 MB in all.

| File | Content | License and source |
|---|---|---|
| `expression_train.csv.gz` | DepMap 24Q4 protein-coding expression, log2(TPM+1), 381 cell lines x 19,193 genes | **CC BY 4.0**, DepMap 24Q4 Public (section 9) |
| `response_train.csv.gz` | PRISM Repurposing 19Q4 AUC, 381 cell lines x 1,448 compounds | **CC BY 4.0**, PRISM Repurposing 19Q4 Dataset (section 2) |
| `tf_train.csv.gz` | TF activities (ULM scores) of the 381 cell lines over 771 TFs, computed from the 24Q4 expression with the CollecTRI snapshot | derived from DepMap 24Q4 (**CC BY 4.0**, section 9) and CollecTRI (composite, section 4) |
| `folds.csv` | 381 rows: `cell_line`, pseudonymized `PatientID`, `OncotreeLineage`, `fold` | donor and lineage from DepMap 25Q3 `Model.csv`, **depmap.org Terms and Conditions** (section 3) |
| `manifest.json` | sizes, the test cell lines, the SHA-256 of the four files above as created and the software environment | metadata written by script 72 |

**Changes made to the originals.** The expression and response files are row subsets of
the matrices of `data/README.md` sections 1.1 and 1.2 (section 2 above for the
response), with expression columns reduced to the gene symbol and compound names
lowercased. Values are not altered. `tf_train.csv.gz` holds computed scores, not source
values. In `folds.csv` the DepMap donor codes are replaced by the pseudonyms of
`data/DepMap/Model.csv` (section 3). The SHA-256 that `manifest.json` records for
`folds.csv` is the one of the file with the DepMap codes, so the shipped file is pinned
instead by the key `hp_folds_pseudonymized_sha256` of
`paper/results/paper_statistics.json`.

The outputs of the search, under `results/v16_hp_search/cesga_11133811/`, are trial
scores and selected hyperparameters and contain no source record.

---

## 7. `results/v9_aurora/aurora_tf_activities.csv` and `paper/results/aurora_tf_activities.csv`

Two byte-identical copies of the AURORA-US TF-activity matrix, 129 samples x 770 TFs,
with pseudonymized sample codes.

**License: none stated by the submitters.** The source is a public record of the Gene
Expression Omnibus. NCBI places no restriction of its own on the use or redistribution
of GEO records and notes that submitters may hold rights in their original data.

- **Source**: GEO series GSE209998, supplementary file `GSE209998_AUR_129_UQN.txt.gz`,
  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209998>
- **Reference**: Garcia-Recio, S. et al. Multiomics in primary and metastatic breast
  tumors from the AURORA US network finds microenvironment and epigenetic drivers of
  metastasis. *Nature Cancer* **4**, 128-147 (2023).
  <https://doi.org/10.1038/s43018-022-00491-x>
- **Regulon**: CollecTRI snapshot (section 4)

**Changes made to the original.** The file holds no expression value. It holds ULM TF
activities computed by `scripts/22_aurora_complementary.py` from the upper-quartile
normalized matrix after `log2(x + 1)`, with the CollecTRI snapshot. Each GEO sample code
is replaced by `AUR-Pnn-<site>`: the 53 patients are numbered `P01` to `P53` in the
sorted order of their AURORA codes, the site field (primary, metastasis or normal tissue
and its number) is kept and the aliquot and plate fields are dropped
(`data/README.md`, section 3.2).

---

## 8. `results/v14_fuscc/` and their copies in `paper/results/`

Thirteen tables of aggregated results of the FUSCC-TNBC analysis (scripts 68 and 69),
each copied byte for byte to `paper/results/`: per-drug Cox statistics over the drugs of
the paper, the SCAN-B TNBC reference vector, the 10,000 permutation replicates of the
correlation statistics, the cohort gate, the composite of the candidates, the
sensitivity arms, the proportional-hazards tests, the power analysis, the comparison of
sequenced and non-sequenced patients, and counts by covariate level and by exclusion
reason.

- **Expression source**: FUSCC-TNBC FPKM matrix, figshare
  <https://doi.org/10.6084/m9.figshare.19783498.v5>, **CC BY 4.0**, Chen et al. 2022
  (section 5)
- **Clinical source**: Supplementary Table S1 of Jiang et al. 2019 (section 5), under
  the **Elsevier user license**, which does not permit redistributing it

**What these tables contain.** Statistics computed over the cohort and counts per
category, with no row per patient and no patient identifier. A few entries are cohort
extremes or small counts, for example the longest observed follow-up
(`max_observed_months` in `fuscc_summary.csv`) and exclusion reasons or covariate
levels that apply to one or two patients. The list of excluded patients that script 69
writes with patient identifiers (`results/v14_fuscc/fuscc_excluded_patients.csv`) and
every patient-level input of the cohort are not distributed. Rebuilding them requires both sources, downloaded as
`data/README.md` section 1.8 describes.

---

## 9. Expression data, downloaded rather than distributed here

The full cell-line expression matrix is not shipped with this package, but the results
are derived from it and the rows of the 381 training cell lines are in the bundle of
section 6, so its license is recorded here too.

**License: CC BY 4.0.** <https://creativecommons.org/licenses/by/4.0/legalcode.en>

- **Title**: DepMap 24Q4 Public
- **Creator**: Broad DepMap
- **Source**: figshare deposit,
  <https://doi.org/10.25452/figshare.plus.27993248.v1>
- **File**: `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, 19,193 protein-coding genes

Note that expression and cell-line annotations come from two different releases, and
therefore from two different licenses: expression from 24Q4 on figshare under CC BY 4.0,
annotations from 25Q3 under the depmap.org terms. The 19,193 genes of the processed
matrices match the 24Q4 column count exactly, whereas 25Q3 has 19,220.

---

## Derived results

The other CSVs under `results/` and `paper/results/` are per-drug, per-TF and per-model
summary statistics and counts, not copies of the source data. Some of them name cell
lines by fields of `Model.csv` (for example `iclac_matches.csv`, with the DepMap
`ModelID`, name, RRID and lineage of each matched line), which come from DepMap 25Q3
(section 3). `iclac_matches.csv` also carries three fields of the ICLAC Register of
Misidentified Cell Lines, version 14, 15 February 2026 (`register_table`,
`registration_id`, `register_name`), for the 13 matched lines. Source: International
Cell Line Authentication Committee, <https://iclac.org>. The trained checkpoints,
`cell_line_embeddings.csv` and the Integrated Gradients tables hold model weights,
embeddings and attribution scores rather than any source record.
