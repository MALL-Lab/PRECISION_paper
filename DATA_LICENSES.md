# Licences of the data files distributed with this repository

The **code** in this repository is under the GNU General Public License v3.0, in
`LICENSE`. The **data files** under `data/` are not: each keeps the licence of its
source, and this file records those licences, the required attribution and the changes
made to each file. Three data files are distributed here. Everything else under `data/`
is downloaded from its provider or rebuilt by a script.

---

## 1. `data/precision_processed/prism_drug_annotations.csv`

**Licence: CC BY 4.0.** <https://creativecommons.org/licenses/by/4.0/legalcode.en>

- **Title of the original work**: PRISM Repurposing 19Q4 Dataset
- **Creators**: Broad DepMap, Steven Corsello, Mustafa Kocak, Todd Golub
- **Source**: figshare deposit, <https://doi.org/10.6084/m9.figshare.9393293.v4>
- **Curation of the annotation fields**: Broad Drug Repurposing Hub. Corsello, S. M.
  et al. The Drug Repurposing Hub: a next-generation drug library and information
  resource. *Nature Medicine* **23**, 405-408 (2017).
  <https://doi.org/10.1038/nm.4306>

**Changes made to the original.** This file is a modified version. It was derived from
`secondary-screen-replicate-treatment-info.csv` of that deposit by reducing it to one
row per compound and by renaming the column `disease.area` to `disease_area`. No value
was altered.

The original work is provided by the licensor on an as-is basis, without warranties of
any kind. See sections 5 and 6 of the licence for the full disclaimer and limitation of
liability.

---

## 2. `data/DepMap/Model.csv`

**Licence: the depmap.org Terms and Conditions.** <https://depmap.org/portal/terms>

Not CC BY 4.0. DepMap releases up to 24Q4 are deposited on figshare under CC BY 4.0,
but this file comes from release **25Q3**, which has no figshare deposit and is
distributed only through the portal. The portal terms are a separate instrument that
grants no Creative Commons licence and that does not permit Commercial Use of the data.

- **Title**: DepMap Public 25Q3
- **Creator**: Broad DepMap
- **Citation requested by DepMap**: DepMap, Broad (2025). DepMap Public 25Q3. Dataset.
  depmap.org
- **Programme reference**: Arafeh, R., Shibue, T., Dempster, J. M., Hahn, W. C. &
  Vazquez, F. The present and future of the Cancer Dependency Map. *Nature Reviews
  Cancer* **25**, 59-73 (2025). <https://doi.org/10.1038/s41568-024-00763-x>

**Changes made to the original.** This file is a **column subset**, not the file as
published. It keeps the 2,132 rows of release 25Q3 and only the four fields the
pipeline reads: `ModelID`, `OncotreeLineage`, `OncotreeSubtype` and
`ModelSubtypeFeatures`. The remaining 45 published columns, which include donor
`PatientID`, `Age`, `Sex`, `PatientRace`, treatment and stage fields, are read by no
script in this package and are deliberately not redistributed. The published file has
md5 `af4472ab734ea3aec974d992b504c7e5` and can be downloaded from
<https://depmap.org/portal>. It is a drop-in replacement, because every loader reads
this path with `index_col=0` and selects columns by name.

Anyone using this file is bound by the depmap.org terms, in particular the prohibition
on Commercial Use and the obligation to preserve the confidentiality of data relating
to identifiable subjects. Machine learning models are permitted to be used with or on
the data for internal use or shared for non-profit research purposes, which is the use
made here.

---

## 3. `data/prior_knowledge/collectri_edges.csv`

**Licence: composite.** A snapshot of the CollecTRI regulon as served by OmniPath.
CollecTRI is built from constituent databases that carry their own licences: CC BY 4.0
for most of them, CC BY-SA 4.0 for TRRUST, LGPL-3.0 for HTRI and AFL 3.0 for GEREDB, as
declared in the OmniPath resource metadata. All of them permit redistribution with
attribution. The CollecTRI code repository itself is GPL-3.0.

- **Source**: <https://github.com/saezlab/CollecTRI> and <https://omnipathdb.org>
- **Reference**: Muller-Dott, S. et al. Expanding the coverage of regulons from public
  resources using selected gene set enrichment methods. *Nucleic Acids Research* (2023).
- **Changes made**: none to the edge values. The file is a snapshot of 42,990 edges
  taken through decoupleR in March 2026 and stored so that results do not drift with
  the live resource.

The OmniPath protein-protein interaction cache is **not** redistributed here, because
its constituent resources have heterogeneous licences that do not all permit
redistribution. The package rebuilds it instead.

---

## 4. Expression data, downloaded rather than distributed here

The cell-line expression matrix is not shipped with this package, but the results are
derived from it, so its licence is recorded here too.

**Licence: CC BY 4.0.** <https://creativecommons.org/licenses/by/4.0/legalcode.en>

- **Title**: DepMap 24Q4 Public
- **Creator**: Broad DepMap
- **Source**: figshare deposit,
  <https://doi.org/10.25452/figshare.plus.27993248.v1>
- **File**: `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, 19,193 protein-coding genes

Note that expression and cell-line annotations come from two different releases, and
therefore from two different licences: expression from 24Q4 on figshare under CC BY 4.0,
annotations from 25Q3 under the depmap.org terms. The 19,193 genes of the processed
matrices match the 24Q4 column count exactly, whereas 25Q3 has 19,220.

---

## Derived results

The CSVs under `results/` and `paper/results/` are per-drug and per-TF summary
statistics, not copies of the source data. The trained checkpoints,
`cell_line_embeddings.csv` and the Integrated Gradients tables hold model weights,
embeddings and attribution scores rather than any source record.
