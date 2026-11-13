#!/usr/bin/env python3
"""Build the processed FUSCC-TNBC inputs that the pipeline reads, from the source files.

The source files are not distributed with the package. Download them from their
sources (below) into data/FUSCC/raw/, the default input directory, or pass the
directory that holds them with --raw-dir. Everyone who only reruns the analysis
starts from the processed files this script writes, as with METABRIC and SCAN-B.

Sources (the FUSCC triple-negative breast cancer cohort of 465 patients, Jiang et
al., Cancer Cell 2019, doi 10.1016/j.ccell.2019.02.001):
  FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv   56,858 genes x 360 samples, FPKM. From
      the figshare deposit of the data descriptor (Chen et al., Sci Data 2022, doi
      10.1038/s41597-022-01681-z), doi 10.6084/m9.figshare.19783498.v5, CC BY 4.0
  mmc2.xlsx, sheet "Table S1"                465 patients x 50 clinical columns.
      Table S1 of Jiang et al. 2019, under the Elsevier user license, which does
      not permit redistributing it

Both are checked before anything is written: the md5 of the expression matrix, the
checksum figshare publishes, and a SHA-256 of the columns of Table S1 that the
analysis uses (CLINICAL_COLUMNS), as read and serialized canonically with floats to
10 significant digits (table_content_sha256). The xlsx gets a content hash and not
a file hash because cloud sync clients can rewrite the internal parts of Office
files, which changes their bytes and leaves every cell as it was. The hash
leaves out the unused columns and the last digits of the floats because saving the
workbook again, even without editing it, rewrites at least one float of an unused
column (Ascat_Ploidy). It was frozen with pandas 2.2.3 and openpyxl 3.1.5, and
another version that reads or writes the sheet differently could make it fail
without any cell having changed.

Writes to data/FUSCC/ (the shared junction):
  fuscc_expression_fpkm.csv.gz   genes x samples, FPKM, duplicated symbols resolved
  fuscc_clinical.csv             the clinical table, one row per patient
  fuscc_sample_ids.txt           the sample identifiers of the matrix, in file order
  README.md                      provenance, licences, checksums and how to rebuild

Two quirks of the raw files are handled here and nowhere else:
  1. The header of the expression matrix separates its first two fields with a space
     instead of a tab, so it carries 360 names for 361 fields per data row. The
     script rebuilds the header explicitly and raises if the counts do not match.
  2. Gene symbols repeat. The row with the highest mean FPKM is kept for each
     symbol, the number dropped is recorded in the README, and nothing is dropped
     silently.
The Sex column of Table S1 carries a trailing space in some rows. It is stripped,
and the script raises unless every patient is recorded as female.

Run from the project root:
    python scripts/00c_prepare_fuscc.py [--raw-dir DIR]
"""

import argparse
import hashlib
import logging
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_fuscc")

# Default input directory: the FPKM matrix of the figshare deposit and the Table S1 of
# Jiang et al. 2019 (mmc2.xlsx), downloaded by hand (see data/README.md, FUSCC-TNBC)
RAW = DATA_DIR / "FUSCC" / "raw"
EXPR_NAME = "FUSCCTNBC_Expression_RNAseqFPKM_hg38.tsv"
CLIN_NAME = "mmc2.xlsx"
CLIN_SHEET = "Table S1"
OUT_DIR = DATA_DIR / "FUSCC"

FIGSHARE_DOI = "10.6084/m9.figshare.19783498.v5"
DESCRIPTOR_DOI = "10.1038/s41597-022-01681-z"
COHORT_DOI = "10.1016/j.ccell.2019.02.001"
# Public deposits of the cohort, as the Data Records of the data descriptor (Chen et al.
# 2022) list them. Nothing here reads them, the README records them for provenance.
NODE_ACCESSION = "OEP000155"   # all the omics data of the cohort and their metadata
GEO_ACCESSION = "GSE118527"    # OncoScan copy-number arrays
SRA_ACCESSION = "SRP157974"    # sequencing reads, whole exome and RNA-seq

EXPECTED_GENES_RAW = 56858
EXPECTED_SAMPLES = 360
EXPECTED_PATIENTS = 465
EXPECTED_RNA_PATIENTS = 360
EXPECTED_EVENTS_RNA = 49
# md5 of the expression matrix, the checksum of the figshare deposit
EXPECTED_EXPR_SRC_MD5 = "44e173a7b74fe71738d245a9ffe09da0"
# SHA-256 of the CLINICAL_COLUMNS of Table S1 as read by read_clinical, see
# table_content_sha256, frozen with the versions of HASH_FROZEN_WITH
EXPECTED_CLIN_SRC_SHA256 = "96ab7091b1bbdcc7ae10d9926ff4f2f4c998a434e3372e2b459493a30ef19646"
HASH_FROZEN_WITH = "pandas 2.2.3 and openpyxl 3.1.5"
HASH_FLOAT_FORMAT = "%.10g"
EXPECTED_SEX = "Female"

CLINICAL_COLUMNS = [
    "Project_ID", "RNA_Seqencing", "mRNA_Subtype", "Intrinsic_Subtype", "Age_at_surgery",
    "Histology", "Histology_extended", "Grade", "T", "N", "Size_cm", "LN_examed",
    "LN_positive", "Ki67", "sTILs", "iTILs", "Followup_Month", "RFS_Status",
    # Days are the unit the analysis uses. Months are kept only so the README can
    # show that they are a rounded version of the same quantity.
    "RFS_time_Days", "RFS_time_Months", "Date_of_surgery",
    "Chemotherapy", "Paclitaxel", "Radiotherapy", "No_Chemotherapy",
    "BRCA1/2 MUT 20160718", "ER_IHC_score", "PR_IHC_score", "ERBB2_IHC_score", "ERBB2_FISH",
    # So that Methods can state the sex of the cohort from the data
    "Sex",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table_s1(path: Path) -> pd.DataFrame:
    """Sheet Table S1 of mmc2.xlsx, all 50 columns. Row 1 is the title, so header = 1."""
    return pd.read_excel(path, sheet_name=CLIN_SHEET, header=1)


def table_content_sha256(table: pd.DataFrame) -> str:
    """SHA-256 of a table serialized canonically: CSV, UTF-8, no index, LF line ends,
    floats to 10 significant digits.

    It changes when a cell changes and not when a sync client rewrites the bytes of the
    xlsx that holds it, nor when saving it again moves the last digits of a float."""
    text = table.to_csv(index=False, lineterminator="\n", float_format=HASH_FLOAT_FORMAT)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_expression(expr_src: Path) -> tuple[pd.DataFrame, int]:
    """FPKM matrix with the header quirk fixed and duplicated gene symbols resolved."""
    with open(expr_src, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        first_row_fields = len(fh.readline().rstrip("\n").split("\t"))
    first = header[0].split()
    if len(first) != 2 or first[0] != "gene_symbol":
        raise ValueError(f"Unexpected first header field: {header[0]!r}")
    names = [first[0], first[1]] + header[1:]
    if len(names) != first_row_fields:
        raise ValueError(f"The header rebuilds to {len(names)} fields and a data row has "
                         f"{first_row_fields}")
    expr = pd.read_csv(expr_src, sep="\t", skiprows=1, header=None, names=names, index_col=0)
    if expr.shape != (EXPECTED_GENES_RAW, EXPECTED_SAMPLES):
        raise ValueError(f"Expression is {expr.shape}, expected "
                         f"{(EXPECTED_GENES_RAW, EXPECTED_SAMPLES)}")
    if expr.isna().any().any():
        raise ValueError("The FPKM matrix has missing values")
    if float(expr.to_numpy().min()) < 0:
        raise ValueError("Negative FPKM values")
    if float(expr.to_numpy().max()) < 100:
        raise ValueError("The matrix does not look like linear FPKM")
    duplicated = int(expr.index.duplicated().sum())
    if duplicated:
        order = expr.mean(axis=1).sort_values(ascending=False).index
        expr = expr.loc[order]
        expr = expr[~expr.index.duplicated(keep="first")].sort_index()
    logger.info("Expression: %d genes x %d samples, %d duplicated symbols resolved by the "
                "highest mean", expr.shape[0], expr.shape[1], duplicated)
    return expr, duplicated


def read_clinical(clin_src: Path, sample_ids) -> tuple[pd.DataFrame, str]:
    """Table S1 of the paper, checked against the samples of the expression matrix.

    Returns the columns the analysis uses and the content hash of those columns."""
    clin = read_table_s1(clin_src)
    missing = [c for c in CLINICAL_COLUMNS if c not in clin.columns]
    if missing:
        raise ValueError(f"Clinical columns missing: {missing}")
    clin = clin[CLINICAL_COLUMNS]
    content_sha256 = table_content_sha256(clin)
    if content_sha256 != EXPECTED_CLIN_SRC_SHA256:
        raise ValueError(f"The columns of {CLIN_SHEET} that the analysis uses hash to {content_sha256}, "
                         f"expected {EXPECTED_CLIN_SRC_SHA256} (frozen with {HASH_FROZEN_WITH}, "
                         f"running pandas {pd.__version__} and openpyxl {version('openpyxl')})")
    if len(clin) != EXPECTED_PATIENTS:
        raise ValueError(f"The clinical table has {len(clin)} rows, expected {EXPECTED_PATIENTS}")
    clin = clin.rename(columns={"BRCA1/2 MUT 20160718": "BRCA12_mutation"})
    # Some rows carry a trailing space ("Female "), which is not information
    clin["Sex"] = clin["Sex"].astype(str).str.strip()
    n_expected_sex = int((clin["Sex"] == EXPECTED_SEX).sum())
    if n_expected_sex != EXPECTED_PATIENTS:
        raise ValueError(f"{n_expected_sex} of {len(clin)} patients recorded as {EXPECTED_SEX}, "
                         f"expected all {EXPECTED_PATIENTS}: "
                         f"{clin['Sex'].value_counts(dropna=False).to_dict()}")
    clin = clin.set_index("Project_ID")
    with_rna = clin.index[clin["RNA_Seqencing"].fillna("NO").str.upper() == "YES"]
    if len(with_rna) != EXPECTED_RNA_PATIENTS:
        raise ValueError(f"{len(with_rna)} patients flagged with RNA-seq, expected "
                         f"{EXPECTED_RNA_PATIENTS}")
    only_matrix = sorted(set(sample_ids) - set(clin.index))
    only_flag = sorted(set(with_rna) - set(sample_ids))
    if only_matrix or only_flag:
        raise ValueError(f"The expression matrix and the RNA-seq flag disagree: "
                         f"{len(only_matrix)} samples in the matrix only, {len(only_flag)} "
                         "patients flagged only (counts, identifiers never leave this script)")
    events = int(clin.loc[with_rna, "RFS_Status"].sum())
    if events != EXPECTED_EVENTS_RNA:
        raise ValueError(f"{events} relapse events among the sequenced patients, expected "
                         f"{EXPECTED_EVENTS_RNA}")
    if clin.loc[with_rna, ["RFS_Status", "RFS_time_Months"]].isna().any().any():
        raise ValueError("A sequenced patient has no relapse-free survival")
    logger.info("Clinical: %d patients, %d with RNA-seq, %d relapse events among them, all %d "
                "recorded as %s", len(clin), len(with_rna), events, n_expected_sex, EXPECTED_SEX)
    return clin, content_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW,
                        help=f"directory holding {EXPR_NAME} and {CLIN_NAME} "
                             "(default: data/FUSCC/raw/)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    expr_src, clin_src = args.raw_dir / EXPR_NAME, args.raw_dir / CLIN_NAME
    for path in (expr_src, clin_src):
        if not path.exists():
            raise SystemExit(f"Source file not found: {path}\n"
                             f"Download {EXPR_NAME} from figshare (doi {FIGSHARE_DOI}) and "
                             f"{CLIN_NAME} (Table S1 of doi {COHORT_DOI}) and pass their "
                             "directory with --raw-dir. Rerunning the analysis does not need "
                             "this step: it reads the processed files of data/FUSCC/.")
    expr_src_md5 = md5(expr_src)
    if expr_src_md5 != EXPECTED_EXPR_SRC_MD5:
        raise ValueError(f"{EXPR_NAME} has md5 {expr_src_md5}, expected the figshare checksum "
                         f"{EXPECTED_EXPR_SRC_MD5}")
    expr, duplicated = read_expression(expr_src)
    clin, clin_src_sha256 = read_clinical(clin_src, expr.columns)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr_out = OUT_DIR / "fuscc_expression_fpkm.csv.gz"
    clin_out = OUT_DIR / "fuscc_clinical.csv"
    ids_out = OUT_DIR / "fuscc_sample_ids.txt"
    expr.to_csv(expr_out, compression={"method": "gzip", "mtime": 0})
    clin.to_csv(clin_out)
    # The sample identifiers on their own, so the header repair can be checked
    # without reading the 47 MB matrix
    ids_out.write_text("\n".join(expr.columns) + "\n", encoding="utf-8")

    with_rna = clin.index[clin["RNA_Seqencing"].fillna("NO").str.upper() == "YES"]
    n_zero_time = int((clin.loc[with_rna, "RFS_time_Days"] <= 0).sum())
    day_month = (clin.loc[with_rna, "RFS_time_Days"] /
                 clin.loc[with_rna, "RFS_time_Months"].replace(0, np.nan))

    n_sex = int((clin["Sex"] == EXPECTED_SEX).sum())

    readme = f"""# FUSCC-TNBC (processed subset)

Triple-negative breast cancer cohort of the Fudan University Shanghai Cancer Center,
465 patients, published by Jiang et al., Cancer Cell 2019 (doi {COHORT_DOI}).
These files are the processed subset the pipeline reads. The source files are not
redistributed and are not needed to reproduce any number of the paper.

## Sources and licences

- Expression: `{EXPR_NAME}`, the FPKM matrix (hg38) of the figshare deposit of the
  data descriptor of the cohort (Chen et al., Sci Data 2022, doi {DESCRIPTOR_DOI}),
  doi {FIGSHARE_DOI}, under CC BY 4.0. Its md5 below is the checksum figshare
  publishes for the file.
- Clinical: sheet `{CLIN_SHEET}` of `{CLIN_NAME}`, Table S1 of Jiang et al. 2019, under
  the Elsevier user license, which does not permit redistributing it.

The clinical table therefore stays in the shared project drive and out of the public
code package. That package can carry this script, with which anyone who downloads
both files from their sources rebuilds the cohort, and the per-drug results of script
69, which contain no patient-level data.

The Data Records of the data descriptor list three more public deposits of the cohort.
The pipeline reads none of them.

- All the omics data of the cohort and their metadata: National Omics Data
  Encyclopedia (NODE), `{NODE_ACCESSION}`.
- OncoScan copy-number arrays: Gene Expression Omnibus, `{GEO_ACCESSION}`.
- Sequencing reads, whole exome and RNA-seq: Sequence Read Archive, `{SRA_ACCESSION}`.

| File | Content | md5 | Read by |
|---|---|---|---|
| `fuscc_expression_fpkm.csv.gz` | FPKM, {expr.shape[0]:,} gene symbols x {expr.shape[1]} samples, linear scale. The log2 is applied by script 68, as with SCAN-B | `{md5(expr_out)}` | 68 |
| `fuscc_clinical.csv` | Table S1 of the paper, {len(clin)} patients, the columns the analysis uses | `{md5(clin_out)}` | 68 |
| `fuscc_sample_ids.txt` | the {expr.shape[1]} sample identifiers in file order | `{md5(ids_out)}` | checks |

Built by `scripts/00c_prepare_fuscc.py` from:

| Source file | Checksum |
|---|---|
| `{EXPR_NAME}` | md5 `{expr_src_md5}` |
| `{CLIN_NAME}` (sheet {CLIN_SHEET}) | SHA-256 of the columns used `{clin_src_sha256}` |

The checksum of `{CLIN_NAME}` is a SHA-256 of the {len(CLINICAL_COLUMNS)} columns of the
sheet that the analysis uses, as read with `header = 1` and written as CSV (UTF-8, no
index, LF line ends, floats to 10 significant digits), not a hash of the file. Cloud
sync clients can rewrite the internal parts of Office files, which changes the
bytes of the xlsx and leaves every cell as it was, and saving the workbook again
rewrites at least one float of an unused column. The checksum was frozen with
{HASH_FROZEN_WITH}. The script checks both checksums before writing anything.

Rebuild with `python scripts/00c_prepare_fuscc.py`, which reads data/FUSCC/raw/, or
with `python scripts/00c_prepare_fuscc.py --raw-dir DIR`, where `DIR` holds the two
source files.

## Quirks of the raw files, resolved once here

1. The header of the expression matrix separates its first two fields with a space
   instead of a tab, so it carries {expr.shape[1]} names for {expr.shape[1] + 1} fields per
   data row. Reading it with the default parser silently renames the first sample.
   The script rebuilds the header and raises if the counts disagree. In Python that is
   `names = header[0].split() + header[1:]` with `skiprows=1, header=None`, and in R the
   equivalent is reading with `data.table::fread` on the file this script writes, whose
   header is already correct.
2. Sheet `Table S1` of `mmc2.xlsx` needs `header = 1`: row 1 is the table title.
3. {duplicated} gene symbols appear more than once. The row with the highest mean FPKM is
   kept for each symbol, leaving {expr.shape[0]:,} unique symbols.
4. `RFS_time_Months` is a rounded version of `RFS_time_Days` (the ratio ranges from
   {float(day_month.min()):.2f} to {float(day_month.max()):.2f} days per month), so the
   analysis uses days only.
5. {n_zero_time} sequenced patients have `RFS_time_Days = 0`. They are kept here and
   excluded by the analysis, which counts them by reason.
6. `Sex` carries a trailing space in some rows ("Female "). It is stripped here.

## Checked at build time

{len(clin)} patients, {int((clin["RNA_Seqencing"].fillna("NO").str.upper() == "YES").sum())} with
RNA-seq, {int(clin.loc[clin["RNA_Seqencing"].fillna("NO").str.upper() == "YES", "RFS_Status"].sum())}
relapse events among them, and the sample identifiers of the matrix equal the patients
flagged with RNA-seq. {n_sex} of {len(clin)} patients are recorded as {EXPECTED_SEX.lower()}.
Scale: FPKM, no missing values, minimum {float(expr.to_numpy().min()):.3f} and maximum
{float(expr.to_numpy().max()):,.1f}.

## Reference effect of the power analysis (script 69)

The power of script 69 is stated against the hazard ratios per standard deviation of
pelitinib and osimertinib in the TNBC subgroup of SCAN-B (script 61), rescaled from the
spread of the predicted AUC in the full SCAN-B Cox cohort to the spread inside the
subgroup. The rescaling uses each drug's own ratio of standard deviations, which gives
1.465 for pelitinib and 1.24 for osimertinib. When the design was reviewed it was
computed with the median ratio over the 12 focal drugs (1.056), which gave 1.46 and
1.26.

## What is not here

TPM, raw counts, RUVg-normalized counts, the GISTIC copy-number table, the mutation MAF
and supplementary tables mmc3 to mmc10 of the source study are not used. Nothing in the
pipeline reads them.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    logger.info("Wrote %s (%.1f MB), %s and %s", expr_out, expr_out.stat().st_size / 1e6,
                clin_out, ids_out)


if __name__ == "__main__":
    sys.exit(main())

