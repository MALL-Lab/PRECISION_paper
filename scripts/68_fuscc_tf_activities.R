#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: TF activities and clinical table for the FUSCC-TNBC cohort
#
# Same route as SCAN-B (script 09) and METABRIC / TCGA (script 20): log2 of the
# linear expression, then ULM of decoupleR against the CollecTRI snapshot shared
# with the graph (load_collectri_snapshot of R/config.R), minsize 5. Using the snapshot and not
# a live download is what keeps this cohort out of the drift documented in
# Supplementary Note 2.
#
# One thing is done here that the earlier cohorts did not need. The ULM statistic
# grows with the number of genes in the matrix, and the clinical transfer applies a
# QuantileTransformer fitted on the PRISM activities, so a cohort computed over a
# much larger gene universe lands outside the training range and its predictions
# are flattened by clipping. Measured: with the full FUSCC transcriptome (55,657
# genes) 71.4% of the activities fall outside that range, against 12.7% for SCAN-B
# and 12.3% for METABRIC. Restricting the matrix to the genes of the PRISM
# expression matrix (18,822 of them are in FUSCC) brings it to 13.9% and leaves the
# activities correlated at 0.976 with the unrestricted ones, so the scale changes
# and the biology does not. The same measurement puts TCGA-BRCA at 80.3%, which is
# a property of a cohort already in the paper and is reported separately.
#
# Inputs (see data/FUSCC/README.md, written by scripts/00c_prepare_fuscc.py):
#   data/FUSCC/fuscc_expression_fpkm.csv.gz   56,858 genes x 360 samples, FPKM
#   data/FUSCC/fuscc_clinical.csv             465 patients, Table S1 of the paper
#
# Produces:
#   data/precision_processed/tf_activities_fuscc.csv   360 samples x TFs
#   data/precision_processed/fuscc_clinical.csv        the 360 sequenced patients,
#                                                      column names normalized, with
#                                                      their sex (all female, checked
#                                                      by 00c and again here)
#
# The script stops instead of writing if any of these fails: the matrix is not
# 56,858 x 360, the scale is not linear FPKM, a gene is not finite, the clinical
# table does not hold 465 patients with 360 sequenced and 49 relapse events, a
# sequenced patient is not recorded as female, or the samples of the matrix are not
# exactly the sequenced patients.
#
# Run from the project root:
#   Rscript scripts/68_fuscc_tf_activities.R
# ==============================================================================

rm(list = ls())

.cfg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                  "..", "R", "config.R")
source(if (file.exists(.cfg)) .cfg else "R/config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(decoupleR)
})

EXPR_PATH <- file.path(DATA_DIR, "FUSCC/fuscc_expression_fpkm.csv.gz")
PRISM_EXPR_PATH <- file.path(DATA_DIR, "PRISM/processed/counts_matched_prism.csv")
CLIN_PATH <- file.path(DATA_DIR, "FUSCC/fuscc_clinical.csv")
OUTPUT_DIR <- file.path(DATA_DIR, "precision_processed")

EXPECTED_GENES <- 56858
EXPECTED_SAMPLES <- 360
EXPECTED_PATIENTS <- 465
EXPECTED_EVENTS <- 49
EXPECTED_PRISM_GENES <- 19193
EXPECTED_SHARED_GENES <- 18822
EXPECTED_ZERO_VARIANCE <- 53
EXPECTED_GENES_USED <- 18769
EXPECTED_TFS <- 769
# The scale checks below also pass for the TPM and the count matrices of the same
# deposit, so the md5 is what identifies the file.
EXPECTED_EXPR_MD5 <- "329f9adae644472e09cb3b848049f300"
# The clinical table gained the Sex column of Table S1 on 30-oct-2026, nothing else
# changed in it (the earlier md5 was 1f5e26322646cc2ff2cf5ccf7a115539)
EXPECTED_CLIN_MD5 <- "c8d781ff61a2c386426a4ad58c458fe1"
EXPECTED_FPKM_MAX <- 745432.3
EXPECTED_LOG2_MAX <- 19.5077  # of the full matrix, checked before restricting

cat("============================================================\n")
cat("PRECISION: FUSCC-TNBC TF activities\n")
cat("============================================================\n\n")

# ------------------------------------------------------------------ expression
cat("--- Expression ---\n")
expr_md5 <- unname(tools::md5sum(EXPR_PATH))
clin_md5 <- unname(tools::md5sum(CLIN_PATH))
if (expr_md5 != EXPECTED_EXPR_MD5) {
  stop(sprintf("Expression md5 is %s, expected %s: rebuild with scripts/00c_prepare_fuscc.py",
               expr_md5, EXPECTED_EXPR_MD5))
}
if (clin_md5 != EXPECTED_CLIN_MD5) {
  stop(sprintf("Clinical md5 is %s, expected %s", clin_md5, EXPECTED_CLIN_MD5))
}
expr_dt <- data.table::fread(EXPR_PATH)
gene_col <- names(expr_dt)[1]
expr_mat <- as.matrix(expr_dt[, -1, with = FALSE])
rownames(expr_mat) <- expr_dt[[gene_col]]
cat("  Loaded:", nrow(expr_mat), "genes x", ncol(expr_mat), "samples\n")

if (nrow(expr_mat) != EXPECTED_GENES || ncol(expr_mat) != EXPECTED_SAMPLES) {
  stop(sprintf("Expression is %d x %d, expected %d x %d",
               nrow(expr_mat), ncol(expr_mat), EXPECTED_GENES, EXPECTED_SAMPLES))
}
if (any(!is.finite(expr_mat))) stop("The expression matrix has non-finite values")
if (min(expr_mat) < 0) stop("Negative FPKM values")
if (abs(max(expr_mat) - EXPECTED_FPKM_MAX) > 0.1) {
  stop(sprintf("Maximum FPKM is %.1f, expected %.1f", max(expr_mat), EXPECTED_FPKM_MAX))
}
if (any(duplicated(rownames(expr_mat)))) stop("Duplicated gene symbols reached the R step")

# FUSCC expression is FPKM (linear), as SCAN-B. log2(x+1) for ULM.
expr_mat <- log2(expr_mat + 1)
if (abs(max(expr_mat) - EXPECTED_LOG2_MAX) > 1e-3) {
  stop(sprintf("Maximum of the log2 matrix is %.4f, expected %.4f",
               max(expr_mat), EXPECTED_LOG2_MAX))
}

# Restrict to the gene universe of the PRISM expression matrix, the one the
# transfer was trained on. See the note in the header for why.
prism_genes <- names(data.table::fread(PRISM_EXPR_PATH, nrows = 1))[-1]
if (length(prism_genes) != EXPECTED_PRISM_GENES) {
  stop(sprintf("The PRISM matrix has %d genes, expected %d",
               length(prism_genes), EXPECTED_PRISM_GENES))
}
shared_genes <- intersect(rownames(expr_mat), prism_genes)
if (length(shared_genes) != EXPECTED_SHARED_GENES) {
  stop(sprintf("%d genes shared with PRISM, expected %d",
               length(shared_genes), EXPECTED_SHARED_GENES))
}
cat("  Restricting to the PRISM gene universe:", length(shared_genes), "of",
    nrow(expr_mat), "genes\n")
expr_mat <- expr_mat[shared_genes, ]

# Genes with no variation carry no information for the linear model and only
# slow it down. They are counted, never dropped silently.
keep <- apply(expr_mat, 1, function(x) stats::sd(x) > 0)
if (sum(!keep) != EXPECTED_ZERO_VARIANCE) {
  stop(sprintf("%d genes without variance, expected %d", sum(!keep), EXPECTED_ZERO_VARIANCE))
}
cat("  Dropping", sum(!keep), "genes with zero variance across the cohort\n")
expr_mat <- expr_mat[keep, ]
if (nrow(expr_mat) != EXPECTED_GENES_USED) {
  stop(sprintf("%d genes reach ULM, expected %d", nrow(expr_mat), EXPECTED_GENES_USED))
}
cat("  After log2:", nrow(expr_mat), "genes x", ncol(expr_mat), "samples,",
    "range [", round(min(expr_mat), 3), ",", round(max(expr_mat), 3), "]\n\n")

# ------------------------------------------------------------------- clinical
cat("--- Clinical ---\n")
clin <- read.csv(CLIN_PATH, stringsAsFactors = FALSE, check.names = FALSE)
if (nrow(clin) != EXPECTED_PATIENTS) {
  stop(sprintf("The clinical table has %d rows, expected %d", nrow(clin), EXPECTED_PATIENTS))
}
clin <- clin %>% filter(toupper(RNA_Seqencing) == "YES")
if (nrow(clin) != EXPECTED_SAMPLES) {
  stop(sprintf("%d sequenced patients, expected %d", nrow(clin), EXPECTED_SAMPLES))
}
if (sum(clin$RFS_Status) != EXPECTED_EVENTS) {
  stop(sprintf("%d relapse events, expected %d", sum(clin$RFS_Status), EXPECTED_EVENTS))
}
if (!setequal(clin$Project_ID, colnames(expr_mat))) {
  stop("The samples of the matrix are not the sequenced patients of the clinical table")
}

# Days are the only time unit that leaves this script. Months and the follow-up
# column are deliberately absent so that no downstream analysis can use the
# rounded version by accident.
clinical_out <- data.frame(
  sample_id       = clin$Project_ID,
  rfs_event       = as.integer(clin$RFS_Status),
  rfs_time_days   = as.numeric(clin$RFS_time_Days),
  subtype_mrna    = clin$mRNA_Subtype,
  subtype_pam50   = clin$Intrinsic_Subtype,
  age             = as.numeric(clin$Age_at_surgery),
  t_stage         = clin$T,
  n_stage         = clin$N,
  grade           = clin$Grade,
  size_cm         = as.numeric(clin$Size_cm),
  ln_positive     = as.numeric(clin$LN_positive),
  ki67            = as.numeric(clin$Ki67),
  stils           = as.numeric(clin$sTILs),
  chemo           = clin$Chemotherapy,
  paclitaxel      = clin$Paclitaxel,
  radiotherapy    = clin$Radiotherapy,
  brca12_mut      = clin$BRCA12_mutation,
  histology       = clin$Histology,
  histology_ext   = clin$Histology_extended,
  erbb2_ihc       = clin$ERBB2_IHC_score,
  erbb2_fish      = clin$ERBB2_FISH,
  surgery_year    = as.integer(format(as.Date(clin$Date_of_surgery), "%Y")),
  sex             = clin$Sex,
  stringsAsFactors = FALSE
)
if (any(is.na(clinical_out$rfs_time_days)) || any(is.na(clinical_out$rfs_event))) {
  stop("A sequenced patient has no relapse-free survival in days")
}
if (any(is.na(clinical_out$surgery_year))) stop("A surgery date did not parse")
if (!identical(sort(unique(clinical_out$sex)), "Female")) {
  stop("Every sequenced patient must be recorded as Female (00c strips the trailing space)")
}
cat("  Sequenced patients:", nrow(clinical_out), "with", sum(clinical_out$rfs_event),
    "relapse events, median observed time",
    round(median(clinical_out$rfs_time_days) / 30.4375, 1), "months\n")
cat("  mRNA subtypes:", paste(sprintf("%s %d", names(table(clinical_out$subtype_mrna)),
                                      table(clinical_out$subtype_mrna)), collapse = ", "), "\n\n")

# --------------------------------------------------------------- TF activities
cat("--- TF activities via decoupleR ULM ---\n")
collectri <- load_collectri_snapshot()

tf_acts_raw <- run_ulm(
  mat = expr_mat,
  net = collectri,
  .source = "source",
  .target = "target",
  .mor = "mor",
  minsize = 5
)

tf_wide <- tf_acts_raw %>%
  filter(statistic == "ulm") %>%
  dplyr::select(source, condition, score) %>%
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")

cat("  TF activities:", nrow(tf_wide), "samples x", ncol(tf_wide), "TFs\n")
if (ncol(tf_wide) != EXPECTED_TFS) {
  stop(sprintf("ULM returned %d TFs, expected %d", ncol(tf_wide), EXPECTED_TFS))
}
if (nrow(tf_wide) != EXPECTED_SAMPLES) {
  stop(sprintf("ULM returned %d samples, expected %d", nrow(tf_wide), EXPECTED_SAMPLES))
}
if (any(!is.finite(as.matrix(tf_wide)))) stop("Non-finite TF activities")
tf_wide <- tf_wide[clinical_out$sample_id, ]

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(tf_wide, file.path(OUTPUT_DIR, "tf_activities_fuscc.csv"))
write.csv(clinical_out, file.path(OUTPUT_DIR, "fuscc_clinical.csv"), row.names = FALSE)

r_summary <- data.frame(
  metric = c("r_version", "decoupleR_version", "data_table_version", "expression_md5",
             "clinical_md5", "n_genes_input", "n_zero_variance_dropped", "n_genes_used",
             "n_samples", "n_tfs", "n_collectri_edges", "n_collectri_tfs",
             "ulm_minsize", "fpkm_max", "log2_max", "n_prism_genes", "n_shared_genes"),
  value = c(R.version.string, as.character(packageVersion("decoupleR")),
            as.character(packageVersion("data.table")), expr_md5, clin_md5,
            EXPECTED_GENES, sum(!keep), nrow(expr_mat), ncol(expr_mat), ncol(tf_wide),
            nrow(collectri), length(unique(collectri$source)), 5,
            sprintf("%.1f", EXPECTED_FPKM_MAX), sprintf("%.4f", max(expr_mat)),
            length(prism_genes), length(shared_genes)),
  stringsAsFactors = FALSE
)
write.csv(r_summary, file.path(OUTPUT_DIR, "fuscc_r_summary.csv"), row.names = FALSE)

cat("\nWrote", file.path(OUTPUT_DIR, "tf_activities_fuscc.csv"), ",",
    file.path(OUTPUT_DIR, "fuscc_clinical.csv"), "and",
    file.path(OUTPUT_DIR, "fuscc_r_summary.csv"), "\n")
