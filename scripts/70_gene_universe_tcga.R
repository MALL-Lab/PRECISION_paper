#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: TCGA-BRCA TF activities recomputed over the gene universe of PRISM
#
# Diagnostic, not a replacement. The published TCGA activities
# (data/precision_processed/tf_activities_tcga.csv, script 20) were computed from
# the GDC SummarizedExperiment, whose gene set is far larger than the one behind
# the PRISM activities. The ULM statistic grows with the number of genes, so those
# activities sit on a different scale (mean 4.58, sd 5.40 against -0.04 and 1.88 in
# PRISM) and the QuantileTransformer of the clinical transfer clips 80.3% of them.
# This script recomputes them over the genes shared with PRISM so that
# scripts/71_gene_universe_check.py can measure what that costs.
#
# It also computes the FUSCC-TNBC activities the other way round, over the full
# transcriptome instead of the PRISM universe that script 68 uses, so that script
# 71 can measure the clipping that made script 68 restrict the matrix (Supplementary
# Note 2). Same route as script 68 (md5 of the FPKM matrix, log2, genes without
# variance dropped, ULM against the snapshot, minsize 5) without the restriction.
#
# Nothing here overwrites a canonical input: the output goes to
# results/v15_gene_universe/, and the published activities stay as they are.
#
# Input:
#   data/TCGA/tcga_brca_log2tpm_reference.rds   17,351 genes x 1,231 samples, log2 TPM
#   data/PRISM/processed/counts_matched_prism.csv  the gene universe of the models
#   data/FUSCC/fuscc_expression_fpkm.csv.gz     56,858 genes x 360 samples, FPKM
#
# Output:
#   results/v15_gene_universe/tf_activities_tcga_prism_universe.csv
#   results/v15_gene_universe/tf_activities_fuscc_full_transcriptome.csv
#   results/v15_gene_universe/gene_universe_r_summary.csv
#
# Run from the project root:
#   Rscript scripts/70_gene_universe_tcga.R
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

EXPR_PATH <- file.path(DATA_DIR, "TCGA/tcga_brca_log2tpm_reference.rds")
PRISM_EXPR_PATH <- file.path(DATA_DIR, "PRISM/processed/counts_matched_prism.csv")
OUT_DIR <- file.path(RESULTS_DIR, "v15_gene_universe")

EXPECTED_TCGA_GENES <- 17351
EXPECTED_TCGA_SAMPLES <- 1231
EXPECTED_PRISM_GENES <- 19193
EXPECTED_SHARED_GENES <- 17170
EXPECTED_TFS <- 749

FUSCC_EXPR_PATH <- file.path(DATA_DIR, "FUSCC/fuscc_expression_fpkm.csv.gz")
EXPECTED_FUSCC_MD5 <- "329f9adae644472e09cb3b848049f300"  # the matrix script 68 reads
EXPECTED_FUSCC_GENES <- 56858
EXPECTED_FUSCC_SAMPLES <- 360
EXPECTED_FUSCC_ZERO_VARIANCE <- 1201
EXPECTED_FUSCC_GENES_USED <- 55657
EXPECTED_FUSCC_TFS <- 770

cat("============================================================\n")
cat("PRECISION: TCGA TF activities over the PRISM gene universe\n")
cat("============================================================\n\n")

expr <- readRDS(EXPR_PATH)
if (inherits(expr, "SummarizedExperiment")) {
  suppressPackageStartupMessages(library(SummarizedExperiment))
  expr <- assay(expr)
}
expr <- as.matrix(expr)
cat("  Reference matrix:", nrow(expr), "genes x", ncol(expr), "samples, range [",
    round(min(expr, na.rm = TRUE), 2), ",", round(max(expr, na.rm = TRUE), 2), "]\n")
if (nrow(expr) != EXPECTED_TCGA_GENES || ncol(expr) != EXPECTED_TCGA_SAMPLES) {
  stop(sprintf("The TCGA reference matrix is %d x %d, expected %d x %d",
               nrow(expr), ncol(expr), EXPECTED_TCGA_GENES, EXPECTED_TCGA_SAMPLES))
}
if (max(expr, na.rm = TRUE) > 25) stop("The TCGA matrix does not look like log2")

prism_genes <- names(data.table::fread(PRISM_EXPR_PATH, nrows = 1))[-1]
if (length(prism_genes) != EXPECTED_PRISM_GENES) {
  stop(sprintf("The PRISM matrix has %d genes, expected %d",
               length(prism_genes), EXPECTED_PRISM_GENES))
}
shared <- intersect(rownames(expr), prism_genes)
if (length(shared) != EXPECTED_SHARED_GENES) {
  stop(sprintf("%d genes shared with PRISM, expected %d", length(shared), EXPECTED_SHARED_GENES))
}
cat("  Restricting to", length(shared), "genes shared with PRISM\n")
mat <- expr[shared, ]
keep <- apply(mat, 1, function(x) stats::sd(x, na.rm = TRUE) > 0 & all(is.finite(x)))
cat("  Dropping", sum(!keep), "genes without variance or with non-finite values\n")
mat <- mat[keep, ]

collectri <- load_collectri_snapshot()
acts <- run_ulm(mat = mat, net = collectri, .source = "source", .target = "target",
                .mor = "mor", minsize = 5)
wide <- acts %>%
  filter(statistic == "ulm") %>%
  dplyr::select(source, condition, score) %>%
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")
cat("  TF activities:", nrow(wide), "samples x", ncol(wide), "TFs, mean",
    round(mean(as.matrix(wide)), 3), "sd", round(sd(as.matrix(wide)), 3), "\n")
if (ncol(wide) != EXPECTED_TFS) {
  stop(sprintf("ULM returned %d TFs, expected %d", ncol(wide), EXPECTED_TFS))
}

# ----------------------------------------- FUSCC over its full transcriptome
cat("\n--- FUSCC-TNBC over its full transcriptome ---\n")
fuscc_md5 <- unname(tools::md5sum(FUSCC_EXPR_PATH))
if (fuscc_md5 != EXPECTED_FUSCC_MD5) {
  stop(sprintf("FUSCC expression md5 is %s, expected %s", fuscc_md5, EXPECTED_FUSCC_MD5))
}
fuscc_dt <- data.table::fread(FUSCC_EXPR_PATH)
fuscc <- as.matrix(fuscc_dt[, -1, with = FALSE])
rownames(fuscc) <- fuscc_dt[[1]]
if (nrow(fuscc) != EXPECTED_FUSCC_GENES || ncol(fuscc) != EXPECTED_FUSCC_SAMPLES) {
  stop(sprintf("The FUSCC matrix is %d x %d, expected %d x %d", nrow(fuscc), ncol(fuscc),
               EXPECTED_FUSCC_GENES, EXPECTED_FUSCC_SAMPLES))
}
fuscc <- log2(fuscc + 1)
fuscc_keep <- apply(fuscc, 1, function(x) stats::sd(x) > 0)
if (sum(!fuscc_keep) != EXPECTED_FUSCC_ZERO_VARIANCE) {
  stop(sprintf("%d FUSCC genes without variance, expected %d", sum(!fuscc_keep),
               EXPECTED_FUSCC_ZERO_VARIANCE))
}
fuscc <- fuscc[fuscc_keep, ]
if (nrow(fuscc) != EXPECTED_FUSCC_GENES_USED) {
  stop(sprintf("%d FUSCC genes reach ULM, expected %d", nrow(fuscc), EXPECTED_FUSCC_GENES_USED))
}
cat("  Full transcriptome:", nrow(fuscc), "genes after dropping", sum(!fuscc_keep),
    "without variance\n")
fuscc_acts <- run_ulm(mat = fuscc, net = collectri, .source = "source", .target = "target",
                      .mor = "mor", minsize = 5)
fuscc_wide <- fuscc_acts %>%
  filter(statistic == "ulm") %>%
  dplyr::select(source, condition, score) %>%
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")
cat("  TF activities:", nrow(fuscc_wide), "samples x", ncol(fuscc_wide), "TFs\n")
if (ncol(fuscc_wide) != EXPECTED_FUSCC_TFS || nrow(fuscc_wide) != EXPECTED_FUSCC_SAMPLES) {
  stop(sprintf("ULM returned %d x %d for FUSCC, expected %d x %d", nrow(fuscc_wide),
               ncol(fuscc_wide), EXPECTED_FUSCC_SAMPLES, EXPECTED_FUSCC_TFS))
}
if (any(!is.finite(as.matrix(fuscc_wide)))) stop("Non-finite FUSCC TF activities")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(wide, file.path(OUT_DIR, "tf_activities_tcga_prism_universe.csv"))
write.csv(fuscc_wide, file.path(OUT_DIR, "tf_activities_fuscc_full_transcriptome.csv"))
summary_df <- data.frame(
  metric = c("r_version", "decoupleR_version", "tcga_reference_md5", "n_tcga_genes",
             "n_prism_genes", "n_shared_genes", "n_genes_used", "n_samples", "n_tfs",
             "activity_mean", "activity_sd", "fuscc_expression_md5", "fuscc_n_genes_input",
             "fuscc_n_zero_variance_dropped", "fuscc_n_genes_used", "fuscc_n_samples",
             "fuscc_n_tfs"),
  value = c(R.version.string, as.character(packageVersion("decoupleR")),
            unname(tools::md5sum(EXPR_PATH)), nrow(expr), length(prism_genes),
            length(shared), nrow(mat), ncol(mat), ncol(wide),
            sprintf("%.4f", mean(as.matrix(wide))), sprintf("%.4f", sd(as.matrix(wide))),
            fuscc_md5, EXPECTED_FUSCC_GENES, sum(!fuscc_keep), nrow(fuscc),
            nrow(fuscc_wide), ncol(fuscc_wide)),
  stringsAsFactors = FALSE
)
write.csv(summary_df, file.path(OUT_DIR, "gene_universe_r_summary.csv"), row.names = FALSE)
cat("\nWrote", file.path(OUT_DIR, "tf_activities_tcga_prism_universe.csv"), "and",
    file.path(OUT_DIR, "tf_activities_fuscc_full_transcriptome.csv"), "\n")
