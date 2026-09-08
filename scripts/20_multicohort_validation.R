#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: Compute TF activities for METABRIC and TCGA-BRCA
#
# Inputs (see data/README.md):
#   data/METABRIC/metabric_expression.rds   log-intensity matrix, genes x samples
#   data/METABRIC/metabric_gene_map.rds     hugoGeneSymbol to entrezGeneId
#   data/METABRIC/metabric_clinical.rds     clinical table (cBioPortal brca_metabric)
#   data/TCGA/tcga_brca_se.rds              SummarizedExperiment from GDC
#                                           (build it with scripts/00_download_tcga_brca.R)
#
# Produces:
#   data/precision_processed/tf_activities_metabric.csv
#   data/precision_processed/metabric_clinical.csv   (adds PAM50 computed with genefu)
#   data/precision_processed/tf_activities_tcga.csv
#   data/precision_processed/tcga_clinical.csv
#
# Usage (any working directory, paths are resolved by R/config.R):
#   Rscript scripts/20_multicohort_validation.R
# ==============================================================================

rm(list = ls())

# Paths (DATA_DIR, PROCESSED_DIR, PRIOR_DIR) and the CollecTRI snapshot loader
# come from R/config.R, next to config.py at the package root.
.cfg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                  "..", "R", "config.R")
source(if (file.exists(.cfg)) .cfg else "R/config.R")

library(dplyr)
library(tidyr)
library(tibble)
library(decoupleR)

OUTPUT_DIR <- PROCESSED_DIR
METABRIC_DIR <- file.path(DATA_DIR, "METABRIC")
TCGA_DIR <- file.path(DATA_DIR, "TCGA")

cat("============================================================\n")
cat("PRECISION: Multi-cohort TF activity computation\n")
cat("============================================================\n\n")

# ==============================================================================
# 1. METABRIC
# ==============================================================================
cat("--- METABRIC ---\n")

clinical_m <- readRDS(file.path(METABRIC_DIR, "metabric_clinical.rds"))
expr_m <- readRDS(file.path(METABRIC_DIR, "metabric_expression.rds"))
gene_map_m <- readRDS(file.path(METABRIC_DIR, "metabric_gene_map.rds"))

cat("  Clinical:", nrow(clinical_m), "patients\n")
cat("  Expression:", nrow(expr_m), "genes x", ncol(expr_m), "samples\n")

# Check survival columns
surv_cols <- grep("surv|os|death|event|vital|follow|time|status", colnames(clinical_m),
                  ignore.case = TRUE, value = TRUE)
cat("  Survival columns:", paste(surv_cols, collapse = ", "), "\n")

# Compute TF activities
cat("  Computing TF activities via ULM...\n")
# Same CollecTRI snapshot the graph was built from (R/config.R)
collectri <- load_collectri_snapshot()

# Expression matrix needs genes in rows
# METABRIC: 6549 genes x 1980 samples, genes in rows (Hugo symbols)
expr_m <- as.matrix(expr_m)
cat("  Expression:", nrow(expr_m), "genes x", ncol(expr_m), "samples\n")
cat("  First genes:", head(rownames(expr_m), 5), "\n")

# Remove NAs/Infs
na_genes <- rowSums(is.na(expr_m)) > 0
inf_genes <- rowSums(!is.finite(expr_m)) > 0
bad <- na_genes | inf_genes
if (any(bad)) {
  cat("  Removing", sum(bad), "genes with NAs/Infs\n")
  expr_m <- expr_m[!bad, ]
}
cat("  Clean expression:", nrow(expr_m), "genes x", ncol(expr_m), "samples\n")

tf_acts_m <- run_ulm(
  mat = expr_m,
  net = collectri,
  .source = "source",
  .target = "target",
  .mor = "mor",
  minsize = 5
)

# Pivot to wide
tf_wide_m <- tf_acts_m %>%
  filter(statistic == "ulm") %>%
  dplyr::select(source, condition, score) %>%  # explicit: genefu attaches packages that mask select()
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")

cat("  TF activities:", nrow(tf_wide_m), "samples x", ncol(tf_wide_m), "TFs\n")

write.csv(tf_wide_m, file.path(OUTPUT_DIR, "tf_activities_metabric.csv"))

# Compute PAM50 via genefu before saving clinical, so the survival pipeline
# always has the PAM50 column available (idempotent: regenerates each run)
cat("  Computing PAM50 with genefu...\n")
suppressPackageStartupMessages(library(genefu))
data(pam50.robust)
data(pam50.scale)
data(pam50)

# Build annotation: probe = Hugo symbol, EntrezGene.ID from gene_map_m
annot_m <- data.frame(
  probe = rownames(expr_m),
  Gene.Symbol = rownames(expr_m),
  stringsAsFactors = FALSE
)
entrez_map <- setNames(gene_map_m$entrezGeneId, gene_map_m$hugoGeneSymbol)
annot_m$EntrezGene.ID <- as.integer(entrez_map[annot_m$Gene.Symbol])
rownames(annot_m) <- annot_m$probe

pam50_result <- molecular.subtyping(
  sbt.model = "pam50",
  data = t(expr_m),
  annot = annot_m,
  do.mapping = TRUE
)
pam50_lookup <- setNames(as.character(pam50_result$subtype),
                          names(pam50_result$subtype))
clinical_m$PAM50 <- pam50_lookup[clinical_m$patientId]
cat("  PAM50 distribution:\n")
print(table(clinical_m$PAM50, useNA = "ifany"))

write.csv(clinical_m, file.path(OUTPUT_DIR, "metabric_clinical.csv"), row.names = TRUE)
cat("  Saved.\n\n")

# ==============================================================================
# 2. TCGA-BRCA
# ==============================================================================
cat("--- TCGA-BRCA ---\n")

se <- readRDS(file.path(TCGA_DIR, "tcga_brca_se.rds"))

# Extract expression and clinical (use TPM for cross-cohort comparability with DepMap)
if (inherits(se, "SummarizedExperiment") || inherits(se, "RangedSummarizedExperiment")) {
  library(SummarizedExperiment)
  available_assays <- assayNames(se)
  cat("  Available assays:", paste(available_assays, collapse = ", "), "\n")
  if ("tpm_unstrand" %in% available_assays) {
    expr_t <- assay(se, "tpm_unstrand")
    cat("  Using TPM assay (tpm_unstrand) for consistency with DepMap log2(TPM+1)\n")
  } else {
    expr_t <- assay(se)
    cat("  WARNING: TPM assay not available, falling back to default assay\n")
  }
  clinical_t <- as.data.frame(colData(se))
  cat("  SummarizedExperiment:", nrow(expr_t), "genes x", ncol(expr_t), "samples\n")

  # Map ENSG IDs to gene symbols
  rd <- rowData(se)
  if ("gene_name" %in% colnames(rd)) {
    gene_names <- rd$gene_name
    valid <- !is.na(gene_names) & gene_names != "" & !duplicated(gene_names)
    expr_t <- expr_t[valid, ]
    rownames(expr_t) <- gene_names[valid]
    cat("  Mapped to gene symbols:", nrow(expr_t), "genes\n")
  }
} else if (is.list(se)) {
  # Might be a list with expression and clinical
  cat("  List with elements:", paste(names(se), collapse = ", "), "\n")
  expr_t <- se$expression
  clinical_t <- se$clinical
} else {
  cat("  Unknown format:", class(se), "\n")
  cat("  Skipping TCGA\n")
  quit(save = "no")
}

cat("  Clinical:", nrow(clinical_t), "patients\n")
surv_cols_t <- grep("surv|os|death|event|vital|follow|time|status|days",
                    colnames(clinical_t), ignore.case = TRUE, value = TRUE)
cat("  Survival columns:", paste(surv_cols_t, collapse = ", "), "\n")

# Clean NAs
na_t <- rowSums(is.na(expr_t)) > 0
inf_t <- rowSums(!is.finite(as.matrix(expr_t))) > 0
bad_t <- na_t | inf_t
if (any(bad_t)) {
  cat("  Removing", sum(bad_t), "genes with NAs/Infs\n")
  expr_t <- expr_t[!bad_t, ]
}
cat("  Clean expression:", nrow(expr_t), "genes x", ncol(expr_t), "samples\n")

# Apply log2(TPM+1) for ULM, consistent with DepMap log2(TPM+1)
cat("  TPM scale check - max:", max(expr_t), "median:", median(as.matrix(expr_t)), "\n")
expr_t <- log2(as.matrix(expr_t) + 1)
cat("  After log2(TPM+1) - max:", round(max(expr_t), 2),
    "median:", round(median(expr_t), 2), "\n")

# Compute TF activities
cat("  Computing TF activities via ULM...\n")
tf_acts_t <- run_ulm(
  mat = expr_t,
  net = collectri,
  .source = "source",
  .target = "target",
  .mor = "mor",
  minsize = 5
)

tf_wide_t <- tf_acts_t %>%
  filter(statistic == "ulm") %>%
  dplyr::select(source, condition, score) %>%  # explicit: genefu attaches packages that mask select()
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")

cat("  TF activities:", nrow(tf_wide_t), "samples x", ncol(tf_wide_t), "TFs\n")

write.csv(tf_wide_t, file.path(OUTPUT_DIR, "tf_activities_tcga.csv"))
# Flatten list columns to character before saving
for (col in colnames(clinical_t)) {
  if (is.list(clinical_t[[col]])) {
    clinical_t[[col]] <- sapply(clinical_t[[col]], function(x) paste(x, collapse=";"))
  }
}
write.csv(clinical_t, file.path(OUTPUT_DIR, "tcga_clinical.csv"), row.names = TRUE)
cat("  Saved.\n\n")

cat("--- DONE ---\n")
