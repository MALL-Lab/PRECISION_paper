#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: Download TCGA-BRCA expression from GDC and cache it as an RDS
#
# Builds data/TCGA/tcga_brca_se.rds, the SummarizedExperiment read by
# scripts/20_multicohort_validation.R. The paper used the GDC STAR-Counts
# workflow (assay "tpm_unstrand", 1,234 samples). GDC data are open access
# (Transcriptome Profiling, Gene Expression Quantification).
#
# Requires TCGAbiolinks (Bioconductor). The download is about 1.5 GB and the
# cached RDS is about 760 MB, which is why the file is not redistributed.
#
# Usage (any working directory, paths are resolved by R/config.R):
#   Rscript scripts/00_download_tcga_brca.R
# ==============================================================================

.cfg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                  "..", "R", "config.R")
source(if (file.exists(.cfg)) .cfg else "R/config.R")

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
})

tcga_dir <- file.path(DATA_DIR, "TCGA")
dir.create(tcga_dir, recursive = TRUE, showWarnings = FALSE)
out_rds <- file.path(tcga_dir, "tcga_brca_se.rds")

if (file.exists(out_rds)) {
  cat("Already present:", out_rds, "\n")
  quit(save = "no")
}

# The GDC download (about 1.5 GB) goes next to the output, not to tempdir(),
# which is often a small tmpfs on Linux.
gdc_cache <- file.path(tcga_dir, "gdc_tmp")
dir.create(gdc_cache, recursive = TRUE, showWarnings = FALSE)

query <- GDCquery(project = "TCGA-BRCA",
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "STAR - Counts")
GDCdownload(query, method = "api", files.per.chunk = 50, directory = gdc_cache)
se <- GDCprepare(query, save = FALSE, directory = gdc_cache)

cat("Assays:", paste(assayNames(se), collapse = ", "), "\n")
cat("Samples:", ncol(se), " genes:", nrow(se), "\n")
saveRDS(se, out_rds)
cat("Saved:", out_rds, "\n")
unlink(gdc_cache, recursive = TRUE)
