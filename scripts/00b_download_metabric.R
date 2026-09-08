#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: Download the METABRIC inputs from the cBioPortal web API
#
# Builds the three files read by scripts/20_multicohort_validation.R:
#   data/METABRIC/metabric_clinical.rds     patient-level clinical attributes
#   data/METABRIC/metabric_gene_map.rds     hugoGeneSymbol to entrezGeneId
#   data/METABRIC/metabric_expression.rds   log-intensity matrix, genes x samples
#
# The expression matrix is deliberately restricted to the target genes of the
# CollecTRI regulon (about 6,500 genes of the Illumina HT-12 array): it is
# enough for TF activity inference, which is the only use of METABRIC in the
# paper. The same restriction was used for the paper (6,549 genes x 1,980
# samples). cBioPortal (study brca_metabric) publishes these data without an
# explicit license, which is why the files are not redistributed.
#
# Requires httr, jsonlite, dplyr, tidyr, tibble. Network access needed.
#
# Usage (any working directory, paths are resolved by R/config.R):
#   Rscript scripts/00b_download_metabric.R
# ==============================================================================

.cfg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                  "..", "R", "config.R")
source(if (file.exists(.cfg)) .cfg else "R/config.R")

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

metabric_dir <- file.path(DATA_DIR, "METABRIC")
dir.create(metabric_dir, recursive = TRUE, showWarnings = FALSE)

CBIO_BASE <- "https://www.cbioportal.org/api"
STUDY_ID <- "brca_metabric"

cbio_get <- function(endpoint, ...) {
  url <- paste0(CBIO_BASE, endpoint)
  resp <- GET(url, query = list(...), add_headers(accept = "application/json"))
  if (status_code(resp) != 200) stop("cBioPortal API error ", status_code(resp), " for ", url)
  fromJSON(content(resp, as = "text", encoding = "UTF-8"), flatten = TRUE)
}

cbio_post <- function(endpoint, body, ...) {
  resp <- POST(paste0(CBIO_BASE, endpoint), body = body, query = list(...),
               content_type_json(), add_headers(accept = "application/json"))
  if (status_code(resp) != 200) return(NULL)
  fromJSON(content(resp, as = "text", encoding = "UTF-8"), flatten = TRUE)
}

# -- 1. Patient-level clinical table ------------------------------------------
clinical_file <- file.path(metabric_dir, "metabric_clinical.rds")
if (file.exists(clinical_file)) {
  cat("Already present:", clinical_file, "\n")
} else {
  cat("Downloading clinical attributes...\n")
  clinical_raw <- cbio_get(paste0("/studies/", STUDY_ID, "/clinical-data"),
                           clinicalDataType = "PATIENT", projection = "DETAILED")
  clinical <- clinical_raw %>%
    select(patientId, clinicalAttributeId, value) %>%
    pivot_wider(names_from = clinicalAttributeId, values_from = value)
  saveRDS(clinical, clinical_file)
  cat("  Patients:", nrow(clinical), " attributes:", ncol(clinical) - 1, "\n")
}

# -- 2. Regulon target genes and their Entrez identifiers ----------------------
net <- load_collectri_snapshot()
regulon_genes <- unique(net$target)
cat("CollecTRI target genes:", length(regulon_genes), "\n")

gene_map_file <- file.path(metabric_dir, "metabric_gene_map.rds")
if (file.exists(gene_map_file)) {
  gene_map <- readRDS(gene_map_file)
  cat("Already present:", gene_map_file, "\n")
} else {
  cat("Resolving Entrez identifiers...\n")
  pieces <- list()
  batch_sz <- 300
  for (i in seq(1, length(regulon_genes), by = batch_sz)) {
    batch <- regulon_genes[i:min(i + batch_sz - 1, length(regulon_genes))]
    gd <- cbio_post("/genes/fetch", body = toJSON(batch, auto_unbox = FALSE),
                    geneIdType = "HUGO_GENE_SYMBOL")
    if (is.data.frame(gd) && nrow(gd) > 0) {
      pieces[[length(pieces) + 1]] <- gd %>% select(entrezGeneId, hugoGeneSymbol)
    }
  }
  gene_map <- bind_rows(pieces) %>% distinct()
  saveRDS(gene_map, gene_map_file)
  cat("  Genes mapped:", nrow(gene_map), "\n")
}

# -- 3. Expression restricted to the regulon genes -----------------------------
expr_file <- file.path(metabric_dir, "metabric_expression.rds")
if (file.exists(expr_file)) {
  cat("Already present:", expr_file, "\n")
} else {
  cat("Downloading expression (mRNA profile, regulon genes only)...\n")
  profiles <- cbio_get(paste0("/studies/", STUDY_ID, "/molecular-profiles"))
  mrna_profile <- profiles %>%
    filter(grepl("mrna", molecularProfileId, ignore.case = TRUE),
           molecularAlterationType == "MRNA_EXPRESSION") %>%
    slice(1)
  profile_id <- mrna_profile$molecularProfileId
  cat("  Profile:", profile_id, "\n")
  samples <- cbio_get(paste0("/studies/", STUDY_ID, "/samples"))
  sample_ids <- samples$sampleId
  all_expr <- list()
  batch_size <- 200
  for (i in seq(1, length(sample_ids), by = batch_size)) {
    batch <- sample_ids[i:min(i + batch_size - 1, length(sample_ids))]
    bd <- cbio_post(paste0("/molecular-profiles/", profile_id, "/molecular-data/fetch"),
                    body = toJSON(list(entrezGeneIds = gene_map$entrezGeneId,
                                       sampleIds = batch), auto_unbox = FALSE))
    if (is.data.frame(bd) && nrow(bd) > 0) {
      bd <- bd %>%
        inner_join(gene_map, by = "entrezGeneId") %>%
        select(sampleId, hugoGeneSymbol, value)
      if (nrow(bd) > 0) all_expr[[length(all_expr) + 1]] <- bd
    }
  }
  expr_wide <- bind_rows(all_expr) %>%
    pivot_wider(names_from = sampleId, values_from = value, values_fn = mean) %>%
    column_to_rownames("hugoGeneSymbol")
  saveRDS(expr_wide, expr_file)
  cat("  Expression:", nrow(expr_wide), "genes x", ncol(expr_wide), "samples\n")
  cat("  Value range:", round(min(expr_wide, na.rm = TRUE), 2), "to",
      round(max(expr_wide, na.rm = TRUE), 2), "(log2 microarray intensities, do not log again)\n")
}

cat("Done. Files under", metabric_dir, "\n")
