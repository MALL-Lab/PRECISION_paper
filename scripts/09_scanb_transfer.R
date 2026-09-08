#!/usr/bin/env Rscript
# ==============================================================================
# PRECISION: TF activities and clinical table for the SCAN-B cohort
#
# Steps:
# 1. Load SCAN-B expression (9,206 breast cancer patients, RNA-seq, batch-adjusted FPKM)
# 2. Compute TF activity profiles (ULM, CollecTRI), same procedure as for cell lines
# 3. Export TF activities and clinical table for the Python transfer step (script 17)
# 4. Sanity check: univariate Cox of three key TFs vs overall survival
# 5. Same sanity check restricted to the TNBC/Basal subgroup
#
# Outputs:
#   data/precision_processed/tf_activities_scanb.csv
#   data/precision_processed/scanb_clinical.csv
#   results/scanb_tf_cox_results.csv (sanity check only, not used in the paper)
#
# Usage (any working directory, paths are resolved by R/config.R):
#   Rscript scripts/09_scanb_transfer.R
# ==============================================================================

rm(list = ls())

# Paths (DATA_DIR, RESULTS_DIR, PROCESSED_DIR, PRIOR_DIR) and the CollecTRI
# snapshot loader come from R/config.R, next to config.py at the package root.
.cfg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
                  "..", "R", "config.R")
source(if (file.exists(.cfg)) .cfg else "R/config.R")

library(dplyr)
library(tidyr)
library(tibble)
library(survival)
library(survminer)

cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
cat("PRECISION: SCAN-B Transfer Validation\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

# Expected layout (see data/README.md, section SCAN-B):
#   data/SCANB/SCANB.9206.genematrix_noNeg.Rdata   batch-adjusted FPKM, genes x samples
#   data/SCANB/clinical_formatted.rds              clinical table
#   data/SCANB/Gene.ID.ann.Rdata                   Ensembl to symbol annotation
SCANB_DIR <- file.path(DATA_DIR, "SCANB")
OUTPUT_DIR <- PROCESSED_DIR

# ==============================================================================
# 1. LOAD SCAN-B DATA
# ==============================================================================
cat("--- Step 1: Loading SCAN-B data ---\n")

# Clinical
clinical <- readRDS(file.path(SCANB_DIR, "clinical_formatted.rds"))
clinical <- clinical %>% filter(include == "yes")
cat("  Clinical samples:", nrow(clinical), "\n")
cat("  PAM50 distribution:\n")
print(table(clinical$pam50, useNA = "ifany"))
cat("  Survival data (os_time):", sum(!is.na(clinical$os_time)), "\n\n")

# Expression
cat("--- Loading expression (may take ~30s) ---\n")
load(file.path(SCANB_DIR, "SCANB.9206.genematrix_noNeg.Rdata"))
expr_mat <- SCANB.9206.genematrix_noNeg
rm(SCANB.9206.genematrix_noNeg)
cat("  Expression:", nrow(expr_mat), "genes x", ncol(expr_mat), "samples\n")

# Gene annotation: map Ensembl to Hugo symbols
load(file.path(SCANB_DIR, "Gene.ID.ann.Rdata"))
gene_map <- Gene.ID.ann %>%
  select(Gene.ID, Gene.Name) %>%
  filter(Gene.Name != "", !is.na(Gene.Name), !duplicated(Gene.Name))

mapped_genes <- intersect(rownames(expr_mat), gene_map$Gene.ID)
expr_mat <- expr_mat[mapped_genes, ]
rownames(expr_mat) <- gene_map$Gene.Name[match(rownames(expr_mat), gene_map$Gene.ID)]
expr_mat <- expr_mat[!duplicated(rownames(expr_mat)), ]

# SCAN-B expression is FPKM (linear scale). Apply log2(x+1) to match
# the log-scale expected by ULM and consistent with DepMap log2(TPM+1).
expr_mat <- log2(expr_mat + 1)

cat("  Mapped expression:", nrow(expr_mat), "genes x", ncol(expr_mat), "samples\n\n")

# ==============================================================================
# 2. COMPUTE TF ACTIVITIES (ALL TFs)
# ==============================================================================
cat("--- Step 2: Computing TF activities via decoupleR ULM ---\n")

library(decoupleR)

# Load CollecTRI: the same snapshot the graph was built from (R/config.R)
collectri <- load_collectri_snapshot()
cat("  CollecTRI:", nrow(collectri), "interactions,",
    length(unique(collectri$source)), "TFs\n")

# decoupleR >= 2.16: genes must be in ROWS (mat = genes x samples)
tf_acts_raw <- run_ulm(
  mat = expr_mat,
  net = collectri,
  .source = "source",
  .target = "target",
  .mor = "mor",
  minsize = 5
)

# Pivot to wide format: samples x TFs
tf_acts_wide <- tf_acts_raw %>%
  filter(statistic == "ulm") %>%
  select(source, condition, score) %>%
  pivot_wider(names_from = source, values_from = score) %>%
  column_to_rownames("condition")

cat("  TF activities:", nrow(tf_acts_wide), "samples x", ncol(tf_acts_wide), "TFs\n\n")

# Save for Python (will be re-saved with short IDs after merge step)
tf_output_path <- file.path(OUTPUT_DIR, "tf_activities_scanb.csv")
cat("  (saving after ID mapping in step 3)\n\n")

# ==============================================================================
# 3. MERGE WITH CLINICAL DATA
# ==============================================================================
cat("--- Step 3: Merging TF activities with clinical data ---\n")

# Expression sample IDs are long (S000001.l.r.m.c.lib.g.k2.a.t),
# clinical IDs are short (S000001). Extract prefix for matching.
tf_acts_wide$short_id <- sub("\\..*", "", rownames(tf_acts_wide))

# Some patients have multiple samples: keep the first per short_id
tf_acts_wide <- tf_acts_wide[!duplicated(tf_acts_wide$short_id), ]
rownames(tf_acts_wide) <- tf_acts_wide$short_id
tf_acts_wide$short_id <- NULL

shared_samples <- intersect(rownames(tf_acts_wide), clinical$sample_id)
cat("  Shared samples:", length(shared_samples), "\n")

# Save TF activities with short IDs for Python
write.csv(tf_acts_wide, tf_output_path)
cat("  TF activities saved to:", tf_output_path, "\n")

# Save clinical data for Python merge
clinical_output <- file.path(OUTPUT_DIR, "scanb_clinical.csv")
write.csv(clinical, clinical_output, row.names = FALSE)
cat("  Clinical data saved to:", clinical_output, "\n\n")

# ==============================================================================
# 4. BASIC TF-SURVIVAL ANALYSIS (sanity check)
# ==============================================================================
cat("--- Step 4: TF-Survival sanity check ---\n")

# Match samples
# Deduplicate clinical by sample_id (some have multiple entries)
clinical_dedup <- clinical[!duplicated(clinical$sample_id), ]

tf_clinical <- tf_acts_wide %>%
  rownames_to_column("sample_id") %>%
  inner_join(clinical_dedup, by = "sample_id") %>%
  filter(!is.na(os_time), !is.na(os_event))

cat("  Samples with TF + survival:", nrow(tf_clinical), "\n")

# Test key TFs from XAI analysis
key_tfs <- c("MYC", "TP53", "E2F1")
key_tfs_present <- key_tfs[key_tfs %in% colnames(tf_clinical)]

cat("\n  Cox regression (univariate) for key TFs:\n")
cox_results <- list()
for (tf in key_tfs_present) {
  tf_clinical[[paste0(tf, "_high")]] <- ifelse(
    tf_clinical[[tf]] > median(tf_clinical[[tf]], na.rm = TRUE), "High", "Low"
  )

  formula <- as.formula(paste0("Surv(os_time, os_event) ~ ", tf, "_high"))
  cox <- coxph(formula, data = tf_clinical)
  s <- summary(cox)

  cat(sprintf("    %s: HR=%.2f (%.2f-%.2f), p=%.2e\n",
              tf, s$conf.int[1], s$conf.int[3], s$conf.int[4],
              s$coefficients[5]))

  cox_results[[tf]] <- data.frame(
    tf = tf,
    HR = s$conf.int[1],
    HR_lower = s$conf.int[3],
    HR_upper = s$conf.int[4],
    pvalue = s$coefficients[5]
  )
}

cox_df <- do.call(rbind, cox_results)
write.csv(cox_df, file.path(RESULTS_DIR, "scanb_tf_cox_results.csv"), row.names = FALSE)

# ==============================================================================
# 5. TNBC SUBGROUP ANALYSIS
# ==============================================================================
cat("\n--- Step 5: TNBC subgroup analysis ---\n")

# Identify TNBC: ER- and HER2- (or PAM50 = Basal)
if ("er" %in% colnames(tf_clinical) && "her2" %in% colnames(tf_clinical)) {
  tnbc <- tf_clinical %>% filter(er == "Negative", her2 == "Negative")
} else if ("pam50" %in% colnames(tf_clinical)) {
  tnbc <- tf_clinical %>% filter(pam50 == "Basal")
} else {
  tnbc <- tf_clinical  # fallback
}
cat("  TNBC/Basal samples:", nrow(tnbc), "\n")

if (nrow(tnbc) > 50) {
  for (tf in key_tfs_present) {
    tnbc[[paste0(tf, "_high")]] <- ifelse(
      tnbc[[tf]] > median(tnbc[[tf]], na.rm = TRUE), "High", "Low"
    )
    formula <- as.formula(paste0("Surv(os_time, os_event) ~ ", tf, "_high"))
    tryCatch({
      cox <- coxph(formula, data = tnbc)
      s <- summary(cox)
      cat(sprintf("    TNBC %s: HR=%.2f (%.2f-%.2f), p=%.2e\n",
                  tf, s$conf.int[1], s$conf.int[3], s$conf.int[4],
                  s$coefficients[5]))
    }, error = function(e) cat(sprintf("    TNBC %s: Cox failed (%s)\n", tf, e$message)))
  }
}

cat("\n--- DONE ---\n")
cat("TF activities saved for Python GNN prediction step.\n")
cat("Next: run Python script to predict drug sensitivity for SCAN-B patients.\n")
