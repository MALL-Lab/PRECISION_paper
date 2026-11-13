#!/usr/bin/env Rscript
# ==============================================================================
# 20b_metabric_pam50.R
# Compute PAM50 molecular subtypes for METABRIC using genefu
# ==============================================================================
#
# Input:
#   data/METABRIC/metabric_expression.rds   (6549 genes x 1980 samples, log2)
#   data/METABRIC/metabric_gene_map.rds     (Hugo -> Entrez mapping)
#   data/precision_processed/metabric_clinical.csv
#
# Output:
#   data/precision_processed/metabric_clinical.csv  (updated with PAM50 column)
#   data/precision_processed/metabric_pam50_proba.csv (subtype probabilities)
#
# Run: Rscript scripts/20b_metabric_pam50.R

rm(list = ls())

suppressPackageStartupMessages({
  library(genefu)
  library(dplyr)
})

cat("=" , rep("=", 59), "\n", sep = "")
cat("METABRIC PAM50 classification via genefu\n")
cat("=" , rep("=", 59), "\n", sep = "")

# -- 1. Load data --------------------------------------------------------------
cat("\n[1] Loading data...\n")

expr <- readRDS(file.path(DATA_DIR, "METABRIC/metabric_expression.rds"))
cat("  Expression:", nrow(expr), "genes x", ncol(expr), "samples\n")
cat("  Range:", round(range(expr, na.rm = TRUE), 2), "\n")

gmap <- readRDS(file.path(DATA_DIR, "METABRIC/metabric_gene_map.rds"))
cat("  Gene map:", nrow(gmap), "entries\n")

clinical <- read.csv(file.path(PROCESSED_DIR, "metabric_clinical.csv"),
                      stringsAsFactors = FALSE)
cat("  Clinical:", nrow(clinical), "patients\n")

# -- 2. Prepare genefu inputs --------------------------------------------------
cat("\n[2] Preparing inputs for genefu...\n")

# Transpose: genefu needs samples in rows, genes in columns
expr_t <- t(expr)
cat("  Transposed matrix:", nrow(expr_t), "samples x", ncol(expr_t), "genes\n")

# Build annotation data.frame
annot <- data.frame(
  probe = rownames(expr),
  Gene.Symbol = rownames(expr),
  stringsAsFactors = FALSE
)

# Map Entrez IDs from gene map
# gmap has columns: hugoGeneSymbol, entrezGeneId (or similar)
gmap_cols <- names(gmap)
cat("  Gene map columns:", paste(gmap_cols, collapse = ", "), "\n")

# Map Entrez IDs: columns are entrezGeneId and hugoGeneSymbol
entrez_map <- setNames(gmap$entrezGeneId, gmap$hugoGeneSymbol)
annot$EntrezGene.ID <- as.integer(entrez_map[annot$Gene.Symbol])
cat("  Entrez IDs mapped:", sum(!is.na(annot$EntrezGene.ID)), "of", nrow(annot), "\n")

rownames(annot) <- annot$probe

# Check PAM50 gene coverage
pam50_genes <- c("ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1",
                 "CDC20","CDC6","CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2",
                 "ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7","KIF2C",
                 "KRT14","KRT17","KRT5","MAPT","MDM2","MELK","MIA","MKI67",
                 "MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6L",
                 "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B",
                 "TYMS","UBE2C","UBE2T")
present <- pam50_genes[pam50_genes %in% colnames(expr_t)]
missing <- pam50_genes[!pam50_genes %in% colnames(expr_t)]
cat("  PAM50 genes:", length(present), "/", length(pam50_genes), "present\n")
cat("  Missing:", paste(missing, collapse = ", "), "\n")

# -- 3. Run PAM50 classification -----------------------------------------------
cat("\n[3] Running PAM50 classification...\n")

# Load PAM50 model data (required by molecular.subtyping internals)
data(pam50.robust)
data(pam50.scale)
data(pam50)

pam50_result <- molecular.subtyping(
  sbt.model = "pam50",
  data = expr_t,
  annot = annot,
  do.mapping = TRUE
)

subtypes <- pam50_result$subtype
proba <- as.data.frame(pam50_result$subtype.proba)

cat("  PAM50 distribution:\n")
print(table(subtypes, useNA = "ifany"))

# -- 4. Validate vs CLAUDIN_SUBTYPE --------------------------------------------
cat("\n[4] Concordance vs CLAUDIN_SUBTYPE...\n")

# Match samples
pam50_df <- data.frame(
  sample = names(subtypes),
  PAM50 = as.character(subtypes),
  stringsAsFactors = FALSE
)

# Find sample ID column in clinical
id_col <- intersect(c("patientId", "Unnamed..0"), names(clinical))[1]
if (is.na(id_col)) id_col <- names(clinical)[1]

# Merge
merged <- merge(
  clinical[, c(id_col, "CLAUDIN_SUBTYPE")],
  pam50_df,
  by.x = id_col, by.y = "sample",
  all.x = FALSE
)

cat("  Matched samples:", nrow(merged), "\n")

# Exclude claudin-low and NA for concordance
comparable <- merged %>%
  filter(!is.na(CLAUDIN_SUBTYPE),
         CLAUDIN_SUBTYPE != "NC",
         CLAUDIN_SUBTYPE != "claudin-low",
         !is.na(PAM50))

cat("  Comparable samples (excl. claudin-low, NA, NC):", nrow(comparable), "\n\n")

# Contingency table
cat("  Confusion matrix (rows=CLAUDIN, cols=PAM50):\n")
ct <- table(comparable$CLAUDIN_SUBTYPE, comparable$PAM50)
print(ct)

# Accuracy: map CLAUDIN to PAM50 equivalents
claudin_to_pam50 <- c("LumA" = "LumA", "LumB" = "LumB",
                       "Her2" = "Her2", "Basal" = "Basal",
                       "Normal" = "Normal")
comparable$expected <- claudin_to_pam50[comparable$CLAUDIN_SUBTYPE]
concordant <- sum(comparable$PAM50 == comparable$expected, na.rm = TRUE)
accuracy <- concordant / nrow(comparable) * 100
cat(sprintf("\n  Overall accuracy: %.1f%% (%d/%d)\n", accuracy, concordant, nrow(comparable)))

# Per-subtype accuracy
cat("\n  Per-subtype concordance:\n")
for (sub in names(claudin_to_pam50)) {
  sub_rows <- comparable[comparable$CLAUDIN_SUBTYPE == sub, ]
  if (nrow(sub_rows) > 0) {
    correct <- sum(sub_rows$PAM50 == claudin_to_pam50[sub], na.rm = TRUE)
    cat(sprintf("    %s: %d/%d (%.0f%%)\n", sub, correct, nrow(sub_rows),
                100 * correct / nrow(sub_rows)))
  }
}

# Also show what happened with claudin-low
cl <- merged %>% filter(CLAUDIN_SUBTYPE == "claudin-low" & !is.na(PAM50))
if (nrow(cl) > 0) {
  cat("\n  claudin-low mapped to PAM50:\n")
  print(table(cl$PAM50))
}

# -- 5. Save results -----------------------------------------------------------
cat("\n[5] Saving results...\n")

# Add PAM50 to clinical
pam50_lookup <- setNames(pam50_df$PAM50, pam50_df$sample)
clinical$PAM50 <- pam50_lookup[clinical[[id_col]]]
cat("  PAM50 assigned:", sum(!is.na(clinical$PAM50)), "of", nrow(clinical), "\n")
cat("  PAM50 distribution in clinical:\n")
print(table(clinical$PAM50, useNA = "ifany"))

write.csv(clinical, file.path(PROCESSED_DIR, "metabric_clinical.csv"),
          row.names = FALSE)
cat("  Saved: data/precision_processed/metabric_clinical.csv\n")

# Save probabilities
proba$sample <- rownames(proba)
write.csv(proba, file.path(PROCESSED_DIR, "metabric_pam50_proba.csv"),
          row.names = FALSE)
cat("  Saved: data/precision_processed/metabric_pam50_proba.csv\n")

cat("\n", rep("=", 60), "\n", sep = "")
cat("Done. PAM50 classification complete.\n")
cat(rep("=", 60), "\n", sep = "")
