# Environment used to produce the results in the paper (R side).
#
# R 4.5.2 with Bioconductor 3.22 and the exact package versions listed below.
# Run from the repository root:
#   Rscript R/install.R
#
# Bioconductor packages are installed from the Bioconductor 3.22 release
# branch through BiocManager. CRAN packages are pinned to the exact version
# with remotes::install_version(), which pulls from the CRAN archive.
# The final block compares the installed versions against the expected ones
# and warns about any mismatch.
#
# scripts/00_download_tcga_brca.R additionally needs TCGAbiolinks (Bioconductor),
# only to fetch the TCGA-BRCA input: BiocManager::install("TCGAbiolinks").

expected_r_version <- "4.5.2"
expected_bioc_version <- "3.22"
cran_repo <- "https://cloud.r-project.org"

# Bioconductor packages and the versions used in the paper.
bioc_versions <- c(
  decoupleR            = "2.16.0",
  genefu               = "2.42.0",
  SummarizedExperiment = "1.40.0",
  OmnipathR            = "3.18.4"
)

# CRAN packages and the versions used in the paper.
# The survival version string follows the CRAN archive naming (3.8-3);
# R reports it as 3.8.3.
cran_versions <- c(
  survival  = "3.8-3",
  survminer = "0.5.2",
  dplyr     = "1.2.0",
  tidyr     = "1.3.2",
  tibble    = "3.3.1"
)

# ---------------------------------------------------------------------------
# R version check
# ---------------------------------------------------------------------------
running_r <- paste(R.version$major, R.version$minor, sep = ".")
if (running_r != expected_r_version) {
  warning(sprintf("Expected R %s, running R %s. Package versions may differ.",
                  expected_r_version, running_r))
}

# ---------------------------------------------------------------------------
# Installers
# ---------------------------------------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = cran_repo)
}
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = cran_repo)
}

# Pin the Bioconductor release. This also sets the matching CRAN snapshot
# used by BiocManager for dependencies.
BiocManager::install(version = expected_bioc_version, ask = FALSE, update = FALSE)

# ---------------------------------------------------------------------------
# CRAN packages, exact versions
# ---------------------------------------------------------------------------
for (pkg in names(cran_versions)) {
  remotes::install_version(
    package = pkg,
    version = cran_versions[[pkg]],
    repos   = cran_repo,
    upgrade = "never"
  )
}

# ---------------------------------------------------------------------------
# Bioconductor packages from the 3.22 release branch
# ---------------------------------------------------------------------------
BiocManager::install(
  names(bioc_versions),
  version = expected_bioc_version,
  ask     = FALSE,
  update  = FALSE
)

# ---------------------------------------------------------------------------
# Verify installed versions against the expected ones
# ---------------------------------------------------------------------------
expected <- c(bioc_versions, cran_versions)
installed <- vapply(names(expected), function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    as.character(packageVersion(pkg))
  } else {
    NA_character_
  }
}, character(1))

# Compare as package_version so that "3.8-3" and "3.8.3" are treated as equal.
matches <- mapply(function(exp, inst) {
  !is.na(inst) && package_version(exp) == package_version(inst)
}, expected, installed)

report <- data.frame(
  package   = names(expected),
  expected  = unname(expected),
  installed = unname(installed),
  ok        = unname(matches),
  row.names = NULL,
  stringsAsFactors = FALSE
)
print(report, row.names = FALSE)

if (!all(matches)) {
  warning(paste0(
    "Some packages do not match the versions used in the paper: ",
    paste(report$package[!report$ok], collapse = ", "),
    ". Bioconductor only serves the latest patch of each release branch, ",
    "so a newer patch version within Bioconductor 3.22 is expected in some cases."
  ))
} else {
  message("All packages match the versions used in the paper.")
}
