# ==============================================================================
# PRECISION paper package: path configuration for the R scripts
#
# Mirrors config.py. Every R script sources this file first, so the scripts
# work from any working directory and the data/results locations can be
# overridden with the same environment variables as the Python side:
#
#   PRECISION_DATA     root of the data tree (default: <package>/data)
#   PRECISION_RESULTS  root of the results tree (default: <package>/results)
#
# Usage at the top of a script:
#   source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "..", "R", "config.R"))
# or, simpler, since every script lives one level below the package root:
#   source("R/config.R")   # when run from the package root
# ==============================================================================

.precision_root <- local({
  # Path of the file being sourced, resolved from the Rscript command line.
  f <- grep("--file=", commandArgs(), value = TRUE)
  if (length(f) > 0) {
    script <- normalizePath(sub("--file=", "", f[1]), winslash = "/", mustWork = FALSE)
    # scripts/<name>.R or paper/scripts/<name>.R: walk up until config.py is found
    d <- dirname(script)
    for (i in 1:4) {
      if (file.exists(file.path(d, "config.py"))) return(d)
      d <- dirname(d)
    }
  }
  # Interactive session or unusual launcher: fall back to the working directory
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
})

PRECISION_ROOT <- .precision_root

.env_path <- function(var, default) {
  v <- Sys.getenv(var, unset = "")
  if (nzchar(v)) normalizePath(v, winslash = "/", mustWork = FALSE) else default
}

DATA_DIR    <- .env_path("PRECISION_DATA",    file.path(PRECISION_ROOT, "data"))
RESULTS_DIR <- .env_path("PRECISION_RESULTS", file.path(PRECISION_ROOT, "results"))
PROCESSED_DIR <- file.path(DATA_DIR, "precision_processed")
PRIOR_DIR     <- file.path(DATA_DIR, "prior_knowledge")
PAPER_DIR     <- file.path(PRECISION_ROOT, "paper")

dir.create(PROCESSED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RESULTS_DIR,   recursive = TRUE, showWarnings = FALSE)

# CollecTRI regulon used by the whole pipeline. The Python side caches the
# network in data/prior_knowledge/collectri_edges.csv (columns source, target,
# weight) and the graph is built from that snapshot. The R scripts read the same
# snapshot so that clinical TF activities and the graph share one regulon.
# Without the snapshot we fall back to a live download, which drifts over time
# (see data/README.md).
load_collectri_snapshot <- function() {
  path <- file.path(PRIOR_DIR, "collectri_edges.csv")
  if (file.exists(path)) {
    net <- read.csv(path, stringsAsFactors = FALSE)
    net <- data.frame(source = net$source, target = net$target, mor = net$weight)
    cat("  CollecTRI snapshot:", path, "(", nrow(net), "edges,",
        length(unique(net$source)), "TFs )\n")
    return(net)
  }
  warning("CollecTRI snapshot not found at ", path,
          ": downloading the live network from OmniPath (results may drift)")
  net <- decoupleR::get_collectri(organism = "human", split_complexes = FALSE)
  as.data.frame(net[, c("source", "target", "mor")])
}
