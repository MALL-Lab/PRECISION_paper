#!/usr/bin/env python3
"""
PRECISION Phase 1, Step 1: Data preparation and prior knowledge download.

This script:
1. Loads DepMap expression and cell line metadata
2. Loads PRISM and GDSC drug response matrices
3. Downloads and caches CollecTRI (TF-target network)
4. Computes TF activity scores for ALL TFs (~700) via ULM
5. Downloads and caches STRING PPI network
6. Builds drug-target edge list from PRISM/GDSC annotations
7. Saves all processed data to data/precision_processed/

Run from project root:
    python scripts/01_prepare_data.py
"""

import sys
import logging
from pathlib import Path

# Anchor imports to the package root regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR

from src.data.load_depmap import load_expression, load_model_info
from src.data.load_drug_response import (
    load_prism_response,
    load_gdsc_response,
    load_prism_drug_annotations,
    load_gdsc_drug_annotations,
    get_shared_cell_lines,
    get_shared_drugs,
)
from src.data.load_prior_knowledge import (
    load_collectri,
    compute_tf_activities,
    load_omnipath_ppi,
    load_string_ppi,
    build_drug_target_edges,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_data")

OUTPUT_DIR = DATA_DIR / "precision_processed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. DepMap expression and metadata ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading DepMap data")
    logger.info("=" * 60)

    expression = load_expression()
    model_info = load_model_info()

    # ── 2. Drug response ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Loading drug response data")
    logger.info("=" * 60)

    prism_response = load_prism_response(use_processed=True)
    gdsc_response = load_gdsc_response(use_processed=True)

    shared_cells = get_shared_cell_lines(prism_response, gdsc_response)
    shared_drugs = get_shared_drugs(prism_response, gdsc_response)

    # ── 3. CollecTRI ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Loading/downloading CollecTRI")
    logger.info("=" * 60)

    collectri = load_collectri(organism="human", cache=True)

    # ── 4. TF activities (ALL TFs) ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Computing TF activities for all TFs")
    logger.info("=" * 60)

    # Use the expression matrix matched to PRISM cell lines
    prism_counts_path = DATA_DIR / "PRISM" / "processed" / "counts_matched_prism.csv"
    gdsc_counts_path = DATA_DIR / "GDSC" / "processed" / "counts_matched_sanger.csv"
    prism_counts = load_prism_response(use_processed=False) if not prism_counts_path.exists() else None

    # Load matched expression counts (already filtered to shared cell lines)
    import pandas as pd
    prism_expr = pd.read_csv(prism_counts_path, index_col=0)
    gdsc_expr = pd.read_csv(gdsc_counts_path, index_col=0)

    tf_acts_prism = compute_tf_activities(prism_expr, collectri)
    tf_acts_gdsc = compute_tf_activities(gdsc_expr, collectri)

    # Save
    tf_acts_prism.to_csv(OUTPUT_DIR / "tf_activities_all_prism.csv")
    tf_acts_gdsc.to_csv(OUTPUT_DIR / "tf_activities_all_gdsc.csv")
    logger.info("Saved TF activities: PRISM %s, GDSC %s",
                tf_acts_prism.shape, tf_acts_gdsc.shape)

    # ── 5. STRING PPI ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Loading/downloading STRING PPI")
    logger.info("=" * 60)

    # Primary: OmniPath (curated PPI, always available)
    ppi = load_omnipath_ppi(cache=True)

    # Optional: STRING bulk download (larger but noisier, often unavailable)
    try:
        string_ppi = load_string_ppi(confidence_threshold=700, cache=True)
        logger.info("STRING PPI also available: %d edges", len(string_ppi))
    except Exception as e:
        logger.info("STRING PPI not available (server down): %s", type(e).__name__)
        string_ppi = None

    # ── 6. Drug-target edges ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Building drug-target edge list")
    logger.info("=" * 60)

    prism_drugs = load_prism_drug_annotations()
    gdsc_drugs = load_gdsc_drug_annotations()
    drug_targets = build_drug_target_edges(prism_drugs, gdsc_drugs)
    drug_targets.to_csv(OUTPUT_DIR / "drug_target_edges.csv", index=False)

    # ── 7. Per-drug PRISM annotations (MOA, target, phase, disease area) ──
    # One row per compound name, used by the candidate selection rule of
    # generate_paper_results.py (Table 5 and Supplementary Table S1). Written
    # from the raw treatment-info file when it is available; otherwise the
    # shipped copy is kept.
    logger.info("=" * 60)
    logger.info("STEP 7: Per-drug PRISM annotations")
    logger.info("=" * 60)
    treatment_info = DATA_DIR / "PRISM" / "raw" / "secondary-screen-replicate-treatment-info.csv"
    annot_path = OUTPUT_DIR / "prism_drug_annotations.csv"
    if treatment_info.exists():
        annot = pd.read_csv(
            treatment_info,
            usecols=["name", "moa", "target", "phase", "disease.area", "indication"],
        )
        annot = (annot.dropna(subset=["name"])
                      .drop_duplicates(subset="name", keep="first")
                      .rename(columns={"disease.area": "disease_area"})
                      .sort_values("name")
                      .reset_index(drop=True))
        annot.to_csv(annot_path, index=False)
        logger.info("PRISM drug annotations: %d compounds -> %s", len(annot), annot_path)
    elif annot_path.exists():
        logger.info("Raw treatment-info not available, keeping %s", annot_path)
    else:
        logger.warning("Raw treatment-info not available and %s missing: the candidate "
                       "selection rule of generate_paper_results.py will fail", annot_path)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("DepMap expression:     %d cell lines x %d genes", *expression.shape)
    logger.info("PRISM response:        %d cell lines x %d drugs", *prism_response.shape)
    logger.info("GDSC response:         %d cell lines x %d drugs", *gdsc_response.shape)
    logger.info("Shared cell lines:     %d", len(shared_cells))
    logger.info("Shared drugs:          %d", len(shared_drugs))
    logger.info("CollecTRI edges:       %d (TFs: %d)",
                len(collectri), collectri["source"].nunique())
    logger.info("TF activities (PRISM): %d cell lines x %d TFs", *tf_acts_prism.shape)
    logger.info("TF activities (GDSC):  %d cell lines x %d TFs", *tf_acts_gdsc.shape)
    logger.info("OmniPath PPI edges:    %d", len(ppi))
    logger.info("STRING PPI edges:      %s", len(string_ppi) if string_ppi is not None else "not downloaded (server down)")
    logger.info("Drug-target edges:     %d", len(drug_targets))
    logger.info("Output directory:      %s", OUTPUT_DIR.resolve())
    logger.info("DONE")


if __name__ == "__main__":
    main()
