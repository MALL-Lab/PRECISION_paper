#!/usr/bin/env python3
"""
PRECISION Phase 1, Step 2: Build heterogeneous biomedical knowledge graph.

Loads all processed data from step 1 and constructs the graph:
- Node types: gene, tf, drug, cell_line
- Edge types: tf→gene (regulates), gene-gene (PPI), drug→gene (targets),
              cell_line-drug (response)
- Node features: expression profiles, TF activity profiles, placeholders for drugs

Saves the graph as a pickle file for downstream model training.

Run from project root (after 01_prepare_data.py):
    python scripts/02_build_graph.py
"""

import sys
import logging
import pickle
from pathlib import Path

# Anchor imports to the package root regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from config import DATA_DIR

from src.data.load_drug_response import load_prism_response, load_prism_drug_annotations
from src.data.load_prior_knowledge import load_collectri, load_omnipath_ppi
from src.graph.build_hetero_graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_graph")

PROCESSED_DIR = DATA_DIR / "precision_processed"
OUTPUT_DIR = DATA_DIR / "precision_graph"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load processed data ───────────────────────────────────────────────
    logger.info("Loading processed data...")

    collectri = load_collectri(cache=True)

    ppi = load_omnipath_ppi(cache=True)

    drug_targets = pd.read_csv(PROCESSED_DIR / "drug_target_edges.csv")
    logger.info("Drug-target edges: %d", len(drug_targets))

    drug_response = load_prism_response(use_processed=True)

    # Expression matched to PRISM cell lines
    expression = pd.read_csv(
        DATA_DIR / "PRISM" / "processed" / "counts_matched_prism.csv", index_col=0
    )
    logger.info("Expression (PRISM-matched): %d x %d", *expression.shape)

    tf_activities = pd.read_csv(PROCESSED_DIR / "tf_activities_all_prism.csv", index_col=0)
    logger.info("TF activities: %d x %d", *tf_activities.shape)

    # Drug features (MOA encoding)
    from src.data.drug_features import get_drug_feature_matrix
    # We need the drug names in graph order: build_graph creates them sorted from drug_response columns
    drug_names_sorted = sorted(drug_response.columns)
    drug_feat = get_drug_feature_matrix(drug_names_sorted)

    # ── Build graph ───────────────────────────────────────────────────────
    graph = build_graph(
        collectri=collectri,
        string_ppi=ppi,
        drug_targets=drug_targets,
        drug_response=drug_response,
        expression=expression,
        tf_activities=tf_activities,
        drug_features=drug_feat,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "hetero_graph_prism.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Graph saved to %s (%.1f MB)", output_path, file_size_mb)

    # ── Summary ───────────────────────────────────────────────────────────
    meta = graph["metadata"]
    logger.info("=" * 60)
    logger.info("GRAPH SUMMARY")
    logger.info("=" * 60)
    logger.info("Nodes:")
    logger.info("  genes:      %d", meta["n_genes"])
    logger.info("  TFs:        %d", meta["n_tfs"])
    logger.info("  drugs:      %d", meta["n_drugs"])
    logger.info("  cell_lines: %d", meta["n_cell_lines"])
    logger.info("Edges:")
    for etype in meta["edge_types"]:
        edge_data = graph["edges"][etype]
        n_edges = edge_data["edge_index"].shape[1]
        has_attr = "edge_attr" in edge_data
        logger.info("  %s: %d edges%s", etype, n_edges,
                    " (with attr)" if has_attr else "")
    logger.info("Features:")
    for ntype, feat in graph["features"].items():
        logger.info("  %s: %s", ntype, feat.shape)

    # ── Quick sanity checks ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SANITY CHECKS")
    logger.info("=" * 60)

    # Check response edges have valid AUC range
    auc = graph["edges"][("cell_line", "responds_to", "drug")]["edge_attr"]
    logger.info("AUC range: [%.3f, %.3f], mean=%.3f, NaN=%d",
                np.nanmin(auc), np.nanmax(auc), np.nanmean(auc), np.isnan(auc).sum())

    # Check TF regulatory weights are signed
    reg_weights = graph["edges"][("tf", "regulates", "gene")]["edge_attr"]
    n_pos = (reg_weights > 0).sum()
    n_neg = (reg_weights < 0).sum()
    logger.info("TF regulatory edges: %d positive, %d negative", n_pos, n_neg)

    # Check overlap between drug-target genes and expression genes
    gene_map = graph["node_maps"]["gene"]
    dt_genes = set(drug_targets["target_gene"]) & set(gene_map.keys())
    expr_genes = set(expression.columns) & set(gene_map.keys())
    overlap = dt_genes & expr_genes
    logger.info("Drug target genes in expression: %d / %d (%.0f%%)",
                len(overlap), len(dt_genes),
                100 * len(overlap) / len(dt_genes) if dt_genes else 0)

    logger.info("DONE")


if __name__ == "__main__":
    main()
