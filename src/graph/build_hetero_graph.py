"""
Build the heterogeneous biomedical knowledge graph for PRECISION.

Node types: gene, tf, drug, cell_line
Edge types: tf_regulates_gene, drug_targets_gene, gene_interacts_gene,
            cell_line_responds_drug, cell_line_expresses_gene

The graph is built from:
- CollecTRI (TF → gene regulatory edges)
- STRING PPI (gene to gene interactions)
- PRISM/GDSC drug annotations (drug → gene target edges)
- Drug response matrices (cell_line to drug AUC values)
- DepMap expression (cell_line to gene expression levels)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_node_mappings(collectri: pd.DataFrame,
                         string_ppi: pd.DataFrame,
                         drug_targets: pd.DataFrame,
                         drug_response: pd.DataFrame,
                         expression: pd.DataFrame,
                         tf_activities: pd.DataFrame) -> dict:
    """Create integer ID mappings for each node type.

    Returns:
        Dict with keys 'gene', 'tf', 'drug', 'cell_line', each mapping
        name -> integer index.
    """
    # Genes: union of all gene names across datasets
    genes_from_collectri = set(collectri["target"])
    genes_from_ppi = set(string_ppi["protein1"]) | set(string_ppi["protein2"])
    genes_from_targets = set(drug_targets["target_gene"])
    genes_from_expression = set(expression.columns)
    all_genes = sorted(
        genes_from_collectri | genes_from_ppi | genes_from_targets | genes_from_expression
    )
    gene_map = {g: i for i, g in enumerate(all_genes)}

    # TFs: from CollecTRI sources
    all_tfs = sorted(collectri["source"].unique())
    tf_map = {tf: i for i, tf in enumerate(all_tfs)}

    # Drugs: from response matrix columns
    all_drugs = sorted(drug_response.columns)
    drug_map = {d: i for i, d in enumerate(all_drugs)}

    # Cell lines: from response matrix rows
    all_cells = sorted(drug_response.index)
    cell_map = {c: i for i, c in enumerate(all_cells)}

    logger.info("Node counts: genes: %d, TFs: %d, drugs: %d, cell_lines: %d",
                len(gene_map), len(tf_map), len(drug_map), len(cell_map))

    return {
        "gene": gene_map,
        "tf": tf_map,
        "drug": drug_map,
        "cell_line": cell_map,
    }


def build_edge_indices(collectri: pd.DataFrame,
                       string_ppi: pd.DataFrame,
                       drug_targets: pd.DataFrame,
                       drug_response: pd.DataFrame,
                       expression: pd.DataFrame,
                       node_maps: dict) -> dict:
    """Build edge index tensors for each edge type.

    Returns:
        Dict of edge_type -> (source_indices, target_indices, optional edge_attr).
    """
    edges = {}
    gene_map = node_maps["gene"]
    tf_map = node_maps["tf"]
    drug_map = node_maps["drug"]
    cell_map = node_maps["cell_line"]

    # 1. TF → Gene (regulates) with signed weights
    valid_reg = collectri[
        collectri["source"].isin(tf_map) & collectri["target"].isin(gene_map)
    ]
    src = np.array([tf_map[s] for s in valid_reg["source"]])
    dst = np.array([gene_map[t] for t in valid_reg["target"]])
    weights = valid_reg["weight"].values.astype(np.float32)
    edges[("tf", "regulates", "gene")] = {
        "edge_index": np.stack([src, dst]),
        "edge_attr": weights,
    }
    logger.info("TF→Gene edges: %d", len(src))

    # 2. Gene to Gene (PPI, undirected, stored in both directions)
    valid_ppi = string_ppi[
        string_ppi["protein1"].isin(gene_map) & string_ppi["protein2"].isin(gene_map)
    ]
    src1 = np.array([gene_map[g] for g in valid_ppi["protein1"]])
    dst1 = np.array([gene_map[g] for g in valid_ppi["protein2"]])
    scores = (valid_ppi["combined_score"].values / 1000.0).astype(np.float32)
    # Both directions for undirected
    src_bi = np.concatenate([src1, dst1])
    dst_bi = np.concatenate([dst1, src1])
    scores_bi = np.concatenate([scores, scores])
    edges[("gene", "interacts", "gene")] = {
        "edge_index": np.stack([src_bi, dst_bi]),
        "edge_attr": scores_bi,
    }
    logger.info("Gene-Gene PPI edges: %d (bidirectional)", len(src_bi))

    # 3. Drug → Gene (targets)
    valid_dt = drug_targets[
        drug_targets["drug"].isin(drug_map) & drug_targets["target_gene"].isin(gene_map)
    ]
    src = np.array([drug_map[d] for d in valid_dt["drug"]])
    dst = np.array([gene_map[g] for g in valid_dt["target_gene"]])
    edges[("drug", "targets", "gene")] = {
        "edge_index": np.stack([src, dst]),
    }
    logger.info("Drug→Gene target edges: %d", len(src))

    # 4. Cell line to Drug (response, only non-NaN AUC values)
    response_long = drug_response.stack().reset_index()
    response_long.columns = ["cell_line", "drug", "auc"]
    valid_resp = response_long[
        response_long["cell_line"].isin(cell_map) & response_long["drug"].isin(drug_map)
    ]
    src = np.array([cell_map[c] for c in valid_resp["cell_line"]])
    dst = np.array([drug_map[d] for d in valid_resp["drug"]])
    auc = valid_resp["auc"].values.astype(np.float32)
    edges[("cell_line", "responds_to", "drug")] = {
        "edge_index": np.stack([src, dst]),
        "edge_attr": auc,
    }
    logger.info("Cell-Drug response edges: %d", len(src))

    return edges


def build_node_features(expression: pd.DataFrame,
                        tf_activities: pd.DataFrame,
                        node_maps: dict,
                        drug_features: np.ndarray | None = None,
                        train_cell_ids: list[str] | None = None) -> dict:
    """Build feature matrices for each node type.

    Returns:
        Dict of node_type -> numpy feature matrix.
    """
    features = {}
    gene_map = node_maps["gene"]
    tf_map = node_maps["tf"]
    cell_map = node_maps["cell_line"]

    # Use only train cells for computing gene/TF static features (avoid leakage)
    if train_cell_ids is not None:
        expr_for_stats = expression.loc[[c for c in train_cell_ids if c in expression.index]]
        tf_for_stats = tf_activities.loc[[c for c in train_cell_ids if c in tf_activities.index]]
        logger.info("Node features computed from %d train cell lines only", len(expr_for_stats))
    else:
        expr_for_stats = expression
        tf_for_stats = tf_activities

    # Gene features: mean expression across train cell lines (static embedding)
    gene_names = sorted(gene_map, key=gene_map.get)
    gene_cols = [g for g in gene_names if g in expr_for_stats.columns]
    full_gene_feat = np.zeros((len(gene_map), 1), dtype=np.float32)
    for g in gene_cols:
        full_gene_feat[gene_map[g], 0] = expr_for_stats[g].mean()
    features["gene"] = full_gene_feat
    logger.info("Gene features: %s", full_gene_feat.shape)

    # TF features: mean activity scores across train cell lines
    tf_names = sorted(tf_map, key=tf_map.get)
    tf_feat = np.zeros((len(tf_map), 1), dtype=np.float32)
    for tf in tf_names:
        if tf in tf_for_stats.columns:
            tf_feat[tf_map[tf], 0] = tf_for_stats[tf].mean()
    features["tf"] = tf_feat
    logger.info("TF features: %s", tf_feat.shape)

    # Cell line features: TF activities + top variable genes expression
    cell_names = sorted(cell_map, key=cell_map.get)
    shared_tfs = [tf for tf in tf_activities.columns if tf in tf_map]

    # Select top 500 most variable genes (from train cells only to avoid leakage)
    n_top_genes = 500
    gene_var = expr_for_stats.var().sort_values(ascending=False)
    top_var_genes = [g for g in gene_var.head(n_top_genes).index if g in expression.columns]

    n_tf_feat = len(shared_tfs)
    n_gene_feat = len(top_var_genes)
    cell_feat = np.zeros((len(cell_map), n_tf_feat + n_gene_feat), dtype=np.float32)

    for i, c in enumerate(cell_names):
        if c in tf_activities.index:
            cell_feat[i, :n_tf_feat] = tf_activities.loc[c, shared_tfs].values
        if c in expression.index:
            cell_feat[i, n_tf_feat:] = expression.loc[c, top_var_genes].values

    features["cell_line"] = cell_feat
    logger.info("Cell line features: %s (TF activities + top %d variable genes)",
                cell_feat.shape, n_gene_feat)

    # Drug features: passed as argument or placeholder
    if drug_features is not None:
        features["drug"] = drug_features
        logger.info("Drug features: %s (provided)", drug_features.shape)
    else:
        features["drug"] = np.ones((len(node_maps["drug"]), 1), dtype=np.float32)
        logger.info("Drug features: %s (placeholder)", features["drug"].shape)

    return features


def build_graph(collectri: pd.DataFrame,
                string_ppi: pd.DataFrame,
                drug_targets: pd.DataFrame,
                drug_response: pd.DataFrame,
                expression: pd.DataFrame,
                tf_activities: pd.DataFrame,
                drug_features: np.ndarray | None = None,
                train_cell_ids: list[str] | None = None) -> dict:
    """Build complete heterogeneous graph.

    Returns:
        Dict with keys: node_maps, edges, features, metadata.
    """
    logger.info("Building heterogeneous biomedical knowledge graph")

    node_maps = _build_node_mappings(
        collectri, string_ppi, drug_targets, drug_response, expression, tf_activities
    )

    edges = build_edge_indices(
        collectri, string_ppi, drug_targets, drug_response, expression, node_maps
    )

    features = build_node_features(expression, tf_activities, node_maps, drug_features, train_cell_ids)

    graph = {
        "node_maps": node_maps,
        "edges": edges,
        "features": features,
        "metadata": {
            "n_genes": len(node_maps["gene"]),
            "n_tfs": len(node_maps["tf"]),
            "n_drugs": len(node_maps["drug"]),
            "n_cell_lines": len(node_maps["cell_line"]),
            "edge_types": list(edges.keys()),
        },
    }

    logger.info("Graph built: %d node types, %d edge types",
                len(node_maps), len(edges))
    return graph
