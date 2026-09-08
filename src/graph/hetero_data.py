"""
Convert the PRECISION graph dict to PyTorch Geometric HeteroData.

This bridges build_hetero_graph.py (numpy-based) with PyG models.
"""

import pickle
import torch
import numpy as np
import logging
from pathlib import Path
from torch_geometric.data import HeteroData

from config import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = DATA_DIR / "precision_graph" / "hetero_graph_prism.pkl"


def graph_dict_to_hetero_data(graph: dict) -> HeteroData:
    """Convert PRECISION graph dict to PyG HeteroData object.

    Args:
        graph: Dict from build_hetero_graph.build_graph().

    Returns:
        HeteroData with node features and edge indices/attributes.
    """
    data = HeteroData()

    # ── Node features ─────────────────────────────────────────────────
    for ntype, feat in graph["features"].items():
        data[ntype].x = torch.tensor(feat, dtype=torch.float32)
        data[ntype].num_nodes = feat.shape[0]

    # ── Edge indices and attributes ───────────────────────────────────
    for etype, edge_data in graph["edges"].items():
        ei = torch.tensor(edge_data["edge_index"], dtype=torch.long)
        data[etype].edge_index = ei

        if "edge_attr" in edge_data:
            attr = edge_data["edge_attr"]
            if attr.ndim == 1:
                attr = attr.reshape(-1, 1)
            data[etype].edge_attr = torch.tensor(attr, dtype=torch.float32)

    logger.info("HeteroData created: %d node types, %d edge types",
                len(graph["features"]), len(graph["edges"]))
    for ntype in graph["features"]:
        logger.info("  %s: %d nodes, features=%s",
                    ntype, data[ntype].num_nodes, list(data[ntype].x.shape))
    for etype in graph["edges"]:
        n = data[etype].edge_index.shape[1]
        has_attr = hasattr(data[etype], "edge_attr")
        logger.info("  %s: %d edges%s", etype, n,
                    f" (attr={list(data[etype].edge_attr.shape)})" if has_attr else "")

    return data


def load_hetero_data(path: str | Path = DEFAULT_GRAPH_PATH) -> HeteroData:
    """Load saved graph and convert to HeteroData."""
    path = Path(path)
    logger.info("Loading graph from %s", path)
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph_dict_to_hetero_data(graph), graph["node_maps"]
