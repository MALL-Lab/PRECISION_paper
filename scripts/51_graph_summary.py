#!/usr/bin/env python3
"""
PRECISION - Node and edge counts of the prebuilt heterogeneous graph.

Reads data/precision_graph/hetero_graph_prism.pkl (built once by
scripts/02_build_graph.py and never rebuilt, because the OmniPath download
drifts over time) and writes results/graph_summary.csv with the node and edge
counts cited in the manuscript (Methods, Figure 1). generate_paper_results.py
reads that CSV instead of the graph itself, which is not redistributed with
the paper package.

Usage: python scripts/51_graph_summary.py (paths from config.py)
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

GRAPH_PATH = DATA_DIR / "precision_graph" / "hetero_graph_prism.pkl"
OUT_PATH = RESULTS_DIR / "graph_summary.csv"


def main() -> int:
    with open(GRAPH_PATH, "rb") as fh:
        graph = pickle.load(fh)
    node_maps, edges = graph["node_maps"], graph["edges"]

    rows = [{"element": "node", "type": node_type, "count": int(len(mapping)),
             "description": f"{node_type} nodes"}
            for node_type, mapping in node_maps.items()]
    for edge_type, payload in edges.items():
        src, rel, dst = edge_type
        rows.append({"element": "edge", "type": f"{src}_{rel}_{dst}",
                     "count": int(payload["edge_index"].shape[1]),
                     "description": f"{src} -> {dst} ({rel}), directed edge count"})
    # Genes regulated by at least one TF (target side of the CollecTRI edges)
    tf_edges = edges[("tf", "regulates", "gene")]["edge_index"]
    n_targets = len(set(int(v) for v in tf_edges[1]))
    rows.append({"element": "edge_targets", "type": "tf_regulates_gene_targets",
                 "count": n_targets,
                 "description": "distinct gene nodes with at least one TF -> gene edge"})
    # OmniPath lists some pairs in both orders, and the graph mirrors every record,
    # so those directed edges are stored twice. The distinct counts are cited in
    # the Figure 1 caption and in Methods.
    ppi_edges = edges[("gene", "interacts", "gene")]["edge_index"]
    ppi_directed = {(int(a), int(b)) for a, b in zip(ppi_edges[0], ppi_edges[1])}
    rows.append({"element": "edge_distinct", "type": "gene_interacts_gene_distinct_directed",
                 "count": len(ppi_directed),
                 "description": "distinct directed gene -> gene edges"})
    rows.append({"element": "edge_distinct", "type": "gene_interacts_gene_distinct_pairs",
                 "count": len({tuple(sorted(pair)) for pair in ppi_directed}),
                 "description": "distinct unordered gene pairs of the PPI edges"})
    summary = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"\nTotal nodes: {summary[summary.element == 'node']['count'].sum()}, "
          f"total directed edges: {summary[summary.element == 'edge']['count'].sum()}")
    print(f"Written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
