#!/usr/bin/env python3
"""
PRECISION - Data constants cited in Methods.

Writes results/data_summary.csv with the sizes of the shared inputs that the
manuscript states and that generate_paper_results.py checks: the PRISM response
matrix (cell lines x drugs), the matched DepMap expression matrix (protein-coding
genes), the GDSC response matrix, the multi-hot MOA encoding of the drug nodes
(annotated drugs and classes, script 02) and the cell-line hold-out split of the
model comparison (scripts 11 and 12, seed 42, 80/20). The generator reads this
CSV instead of the matrices, which are not redistributed with the paper package
(same pattern as 51_graph_summary.py for the graph).

Usage: python scripts/53_data_summary.py (paths from config.py)
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data.drug_names import normalize_drug_columns  # noqa: E402
from src.data.load_drug_response import load_prism_response  # noqa: E402
from src.data.splits import cell_line_holdout_split  # noqa: E402

DATA = DATA_DIR
OUT_PATH = RESULTS_DIR / "data_summary.csv"
TEST_FRAC = 0.2
SEED = 42


def main() -> int:
    resp = normalize_drug_columns(load_prism_response(use_processed=True))
    train, test = cell_line_holdout_split(resp, TEST_FRAC, SEED)
    n_genes = pd.read_csv(DATA / "PRISM" / "processed" / "counts_matched_prism.csv",
                          index_col=0, nrows=0).shape[1]
    gdsc = pd.read_csv(DATA / "GDSC" / "processed" / "drug_response_sanger.csv", index_col=0)
    moa_names = pd.read_csv(DATA / "precision_processed" / "drug_moa_names.csv")
    moa_cats = pd.read_csv(DATA / "precision_processed" / "drug_moa_categories.csv")
    n_tf_features = pd.read_csv(DATA / "precision_processed" / "tf_activities_all_prism.csv",
                                index_col=0, nrows=0).shape[1]
    with open(DATA / "precision_graph" / "hetero_graph_prism.pkl", "rb") as f:
        n_cell_features = pickle.load(f)["features"]["cell_line"].shape[1]
    # METABRIC was downloaded restricted to CollecTRI targets (Methods)
    metabric_genes = set(pyreadr.read_r(str(DATA / "METABRIC" / "metabric_expression.rds"))[None].index)
    snapshot_targets = set(pd.read_csv(DATA / "prior_knowledge" / "collectri_edges.csv")["target"])

    rows = [
        ("matrix", "prism_n_cell_lines", resp.shape[0], "PRISM response matrix: cell lines (rows)"),
        ("matrix", "prism_n_drugs", resp.shape[1], "PRISM response matrix: drugs (columns)"),
        ("matrix", "expression_n_genes", n_genes, "matched DepMap expression matrix: protein-coding genes"),
        ("matrix", "gdsc_n_cell_lines", gdsc.shape[0], "GDSC response matrix: cell lines (rows)"),
        ("matrix", "gdsc_n_drugs", gdsc.shape[1], "GDSC response matrix: drugs (columns)"),
        ("encoding", "moa_n_drugs_annotated", len(moa_names),
         "drugs with a MOA annotation in the multi-hot encoding (script 02)"),
        ("encoding", "moa_n_categories", len(moa_cats),
         "MOA classes of the multi-hot encoding (classes with fewer than 3 drugs removed)"),
        ("split", "holdout_n_train_cells", len(train),
         f"cell lines in the training split (seed {SEED}, test fraction {TEST_FRAC})"),
        ("split", "holdout_n_test_cells", len(test), "cell lines in the hold-out test split"),
        ("matrix", "prism_n_tf_features", n_tf_features, "TF activity features of the PRISM cell lines"),
        ("graph", "cell_line_n_features", n_cell_features,
         "cell-line node features: TF activities plus the top-variable genes"),
        ("matrix", "metabric_n_genes_matrix", len(metabric_genes), "genes of the METABRIC expression matrix"),
        ("matrix", "metabric_n_collectri_targets_covered", len(metabric_genes & snapshot_targets),
         "METABRIC genes that are targets of the CollecTRI snapshot"),
        ("matrix", "collectri_snapshot_n_targets", len(snapshot_targets), "target genes of the CollecTRI snapshot"),
    ]
    df = pd.DataFrame(rows, columns=["element", "type", "count", "description"])
    df["count"] = df["count"].astype(int)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
