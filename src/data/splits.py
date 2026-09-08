"""
Unified train/test splitting for fair model comparison.

Uses CELL LINE hold-out (not edge-level) so that:
- A cell line is either fully in train or fully in test
- All models (baselines and GNN) use the exact same split
- No data leakage from shared cell lines across train/test

This fixes the audit issue where baselines split by cell lines
but GNN split by edges, inflating GNN metrics.
"""

import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)

SEED = 42


def set_all_seeds(seed: int = SEED):
    """Set seeds for all random generators (numpy, torch, python)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("All seeds set to %d", seed)


def cell_line_holdout_split(drug_response: pd.DataFrame,
                            test_frac: float = 0.2,
                            seed: int = SEED) -> tuple[list[str], list[str]]:
    """Split cell lines into train/test sets.

    Args:
        drug_response: AUC matrix (index=depmap_id, columns=drugs).
        test_frac: Fraction of cell lines for test.
        seed: Random seed.

    Returns:
        (train_cell_ids, test_cell_ids)
    """
    all_cells = sorted(drug_response.index.tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_cells)

    n_test = max(1, int(len(all_cells) * test_frac))
    test_cells = all_cells[:n_test]
    train_cells = all_cells[n_test:]

    logger.info("Cell line hold-out split: %d train, %d test (%.0f%%)",
                len(train_cells), len(test_cells), 100 * test_frac)
    return train_cells, test_cells


def split_response_edges(drug_response: pd.DataFrame,
                         node_maps: dict,
                         train_cells: list[str],
                         test_cells: list[str],
                         device: torch.device = torch.device("cpu")
                         ) -> dict:
    """Convert cell-line split into train/test edge tensors for GNN.

    Args:
        drug_response: AUC matrix.
        node_maps: Graph node mappings.
        train_cells: Cell line IDs for training.
        test_cells: Cell line IDs for testing.
        device: Torch device.

    Returns:
        Dict with train_cell, train_drug, train_auc, test_cell, test_drug, test_auc.
    """
    cell_map = node_maps["cell_line"]
    drug_map = node_maps["drug"]

    def _extract_edges(cell_ids):
        cells, drugs, aucs = [], [], []
        for cell in cell_ids:
            if cell not in cell_map:
                continue
            for drug in drug_response.columns:
                if drug not in drug_map:
                    continue
                auc = drug_response.loc[cell, drug]
                if pd.notna(auc):
                    cells.append(cell_map[cell])
                    drugs.append(drug_map[drug])
                    aucs.append(auc)
        return (torch.tensor(cells, device=device),
                torch.tensor(drugs, device=device),
                torch.tensor(aucs, dtype=torch.float32, device=device))

    train = _extract_edges(train_cells)
    test = _extract_edges(test_cells)

    logger.info("Train edges: %d, Test edges: %d", len(train[2]), len(test[2]))
    return {
        "train_cell": train[0], "train_drug": train[1], "train_auc": train[2],
        "test_cell": test[0], "test_drug": test[1], "test_auc": test[2],
    }
