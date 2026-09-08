#!/usr/bin/env python3
"""
Training dynamics of the SAGE GNN (Supplementary Figure on training curves).

Retrains the model with exactly the protocol of scripts/13_xai_v2_and_scanb.py
(cell-line hold-out split, seed 42, 3 SAGE layers, 128 hidden units, dropout
0.33, Adam lr 1.3e-3, cosine annealing, Huber loss, 800 epochs) and records,
at every epoch, the training loss, the test loss and the test Pearson
correlation. The checkpoint itself is not saved: the canonical weights are the
ones produced by script 13 (best_sage_v4.pt).

Output:
  results/gnn_training_history.csv   columns epoch, train_loss, test_loss, test_pearson

Usage: python scripts/50_training_history.py (paths from config.py)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds, split_response_edges
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import RESULTS_DIR  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 800
LR = 1.3e-3


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    data, node_maps = load_hetero_data()
    data = data.to(DEVICE)
    train_cells, test_cells = cell_line_holdout_split(drug_response, test_frac=0.2, seed=SEED)
    edges = split_response_edges(drug_response, node_maps, train_cells, test_cells, DEVICE)

    rk = ("cell_line", "responds_to", "drug")
    mp_edges = {}
    for et in data.edge_types:
        if et == rk:
            continue
        mp_edges[et] = data[et].edge_index
        s, r, d = et
        mp_edges[(d, f"rev_{r}", s)] = data[et].edge_index.flip(0)
    x_dict = {nt: data[nt].x for nt in data.node_types}
    nfd = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3, dropout=0.33,
        conv_type="sage", edge_types=list(mp_edges.keys()), n_drugs=len(node_maps["drug"]),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    logger.info("Training SAGE on %s: %d train / %d test cell lines, %d epochs",
                DEVICE, len(train_cells), len(test_cells), EPOCHS)

    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict, mp_edges, edges["train_cell"], edges["train_drug"])
        loss = F.huber_loss(pred, edges["train_auc"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tp = model(x_dict, mp_edges, edges["test_cell"], edges["test_drug"])
            tl = F.huber_loss(tp, edges["test_auc"]).item()
            pr, _ = pearsonr(edges["test_auc"].cpu().numpy(), tp.cpu().numpy())
        history.append({"epoch": ep, "train_loss": float(loss.item()),
                        "test_loss": float(tl), "test_pearson": float(pr)})
        if ep % 100 == 0 or ep == 1:
            logger.info("Epoch %4d | train loss %.4f | test loss %.4f | test Pearson %.4f",
                        ep, loss.item(), tl, pr)

    df = pd.DataFrame(history)
    out = RESULTS_DIR / "gnn_training_history.csv"
    df.to_csv(out, index=False)
    best = df.loc[df["test_pearson"].idxmax()]
    logger.info("Saved %s. Final test Pearson %.4f, best %.4f at epoch %d",
                out, df["test_pearson"].iloc[-1], best["test_pearson"], int(best["epoch"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
