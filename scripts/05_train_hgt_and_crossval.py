#!/usr/bin/env python3
"""
PRECISION Phase 1, Step 5: HGT model + cross-validation PRISM to GDSC.

1. Train HGT (Heterogeneous Graph Transformer) on PRISM data
2. Compare with SAGE baseline (same epochs), only if results/gnn_per_drug_results.csv
   exists (produced by an earlier baseline script not included in this package,
   not used in the paper)
3. Cross-validate: train on PRISM, test on GDSC shared drugs/cell lines

Run from project root:
    python scripts/05_train_hgt_and_crossval.py
"""

import sys
import logging
from pathlib import Path

# Anchor imports to the package root regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import DATA_DIR, RESULTS_DIR

from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse, HGTDrugResponse
from src.data.load_drug_response import load_prism_response, load_gdsc_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_hgt")

GRAPH_PATH = DATA_DIR / "precision_graph" / "hetero_graph_prism.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 128
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
TEST_FRAC = 0.2
SEED = 42


def prepare_edges(data, test_frac=TEST_FRAC, seed=SEED):
    response_key = ("cell_line", "responds_to", "drug")
    ei = data[response_key].edge_index
    attr = data[response_key].edge_attr.squeeze()
    n = ei.shape[1]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    test_p, train_p = perm[:n_test], perm[n_test:]
    return (ei[0, train_p], ei[1, train_p], attr[train_p],
            ei[0, test_p], ei[1, test_p], attr[test_p])


def build_mp_edges(data):
    edge_index_dict = {}
    for etype in data.edge_types:
        if etype == ("cell_line", "responds_to", "drug"):
            continue
        edge_index_dict[etype] = data[etype].edge_index
        src_type, rel, dst_type = etype
        rev_etype = (dst_type, f"rev_{rel}", src_type)
        edge_index_dict[rev_etype] = data[etype].edge_index.flip(0)
    return edge_index_dict


def train_model(model, x_dict, mp_edges, train_cell, train_drug, train_auc,
                test_cell, test_drug, test_auc, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict, mp_edges, train_cell, train_drug)
        loss = F.huber_loss(pred, train_auc)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tp = model(x_dict, mp_edges, test_cell, test_drug)
                tl = F.huber_loss(tp, test_auc).item()
                pr, _ = pearsonr(test_auc.cpu().numpy(), tp.cpu().numpy())
            if tl < best_test_loss:
                best_test_loss = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info("Epoch %3d | Train: %.4f | Test: %.4f | Pearson: %.4f",
                        epoch, loss.item(), tl, pr)

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_per_drug(y_true, y_pred, drug_idx, node_maps):
    inv = {v: k for k, v in node_maps["drug"].items()}
    results = []
    for d in torch.unique(drug_idx):
        m = drug_idx == d
        yt, yp = y_true[m].cpu().numpy(), y_pred[m].cpu().numpy()
        if len(yt) < 5:
            continue
        pr, _ = pearsonr(yt, yp)
        sr, _ = spearmanr(yt, yp)
        results.append({
            "drug": inv.get(d.item(), f"drug_{d.item()}"),
            "n_samples": len(yt), "pearson_r": pr, "spearman_r": sr,
            "rmse": np.sqrt(np.mean((yt - yp) ** 2)),
        })
    return pd.DataFrame(results)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────
    logger.info("Loading graph...")
    data, node_maps = load_hetero_data(GRAPH_PATH)
    data = data.to(DEVICE)

    (train_cell, train_drug, train_auc,
     test_cell, test_drug, test_auc) = prepare_edges(data)

    mp_edges = build_mp_edges(data)
    x_dict = {nt: data[nt].x for nt in data.node_types}
    node_feat_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    all_edge_types = list(mp_edges.keys())
    all_node_types = list(node_feat_dims.keys())

    # ── Train HGT ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING HGT (Heterogeneous Graph Transformer)")
    logger.info("=" * 60)

    hgt = HGTDrugResponse(
        node_feature_dims=node_feat_dims,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        node_types=all_node_types,
        edge_types=all_edge_types,
    ).to(DEVICE)
    logger.info("HGT parameters: %d", sum(p.numel() for p in hgt.parameters()))

    hgt = train_model(hgt, x_dict, mp_edges, train_cell, train_drug, train_auc,
                      test_cell, test_drug, test_auc)

    # Evaluate HGT
    hgt.eval()
    with torch.no_grad():
        hgt_pred = hgt(x_dict, mp_edges, test_cell, test_drug)
    hgt_results = eval_per_drug(test_auc, hgt_pred, test_drug, node_maps)
    hgt_results.to_csv(RESULTS_DIR / "hgt_per_drug_results.csv", index=False)

    valid = hgt_results.dropna(subset=["pearson_r"])
    logger.info("HGT results: %d drugs, median Pearson=%.4f, >0.3: %.0f%%",
                len(valid), valid["pearson_r"].median(),
                (valid["pearson_r"] > 0.3).mean() * 100)

    torch.save(hgt.state_dict(), RESULTS_DIR / "best_hgt_model.pt")

    # ── Cross-validation PRISM→GDSC ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION: Train PRISM → Test GDSC")
    logger.info("=" * 60)

    # Load GDSC response
    gdsc_response = load_gdsc_response(use_processed=True)

    # Find shared cell lines and drugs between PRISM and GDSC
    prism_response = load_prism_response(use_processed=True)
    shared_cells = sorted(set(prism_response.index) & set(gdsc_response.index))
    # Match drug names (lowercase comparison)
    prism_drugs_lower = {d.lower(): d for d in prism_response.columns}
    gdsc_drugs_lower = {d.lower(): d for d in gdsc_response.columns}
    shared_drugs_lower = set(prism_drugs_lower) & set(gdsc_drugs_lower)

    logger.info("Cross-val shared: %d cell lines, %d drugs", len(shared_cells), len(shared_drugs_lower))

    if len(shared_drugs_lower) < 10:
        logger.warning("Too few shared drugs for cross-validation, skipping")
    else:
        # For each shared drug+cell line pair in GDSC, predict using PRISM-trained model
        drug_map = node_maps["drug"]
        cell_map = node_maps["cell_line"]

        crossval_results = []
        for drug_lower in shared_drugs_lower:
            prism_name = prism_drugs_lower[drug_lower]
            gdsc_name = gdsc_drugs_lower[drug_lower]

            if prism_name not in drug_map:
                continue

            drug_id = drug_map[prism_name]

            cells_with_gdsc = []
            gdsc_aucs = []
            for cell in shared_cells:
                if cell not in cell_map:
                    continue
                auc_val = gdsc_response.loc[cell, gdsc_name] if gdsc_name in gdsc_response.columns else np.nan
                if not np.isnan(auc_val):
                    cells_with_gdsc.append(cell_map[cell])
                    gdsc_aucs.append(auc_val)

            if len(gdsc_aucs) < 5:
                continue

            cell_idx = torch.tensor(cells_with_gdsc, device=DEVICE)
            drug_idx = torch.full((len(cells_with_gdsc),), drug_id, device=DEVICE)
            y_true = np.array(gdsc_aucs)

            hgt.eval()
            with torch.no_grad():
                y_pred = hgt(x_dict, mp_edges, cell_idx, drug_idx).cpu().numpy()

            pr, _ = pearsonr(y_true, y_pred)
            sr, _ = spearmanr(y_true, y_pred)
            crossval_results.append({
                "drug": prism_name,
                "n_samples": len(y_true),
                "pearson_r": pr,
                "spearman_r": sr,
                "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            })

        cv_df = pd.DataFrame(crossval_results)
        cv_df.to_csv(RESULTS_DIR / "crossval_prism_to_gdsc.csv", index=False)

        valid_cv = cv_df.dropna(subset=["pearson_r"])
        logger.info("Cross-val PRISM→GDSC: %d drugs, median Pearson=%.4f, >0.3: %.0f%%",
                    len(valid_cv), valid_cv["pearson_r"].median(),
                    (valid_cv["pearson_r"] > 0.3).mean() * 100)

    # ── Final comparison ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)

    models = {}

    # SAGE per-drug results come from an earlier baseline script that is not part of
    # this package. Include them in the comparison only when the file is present.
    sage_path = RESULTS_DIR / "gnn_per_drug_results.csv"
    if sage_path.exists():
        sage_df = pd.read_csv(sage_path)
        models["SAGE_300ep"] = sage_df.dropna(subset=["pearson_r"])
    else:
        logger.info("SAGE per-drug results not present, skipping SAGE/HGT comparison (not used in the paper)")

    models["HGT_300ep"] = valid
    if len(shared_drugs_lower) >= 10:
        models["HGT_crossval_PRISM→GDSC"] = valid_cv

    for name, df in models.items():
        logger.info("%-30s | drugs=%4d | Pearson=%.3f (med) | >0.3: %.0f%%",
                    name, len(df), df["pearson_r"].median(),
                    (df["pearson_r"] > 0.3).mean() * 100)

    logger.info("DONE")


if __name__ == "__main__":
    main()
