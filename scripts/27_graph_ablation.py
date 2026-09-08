#!/usr/bin/env python3
"""Graph component ablation: train GNN removing one edge type at a time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import torch, numpy as np, pandas as pd, logging
import torch.nn.functional as F
from scipy.stats import pearsonr
from config import DATA_DIR, RESULTS_DIR
from src.data.splits import set_all_seeds, cell_line_holdout_split, split_response_edges
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_all_seeds(42)

try:
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
except FileNotFoundError:
    drug_response = pd.read_csv(DATA_DIR / "PRISM" / "processed" / "drug_response_prism.csv", index_col=0)
    drug_response.columns = [c.strip().lower() for c in drug_response.columns]
data, node_maps = load_hetero_data()
data = data.to(DEVICE)
train_cells, test_cells = cell_line_holdout_split(drug_response, 0.2, 42)
edges = split_response_edges(drug_response, node_maps, train_cells, test_cells, DEVICE)

rk = ("cell_line", "responds_to", "drug")

# Build full MP edges
def build_mp(data, exclude_types=None):
    mp = {}
    if exclude_types is None:
        exclude_types = set()
    for et in data.edge_types:
        if et == rk:
            continue
        # Check if this edge type should be excluded
        skip = False
        for ex in exclude_types:
            if ex in str(et):
                skip = True
        if skip:
            continue
        mp[et] = data[et].edge_index
        s, r, d = et
        rev = (d, f"rev_{r}", s)
        mp[rev] = data[et].edge_index.flip(0)
    return mp

configs = {
    "Full graph": set(),
    "No PPI (gene-gene)": {"interacts"},
    "No CollecTRI (TF-gene)": {"regulates"},
    "No drug-target": {"targets"},
    "No PPI + No drug-target": {"interacts", "targets"},
    "Only response edges (no prior knowledge)": {"interacts", "regulates", "targets"},
}

x_dict = {nt: data[nt].x for nt in data.node_types}
nfd = {nt: data[nt].x.shape[1] for nt in data.node_types}
n_drugs = len(node_maps["drug"])

results = []
for name, exclude in configs.items():
    logging.info("=== %s ===", name)
    mp = build_mp(data, exclude)
    if not mp:
        # No edges at all: skip message passing
        logging.info("  No MP edges, using projection-only model")
        mp_types = []
    else:
        mp_types = list(mp.keys())

    model = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3, dropout=0.33,
        conv_type="sage", edge_types=mp_types if mp_types else None, n_drugs=n_drugs,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1.3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)

    best_pr = -1
    for ep in range(1, 501):
        model.train(); opt.zero_grad()
        pred = model(x_dict, mp, edges["train_cell"], edges["train_drug"])
        F.huber_loss(pred, edges["train_auc"]).backward()
        opt.step(); sch.step()

        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                tp = model(x_dict, mp, edges["test_cell"], edges["test_drug"])
                pr, _ = pearsonr(edges["test_auc"].cpu().numpy(), tp.cpu().numpy())
            best_pr = max(best_pr, pr)

    results.append({"config": name, "test_pearson": best_pr,
                    "n_edge_types": len(mp_types), "excluded": str(exclude)})
    logging.info("  Best Pearson: %.4f", best_pr)

df = pd.DataFrame(results)
out_dir = RESULTS_DIR / "v6_strengthen"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "graph_ablation.csv", index=False)
logging.info("\nABLATION RESULTS:")
for _, r in df.iterrows():
    logging.info("  %-45s Pearson=%.4f", r["config"], r["test_pearson"])
