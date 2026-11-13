#!/usr/bin/env python3
"""
PRECISION - GNN cell-line embeddings as features of per-drug Ridge models.

Loads the five SAGE models of Table 1 (scripts/57_train_gnns.py, seed-42
cell-line split, initialization seeds 42, 123, 456, 789 and 1011, final-epoch
weights), extracts the 128-dimensional cell-line embeddings of each one and
fits one Ridge model per drug (alpha 100, standardized features) on the train
cell lines of the same split, evaluated on its test cell lines. Cell lines
receive no messages (response edges are outside message passing), so their
embedding is the ReLU of a learned linear projection of their node features.
This script used to train its own SAGE encoder, on which only 947 of the
1,448 drugs reached training because of a drug-name case mismatch.

Rows of embedding_comparison.csv, all on the same split:
1. Ridge(expression_19K)       - 19K raw gene features
2. Ridge(TF_activities_771)    - 771 TF features
3. Ridge(GNN_cell_emb_128)     - 128d cell embeddings of the SAGE model
4. Ridge(expression+GNN_emb)   - both feature sets together (hybrid row of
                                 Table 1)
Rows 1 and 2 do not depend on the GNN and are fitted once (n_seeds 1, empty
*_sd columns). Rows 3 and 4 are fitted on the embeddings of each of the five
models: global_pearson, median_pearson, mean_pearson and pct_gt_03 are the
mean over them, the *_sd columns their sample standard deviation (ddof 1) and
n_seeds is 5.

Outputs in results/v3_embeddings/:
- embedding_comparison.csv: one row per feature set, sorted by median
  per-drug Pearson.
- embedding_comparison_per_seed.csv: rows 3 and 4 for each SAGE model.
- per_drug_<feature set>.csv: per-drug test metrics. For rows 3 and 4 they
  come from the canonical model (sage_split42_init42_full).
- cell_line_embeddings.csv, drug_embeddings.csv: embeddings of the canonical
  model, indexed by cell line and by lowercase drug name.

Run from project root, after scripts/57_train_gnns.py --set table1:
    python scripts/12_gnn_embeddings_perdrug.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR as RESULTS_ROOT  # noqa: E402
from src.data.splits import cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.models import training as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gnn_emb")

RESULTS_DIR = RESULTS_ROOT / "v3_embeddings"
SEED = 42  # split seed of the SAGE runs and of the Ridge models

# SAGE runs of Table 1 (scripts/57_train_gnns.py) on the seed-42 split
INIT_SEEDS = (42, 123, 456, 789, 1011)
CANONICAL_INIT = 42  # sage_split42_init42_full
METRICS = ["global_pearson", "median_pearson", "mean_pearson", "pct_gt_03"]
COLUMNS = ["model", *METRICS, "n_drugs", *(f"{m}_sd" for m in METRICS), "n_seeds"]


def extract_embeddings(model, graph, mp):
    """Cell-line and drug embeddings of a trained model, as DataFrames."""
    with torch.no_grad():
        h_dict = model.encode(graph.x_dict, mp)

    def _frame(ntype, prefix, node_map):
        inv = {v: k for k, v in node_map.items()}
        if sorted(inv) != list(range(len(inv))):
            raise ValueError(f"{ntype} node indices are not 0..{len(inv) - 1}")
        emb = h_dict[ntype].cpu().numpy()
        return pd.DataFrame(emb, index=[inv[i] for i in range(len(inv))],
                            columns=[f"{prefix}_{i}" for i in range(emb.shape[1])])

    return (_frame("cell_line", "gnn_cell", graph.node_maps["cell_line"]),
            _frame("drug", "gnn_drug", graph.node_maps["drug"]))


def run_perdrug_comparison(features_dict, drug_response, train_cells, test_cells,
                           min_samples=30):
    """Run per-drug Ridge on multiple feature sets."""
    results = {}

    for feat_name, features in features_dict.items():
        shared = sorted(set(features.index) & set(drug_response.index))
        feat = features.loc[shared]
        resp = drug_response.loc[shared]

        train_mask = feat.index.isin(train_cells)
        test_mask = feat.index.isin(test_cells)

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            logger.warning("%s: not enough samples (train=%d, test=%d)",
                           feat_name, train_mask.sum(), test_mask.sum())
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(feat[train_mask])
        X_test = scaler.transform(feat[test_mask])

        per_drug_results = []
        all_yt, all_yp = [], []

        for drug in resp.columns:
            y_train = resp.loc[train_mask, drug].values
            y_test = resp.loc[test_mask, drug].values
            tr_ok = ~np.isnan(y_train)
            te_ok = ~np.isnan(y_test)

            if tr_ok.sum() < min_samples or te_ok.sum() < 5:
                continue

            model = Ridge(alpha=100)
            model.fit(X_train[tr_ok], y_train[tr_ok])
            pred = model.predict(X_test[te_ok])

            pr, _ = pearsonr(y_test[te_ok], pred)
            sr, _ = spearmanr(y_test[te_ok], pred)
            per_drug_results.append({
                "drug": drug, "n_samples": te_ok.sum(),
                "pearson_r": pr, "spearman_r": sr,
                "rmse": np.sqrt(np.mean((y_test[te_ok] - pred) ** 2)),
            })
            all_yt.extend(y_test[te_ok])
            all_yp.extend(pred)

        pdf = pd.DataFrame(per_drug_results)
        valid = pdf.dropna(subset=["pearson_r"])
        global_pr, _ = pearsonr(all_yt, all_yp) if all_yt else (0, 0)

        results[feat_name] = {
            "per_drug": pdf,
            "global_pearson": global_pr,
            "median_pearson": valid["pearson_r"].median(),
            "mean_pearson": valid["pearson_r"].mean(),
            "n_drugs": len(valid),
            "pct_gt_03": (valid["pearson_r"] > 0.3).mean() * 100,
        }

        logger.info("%-35s | global=%.4f | median=%.4f | >0.3: %.0f%% | drugs=%d",
                    feat_name, global_pr, valid["pearson_r"].median(),
                    results[feat_name]["pct_gt_03"], len(valid))

    return results


def main():
    # Checked before the Ridge fits so that a missing checkpoint stops the script at once
    runs = [T.run_name("sage", SEED, s) for s in INIT_SEEDS]
    missing = [r for r in runs
               if not all((T.RUNS_DIR / f"{r}{ext}").exists() for ext in (".json", ".pt"))]
    if missing:
        raise FileNotFoundError(f"SAGE runs missing from {T.RUNS_DIR}: {missing}. "
                                "Train them with scripts/57_train_gnns.py --set table1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    expression = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    tf_activities = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0)

    # The split the SAGE runs were trained on
    train_cells, test_cells = cell_line_holdout_split(drug_response, test_frac=T.TEST_FRAC, seed=SEED)

    # ── Feature sets that do not depend on the GNN ────────────────────────
    logger.info("=" * 60)
    logger.info("PER-DRUG RIDGE WITHOUT GNN")
    logger.info("=" * 60)
    fixed = run_perdrug_comparison(
        {"Ridge(expression_19K)": expression, "Ridge(TF_activities_771)": tf_activities},
        drug_response, train_cells, test_cells,
    )

    # ── Embeddings of each trained SAGE model ─────────────────────────────
    dev = T.device()
    graph = T.load_graph(dev)
    per_seed, per_drug = [], {}
    for init_seed, run in zip(INIT_SEEDS, runs):
        logger.info("=" * 60)
        logger.info("PER-DRUG RIDGE ON THE EMBEDDINGS OF %s", run)
        logger.info("=" * 60)
        model, mp, _ = T.load_run("sage", SEED, init_seed, graph)
        cell_emb, drug_emb = extract_embeddings(model, graph, mp)
        del model
        if init_seed == CANONICAL_INIT:
            cell_emb.to_csv(RESULTS_DIR / "cell_line_embeddings.csv")
            drug_emb.to_csv(RESULTS_DIR / "drug_embeddings.csv")
            logger.info("Canonical embeddings: cells %s, drugs %s", cell_emb.shape, drug_emb.shape)

        shared_idx = sorted(set(expression.index) & set(cell_emb.index))
        combined = pd.concat([expression.loc[shared_idx], cell_emb.loc[shared_idx]], axis=1)
        res = run_perdrug_comparison(
            {"Ridge(GNN_cell_emb_128)": cell_emb, "Ridge(expression+GNN_emb)": combined},
            drug_response, train_cells, test_cells,
        )
        for name, r in res.items():
            per_seed.append({"model": name, "run": run, "init_seed": init_seed,
                             **{m: r[m] for m in METRICS}, "n_drugs": r["n_drugs"]})
            if init_seed == CANONICAL_INIT:
                per_drug[name] = r["per_drug"]
    per_seed = pd.DataFrame(per_seed)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)

    summary = []
    for name, r in fixed.items():
        summary.append({"model": name, **{m: r[m] for m in METRICS},
                        "n_drugs": r["n_drugs"], "n_seeds": 1})
        per_drug[name] = r["per_drug"]
    for name, g in per_seed.groupby("model", sort=False):
        if g["n_drugs"].nunique() != 1:
            raise ValueError(f"{name}: the number of drugs changes between SAGE models: "
                             f"{dict(zip(g['run'], g['n_drugs']))}")
        summary.append({"model": name, **{m: g[m].mean() for m in METRICS},
                        "n_drugs": int(g["n_drugs"].iloc[0]),
                        **{f"{m}_sd": g[m].std() for m in METRICS}, "n_seeds": len(g)})

    summary_df = (pd.DataFrame(summary)[COLUMNS]
                  .sort_values("median_pearson", ascending=False, ignore_index=True))
    summary_df.to_csv(RESULTS_DIR / "embedding_comparison.csv", index=False)
    per_seed.to_csv(RESULTS_DIR / "embedding_comparison_per_seed.csv", index=False)
    for name, pdf in per_drug.items():
        pdf.to_csv(RESULTS_DIR / f"per_drug_{name.replace('/', '_')}.csv", index=False)

    for _, row in summary_df.iterrows():
        logger.info("%-35s | global=%.4f (sd %.4f) | median=%.4f (sd %.4f) | >0.3: %.0f%% | seeds=%d",
                    row["model"], row["global_pearson"], row["global_pearson_sd"],
                    row["median_pearson"], row["median_pearson_sd"], row["pct_gt_03"],
                    row["n_seeds"])

    logger.info("DONE. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
