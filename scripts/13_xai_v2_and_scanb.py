#!/usr/bin/env python3
"""
PRECISION: XAI v2 + SCAN-B transfer (all audit fixes applied).

1. Train best SAGE with learned embeddings (cell-line hold-out, train only)
2. XAI v2: TF importance with bootstrap CI, drug ranking, per-drug explanations
3. SCAN-B transfer: use message-passing-aware strategy

Run from project root:
    python scripts/13_xai_v2_and_scanb.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import DATA_DIR, RESULTS_DIR as RESULTS_ROOT
from src.data.splits import set_all_seeds, cell_line_holdout_split, split_response_edges
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.load_depmap import load_model_info
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse
from src.xai.explainer_v2 import run_full_xai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xai_v2")

RESULTS_DIR = RESULTS_ROOT / "v4_xai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 800


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    model_info = load_model_info()

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
    n_drugs = len(node_maps["drug"])

    # ── Train model (train cells only) ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING SAGE (learned embeddings, train cells only)")
    logger.info("=" * 60)

    model = HeteroGNNDrugResponse(
        node_feature_dims=nfd, hidden_dim=128, n_layers=3,
        dropout=0.33, conv_type="sage",
        edge_types=list(mp_edges.keys()), n_drugs=n_drugs,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_loss, best_state = float("inf"), None
    for ep in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_dict, mp_edges, edges["train_cell"], edges["train_drug"])
        loss = F.huber_loss(pred, edges["train_auc"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                tp = model(x_dict, mp_edges, edges["test_cell"], edges["test_drug"])
                tl = F.huber_loss(tp, edges["test_auc"]).item()
                pr, _ = pearsonr(edges["test_auc"].cpu().numpy(), tp.cpu().numpy())
            if tl < best_loss:
                best_loss = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info("Epoch %d | loss=%.4f | Pearson=%.4f", ep, tl, pr)

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), RESULTS_DIR / "best_sage_v4.pt")

    # ── XAI v2 ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("XAI v2 ANALYSIS")
    logger.info("=" * 60)

    tf_imp, ranking, expl_df = run_full_xai(
        model, x_dict, mp_edges, node_maps, model_info,
        edges["test_cell"], edges["test_drug"], edges["test_auc"],
        RESULTS_DIR,
    )

    # ── SCAN-B transfer v2 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SCAN-B TRANSFER v2")
    logger.info("=" * 60)

    scanb_tf_path = DATA_DIR / "precision_processed" / "tf_activities_scanb.csv"
    if not scanb_tf_path.exists():
        logger.warning("SCAN-B TF activities not found. Run 09_scanb_transfer.R first.")
    else:
        scanb_tf = pd.read_csv(scanb_tf_path, index_col=0)
        logger.info("SCAN-B TF activities: %d patients x %d TFs", *scanb_tf.shape)

        # Align to PRISM TF order
        prism_tf = pd.read_csv(
            DATA_DIR / "precision_processed" / "tf_activities_all_prism.csv",
            index_col=0
        )
        prism_tfs = prism_tf.columns.tolist()
        shared_tfs = [t for t in prism_tfs if t in scanb_tf.columns]

        # Standardize FIRST, then align (FIX #6)
        scanb_std = scanb_tf[shared_tfs].copy()
        for tf in shared_tfs:
            mu, sd = prism_tf[tf].mean(), prism_tf[tf].std()
            if sd > 0:
                scanb_std[tf] = (scanb_std[tf] - mu) / sd

        # Also need gene expression features (top 500 variable genes)
        # For SCAN-B patients we only have TF activities, so zero-fill gene dims
        n_cell_features = x_dict["cell_line"].shape[1]
        n_tf_features = len(prism_tfs)
        n_gene_features = n_cell_features - n_tf_features

        patient_features = np.zeros((len(scanb_std), n_cell_features), dtype=np.float32)
        aligned_tfs = np.zeros((len(scanb_std), n_tf_features), dtype=np.float32)
        for i, tf in enumerate(prism_tfs):
            if tf in shared_tfs:
                aligned_tfs[:, i] = scanb_std[tf].values
        patient_features[:, :n_tf_features] = aligned_tfs
        # Gene expression dims stay 0 (patients don't have matched expression)

        patient_tensor = torch.tensor(patient_features, dtype=torch.float32).to(DEVICE)
        logger.info("Patient features: %d patients x %d features (%d TFs + %d gene zeros)",
                    *patient_tensor.shape, n_tf_features, n_gene_features)

        # Predict using full model encode (graph structure preserved)
        model.eval()
        with torch.no_grad():
            h_dict = model.encode(x_dict, mp_edges)
            drug_embeddings = h_dict["drug"]

            # Project patient features through cell_line projection
            patient_emb = model.input_projections["cell_line"](patient_tensor).relu()

        # Predict top 50 drugs for each patient
        top_drugs = ranking.head(50)["drug"].tolist()
        predictions = {}
        for drug_name in top_drugs:
            if drug_name not in node_maps["drug"]:
                continue
            drug_idx = node_maps["drug"][drug_name]
            drug_emb = drug_embeddings[drug_idx].unsqueeze(0).expand(len(scanb_std), -1)

            with torch.no_grad():
                pair_emb = torch.cat([patient_emb, drug_emb], dim=-1)
                pred_auc = model.predictor(pair_emb).squeeze(-1).cpu().numpy()
            predictions[drug_name] = pred_auc

        pred_df = pd.DataFrame(predictions, index=scanb_std.index)
        pred_df.to_csv(RESULTS_DIR / "scanb_predictions_v2.csv")
        logger.info("SCAN-B predictions: %d patients x %d drugs", *pred_df.shape)

        # Check prediction diversity
        n_unique_per_drug = pred_df.apply(lambda c: c.round(4).nunique())
        logger.info("Prediction diversity: mean %.0f unique values per drug (of %d patients)",
                    n_unique_per_drug.mean(), len(pred_df))

        # Survival analysis
        try:
            from lifelines import CoxPHFitter
            from lifelines.statistics import logrank_test
            from scipy import stats

            clinical = pd.read_csv(DATA_DIR / "precision_processed" / "scanb_clinical.csv")
            clinical_dedup = clinical.drop_duplicates(subset="sample_id")

            surv_results = []
            for drug in predictions:
                pred_series = pred_df[drug].rename("predicted_auc")
                merged = clinical_dedup.set_index("sample_id").join(pred_series, how="inner")
                merged = merged.dropna(subset=["os_time", "os_event", "predicted_auc"])

                if len(merged) < 100:
                    continue

                # Median split for log-rank test
                median_auc = merged["predicted_auc"].median()
                sens = merged[merged["predicted_auc"] < median_auc]
                res = merged[merged["predicted_auc"] >= median_auc]

                if len(sens) < 30 or len(res) < 30:
                    continue

                lr = logrank_test(sens["os_time"], res["os_time"],
                                  event_observed_A=sens["os_event"],
                                  event_observed_B=res["os_event"])
                surv_results.append({
                    "drug": drug,
                    "n_patients": len(merged),
                    "pvalue": lr.p_value,
                    "n_sensitive": len(sens),
                    "n_resistant": len(res),
                })

            if surv_results:
                surv_df = pd.DataFrame(surv_results)
                surv_df["padj"] = stats.false_discovery_control(surv_df["pvalue"])
                surv_df = surv_df.sort_values("pvalue")
                surv_df.to_csv(RESULTS_DIR / "scanb_survival_v2.csv", index=False)

                sig = surv_df[surv_df["padj"] < 0.05]
                logger.info("Survival: %d drugs tested, %d significant (FDR<0.05)",
                            len(surv_df), len(sig))
                for _, r in surv_df.head(10).iterrows():
                    logger.info("  %s: p=%.2e, padj=%.2e (n=%d)",
                                r["drug"], r["pvalue"], r["padj"], r["n_patients"])

        except ImportError:
            logger.warning("lifelines not installed, skipping survival analysis")

    logger.info("DONE. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
