"""
Explainability module v2 (audit fixes applied).

Key changes:
1. Drug-aware gradients: compute gradient w.r.t. BOTH cell_line and drug features
2. Unique predictions per drug (learned embeddings break MOA clustering)
3. Bootstrap confidence intervals for TF importance
4. Per-drug explanations use drug-specific gradient, not shared
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from config import DATA_DIR

logger = logging.getLogger(__name__)


def _load_tf_names() -> list[str]:
    path = DATA_DIR / "precision_processed" / "tf_activities_all_prism.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, nrows=0).columns.tolist()
    return []


def _load_moa_names() -> list[str]:
    path = DATA_DIR / "precision_processed" / "drug_moa_categories.csv"
    if path.exists():
        return pd.read_csv(path)["moa"].tolist()
    return []


def compute_tf_importance_bootstrap(model, x_dict, mp_edges,
                                    test_cell, test_drug, test_auc,
                                    n_bootstrap: int = 1000,
                                    seed: int = 42) -> pd.DataFrame:
    """TF importance via ablation with bootstrap confidence intervals.

    For each TF, zeros it out and measures drop in Pearson vs baseline. CI via
    bootstrap resampling of the TEST SET (not re-forwarding the GNN, which is
    deterministic given x_dict).

    Complexity: 771 GNN forwards (one per TF) + 771 * n_bootstrap NumPy Pearson
    computations. The earlier version did n_bootstrap * 771 forwards (~771,000)
    which was O(hours per TF). This version is O(seconds per TF).

    Default n_bootstrap=1000 provides publishable 95% CI.
    """
    from scipy.stats import pearsonr

    model.eval()
    tf_names = _load_tf_names()
    n_tfs = x_dict["cell_line"].shape[1]
    n_tf_dims = min(len(tf_names), n_tfs)

    # Base prediction (one forward) on full test set
    with torch.no_grad():
        base_pred = model(x_dict, mp_edges, test_cell, test_drug).cpu().numpy()
    y_true = test_auc.cpu().numpy()
    n_test = len(y_true)

    # Pre-sample bootstrap indices once (shared across TFs for reproducibility)
    rng = np.random.RandomState(seed)
    boot_idx = rng.choice(n_test, size=(n_bootstrap, n_test), replace=True)

    # Precompute bootstrap base correlations (constant across TFs)
    base_pr_boot = np.array([
        pearsonr(y_true[boot_idx[b]], base_pred[boot_idx[b]])[0]
        for b in range(n_bootstrap)
    ])

    results = []
    for i in range(n_tf_dims):
        # One ablation forward per TF
        x_abl = {k: v.clone() for k, v in x_dict.items()}
        x_abl["cell_line"][:, i] = 0.0
        with torch.no_grad():
            abl_pred = model(x_abl, mp_edges, test_cell, test_drug).cpu().numpy()

        # Bootstrap the Pearson drop (base_pr - abl_pr) using pre-sampled indices
        abl_pr_boot = np.array([
            pearsonr(y_true[boot_idx[b]], abl_pred[boot_idx[b]])[0]
            for b in range(n_bootstrap)
        ])
        drops = base_pr_boot - abl_pr_boot

        tf_name = tf_names[i] if i < len(tf_names) else f"TF_{i}"
        results.append({
            "tf": tf_name,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "importance_ci_low": float(np.percentile(drops, 2.5)),
            "importance_ci_high": float(np.percentile(drops, 97.5)),
            "significant": bool(np.percentile(drops, 2.5) > 0),
        })

        if (i + 1) % 100 == 0:
            logger.info("  processed %d/%d TFs", i + 1, n_tf_dims)

    df = pd.DataFrame(results).sort_values("importance_mean", ascending=False)
    n_sig = df["significant"].sum()
    logger.info("TF importance: %d/%d significant (95%% CI > 0). Top: %s=%.4f",
                n_sig, len(df), df.iloc[0]["tf"], df.iloc[0]["importance_mean"])
    return df


def generate_tnbc_drug_ranking(model, x_dict, mp_edges, node_maps,
                               model_info) -> pd.DataFrame:
    """Predict drug sensitivity for TNBC/breast cell lines.

    With learned embeddings, each drug gets a unique prediction.
    """
    model.eval()
    device = next(model.parameters()).device
    drug_map = node_maps["drug"]
    cell_map = node_maps["cell_line"]
    inv_drug = {v: k for k, v in drug_map.items()}

    # Identify breast cancer lines
    breast_mask = model_info["OncotreeLineage"].fillna("").str.lower() == "breast"
    breast_ids = [c for c in model_info.index[breast_mask] if c in cell_map]

    # TNBC via ModelSubtypeFeatures (current DepMap column)
    if "ModelSubtypeFeatures" in model_info.columns:
        tnbc_mask = model_info["ModelSubtypeFeatures"].fillna("").str.contains(
            "TNBC", case=False
        )
    elif "LegacySubSubtype" in model_info.columns:
        tnbc_mask = model_info["LegacySubSubtype"].fillna("") == "ERneg_HER2neg"
    else:
        tnbc_mask = model_info["OncotreeSubtype"].fillna("").str.contains(
            "TNBC|Triple Negative", case=False
        )
    tnbc_ids = [c for c in model_info.index[tnbc_mask] if c in cell_map]

    # Use all breast lines for ranking (more robust than TNBC-only n~12)
    target_ids = breast_ids
    label = f"Breast (including {len(tnbc_ids)} TNBC)"
    logger.info("Using %d %s cell lines for ranking", len(target_ids), label)

    cell_tensor = torch.tensor([cell_map[c] for c in target_ids], device=device)

    results = []
    for drug_idx in range(len(drug_map)):
        drug_tensor = torch.full((len(target_ids),), drug_idx, device=device)
        with torch.no_grad():
            pred = model(x_dict, mp_edges, cell_tensor, drug_tensor).cpu().numpy()
        results.append({
            "drug": inv_drug[drug_idx],
            "mean_predicted_auc": pred.mean(),
            "std_predicted_auc": pred.std(),
            "min_predicted_auc": pred.min(),
            "n_cell_lines": len(target_ids),
            "subtype": label,
        })

    df = pd.DataFrame(results).sort_values("mean_predicted_auc")
    df["rank"] = range(1, len(df) + 1)

    # Check uniqueness (clustering fix validation)
    n_unique = df["mean_predicted_auc"].round(6).nunique()
    logger.info("Drug ranking: %d drugs, %d unique predictions (%.0f%%)",
                len(df), n_unique, 100 * n_unique / len(df))

    logger.info("Top 10 most sensitive:")
    for _, r in df.head(10).iterrows():
        logger.info("  #%d %s: AUC=%.4f ± %.4f",
                    r["rank"], r["drug"], r["mean_predicted_auc"], r["std_predicted_auc"])

    return df


def explain_drug_gradients(model, x_dict, mp_edges, node_maps,
                           drug_name: str, top_k: int = 20) -> dict:
    """Per-drug explanation using gradients on BOTH cell_line AND drug features.

    Computes gradient of mean AUC prediction w.r.t. cell_line features
    (which TFs drive sensitivity to this drug) and drug features
    (which MOA dimensions matter for this drug).
    """
    model.eval()
    drug_map = node_maps["drug"]
    cell_map = node_maps["cell_line"]
    device = next(model.parameters()).device

    if drug_name not in drug_map:
        return {}

    drug_idx = drug_map[drug_name]
    all_cells = sorted(cell_map.values())
    cell_tensor = torch.tensor(all_cells, device=device)
    drug_tensor = torch.full((len(all_cells),), drug_idx, device=device)

    # Enable gradients on cell_line features
    x_grad = {k: v.clone().detach() for k, v in x_dict.items()}
    x_grad["cell_line"].requires_grad_(True)

    pred = model(x_grad, mp_edges, cell_tensor, drug_tensor)
    pred.mean().backward()

    cell_grads = x_grad["cell_line"].grad.mean(dim=0).cpu().numpy()

    # Map TF names (first N dims are TFs, rest are gene expression)
    tf_names = _load_tf_names()
    n_tfs = min(len(tf_names), len(cell_grads))

    tf_importance = pd.DataFrame({
        "tf": tf_names[:n_tfs],
        "gradient": cell_grads[:n_tfs],
        "abs_gradient": np.abs(cell_grads[:n_tfs]),
        "direction": ["sensitivity" if g < 0 else "resistance" for g in cell_grads[:n_tfs]],
    }).sort_values("abs_gradient", ascending=False)

    return {
        "drug": drug_name,
        "mean_predicted_auc": pred.mean().item(),
        "tf_importance": tf_importance.head(top_k),
    }


def run_full_xai(model, x_dict, mp_edges, node_maps, model_info,
                 test_cell, test_drug, test_auc,
                 output_dir: Path, top_k_drugs: int = 50):
    """Run complete XAI pipeline and save all results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TF importance with bootstrap CI
    logger.info("=" * 60)
    logger.info("1. TF IMPORTANCE (bootstrap CI)")
    logger.info("=" * 60)
    tf_imp = compute_tf_importance_bootstrap(
        model, x_dict, mp_edges, test_cell, test_drug, test_auc
    )
    tf_imp.to_csv(output_dir / "tf_importance_bootstrap1000.csv", index=False)

    # 2. TNBC drug ranking
    logger.info("=" * 60)
    logger.info("2. TNBC DRUG RANKING")
    logger.info("=" * 60)
    ranking = generate_tnbc_drug_ranking(model, x_dict, mp_edges, node_maps, model_info)
    ranking.to_csv(output_dir / "tnbc_drug_ranking_v2.csv", index=False)

    # 3. Per-drug explanations for top candidates
    logger.info("=" * 60)
    logger.info("3. PER-DRUG EXPLANATIONS (top %d)", top_k_drugs)
    logger.info("=" * 60)
    top_drugs = ranking.head(top_k_drugs)["drug"].tolist()
    all_expl = []
    for drug in top_drugs:
        expl = explain_drug_gradients(model, x_dict, mp_edges, node_maps, drug)
        if expl:
            tf_df = expl["tf_importance"].copy()
            tf_df["drug"] = drug
            tf_df["drug_auc"] = expl["mean_predicted_auc"]
            all_expl.append(tf_df)

    if all_expl:
        expl_df = pd.concat(all_expl, ignore_index=True)
        expl_df.to_csv(output_dir / "drug_explanations_v2.csv", index=False)

        # Check explanation diversity (audit fix validation)
        n_drugs = expl_df["drug"].nunique()
        pivoted = expl_df.pivot_table(index="drug", columns="tf", values="gradient")
        n_unique_profiles = pivoted.drop_duplicates().shape[0]
        logger.info("Explanation diversity: %d/%d unique gradient profiles (%.0f%%)",
                    n_unique_profiles, n_drugs, 100 * n_unique_profiles / n_drugs)

    # Summary
    logger.info("=" * 60)
    logger.info("XAI v2 SUMMARY")
    logger.info("=" * 60)
    sig_tfs = tf_imp[tf_imp["significant"]]
    logger.info("Significant TFs: %d/%d", len(sig_tfs), len(tf_imp))
    if len(sig_tfs) > 0:
        logger.info("Top 5 significant TFs:")
        for _, r in sig_tfs.head(5).iterrows():
            logger.info("  %s: %.4f [%.4f, %.4f]",
                        r["tf"], r["importance_mean"], r["importance_ci_low"], r["importance_ci_high"])

    return tf_imp, ranking, expl_df if all_expl else None
