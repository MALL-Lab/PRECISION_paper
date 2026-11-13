#!/usr/bin/env python3
"""Diagnostics of the Integrated Gradients attributions of the canonical SAGE model.

Script 32 runs Captum IG (Gauss-Legendre, 50 steps) over the whole 1,271-dim
cell-line feature row of the canonical SAGE model (sage_split42_init42_full),
with a baseline x0 equal to the mean row over the 381 training cell lines of
the seed-42 split, and keeps the first 771 dimensions (TF activities). For one
feature k the attribution is

    attr_k = (x_k - x0_k) * g_k

where g_k is the path-averaged gradient of the predicted AUC along the straight
path from x0 to x. The attribution is therefore relative to the baseline. Its
sign is the sign of g_k times the sign of (x_k - x0_k), and only g_k gives the
direction in which higher TF activity moves the predicted AUC. This script
measures how far the two readings diverge on the matrix of Figure 3C and
collects the other IG diagnostics of the re-audit:

(1) Gradient direction versus attribution sign on the top-20 TFs x 11 drugs of
    Figure 3C over the 12 TNBC lines. g_k is recovered as attr_k / (x_k - x0_k)
    (lines with |x_k - x0_k| < 1e-8 are skipped and counted). The division is
    confirmed by recomputing g with autograd at the Gauss-Legendre points of
    Captum for two drugs (erlotinib, a Table 5 candidate, and epirubicin, the
    control with the most distinct attributions) on the 12 lines. That
    recomputation also has to reproduce the attributions of script 32.
(2) Cross-drug similarity: mean pairwise Pearson correlation, over the 11
    drugs, of the 771-TF vectors of mean attribution across the 12 lines.
(3) Split membership of the 12 TNBC IG lines in the seed-42 cell-line split.
(4) Number of TFs in the top 20 of at least one of the three IG seeds of
    script 35.
(5) Seed-42 per-drug median Pearson of SAGE, HGT and the Ridge hybrid
    (expression plus GNN embedding), from the per-drug CSVs the generator
    copies to paper/results.

Reproduction checks (the script raises if any fails): the signed matrix of
Figure 3C and its 220 same-sign cells, the top IG TF and its importance, the
381 training lines of the baseline, the 12 TNBC lines, the multi-seed top-20
overlaps, the Table 1 per-drug medians and the attributions of script 32 for
the recomputed pairs.

Inputs:
  results/v4_xai/ig_full_attributions.csv                  (script 32)
  results/v4_xai/tf_drug_ig_signed.csv                     (script 32)
  results/v4_xai/tf_importance_ig_multi_seed_stability.csv (script 35)
  results/v12_gnn/sage_split42_init42_full.pt and .json    (script 57)
  results/v2_fair/fair_comparison_per_seed.csv,
  results/v2_fair/per_drug_SAGE_learned_emb.csv,
  results/v2_fair/per_drug_HGT_learned_emb.csv             (script 11)
  results/v3_embeddings/embedding_comparison_per_seed.csv,
  results/v3_embeddings/per_drug_Ridge(expression+GNN_emb).csv (script 12)
  data/precision_graph/hetero_graph_prism.pkl, PRISM responses, DepMap Model.csv
Outputs in results/v13_reaudit/:
  ig_gradient_sign.csv         one row per (TF, drug) of Figure 3C: tf, drug,
                               mean_attr, mean_grad, mean_dx, attr_sign,
                               grad_sign, disagree, n_lines (lines with
                               |dx| >= 1e-8 behind mean_grad)
  ig_gradient_sign_by_tf.csv   one row per top-20 TF: mean_dx over the 12
                               lines, drugs with positive and negative mean
                               gradient, majority gradient sign, majority
                               attribution sign, disagreeing drugs
  ig_gradient_check.csv        one row per recomputed (line, drug): agreement
                               of autograd and script 32, ZEB1 gradients
  ig_diagnostics_summary.csv   metric, value

Run from project root:
    python scripts/65_ig_diagnostics.py
"""

import importlib.util
import logging
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import RESULTS_DIR, PAPER_RESULTS  # noqa: E402

import numpy as np
import pandas as pd
import torch
from captum.attr._utils.approximation_methods import approximation_parameters

from src.data.splits import cell_line_holdout_split, set_all_seeds
from src.models import training as T

# Reuse the loader, the TNBC selection and the Captum wrapper of script 32
_spec = importlib.util.spec_from_file_location("script_32", Path(__file__).parent / "32_xai_ig.py")
_mod32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod32)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ig_diagnostics")

XAI_DIR = RESULTS_DIR / "v4_xai"
OUT_DIR = RESULTS_DIR / "v13_reaudit"
FULL_PATH = XAI_DIR / "ig_full_attributions.csv"
SIGNED_PATH = XAI_DIR / "tf_drug_ig_signed.csv"
SIGNED_PAPER_PATH = PAPER_RESULTS / "tf_drug_ig_signed.csv"
STABILITY_PATH = XAI_DIR / "tf_importance_ig_multi_seed_stability.csv"
PER_SEED_GNN = RESULTS_DIR / "v2_fair/fair_comparison_per_seed.csv"
PER_SEED_EMB = RESULTS_DIR / "v3_embeddings/embedding_comparison_per_seed.csv"
PER_DRUG = {
    "SAGE": RESULTS_DIR / "v2_fair/per_drug_SAGE_learned_emb.csv",
    "HGT": RESULTS_DIR / "v2_fair/per_drug_HGT_learned_emb.csv",
    "Ridge_expr_GNN_emb": RESULTS_DIR / "v3_embeddings/per_drug_Ridge(expression+GNN_emb).csv",
}
PER_SEED_KEY = {
    "SAGE": (PER_SEED_GNN, "SAGE_learned_emb"),
    "HGT": (PER_SEED_GNN, "HGT_learned_emb"),
    "Ridge_expr_GNN_emb": (PER_SEED_EMB, "Ridge(expression+GNN_emb)"),
}

SEED = 42
N_STEPS = 50                      # n_steps of script 32
DX_MIN = 1e-8                     # lines with |x - x0| below this are skipped
CHECK_DRUGS = ["erlotinib", "epirubicin"]
CHECK_TF = "ZEB1"
RTOL_ATTR = 1e-3                  # autograd against script 32, relative to max |attr|

# Published values, read once from paper/results/paper_statistics.json
# (2026-09-18) and hard-coded here.
PUBLISHED = {
    "breast_n_tnbc": 12,
    "ig_n_samples_per_tf": 132,
    "ig_top1_name": "RELA",
    "ig_top1_importance": 0.00277,
    "ig_top20_names": ["RELA", "ZEB1", "IRX1", "BARX1", "HOXB7", "ZNF362", "AEBP1", "MZF1",
                       "RUNX2", "ZBTB7B", "NFKBIZ", "HIVEP2", "HOXB9", "SPDEF", "TET1", "LHX1",
                       "JUND", "ARID1A", "E2F1", "HIVEP1"],
    "ig_signed_n_cells": 220,
    "ig_signed_n_cells_same_sign": 220,
    "holdout_n_train_cells": 381,
    "holdout_n_test_cells": 95,
    "ig_multi_seed_top20_all_n": 2,
    "ig_multi_seed_top20_overlap_min": 4,
    "ig_multi_seed_top20_overlap_max": 7,
    "table1_SAGE_learned_emb_median_perdrug": 0.032,
    "table1_HGT_learned_emb_median_perdrug": 0.048,
    "table1_Ridge_expr_GNN_emb_median_perdrug": 0.112,
}


def check(ok: bool, what: str, expected, got) -> None:
    """Raise if a reproduction check fails, log it otherwise."""
    if not ok:
        raise RuntimeError(f"Reproduction check failed: {what}: expected {expected}, got {got}")
    logger.info("Reproduced %s: %s", what, got)


def sign(v):
    return np.sign(v).astype(int)


# =============================================================================
# Attributions, signed matrix and published IG numbers
# =============================================================================
def load_attributions():
    """Long-form attributions of script 32 and the Figure 3C matrix, checked
    against the published numbers."""
    full = pd.read_csv(FULL_PATH)
    check(set(full["seed"]) == {SEED}, "seed of ig_full_attributions.csv", {SEED}, set(full["seed"]))
    n_pairs = full[["cell_depmap_id", "drug"]].drop_duplicates().shape[0]
    check(n_pairs == PUBLISHED["ig_n_samples_per_tf"], "IG (line, drug) pairs",
          PUBLISHED["ig_n_samples_per_tf"], n_pairs)

    glob = (full.assign(a=full["attribution"].abs()).groupby("tf")["a"].mean()
            .sort_values(ascending=False))
    top1 = round(float(glob.iloc[0]), 5)
    check(glob.index[0] == PUBLISHED["ig_top1_name"] and top1 == PUBLISHED["ig_top1_importance"],
          "top IG TF and mean |attr|",
          (PUBLISHED["ig_top1_name"], PUBLISHED["ig_top1_importance"]), (glob.index[0], top1))
    top20 = glob.index[:20].tolist()
    check(top20 == PUBLISHED["ig_top20_names"], "IG top-20 TFs", PUBLISHED["ig_top20_names"], top20)

    signed = pd.read_csv(SIGNED_PATH, comment="#").set_index("tf")
    signed_paper = pd.read_csv(SIGNED_PAPER_PATH, comment="#").set_index("tf")
    drugs = signed.columns.tolist()
    recomputed = (full[full["tf"].isin(top20)].groupby(["tf", "drug"])["attribution"].mean()
                  .unstack("drug").reindex(index=top20, columns=drugs))
    for label, ref in (("results/v4_xai", signed), ("paper/results", signed_paper)):
        diff = float(np.abs(recomputed.values - ref.reindex(index=top20, columns=drugs).values).max())
        check(diff < 1e-12, f"Figure 3C matrix from the full attributions ({label})", "< 1e-12", diff)
    same = int((sign(signed.values) == sign(signed.mean(axis=1).values)[:, None]).sum())
    check(signed.size == PUBLISHED["ig_signed_n_cells"]
          and same == PUBLISHED["ig_signed_n_cells_same_sign"],
          "Figure 3C cells and cells with the sign of their row mean",
          (PUBLISHED["ig_signed_n_cells"], PUBLISHED["ig_signed_n_cells_same_sign"]),
          (signed.size, same))
    return full, top20, drugs


# =============================================================================
# Model, baseline and x - x0
# =============================================================================
def load_model_and_features():
    """Canonical SAGE model, the cell-line features, the mean-train baseline
    (computed as in script 32) and the seed-42 split."""
    sage, x_dict, mp_edges, node_maps, model_info, drug_response, train_cells = \
        _mod32.load_sage_and_data(SEED, SEED)
    cell_map = node_maps["cell_line"]
    cell_feat = x_dict["cell_line"]
    train_idx = torch.tensor([cell_map[c] for c in train_cells if c in cell_map],
                             device=cell_feat.device, dtype=torch.long)
    _, test_cells = cell_line_holdout_split(drug_response, test_frac=T.TEST_FRAC, seed=SEED)
    n_test_graph = sum(c in cell_map for c in test_cells)
    check(len(train_idx) == PUBLISHED["holdout_n_train_cells"]
          and n_test_graph == PUBLISHED["holdout_n_test_cells"],
          "seed-42 split lines in the graph (train, test)",
          (PUBLISHED["holdout_n_train_cells"], PUBLISHED["holdout_n_test_cells"]),
          (len(train_idx), n_test_graph))
    baseline = cell_feat[train_idx].mean(dim=0).detach()
    return (sage, x_dict, mp_edges, node_maps, model_info, drug_response,
            train_cells, test_cells, cell_feat, baseline)


# =============================================================================
# (1) Gradient direction versus attribution sign
# =============================================================================
def gradient_sign_table(full, top20, drugs, dx):
    """Per (TF, drug) of Figure 3C: mean attribution and mean path-averaged
    gradient over the TNBC lines. dx is a DataFrame lines x TFs of x - x0."""
    sub = full[full["tf"].isin(top20)].copy()
    sub["dx"] = [dx.at[c, t] for c, t in zip(sub["cell_depmap_id"], sub["tf"])]
    ok = sub["dx"].abs() >= DX_MIN
    sub["grad"] = np.where(ok, sub["attribution"] / sub["dx"].where(ok, 1.0), np.nan)
    agg = sub.groupby(["tf", "drug"]).agg(mean_attr=("attribution", "mean"),
                                         mean_grad=("grad", "mean"),
                                         n_lines=("grad", "count")).reset_index()
    # (TF, drug) cells whose per-line gradients all share one sign
    per_line_sign = sub.dropna(subset=["grad"]).groupby(["tf", "drug"])["grad"].agg(
        lambda g: len(set(np.sign(g))) == 1)
    n_uniform = int(per_line_sign.sum())
    agg["mean_dx"] = agg["tf"].map(dx[top20].mean())
    agg["attr_sign"] = sign(agg["mean_attr"])
    agg["grad_sign"] = sign(agg["mean_grad"])
    agg["disagree"] = agg["attr_sign"] != agg["grad_sign"]
    agg["tf"] = pd.Categorical(agg["tf"], top20, ordered=True)
    agg["drug"] = pd.Categorical(agg["drug"], drugs, ordered=True)
    agg = agg.sort_values(["tf", "drug"]).reset_index(drop=True)
    agg["tf"] = agg["tf"].astype(str)
    agg["drug"] = agg["drug"].astype(str)
    cols = ["tf", "drug", "mean_attr", "mean_grad", "mean_dx", "attr_sign", "grad_sign",
            "disagree", "n_lines"]
    return agg[cols], int((~ok).sum()), n_uniform


def by_tf_table(sign_tab, top20):
    """Per TF: gradient sign across drugs and the disagreements."""
    rows = []
    for tf in top20:
        t = sign_tab[sign_tab["tf"] == tf]
        n_pos, n_neg = int((t["grad_sign"] > 0).sum()), int((t["grad_sign"] < 0).sum())
        a_pos, a_neg = int((t["attr_sign"] > 0).sum()), int((t["attr_sign"] < 0).sum())
        rows.append({"tf": tf, "mean_dx": float(t["mean_dx"].iloc[0]),
                     "n_drugs_grad_pos": n_pos, "n_drugs_grad_neg": n_neg,
                     "majority_grad_sign": int(np.sign(n_pos - n_neg)),
                     "majority_attr_sign": int(np.sign(a_pos - a_neg)),
                     "n_drugs_disagree": int(t["disagree"].sum())})
    return pd.DataFrame(rows)


def gl_gradients(wrapper, x, x0, n_steps):
    """Path-averaged gradient with the Gauss-Legendre points and weights that
    Captum IntegratedGradients uses by default, plus the local gradients at the
    baseline and at the input and the predictions there."""
    step_fn, alpha_fn = approximation_parameters("gausslegendre")
    diff = x - x0
    total = torch.zeros_like(x)
    for w, a in zip(step_fn(n_steps), alpha_fn(n_steps)):
        pt = (x0 + a * diff).clone().requires_grad_(True)
        g = torch.autograd.grad(wrapper(pt.unsqueeze(0)).sum(), pt)[0]
        total = total + w * g
    ends = {}
    for label, pt in (("x0", x0), ("x", x)):
        pt = pt.clone().requires_grad_(True)
        out = wrapper(pt.unsqueeze(0)).sum()
        ends[label] = (torch.autograd.grad(out, pt)[0], float(out.detach()))
    return total, ends


def gradient_check(sage, x_dict, mp_edges, cell_map, drug_map, cell_feat, baseline,
                   full, tnbc, tf_names):
    """Recompute IG with autograd for CHECK_DRUGS on every TNBC line, compare
    with the attributions of script 32 and with the division estimate."""
    n_tfs = len(tf_names)
    k = tf_names.index(CHECK_TF)
    wide = full.pivot_table(index=["cell_depmap_id", "drug"], columns="tf_index",
                            values="attribution")
    rows = []
    for drug in CHECK_DRUGS:
        for cell in tnbc:
            wrapper = _mod32.GNNCellLineWrapper(sage, mp_edges, x_dict, cell_map[cell],
                                               drug_map[drug], _mod32.DEVICE)
            x = cell_feat[cell_map[cell]].detach()
            g, ends = gl_gradients(wrapper, x, baseline, N_STEPS)
            dx = (x - baseline).cpu().numpy().astype(np.float64)
            g = g.cpu().numpy().astype(np.float64)
            attr_auto = dx[:n_tfs] * g[:n_tfs]
            attr_32 = wide.loc[(cell, drug)].reindex(range(n_tfs)).values
            ok = np.abs(dx[:n_tfs]) >= DX_MIN
            g_div = np.where(ok, attr_32 / np.where(ok, dx[:n_tfs], 1.0), np.nan)
            scale_g = np.nanmax(np.abs(g[:n_tfs][ok]))
            rows.append({
                "cell_depmap_id": cell, "drug": drug,
                "max_abs_attr": float(np.abs(attr_32).max()),
                "attr_max_rel_diff": float(np.abs(attr_auto - attr_32).max() / np.abs(attr_32).max()),
                "grad_max_rel_diff": float(np.nanmax(np.abs(g_div - g[:n_tfs])) / scale_g),
                "grad_sign_agree_all_tfs": int((sign(g_div[ok]) == sign(g[:n_tfs][ok])).sum()),
                "n_tfs_dx_ok": int(ok.sum()),
                "zeb1_dx": dx[k], "zeb1_attr_script32": attr_32[k],
                "zeb1_grad_division": g_div[k], "zeb1_grad_autograd": g[k],
                "zeb1_grad_at_baseline": float(ends["x0"][0][k]),
                "zeb1_grad_at_input": float(ends["x"][0][k]),
                "completeness_sum_attr_all_dims": float((dx * g).sum()),
                "completeness_f_x_minus_f_x0": ends["x"][1] - ends["x0"][1],
                "share_abs_attr_tf_dims": float(np.abs(dx[:n_tfs] * g[:n_tfs]).sum()
                                                / np.abs(dx * g).sum()),
            })
        logger.info("Gradient check done for %s on %d lines", drug, len(tnbc))
    return pd.DataFrame(rows)


# =============================================================================
# (2) Cross-drug similarity, (4) multi-seed union, (5) per-drug medians
# =============================================================================
def cross_drug_similarity(full, drugs):
    vec = full.groupby(["tf", "drug"])["attribution"].mean().unstack("drug")[drugs]
    r = [float(np.corrcoef(vec[a], vec[b])[0, 1]) for a, b in combinations(drugs, 2)]
    return float(np.mean(r)), float(np.min(r)), float(np.max(r)), len(r), vec.shape[0]


def multi_seed_union():
    stab = pd.read_csv(STABILITY_PATH)
    seeds = [c.split("_", 1)[1] for c in stab.columns if c.startswith("rank_")]
    tops = {s: set(stab.nsmallest(20, f"rank_{s}")["tf"]) for s in seeds}
    overlaps = [len(tops[a] & tops[b]) for a, b in combinations(seeds, 2)]
    n_all = int(stab["in_top20_all_seeds"].sum())
    check(n_all == PUBLISHED["ig_multi_seed_top20_all_n"]
          and min(overlaps) == PUBLISHED["ig_multi_seed_top20_overlap_min"]
          and max(overlaps) == PUBLISHED["ig_multi_seed_top20_overlap_max"],
          "multi-seed top-20 (in all seeds, min and max pairwise overlap)",
          (PUBLISHED["ig_multi_seed_top20_all_n"], PUBLISHED["ig_multi_seed_top20_overlap_min"],
           PUBLISHED["ig_multi_seed_top20_overlap_max"]),
          (n_all, min(overlaps), max(overlaps)))
    union = set().union(*tops.values())
    return len(union), sorted(union), seeds


def per_drug_medians():
    out = {}
    for model, path in PER_DRUG.items():
        med = float(pd.read_csv(path)["pearson_r"].median())
        per_seed_path, label = PER_SEED_KEY[model]
        ps = pd.read_csv(per_seed_path)
        ps = ps[ps["model"] == label]
        seed42 = float(ps.loc[ps["run"] == T.run_name("sage" if model != "HGT" else "hgt", SEED, SEED),
                              "median_pearson"].iloc[0])
        check(abs(med - seed42) < 1e-6, f"{model} seed-42 per-drug median ({per_seed_path.name})",
              seed42, med)
        key = {"SAGE": "table1_SAGE_learned_emb_median_perdrug",
               "HGT": "table1_HGT_learned_emb_median_perdrug",
               "Ridge_expr_GNN_emb": "table1_Ridge_expr_GNN_emb_median_perdrug"}[model]
        mean5 = round(float(ps["median_pearson"].mean()), 3)
        check(len(ps) == 5 and mean5 == PUBLISHED[key], f"Table 1 {key} (mean of 5 seeds)",
              PUBLISHED[key], mean5)
        out[model] = med
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    set_all_seeds(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, top20, drugs = load_attributions()
    tf_names = _mod32.load_tf_names()
    check(full.drop_duplicates("tf_index").set_index("tf_index")["tf"].sort_index().tolist()
          == tf_names, "TF order of the attributions", "load_tf_names()", "same order")

    (sage, x_dict, mp_edges, node_maps, model_info, drug_response, train_cells,
     test_cells, cell_feat, baseline) = load_model_and_features()
    cell_map, drug_map = node_maps["cell_line"], node_maps["drug"]
    tnbc = _mod32.identify_tnbc_cells(model_info, drug_response, cell_map)
    ig_cells = full["cell_depmap_id"].drop_duplicates().tolist()
    check(sorted(tnbc) == sorted(ig_cells) and len(tnbc) == PUBLISHED["breast_n_tnbc"],
          "TNBC IG lines", PUBLISHED["breast_n_tnbc"], len(tnbc))

    # x - x0 over the TF dimensions, float64 from the float32 tensors of script 32
    idx = torch.tensor([cell_map[c] for c in tnbc], device=cell_feat.device)
    dx = pd.DataFrame((cell_feat[idx] - baseline).detach().cpu().numpy()[:, :len(tf_names)]
                      .astype(np.float64), index=tnbc, columns=tf_names)

    # (1) gradient direction versus attribution sign
    sign_tab, n_skip_triplets, n_uniform = gradient_sign_table(full, top20, drugs, dx)
    tf_tab = by_tf_table(sign_tab, top20)
    n_small_top20 = int((dx[top20].abs() < DX_MIN).values.sum())
    n_small_all = int((dx.abs() < DX_MIN).values.sum())
    n_dis = int(sign_tab["disagree"].sum())
    n_dis_negdx = int((sign_tab["disagree"] & (sign_tab["mean_dx"] < 0)).sum())
    logger.info("Figure 3C cells with attribution sign != gradient sign: %d of %d "
                "(%d of them with negative mean x - x0)", n_dis, len(sign_tab), n_dis_negdx)

    check_tab = gradient_check(sage, x_dict, mp_edges, cell_map, drug_map, cell_feat, baseline,
                               full, tnbc, tf_names)
    worst = float(check_tab["attr_max_rel_diff"].max())
    check(worst < RTOL_ATTR, "script 32 attributions recomputed with autograd (max rel diff)",
          f"< {RTOL_ATTR}", worst)
    zeb = check_tab
    zeb_agree = int((sign(zeb["zeb1_grad_division"]) == sign(zeb["zeb1_grad_autograd"])).sum())
    check(zeb_agree == len(zeb), f"{CHECK_TF} gradient sign, division against autograd",
          len(zeb), zeb_agree)

    # (2) cross-drug similarity
    r_mean, r_min, r_max, n_pairs, n_tfs = cross_drug_similarity(full, drugs)
    # (3) split membership
    train_set, test_set = set(train_cells), set(test_cells)
    name = model_info["StrippedCellLineName"].to_dict()
    in_train = [c for c in tnbc if c in train_set]
    in_test = [c for c in tnbc if c in test_set]
    if len(in_train) + len(in_test) != len(tnbc):
        raise RuntimeError("Some TNBC IG line is in neither side of the seed-42 split")
    # (4) multi-seed union, (5) per-drug medians
    n_union, union, seeds = multi_seed_union()
    med = per_drug_medians()

    summary = [
        ("n_tnbc_ig_lines", len(tnbc)),
        ("n_drugs", len(drugs)),
        ("n_fig3c_cells", len(sign_tab)),
        ("dx_min_threshold", DX_MIN),
        ("n_tf_line_small_dx_top20", n_small_top20),
        ("n_tf_drug_line_skipped_top20", n_skip_triplets),
        ("n_tf_line_small_dx_all_771", n_small_all),
        ("n_fig3c_cells_attr_sign_ne_grad_sign", n_dis),
        ("n_fig3c_cells_disagree_with_negative_mean_dx", n_dis_negdx),
        ("n_fig3c_cells_grad_sign_same_all_lines", n_uniform),
        ("n_top20_tfs_negative_mean_dx", int((tf_tab["mean_dx"] < 0).sum())),
        ("n_top20_tfs_with_any_disagreement", int((tf_tab["n_drugs_disagree"] > 0).sum())),
        ("top20_tfs_with_disagreement",
         ", ".join(tf_tab.loc[tf_tab["n_drugs_disagree"] > 0, "tf"])),
        ("n_top20_tfs_disagree_all_11_drugs", int((tf_tab["n_drugs_disagree"] == len(drugs)).sum())),
        ("n_top20_tfs_grad_sign_same_all_drugs",
         int(((tf_tab["n_drugs_grad_pos"] == len(drugs)) | (tf_tab["n_drugs_grad_neg"] == len(drugs))).sum())),
        ("n_top20_tfs_majority_grad_positive", int((tf_tab["majority_grad_sign"] > 0).sum())),
        ("n_top20_tfs_majority_grad_negative", int((tf_tab["majority_grad_sign"] < 0).sum())),
        ("top20_tfs_majority_grad_positive",
         ", ".join(tf_tab.loc[tf_tab["majority_grad_sign"] > 0, "tf"])),
        ("top20_tfs_majority_grad_negative",
         ", ".join(tf_tab.loc[tf_tab["majority_grad_sign"] < 0, "tf"])),
        ("n_top20_tfs_majority_grad_ne_majority_attr",
         int((tf_tab["majority_grad_sign"] != tf_tab["majority_attr_sign"]).sum())),
        ("gradcheck_drugs", ", ".join(CHECK_DRUGS)),
        ("gradcheck_n_pairs", len(check_tab)),
        ("gradcheck_attr_max_rel_diff", worst),
        ("gradcheck_grad_max_rel_diff_all_tfs", float(check_tab["grad_max_rel_diff"].max())),
        ("gradcheck_grad_sign_agree_all_tfs",
         f"{int(check_tab['grad_sign_agree_all_tfs'].sum())} of {int(check_tab['n_tfs_dx_ok'].sum())}"),
        (f"gradcheck_{CHECK_TF}_sign_agree_division_autograd", f"{zeb_agree} of {len(zeb)}"),
        (f"gradcheck_{CHECK_TF}_grad_negative_pairs", int((zeb["zeb1_grad_autograd"] < 0).sum())),
        (f"gradcheck_{CHECK_TF}_local_grad_at_input_negative_pairs",
         int((zeb["zeb1_grad_at_input"] < 0).sum())),
        (f"gradcheck_{CHECK_TF}_local_grad_at_baseline_negative_pairs",
         int((zeb["zeb1_grad_at_baseline"] < 0).sum())),
        ("gradcheck_completeness_max_abs_delta",
         float((check_tab["completeness_sum_attr_all_dims"]
                - check_tab["completeness_f_x_minus_f_x0"]).abs().max())),
        ("gradcheck_mean_share_abs_attr_tf_dims", float(check_tab["share_abs_attr_tf_dims"].mean())),
        ("cross_drug_mean_pairwise_pearson", r_mean),
        ("cross_drug_min_pairwise_pearson", r_min),
        ("cross_drug_max_pairwise_pearson", r_max),
        ("cross_drug_n_pairs", n_pairs),
        ("cross_drug_n_tfs", n_tfs),
        ("n_ig_lines_train", len(in_train)),
        ("n_ig_lines_test", len(in_test)),
        ("ig_lines_train", ", ".join(f"{c} ({name.get(c, '?')})" for c in in_train)),
        ("ig_lines_test", ", ".join(f"{c} ({name.get(c, '?')})" for c in in_test)),
        ("ig_multi_seed_seeds", ", ".join(seeds)),
        ("n_tfs_top20_any_seed", n_union),
        ("tfs_top20_any_seed", ", ".join(union)),
        ("seed42_median_pearson_SAGE", med["SAGE"]),
        ("seed42_median_pearson_HGT", med["HGT"]),
        ("seed42_median_pearson_Ridge_expr_GNN_emb", med["Ridge_expr_GNN_emb"]),
    ]
    summary = pd.DataFrame(summary, columns=["metric", "value"])

    sign_tab.to_csv(OUT_DIR / "ig_gradient_sign.csv", index=False)
    tf_tab.to_csv(OUT_DIR / "ig_gradient_sign_by_tf.csv", index=False)
    check_tab.to_csv(OUT_DIR / "ig_gradient_check.csv", index=False)
    summary.to_csv(OUT_DIR / "ig_diagnostics_summary.csv", index=False)
    for _, r in summary.iterrows():
        logger.info("  %s = %s", r["metric"], r["value"])
    logger.info("Per TF:\n%s", tf_tab.to_string(index=False))
    logger.info("Wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
