#!/usr/bin/env python3
"""Reference predictors for Table 1 and held-out accuracy of the clinical transfer model.

Both analyses use the seed-42 cell-line hold-out split of Table 1
(src/data/splits.py, 381 training and 95 test cell lines) and its evaluation
universe: drugs with at least 30 training and 5 test cell lines with an AUC, as
in scripts/11_rerun_fair_comparison.py and src/models/training.py.

1. Drug-mean reference. Each test pair (cell line, drug) is predicted with the
   mean AUC of the drug over the training cell lines. The prediction carries no
   information about the cell line, so its global Pearson r shows how much of
   the pooled metric of Table 1 comes from differences between drugs alone.
   Its per-drug r is undefined (constant predictor) and is reported as missing.
   The share of the test-pair variance that lies between drugs is reported too:
   the sum over pairs of (drug mean in test minus grand mean) squared, divided
   by the total sum of squares.
2. Held-out accuracy of the clinical transfer model. The per-drug Ridge of
   scripts/17_fix_all_critical.py (PRISM TF activities restricted to the TFs
   shared with SCAN-B, QuantileTransformer and StandardScaler fitted on the
   training cell lines, Ridge alpha 100 per drug with at least 30 training
   responses) is refitted exactly as there and evaluated on the test cell lines
   of the same split. Script 17 only applies it to SCAN-B.

Both rows also report within_drug_pearson, the pooled Pearson r after centring
the observed and the predicted AUC on the test mean of each drug, which removes
the between-drug component that the drug mean captures. It is missing for the
drug mean, whose centred prediction is zero everywhere.

Reproduction checks, the script raises if one fails:
- the universe holds 1,447 drugs and equals the drug set of script 11
  (results/v2_fair/per_drug_Ridge_expression.csv) and of the canonical SAGE run
  (results/v12_gnn/per_drug/sage_split42_init42_full.csv). Every seed-42 run of
  results/v12_gnn/runs.csv has 1,447 evaluation drugs, and the number of test
  pairs equals n_eval_test_edges of results/v12_gnn/sage_split42_init42_full.json.
- Ridge on expression, refitted with run_baseline of script 11 and the alpha
  of its RIDGE_ALPHA constant (read with ast, so the two scripts cannot
  disagree), gives the global Pearson r of results/v2_fair/fair_comparison.csv
  (the Ridge row of Table 1) and the per-drug r of
  results/v2_fair/per_drug_Ridge_expression.csv.
- the refitted transfer Ridge, applied to SCAN-B, gives
  results/v5_final/scanb_perdrug_predictions.csv for every drug and sample.

Inputs:
  data/PRISM/processed/drug_response_prism.csv, counts_matched_prism.csv
  data/precision_processed/tf_activities_all_prism.csv, tf_activities_scanb.csv
  results/v2_fair/fair_comparison.csv, per_drug_Ridge_expression.csv (script 11)
  results/v12_gnn/runs.csv, sage_split42_init42_full.json and per_drug/ (script 57)
  results/v5_final/scanb_perdrug_predictions.csv (script 17)
Outputs in results/v13_reaudit/:
  reference_baselines.csv: one row per reference model with columns model,
    global_pearson, median_perdrug, pct_gt_03 (percentage of drugs with r above
    0.3), n_drugs, n_test_pairs, within_drug_pearson and
    between_drug_variance_share (drug-mean row only)
  transfer_ridge_heldout_perdrug.csv: drug, pearson_r, n_test, role
    (candidate, positive_control or empty)

Run from project root (a few minutes, most of it the Ridge refit of script 11):
    python scripts/59_reference_baselines.py
"""

import ast
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import QuantileTransformer, StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data.drug_names import normalize_drug_columns  # noqa: E402
from src.data.load_drug_response import load_prism_response  # noqa: E402
from src.data.splits import cell_line_holdout_split, set_all_seeds  # noqa: E402
from src.models import training as T  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reference_baselines")

DATA = DATA_DIR
PROCESSED = DATA / "precision_processed"
RESULTS = RESULTS_DIR
OUT = RESULTS / "v13_reaudit"

SEED = 42                # split seed of Table 1 and random_state of the quantile transform
N_UNIVERSE = 1447        # drugs of Table 1
SCRIPT_11 = ROOT / "scripts" / "11_rerun_fair_comparison.py"
TRANSFER_ALPHA = 100     # per-drug transfer Ridge, script 17
TRANSFER_MIN_TRAIN = 30  # training responses a drug needs in script 17
CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib", "pelitinib",
              "trametinib", "entinostat"]
POSITIVE_CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
R_TOL = 1e-8     # reproduced correlations
PRED_TOL = 1e-8  # reproduced SCAN-B predictions (AUC scale)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def script11_ridge_alpha() -> float:
    """RIDGE_ALPHA of script 11, the alpha of Ridge on expression in Table 1, read with ast."""
    for node in ast.parse(SCRIPT_11.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "RIDGE_ALPHA"):
            return ast.literal_eval(node.value)
    raise ValueError(f"RIDGE_ALPHA not found in {SCRIPT_11}")


def evaluation_universe(resp: pd.DataFrame, train_cells: list, test_cells: list) -> list:
    """Drugs with at least T.MIN_TRAIN_CELLS training and T.MIN_TEST_CELLS test responses."""
    n_train = resp.loc[train_cells].notna().sum()
    n_test = resp.loc[test_cells].notna().sum()
    keep = (n_train >= T.MIN_TRAIN_CELLS) & (n_test >= T.MIN_TEST_CELLS)
    return sorted(resp.columns[keep])


def long_pairs(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Cell line by drug matrix to one row per non-missing (cell, drug) pair."""
    out = frame.rename_axis(index="cell", columns="drug").reset_index()
    out = out.melt(id_vars="cell", var_name="drug", value_name=value)
    return out.dropna(subset=[value]).reset_index(drop=True)


def check_universe(universe: list, n_test_pairs: int) -> None:
    """The universe must be the one of Table 1, in script 11 and in the GNN runs of script 57."""
    if len(universe) != N_UNIVERSE:
        raise ValueError(f"Evaluation universe has {len(universe)} drugs, expected {N_UNIVERSE}")
    s11 = set(pd.read_csv(RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv")["drug"])
    if s11 != set(universe):
        raise ValueError(f"Universe differs from script 11: {len(s11 - set(universe))} drugs only there, "
                         f"{len(set(universe) - s11)} only here")
    runs_path = T.RUNS_DIR / "runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        bad = runs[(runs["split_seed"] == SEED) & (runs["n_eval_drugs"] != len(universe))]
        if len(bad):
            raise ValueError(f"Seed-{SEED} GNN runs with a different universe: "
                             f"{dict(zip(bad['run'], bad['n_eval_drugs']))}")
        canonical = T.run_name("sage", SEED, SEED)
        sage = set(pd.read_csv(T.RUNS_DIR / "per_drug" / f"{canonical}.csv")["drug"])
        if sage != set(universe):
            raise ValueError(f"Universe differs from the drugs of {canonical}")
        sidecar = json.loads((T.RUNS_DIR / f"{canonical}.json").read_text())
        if sidecar["n_eval_test_edges"] != n_test_pairs or sidecar["n_eval_drugs"] != len(universe):
            raise ValueError(f"{canonical} evaluates {sidecar['n_eval_test_edges']} test edges of "
                             f"{sidecar['n_eval_drugs']} drugs, here {n_test_pairs} pairs of {len(universe)}")
        logger.info("Universe matches script 11, %d GNN runs of split %d and %s (%d test pairs)",
                    int((runs["split_seed"] == SEED).sum()), SEED, canonical, n_test_pairs)
    else:
        logger.warning("%s not found, universe checked against script 11 only", runs_path)


def between_drug_share(pairs: pd.DataFrame) -> float:
    y = pairs["auc"]
    drug_mean = pairs.groupby("drug")["auc"].transform("mean")
    return float(((drug_mean - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def within_drug_pearson(pairs: pd.DataFrame) -> float:
    """Pooled r after centring observed and predicted AUC on the test mean of each drug."""
    g = pairs.groupby("drug")
    yt = pairs["auc"] - g["auc"].transform("mean")
    yp = pairs["pred"] - g["pred"].transform("mean")
    return float(pearsonr(yt, yp)[0])


def drug_mean_reference(resp, train_cells, pairs, universe) -> dict:
    train_mean = resp.loc[train_cells, universe].mean()
    pred = pairs["drug"].map(train_mean)
    r = float(pearsonr(pairs["auc"], pred)[0])
    share = between_drug_share(pairs)
    logger.info("Drug mean: global r = %.4f over %d pairs of %d drugs, between-drug variance share = %.4f "
                "(r squared = %.4f)", r, len(pairs), len(universe), share, r ** 2)
    return {"model": "Drug_mean_train", "global_pearson": r, "median_perdrug": np.nan,
            "pct_gt_03": np.nan, "n_drugs": len(universe), "n_test_pairs": len(pairs),
            "within_drug_pearson": np.nan, "between_drug_variance_share": share}


def reproduce_ridge_expression(resp, train_cells, test_cells) -> None:
    """Refit Ridge on expression with run_baseline of script 11 and compare with its outputs."""
    m11 = load_module("s11_rerun_fair_comparison", SCRIPT_11)
    expression = pd.read_csv(DATA / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    alpha = script11_ridge_alpha()
    per_drug, global_r = m11.run_baseline("Ridge_expression", Ridge, {"alpha": alpha},
                                          expression, resp, train_cells, test_cells)
    fair = pd.read_csv(RESULTS / "v2_fair" / "fair_comparison.csv").set_index("model")
    published = float(fair.loc["Ridge_expression", "global_pearson"])
    if abs(global_r - published) > R_TOL:
        raise ValueError(f"Ridge on expression: global r {global_r:.10f}, script 11 gave {published:.10f}")
    ref = pd.read_csv(RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv").set_index("drug")["pearson_r"]
    new = per_drug.set_index("drug")["pearson_r"]
    if set(new.index) != set(ref.index) or not np.allclose(new.loc[ref.index], ref, rtol=0, atol=R_TOL):
        raise ValueError("Ridge on expression: per-drug r differs from results/v2_fair/per_drug_Ridge_expression.csv")
    logger.info("Reproduced Ridge on expression (alpha %s): global r = %.4f (script 11: %.4f), %d drugs",
                alpha, global_r, published, len(new))


def fit_transfer_ridge(resp, train_cells, test_cells):
    """Per-drug Ridge of scripts/17_fix_all_critical.py, lines 78-127, kept step by step.

    Returns (test predictions as test cells by drugs, SCAN-B predictions as samples by drugs).
    """
    tf_activities = pd.read_csv(PROCESSED / "tf_activities_all_prism.csv", index_col=0)
    scanb_tf = pd.read_csv(PROCESSED / "tf_activities_scanb.csv", index_col=0)
    shared_tfs = sorted(set(tf_activities.columns) & set(scanb_tf.columns))
    logger.info("Transfer Ridge: %d shared TFs, %d SCAN-B samples", len(shared_tfs), len(scanb_tf))

    prism_tf = tf_activities[shared_tfs]
    scanb_tf_aligned = scanb_tf[shared_tfs]
    train_prism_tf = prism_tf.loc[[c for c in train_cells if c in prism_tf.index]]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism_tf)
    prism_tf_qt = pd.DataFrame(qt.transform(prism_tf), index=prism_tf.index, columns=shared_tfs)
    scanb_tf_qt = pd.DataFrame(qt.transform(scanb_tf_aligned), index=scanb_tf_aligned.index,
                               columns=shared_tfs)

    train_tf = prism_tf_qt.loc[[c for c in train_cells if c in prism_tf_qt.index]]
    test_tf = prism_tf_qt.loc[[c for c in test_cells if c in prism_tf_qt.index]]
    train_resp = resp.loc[train_tf.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_tf)
    X_test = scaler.transform(test_tf)
    X_scanb = scaler.transform(scanb_tf_qt)

    test_pred, coefs, intercepts, drugs = {}, [], [], []
    for drug in resp.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < TRANSFER_MIN_TRAIN:
            continue
        ridge = Ridge(alpha=TRANSFER_ALPHA)
        ridge.fit(X_train[mask], y[mask])
        test_pred[drug] = ridge.predict(X_test)
        coefs.append(ridge.coef_)
        intercepts.append(ridge.intercept_)
        drugs.append(drug)
    scanb_pred = pd.DataFrame(X_scanb @ np.vstack(coefs).T + np.array(intercepts),
                              index=scanb_tf_qt.index, columns=drugs)
    logger.info("Transfer Ridge fitted for %d drugs on %d training cell lines", len(drugs), len(train_tf))
    return pd.DataFrame(test_pred, index=test_tf.index), scanb_pred


def check_scanb_predictions(scanb_pred: pd.DataFrame) -> None:
    """The refitted transfer Ridge must give the SCAN-B predictions of script 17."""
    pub = pd.read_csv(RESULTS / "v5_final" / "scanb_perdrug_predictions.csv", index_col=0)
    if set(pub.columns) != set(scanb_pred.columns):
        raise ValueError(f"SCAN-B predictions: {len(set(pub.columns) ^ set(scanb_pred.columns))} drugs "
                         "differ from results/v5_final/scanb_perdrug_predictions.csv")
    if not pub.index.astype(str).equals(scanb_pred.index.astype(str)):
        raise ValueError("SCAN-B predictions: samples differ from results/v5_final/scanb_perdrug_predictions.csv")
    diff = np.abs(pub.to_numpy() - scanb_pred[pub.columns].to_numpy())
    if not np.isfinite(diff).all() or diff.max() > PRED_TOL:
        raise ValueError(f"SCAN-B predictions differ from script 17 (max abs diff {np.nanmax(diff):.3e})")
    logger.info("Reproduced script 17 SCAN-B predictions: %d samples x %d drugs, max abs diff %.2e",
                *pub.shape, diff.max())


def transfer_heldout(pairs, test_pred, universe):
    """Per-drug and pooled accuracy of the transfer Ridge on the test cell lines."""
    missing = sorted(set(universe) - set(test_pred.columns))
    if missing:
        raise ValueError(f"{len(missing)} universe drugs have no transfer model: {missing[:10]}")
    pred_long = long_pairs(test_pred[universe], "pred")
    p = pairs.merge(pred_long, on=["cell", "drug"], how="left", validate="one_to_one")
    if p["pred"].isna().any():
        raise ValueError(f"{int(p['pred'].isna().sum())} test pairs have no transfer prediction")
    rows = []
    for drug, g in p.groupby("drug", sort=True):
        rows.append({"drug": drug, "pearson_r": float(pearsonr(g["auc"], g["pred"])[0]), "n_test": len(g)})
    per_drug = pd.DataFrame(rows)
    role = {**{d: "candidate" for d in CANDIDATES}, **{d: "positive_control" for d in POSITIVE_CONTROLS}}
    per_drug["role"] = per_drug["drug"].map(role).fillna("")
    valid = per_drug.dropna(subset=["pearson_r"])
    row = {"model": "Transfer_Ridge_TF", "global_pearson": float(pearsonr(p["auc"], p["pred"])[0]),
           "median_perdrug": float(valid["pearson_r"].median()),
           "pct_gt_03": float((valid["pearson_r"] > 0.3).mean() * 100), "n_drugs": len(valid),
           "n_test_pairs": len(p), "within_drug_pearson": within_drug_pearson(p),
           "between_drug_variance_share": np.nan}
    logger.info("Transfer Ridge held out: global r = %.4f, within-drug r = %.4f, median per-drug r = %.4f, "
                "%.1f%% of %d drugs above 0.3", row["global_pearson"], row["within_drug_pearson"],
                row["median_perdrug"], row["pct_gt_03"], row["n_drugs"])
    for d in CANDIDATES + POSITIVE_CONTROLS:
        r = per_drug.set_index("drug").loc[d]
        logger.info("  %-16s %-16s r = %+.3f (n = %d)", d, r["role"], r["pearson_r"], r["n_test"])
    return row, per_drug


def main():
    os.makedirs(OUT, exist_ok=True)
    set_all_seeds(SEED)

    resp = normalize_drug_columns(load_prism_response(use_processed=True))
    train_cells, test_cells = cell_line_holdout_split(resp, T.TEST_FRAC, SEED)
    logger.info("Split %d: %d train, %d test cell lines", SEED, len(train_cells), len(test_cells))

    universe = evaluation_universe(resp, train_cells, test_cells)
    pairs = long_pairs(resp.loc[test_cells, universe], "auc")
    check_universe(universe, len(pairs))
    reproduce_ridge_expression(resp, train_cells, test_cells)

    rows = [drug_mean_reference(resp, train_cells, pairs, universe)]

    test_pred, scanb_pred = fit_transfer_ridge(resp, train_cells, test_cells)
    check_scanb_predictions(scanb_pred)
    row, per_drug = transfer_heldout(pairs, test_pred, universe)
    rows.append(row)

    summary = pd.DataFrame(rows)[["model", "global_pearson", "median_perdrug", "pct_gt_03", "n_drugs",
                                  "n_test_pairs", "within_drug_pearson", "between_drug_variance_share"]]
    summary.to_csv(OUT / "reference_baselines.csv", index=False)
    per_drug.to_csv(OUT / "transfer_ridge_heldout_perdrug.csv", index=False)
    logger.info("Wrote %s and %s", OUT / "reference_baselines.csv", OUT / "transfer_ridge_heldout_perdrug.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
