#!/usr/bin/env python3
"""
PRECISION - Table 1 model comparison on the seed-42 cell-line hold-out split.

The Ridge (expression) and random forest (TF activities) baselines are fitted
here, one model per drug, on the train cell lines of the split and evaluated on
its test cell lines. Ridge is deterministic and is fitted once. The random
forest is fitted once per seed of RF_SEEDS (its random_state), the five
initialization seeds of the GNNs. The GNN rows are not trained here. They are
read from the runs of scripts/57_train_gnns.py (src/models/training.py): SAGE
and HGT on the same split with initialization seeds 42, 123, 456, 789 and 1011,
final-epoch weights and all 1,448 drugs in training. This script used to train
one SAGE and one HGT itself, keeping the checkpoint with the lowest test loss,
and only 947 drugs reached training because of a drug-name case mismatch.

Hyperparameters of the baselines (RIDGE_ALPHA and RF_PARAMS). They come from
scripts/72_baseline_hp_search.py, run with the design of
docs/plan_hp_baselines.md. The plan and the script were versioned together in
commit 3d539a0 before the search ran. The search space, the ten trials per model
and the objective (median per-drug Pearson over the 200 drugs with most
non-missing AUCs) are those of the development search. The trials are ten random
draws of optuna's TPESampler with seed 42, and each is scored by 5-fold
cross-validation inside the 381 train cell lines, grouped by DepMap PatientID and
stratified by OncotreeLineage. The input bundle of the search holds the train
lines only. The search ran on CESGA (SLURM job 11133811) and its outputs are in
results/v16_hp_search/cesga_11133811/. Ridge takes the alpha of the best trial
(506.158) rounded to three significant figures, with a cross-validated objective
of 0.1622 +/- 0.0155 (mean and sd over the folds). The random forest takes the
best trial as it is (0.1794 +/- 0.0287). Until October 2026 the values were
alpha 730 and 200 trees, maximum depth 19 and 9 samples per leaf. They were the
best of ten unseeded random trials scored on the 95 test cell lines of this
split (scripts/06_optuna_tuning.py, commit 089d315, deleted in April 2026).

Outputs in results/v2_fair/:
- fair_comparison.csv: one row per model. For RF_TF, SAGE_learned_emb and
  HGT_learned_emb, global_pearson, median_pearson, mean_pearson and pct_gt_0.3
  are the mean over the five seeds, the *_sd columns their sample standard
  deviation (ddof 1) and n_seeds is 5. Ridge is fitted once, so its n_seeds is 1
  and its *_sd columns are empty.
- fair_comparison_per_seed.csv: one row per random forest fit and per GNN run
  behind those means. For the random forest, init_seed is its random_state.
- per_drug_Ridge_expression.csv: per-drug test metrics of Ridge.
- per_drug_RF_TF.csv: per-drug test metrics of the random forest with
  random_state 42, as the GNN tables below are those of their init-42 runs.
- per_drug_SAGE_learned_emb.csv, per_drug_HGT_learned_emb.csv: per-drug test
  metrics of the init-42 runs (sage_split42_init42_full, the canonical model,
  and hgt_split42_init42_full), copied from results/v12_gnn/per_drug/.

Drug names are lowercase everywhere. The script stops if any of the ten GNN
runs is missing from results/v12_gnn/runs.csv, if a GNN run is evaluated on a
different number of drugs than the baselines, or if a random forest fit is
evaluated on different drugs than Ridge.

Run from project root, after scripts/57_train_gnns.py --set table1:
    python scripts/11_rerun_fair_comparison.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RESULTS_DIR as RESULTS_ROOT  # noqa: E402
from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns, normalize_drug_series
from src.data.load_drug_response import load_prism_response
from src.models import training as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fair_comparison")

RESULTS_DIR = RESULTS_ROOT / "v2_fair"
SEED = 42  # split seed of every model

# Baseline hyperparameters, chosen by scripts/72_baseline_hp_search.py (see the
# docstring): best_params_ridge.json and best_params_rf.json of
# results/v16_hp_search/cesga_11133811/
RIDGE_ALPHA = 506
RF_PARAMS = {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 13}
RF_SEEDS = (42, 123, 456, 789, 1011)  # random_state of each random forest fit

# GNN rows of Table 1: runs of scripts/57_train_gnns.py on the seed-42 split
GNN_MODELS = {"SAGE_learned_emb": "sage", "HGT_learned_emb": "hgt"}
INIT_SEEDS = (42, 123, 456, 789, 1011)
PER_DRUG_INIT = 42  # seed of the random forest fit and GNN runs whose per-drug tables are exported
METRICS = ["global_pearson", "median_pearson", "mean_pearson", "pct_gt_0.3"]
COLUMNS = ["model", "global_pearson", "n_drugs", "median_pearson", "mean_pearson", "pct_gt_0.3",
           *(f"{m}_sd" for m in METRICS), "n_seeds"]
PER_SEED_COLUMNS = ["model", "run", "init_seed", "global_pearson", "n_drugs",
                    "median_pearson", "mean_pearson", "pct_gt_0.3"]


def eval_per_drug(y_true, y_pred, drug_names, min_samples=5):
    """Per-drug Pearson/Spearman on test set."""
    results = []
    for drug in sorted(set(drug_names)):
        mask = [d == drug for d in drug_names]
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < min_samples:
            continue
        pr, _ = pearsonr(yt, yp)
        sr, _ = spearmanr(yt, yp)
        results.append({"drug": drug, "n_samples": len(yt),
                        "pearson_r": pr, "spearman_r": sr,
                        "rmse": np.sqrt(np.mean((yt - yp) ** 2))})
    return pd.DataFrame(results)


def run_baseline(name, model_class, model_kwargs, features, drug_response,
                 train_cells, test_cells, min_drug_samples=30):
    """Run a per-drug baseline with cell-line hold-out split."""
    shared = sorted(set(features.index) & set(drug_response.index))
    features = features.loc[shared]
    drug_response = drug_response.loc[shared]

    train_mask = features.index.isin(train_cells)
    test_mask = features.index.isin(test_cells)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_mask])
    X_test = scaler.transform(features[test_mask])

    all_yt, all_yp, all_drugs = [], [], []
    for drug in drug_response.columns:
        y_train = drug_response.loc[train_mask, drug].values
        y_test = drug_response.loc[test_mask, drug].values
        tr_ok = ~np.isnan(y_train)
        te_ok = ~np.isnan(y_test)
        if tr_ok.sum() < min_drug_samples or te_ok.sum() < 5:
            continue
        model = model_class(**model_kwargs)
        model.fit(X_train[tr_ok], y_train[tr_ok])
        pred = model.predict(X_test[te_ok])
        all_yt.extend(y_test[te_ok])
        all_yp.extend(pred)
        all_drugs.extend([drug] * te_ok.sum())

    all_yt, all_yp = np.array(all_yt), np.array(all_yp)
    global_pr, _ = pearsonr(all_yt, all_yp)
    per_drug = eval_per_drug(all_yt, all_yp, all_drugs)

    logger.info("%s: global Pearson=%.4f, median per-drug=%.4f, drugs=%d",
                name, global_pr, per_drug["pearson_r"].median(), len(per_drug))
    return per_drug, global_pr


def baseline_metrics(per_drug, global_r):
    """Table 1 metrics of one baseline fit and the drugs it is evaluated on."""
    valid = per_drug.dropna(subset=["pearson_r"])
    row = {"global_pearson": global_r,
           "n_drugs": len(valid),
           "median_pearson": valid["pearson_r"].median(),
           "mean_pearson": valid["pearson_r"].mean(),
           "pct_gt_0.3": (valid["pearson_r"] > 0.3).mean() * 100}
    return row, set(valid["drug"])


def load_gnn_runs():
    """The ten GNN runs of Table 1 from T.RUNS_DIR/runs.csv, and the per-drug
    tables of the init-42 runs.

    Raises FileNotFoundError without runs.csv, and ValueError if a run is
    missing or listed twice, so that Table 1 is never built from a partial set
    of seeds.
    """
    path = T.RUNS_DIR / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the GNNs with "
                                "scripts/57_train_gnns.py --set table1")
    runs = pd.read_csv(path)
    wanted = {T.run_name(arch, SEED, s): model
              for model, arch in GNN_MODELS.items() for s in INIT_SEEDS}
    sel = runs[runs["run"].isin(list(wanted))]
    missing = sorted(set(wanted) - set(sel["run"]))
    if missing:
        raise ValueError(f"{len(missing)} of the {len(wanted)} Table 1 GNN runs are missing from "
                         f"{path}: {missing}. Train them with scripts/57_train_gnns.py --set table1")
    if sel["run"].duplicated().any():
        raise ValueError(f"Runs listed more than once in {path}: "
                         f"{sorted(sel.loc[sel['run'].duplicated(), 'run'])}")
    sel = sel.assign(model=sel["run"].map(wanted))
    per_drug = {model: pd.read_csv(T.RUNS_DIR / "per_drug" / f"{T.run_name(arch, SEED, PER_DRUG_INIT)}.csv")
                for model, arch in GNN_MODELS.items()}
    return sel, per_drug


def summarize_gnn_runs(sel: pd.DataFrame, per_drug: dict, baseline_drugs: set):
    """Mean and sd over seeds of each GNN model, after checking its drug universe.

    Every run must be evaluated on as many drugs as the baselines, and the
    exported per-drug tables must hold exactly the drugs of the baselines, with
    lowercase names. Returns (summary rows, one row per run).
    """
    n_drugs = len(baseline_drugs)
    bad = sel[sel["n_drugs"] != n_drugs]
    if len(bad):
        raise ValueError(f"GNN runs evaluated on a different number of drugs than the baselines "
                         f"({n_drugs}): {dict(zip(bad['run'], bad['n_drugs']))}")
    for model, df in per_drug.items():
        if not df["drug"].equals(normalize_drug_series(df["drug"])):
            raise ValueError(f"{model}: per-drug table has drug names that are not normalized")
        drugs = set(df.dropna(subset=["pearson_r"])["drug"])
        if drugs != baseline_drugs:
            raise ValueError(f"{model}: {len(drugs - baseline_drugs)} drugs not in the baselines and "
                             f"{len(baseline_drugs - drugs)} baseline drugs missing")

    rows, per_seed = [], []
    for model in GNN_MODELS:
        g = sel[sel["model"] == model].sort_values("init_seed")
        row = {"model": model, "n_drugs": n_drugs, "n_seeds": len(g)}
        for m in METRICS:
            row[m] = g[m].mean()
            row[f"{m}_sd"] = g[m].std()
        rows.append(row)
        per_seed.append(g[PER_SEED_COLUMNS])
    return rows, pd.concat(per_seed, ignore_index=True)


def main():
    # Checked before the baselines so that a missing run stops the script at once
    gnn_runs, gnn_per_drug = load_gnn_runs()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    expression = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    tf_activities = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0)

    # Unified cell-line hold-out split, the one the GNN runs were trained on
    train_cells, test_cells = cell_line_holdout_split(drug_response, test_frac=T.TEST_FRAC, seed=SEED)

    # ── Baselines ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINES (cell-line hold-out)")
    logger.info("=" * 60)

    # Ridge on expression, fitted once
    pd_ridge_expr, gp = run_baseline(
        "Ridge_expression", Ridge, {"alpha": RIDGE_ALPHA},
        expression, drug_response, train_cells, test_cells)
    ridge_row, baseline_drugs = baseline_metrics(pd_ridge_expr, gp)
    summary = [{"model": "Ridge_expression", **ridge_row, "n_seeds": 1}]
    pd_ridge_expr.to_csv(RESULTS_DIR / "per_drug_Ridge_expression.csv", index=False)

    # Random forest on TF activities, one fit per random_state
    rf_per_seed, rf_per_drug = [], {}
    for seed in RF_SEEDS:
        per_drug, gp = run_baseline(
            f"RF_TF random_state {seed}", RandomForestRegressor,
            {**RF_PARAMS, "n_jobs": -1, "random_state": seed},
            tf_activities, drug_response, train_cells, test_cells)
        row, drugs = baseline_metrics(per_drug, gp)
        if drugs != baseline_drugs:
            raise ValueError(f"The random forest with random_state {seed} and Ridge are evaluated on "
                             f"different drugs")
        rf_per_seed.append({"model": "RF_TF", "run": f"rf_tf_split{SEED}_seed{seed}", "init_seed": seed, **row})
        rf_per_drug[seed] = per_drug
    rf_per_seed = pd.DataFrame(rf_per_seed)[PER_SEED_COLUMNS]
    rf_row = {"model": "RF_TF", "n_drugs": len(baseline_drugs), "n_seeds": len(rf_per_seed)}
    for m in METRICS:
        rf_row[m] = rf_per_seed[m].mean()
        rf_row[f"{m}_sd"] = rf_per_seed[m].std()
    summary.append(rf_row)
    rf_per_drug[PER_DRUG_INIT].to_csv(RESULTS_DIR / "per_drug_RF_TF.csv", index=False)

    # ── GNN (trained by scripts/57_train_gnns.py) ─────────────────────────
    gnn_rows, gnn_per_seed = summarize_gnn_runs(gnn_runs, gnn_per_drug, baseline_drugs)
    summary += gnn_rows
    for model, df in gnn_per_drug.items():
        df.to_csv(RESULTS_DIR / f"per_drug_{model}.csv", index=False)
    pd.concat([rf_per_seed, gnn_per_seed], ignore_index=True).to_csv(
        RESULTS_DIR / "fair_comparison_per_seed.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FAIR COMPARISON (cell-line hold-out, same split)")
    logger.info("=" * 60)

    summary = pd.DataFrame(summary)[COLUMNS]
    summary.to_csv(RESULTS_DIR / "fair_comparison.csv", index=False)
    for _, row in summary.iterrows():
        logger.info("%-18s | global=%.4f (sd %.4f) | median=%.4f (sd %.4f) | >0.3: %.1f%% | "
                    "drugs=%d | seeds=%d", row["model"], row["global_pearson"],
                    row["global_pearson_sd"], row["median_pearson"], row["median_pearson_sd"],
                    row["pct_gt_0.3"], row["n_drugs"], row["n_seeds"])

    logger.info("DONE. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
