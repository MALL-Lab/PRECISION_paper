#!/usr/bin/env python3
"""
PRECISION - Hyperparameter search of the Table 1 baselines with internal validation.

Design declared in docs/plan_hp_baselines.md before running. The Ridge
(expression) and random forest (TF activities) hyperparameters of Table 1 were
chosen during development on the 95 test cell lines of the seed-42 split (legacy
scripts/06_optuna_tuning.py, commit 089d315, ten unseeded random trials). This
script repeats that search with one change: each trial is scored by 5-fold
cross-validation inside the 381 train cell lines, grouped by DepMap PatientID and
stratified by OncotreeLineage. The test cell lines never leave the machine that
runs `prepare`: the input bundle holds the train lines only.

Subcommands:
    prepare  (local)  write results/v16_hp_search/inputs/: train-only response,
             expression and TF matrices, folds.csv and manifest.json (sha256 of
             each file, the test cell IDs to exclude, environment).
    search   (CESGA)  run the search for one model on that bundle and write
             trials_<model>.csv, cv_perdrug_<model>.csv and best_params_<model>.json.

Kept from the original search: search space, ten trials, objective (median
per-drug Pearson over the 200 drugs with most non-missing AUCs), StandardScaler
and the per-drug minimum of 30 fitting and 5 evaluation cell lines. The drug
completeness is counted on the train lines only, with ties broken alphabetically.

Run from project root:
    python scripts/72_baseline_hp_search.py prepare
    python scripts/72_baseline_hp_search.py search --model ridge --n-jobs 64
    python scripts/72_baseline_hp_search.py search --model rf --n-jobs 64
"""

import sys
import json
import time
import hashlib
import logging
import argparse
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("baseline_hp_search")

OUT_DIR = Path("results/v16_hp_search")
INPUT_DIR = OUT_DIR / "inputs"
DATA_DIR = Path("data")
SEED = 42            # split seed, fold seed, sampler seed and random_state of the RF search
N_FOLDS = 5
N_TRIALS = 10        # as in the original search
N_OBJECTIVE_DRUGS = 200
MIN_FIT, MIN_EVAL = 30, 5   # per-drug minimum of fitting and evaluation cell lines, as in script 11
INPUT_FILES = ("response_train.csv.gz", "expression_train.csv.gz", "tf_train.csv.gz", "folds.csv")

# Original search space (scripts/06_optuna_tuning.py@089d315, L105 and L125-127)
RIDGE_ALPHA = (1e-3, 1e3)
RF_N_ESTIMATORS = (50, 300, 50)
RF_MAX_DEPTH = (3, 20)
RF_MIN_SAMPLES_LEAF = (2, 20)

# Edge of the range (outer 10%) and the point evaluated beyond it, as a diagnostic only
EDGE_CHECKS = {
    "alpha": [(lambda v: v > 10 ** 2.4, 10 ** 3.5), (lambda v: v < 10 ** -2.4, 10 ** -3.5)],
    "max_depth": [(lambda v: v >= 19, 25)],
    "min_samples_leaf": [(lambda v: v <= 3, 1)],
    "n_estimators": [(lambda v: v == 300, 400)],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    import sklearn
    import scipy
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "host": platform.node()}
    try:
        import optuna
        out["optuna"] = optuna.__version__
    except ImportError:
        out["optuna"] = None
    return out


# ── prepare (local) ───────────────────────────────────────────────────────

def prepare():
    import subprocess
    from sklearn.model_selection import StratifiedGroupKFold
    from src.data.splits import cell_line_holdout_split
    from src.data.drug_names import normalize_drug_columns
    from src.data.load_drug_response import load_prism_response
    from src.models import training as T

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    expression = pd.read_csv(DATA_DIR / "PRISM/processed/counts_matched_prism.csv", index_col=0)
    tf_activities = pd.read_csv(DATA_DIR / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    train_cells, test_cells = cell_line_holdout_split(drug_response, test_frac=T.TEST_FRAC, seed=SEED)
    if (len(train_cells), len(test_cells)) != (381, 95):
        raise ValueError(f"Unexpected split: {len(train_cells)} train, {len(test_cells)} test (expected 381, 95)")

    # Same restriction as run_baseline of script 11: cell lines with features and response
    for name, feats in (("expression", expression), ("tf", tf_activities)):
        missing = sorted(set(train_cells) - set(feats.index))
        if missing:
            raise ValueError(f"{len(missing)} train cell lines have no {name} features: {missing[:5]}")
    train = sorted(train_cells)

    models = pd.read_csv(DATA_DIR / "DepMap/Model.csv", usecols=["ModelID", "PatientID", "OncotreeLineage"])
    meta = models.set_index("ModelID").reindex(train)
    if meta.isna().any().any():
        raise ValueError(f"Train cell lines without PatientID or OncotreeLineage: "
                         f"{meta[meta.isna().any(axis=1)].index.tolist()}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=train, name="fold")
    for k, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(train)), meta["OncotreeLineage"], meta["PatientID"])):
        fold.iloc[val_idx] = k
    folds = meta.assign(fold=fold).rename_axis("cell_line").reset_index()
    if (folds.groupby("PatientID")["fold"].nunique() > 1).any():
        raise ValueError("A patient is split across folds")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    drug_response.loc[train].to_csv(INPUT_DIR / "response_train.csv.gz")
    expression.loc[train].to_csv(INPUT_DIR / "expression_train.csv.gz")
    tf_activities.loc[train].to_csv(INPUT_DIR / "tf_train.csv.gz")
    folds.to_csv(INPUT_DIR / "folds.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    manifest = {
        "created_by": "scripts/72_baseline_hp_search.py prepare",
        "git_commit": commit, "git_dirty": dirty,
        "n_train": len(train), "n_test": len(test_cells),
        "test_cell_ids": sorted(test_cells),
        "n_drugs": int(drug_response.shape[1]),
        "n_expression_features": int(expression.shape[1]), "n_tf_features": int(tf_activities.shape[1]),
        "fold_sizes": folds["fold"].value_counts().sort_index().tolist(),
        "sha256": {f: sha256(INPUT_DIR / f) for f in INPUT_FILES},
        "environment": versions(),
    }
    (INPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Bundle written to %s: %d train lines, folds %s", INPUT_DIR, len(train), manifest["fold_sizes"])


# ── search (CESGA) ────────────────────────────────────────────────────────

def load_bundle(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    for f, digest in manifest["sha256"].items():
        if sha256(input_dir / f) != digest:
            raise ValueError(f"{f} does not match its sha256 in the manifest")
    response = pd.read_csv(input_dir / "response_train.csv.gz", index_col=0)
    expression = pd.read_csv(input_dir / "expression_train.csv.gz", index_col=0)
    tf = pd.read_csv(input_dir / "tf_train.csv.gz", index_col=0)
    folds = pd.read_csv(input_dir / "folds.csv").set_index("cell_line")

    cells = set(response.index)
    leaked = cells & set(manifest["test_cell_ids"])
    if leaked:
        raise ValueError(f"{len(leaked)} test cell lines are in the bundle")
    if not (cells == set(expression.index) == set(tf.index) == set(folds.index)):
        raise ValueError("The matrices and folds.csv cover different cell lines")
    if (folds.groupby("PatientID")["fold"].nunique() > 1).any():
        raise ValueError("A patient is split across folds")
    return manifest, response, expression, tf, folds


def objective_drugs(response, n):
    counts = response.notna().sum()
    order = sorted(counts.index, key=lambda d: (-counts[d], d))
    return order[:n]


def make_model(model, params, n_jobs=1):
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params["alpha"])
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                 min_samples_leaf=params["min_samples_leaf"],
                                 n_jobs=n_jobs, random_state=SEED)


def fit_one(model, params, X_fit, y_fit, X_val, y_val):
    from scipy.stats import pearsonr
    fit_ok, val_ok = ~np.isnan(y_fit), ~np.isnan(y_val)
    n_fit, n_val = int(fit_ok.sum()), int(val_ok.sum())
    if n_fit < MIN_FIT or n_val < MIN_EVAL:
        return np.nan, n_fit, n_val, "too_few"
    m = make_model(model, params)
    m.fit(X_fit[fit_ok], y_fit[fit_ok])
    pred = m.predict(X_val[val_ok])
    yv = y_val[val_ok]
    if np.std(pred) == 0 or np.std(yv) == 0:
        return np.nan, n_fit, n_val, "constant"
    return float(pearsonr(yv, pred)[0]), n_fit, n_val, "ok"


def evaluate(model, params, features, response, folds, drugs, n_jobs):
    """Mean over folds of the median per-drug Pearson in the validation fold."""
    from joblib import Parallel, delayed
    from sklearn.preprocessing import StandardScaler
    rows, medians = [], []
    for k in range(N_FOLDS):
        fit_cells = folds.index[folds["fold"] != k]
        val_cells = folds.index[folds["fold"] == k]
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(features.loc[fit_cells])
        X_val = scaler.transform(features.loc[val_cells])
        Y_fit = response.loc[fit_cells, drugs].to_numpy()
        Y_val = response.loc[val_cells, drugs].to_numpy()
        res = Parallel(n_jobs=n_jobs)(delayed(fit_one)(model, params, X_fit, Y_fit[:, j], X_val, Y_val[:, j])
                                      for j in range(len(drugs)))
        r = np.array([x[0] for x in res], dtype=float)
        medians.append(float(np.nanmedian(r)))
        rows += [{"fold": k, "drug": d, "pearson": x[0], "n_fit": x[1], "n_val": x[2], "status": x[3]}
                 for d, x in zip(drugs, res)]
    return float(np.mean(medians)), float(np.std(medians, ddof=1)), medians, rows


def suggest(trial, model):
    # Same calls, bounds and order as the original search
    if model == "ridge":
        return {"alpha": trial.suggest_float("alpha", *RIDGE_ALPHA, log=True)}
    lo, hi, step = RF_N_ESTIMATORS
    return {"n_estimators": trial.suggest_int("n_estimators", lo, hi, step=step),
            "max_depth": trial.suggest_int("max_depth", *RF_MAX_DEPTH),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *RF_MIN_SAMPLES_LEAF)}


def search(model, input_dir, out_dir, n_jobs, n_trials, n_drugs):
    import optuna
    manifest, response, expression, tf, folds = load_bundle(input_dir)
    features = expression if model == "ridge" else tf
    drugs = objective_drugs(response, n_drugs)
    logger.info("%s: %d train lines, %d objective drugs (least complete has %d AUCs), %d trials, n_jobs=%d",
                model, len(response), len(drugs), int(response[drugs[-1]].notna().sum()), n_trials, n_jobs)

    trials, perdrug = [], []

    def objective(trial):
        params = suggest(trial, model)
        t0 = time.time()
        mean, sd, medians, rows = evaluate(model, params, features, response, folds, drugs, n_jobs)
        n_nan = sum(1 for r in rows if np.isnan(r["pearson"]))
        trials.append({"trial": trial.number, **params, "cv_mean": mean, "cv_sd": sd,
                       **{f"fold{k}_median": m for k, m in enumerate(medians)},
                       "n_nan": n_nan, "seconds": round(time.time() - t0, 1)})
        perdrug.extend({"trial": trial.number, **r} for r in rows)
        logger.info("trial %d %s: cv %.4f (sd %.4f), %d NaN, %.0f s", trial.number, params, mean, sd, n_nan,
                    time.time() - t0)
        return mean

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best = dict(study.best_trial.params)

    # Ridge alpha rounded to three significant figures, its objective recomputed with that value.
    # For the RF, the same evaluation is repeated to check that a trial is reproducible.
    chosen = dict(best)
    if model == "ridge":
        chosen["alpha"] = float(f"{best['alpha']:.3g}")
    rmean, rsd, rmed, _ = evaluate(model, chosen, features, response, folds, drugs, n_jobs)
    if model == "rf" and abs(rmean - study.best_value) > 1e-12:
        raise RuntimeError(f"Repeating the best RF trial gave {rmean}, not {study.best_value}")

    edge = []
    for name, value in best.items():
        for at_edge, beyond in EDGE_CHECKS.get(name, []):
            if at_edge(value):
                probe = {**chosen, name: beyond}
                pm, ps, _, _ = evaluate(model, probe, features, response, folds, drugs, n_jobs)
                edge.append({"param": name, "best": value, "probe": beyond, "cv_mean": pm, "cv_sd": ps})
                logger.info("Edge diagnostic %s=%s: cv %.4f (sd %.4f)", name, beyond, pm, ps)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(out_dir / f"trials_{model}.csv", index=False)
    pd.DataFrame(perdrug).to_csv(out_dir / f"cv_perdrug_{model}.csv", index=False)
    result = {
        "model": model, "best_trial": study.best_trial.number, "best_params": best,
        "best_cv_mean": study.best_value, "chosen_params": chosen,
        "chosen_cv_mean": rmean, "chosen_cv_sd": rsd, "chosen_fold_medians": rmed,
        "edge_diagnostic": edge,
        "design": {"n_folds": N_FOLDS, "n_trials": n_trials, "n_objective_drugs": n_drugs, "seed": SEED,
                   "min_fit": MIN_FIT, "min_eval": MIN_EVAL, "sampler": "TPESampler(seed=42)",
                   "official": n_trials == N_TRIALS and n_drugs == N_OBJECTIVE_DRUGS},
        "bundle_sha256": manifest["sha256"], "bundle_git_commit": manifest["git_commit"],
        "environment": versions(),
    }
    (out_dir / f"best_params_{model}.json").write_text(json.dumps(result, indent=2))
    logger.info("%s: best %s (cv %.4f), chosen %s (cv %.4f). Written to %s",
                model, best, study.best_value, chosen, rmean, out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("search")
    s.add_argument("--model", choices=["ridge", "rf"], required=True)
    s.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    s.add_argument("--n-jobs", type=int, default=1)
    s.add_argument("--n-trials", type=int, default=N_TRIALS, help="only for smoke tests")
    s.add_argument("--n-drugs", type=int, default=N_OBJECTIVE_DRUGS, help="only for smoke tests")
    a = p.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        search(a.model, a.input_dir, a.out_dir, a.n_jobs, a.n_trials, a.n_drugs)


if __name__ == "__main__":
    main()
