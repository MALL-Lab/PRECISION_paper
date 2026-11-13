#!/usr/bin/env python3
"""
PRECISION - Graph component ablation (Supplementary Figure 12).

Summarizes the ablation runs trained by scripts/57_train_gnns.py and trains
nothing. Every configuration of ABLATIONS in src/models/training.py is SAGE
on the seed-42 cell-line split with the five initialization seeds of Table 1.
A reduced graph removes whole prior-knowledge relations (PPI, CollecTRI,
drug-target) and their reverses from message passing. The control of SHUFFLED
keeps every relation and rewires the drug-target edges with degree-preserving
swaps, one rewiring per seed. The Full graph row is
made of the Table 1 SAGE runs (tag full). The metrics are the ones recorded in
the sidecar of each run: global Pearson over the test edges of the evaluation
universe of drugs, computed with the final-epoch weights.

The ablation used to be trained here with its own protocol (500 epochs, one
seed and the best test Pearson over checkpoints). It now shares the protocol,
the seeds and the full-graph checkpoints of Table 1, and the test set never
chooses a model.

Outputs in results/v6_strengthen/:
  graph_ablation.csv           one row per configuration: config (label of
                               ABLATIONS or SHUFFLED), test_pearson (mean over
                               the seeds), test_pearson_sd (sample standard
                               deviation),
                               n_seeds, n_edge_types (message-passing
                               relations, reverses included), excluded and
                               shuffled (the rewired relation, or none)
  graph_ablation_per_seed.csv  one row per run, with the number of original
                               drug-target edges each rewiring keeps

The script fails if any of the runs is missing or was not trained with the
shared protocol.

Run from project root, after scripts/57_train_gnns.py:
    python scripts/27_graph_ablation.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import pandas as pd

from config import RESULTS_DIR  # noqa: E402
from src.models import training as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graph_ablation")

OUT_DIR = RESULTS_DIR / "v6_strengthen"
ARCH, SPLIT_SEED = "sage", 42
# The initialization seeds of Table 1 (TABLE1_SEEDS in scripts/57_train_gnns.py).
INIT_SEEDS = (42, 123, 456, 789, 1011)
# The configuration without prior knowledge removes every prior-knowledge relation.
PRIOR_RELATIONS = set(T.ABLATIONS["response_only"][1])
TAGS = (*T.ABLATIONS, *T.SHUFFLED)


def load_sidecars() -> list[dict]:
    """Sidecars of every ablation run, in the order of TAGS and INIT_SEEDS."""
    paths = {(tag, seed): T.RUNS_DIR / f"{T.run_name(ARCH, SPLIT_SEED, seed, tag)}.json"
             for tag in TAGS for seed in INIT_SEEDS}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} runs are missing in {T.RUNS_DIR} "
                                f"(train them with scripts/57_train_gnns.py): {missing}")
    sidecars = []
    for (tag, seed), path in paths.items():
        s = json.loads(path.read_text())
        exclude, shuffle = T.ablation_config(tag)
        if (s["excluded_relations"] != list(exclude) or s.get("shuffled_relation") != shuffle
                or s.get("shuffle_seed") != (seed if shuffle else None)
                or s["checkpoint_selection"] != "final epoch"):
            raise ValueError(f"{path.name} does not match the ablation protocol of src/models/training.py")
        sidecars.append(s)
    protocols = {json.dumps([s["hyperparameters"], s["optimizer"]], sort_keys=True) for s in sidecars}
    if len(protocols) > 1:
        raise ValueError(f"The ablation runs were trained with {len(protocols)} different protocols")
    return sidecars


def main():
    sidecars = load_sidecars()
    per_seed = pd.DataFrame([{
        "config": T.config_label(s["tag"]), "tag": s["tag"], "init_seed": s["init_seed"], "run": s["run"],
        "test_pearson": s["metrics"]["global_pearson"], "median_pearson": s["metrics"]["median_pearson"],
        "n_drugs": s["metrics"]["n_drugs"],
        "shuffled_edges_kept": s.get("shuffled_edges_kept"), "shuffled_edges_total": s.get("shuffled_edges_total"),
    } for s in sidecars])

    rows = []
    for tag in TAGS:
        excluded, shuffle = T.ablation_config(tag)
        r = per_seed.loc[per_seed["tag"] == tag, "test_pearson"]
        rows.append({"config": T.config_label(tag), "test_pearson": r.mean(), "test_pearson_sd": r.std(ddof=1),
                     "n_seeds": len(r), "n_edge_types": 2 * len(PRIOR_RELATIONS - set(excluded)),
                     "excluded": " + ".join(excluded) or "none", "shuffled": shuffle or "none"})
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "graph_ablation.csv", index=False)
    per_seed.to_csv(OUT_DIR / "graph_ablation_per_seed.csv", index=False)
    logger.info("Graph ablation, %s on split %d, %d seeds per configuration:", ARCH, SPLIT_SEED, len(INIT_SEEDS))
    for _, r in df.iterrows():
        logger.info("  %-42s Pearson %.4f +/- %.4f", r["config"], r["test_pearson"], r["test_pearson_sd"])


if __name__ == "__main__":
    main()
