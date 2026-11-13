#!/usr/bin/env python3
"""Train every GNN of Paper 1 once, with the protocol of src/models/training.py.

Sets (``--set``, default ``all``):
- table1: SAGE and HGT on the seed-42 cell-line split, initialization seeds
  42, 123, 456, 789 and 1011. The run sage_split42_init42_full is the
  canonical model for Integrated Gradients, and its per-epoch history is
  Supplementary Figure 1.
- ig_seeds: SAGE with split and initialization seeds 123 and 456, for the
  multi-seed Integrated Gradients analysis (script 35).
- ablation: SAGE on the seed-42 split with the five initialization seeds of
  Table 1 for each reduced graph and for the control with the drug-target
  edges rewired (SHUFFLED in src/models/training.py). The full graph runs
  are those of table1.

Outputs in results/v12_gnn/: one checkpoint (.pt) and one sidecar (.json) per
run, history/ and per_drug/ tables, and runs.csv with one row per run.
Existing runs are kept unless --force is given.

Run from project root:
    python scripts/57_train_gnns.py [--set table1|ig_seeds|ablation|all] [--force]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models import training as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gnns")

TABLE1_SEEDS = (42, 123, 456, 789, 1011)
IG_SEEDS = (42, 123, 456)
CANONICAL = ("sage", 42, 42, "full")


def plan(which: str) -> list[tuple]:
    """(arch, split_seed, init_seed, tag) of every run of the requested set."""
    runs = []
    if which in ("table1", "all"):
        runs += [("sage", 42, s, "full") for s in TABLE1_SEEDS]
    if which in ("ig_seeds", "all"):
        runs += [("sage", s, s, "full") for s in IG_SEEDS if s != 42]
    if which in ("table1", "all"):
        runs += [("hgt", 42, s, "full") for s in TABLE1_SEEDS]
    if which in ("ablation", "all"):
        runs += [("sage", 42, s, tag) for tag in (*T.ABLATIONS, *T.SHUFFLED) if tag != "full"
                 for s in TABLE1_SEEDS]
    return runs


def write_runs_table():
    rows = []
    for f in sorted(T.RUNS_DIR.glob("*.json")):
        s = json.loads(f.read_text())
        rows.append({"run": s["run"], "arch": s["arch"], "split_seed": s["split_seed"],
                     "init_seed": s["init_seed"], "tag": s["tag"],
                     "config": T.config_label(s["tag"]), **s["metrics"],
                     "n_train_edges": s["n_train_edges"], "n_test_edges": s["n_test_edges"],
                     "n_train_drugs": s["n_train_drugs"], "n_eval_drugs": s["n_eval_drugs"],
                     "seconds": s["seconds"], "sha256": s["sha256"]})
    pd.DataFrame(rows).to_csv(T.RUNS_DIR / "runs.csv", index=False)
    logger.info("runs.csv: %d runs", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["table1", "ig_seeds", "ablation", "all"])
    ap.add_argument("--force", action="store_true", help="retrain runs that already exist")
    args = ap.parse_args()

    dev = T.device()
    logger.info("Device: %s", dev)
    graph = T.load_graph(dev)
    splits = {}
    todo = plan(args.set)
    logger.info("%d runs requested", len(todo))
    for i, (arch, split_seed, init_seed, tag) in enumerate(todo, 1):
        name = T.run_name(arch, split_seed, init_seed, tag)
        if not args.force and (T.RUNS_DIR / f"{name}.json").exists():
            logger.info("[%d/%d] %s exists, skipped", i, len(todo), name)
            continue
        if split_seed not in splits:
            splits[split_seed] = T.prepare_split(split_seed, graph, dev)
        logger.info("[%d/%d] training %s", i, len(todo), name)
        # The canonical run records every epoch for Supplementary Figure 1.
        every = 1 if (arch, split_seed, init_seed, tag) == CANONICAL else 50
        T.train_and_save(arch, graph, splits[split_seed], init_seed, tag, history_every=every)
        write_runs_table()
    write_runs_table()


if __name__ == "__main__":
    main()
