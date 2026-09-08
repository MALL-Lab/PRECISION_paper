#!/usr/bin/env python3
"""
PRECISION - Robustness of the IG attributions to the candidate set
(Supplementary Note 1).

Compares the main Integrated Gradients run (the 7 candidates of Table 5 plus
the 4 positive controls, 11 drugs) with the run on the extended candidate set
(the 16 kinase or HDAC inhibitors of Supplementary Table S1 plus the 4
controls, 20 drugs), produced by scripts 32, 35 and 36 with
--candidate-set extended.

Inputs (results/v4_xai and results/v9_aurora):
  tf_importance_ig_global.csv,                tf_importance_ig_global_extended.csv
  tf_importance_ig_multi_seed_stability.csv,  tf_importance_ig_multi_seed_stability_extended.csv
  aurora_tf_ig_validation.csv,                aurora_tf_ig_validation_extended.csv
Output:
  results/v4_xai/ig_extended_comparison.csv   one row per metric
  results/v4_xai/ig_extended_core_tfs.csv     the ten seed-stable core TFs of each run, side by side

Usage: python scripts/49_ig_extended_comparison.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ig_extended")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR  # noqa: E402

V4 = RESULTS_DIR / "v4_xai"
V9 = RESULTS_DIR / "v9_aurora"
TOP_N = 20
CORE_N = 10


def main() -> int:
    g_main = pd.read_csv(V4 / "tf_importance_ig_global.csv", comment="#")
    g_ext = pd.read_csv(V4 / "tf_importance_ig_global_extended.csv", comment="#")
    s_main = pd.read_csv(V4 / "tf_importance_ig_multi_seed_stability.csv")
    s_ext = pd.read_csv(V4 / "tf_importance_ig_multi_seed_stability_extended.csv")
    a_main = pd.read_csv(V9 / "aurora_tf_ig_validation.csv")
    a_ext = pd.read_csv(V9 / "aurora_tf_ig_validation_extended.csv")

    # Global importance (seed 42 model): top-20 overlap and rank correlation
    top_main = set(g_main.head(TOP_N)["tf"])
    top_ext = set(g_ext.head(TOP_N)["tf"])
    merged = g_main[["tf", "importance_mean"]].merge(
        g_ext[["tf", "importance_mean"]], on="tf", suffixes=("_main", "_ext"))
    rho, _ = spearmanr(merged["importance_mean_main"], merged["importance_mean_ext"])

    # Seed-stable core TFs (lowest mean rank across the three seeds)
    core_main = s_main.sort_values("mean_rank").head(CORE_N)["tf"].tolist()
    core_ext = s_ext.sort_values("mean_rank").head(CORE_N)["tf"].tolist()
    core_overlap = len(set(core_main) & set(core_ext))
    stab_merged = s_main[["tf", "mean_rank"]].merge(
        s_ext[["tf", "mean_rank"]], on="tf", suffixes=("_main", "_ext"))
    rho_stab, _ = spearmanr(stab_merged["mean_rank_main"], stab_merged["mean_rank_ext"])

    # AURORA validation of the core TFs of each run
    def aurora_summary(a: pd.DataFrame) -> dict:
        n = len(a)
        bonf = int((a["pvalue"] < 0.05 / n).sum()) if "pvalue" in a else None
        nominal = int((a["pvalue"] < 0.05).sum()) if "pvalue" in a else None
        return {"n": n, "bonferroni": bonf, "nominal": nominal}

    au_main, au_ext = aurora_summary(a_main), aurora_summary(a_ext)
    core_shared = sorted(set(core_main) & set(core_ext))

    rows = [
        ("n_drugs_main", 11), ("n_drugs_extended", 20),
        ("global_top20_overlap", len(top_main & top_ext)),
        ("global_top20_only_main", ", ".join(sorted(top_main - top_ext))),
        ("global_top20_only_extended", ", ".join(sorted(top_ext - top_main))),
        ("global_spearman_all_tfs", round(float(rho), 4)),
        ("core10_overlap", core_overlap),
        ("core10_shared", ", ".join(core_shared)),
        ("core10_only_main", ", ".join(t for t in core_main if t not in core_ext)),
        ("core10_only_extended", ", ".join(t for t in core_ext if t not in core_main)),
        ("stability_spearman_mean_rank", round(float(rho_stab), 4)),
        ("aurora_main_n_tfs", au_main["n"]), ("aurora_main_bonferroni", au_main["bonferroni"]),
        ("aurora_main_nominal", au_main["nominal"]),
        ("aurora_extended_n_tfs", au_ext["n"]), ("aurora_extended_bonferroni", au_ext["bonferroni"]),
        ("aurora_extended_nominal", au_ext["nominal"]),
    ]
    out = pd.DataFrame(rows, columns=["metric", "value"])
    out.to_csv(V4 / "ig_extended_comparison.csv", index=False)

    core = pd.DataFrame({
        "rank": range(1, CORE_N + 1),
        "core_tf_main": core_main,
        "mean_rank_main": s_main.sort_values("mean_rank").head(CORE_N)["mean_rank"].round(2).tolist(),
        "core_tf_extended": core_ext,
        "mean_rank_extended": s_ext.sort_values("mean_rank").head(CORE_N)["mean_rank"].round(2).tolist(),
    })
    core.to_csv(V4 / "ig_extended_core_tfs.csv", index=False)

    logger.info("\n%s", out.to_string(index=False))
    logger.info("\n%s", core.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
