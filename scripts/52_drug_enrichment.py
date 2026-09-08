#!/usr/bin/env python3
"""
PRECISION - Mechanism-of-action and target enrichment among survival-associated drugs.

For the drugs with a significant Fisher meta-analysis association (FDR < 0.05,
script 21), tests whether each PRISM mechanism-of-action class and each
annotated molecular target is over-represented (one-sided Fisher exact test
against the 1,447 tested drugs, Benjamini-Hochberg over the tested classes).
Annotations come from data/precision_processed/prism_drug_annotations.csv
(PRISM 19Q4 treatment metadata, one row per compound, written by script 01).

Outputs (results/v8_multicohort/):
    moa_enrichment.csv     classes with at least MIN_DRUGS_MOA annotated drugs
    target_enrichment.csv  targets with at least MIN_DRUGS_TARGET annotated drugs

Both feed Supplementary Figures S6 and S7 (generate_paper_figures.py) and the
enrichment keys of paper_statistics.json.

Usage: python scripts/52_drug_enrichment.py (paths from config.py)
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

DATA = DATA_DIR
RESULTS = RESULTS_DIR / "v8_multicohort"
MIN_DRUGS_MOA = 5      # classes tested (same family as the figure since April 2026)
MIN_DRUGS_TARGET = 3   # targets tested
FDR_META = 0.05


def explode_annotation(annot: pd.DataFrame, column: str, lower: bool) -> pd.DataFrame:
    """One row per (drug, term) for a comma-separated annotation column."""
    df = annot[["name", column]].dropna().copy()
    df[column] = df[column].str.split(",")
    df = df.explode(column)
    df[column] = df[column].str.strip()
    if lower:
        df[column] = df[column].str.lower()
    return df[df[column] != ""]


def enrichment(pairs: pd.DataFrame, column: str, sig: set, universe: set,
               min_drugs: int) -> pd.DataFrame:
    rows = []
    for term, grp in pairs.groupby(column):
        drugs_with = set(grp["name"]) & universe
        n_total = len(drugs_with)
        if n_total < min_drugs:
            continue
        a = len(drugs_with & sig)
        b = len(sig - drugs_with)
        c = n_total - a
        d = len(universe - sig - drugs_with)
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({column: term, "n_sig": a, "n_total": n_total,
                     "pct_sig": 100.0 * a / n_total, "odds_ratio": odds, "pvalue": p})
    out = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    out["padj"] = stats.false_discovery_control(out["pvalue"])
    return out


def main() -> int:
    annot = pd.read_csv(DATA / "precision_processed" / "prism_drug_annotations.csv")
    annot = annot.drop_duplicates(subset="name").copy()
    annot["name"] = annot["name"].str.lower()

    meta = pd.read_csv(RESULTS / "meta_analysis_fisher.csv")
    universe = set(meta["drug"])
    sig = set(meta.loc[meta["fisher_padj"] < FDR_META, "drug"])
    print(f"Universe: {len(universe)} drugs tested, {len(sig)} with Fisher FDR < {FDR_META}")

    moa = enrichment(explode_annotation(annot, "moa", lower=True), "moa", sig, universe, MIN_DRUGS_MOA)
    target = enrichment(explode_annotation(annot, "target", lower=False), "target", sig, universe, MIN_DRUGS_TARGET)

    RESULTS.mkdir(parents=True, exist_ok=True)
    moa.to_csv(RESULTS / "moa_enrichment.csv", index=False)
    target.to_csv(RESULTS / "target_enrichment.csv", index=False)

    print(f"\nMOA classes tested: {len(moa)}, FDR < 0.05: {int((moa['padj'] < 0.05).sum())}")
    print(moa.head(5).to_string(index=False))
    print(f"\nTargets tested: {len(target)}, FDR < 0.05: {int((target['padj'] < 0.05).sum())}")
    print(target.head(6).to_string(index=False))
    print(f"\nWritten: {RESULTS / 'moa_enrichment.csv'}, {RESULTS / 'target_enrichment.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
