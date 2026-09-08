#!/usr/bin/env python3
"""
PRECISION - AURORA site-stratified TF activity (internal clinical review).

Internal clinical review: AURORA metastases come from heterogeneous
tissue sites (breast, brain, liver, lung, lymph node, bone, ...). The paired
Wilcoxon primary-vs-metastasis test does not account for tissue-specific
transcriptional programmes that may drive TF activity changes independently
of the metastatic state itself.

This script stratifies the primary vs metastasis comparison by metastatic
biopsy tissue site for the four Bonferroni-significant IG TFs (CREB3L1,
KLF8, AEBP1, SPDEF). Each site with >=3 paired patients yields a per-site
delta and Wilcoxon p.

Outputs in paper/results/:
  aurora_tf_by_site.csv         - rows = (TF, site)
  aurora_tf_by_site_summary.csv - pivot summary

Run from project root:
    python scripts/39_aurora_site_stratified.py
"""

from __future__ import annotations

import gzip
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_DIR, RESULTS_DIR

# Outputs go to RESULTS_DIR; generate_paper_results.py copies the two tables
# cited in the manuscript to paper/results/.
OUT_DIR = RESULTS_DIR / "v9_aurora"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aurora_site")

BONF_TFS = ["CREB3L1", "KLF8", "AEBP1", "SPDEF"]
EXTRA_TFS = ["MZF1", "HIVEP1", "REL", "IRX1", "ASXL1", "BCOR"]


def parse_geo_series_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        lines = f.readlines()

    def _split(line: str) -> list[str]:
        parts = line.strip().split("\t")
        return [p.strip().strip('"') for p in parts[1:]]

    titles, tissues, disease, treatment, time_col = None, None, None, None, None
    for line in lines:
        if line.startswith("!Sample_title"):
            titles = _split(line)
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = _split(line)
            if vals and vals[0].startswith("disease"):
                disease = [v.split(":", 1)[-1].strip() for v in vals]
            elif vals and vals[0].startswith("tissue"):
                tissues = [v.split(":", 1)[-1].strip() for v in vals]
            elif vals and vals[0].startswith("treatment"):
                treatment = [v.split(":", 1)[-1].strip() for v in vals]
            elif vals and vals[0].startswith("time"):
                time_col = [v.split(":", 1)[-1].strip() for v in vals]

    ids = []
    for t in titles:
        m = re.search(r"\[(AUR-[^\]]+)\]", t)
        ids.append(m.group(1) if m else t)

    n = len(ids)
    return pd.DataFrame({
        "sample_id": ids,
        "disease": disease or [""] * n,
        "tissue_site": tissues,
        "treatment": treatment or [""] * n,
        "time": time_col or [""] * n,
    }).set_index("sample_id")


def parse_patient_tissue(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("-")
    if len(parts) >= 3:
        pid = parts[1]
        tt = parts[2]
        if tt.startswith("TTP"):
            return pid, "primary"
        if tt.startswith("TTM"):
            return pid, "metastasis"
    return sample_id, "unknown"


def main() -> int:
    tf_acts = pd.read_csv(RESULTS_DIR / "v9_aurora" / "aurora_tf_activities.csv",
                          index_col=0)
    logger.info("TF activities: %d samples x %d TFs", *tf_acts.shape)

    meta = parse_geo_series_matrix(
        DATA_DIR / "AURORA-US" / "GSE209998_series_matrix.txt.gz",
    )
    logger.info("GEO metadata: %d samples", len(meta))
    logger.info("Tissue sites: %s", dict(meta["tissue_site"].value_counts()))

    patient_list, role_list, site_list = [], [], []
    for sid in tf_acts.index.astype(str):
        pid, role = parse_patient_tissue(sid)
        patient_list.append(pid)
        role_list.append(role)
        site_list.append(meta["tissue_site"].get(sid, "Unknown"))

    info = pd.DataFrame({
        "sample_id": tf_acts.index,
        "patient_id": patient_list,
        "role": role_list,
        "site": site_list,
    }).set_index("sample_id")

    prim = info[info["role"] == "primary"]
    met = info[info["role"] == "metastasis"]
    paired_pids = sorted(set(prim["patient_id"]) & set(met["patient_id"]))
    logger.info("Paired patients: %d", len(paired_pids))

    site_counts = (met[met["patient_id"].isin(paired_pids)]
                   ["site"].value_counts())
    sites_to_keep = [s for s, n in site_counts.items() if n >= 3]
    logger.info("Sites with >=3 biopsies in paired patients: %s", sites_to_keep)

    all_tfs = [tf for tf in (BONF_TFS + EXTRA_TFS) if tf in tf_acts.columns]

    rows = []
    for tf in all_tfs:
        for site in sites_to_keep:
            p_vals, m_vals = [], []
            for pid in paired_pids:
                p_samples = prim[prim["patient_id"] == pid].index
                m_samples_site = met[(met["patient_id"] == pid)
                                      & (met["site"] == site)].index
                if len(p_samples) == 0 or len(m_samples_site) == 0:
                    continue
                p_vals.append(tf_acts.loc[p_samples, tf].mean())
                m_vals.append(tf_acts.loc[m_samples_site, tf].mean())
            if len(p_vals) < 3:
                pvalue = np.nan
                delta = np.nan
            else:
                p_arr = np.array(p_vals)
                m_arr = np.array(m_vals)
                delta = float(np.mean(m_arr - p_arr))
                if np.allclose(p_arr, m_arr):
                    pvalue = np.nan
                else:
                    _, pvalue = wilcoxon(p_arr, m_arr, alternative="two-sided")
                    pvalue = float(pvalue)
            rows.append({
                "tf": tf,
                "site": site,
                "n_paired": len(p_vals),
                "primary_mean": float(np.mean(p_vals)) if p_vals else np.nan,
                "metastasis_mean": float(np.mean(m_vals)) if m_vals else np.nan,
                "delta": delta,
                "pvalue": pvalue,
            })

    df = pd.DataFrame(rows).sort_values(["tf", "site"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "aurora_tf_by_site.csv", index=False)
    logger.info("\nBONF TFs by site:\n%s",
                df[df["tf"].isin(BONF_TFS)].to_string(index=False))

    pivot_d = df.pivot_table(index="tf", columns="site", values="delta")
    pivot_n = df.pivot_table(index="tf", columns="site", values="n_paired")
    pivot_p = df.pivot_table(index="tf", columns="site", values="pvalue")
    with open(OUT_DIR / "aurora_tf_by_site_summary.csv", "w",
              encoding="utf-8") as f:
        f.write("# per-site delta (meta - primary)\n")
        pivot_d.round(3).to_csv(f)
        f.write("\n# per-site Wilcoxon p-value\n")
        pivot_p.map(
            lambda x: f"{x:.2e}" if pd.notna(x) else "n.a."
        ).to_csv(f)
        f.write("\n# per-site n_paired\n")
        pivot_n.astype("Int64").to_csv(f)

    bonf_df = df[df["tf"].isin(BONF_TFS) & df["n_paired"].ge(3) &
                  df["delta"].notna()]
    logger.info("\nBONF TF direction concordance across sites:")
    for tf in BONF_TFS:
        sub = bonf_df[bonf_df["tf"] == tf]["delta"]
        if len(sub):
            signs = np.sign(sub)
            same = int((signs == signs.iloc[0]).sum())
            direction = "+" if sub.mean() > 0 else "-"
            logger.info("  %-10s  %d/%d sites same sign  global=%s",
                        tf, same, len(sub), direction)

    return 0


if __name__ == "__main__":
    sys.exit(main())
