#!/usr/bin/env python3
"""
PRECISION - AURORA paired Wilcoxon validation for IG top-10 TFs.

The previous AURORA validation (Table 4, Figure 6) used the 10 top TFs from
the ablation-bootstrap method (MYC, TP53, DNMT3A, IRF4, KLF8, SRSF2, E2F1,
AR, SPI1, HIF1A). Script 34 demonstrated that the ablation ranking is
unstable across seeds (top20 overlap 2-3/20). Script 35 showed IG is more
stable (top20 overlap 5-10/20). Three TFs (IRX1, KLF8, AEBP1) are in
top 20 across all 3 seeds under IG.

This script re-validates the narrative using the 10 most stable IG TFs
(ranked by mean rank across seeds in
tf_importance_ig_multi_seed_stability.csv).

Output: results/v9_aurora/aurora_tf_ig_validation.csv

Run from project root:
    python scripts/36_aurora_ig_validation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aurora_ig_val")

V9 = RESULTS_DIR / "v9_aurora"


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="AURORA validation of the IG core TFs")
    ap.add_argument("--candidate-set", choices=["table5", "extended"], default="table5",
                    help="which IG run to validate (extended reads the _extended stability table)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    suffix = "" if args.candidate_set == "table5" else f"_{args.candidate_set}"
    # Top 10 IG TFs by mean rank across seeds (from stability table).
    stab = pd.read_csv(
        RESULTS_DIR / "v4_xai" / f"tf_importance_ig_multi_seed_stability{suffix}.csv"
    )
    top10 = stab.head(10)["tf"].tolist()
    logger.info("Top 10 IG TFs by mean rank across seeds: %s", top10)

    # AURORA TF activities (already computed: 129 samples x 770 TFs).
    tf_acts = pd.read_csv(V9 / "aurora_tf_activities.csv", index_col=0)
    logger.info("AURORA TF activities: %d samples x %d TFs",
                tf_acts.shape[0], tf_acts.shape[1])

    # Paired primary/metastasis comparison on the 10 IG TFs. Script 22 does
    # the same averaging per patient for its own TF set; here the pairs are
    # rebuilt from the sample identifiers of the TF activity matrix so that
    # no external metadata file is needed.

    missing = [tf for tf in top10 if tf not in tf_acts.columns]
    if missing:
        logger.warning("IG TFs missing from AURORA TF activities: %s", missing)
    avail = [tf for tf in top10 if tf in tf_acts.columns]
    logger.info("IG TFs available in AURORA: %s", avail)

    # Derive per-patient paired values using the prior script 22 approach.
    # Need: for each sample, know its patient and primary/meta status.
    # Quickest: pull from aurora_drug_primary_vs_meta metadata? No, that's
    # drug-level. We need sample-level. Load from the script 22 intermediate
    # if available, otherwise compute from expression metadata.

    # Look for n_paired used (39 patients). In script 22 line ~240, paired
    # patients were computed via averaging metastatic biopsies per patient.
    # Here we replicate that averaging using sample IDs parsed from the
    # AURORA GEO metadata. Since the raw sample metadata is not bundled with
    # the package (see data/README.md), we'll rely on a helper if it exists;
    # otherwise we compute empirically.

    # ----- Empirical pairing from sample names in tf_acts.index -----
    # AURORA sample ID format: "AUR01_P", "AUR01_M1", etc. Parse patient id.
    idx = tf_acts.index.astype(str)
    # Split convention: first token = patient, rest = tissue/biopsy info
    def _parse(sid: str):
        # AURORA format: AUR-{patient}-TT{P|M}{n}-A-...
        # e.g. AUR-AFEA-TTP1-A-1-0-R-A741-41  (primary)
        #      AUR-AFEA-TTM4-A-1-1-R-A742-41  (metastasis biopsy 4)
        parts = sid.split("-")
        if len(parts) >= 3:
            pid = parts[1]
            tt = parts[2]
            if tt.startswith("TTP"):
                return pid, "primary"
            if tt.startswith("TTM"):
                return pid, "metastasis"
        return sid, "unknown"

    parsed = pd.DataFrame({
        "sample_id": idx,
        "patient_id": [_parse(s)[0] for s in idx],
        "tissue": [_parse(s)[1] for s in idx],
    })
    logger.info("Parsed samples: %d primary, %d metastasis, %d unknown",
                (parsed["tissue"] == "primary").sum(),
                (parsed["tissue"] == "metastasis").sum(),
                (parsed["tissue"] == "unknown").sum())

    # If primary/meta detection yields <20 per group, try simpler heuristics.
    if (parsed["tissue"] == "primary").sum() < 10 or \
       (parsed["tissue"] == "metastasis").sum() < 10:
        logger.warning("Sample pairing heuristic failed. Fall back: recompute"
                       " unpaired Mann-Whitney on all samples per tissue.")
        # Leverage clinical CSV if accessible.
        clin_path = DATA_DIR / "AURORA-US" / "AURORA_clinical.csv"
        if clin_path.exists():
            clin = pd.read_csv(clin_path)
            logger.info("Clinical cols: %s", list(clin.columns)[:15])
        else:
            logger.warning("Clinical file not accessible. Proceeding with"
                           " simplified group-level comparison.")
            # Assume all non-primary are metastasis for a rough check
            return _fallback_unpaired(tf_acts, avail)

    # Compute per-patient primary and averaged-metastasis values per TF
    prim = parsed[parsed["tissue"] == "primary"]
    meta = parsed[parsed["tissue"] == "metastasis"]
    patients_with_both = set(prim["patient_id"]) & set(meta["patient_id"])
    logger.info("Patients with both primary & metastasis: %d",
                len(patients_with_both))

    # For each IG TF, perform paired Wilcoxon signed-rank
    rows = []
    for tf in avail:
        p_vals, m_vals = [], []
        for pid in sorted(patients_with_both):
            p_samples = prim[prim["patient_id"] == pid]["sample_id"]
            m_samples = meta[meta["patient_id"] == pid]["sample_id"]
            p_act = tf_acts.loc[p_samples, tf].mean()
            m_act = tf_acts.loc[m_samples, tf].mean()
            p_vals.append(p_act)
            m_vals.append(m_act)
        p_vals = np.array(p_vals)
        m_vals = np.array(m_vals)
        delta = float(np.mean(m_vals - p_vals))
        if len(p_vals) >= 3 and not np.allclose(p_vals, m_vals):
            stat, pvalue = wilcoxon(p_vals, m_vals, alternative="two-sided")
            pvalue = float(pvalue)
        else:
            pvalue = np.nan
        rows.append({
            "tf": tf,
            "primary_mean": float(np.mean(p_vals)),
            "metastasis_mean": float(np.mean(m_vals)),
            "delta": delta,
            "pvalue": pvalue,
            "n_paired": len(p_vals),
        })

    out = pd.DataFrame(rows)
    out["bonferroni_p"] = out["pvalue"] * len(out)
    out["bonferroni_p"] = out["bonferroni_p"].clip(upper=1.0)
    out["bonferroni_sig"] = out["bonferroni_p"] < 0.05
    out["nominal_sig"] = out["pvalue"] < 0.05
    out = out.sort_values("pvalue").reset_index(drop=True)
    out.to_csv(V9 / f"aurora_tf_ig_validation{suffix}.csv", index=False)
    logger.info("Wrote aurora_tf_ig_validation.csv")
    logger.info("Summary:")
    for _, r in out.iterrows():
        sig = "*" if r["bonferroni_sig"] else ("n" if r["nominal_sig"] else ".")
        logger.info("  %-10s  primary=%.3f  meta=%.3f  delta=%+.3f  "
                    "p=%.2e  bonf=%.2e  %s",
                    r["tf"], r["primary_mean"], r["metastasis_mean"],
                    r["delta"], r["pvalue"], r["bonferroni_p"], sig)
    n_bonf = int(out["bonferroni_sig"].sum())
    n_nom = int(out["nominal_sig"].sum())
    logger.info("Bonferroni sig: %d/%d, nominal sig: %d/%d",
                n_bonf, len(out), n_nom, len(out))
    return 0


def _fallback_unpaired(tf_acts, tfs):
    """Unpaired Mann-Whitney if primary/meta pairing failed."""
    from scipy.stats import mannwhitneyu
    logger.info("Running unpaired Mann-Whitney (fallback)")
    # Without group labels, can't do. Abort.
    logger.error("Cannot determine groups without clinical. Abort.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
