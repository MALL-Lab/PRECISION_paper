#!/usr/bin/env python3
"""
generate_paper_results.py
=========================
Consolidates ALL the results cited in the PRECISION manuscript from the CSVs
already present under RESULTS_DIR (results/ by default). It does not retrain
models nor run any heavy analysis.

Output (under paper/results/):
  table1_model_comparison.csv
  table2_multicohort_summary.csv
  table3_positive_controls.csv
  table4_aurora_tfs.csv
  table5_candidates.csv
  supp_table_S1_candidate_selection.csv
  shared_drugs.csv
  paper_statistics.json
  plus the intermediate CSVs copied from RESULTS_DIR that
  generate_paper_figures.py reads (see the `copies` list below).

Usage (from the package root, any cwd works):
  python paper/scripts/generate_paper_results.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# paper/scripts/ -> package root, so `config` is importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, RESULTS_DIR, PAPER_RESULTS  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = RESULTS_DIR
DATA = DATA_DIR
OUT = PAPER_RESULTS
OUT.mkdir(parents=True, exist_ok=True)

stats = {}  # all key numbers for paper_statistics.json


def load_csv(path, label=None):
    """Load CSV with warning if missing (non-critical inputs)."""
    p = Path(path)
    if not p.exists():
        tag = label or p.name
        print(f"  [WARNING] Not found: {tag} -> {p}")
        return None
    return pd.read_csv(p)


def load_csv_required(path, label=None):
    """Load CSV and fail loudly if missing (critical inputs that must exist)."""
    p = Path(path)
    if not p.exists():
        tag = label or p.name
        raise FileNotFoundError(
            f"[REQUIRED CSV missing] {tag} -> {p}. "
            f"Run the generating script (see README.md) before generate_paper_results.py."
        )
    return pd.read_csv(p)


REQUIRED_INPUTS = [
    RESULTS / "v2_fair" / "fair_comparison.csv",
    RESULTS / "v3_embeddings" / "embedding_comparison.csv",
    RESULTS / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv",
    RESULTS / "v6_strengthen" / "graph_ablation.csv",
    RESULTS / "v8_multicohort" / "survival_SCANB.csv",
    RESULTS / "v8_multicohort" / "survival_METABRIC.csv",
    RESULTS / "v8_multicohort" / "survival_TCGA.csv",
    RESULTS / "v8_multicohort" / "meta_analysis_fisher.csv",
    RESULTS / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",
    # Cohort sizes cited in Table 2, Methods and Results (scripts 21 and 22)
    RESULTS / "v8_multicohort" / "cohort_summary.csv",
    RESULTS / "v9_aurora" / "aurora_cohort_summary.csv",
    # Node and edge counts of the prebuilt graph (script 51)
    RESULTS / "graph_summary.csv",
    # Data constants cited in Methods (script 53)
    RESULTS / "data_summary.csv",
    # MOA and target enrichment among survival-associated drugs (script 52)
    RESULTS / "v8_multicohort" / "moa_enrichment.csv",
    RESULTS / "v8_multicohort" / "target_enrichment.csv",
    # Long-form IG attributions (script 32): cells x drugs scope of the XAI
    RESULTS / "v4_xai" / "ig_full_attributions.csv",
    # Sensitivity analyses cited in Results and Discussion (scripts 37, 39, 40, 41)
    RESULTS / "v8_multicohort" / "cox_clinical_comparison.csv",
    RESULTS / "v9_aurora" / "aurora_tf_by_site.csv",
    RESULTS / "v9_aurora" / "aurora_tf_by_site_summary.csv",
    RESULTS / "v8_multicohort" / "cox_rfs_comparison.csv",
    RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv",
]


def check_required_inputs():
    """Verify all critical CSVs exist before continuing."""
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        print("\n[ERROR] Required CSVs for the paper results are missing:")
        for p in missing:
            print(f"  - {p}")
        print("\nRun the full pipeline (pipeline.py) before this script.")
        raise SystemExit(1)
    print(f"[OK] All {len(REQUIRED_INPUTS)} required CSVs present.\n")


check_required_inputs()


# ===================================================================
# TABLE 1 - Model comparison (v2_fair/fair_comparison.csv)
# ===================================================================
print("=" * 60)
print("TABLE 1: Model comparison")
print("=" * 60)

df_fair = load_csv(RESULTS / "v2_fair" / "fair_comparison.csv", "fair_comparison")
# Ridge(expr+GNN emb) hybrid comes from the v3_embeddings benchmark (script 12)
# and the manuscript reports it as the 4th row of Table 1.
df_hybrid = load_csv(RESULTS / "v3_embeddings" / "embedding_comparison.csv", "embedding_comparison")

if df_fair is not None:
    t1 = df_fair[["model", "global_pearson", "median_pearson", "pct_gt_0.3"]].copy()
    t1.columns = ["Model", "Global_Pearson", "Median_PerDrug_Pearson", "Pct_Drugs_GT_0.3"]

    # Append Ridge(expr + GNN emb) hybrid row (matches manuscript Table 1)
    if df_hybrid is not None:
        hybrid_row = df_hybrid[df_hybrid["model"] == "Ridge(expression+GNN_emb)"]
        if len(hybrid_row):
            r = hybrid_row.iloc[0]
            t1 = pd.concat([t1, pd.DataFrame([{
                "Model": "Ridge_expr_GNN_emb",
                "Global_Pearson": r["global_pearson"],
                "Median_PerDrug_Pearson": r["median_pearson"],
                "Pct_Drugs_GT_0.3": r["pct_gt_03"],
            }])], ignore_index=True)

    t1["Global_Pearson"] = t1["Global_Pearson"].round(3)
    t1["Median_PerDrug_Pearson"] = t1["Median_PerDrug_Pearson"].round(3)
    t1["Pct_Drugs_GT_0.3"] = t1["Pct_Drugs_GT_0.3"].round(1)
    t1.to_csv(OUT / "table1_model_comparison.csv", index=False)
    print(t1.to_string(index=False))

    for _, row in t1.iterrows():
        stats[f"table1_{row['Model']}_global_pearson"] = row["Global_Pearson"]
        stats[f"table1_{row['Model']}_median_perdrug"] = row["Median_PerDrug_Pearson"]


# ===================================================================
# TABLE 2 - Multi-cohort survival summary
# ===================================================================
print("\n" + "=" * 60)
print("TABLE 2: Multi-cohort survival summary")
print("=" * 60)

# Cohort sizes come from scripts/21_multicohort_survival.py (cohort_summary.csv):
# clinical rows, patients after deduplication, patients with survival, events
# and TFs per cohort. They feed Table 2 and the Methods dedup chains.
cohort_summary = load_csv_required(
    RESULTS / "v8_multicohort" / "cohort_summary.csv", "cohort_summary.csv"
).set_index("cohort")
cohort_info = {
    "SCANB":    {"file": "survival_SCANB.csv",    "tech": "RNA-seq"},
    "METABRIC": {"file": "survival_METABRIC.csv", "tech": "Microarray"},
    "TCGA":     {"file": "survival_TCGA.csv",     "tech": "RNA-seq"},
}
for cohort, info in cohort_info.items():
    row = cohort_summary.loc[cohort]
    info["n_patients"] = int(row["n_patients_with_survival"])
    info["events"] = int(row["n_events"])
    stats[f"{cohort}_n_clinical_rows"] = int(row["n_clinical_rows"])
    stats[f"{cohort}_n_tf_samples"] = int(row["n_tf_samples"])
    stats[f"{cohort}_n_tfs"] = int(row["n_tfs_cohort"])
    stats[f"{cohort}_n_patients_after_dedup"] = int(row["n_patients_after_dedup"])
    stats[f"{cohort}_n_patients"] = info["n_patients"]
    stats[f"{cohort}_n_events"] = info["events"]

rows_t2 = []
cohort_dfs = {}
for cohort, info in cohort_info.items():
    df = load_csv(RESULTS / "v8_multicohort" / info["file"], info["file"])
    if df is not None:
        cohort_dfs[cohort] = df
        n_sig = (df["padj"] < 0.05).sum()
        n_total = len(df)
        pct = round(100 * n_sig / n_total, 1) if n_total > 0 else 0
        rows_t2.append({
            "Cohort": cohort,
            "Technology": info["tech"],
            "N_patients": info["n_patients"],
            "N_events": info["events"],
            "N_drugs_tested": n_total,
            "Drugs_FDR_005": n_sig,
            "Pct_significant": pct,
        })
        stats[f"{cohort}_drugs_FDR005"] = int(n_sig)
        stats[f"{cohort}_n_drugs_tested"] = int(n_total)

# Fisher meta-analysis
df_fisher = load_csv(RESULTS / "v8_multicohort" / "meta_analysis_fisher.csv", "meta_analysis_fisher")
if df_fisher is not None:
    fisher_sig = (df_fisher["fisher_padj"] < 0.05).sum()
    fisher_total = len(df_fisher)
    rows_t2.append({
        "Cohort": "Fisher_meta",
        "Technology": "Combined",
        "N_patients": "-",
        "N_events": "-",
        "N_drugs_tested": fisher_total,
        "Drugs_FDR_005": fisher_sig,
        "Pct_significant": round(100 * fisher_sig / fisher_total, 1),
    })
    stats["Fisher_drugs_FDR005"] = int(fisher_sig)

t2 = pd.DataFrame(rows_t2)
t2.to_csv(OUT / "table2_multicohort_summary.csv", index=False)
print(t2.to_string(index=False))


# ===================================================================
# TABLE 3 - Positive control drugs
# ===================================================================
print("\n" + "=" * 60)
print("TABLE 3: Positive control drugs")
print("=" * 60)

positive_controls = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
rows_t3 = []

for drug in positive_controls:
    row = {}
    row["Drug"] = drug

    # SCAN-B
    if "SCANB" in cohort_dfs:
        s = cohort_dfs["SCANB"]
        match = s[s["drug"] == drug]
        if len(match):
            row["SCANB_HR"] = round(match.iloc[0]["HR"], 2)
            row["SCANB_padj"] = f"{match.iloc[0]['padj']:.2e}"
        else:
            row["SCANB_HR"] = "n.a."
            row["SCANB_padj"] = "n.a."

    # METABRIC
    if "METABRIC" in cohort_dfs:
        m = cohort_dfs["METABRIC"]
        match = m[m["drug"] == drug]
        if len(match):
            row["METABRIC_HR"] = round(match.iloc[0]["HR"], 2)
            padj_m = match.iloc[0]["padj"]
            row["METABRIC_padj"] = f"{padj_m:.2e}" if padj_m < 0.05 else "n.s."
        else:
            row["METABRIC_HR"] = "n.a."
            row["METABRIC_padj"] = "n.a."

    # Fisher
    if df_fisher is not None:
        f_match = df_fisher[df_fisher["drug"] == drug]
        if len(f_match):
            row["Fisher_padj"] = f"{f_match.iloc[0]['fisher_padj']:.2e}"
            row["Fisher_mean_HR"] = round(f_match.iloc[0]["mean_HR"], 2)
        else:
            row["Fisher_padj"] = "n.a."
            row["Fisher_mean_HR"] = "n.a."

    rows_t3.append(row)
    stats[f"positive_control_{drug}_fisher_padj"] = row.get("Fisher_padj", "n.a.")

t3 = pd.DataFrame(rows_t3)
t3.to_csv(OUT / "table3_positive_controls.csv", index=False)
print(t3.to_string(index=False))


# ===================================================================
# TABLE 4 - AURORA TF comparison (v9_aurora)
# ===================================================================
print("\n" + "=" * 60)
print("TABLE 4: AURORA TF changes (primary vs metastasis)")
print("=" * 60)

df_aurora = load_csv(RESULTS / "v9_aurora" / "aurora_tf_ig_validation.csv", "aurora_tf_ig_validation")
if df_aurora is not None:
    t4 = df_aurora[["tf", "primary_mean", "metastasis_mean", "delta",
                    "pvalue", "bonferroni_p", "bonferroni_sig"]].copy()
    t4.columns = ["TF", "Primary_mean", "Metastasis_mean", "Delta",
                  "pvalue", "Bonferroni_p", "Bonferroni_sig"]
    t4["Primary_mean"] = t4["Primary_mean"].round(2)
    t4["Metastasis_mean"] = t4["Metastasis_mean"].round(2)
    t4["Delta"] = t4["Delta"].round(2)
    t4.to_csv(OUT / "table4_aurora_tfs.csv", index=False)
    print(t4.to_string(index=False))

    n_bonf = t4["Bonferroni_sig"].sum()
    stats["aurora_bonferroni_sig"] = int(n_bonf)
    stats["aurora_total_tfs_tested"] = len(t4)
    stats["aurora_all_nominal_sig"] = int((t4["pvalue"] < 0.05).sum())


# ===================================================================
# TABLE 5 - Rule-based candidates (PAM50-adjusted Cox)
# ===================================================================
print("\n" + "=" * 60)
print("TABLE 5: Rule-based repurposing candidates (PAM50-adjusted)")
print("=" * 60)

df_pam = load_csv(RESULTS / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv", "cox_multivar_pam50_adjusted")
df_pam05 = load_csv(RESULTS / "v6_strengthen" / "cox_multivar_pam50_pen05.csv", "cox_multivar_pam50_pen05")

# Rule-based candidate selection (Methods, "Rule-based candidate selection").
# Four filters and one cutoff, applied in this order to the PAM50-adjusted
# penalized Cox results. No expert scoring: the mechanism class and the
# development stage come from the PRISM annotations as distributed (19Q4),
# and the rank cutoff is an explicit choice whose sensitivity is reported in
# Supplementary Table S1.
SELECTION_RULE = {
    "padj_max": 0.05,            # filter (i), penalizer 0.1
    "hr_range": (1.5, 10.0),     # filter (i), penalizer 0.1
    "stability_padj_max": 0.05,  # filter (ii), penalizer 0.5
    "min_phase": 2,              # filter (iv), PRISM phase annotation
    "rank_cutoff": 30,           # cutoff (v), rank in the filter (i) ranking
}
# PRISM MOA strings denoting protein kinase or HDAC inhibitors (filter iii).
# Kept explicit so that the class filter is reproducible from the annotation.
KINASE_HDAC_MOA = re.compile(
    r"kinase|egfr|\balk\b|mek|\bsrc\b|hdac|mtor|pi3k|cdk|braf|vegfr|jak|fak|"
    r"aurora|plk|bcr-abl|\babl\b|flt3|btk|igf|fgfr|pdgfr|\bkit\b|\braf\b|erk|"
    r"akt|syk|lck|her2|erbb|ros1|\bret\b|\bmet\b|trk|axl|tyrosine",
    re.IGNORECASE,
)
# PRISM phase labels mapped to a development stage number.
PHASE_RANK = {
    "Preclinical": 0, "Phase 1": 1, "Phase 1/Phase 2": 2, "Phase 2": 2,
    "Phase 2/Phase 3": 3, "Phase 3": 3, "Launched": 4, "Withdrawn": 4,
}
SENSITIVITY_CUTOFFS = (26, 30, 35, 50)

if df_pam is not None and df_pam05 is not None:
    annot = load_csv_required(DATA / "precision_processed" / "prism_drug_annotations.csv",
                              "prism_drug_annotations")
    annot = annot.drop_duplicates(subset="name")
    annot["name_l"] = annot["name"].str.lower()

    lo, hi = SELECTION_RULE["hr_range"]
    step1 = df_pam[(df_pam["padj"] < SELECTION_RULE["padj_max"])
                   & (df_pam["HR"] >= lo) & (df_pam["HR"] <= hi)].copy()
    step1 = step1.sort_values("padj").reset_index(drop=True)
    step1["rank"] = step1.index + 1
    step1["name_l"] = step1["drug"].str.lower()
    step1 = step1.merge(annot[["name_l", "moa", "target", "phase", "disease_area"]],
                        on="name_l", how="left").drop(columns="name_l")
    pam05_idx = df_pam05.set_index("drug")
    step1["HR_pen05"] = step1["drug"].map(pam05_idx["HR"])
    step1["padj_pen05"] = step1["drug"].map(pam05_idx["padj"])
    step1["pass_stability"] = step1["padj_pen05"] < SELECTION_RULE["stability_padj_max"]
    step1["pass_class"] = step1["moa"].fillna("").str.contains(KINASE_HDAC_MOA)
    step1["pass_phase"] = (step1["phase"].map(lambda p: PHASE_RANK.get(str(p).strip(), 0))
                           >= SELECTION_RULE["min_phase"])
    step1["pass_rank"] = step1["rank"] <= SELECTION_RULE["rank_cutoff"]
    extended_mask = step1[["pass_stability", "pass_class", "pass_phase"]].all(axis=1)
    step1["selected"] = extended_mask & step1["pass_rank"]
    extended = step1[extended_mask]
    selected = step1[step1["selected"]]

    stats["candidates_n_statistical_filter"] = int(len(step1))
    stats["candidates_n_stable_pen05"] = int(step1["pass_stability"].sum())
    stats["candidates_n_kinase_hdac"] = int((step1["pass_stability"] & step1["pass_class"]).sum())
    stats["candidates_n_extended_set"] = int(len(extended))
    stats["candidates_n_selected"] = int(len(selected))
    stats["candidates_rank_cutoff"] = SELECTION_RULE["rank_cutoff"]
    stats["candidates_selected"] = selected["drug"].tolist()
    stats["candidates_extended_set"] = extended["drug"].tolist()
    for n in SENSITIVITY_CUTOFFS:
        stats[f"candidates_n_at_cutoff_{n}"] = int((extended["rank"] <= n).sum())
    stats["candidates_extra_at_cutoff_50"] = extended[
        (extended["rank"] > SELECTION_RULE["rank_cutoff"]) & (extended["rank"] <= 50)
    ]["drug"].tolist()

    # Supplementary Table S1: every drug passing filter (i), with all flags.
    supp_cols = ["rank", "drug", "HR", "padj", "HR_pen05", "padj_pen05", "moa", "target",
                 "phase", "disease_area", "pass_stability", "pass_class", "pass_phase",
                 "pass_rank", "selected"]
    step1[supp_cols].to_csv(OUT / "supp_table_S1_candidate_selection.csv", index=False)

    # Table 5, ordered by HR at penalizer 0.1 as in the manuscript.
    t5 = selected.sort_values("HR", ascending=False)
    t5 = pd.DataFrame({
        "Drug": t5["drug"],
        "MOA": t5["moa"],
        "Phase": t5["phase"],
        "HR_PAM50adj": t5["HR"].round(2),
        "padj_PAM50adj": t5["padj"].map(lambda v: f"{v:.2e}"),
        "HR_pen05": t5["HR_pen05"].round(2),
        "padj_pen05": t5["padj_pen05"].map(lambda v: f"{v:.2e}"),
    })
    t5.to_csv(OUT / "table5_candidates.csv", index=False)
    print(f"  filter (i) {len(step1)} -> stability {stats['candidates_n_stable_pen05']} "
          f"-> kinase/HDAC {stats['candidates_n_kinase_hdac']} -> phase>=2 {len(extended)} "
          f"-> rank<={SELECTION_RULE['rank_cutoff']} {len(selected)}")
    print(t5.to_string(index=False))


# ===================================================================
# Supplementary Note 1: IG on the extended candidate set (scripts 32, 35,
# 36 and 49 with --candidate-set extended). Optional: the keys are only
# written when the comparison exists.
# ===================================================================
ext_cmp_path = RESULTS / "v4_xai" / "ig_extended_comparison.csv"
if ext_cmp_path.exists():
    ext_cmp = pd.read_csv(ext_cmp_path).set_index("metric")["value"]
    for key in ("n_drugs_main", "n_drugs_extended", "global_top20_overlap",
                "core10_overlap", "aurora_main_bonferroni", "aurora_extended_bonferroni",
                "aurora_extended_nominal", "aurora_extended_n_tfs"):
        if key in ext_cmp.index:
            stats[f"ig_extended_{key}"] = int(float(ext_cmp[key]))
    for key in ("global_spearman_all_tfs", "stability_spearman_mean_rank"):
        if key in ext_cmp.index:
            stats[f"ig_extended_{key}"] = round(float(ext_cmp[key]), 4)
    for key in ("core10_shared", "core10_only_main", "core10_only_extended",
                "global_top20_only_main", "global_top20_only_extended"):
        if key in ext_cmp.index:
            stats[f"ig_extended_{key}"] = str(ext_cmp[key]) if pd.notna(ext_cmp[key]) else ""
    print("\nSupplementary Note 1 (extended candidate set):")
    print(ext_cmp.to_string())
else:
    print("\n  [WARNING] ig_extended_comparison.csv not found: Supplementary Note 1 keys not written")


# ===================================================================
# Within-subtype survival screens (script 19b): Basal alone and the
# Basal + HER2 pool, one sample per patient (Results, SFig12).
# ===================================================================
sub_path = RESULTS / "v7_weaknesses" / "subtype_pooling_summary.csv"
if sub_path.exists():
    sub = pd.read_csv(sub_path).set_index("subset")
    for name, key in (("Basal", "subtype_basal"), ("Basal_HER2", "subtype_pooled")):
        if name in sub.index:
            stats[f"{key}_n_patients"] = int(sub.loc[name, "n_patients"])
            stats[f"{key}_n_events"] = int(sub.loc[name, "n_events"])
            stats[f"{key}_n_tested"] = int(sub.loc[name, "n_tested"])
            stats[f"{key}_drugs_FDR005"] = int(sub.loc[name, "n_fdr05"])
            stats[f"{key}_drugs_FDR010"] = int(sub.loc[name, "n_fdr10"])
    print("\nWithin-subtype screens (script 19b):")
    print(sub.to_string())
else:
    print("\n  [WARNING] subtype_pooling_summary.csv not found: within-subtype keys not written")


# ===================================================================
# Sensitivity analyses and training history: keys for the numbers quoted in
# Methods, Results and Discussion (scripts 50, 40, 41, 37, 39).
# ===================================================================
hist_path = RESULTS / "gnn_training_history.csv"
if hist_path.exists():
    hist = pd.read_csv(hist_path)
    best_i = hist["test_pearson"].idxmax()
    stats["training_epochs"] = int(len(hist))
    stats["training_final_test_pearson"] = round(float(hist["test_pearson"].iloc[-1]), 3)
    stats["training_best_test_pearson"] = round(float(hist.loc[best_i, "test_pearson"]), 3)
    stats["training_best_epoch"] = int(hist.loc[best_i, "epoch"])
    print(f"\nTraining history (script 50): {stats['training_epochs']} epochs, final test Pearson "
          f"{stats['training_final_test_pearson']}, best {stats['training_best_test_pearson']} "
          f"at epoch {stats['training_best_epoch']}")

rfs_path = RESULTS / "v8_multicohort" / "cox_rfs_comparison.csv"
if rfs_path.exists():
    rfs = pd.read_csv(rfs_path)
    rfs["padj_RFS_num"] = pd.to_numeric(rfs["padj_RFS"], errors="coerce")
    sig = rfs[rfs["padj_RFS_num"] < 0.05]
    stats["rfs_n_drugs"] = int(len(rfs))
    stats["rfs_n_direction_concordant"] = int((rfs["same_direction"].astype(str).str.lower() == "yes").sum())
    stats["rfs_n_fdr05"] = int(len(sig))
    stats["rfs_fdr05_drugs"] = sig["drug"].tolist()
    print(f"RFS (script 40): {stats['rfs_n_direction_concordant']}/{stats['rfs_n_drugs']} same direction, "
          f"{stats['rfs_n_fdr05']} at FDR<0.05: {stats['rfs_fdr05_drugs']}")

combat_path = RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv"
if combat_path.exists():
    cb = pd.read_csv(combat_path).set_index("metric")["value"]
    stats["combat_n_sig"] = int(cb["n_sig_combat"])
    stats["qt_n_sig"] = int(cb["n_sig_qt"])
    stats["combat_qt_n_both"] = int(cb["n_sig_both"])
    stats["combat_qt_direction_pct"] = round(float(cb["direction_concordant_pct_among_both_sig"]), 1)
    print(f"ComBat vs QT (script 41): {stats['combat_n_sig']} vs {stats['qt_n_sig']}, "
          f"{stats['combat_qt_n_both']} both, {stats['combat_qt_direction_pct']}% concordant")

clin_path = RESULTS / "v8_multicohort" / "cox_clinical_comparison.csv"
if clin_path.exists():
    cl = pd.read_csv(clin_path)
    cand = cl[~cl["drug"].isin(positive_controls)]
    sig_s = cand[pd.to_numeric(cand["padj_clinical_scanb"], errors="coerce") < 0.05]
    sig_m = cand[pd.to_numeric(cand["padj_clinical_metabric"], errors="coerce") < 0.05]
    stats["clinical_adj_n_candidates"] = int(len(cand))
    stats["clinical_adj_scanb_n_candidates_fdr05"] = int(len(sig_s))
    stats["clinical_adj_scanb_candidates_fdr05"] = sig_s["drug"].tolist()
    stats["clinical_adj_scanb_hr_min"] = round(float(sig_s["HR_clinical_scanb"].min()), 2)
    stats["clinical_adj_scanb_hr_max"] = round(float(sig_s["HR_clinical_scanb"].max()), 2)
    stats["clinical_adj_metabric_n_candidates_fdr05"] = int(len(sig_m))
    stats["clinical_adj_metabric_candidates_fdr05"] = sig_m["drug"].tolist()
    print(f"Clinical covariates (script 37): SCAN-B {len(sig_s)}/{len(cand)} candidates FDR<0.05 "
          f"(HR {stats['clinical_adj_scanb_hr_min']}-{stats['clinical_adj_scanb_hr_max']}), METABRIC {sig_m['drug'].tolist()}")

site_path = RESULTS / "v9_aurora" / "aurora_tf_by_site_summary.csv"
if site_path.exists():
    import io as _io
    sections = []
    for chunk in open(site_path, encoding="utf-8").read().split("#"):
        body = "\n".join(l for l in chunk.strip().splitlines()[1:] if l.strip()) if chunk.strip() else ""
        if body.startswith("tf,"):
            sections.append(pd.read_csv(_io.StringIO(body)).set_index("tf"))
    if sections:
        delta = sections[0]
        signs = delta.gt(0).sum(axis=1)
        fully = delta.index[(signs == delta.shape[1]) | (signs == 0)].tolist()
        stats["aurora_site_n_sites"] = int(delta.shape[1])
        stats["aurora_site_fully_concordant_tfs"] = fully
        print(f"AURORA by site (script 39): {delta.shape[1]} sites, fully concordant TFs: {fully}")


# ===================================================================
# SHARED DRUGS + direction consistency
# ===================================================================
print("\n" + "=" * 60)
print("SHARED DRUGS (FDR<0.05 in both SCAN-B and METABRIC)")
print("=" * 60)

if "SCANB" in cohort_dfs and "METABRIC" in cohort_dfs:
    s_sig = cohort_dfs["SCANB"][cohort_dfs["SCANB"]["padj"] < 0.05]
    m_sig = cohort_dfs["METABRIC"][cohort_dfs["METABRIC"]["padj"] < 0.05]
    shared_drugs = set(s_sig["drug"]) & set(m_sig["drug"])
    stats["shared_drugs_both_cohorts"] = len(shared_drugs)

    merged = s_sig.merge(m_sig, on="drug", suffixes=("_s", "_m"))
    merged["same_direction"] = ((merged["HR_s"] > 1) & (merged["HR_m"] > 1)) | \
                               ((merged["HR_s"] < 1) & (merged["HR_m"] < 1))
    n_consistent = int(merged["same_direction"].sum())
    n_inconsistent = len(merged) - n_consistent
    stats["shared_drugs_direction_consistent"] = n_consistent
    stats["shared_drugs_direction_inconsistent"] = n_inconsistent
    stats["shared_drugs_direction_pct"] = round(100 * n_consistent / len(merged), 1)

    shared_out = merged[["drug", "HR_s", "padj_s", "HR_m", "padj_m", "same_direction"]].copy()
    shared_out = shared_out.sort_values("padj_s")
    shared_out.to_csv(OUT / "shared_drugs.csv", index=False)

    print(f"  Shared drugs: {len(shared_drugs)}")
    print(f"  Direction consistent: {n_consistent}")
    print(f"  Direction inconsistent: {n_inconsistent}")
    print(f"  Concordance: {stats['shared_drugs_direction_pct']}%")


# ===================================================================
# TF IMPORTANCE via Integrated Gradients (Captum, canonical XAI)
# ===================================================================
print("\n" + "=" * 60)
print("TF IMPORTANCE via Integrated Gradients (canonical)")
print("=" * 60)

ig_path = RESULTS / "v4_xai" / "tf_importance_ig_seed42.csv"
df_ig = pd.read_csv(ig_path, comment="#") if ig_path.exists() else None
if df_ig is not None:
    # Samples per TF = (TNBC cell line, drug) pairs attributed per seed, read
    # from the long-form attribution table of script 32 (12 cells x 11 drugs)
    ig_pairs = pd.read_csv(
        RESULTS / "v4_xai" / "ig_full_attributions.csv",
        usecols=["seed", "cell_depmap_id", "drug"],
    ).drop_duplicates()
    pairs_per_seed = ig_pairs.groupby("seed").size()
    stats["ig_n_seeds"] = int(len(pairs_per_seed))
    stats["ig_n_cells"] = int(ig_pairs["cell_depmap_id"].nunique())
    stats["ig_n_drugs"] = int(ig_pairs["drug"].nunique())
    stats["ig_n_samples_per_tf"] = int(pairs_per_seed.iloc[0]) if pairs_per_seed.nunique() == 1 else None
    stats["ig_n_steps"] = 50  # n_steps of Captum IntegratedGradients in script 32
    for i in range(min(10, len(df_ig))):
        stats[f"ig_top{i+1}_name"] = df_ig.iloc[i]["tf"]
        stats[f"ig_top{i+1}_importance"] = round(df_ig.iloc[i]["importance_mean"], 5)
    print(df_ig.head(10)[["tf", "importance_mean", "rank"]].to_string(index=False))

# Multi-seed IG stability (internal review item)
ig_stab_path = RESULTS / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv"
if ig_stab_path.exists():
    df_stab = pd.read_csv(ig_stab_path)
    robust_top20 = df_stab[df_stab["in_top20_all_seeds"]]["tf"].tolist()
    stats["ig_multi_seed_top20_all_n"] = len(robust_top20)
    stats["ig_multi_seed_top20_all_names"] = robust_top20
    print(f"  Multi-seed stable TFs (top-20 across 3 seeds): {robust_top20}")

# IG baseline robustness (internal review item)
ig_base_path = RESULTS / "v4_xai" / "ig_baseline_stability.csv"
if ig_base_path.exists():
    from scipy.stats import spearmanr
    m = pd.read_csv(RESULTS / "v4_xai" / "ig_baseline_mean.csv")
    r = pd.read_csv(RESULTS / "v4_xai" / "ig_baseline_random.csv")
    merged = m[["tf", "rank"]].merge(r[["tf", "rank"]], on="tf",
                                      suffixes=("_mean", "_random"))
    rho_mr, _ = spearmanr(merged["rank_mean"], merged["rank_random"])
    stats["ig_baseline_spearman_mean_vs_random"] = round(float(rho_mr), 3)
    print(f"  IG baseline stability (mean vs random): rho={rho_mr:.3f}")


# ===================================================================
# Additional paper statistics
# ===================================================================
print("\n" + "=" * 60)
print("ADDITIONAL STATISTICS")
print("=" * 60)

# Knowledge graph stats: node and edge counts of the prebuilt graph, written by
# scripts/51_graph_summary.py (the graph itself is not redistributed)
graph_summary = load_csv_required(RESULTS / "graph_summary.csv", "graph_summary.csv")
kg = dict(zip(graph_summary["type"], graph_summary["count"].astype(int)))
stats["kg_genes"] = kg["gene"]
stats["kg_tfs"] = kg["tf"]
stats["kg_drugs"] = kg["drug"]
stats["kg_cell_lines"] = kg["cell_line"]
stats["kg_nodes"] = stats["kg_genes"] + stats["kg_tfs"] + stats["kg_drugs"] + stats["kg_cell_lines"]
stats["collectri_edges"] = kg["tf_regulates_gene"]
stats["ppi_edges_bidir"] = kg["gene_interacts_gene"]
stats["ppi_interactions"] = stats["ppi_edges_bidir"] // 2
stats["drug_target_edges"] = kg["drug_targets_gene"]
stats["response_edges"] = kg["cell_line_responds_to_drug"]
stats["kg_edges_total"] = (stats["collectri_edges"] + stats["ppi_edges_bidir"]
                           + stats["drug_target_edges"] + stats["response_edges"])
stats["kg_edges_approx"] = int(round(stats["kg_edges_total"], -4))
print(f"Knowledge graph: {stats['kg_nodes']} nodes, {stats['kg_edges_total']} edges "
      f"(CollecTRI {stats['collectri_edges']}, PPI {stats['ppi_interactions']} x2, "
      f"drug-target {stats['drug_target_edges']}, response {stats['response_edges']})")

# METABRIC PAM50 from genefu (script 20) vs the cBioPortal CLAUDIN_SUBTYPE labels,
# computed by script 21 on the clinical table (Methods: 73% excluding claudin-low)
met_row = cohort_summary.loc["METABRIC"]
if "pam50_genefu_concordance_pct" in met_row.index and pd.notna(met_row["pam50_genefu_concordance_pct"]):
    stats["metabric_pam50_genefu_n_compared"] = int(met_row["pam50_genefu_n_compared"])
    stats["metabric_pam50_genefu_concordance_pct"] = round(float(met_row["pam50_genefu_concordance_pct"]), 1)
    stats["metabric_pam50_genefu_concordance_all_pct"] = round(float(met_row["pam50_genefu_concordance_all_pct"]), 1)
    print(f"METABRIC genefu PAM50 vs CLAUDIN_SUBTYPE: {stats['metabric_pam50_genefu_concordance_pct']}% "
          f"of {stats['metabric_pam50_genefu_n_compared']} (all labels: {stats['metabric_pam50_genefu_concordance_all_pct']}%)")

# Cross-screen validation PRISM -> GDSC (script 05, SFig07, Discussion)
cv = load_csv(RESULTS / "crossval_prism_to_gdsc.csv", "crossval_prism_to_gdsc.csv")
if cv is not None:
    stats["crossval_n_drugs"] = int(len(cv))
    stats["crossval_median_pearson"] = round(float(cv["pearson_r"].median()), 3)
    stats["crossval_mean_pearson"] = round(float(cv["pearson_r"].mean()), 3)
    print(f"Cross-screen PRISM->GDSC: {len(cv)} drugs, median Pearson {stats['crossval_median_pearson']}")

# AURORA-US cohort sizes (script 22, aurora_cohort_summary.csv): Methods and Results
aur_summary = load_csv_required(
    RESULTS / "v9_aurora" / "aurora_cohort_summary.csv", "aurora_cohort_summary.csv"
).iloc[0]
for key, value in aur_summary.items():
    stats[f"aurora_cohort_{key}"] = int(value)
print(f"AURORA cohort: {aur_summary.to_dict()}")

# Follow-up of the analysed cohorts (script 21): median observed time and
# reverse Kaplan-Meier median follow-up
for cohort in cohort_info:
    row = cohort_summary.loc[cohort]
    for col in ("median_os_time_years", "median_followup_years_reverse_km"):
        if col in row.index and pd.notna(row[col]):
            stats[f"{cohort}_{col}"] = round(float(row[col]), 2)

# QQ-plot genomic inflation per cohort (SFig05 caption): median chi2 / 0.4549
from scipy.stats import chi2 as _chi2
for cohort, df_c in cohort_dfs.items():
    med_chi2 = float(np.median(_chi2.isf(df_c["pvalue"].clip(lower=1e-300), 1)))
    stats[f"{cohort}_qq_lambda"] = round(med_chi2 / _chi2.ppf(0.5, 1), 2)
print("QQ lambda:", {c: stats[f'{c}_qq_lambda'] for c in cohort_dfs})

# Data description constants cited in Methods, written by scripts/53_data_summary.py
# (the matrices themselves are not redistributed): PRISM and GDSC response
# matrices, expression genes, MOA encoding, hold-out split. CollecTRI target
# genes come from the graph summary (script 51).
data_summary = load_csv_required(RESULTS / "data_summary.csv", "data_summary.csv")
for key, value in zip(data_summary["type"], data_summary["count"].astype(int)):
    stats[key] = int(value)
if "tf_regulates_gene_targets" in kg:
    stats["collectri_target_genes"] = int(kg["tf_regulates_gene_targets"])
print(f"Data constants: PRISM {stats.get('prism_n_cell_lines')}x{stats.get('prism_n_drugs')}, "
      f"expression genes {stats.get('expression_n_genes')}, GDSC {stats.get('gdsc_n_cell_lines')}x{stats.get('gdsc_n_drugs')}, "
      f"MOA {stats.get('moa_n_drugs_annotated')} drugs / {stats.get('moa_n_categories')} classes, "
      f"CollecTRI targets {stats.get('collectri_target_genes')}, split {stats.get('holdout_n_train_cells')}/{stats.get('holdout_n_test_cells')}")

# MOA and target enrichment among Fisher-significant drugs (script 52, SFig06 and SFig07)
moa_enr = load_csv_required(RESULTS / "v8_multicohort" / "moa_enrichment.csv", "moa_enrichment.csv")
tgt_enr = load_csv_required(RESULTS / "v8_multicohort" / "target_enrichment.csv", "target_enrichment.csv")
top_moa = moa_enr.sort_values("pvalue").iloc[0]
stats["moa_enrichment_n_classes_tested"] = int(len(moa_enr))
stats["moa_enrichment_n_classes_fdr05"] = int((moa_enr["padj"] < 0.05).sum())
stats["moa_enrichment_top_class"] = str(top_moa["moa"])
stats["moa_enrichment_top_n_sig"] = int(top_moa["n_sig"])
stats["moa_enrichment_top_n_total"] = int(top_moa["n_total"])
stats["moa_enrichment_top_pct"] = round(float(top_moa["pct_sig"]), 1)
stats["moa_enrichment_top_padj"] = float(f"{top_moa['padj']:.2e}")
sig_targets = tgt_enr[tgt_enr["padj"] < 0.05].sort_values("pvalue")["target"].tolist()
stats["target_enrichment_n_targets_tested"] = int(len(tgt_enr))
stats["target_enrichment_n_fdr05"] = int(len(sig_targets))
stats["target_enrichment_fdr05_targets"] = sig_targets
print(f"Enrichment: {stats['moa_enrichment_top_class']} {stats['moa_enrichment_top_n_sig']}/{stats['moa_enrichment_top_n_total']} "
      f"(padj {stats['moa_enrichment_top_padj']}), targets FDR<0.05: {sig_targets}")

# Within-subtype pooled screen (SFig12 legend): nominal and Bonferroni counts
pooled_path = RESULTS / "v7_weaknesses" / "basal_her2_survival.csv"
if pooled_path.exists():
    pooled = pd.read_csv(pooled_path)
    stats["subtype_pooled_n_nominal_p05"] = int((pooled["pvalue"] < 0.05).sum())
    stats["subtype_pooled_n_bonferroni"] = int((pooled["pvalue"] < 0.05 / len(pooled)).sum())

# SCAN-B vs METABRIC p-value concordance (SFig13 annotation)
if "SCANB" in cohort_dfs and "METABRIC" in cohort_dfs:
    _m = cohort_dfs["SCANB"][["drug", "pvalue"]].merge(cohort_dfs["METABRIC"][["drug", "pvalue"]],
                                                       on="drug", suffixes=("_s", "_m"))
    _rho, _ = spearmanr(-np.log10(_m["pvalue_s"].clip(lower=1e-30)), -np.log10(_m["pvalue_m"].clip(lower=1e-30)))
    stats["scanb_metabric_neglogp_spearman"] = round(float(_rho), 3)

# IG baseline robustness: top-20 overlaps between baselines (Methods)
if ig_base_path.exists():
    _b = pd.read_csv(ig_base_path)
    stats["ig_baseline_top20_overlap_mean_random"] = int((_b["in_top20_mean"] & _b["in_top20_random"]).sum())
    stats["ig_baseline_top20_overlap_mean_zero"] = int((_b["in_top20_mean"] & _b["in_top20_zero"]).sum())
    stats["ig_baseline_top20_overlap_all_three"] = int(_b["in_top20_all"].sum())

# Subtype analysis (v6_strengthen)
df_sub = load_csv(RESULTS / "v6_strengthen" / "test3_subtype_summary.csv", "subtype_summary")
if df_sub is not None:
    for _, row in df_sub.iterrows():
        name = row.iloc[0] if pd.notna(row.iloc[0]) else "unnamed"
        # script 18 keeps every sample (no patient-level dedup) and is not cited: kept for reference
        stats[f"test3_samplelevel_{name}_n_sig"] = int(row["n_sig"]) if pd.notna(row["n_sig"]) else 0

# Graph ablation
df_abl = load_csv(RESULTS / "v6_strengthen" / "graph_ablation.csv", "graph_ablation")
if df_abl is not None:
    for _, row in df_abl.iterrows():
        stats[f"ablation_{row['config']}_pearson"] = round(row["test_pearson"], 3)
    print("\n  Graph ablation:")
    print(df_abl[["config", "test_pearson"]].to_string(index=False))

# TCGA nominal paclitaxel
if "TCGA" in cohort_dfs:
    tcga = cohort_dfs["TCGA"]
    pac = tcga[tcga["drug"] == "paclitaxel"]
    if len(pac):
        stats["tcga_paclitaxel_pvalue"] = round(pac.iloc[0]["pvalue"], 4)
        stats["tcga_paclitaxel_HR"] = round(pac.iloc[0]["HR"], 2)
        print(f"\n  TCGA paclitaxel: p={pac.iloc[0]['pvalue']:.4f}, HR={pac.iloc[0]['HR']:.2f}")


# ===================================================================
# SAVE paper_statistics.json
# ===================================================================
# Convert numpy types to native Python for JSON serialization
def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

stats_clean = {k: convert_types(v) for k, v in stats.items()}

with open(OUT / "paper_statistics.json", "w") as f:
    json.dump(stats_clean, f, indent=2, default=str)


# ===================================================================
# FINAL SUMMARY - all key paper numbers
# ===================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY - key numbers of the paper")
print("=" * 60)

print(f"""
KNOWLEDGE GRAPH:
  Nodes: {stats.get('kg_nodes')} | Edges: ~{stats.get('kg_edges_approx')}
  TFs: {stats.get('kg_tfs')} | Genes: {stats.get('kg_genes')} | Drugs: {stats.get('kg_drugs')} | Cell lines: {stats.get('kg_cell_lines')}

MODEL COMPARISON (Table 1):
  RF (TF): global r={stats.get('table1_RF_TF_global_pearson')}, median per-drug={stats.get('table1_RF_TF_median_perdrug')}
  Ridge (expr): global r={stats.get('table1_Ridge_expression_global_pearson')}, median per-drug={stats.get('table1_Ridge_expression_median_perdrug')}
  SAGE GNN: global r={stats.get('table1_SAGE_learned_emb_global_pearson')}, median per-drug={stats.get('table1_SAGE_learned_emb_median_perdrug')}

TF IMPORTANCE:
  Top 5: {stats.get('tf_top1_name')} ({stats.get('tf_top1_importance')}), {stats.get('tf_top2_name')} ({stats.get('tf_top2_importance')}), {stats.get('tf_top3_name')}, {stats.get('tf_top4_name')}, {stats.get('tf_top5_name')}

MULTI-COHORT SURVIVAL (Table 2):
  SCAN-B:   {stats.get('SCANB_drugs_FDR005')} drugs FDR<0.05
  METABRIC: {stats.get('METABRIC_drugs_FDR005')} drugs FDR<0.05
  TCGA:     {stats.get('TCGA_drugs_FDR005')} drugs FDR<0.05
  Fisher:   {stats.get('Fisher_drugs_FDR005')} drugs FDR<0.05

SHARED DRUGS:
  Both cohorts: {stats.get('shared_drugs_both_cohorts')}
  Direction consistent: {stats.get('shared_drugs_direction_consistent')} ({stats.get('shared_drugs_direction_pct')}%)

POSITIVE CONTROLS (Table 3):
  Paclitaxel: Fisher padj = {stats.get('positive_control_paclitaxel_fisher_padj')}
  Docetaxel:  Fisher padj = {stats.get('positive_control_docetaxel_fisher_padj')}
  Epirubicin: Fisher padj = {stats.get('positive_control_epirubicin_fisher_padj')}
  Olaparib:   Fisher padj = {stats.get('positive_control_olaparib_fisher_padj')}

AURORA (Table 4):
  TFs Bonferroni sig: {stats.get('aurora_bonferroni_sig')} / {stats.get('aurora_total_tfs_tested')}
  All nominally sig:  {stats.get('aurora_all_nominal_sig')} / {stats.get('aurora_total_tfs_tested')}
""")


# ===================================================================
# COPY intermediate CSVs needed by generate_paper_figures.py
# ===================================================================
print("\n" + "=" * 60)
print("COPY: intermediate CSVs for figure generation")
print("=" * 60)

import shutil

copies = [
    # Multi-cohort survival (Fig5, SFig08-10, SFig15)
    (RESULTS / "v8_multicohort" / "meta_analysis_fisher.csv",        OUT / "meta_analysis_fisher.csv"),
    (RESULTS / "v8_multicohort" / "survival_SCANB.csv",              OUT / "survival_SCANB.csv"),
    (RESULTS / "v8_multicohort" / "survival_METABRIC.csv",           OUT / "survival_METABRIC.csv"),
    (RESULTS / "v8_multicohort" / "survival_TCGA.csv",               OUT / "survival_TCGA.csv"),
    # Cohort sizes (Table 2, Methods dedup chains, AURORA sample counts)
    (RESULTS / "v8_multicohort" / "cohort_summary.csv",              OUT / "cohort_summary.csv"),
    (RESULTS / "v9_aurora" / "aurora_cohort_summary.csv",            OUT / "aurora_cohort_summary.csv"),
    (RESULTS / "graph_summary.csv",                                  OUT / "graph_summary.csv"),
    (RESULTS / "data_summary.csv",                                   OUT / "data_summary.csv"),
    # MOA and target enrichment (SFig06, SFig07, script 52)
    (RESULTS / "v8_multicohort" / "moa_enrichment.csv",              OUT / "moa_enrichment.csv"),
    (RESULTS / "v8_multicohort" / "target_enrichment.csv",           OUT / "target_enrichment.csv"),
    # SCAN-B per-drug predictions (Fig6, Fig7): NOT copied here (220 MB).
    # generate_paper_figures.py reads it directly from RESULTS_DIR/v5_final/
    # (see data/README.md). It is regenerated by scripts/17_fix_all_critical.py.
    # AURORA TF data (Fig8)
    (RESULTS / "v9_aurora" / "aurora_tf_activities.csv",             OUT / "aurora_tf_activities.csv"),
    # XAI explanations (SFig01, SFig02) - ablation-based subgraph kept as
    # illustrative example only; canonical XAI method is Integrated Gradients.
    (RESULTS / "v4_xai" / "drug_explanations_v2.csv",               OUT / "drug_explanations_v2.csv"),
    (RESULTS / "v4_xai" / "tnbc_drug_ranking_v2.csv",               OUT / "tnbc_drug_ranking_v2.csv"),
    # Canonical IG attributions (Fig 5)
    (RESULTS / "v4_xai" / "tf_importance_ig_seed42.csv",            OUT / "tf_importance_ig_top20.csv"),
    (RESULTS / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv",
                                                                     OUT / "tf_importance_ig_stability.csv"),
    (RESULTS / "v4_xai" / "ig_baseline_stability.csv",              OUT / "ig_baseline_stability.csv"),
    (RESULTS / "v4_xai" / "tf_drug_ig_signed.csv",                  OUT / "tf_drug_ig_signed.csv"),
    # Cell line embeddings (SFig03)
    (RESULTS / "v3_embeddings" / "cell_line_embeddings.csv",         OUT / "cell_line_embeddings.csv"),
    # Training history (SFig04)
    (RESULTS / "gnn_training_history.csv",                           OUT / "gnn_training_history.csv"),
    # Per-drug model results (SFig05, SFig06)
    (RESULTS / "v2_fair" / "per_drug_RF_TF.csv",                    OUT / "per_drug_RF_TF.csv"),
    (RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv",          OUT / "per_drug_Ridge_expression.csv"),
    (RESULTS / "v2_fair" / "per_drug_SAGE_learned_emb.csv",          OUT / "per_drug_SAGE_learned_emb.csv"),
    # Cross-validation PRISM to GDSC (SFig07)
    (RESULTS / "crossval_prism_to_gdsc.csv",                         OUT / "crossval_prism_to_gdsc.csv"),
    # Basal+HER2 pooled survival (SFig12, script 19b)
    (RESULTS / "v7_weaknesses" / "basal_her2_survival.csv",          OUT / "basal_her2_survival.csv"),
    (RESULTS / "v7_weaknesses" / "basal_only_survival.csv",          OUT / "basal_only_survival.csv"),
    (RESULTS / "v7_weaknesses" / "subtype_pooling_summary.csv",      OUT / "subtype_pooling_summary.csv"),
    # AURORA drug sensitivity (SFig13)
    (RESULTS / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",      OUT / "aurora_drug_primary_vs_meta.csv"),
    # Graph ablation (SFig17)
    (RESULTS / "v6_strengthen" / "graph_ablation.csv",               OUT / "graph_ablation.csv"),
    # PCA / PC1 residualization (Discussion: shared prognostic axis)
    (RESULTS / "v8_multicohort" / "pca_residualization_summary.csv",  OUT / "pca_residualization_summary.csv"),
    (RESULTS / "v8_multicohort" / "pca_residualization_perdrug.csv",  OUT / "pca_residualization_perdrug.csv"),
    # Positive-control forest plot (SFig10): real Cox 95% CIs, SCAN-B vs METABRIC
    (RESULTS / "v8_multicohort" / "forest_positive_controls.csv",     OUT / "forest_positive_controls.csv"),
    # TF-level univariate Cox on SCAN-B OS (Results: MYC/TP53/E2F1 examples, script 44)
    (RESULTS / "v8_multicohort" / "tf_level_survival_scanb.csv",       OUT / "tf_level_survival_scanb.csv"),
    # Cox penalizer sweep (SFig11 + Discussion HR-range claims, script 45)
    (RESULTS / "v8_multicohort" / "penalizer_sweep.csv",              OUT / "penalizer_sweep.csv"),
    # Sensitivity analyses cited in Results and Discussion (scripts 37, 39, 40, 41)
    (RESULTS / "v8_multicohort" / "cox_clinical_comparison.csv",     OUT / "cox_clinical_comparison.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_by_site.csv",                OUT / "aurora_tf_by_site.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_by_site_summary.csv",        OUT / "aurora_tf_by_site_summary.csv"),
    (RESULTS / "v8_multicohort" / "cox_rfs_comparison.csv",          OUT / "cox_rfs_comparison.csv"),
    (RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv",        OUT / "combat_vs_qt_summary.csv"),
    # Supplementary Note 1: IG on the extended candidate set (scripts 32/35/36/49, --candidate-set extended)
    (RESULTS / "v4_xai" / "ig_extended_comparison.csv",              OUT / "ig_extended_comparison.csv"),
    (RESULTS / "v4_xai" / "ig_extended_core_tfs.csv",                OUT / "ig_extended_core_tfs.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_ig_validation_extended.csv",  OUT / "aurora_tf_ig_validation_extended.csv"),
]

n_copied = 0
for src, dst in copies:
    if src.exists():
        shutil.copy2(src, dst)
        n_copied += 1
        print(f"  COPY: {src.name}")
    else:
        print(f"  [WARNING] Not found: {src}")

print(f"\n  Copied: {n_copied} / {len(copies)} files")


print(f"\nResults saved to: {OUT}")
print(f"  - {len(list(OUT.glob('*.csv')))} CSVs")
print(f"  - paper_statistics.json ({len(stats_clean)} values before the sanity-check block)")


# ===================================================================
# SANITY CHECKS (regression asserts)
# ===================================================================
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

sanity_failures = []


def _check(name, condition, detail=""):
    if condition:
        print(f"  [OK] {name}")
    else:
        sanity_failures.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}: {detail}")


# Knowledge graph size
_check(
    "KG has ~23,498 nodes",
    stats_clean.get("kg_nodes", 0) == 23498,
    f"got {stats_clean.get('kg_nodes')}",
)
_check(
    "CollecTRI 42,990 edges",
    stats_clean.get("collectri_edges", 0) == 42990,
    f"got {stats_clean.get('collectri_edges')}",
)
_check(
    "KG nodes 20,389 genes / 1,185 TFs / 1,448 drugs / 476 cell lines (Methods)",
    (stats_clean.get("kg_genes"), stats_clean.get("kg_tfs"), stats_clean.get("kg_drugs"),
     stats_clean.get("kg_cell_lines")) == (20389, 1185, 1448, 476),
    f"got {stats_clean.get('kg_genes')}/{stats_clean.get('kg_tfs')}/{stats_clean.get('kg_drugs')}/{stats_clean.get('kg_cell_lines')}",
)
_check(
    "KG edges: PPI 84,587 interactions (169,174 directed), drug-target 3,442, response 614,325, total ~830,000",
    stats_clean.get("ppi_interactions") == 84587 and stats_clean.get("ppi_edges_bidir") == 169174
    and stats_clean.get("drug_target_edges") == 3442 and stats_clean.get("response_edges") == 614325
    and stats_clean.get("kg_edges_approx") == 830000,
    f"got ppi={stats_clean.get('ppi_interactions')} dt={stats_clean.get('drug_target_edges')} "
    f"resp={stats_clean.get('response_edges')} total={stats_clean.get('kg_edges_total')}",
)

# Cohort sizes (script 21) cited in Table 2 and Methods
_check(
    "Table 2 cohorts: SCAN-B 7,397 / 1,188 events, METABRIC 1,979 / 1,143, TCGA 1,072 / 150",
    (stats_clean.get("SCANB_n_patients"), stats_clean.get("SCANB_n_events")) == (7397, 1188)
    and (stats_clean.get("METABRIC_n_patients"), stats_clean.get("METABRIC_n_events")) == (1979, 1143)
    and (stats_clean.get("TCGA_n_patients"), stats_clean.get("TCGA_n_events")) == (1072, 150),
    f"got SCANB {stats_clean.get('SCANB_n_patients')}/{stats_clean.get('SCANB_n_events')}, "
    f"METABRIC {stats_clean.get('METABRIC_n_patients')}/{stats_clean.get('METABRIC_n_events')}, "
    f"TCGA {stats_clean.get('TCGA_n_patients')}/{stats_clean.get('TCGA_n_events')}",
)
_check(
    "Methods dedup chains: SCAN-B 8,269 samples -> 7,662 patients, TCGA 1,231 -> 1,095",
    (stats_clean.get("SCANB_n_clinical_rows"), stats_clean.get("SCANB_n_patients_after_dedup")) == (8269, 7662)
    and (stats_clean.get("TCGA_n_clinical_rows"), stats_clean.get("TCGA_n_patients_after_dedup")) == (1231, 1095),
    f"got SCANB {stats_clean.get('SCANB_n_clinical_rows')}->{stats_clean.get('SCANB_n_patients_after_dedup')}, "
    f"TCGA {stats_clean.get('TCGA_n_clinical_rows')}->{stats_clean.get('TCGA_n_patients_after_dedup')}",
)
_check(
    "TFs per clinical cohort: SCAN-B 769, METABRIC 767, TCGA 772 (Methods)",
    (stats_clean.get("SCANB_n_tfs"), stats_clean.get("METABRIC_n_tfs"), stats_clean.get("TCGA_n_tfs")) == (769, 767, 772),
    f"got {stats_clean.get('SCANB_n_tfs')}/{stats_clean.get('METABRIC_n_tfs')}/{stats_clean.get('TCGA_n_tfs')}",
)
_check(
    "METABRIC genefu PAM50 concordance 73% excluding claudin-low (Methods)",
    73.0 <= (stats_clean.get("metabric_pam50_genefu_concordance_pct") or 0) < 74.0,
    f"got {stats_clean.get('metabric_pam50_genefu_concordance_pct')}",
)
_check(
    "Cross-screen PRISM->GDSC median per-drug Pearson = 0.052 (Discussion, SFig07)",
    stats_clean.get("crossval_median_pearson") == 0.052,
    f"got {stats_clean.get('crossval_median_pearson')}",
)
_check(
    "IG scope: 12 TNBC cells x 11 drugs = 132 pairs per seed, 3 seeds (Methods)",
    (stats_clean.get("ig_n_cells"), stats_clean.get("ig_n_drugs"), stats_clean.get("ig_n_samples_per_tf"),
     stats_clean.get("ig_n_seeds")) == (12, 11, 132, 3),
    f"got {stats_clean.get('ig_n_cells')}x{stats_clean.get('ig_n_drugs')}={stats_clean.get('ig_n_samples_per_tf')}, seeds {stats_clean.get('ig_n_seeds')}",
)
_check(
    "Median follow-up (reverse Kaplan-Meier): SCAN-B 7.0 years, TCGA 2.7 years (Results, Discussion)",
    round(stats_clean.get("SCANB_median_followup_years_reverse_km") or 0, 1) == 7.0
    and round(stats_clean.get("TCGA_median_followup_years_reverse_km") or 0, 1) == 2.7,
    f"got SCAN-B {stats_clean.get('SCANB_median_followup_years_reverse_km')}, TCGA {stats_clean.get('TCGA_median_followup_years_reverse_km')}",
)
_check(
    "QQ-plot lambda: SCAN-B 7.98, METABRIC 2.72, TCGA 0.84 (SFig05 caption)",
    (stats_clean.get("SCANB_qq_lambda"), stats_clean.get("METABRIC_qq_lambda"), stats_clean.get("TCGA_qq_lambda")) == (7.98, 2.72, 0.84),
    f"got {stats_clean.get('SCANB_qq_lambda')}/{stats_clean.get('METABRIC_qq_lambda')}/{stats_clean.get('TCGA_qq_lambda')}",
)
_check(
    "Data constants: PRISM 476 x 1,448, 19,193 expression genes, GDSC 696 x 397, MOA 1,079 drugs / 155 classes, 6,675 CollecTRI targets (Methods)",
    (stats_clean.get("prism_n_cell_lines"), stats_clean.get("prism_n_drugs")) == (476, 1448)
    and stats_clean.get("expression_n_genes") == 19193
    and (stats_clean.get("gdsc_n_cell_lines"), stats_clean.get("gdsc_n_drugs")) == (696, 397)
    and (stats_clean.get("moa_n_drugs_annotated"), stats_clean.get("moa_n_categories")) == (1079, 155)
    and stats_clean.get("collectri_target_genes") == 6675,
    f"got PRISM {stats_clean.get('prism_n_cell_lines')}x{stats_clean.get('prism_n_drugs')}, genes {stats_clean.get('expression_n_genes')}, "
    f"GDSC {stats_clean.get('gdsc_n_cell_lines')}x{stats_clean.get('gdsc_n_drugs')}, MOA {stats_clean.get('moa_n_drugs_annotated')}/{stats_clean.get('moa_n_categories')}, "
    f"targets {stats_clean.get('collectri_target_genes')}",
)
_check(
    "Cell-line hold-out split: 381 train / 95 test (Methods)",
    (stats_clean.get("holdout_n_train_cells"), stats_clean.get("holdout_n_test_cells")) == (381, 95),
    f"got {stats_clean.get('holdout_n_train_cells')}/{stats_clean.get('holdout_n_test_cells')}",
)
_check(
    "Enrichment: tubulin polymerization inhibitors 25/27 (93%, FDR 3.2e-7), 4 tubulin targets at FDR<0.05 (SFig06, SFig07)",
    stats_clean.get("moa_enrichment_top_class") == "tubulin polymerization inhibitor"
    and (stats_clean.get("moa_enrichment_top_n_sig"), stats_clean.get("moa_enrichment_top_n_total")) == (25, 27)
    and 3.0e-7 <= (stats_clean.get("moa_enrichment_top_padj") or 0) <= 3.3e-7
    and stats_clean.get("moa_enrichment_n_classes_fdr05") == 1
    and set(stats_clean.get("target_enrichment_fdr05_targets") or []) == {"TUBB", "TUBB1", "TUBA4A", "TUBA1A"},
    f"got {stats_clean.get('moa_enrichment_top_class')} {stats_clean.get('moa_enrichment_top_n_sig')}/{stats_clean.get('moa_enrichment_top_n_total')} "
    f"padj {stats_clean.get('moa_enrichment_top_padj')}, classes {stats_clean.get('moa_enrichment_n_classes_fdr05')}, targets {stats_clean.get('target_enrichment_fdr05_targets')}",
)
_check(
    "SFig12 legend: 434 drugs at nominal p<0.05 and 22 after Bonferroni in the Basal+HER2 pool",
    (stats_clean.get("subtype_pooled_n_nominal_p05"), stats_clean.get("subtype_pooled_n_bonferroni")) == (434, 22),
    f"got {stats_clean.get('subtype_pooled_n_nominal_p05')}/{stats_clean.get('subtype_pooled_n_bonferroni')}",
)
_check(
    "SFig13: Spearman rho of -log10 p between SCAN-B and METABRIC = 0.198",
    stats_clean.get("scanb_metabric_neglogp_spearman") == 0.198,
    f"got {stats_clean.get('scanb_metabric_neglogp_spearman')}",
)
_check(
    "AURORA cohort: 129 samples, 53 patients, 44 primary / 79 metastatic / 6 normal, 39 paired, 43 multi-sample, up to 8 metastases (Methods)",
    (stats_clean.get("aurora_cohort_n_samples"), stats_clean.get("aurora_cohort_n_patients"),
     stats_clean.get("aurora_cohort_n_primary"), stats_clean.get("aurora_cohort_n_metastasis"),
     stats_clean.get("aurora_cohort_n_unclassified"), stats_clean.get("aurora_cohort_n_paired_patients"),
     stats_clean.get("aurora_cohort_n_patients_multiple_samples"),
     stats_clean.get("aurora_cohort_max_metastatic_samples_per_patient")) == (129, 53, 44, 79, 6, 39, 43, 8),
    f"got {[stats_clean.get(k) for k in ('aurora_cohort_n_samples', 'aurora_cohort_n_patients', 'aurora_cohort_n_primary', 'aurora_cohort_n_metastasis', 'aurora_cohort_n_unclassified', 'aurora_cohort_n_paired_patients', 'aurora_cohort_n_patients_multiple_samples', 'aurora_cohort_max_metastatic_samples_per_patient')]}",
)

# Survival cohorts
_check(
    "SCAN-B drugs_FDR<0.05 in [550, 700]",
    550 <= stats_clean.get("SCANB_drugs_FDR005", 0) <= 700,
    f"got {stats_clean.get('SCANB_drugs_FDR005')}",
)
_check(
    "METABRIC drugs_FDR<0.05 in [50, 150]",
    50 <= stats_clean.get("METABRIC_drugs_FDR005", 0) <= 150,
    f"got {stats_clean.get('METABRIC_drugs_FDR005')}",
)
_check(
    "Fisher meta drugs_FDR<0.05 in [400, 700]",
    400 <= stats_clean.get("Fisher_drugs_FDR005", 0) <= 700,
    f"got {stats_clean.get('Fisher_drugs_FDR005')}",
)

# Shared drugs direction
_check(
    "shared_drugs_direction_pct >= 95",
    stats_clean.get("shared_drugs_direction_pct", 0) >= 95,
    f"got {stats_clean.get('shared_drugs_direction_pct')}",
)

# IG XAI stability (internal review items)
_check(
    "IG multi-seed top-20 robust in [2, 10]",
    2 <= stats_clean.get("ig_multi_seed_top20_all_n", 0) <= 10,
    f"got {stats_clean.get('ig_multi_seed_top20_all_n')}",
)
_check(
    "IG baseline stability rho(mean vs random) >= 0.85",
    stats_clean.get("ig_baseline_spearman_mean_vs_random", 0) >= 0.85,
    f"got {stats_clean.get('ig_baseline_spearman_mean_vs_random')}",
)

# AURORA (IG TFs)
_check(
    "AURORA Bonferroni sig in [2, 6] (IG top 10)",
    2 <= stats_clean.get("aurora_bonferroni_sig", 0) <= 6,
    f"got {stats_clean.get('aurora_bonferroni_sig')}",
)
_check(
    "AURORA TFs tested == 10",
    stats_clean.get("aurora_total_tfs_tested", 0) == 10,
    f"got {stats_clean.get('aurora_total_tfs_tested')}",
)

# Table 5: the selection rule must reproduce the manuscript (Methods, Results,
# Supplementary Table S1) and its HRs must be biologically plausible.
t5_path = OUT / "table5_candidates.csv"
if t5_path.exists():
    t5 = pd.read_csv(t5_path)
    max_hr = t5["HR_PAM50adj"].astype(float).max()
    _check(
        "Table 5 max HR <= 10 (penalized Cox must avoid extreme HRs)",
        max_hr <= 10.0,
        f"max HR={max_hr:.2f}: check that cox_multivar_pam50_adjusted.csv uses penalizer>=0.1",
    )
    _check(
        "Candidate rule yields the seven drugs of Table 5",
        set(stats_clean.get("candidates_selected", [])) == {
            "osimertinib", "saracatinib", "erlotinib", "brigatinib",
            "pelitinib", "entinostat", "trametinib"},
        f"got {stats_clean.get('candidates_selected')}",
    )
    _check(
        "Candidate rule counts match the manuscript (143 / 52 / 21 / 16 / 7)",
        (stats_clean.get("candidates_n_statistical_filter"),
         stats_clean.get("candidates_n_stable_pen05"),
         stats_clean.get("candidates_n_kinase_hdac"),
         stats_clean.get("candidates_n_extended_set"),
         stats_clean.get("candidates_n_selected")) == (143, 52, 21, 16, 7),
        f"got {[stats_clean.get(k) for k in ('candidates_n_statistical_filter', 'candidates_n_stable_pen05', 'candidates_n_kinase_hdac', 'candidates_n_extended_set', 'candidates_n_selected')]}",
    )
    if "ig_extended_global_top20_overlap" in stats_clean:
        _check(
            "Supplementary Note 1: extended candidate set keeps the global top-20 TFs (20/20)",
            stats_clean.get("ig_extended_global_top20_overlap") == 20,
            f"got {stats_clean.get('ig_extended_global_top20_overlap')}",
        )
        _check(
            "Supplementary Note 1: extended candidate set shares >= 9 of the 10 core TFs",
            stats_clean.get("ig_extended_core10_overlap", 0) >= 9,
            f"got {stats_clean.get('ig_extended_core10_overlap')}",
        )
        _check(
            "Supplementary Note 1: global IG profiles Spearman >= 0.99 between candidate sets",
            stats_clean.get("ig_extended_global_spearman_all_tfs", 0) >= 0.99,
            f"got {stats_clean.get('ig_extended_global_spearman_all_tfs')}",
        )
        _check(
            "Supplementary Note 1: AURORA Bonferroni-significant core TFs equal in both runs (4)",
            stats_clean.get("ig_extended_aurora_extended_bonferroni") == 4
            and stats_clean.get("ig_extended_aurora_main_bonferroni") == 4,
            f"got main={stats_clean.get('ig_extended_aurora_main_bonferroni')} "
            f"extended={stats_clean.get('ig_extended_aurora_extended_bonferroni')}",
        )
    _check(
        "Candidate rule cutoff sensitivity matches Supplementary Table S1 (26:7, 35:10, 50:14)",
        (stats_clean.get("candidates_n_at_cutoff_26"),
         stats_clean.get("candidates_n_at_cutoff_35"),
         stats_clean.get("candidates_n_at_cutoff_50")) == (7, 10, 14),
        f"got {[stats_clean.get(f'candidates_n_at_cutoff_{n}') for n in (26, 35, 50)]}",
    )

# Within-subtype screens must match the manuscript (Results paragraph, SFig12 caption)
if "subtype_pooled_drugs_FDR005" in stats_clean:
    _check(
        "Within-subtype: Basal alone n/events/FDR<0.05 == 686/167/117",
        (stats_clean.get("subtype_basal_n_patients"), stats_clean.get("subtype_basal_n_events"),
         stats_clean.get("subtype_basal_drugs_FDR005")) == (686, 167, 117),
        f"got {[stats_clean.get(k) for k in ('subtype_basal_n_patients', 'subtype_basal_n_events', 'subtype_basal_drugs_FDR005')]}",
    )
    _check(
        "Within-subtype: Basal+HER2 pool n/events/FDR<0.05 == 1362/314/216",
        (stats_clean.get("subtype_pooled_n_patients"), stats_clean.get("subtype_pooled_n_events"),
         stats_clean.get("subtype_pooled_drugs_FDR005")) == (1362, 314, 216),
        f"got {[stats_clean.get(k) for k in ('subtype_pooled_n_patients', 'subtype_pooled_n_events', 'subtype_pooled_drugs_FDR005')]}",
    )

# Training history (Methods: 800 epochs, SFig04), sensitivity analyses (Results, Discussion)
if "training_epochs" in stats_clean:
    _check(
        "Training history: 800 epochs and final test Pearson in [0.74, 0.77] (Table 1 SAGE 0.756)",
        stats_clean.get("training_epochs") == 800
        and 0.74 <= stats_clean.get("training_final_test_pearson", 0) <= 0.77,
        f"got epochs={stats_clean.get('training_epochs')} pearson={stats_clean.get('training_final_test_pearson')}",
    )
if "rfs_n_fdr05" in stats_clean:
    _check(
        "RFS sensitivity: 11/11 direction-concordant and 5 drugs FDR<0.05 (Discussion)",
        stats_clean.get("rfs_n_direction_concordant") == 11 and stats_clean.get("rfs_n_fdr05") == 5,
        f"got concordant={stats_clean.get('rfs_n_direction_concordant')} fdr05={stats_clean.get('rfs_n_fdr05')}",
    )
if "combat_n_sig" in stats_clean:
    _check(
        "ComBat vs QT: 634 / 623 / 367 both / 98.6% concordant (Discussion)",
        (stats_clean.get("combat_n_sig"), stats_clean.get("qt_n_sig"), stats_clean.get("combat_qt_n_both"),
         stats_clean.get("combat_qt_direction_pct")) == (634, 623, 367, 98.6),
        f"got {[stats_clean.get(k) for k in ('combat_n_sig', 'qt_n_sig', 'combat_qt_n_both', 'combat_qt_direction_pct')]}",
    )
if "clinical_adj_scanb_n_candidates_fdr05" in stats_clean:
    _check(
        "Clinical-covariate Cox: 5 candidates FDR<0.05 in SCAN-B (HR 1.9-3.4), brigatinib only in METABRIC (Results)",
        stats_clean.get("clinical_adj_scanb_n_candidates_fdr05") == 5
        and 1.85 <= stats_clean.get("clinical_adj_scanb_hr_min", 0) <= 1.95
        and 3.35 <= stats_clean.get("clinical_adj_scanb_hr_max", 0) <= 3.45
        and stats_clean.get("clinical_adj_metabric_candidates_fdr05") == ["brigatinib"],
        f"got scanb n={stats_clean.get('clinical_adj_scanb_n_candidates_fdr05')} "
        f"HR=[{stats_clean.get('clinical_adj_scanb_hr_min')}, {stats_clean.get('clinical_adj_scanb_hr_max')}] "
        f"metabric={stats_clean.get('clinical_adj_metabric_candidates_fdr05')}",
    )
if "aurora_site_n_sites" in stats_clean:
    _check(
        "AURORA by site: 7 sites, CREB3L1 and KLF8 concordant in every site (Results)",
        stats_clean.get("aurora_site_n_sites") == 7
        and {"CREB3L1", "KLF8"} <= set(stats_clean.get("aurora_site_fully_concordant_tfs", [])),
        f"got sites={stats_clean.get('aurora_site_n_sites')} tfs={stats_clean.get('aurora_site_fully_concordant_tfs')}",
    )

# AURORA drug sensitivity counts in manuscript text
aurora_drug_path = OUT / "aurora_drug_primary_vs_meta.csv"
if aurora_drug_path.exists():
    df_ad = pd.read_csv(aurora_drug_path)
    n_sig = int((df_ad["pvalue"] < 0.05).sum())
    stats_clean["aurora_drugs_sig"] = n_sig
    stats_clean["aurora_drugs_more_effective_meta"] = int(
        ((df_ad["pvalue"] < 0.05) & (df_ad["delta_auc"] < 0)).sum()
    )
    stats_clean["aurora_drugs_less_effective_meta"] = int(
        ((df_ad["pvalue"] < 0.05) & (df_ad["delta_auc"] > 0)).sum()
    )
    _check(
        "AURORA significant drugs in [700, 900] (manuscript claims)",
        700 <= n_sig <= 900,
        f"got {n_sig}",
    )

# PCA / PC1 residualization (Discussion: shared prognostic axis)
pca_path = RESULTS / "v8_multicohort" / "pca_residualization_summary.csv"
if pca_path.exists():
    pca = pd.read_csv(pca_path).set_index("metric")["value"]
    stats_clean["pca_pc1_explained_var_pct"] = float(pca.get("pc1_explained_var_pct"))
    stats_clean["pca_pc1_survival_pvalue"] = float(pca.get("pc1_survival_pvalue"))
    stats_clean["pca_inter_drug_corr_mean"] = float(pca.get("inter_drug_corr_mean"))
    stats_clean["pca_n_nominal_sig_raw"] = int(pca.get("n_nominal_sig_raw"))
    stats_clean["pca_n_nominal_sig_residualized"] = int(pca.get("n_nominal_sig_residualized"))
    _check(
        "PCA: residualizing PC1 does NOT increase nominal count "
        "(manuscript must not claim drug-specific increase)",
        stats_clean["pca_n_nominal_sig_residualized"] <= stats_clean["pca_n_nominal_sig_raw"],
        f"raw={stats_clean['pca_n_nominal_sig_raw']} "
        f"resid={stats_clean['pca_n_nominal_sig_residualized']}",
    )
    _check(
        "PCA: inter-drug corr matches manuscript (~0.07)",
        0.05 <= stats_clean["pca_inter_drug_corr_mean"] <= 0.10,
        f"got {stats_clean['pca_inter_drug_corr_mean']}",
    )

# TF-level univariate Cox on SCAN-B OS (Results MYC/TP53/E2F1 examples, script 44)
tfcox_path = OUT / "tf_level_survival_scanb.csv"
if tfcox_path.exists():
    tfc = pd.read_csv(tfcox_path).set_index("tf")
    for g in ["MYC", "TP53", "E2F1"]:
        if g in tfc.index:
            stats_clean[f"tfcox_{g}_HR"] = round(float(tfc.loc[g, "HR"]), 2)
            stats_clean[f"tfcox_{g}_p"] = float(tfc.loc[g, "pvalue"])
    # Biological direction: MYC/E2F1 proliferative -> HR>1 (worse OS);
    # TP53 activity -> HR<1 (better OS). Two of these were reversed in the text.
    _check(
        "TF-Cox MYC HR>1 (higher MYC activity, worse OS)",
        stats_clean.get("tfcox_MYC_HR", 0) > 1,
        f"got {stats_clean.get('tfcox_MYC_HR')}",
    )
    _check(
        "TF-Cox TP53 HR<1 (higher TP53 activity, better OS)",
        0 < stats_clean.get("tfcox_TP53_HR", 9) < 1,
        f"got {stats_clean.get('tfcox_TP53_HR')}",
    )
    _check(
        "TF-Cox E2F1 HR>1 (higher E2F1 activity, worse OS)",
        stats_clean.get("tfcox_E2F1_HR", 0) > 1,
        f"got {stats_clean.get('tfcox_E2F1_HR')}",
    )

# Cox penalizer sweep (SFig11 + Discussion HR-range claims, script 45)
pensweep_path = OUT / "penalizer_sweep.csv"
if pensweep_path.exists():
    psw = pd.read_csv(pensweep_path)

    def _prow(pen):
        return psw.loc[(psw["penalizer"] - pen).abs() < 1e-6].iloc[0]

    r001, r05 = _prow(0.01), _prow(0.5)
    stats_clean["pensweep_n_fdr_pen05"] = int(r05["n_fdr_all"])
    stats_clean["pensweep_hr_min_pen05"] = round(float(r05["hr_min_all"]), 2)
    stats_clean["pensweep_hr_max_pen05"] = round(float(r05["hr_max_all"]), 2)
    stats_clean["pensweep_hr_max_pen001"] = round(float(r001["hr_max_all"]), 1)
    # Discussion line 343: pen=0.5 retains 306 drugs at FDR<0.05, HR in [0.13, 5.53];
    # near-unpenalized pen=0.01 inflates HR up to 245.
    _check(
        "Penalizer sweep pen=0.5 FDR<0.05 == 306 (Discussion text)",
        stats_clean["pensweep_n_fdr_pen05"] == 306,
        f"got {stats_clean['pensweep_n_fdr_pen05']}",
    )
    _check(
        "Penalizer sweep pen=0.01 max HR ~245 (Discussion, [240,250])",
        240 <= stats_clean["pensweep_hr_max_pen001"] <= 250,
        f"got {stats_clean['pensweep_hr_max_pen001']}",
    )

# Re-write paper_statistics.json with updated AURORA drug counts
with open(OUT / "paper_statistics.json", "w") as f:
    json.dump(stats_clean, f, indent=2)

if sanity_failures:
    print(f"\n[FAILURE] {len(sanity_failures)} sanity check(s) failed:")
    for f_msg in sanity_failures:
        print(f"  - {f_msg}")
    raise SystemExit(2)

print(f"\n[OK] All sanity checks passed.")
