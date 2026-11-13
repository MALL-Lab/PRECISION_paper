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
  supp_table_S1_candidate_selection.csv   (Supplementary Data 1)
  supp_table_S3_robustness.csv            (Supplementary Table S3, candidates and controls in SCAN-B)
  supp_table_S4_replication.csv           (Supplementary Table S1, METABRIC and chemotherapy)
  supp_table_hp_search.csv                (Supplementary Table S5, search of the Ridge and
                                           random forest hyperparameters, scripts/72, and
                                           the values used before it)
  shared_drugs.csv
  tf_importance_ig_top20.csv
  tf_importance_ig_top20_per_line.csv     (points of Figure 3a, one per TNBC line)
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
PAPER_DIR = Path(__file__).resolve().parent.parent
REPO = PAPER_DIR.parent
sys.path.insert(0, str(REPO))
from src.analysis.candidate_selection import SELECTION_RULE, apply_selection_rule  # noqa: E402
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
    RESULTS / "v2_fair" / "fair_comparison_per_seed.csv",
    # Search of the Ridge and random forest hyperparameters (scripts/72 on CESGA, job
    # 11133811) and its input bundle
    RESULTS / "v16_hp_search" / "cesga_11133811" / "best_params_ridge.json",
    RESULTS / "v16_hp_search" / "cesga_11133811" / "best_params_rf.json",
    RESULTS / "v16_hp_search" / "cesga_11133811" / "trials_ridge.csv",
    RESULTS / "v16_hp_search" / "cesga_11133811" / "trials_rf.csv",
    RESULTS / "v16_hp_search" / "cesga_11133811" / "script_sha256.txt",
    RESULTS / "v16_hp_search" / "inputs" / "manifest.json",
    RESULTS / "v16_hp_search" / "inputs" / "folds.csv",
    RESULTS / "v3_embeddings" / "embedding_comparison.csv",
    RESULTS / "v6_strengthen" / "cox_multivar_pam50_adjusted.csv",
    # Fits of script 31 per penalizer and the failed ones, so BH is known to run over all drugs
    RESULTS / "v6_strengthen" / "cox_multivar_pam50_fit_summary.csv",
    RESULTS / "v6_strengthen" / "cox_multivar_pam50_failures.csv",
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
    # Reporting Summary inputs: 95% CIs of Table 3 (script 43), cohort demographics
    # (script 54) and the ICLAC cross-check (script 55)
    RESULTS / "v8_multicohort" / "forest_positive_controls.csv",
    RESULTS / "v10_reporting" / "cohort_demographics.csv",
    RESULTS / "v10_reporting" / "iclac_summary.csv",
    RESULTS / "v10_reporting" / "iclac_matches.csv",
    RESULTS / "v11_collectri_snapshot" / "collectri_snapshot_sensitivity.csv",
    RESULTS / "v5_final" / "scanb_km_logrank.csv",
    # Re-audit of September 2026 (pipeline stage "reaudit", scripts 59 to 66, plus
    # the extensions of 42 and 58). Every headline number of the re-audit has a key
    # and a sanity check here, so a missing file has to stop the generator.
    RESULTS / "v13_reaudit" / "reference_baselines.csv",               # 59
    RESULTS / "v13_reaudit" / "transfer_ridge_heldout_perdrug.csv",
    RESULTS / "v13_reaudit" / "aurora_background_summary.csv",         # 60
    RESULTS / "v13_reaudit" / "aurora_tf_background.csv",
    RESULTS / "v13_reaudit" / "aurora_site_background.csv",
    RESULTS / "v13_reaudit" / "aurora_drug_paired.csv",
    RESULTS / "v13_reaudit" / "tnbc_subgroup_summary.csv",             # 61
    RESULTS / "v13_reaudit" / "tnbc_subgroup_cox.csv",
    RESULTS / "v13_reaudit" / "pam50_predicted_auc.csv",
    RESULTS / "v13_reaudit" / "pam50_composition_sensitive_half.csv",
    RESULTS / "v13_reaudit" / "treatment_summary.csv",                 # 62
    RESULTS / "v13_reaudit" / "treatment_adjusted_cox.csv",
    RESULTS / "v13_reaudit" / "treatment_interaction_cox.csv",
    RESULTS / "v13_reaudit" / "meta_dss_summary.csv",                  # 63
    RESULTS / "v13_reaudit" / "signed_meta_stouffer.csv",
    RESULTS / "v13_reaudit" / "direction_agreement_summary.csv",
    RESULTS / "v13_reaudit" / "positive_control_baserate.csv",
    RESULTS / "v13_reaudit" / "metabric_dss_survival.csv",
    RESULTS / "v13_reaudit" / "ph_summary.csv",                        # 64
    RESULTS / "v13_reaudit" / "ph_schoenfeld.csv",
    RESULTS / "v13_reaudit" / "cox_pam50_stratified.csv",
    RESULTS / "v13_reaudit" / "hr_per_sd_all_drugs.csv",
    RESULTS / "v13_reaudit" / "ph_summary_metabric.csv",
    RESULTS / "v13_reaudit" / "ph_schoenfeld_metabric.csv",
    RESULTS / "v13_reaudit" / "cox_pam50_stratified_metabric.csv",
    RESULTS / "v13_reaudit" / "pc1_orientation.csv",                   # 42
    RESULTS / "v13_reaudit" / "scanb_km_extras.csv",                   # 58
    RESULTS / "v13_reaudit" / "scanb_km_at_risk.csv",
    RESULTS / "v13_reaudit" / "ig_diagnostics_summary.csv",            # 65
    RESULTS / "v13_reaudit" / "ig_gradient_sign.csv",
    RESULTS / "v13_reaudit" / "ig_gradient_sign_by_tf.csv",
    RESULTS / "v13_reaudit" / "ig_gradient_check.csv",
    RESULTS / "v13_reaudit" / "tnbc_selectivity_primary.csv",          # 67
    RESULTS / "v13_reaudit" / "tnbc_selectivity_background.csv",
    RESULTS / "v13_reaudit" / "tnbc_set_test.csv",
    RESULTS / "v13_reaudit" / "tnbc_permutation_null.csv",
    RESULTS / "v13_reaudit" / "tnbc_sensitivity.csv",
    RESULTS / "v13_reaudit" / "tnbc_power_mde.csv",
    RESULTS / "v13_reaudit" / "tnbc_excluded_drugs.csv",
    RESULTS / "v13_reaudit" / "tnbc_moa_matched_null.csv",
    RESULTS / "v13_reaudit" / "tnbc_orthogonality.csv",
    RESULTS / "v13_reaudit" / "assay_sensitivity_gate.csv",
    RESULTS / "v13_reaudit" / "gdsc_replication.csv",
    RESULTS / "v13_reaudit" / "tnbc_cell_lines_summary.csv",
    RESULTS / "v14_fuscc" / "fuscc_primary_correlation.csv",             # 69
    RESULTS / "v14_fuscc" / "fuscc_permutation_null.csv",
    RESULTS / "v14_fuscc" / "fuscc_cox_perdrug.csv",
    RESULTS / "v14_fuscc" / "fuscc_composite.csv",
    RESULTS / "v14_fuscc" / "fuscc_sensitivity.csv",
    RESULTS / "v14_fuscc" / "fuscc_cohort_gate.csv",
    RESULTS / "v14_fuscc" / "fuscc_scanb_tnbc_reference_z.csv",
    RESULTS / "v14_fuscc" / "fuscc_excluded_summary.csv",   # counts only, the file with identifiers stays out
    RESULTS / "v14_fuscc" / "fuscc_rnaseq_vs_not.csv",
    RESULTS / "v14_fuscc" / "fuscc_power_mde.csv",
    RESULTS / "v14_fuscc" / "fuscc_ph_tests.csv",
    RESULTS / "v14_fuscc" / "fuscc_summary.csv",
    RESULTS / "v15_gene_universe" / "cohort_clipping.csv",                # 70, 71
    RESULTS / "v15_gene_universe" / "tcga_screen_comparison.csv",
    RESULTS / "v15_gene_universe" / "fisher_meta_comparison.csv",
    RESULTS / "v15_gene_universe" / "tcga_ph_schoenfeld.csv",
    RESULTS / "v15_gene_universe" / "gene_universe_summary.csv",
    RESULTS / "v15_gene_universe" / "gene_universe_r_summary.csv",
    RESULTS / "v13_reaudit" / "multiplicity_summary.csv",              # 66
    RESULTS / "v13_reaudit" / "ablation_perdrug_median.csv",
    RESULTS / "v13_reaudit" / "metabric_pam50_coverage.csv",
    # Tables the re-audit keys are derived from: unrounded fits of scripts 37 and 40
    # (twelve drugs since talazoparib joined), ComBat (41), TF-level Cox (44),
    # penalizer sweep (45), PC1 residualization (42) and the per-seed ablation (27)
    RESULTS / "v8_multicohort" / "cox_clinical_adjusted_scanb.csv",
    RESULTS / "v8_multicohort" / "cox_clinical_adjusted_metabric.csv",
    RESULTS / "v8_multicohort" / "cox_rfs_scanb.csv",
    RESULTS / "v8_multicohort" / "combat_vs_qt_comparison.csv",
    RESULTS / "v8_multicohort" / "combat_cox_scanb.csv",
    RESULTS / "v8_multicohort" / "tf_level_survival_scanb.csv",
    RESULTS / "v8_multicohort" / "penalizer_sweep.csv",
    RESULTS / "v8_multicohort" / "pca_residualization_perdrug.csv",
    RESULTS / "v6_strengthen" / "graph_ablation_per_seed.csv",
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
    # GNN rows and the hybrid row are means over five initialization seeds (script 57),
    # and the random forest row over five random_state values (script 11), with their
    # standard deviation. Ridge is fitted once.
    t1 = df_fair[["model", "global_pearson", "global_pearson_sd", "median_pearson", "median_pearson_sd",
                  "pct_gt_0.3", "n_drugs", "n_seeds"]].copy()
    t1.columns = ["Model", "Global_Pearson", "Global_Pearson_SD", "Median_PerDrug_Pearson",
                  "Median_PerDrug_Pearson_SD", "Pct_Drugs_GT_0.3", "N_drugs", "N_seeds"]

    # Append Ridge(expr + GNN emb) hybrid row (matches manuscript Table 1)
    if df_hybrid is not None:
        hybrid_row = df_hybrid[df_hybrid["model"] == "Ridge(expression+GNN_emb)"]
        if len(hybrid_row):
            r = hybrid_row.iloc[0]
            t1 = pd.concat([t1, pd.DataFrame([{
                "Model": "Ridge_expr_GNN_emb",
                "Global_Pearson": r["global_pearson"],
                "Global_Pearson_SD": r["global_pearson_sd"],
                "Median_PerDrug_Pearson": r["median_pearson"],
                "Median_PerDrug_Pearson_SD": r["median_pearson_sd"],
                "Pct_Drugs_GT_0.3": r["pct_gt_03"],
                "N_drugs": r["n_drugs"],
                "N_seeds": r["n_seeds"],
            }])], ignore_index=True)

    t1["Row_type"] = "model"
    t1["Label"] = t1["Model"]

    # Two reference rows added by the September 2026 re-audit (script 59). Neither is a
    # competing model: the drug mean carries no cell-line information and shows how much
    # of the pooled Pearson correlation comes from differences between drugs alone, and
    # the transfer Ridge is the clinical model of the second stage scored on the same
    # hold-out. Row_type tells them apart, so downstream code can keep the five models.
    df_ref = load_csv_required(RESULTS / "v13_reaudit" / "reference_baselines.csv",
                               "reference_baselines")
    ref_labels = {
        "Drug_mean_train": ("Drug_mean_reference", "Drug mean (no cell-line features)"),
        "Transfer_Ridge_TF": ("Transfer_Ridge_TF", "Ridge on TF activities (clinical transfer model)"),
    }
    for _, r in df_ref.iterrows():
        model, label = ref_labels[r["model"]]
        t1 = pd.concat([t1, pd.DataFrame([{
            "Model": model,
            "Global_Pearson": r["global_pearson"],
            "Global_Pearson_SD": np.nan,
            # The per-drug correlation of a constant predictor is undefined, so the
            # drug-mean row leaves these two empty instead of writing a zero.
            "Median_PerDrug_Pearson": r["median_perdrug"],
            "Median_PerDrug_Pearson_SD": np.nan,
            "Pct_Drugs_GT_0.3": r["pct_gt_03"],
            "N_drugs": r["n_drugs"],
            "N_seeds": 1,
            "Row_type": "reference",
            "Label": label,
        }])], ignore_index=True)

    for c in ("Global_Pearson", "Global_Pearson_SD", "Median_PerDrug_Pearson", "Median_PerDrug_Pearson_SD"):
        t1[c] = t1[c].round(3)
    t1["Pct_Drugs_GT_0.3"] = t1["Pct_Drugs_GT_0.3"].round(1)
    t1.to_csv(OUT / "table1_model_comparison.csv", index=False)
    print(t1.to_string(index=False))

    for _, row in t1.iterrows():
        stats[f"table1_{row['Model']}_global_pearson"] = row["Global_Pearson"]
        # None, not NaN: an undefined per-drug metric must be absent from the JSON
        # rather than an unquoted NaN literal.
        stats[f"table1_{row['Model']}_median_perdrug"] = (
            None if pd.isna(row["Median_PerDrug_Pearson"]) else row["Median_PerDrug_Pearson"])
        stats[f"table1_{row['Model']}_pct_gt_03"] = (
            None if pd.isna(row["Pct_Drugs_GT_0.3"]) else row["Pct_Drugs_GT_0.3"])
        stats[f"table1_{row['Model']}_n_drugs"] = int(row["N_drugs"])
        stats[f"table1_{row['Model']}_n_seeds"] = int(row["N_seeds"])
        stats[f"table1_{row['Model']}_row_type"] = row["Row_type"]
        if pd.notna(row["Global_Pearson_SD"]):
            stats[f"table1_{row['Model']}_global_pearson_sd"] = row["Global_Pearson_SD"]
            stats[f"table1_{row['Model']}_median_perdrug_sd"] = row["Median_PerDrug_Pearson_SD"]


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
    # Table 2 reports the population of the PAM50-adjusted Cox models, the one every
    # hazard ratio is fitted on. Until September 2026 it reported the survival cohort
    # before the PAM50 filter, which is larger (SCAN-B 7,397 against 7,215). That
    # survival cohort is still kept, under its own keys, for the Methods chain.
    info["n_patients"] = int(row["n_patients_cox"])
    info["events"] = int(row["n_events_cox"])
    stats[f"{cohort}_n_clinical_rows"] = int(row["n_clinical_rows"])
    stats[f"{cohort}_n_tf_samples"] = int(row["n_tf_samples"])
    stats[f"{cohort}_n_tfs"] = int(row["n_tfs_cohort"])
    stats[f"{cohort}_n_patients_after_dedup"] = int(row["n_patients_after_dedup"])
    stats[f"{cohort}_n_patients"] = int(row["n_patients_with_survival"])
    stats[f"{cohort}_n_events"] = int(row["n_events"])
    stats[f"{cohort}_n_patients_cox"] = int(row["n_patients_cox"])
    stats[f"{cohort}_n_events_cox"] = int(row["n_events_cox"])

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
        stats[f"{cohort}_drugs_FDR005_pct"] = pct

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
    stats["Fisher_drugs_FDR005_pct"] = round(100 * fisher_sig / fisher_total, 1)

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
            # The actual adjusted p-value, never 'n.s.' (journal policy: exact P values).
            row["METABRIC_padj"] = f"{match.iloc[0]['padj']:.2e}"
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
# 95% CIs come from script 43, which refits the same penalized Cox models. Its hazard
# ratios equal those of survival_*.csv to machine precision, checked here every run so a
# CI can never be attached to a hazard ratio from a different fit.
forest = load_csv_required(RESULTS / "v8_multicohort" / "forest_positive_controls.csv",
                           "forest_positive_controls")
forest["cohort_key"] = forest["cohort"].str.upper().str.replace("-", "", regex=False)
for coh in ("SCANB", "METABRIC"):
    fc = forest[forest["cohort_key"] == coh].set_index("drug")
    src = cohort_dfs[coh].set_index("drug")
    common = [d for d in positive_controls if d in fc.index and d in src.index]
    max_diff = float((fc.loc[common, "HR"] - src.loc[common, "HR"]).abs().max())
    if max_diff > 1e-6:
        raise AssertionError(f"forest_positive_controls HR differs from survival_{coh} by {max_diff}")
    t3[f"{coh}_CI95"] = t3["Drug"].map(
        lambda d: f"{fc.at[d, 'CI_low']:.2f}-{fc.at[d, 'CI_high']:.2f}" if d in fc.index else "n.a.")
    for d in common:
        stats[f"positive_control_{d}_{coh}_ci_low"] = round(float(fc.at[d, "CI_low"]), 2)
        stats[f"positive_control_{d}_{coh}_ci_high"] = round(float(fc.at[d, "CI_high"]), 2)
# Signed Stouffer combination of the September 2026 re-audit (script 63), which unlike
# Fisher requires a consistent direction across cohorts. Two significant digits, as
# printed, rounded once (the key is stouffer_<drug>_padj of the re-audit block).
_stouffer_padj = load_csv_required(RESULTS / "v13_reaudit" / "signed_meta_stouffer.csv",
                                   "signed_meta_stouffer").set_index("drug")["stouffer_padj"]
t3["Stouffer_padj"] = t3["Drug"].map(lambda d: f"{float(_stouffer_padj[d]):.1e}")
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
    stats["aurora_bonferroni_tfs"] = t4.loc[t4["Bonferroni_sig"].astype(bool), "TF"].tolist()
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
# The rule lives in src/analysis/candidate_selection.py so that Supplementary
# Note 2 (script 56) applies exactly the same filters.
SENSITIVITY_CUTOFFS = (26, 30, 35, 50)

if df_pam is not None and df_pam05 is not None:
    annot = load_csv_required(DATA / "precision_processed" / "prism_drug_annotations.csv",
                              "prism_drug_annotations")
    step1 = apply_selection_rule(df_pam, df_pam05, annot)
    extended_mask = step1[["pass_stability", "pass_class", "pass_phase"]].all(axis=1)
    extended = step1[extended_mask]
    selected = step1[step1["selected"]]

    # HDAC inhibitors in the funnel (Results: entinostat is the only one that
    # passes both the statistical and the stability filters) and the two HDAC
    # inhibitors named in the text as showing no association in SCAN-B.
    hdac = step1[step1["moa"].fillna("").str.contains("HDAC", case=False)]
    stats["candidates_hdac_statistical_filter"] = hdac["drug"].tolist()
    stats["candidates_hdac_stable"] = hdac.loc[hdac["pass_stability"], "drug"].tolist()
    pam_idx = df_pam.set_index("drug")
    for d in ("panobinostat", "sb-939"):
        stats[f"scanb_{d.replace('-', '')}_padj"] = round(float(pam_idx.loc[d, "padj"]), 2)

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

    # Supplementary Data 1: every drug passing filter (i), with all flags.
    supp_cols = ["rank", "drug", "HR", "CI_low", "CI_high", "padj", "HR_pen05", "padj_pen05", "moa", "target",
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
        "CI95_PAM50adj": t5["CI_low"].map(lambda v: f"{v:.2f}") + "-" + t5["CI_high"].map(lambda v: f"{v:.2f}"),
        # Two significant digits, as printed, so the manuscript never rounds twice.
        "padj_PAM50adj": t5["padj"].map(lambda v: f"{v:.1e}"),
        "HR_pen05": t5["HR_pen05"].round(2),
        "padj_pen05": t5["padj_pen05"].map(lambda v: f"{v:.1e}"),
    })
    for _, r in selected.iterrows():
        # HR_0.1 and p_adj,0.1 of Table 5, rounded once as printed (two decimals, two
        # significant digits), from the same rows as the CSV columns HR_PAM50adj and
        # padj_PAM50adj.
        stats[f"candidate_{r['drug']}_hr"] = round(float(r["HR"]), 2)
        stats[f"candidate_{r['drug']}_padj"] = float(f"{r['padj']:.1e}")
        stats[f"candidate_{r['drug']}_ci_low"] = round(float(r["CI_low"]), 2)
        stats[f"candidate_{r['drug']}_ci_high"] = round(float(r["CI_high"]), 2)
        stats[f"candidate_{r['drug']}_hr_pen05"] = round(float(r["HR_pen05"]), 2)
        stats[f"candidate_{r['drug']}_padj_pen05"] = float(f"{r['padj_pen05']:.1e}")
    # Two columns of the September 2026 re-audit, rounded once as printed: the hazard
    # ratio per standard deviation of the predicted AUC across the Cox cohort (script 64,
    # key hrsd_<drug>) and the Pearson r of the transfer model on the 95 held-out cell
    # lines (script 59, key candidate_<drug>_heldout_r).
    _hr_per_sd = load_csv_required(RESULTS / "v13_reaudit" / "hr_per_sd_all_drugs.csv",
                                   "hr_per_sd_all_drugs").set_index("drug")["HR_per_sd"]
    _r_test = load_csv_required(RESULTS / "v13_reaudit" / "transfer_ridge_heldout_perdrug.csv",
                                "transfer_ridge_heldout_perdrug").set_index("drug")["pearson_r"]
    t5.insert(t5.columns.get_loc("padj_PAM50adj") + 1, "HR_per_SD",
              t5["Drug"].map(lambda d: round(float(_hr_per_sd[d]), 2)))
    t5["r_test"] = t5["Drug"].map(lambda d: round(float(_r_test[d]), 2))
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

# Breast cell lines of the graph by receptor subtype (Supplementary Figure 3B).
from src.data.breast_subtypes import breast_subtypes  # noqa: E402
from src.data.drug_names import normalize_drug_name  # noqa: E402
_graph_lines = pd.read_csv(RESULTS / "v3_embeddings" / "cell_line_embeddings.csv", usecols=[0]).iloc[:, 0]
_breast = breast_subtypes(pd.read_csv(DATA / "DepMap" / "Model.csv", index_col="ModelID"), _graph_lines)
stats["breast_n_cell_lines"] = int(len(_breast))
for _label, _key in (("TNBC", "tnbc"), ("ER+", "erpos"), ("HER2+", "her2pos"), ("Other", "other")):
    stats[f"breast_n_{_key}"] = int((_breast == _label).sum())

# Kaplan-Meier panels of Figure 4A (script 58): survival cohort, median split.
km = load_csv_required(RESULTS / "v5_final" / "scanb_km_logrank.csv", "scanb_km_logrank")
stats["SCANB_km_n_patients"] = int(km["n_patients"].iloc[0])
stats["SCANB_km_n_events"] = int(km["n_events"].iloc[0])
for _, r in km.iterrows():
    stats[f"SCANB_km_{r['drug']}_logrank_p"] = float(f"{r['p']:.2e}")
    stats[f"SCANB_km_{r['drug']}_sensitive_worse"] = bool(r["sensitive_worse"])

# ===================================================================
# Supplementary Note 2: clinical results with the CollecTRI snapshot
# (script 56). Every count of the Note is read from here.
# ===================================================================
snap = load_csv_required(RESULTS / "v11_collectri_snapshot" / "collectri_snapshot_sensitivity.csv",
                         "collectri_snapshot_sensitivity")
for _, r in snap.iterrows():
    v = str(r["value"])
    if v.startswith("["):
        val = json.loads(v)
    else:
        num = pd.to_numeric(v, errors="coerce")
        val = v if pd.isna(num) else (int(num) if float(num).is_integer() else float(num))
    stats[f"collectri_snap_{r['metric']}"] = val
print(f"\nSupplementary Note 2 (script 56): {len(snap)} metrics, SCAN-B "
      f"{stats['collectri_snap_SCANB_drugs_FDR005']} significant drugs with the snapshot")


# ===================================================================
# Within-subtype survival screens (script 19b): Basal alone and the
# Basal + HER2 pool, one sample per patient (Results, Supplementary Figure 13).
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
# Per-epoch history of the canonical SAGE model (script 57), the model used
# for Integrated Gradients (Supplementary Figure 1)
hist_path = RESULTS / "v12_gnn" / "history" / "sage_split42_init42_full.csv"
if hist_path.exists():
    hist = pd.read_csv(hist_path)
    best_i = hist["test_pearson"].idxmax()
    stats["training_epochs"] = int(len(hist))
    stats["training_final_test_pearson"] = round(float(hist["test_pearson"].iloc[-1]), 3)
    stats["training_best_test_pearson"] = round(float(hist.loc[best_i, "test_pearson"]), 3)
    stats["training_best_epoch"] = int(hist.loc[best_i, "epoch"])
    print(f"\nTraining history (script 57): {stats['training_epochs']} epochs, final test Pearson "
          f"{stats['training_final_test_pearson']}, best {stats['training_best_test_pearson']} "
          f"at epoch {stats['training_best_epoch']}")

# Scripts 37 and 40 test the seven candidates and the five positive controls, with
# Benjamini-Hochberg over the twelve drugs since talazoparib joined them (September
# 2026, before that 11). Their comparison tables round HR to two decimals and padj
# to three significant digits, so every key is read from the unrounded per-cohort
# fits and rounded once here. The comparison tables are still read for the
# direction flag of script 40 and for the order of the drugs.
rfs = load_csv_required(RESULTS / "v8_multicohort" / "cox_rfs_comparison.csv", "cox_rfs_comparison")
rfs_fit = load_csv_required(RESULTS / "v8_multicohort" / "cox_rfs_scanb.csv", "cox_rfs_scanb").set_index("drug")
rfs_sig = [d for d in rfs["drug"] if rfs_fit.at[d, "padj"] < 0.05]
stats["rfs_n_drugs"] = int(len(rfs_fit))
stats["rfs_n_direction_concordant"] = int((rfs["same_direction"].astype(str).str.lower() == "yes").sum())
stats["rfs_n_fdr05"] = len(rfs_sig)
stats["rfs_fdr05_drugs"] = rfs_sig
# Osimertinib and saracatinib sit just below 0.05 with the 12-drug family (0.052 with
# 11), so they keep three significant digits: two would print 0.050 and 0.049.
for d in ("osimertinib", "saracatinib"):
    stats[f"rfs_{d}_padj"] = float(f"{rfs_fit.at[d, 'padj']:.2e}")
print(f"RFS (script 40): {stats['rfs_n_direction_concordant']}/{stats['rfs_n_drugs']} same direction, "
      f"{stats['rfs_n_fdr05']} at FDR<0.05: {stats['rfs_fdr05_drugs']}")

combat_path = RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv"
if combat_path.exists():
    cb = pd.read_csv(combat_path).set_index("metric")["value"]
    stats["combat_n_sig"] = int(cb["n_sig_combat"])
    stats["qt_n_sig"] = int(cb["n_sig_qt"])
    stats["combat_qt_n_both"] = int(cb["n_sig_both"])
    # From the counts, because the summary already rounds the percentage (98.6)
    stats["combat_qt_direction_pct"] = round(
        100 * float(cb["direction_concordant_among_both_sig"]) / float(cb["n_sig_both"]), 1)
    print(f"ComBat vs QT (script 41): {stats['combat_n_sig']} vs {stats['qt_n_sig']}, "
          f"{stats['combat_qt_n_both']} both, {stats['combat_qt_direction_pct']}% concordant")

cl = load_csv_required(RESULTS / "v8_multicohort" / "cox_clinical_comparison.csv", "cox_clinical_comparison")
cl_fit = {
    "scanb": load_csv_required(RESULTS / "v8_multicohort" / "cox_clinical_adjusted_scanb.csv",
                               "cox_clinical_adjusted_scanb").set_index("drug"),
    "metabric": load_csv_required(RESULTS / "v8_multicohort" / "cox_clinical_adjusted_metabric.csv",
                                  "cox_clinical_adjusted_metabric").set_index("drug"),
}
cand = [d for d in cl["drug"] if d not in positive_controls]      # order of Table 5 in script 37
sig_s = [d for d in cand if cl_fit["scanb"].at[d, "padj"] < 0.05]
sig_m = [d for d in cand if cl_fit["metabric"].at[d, "padj"] < 0.05]
stats["clinical_adj_n_drugs"] = int(len(cl_fit["scanb"]))
stats["clinical_adj_n_candidates"] = len(cand)
stats["clinical_adj_scanb_n_candidates_fdr05"] = len(sig_s)
stats["clinical_adj_scanb_candidates_fdr05"] = sig_s
stats["clinical_adj_scanb_hr_min"] = round(float(cl_fit["scanb"].loc[sig_s, "HR_clinical"].min()), 2)
stats["clinical_adj_scanb_hr_max"] = round(float(cl_fit["scanb"].loc[sig_s, "HR_clinical"].max()), 2)
stats["clinical_adj_metabric_n_candidates_fdr05"] = len(sig_m)
stats["clinical_adj_metabric_candidates_fdr05"] = sig_m
for coh in ("scanb", "metabric"):
    stats[f"clinical_adj_{coh}_controls_fdr05"] = [
        d for d in positive_controls if cl_fit[coh].at[d, "padj"] < 0.05]
# HR and padj of each candidate surviving adjustment in METABRIC, quoted in Results,
# with the two significant digits the text prints
for d in sig_m:
    stats[f"clinical_adj_metabric_{d}_hr"] = round(float(cl_fit["metabric"].at[d, "HR_clinical"]), 2)
    stats[f"clinical_adj_metabric_{d}_padj"] = float(f"{cl_fit['metabric'].at[d, 'padj']:.1e}")
for d in cand:
    if d not in sig_s:
        stats[f"clinical_adj_scanb_{d}_padj"] = float(f"{cl_fit['scanb'].at[d, 'padj']:.1e}")
# Olaparib, the one control without FDR < 0.05 after adjustment in SCAN-B, three
# significant digits (0.141) as script 66 reports it
stats["clinical_adj_scanb_olaparib_padj"] = float(f"{cl_fit['scanb'].at['olaparib', 'padj']:.2e}")
print(f"Clinical covariates (script 37, {stats['clinical_adj_n_drugs']} drugs): SCAN-B {len(sig_s)}/{len(cand)} "
      f"candidates FDR<0.05 (HR {stats['clinical_adj_scanb_hr_min']}-{stats['clinical_adj_scanb_hr_max']}), "
      f"controls {stats['clinical_adj_scanb_controls_fdr05']}, METABRIC {sig_m}")

# Reporting Summary descriptors. Demographics (script 54) are written as long rows
# (cohort, population, variable, value) and every row becomes a key, so any figure the
# form quotes is traceable. The ICLAC summary (script 55) is already key-shaped.
demo = load_csv_required(RESULTS / "v10_reporting" / "cohort_demographics.csv", "cohort_demographics")
for _, r in demo.iterrows():
    key = f"demo_{r['cohort']}_{r['population']}_{r['variable']}".replace(" ", "_")
    val = pd.to_numeric(r["value"], errors="coerce")
    stats[key] = (int(val) if float(val).is_integer() else float(val)) if pd.notna(val) else r["value"]
iclac = load_csv_required(RESULTS / "v10_reporting" / "iclac_summary.csv", "iclac_summary").iloc[0]
for k, v in iclac.items():
    stats[k] = v.item() if hasattr(v, "item") else v
print(f"Demographics (script 54): {len(demo)} descriptors | ICLAC (script 55): "
      f"{stats['iclac_n_graph_matched']}/{stats['iclac_n_graph_lines']} graph lines, "
      f"{stats['iclac_n_ig_matched']}/{stats['iclac_n_ig_lines']} IG lines")

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
        # Sites whose delta has the sign of the paired whole-cohort delta (Table 4).
        overall = pd.read_csv(RESULTS / "v9_aurora" / "aurora_tf_ig_validation.csv").set_index("tf")["delta"]
        for tf in delta.index.intersection(overall.index):
            stats[f"aurora_site_n_concordant_{tf}"] = int((np.sign(delta.loc[tf]) == np.sign(overall[tf])).sum())
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
    stats["ig_top20_names"] = df_ig.nsmallest(20, "rank")["tf"].tolist()
    print(df_ig.head(10)[["tf", "importance_mean", "rank"]].to_string(index=False))

    # Signed per-drug attributions of Figure 3C: top 20 TFs x 11 drugs
    _sg = pd.read_csv(RESULTS / "v4_xai" / "tf_drug_ig_signed.csv", comment="#").set_index("tf")
    _mean_sign = np.sign(_sg.mean(axis=1)).values[:, None]
    stats["ig_signed_n_cells"] = int(_sg.size)
    stats["ig_signed_n_cells_same_sign"] = int((np.sign(_sg.values) == _mean_sign).sum())
    _m = _sg.mean(axis=1).sort_values()
    stats["ig_signed_top5_sensitivity"] = _m.head(5).index.tolist()
    stats["ig_signed_top4_resistance"] = _m.tail(4).index[::-1].tolist()

# Multi-seed IG stability (internal review item)
ig_stab_path = RESULTS / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv"
if ig_stab_path.exists():
    df_stab = pd.read_csv(ig_stab_path)
    robust_top20 = df_stab[df_stab["in_top20_all_seeds"]]["tf"].tolist()
    stats["ig_multi_seed_top20_all_n"] = len(robust_top20)
    stats["ig_multi_seed_top20_all_names"] = robust_top20
    # The core set carried to AURORA-US and Table 4 (script 36): the ten TFs
    # with the lowest mean rank across the seeds.
    stats["ig_multi_seed_core10_names"] = df_stab.sort_values("mean_rank").head(10)["tf"].tolist()
    stats["ig_multi_seed_n_tfs"] = int(len(df_stab))
    # One trained model per seed (script 35), so the seed count comes from here.
    seeds = [c.split("_", 1)[1] for c in df_stab.columns if c.startswith("rank_")]
    stats["ig_n_seeds"] = len(seeds)
    overlaps, rhos = [], []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            top_a = set(df_stab.nsmallest(20, f"rank_{a}")["tf"])
            top_b = set(df_stab.nsmallest(20, f"rank_{b}")["tf"])
            overlaps.append(len(top_a & top_b))
            rhos.append(spearmanr(df_stab[f"rank_{a}"], df_stab[f"rank_{b}"])[0])
    stats["ig_multi_seed_top20_overlap_min"] = int(min(overlaps))
    stats["ig_multi_seed_top20_overlap_max"] = int(max(overlaps))
    stats["ig_multi_seed_spearman_min"] = round(float(min(rhos)), 3)
    stats["ig_multi_seed_spearman_max"] = round(float(max(rhos)), 3)
    print(f"  Multi-seed stable TFs (top-20 across {len(seeds)} seeds): {robust_top20}")
    print(f"  Pairwise top-20 overlap {min(overlaps)}-{max(overlaps)}, Spearman "
          f"{min(rhos):.3f}-{max(rhos):.3f}, core 10: {stats['ig_multi_seed_core10_names']}")

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
stats["ppi_edges_distinct_directed"] = kg["gene_interacts_gene_distinct_directed"]
stats["ppi_gene_pairs_distinct"] = kg["gene_interacts_gene_distinct_pairs"]
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

# Cross-screen validation PRISM -> GDSC (script 05, Supplementary Figure 14, Discussion)
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

# MOA and target enrichment among Fisher-significant drugs (script 52, Supplementary Figures 8 and 9)
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

# Within-subtype pooled screen (Supplementary Figure 13 legend): nominal and Bonferroni counts
pooled_path = RESULTS / "v7_weaknesses" / "basal_her2_survival.csv"
if pooled_path.exists():
    pooled = pd.read_csv(pooled_path)
    stats["subtype_pooled_n_nominal_p05"] = int((pooled["pvalue"] < 0.05).sum())
    stats["subtype_pooled_n_bonferroni"] = int((pooled["pvalue"] < 0.05 / len(pooled)).sum())

# SCAN-B vs METABRIC p-value concordance (Supplementary Figure 6 annotation)
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
    # TFs leading the zero-baseline ranking (Results)
    _z = pd.read_csv(RESULTS / "v4_xai" / "ig_baseline_zero.csv")
    stats["ig_baseline_zero_top3_names"] = _z.nsmallest(3, "rank")["tf"].tolist()

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
    # Means over five initialization seeds (script 27 reads the runs of script 57)
    for _, row in df_abl.iterrows():
        stats[f"ablation_{row['config']}_pearson"] = round(row["test_pearson"], 3)
        stats[f"ablation_{row['config']}_pearson_sd"] = round(row["test_pearson_sd"], 3)
        stats[f"ablation_{row['config']}_n_seeds"] = int(row["n_seeds"])
    print("\n  Graph ablation:")
    print(df_abl[["config", "test_pearson", "test_pearson_sd", "n_seeds"]].to_string(index=False))

# TCGA nominal paclitaxel
if "TCGA" in cohort_dfs:
    tcga = cohort_dfs["TCGA"]
    pac = tcga[tcga["drug"] == "paclitaxel"]
    if len(pac):
        stats["tcga_paclitaxel_pvalue"] = round(pac.iloc[0]["pvalue"], 4)
        stats["tcga_paclitaxel_HR"] = round(pac.iloc[0]["HR"], 2)
        print(f"\n  TCGA paclitaxel: p={pac.iloc[0]['pvalue']:.4f}, HR={pac.iloc[0]['HR']:.2f}")


# ===================================================================
# RE-AUDIT (September 2026): scripts 59 to 66 in results/v13_reaudit/, plus the
# extensions of scripts 42 (PC1 orientation) and 58 (Kaplan-Meier extras), and
# the keys the re-audit derives from tables that existed before it (ComBat,
# candidate ranks, METABRIC replication, TF-level Cox, PC1 residualization,
# hyperparameters and the Fisher recomputation of script 21).
#
# Every number of the re-audit that the manuscript or Supplementary Tables S1
# and S3 cite gets a key here, rounded once to the precision the text prints:
# hazard ratios, confidence limits and correlations to two decimals, p-values to
# two significant digits (_p2) and percentages to one decimal, or to whole
# numbers where the text prints them so. The values come from the per-drug and
# per-TF tables, which keep full precision, and percentages are recomputed from
# the counts, because the metric/value summaries of scripts 60 to 63 round some
# of their values. Script 42 only writes rounded values (four decimals for the
# correlations), so those keys go through _round_again, which raises when
# rounding the rounded value could differ from rounding the original one.
# Every key registered with _add must appear in the frozen-value checks of the
# sanity block, and the cross-checks collected in REAUDIT_XCHECKS compare the
# keys with the summaries of the scripts and with the other tables.
# ===================================================================
print("\n" + "=" * 60)
print("RE-AUDIT (scripts 59-66, results/v13_reaudit/)")
print("=" * 60)

from scipy.stats import binom, false_discovery_control, hypergeom, norm  # noqa: E402

REAUDIT = RESULTS / "v13_reaudit"
FUSCC = RESULTS / "v14_fuscc"
GENE_UNIVERSE = RESULTS / "v15_gene_universe"
CANDIDATES = ["erlotinib", "brigatinib", "osimertinib", "saracatinib",
              "pelitinib", "trametinib", "entinostat"]
TWELVE_DRUGS = CANDIDATES + positive_controls
REAUDIT_KEYS = []      # every key registered by _add, all of them frozen in the checks
REAUDIT_XCHECKS = []   # (name, condition, detail), reported in the sanity block


def _p2(value):
    """P-value with the two significant digits the manuscript prints."""
    return float(f"{float(value):.1e}")


def _p3(value):
    """P-value with three significant digits, for values next to a threshold."""
    return float(f"{float(value):.2e}")


def _r2(value):
    """Hazard ratio, confidence limit or correlation as printed, two decimals."""
    return round(float(value), 2)


def _pct(num, den):
    """Percentage from counts, one decimal."""
    return round(100 * float(num) / float(den), 1)


def _round_again(value, ndigits, src_ndigits):
    """Round a value that its script already rounded to src_ndigits decimals.

    Any unrounded value within half a unit of the last source digit must round to
    the same result, otherwise rounding twice could differ from rounding once and
    the function raises. Returns an int when ndigits is 0."""
    v = float(value)
    half = 0.5 * 10 ** -src_ndigits * (1 - 1e-9)
    if round(v - half, ndigits) != round(v + half, ndigits):
        raise AssertionError(f"{v}, rounded to {src_ndigits} decimals by its script, cannot be "
                             f"rounded to {ndigits} decimals without the unrounded value")
    out = round(v, ndigits)
    return int(out) if ndigits == 0 else out


def _cell(value):
    """Value of a metric/value CSV as int, float, list, dict or string."""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") or s.startswith("{"):
            return json.loads(s)
        num = pd.to_numeric(s, errors="coerce")
        if pd.isna(num):
            return s
        value = num
    if pd.isna(value):
        return None
    f = float(value)
    return int(f) if f.is_integer() else f


def _summary(path, label):
    """metric -> value of a metric/value CSV, parsed with _cell."""
    df = load_csv_required(path, label)
    return {str(m): _cell(v) for m, v in zip(df["metric"], df["value"])}


def _one(series, what):
    """The single value a column repeats on every row (cohort sizes per drug)."""
    values = pd.unique(series.dropna())
    if len(values) != 1:
        raise AssertionError(f"{what} is not constant: {values}")
    return values[0]


def _add(key, value):
    """Register a re-audit key, refusing to overwrite one that already exists."""
    if key in stats:
        raise AssertionError(f"re-audit key collides with an existing key: {key}")
    stats[key] = value
    REAUDIT_KEYS.append(key)


def _xcheck(name, condition, detail=""):
    REAUDIT_XCHECKS.append((name, bool(condition), detail))


def _role(drug):
    return "candidate" if drug in CANDIDATES else "control"


# --- script 59: reference predictors of Table 1 and held-out transfer Ridge ---
# The two reference rows of Table 1 already have their table1_* keys (Table 1 loop).
_ref = df_ref.set_index("model")
_add("ref_drug_mean_between_drug_variance_share",
     _r2(_ref.at["Drug_mean_train", "between_drug_variance_share"]))
_add("ref_n_test_pairs", int(_one(df_ref["n_test_pairs"], "n_test_pairs of reference_baselines.csv")))
tr_held = load_csv_required(REAUDIT / "transfer_ridge_heldout_perdrug.csv",
                            "transfer_ridge_heldout_perdrug").set_index("drug")
_add("transfer_heldout_n_drugs", int(tr_held["pearson_r"].notna().sum()))
for d in TWELVE_DRUGS:
    _add(f"{_role(d)}_{d}_heldout_r", _r2(tr_held.at[d, "pearson_r"]))
_xcheck("Transfer Ridge row of Table 1 = per-drug file of script 59 (median r and share above 0.3)",
        np.isclose(tr_held["pearson_r"].median(), _ref.at["Transfer_Ridge_TF", "median_perdrug"], rtol=0, atol=1e-12)
        and np.isclose(100 * (tr_held["pearson_r"] > 0.3).mean(), _ref.at["Transfer_Ridge_TF", "pct_gt_03"],
                       rtol=0, atol=1e-9)
        and len(tr_held) == int(_ref.at["Transfer_Ridge_TF", "n_drugs"]),
        f"median {tr_held['pearson_r'].median()} vs {_ref.at['Transfer_Ridge_TF', 'median_perdrug']}")
print(f"Reference predictors (script 59): drug mean r={stats['table1_Drug_mean_reference_global_pearson']} "
      f"(between-drug variance share {stats['ref_drug_mean_between_drug_variance_share']}), transfer Ridge "
      f"r={stats['table1_Transfer_Ridge_TF_global_pearson']}, held-out r of the candidates "
      f"{[stats[f'candidate_{d}_heldout_r'] for d in CANDIDATES]}")

# --- script 42 extended: orientation and meaning of PC1 ---
pc1 = _summary(REAUDIT / "pc1_orientation.csv", "pc1_orientation")
_add("pc1_orient_n_patients", int(pc1["n_patients"]))
_add("pc1_orient_n_events", int(pc1["n_events"]))
_add("pc1_orient_sign_flipped", bool(pc1["pc1_sign_flipped"]))
_add("pc1_orient_pc2_explained_var_pct", _round_again(pc1["pc2_explained_var_pct"], 0, 2))
for tf in ("E2F1", "MYC", "TP53", "ESR1"):
    _add(f"pc1_orient_corr_{tf}", _round_again(pc1[f"pc1_oriented_corr_tf_{tf}"], 2, 4))
for d in ("paclitaxel", "talazoparib", "olaparib"):
    _add(f"pc1_orient_corr_{d}_auc", _round_again(pc1[f"pc1_oriented_corr_{d}_auc"], 2, 4))
_add("pc1_orient_survival_z", _round_again(pc1["pc1_oriented_survival_z"], 2, 3))
_add("pc1_orient_inter_drug_abs_corr_mean", _round_again(pc1["inter_drug_abs_corr_mean"], 2, 4))
_add("pc1_orient_pct_drug_pairs_negative", _round_again(pc1["pct_drug_pairs_negative_corr"], 0, 2))
_add("pc1_orient_n_drug_pairs", int(pc1["n_drug_pairs"]))
_pca_sum = load_csv_required(RESULTS / "v8_multicohort" / "pca_residualization_summary.csv",
                             "pca_residualization_summary").set_index("metric")["value"]
_xcheck("PC1 orientation (script 42) matches the PCA summary: 12.0% variance, mean inter-drug r, "
        "|z| consistent with the PC1 Cox p",
        float(pc1["pc1_explained_var_pct"]) == float(_pca_sum["pc1_explained_var_pct"])
        and abs(float(pc1["inter_drug_corr_mean"]) - float(_pca_sum["inter_drug_corr_mean"])) < 5e-5
        and np.isclose(2 * norm.sf(abs(float(pc1["pc1_oriented_survival_z"]))),
                       float(_pca_sum["pc1_survival_pvalue"]), rtol=1e-2),
        f"PC1 {pc1['pc1_explained_var_pct']} vs {_pca_sum['pc1_explained_var_pct']}, z {pc1['pc1_oriented_survival_z']} "
        f"vs p {_pca_sum['pc1_survival_pvalue']}")
print(f"PC1 orientation (script 42): E2F1 {stats['pc1_orient_corr_E2F1']}, MYC {stats['pc1_orient_corr_MYC']}, "
      f"TP53 {stats['pc1_orient_corr_TP53']}, ESR1 {stats['pc1_orient_corr_ESR1']}, talazoparib AUC "
      f"{stats['pc1_orient_corr_talazoparib_auc']}, PC2 {stats['pc1_orient_pc2_explained_var_pct']}%")

# --- script 58 extended: Kaplan-Meier extras of Figure 4 ---
km_extra = load_csv_required(REAUDIT / "scanb_km_extras.csv", "scanb_km_extras").set_index("drug")
for col in ("n_patients", "n_events", "pam50_n_patients", "pam50_n_events"):
    _add(f"km_extra_{col}", int(_one(km_extra[col], f"{col} of scanb_km_extras.csv")))
for d, r in km_extra.iterrows():
    _add(f"km_{d}_os10_sensitive_pct", round(float(r["os10_sensitive_pct"]), 1))
    _add(f"km_{d}_os10_resistant_pct", round(float(r["os10_resistant_pct"]), 1))
    _add(f"km_{d}_os10_diff_pp", round(float(r["os10_diff_pp"]), 1))
    _add(f"km_{d}_pam50_strat_p", _p2(r["pam50_strat_p"]))
    _add(f"km_{d}_trend_p", _p2(r["trend_p"]))
    # A positive score U means the hazard rises with the predicted AUC (script 58)
    _add(f"km_{d}_trend_hazard_rises_with_auc", bool(float(r["trend_score_u"]) > 0))
    for sub in ("luma", "basal"):
        for half in ("sensitive", "resistant"):
            _add(f"km_{d}_pct_{sub}_{half}", round(float(r[f"pct_{sub}_{half}"]), 1))
_xcheck("Kaplan-Meier extras (script 58): the sign of the 10-year difference agrees with the "
        "sensitive_worse flag of the log-rank table, and the median split score test equals the log-rank chi2",
        all((float(km_extra.at[d, "os10_diff_pp"]) < 0) == bool(km.set_index("drug").at[d, "sensitive_worse"])
            for d in km_extra.index)
        # 1e-3: the Breslow score variance differs slightly from the hypergeometric
        # log-rank variance with tied event times, the tolerance script 58 uses itself
        and np.allclose(km_extra["median_split_score_chi2"], km.set_index("drug").loc[km_extra.index, "chi2"],
                        rtol=1e-3, atol=0),
        "sign or chi2 mismatch between scanb_km_extras.csv and scanb_km_logrank.csv")
# Numbers at risk printed under the panels of Figure 4
km_risk = load_csv_required(REAUDIT / "scanb_km_at_risk.csv", "scanb_km_at_risk")
_risk_half = km_risk[km_risk["split"] == "median"].set_index(["drug", "group", "time_years"])["n_at_risk"]
_risk_total = km_risk.groupby(["drug", "split", "time_years"])["n_at_risk"].sum().unstack("time_years")
_add("km_at_risk_10y_total", int(_one(_risk_total[10], "total at risk at 10 years of scanb_km_at_risk.csv")))
_add("km_at_risk_step_years",
     int(_one(pd.Series(np.diff(np.sort(km_risk["time_years"].unique()))), "step of the times of scanb_km_at_risk.csv")))
_km_lr = km.set_index("drug")
_xcheck("Numbers at risk of Figure 4 (script 58): at 0 years the halves are those of the log-rank table and add "
        "up to the cohort, at 10 years they are those of the extras, and the total per time is the same for all "
        "drugs and both splits",
        set(km_risk["drug"]) == set(km_extra.index)
        and all((_risk_half[d, "sensitive", 0], _risk_half[d, "resistant", 0])
                == (_km_lr.at[d, "n_sensitive"], _km_lr.at[d, "n_resistant"])
                and (_risk_half[d, "sensitive", 10], _risk_half[d, "resistant", 10])
                == (km_extra.at[d, "n_at_risk_10y_sensitive"], km_extra.at[d, "n_at_risk_10y_resistant"])
                for d in km_extra.index)
        and (_risk_total[0] == stats["SCANB_km_n_patients"]).all()
        and _risk_total.nunique().max() == 1,
        f"scanb_km_at_risk.csv disagrees with scanb_km_logrank.csv, scanb_km_extras.csv or "
        f"SCANB_km_n_patients: totals per time\n{_risk_total}")
print(f"Kaplan-Meier extras (script 58): 10-year OS difference "
      f"{[stats[f'km_{d}_os10_diff_pp'] for d in km_extra.index]} points, PAM50-stratified p "
      f"{[stats[f'km_{d}_pam50_strat_p'] for d in km_extra.index]}, at risk at 10 years "
      f"{stats['km_at_risk_10y_total']} of {stats['SCANB_km_n_patients']}")

# --- script 63: directional agreement, signed Stouffer, control base rate, DSS ---
meta_sum = _summary(REAUDIT / "meta_dss_summary.csv", "meta_dss_summary")
dir_df = load_csv_required(REAUDIT / "direction_agreement_summary.csv",
                           "direction_agreement_summary").set_index("subset")
# The agreement expected if the cohorts chose their directions independently is
# recomputed from the per-cohort hazard ratios, since the summary rounds it: with
# shares a and b of HR > 1 in SCAN-B and METABRIC it is a * b + (1 - a) * (1 - b).
_hr_s = cohort_dfs["SCANB"].set_index("drug")
_hr_m = cohort_dfs["METABRIC"].set_index("drug").loc[_hr_s.index]
_sig_s = _hr_s["padj"] < 0.05
_sig_m = _hr_m["padj"] < 0.05
for label, tag, mask in (("all drugs tested in both cohorts", "all", pd.Series(True, index=_hr_s.index)),
                         ("SCAN-B FDR < 0.05", "scanb_sig", _sig_s),
                         ("FDR < 0.05 in both cohorts", "both_sig", _sig_s & _sig_m)):
    r = dir_df.loc[label]
    a = float((_hr_s.loc[mask, "HR"] > 1).mean())
    b = float((_hr_m.loc[mask, "HR"] > 1).mean())
    expected = 100 * (a * b + (1 - a) * (1 - b))
    same = int(((_hr_s.loc[mask, "HR"] > 1) == (_hr_m.loc[mask, "HR"] > 1)).sum())
    _add(f"direction_{tag}_n", int(mask.sum()))
    _add(f"direction_{tag}_n_same", same)
    _add(f"direction_{tag}_pct_same", _pct(same, mask.sum()))
    _add(f"direction_{tag}_pct_expected", round(expected, 1))
    _xcheck(f"Directional agreement '{label}' recomputed from survival_*.csv = script 63",
            (int(mask.sum()), same) == (int(r["n_drugs"]), int(r["n_same_direction"]))
            and round(expected, 1) == float(r["pct_same_expected_independent"]),
            f"{int(mask.sum())}/{same}/{expected:.2f} vs {r['n_drugs']}/{r['n_same_direction']}/"
            f"{r['pct_same_expected_independent']}")

stouffer = load_csv_required(REAUDIT / "signed_meta_stouffer.csv", "signed_meta_stouffer").set_index("drug")
_st_sig = stouffer["stouffer_padj"] < 0.05
_fi_sig = stouffer["fisher_padj"] < 0.05
_opposite = ~stouffer["scanb_metabric_same_direction"].astype(bool)
_add("stouffer_n_drugs", int(len(stouffer)))
_add("stouffer_n_sig", int(_st_sig.sum()))
_add("stouffer_pct_sig", _pct(_st_sig.sum(), len(stouffer)))
# z = sign(log HR) * |z|, so a positive combined z is a combined HR above 1
_add("stouffer_n_sig_hr_gt1", int((_st_sig & (stouffer["stouffer_z"] > 0)).sum()))
_add("stouffer_n_sig_hr_lt1", int((_st_sig & (stouffer["stouffer_z"] < 0)).sum()))
_add("stouffer_fisher_overlap", int((_st_sig & _fi_sig).sum()))
_add("stouffer_fisher_sig_lost", int((_fi_sig & ~_st_sig).sum()))
_add("stouffer_sig_not_fisher", int((_st_sig & ~_fi_sig).sum()))
_add("stouffer_fisher_sig_opposite_direction", int((_fi_sig & _opposite).sum()))
_add("stouffer_fisher_sig_opposite_lost", int((_fi_sig & _opposite & ~_st_sig).sum()))
for d in TWELVE_DRUGS:
    _add(f"stouffer_{d}_padj", _p2(stouffer.at[d, "stouffer_padj"]))
_add("stouffer_controls_fdr05", [d for d in positive_controls if stouffer.at[d, "stouffer_padj"] < 0.05])
_add("stouffer_candidates_n_fdr05", int(sum(stouffer.at[d, "stouffer_padj"] < 0.05 for d in CANDIDATES)))
# Stouffer-significant drugs whose SCAN-B and METABRIC hazard ratios point in opposite directions
_add("stouffer_n_sig_opposite_direction", int((_st_sig & _opposite).sum()))
_opp_hr = (np.sign(np.log(_hr_s["HR"])) != np.sign(np.log(_hr_m["HR"]))).loc[stouffer.index]
_xcheck("Stouffer-significant drugs with opposite SCAN-B and METABRIC directions recomputed from survival_*.csv "
        "= the flag of script 63 = its summary (Stouffer-significant minus same direction)",
        (_opp_hr == _opposite).all()
        and int((_st_sig & _opp_hr).sum()) == stats["stouffer_n_sig_opposite_direction"]
        == meta_sum["direction_stouffer_sig_n"] - meta_sum["direction_stouffer_sig_n_same"]
        and meta_sum["direction_stouffer_sig_n"] == stats["stouffer_n_sig"],
        f"{stats['stouffer_n_sig_opposite_direction']} opposite")
_xcheck("Signed Stouffer counts from the per-drug table = meta_dss_summary.csv (script 63)",
        [stats["stouffer_n_sig"], stats["stouffer_n_sig_hr_gt1"], stats["stouffer_n_sig_hr_lt1"],
         stats["stouffer_fisher_overlap"], stats["stouffer_fisher_sig_lost"], stats["stouffer_sig_not_fisher"],
         stats["stouffer_fisher_sig_opposite_direction"], stats["stouffer_fisher_sig_opposite_lost"]]
        == [meta_sum[k] for k in ("stouffer_n_sig", "stouffer_n_sig_hr_gt1", "stouffer_n_sig_hr_lt1",
                                  "stouffer_fisher_overlap", "fisher_sig_lost_under_stouffer",
                                  "stouffer_sig_not_fisher", "fisher_sig_opposite_direction",
                                  "fisher_sig_opposite_direction_lost_under_stouffer")]
        and all(np.isclose(meta_sum[f"stouffer_{d}_padj"], stouffer.at[d, "stouffer_padj"], rtol=1e-12)
                for d in TWELVE_DRUGS),
        "per-drug table and summary of script 63 disagree")
_xcheck("Fisher columns of signed_meta_stouffer.csv = meta_analysis_fisher.csv (script 21, chi2.sf)",
        np.allclose(stouffer.loc[df_fisher["drug"], "fisher_padj"].values, df_fisher["fisher_padj"].values,
                    rtol=1e-9, atol=0),
        "the Stouffer table was built from an older Fisher meta-analysis")

baserate = load_csv_required(REAUDIT / "positive_control_baserate.csv", "positive_control_baserate")
for analysis, tag in (("Fisher meta-analysis", "fisher"), ("Signed Stouffer meta-analysis", "stouffer")):
    b = baserate[baserate["analysis"] == analysis].set_index("drug")
    _add(f"controls_baserate_{tag}_frac_sig", round(float(_one(b["frac_drugs_significant"], analysis)), 3))
    _add(f"controls_baserate_{tag}_n_sig", int(_one(b["n_controls_significant"], analysis)))
    _add(f"controls_baserate_{tag}_p_at_least_observed",
         _p2(_one(b["prob_at_least_n_controls_significant"], analysis)))
    for d in positive_controls:
        _add(f"controls_baserate_{tag}_{d}_rank", int(b.at[d, "rank_by_p"]))
    _add(f"controls_baserate_{tag}_rank_min", int(b["rank_by_p"].min()))
    _add(f"controls_baserate_{tag}_rank_max", int(b["rank_by_p"].max()))
_fisher_rank = df_fisher.set_index("drug")["fisher_p"].rank(method="min")
_frac = stats["Fisher_drugs_FDR005"] / len(df_fisher)
_xcheck("Control base rate (script 63): Fisher share, binomial P(at least 4 of 5) and ranks recomputed "
        "from meta_analysis_fisher.csv",
        round(_frac, 3) == stats["controls_baserate_fisher_frac_sig"]
        and _p2(binom.sf(stats["controls_baserate_fisher_n_sig"] - 1, 5, _frac))
        == stats["controls_baserate_fisher_p_at_least_observed"]
        and stats["controls_baserate_fisher_n_sig"] == sum(
            str(stats[f"positive_control_{d}_fisher_padj"]) != "n.a."
            and float(stats[f"positive_control_{d}_fisher_padj"]) < 0.05 for d in positive_controls)
        and all(int(_fisher_rank[d]) == stats[f"controls_baserate_fisher_{d}_rank"] for d in positive_controls),
        f"share {_frac}, n {stats['controls_baserate_fisher_n_sig']}")

dss = load_csv_required(REAUDIT / "metabric_dss_survival.csv", "metabric_dss_survival").set_index("drug")
for m in ("n_patients", "n_deaths", "n_dss_deaths", "n_other_cause_deaths", "n_deaths_unknown_cause_censored"):
    _add(f"dss_metabric_{m}", int(meta_sum[f"metabric_{m}"]))
_add("dss_metabric_pct_deaths_other_causes",
     _pct(meta_sum["metabric_n_other_cause_deaths"], meta_sum["metabric_n_deaths"]))
_dss_sig = dss["padj_DSS"] < 0.05
_os_sig = dss["padj_OS"] < 0.05
_add("dss_metabric_n_sig", int(_dss_sig.sum()))
_add("dss_metabric_os_n_sig", int(_os_sig.sum()))
_add("dss_metabric_os_dss_overlap", int((_dss_sig & _os_sig).sum()))
_add("dss_metabric_n_sig_hr_gt1", int((_dss_sig & (dss["HR_DSS"] > 1)).sum()))
for d in TWELVE_DRUGS:
    _add(f"dss_{d}_hr", _r2(dss.at[d, "HR_DSS"]))
    _add(f"dss_{d}_padj", _p2(dss.at[d, "padj_DSS"]))
_add("dss_candidates_fdr05", [d for d in CANDIDATES if dss.at[d, "padj_DSS"] < 0.05])
_add("dss_candidates_n_fdr05", len(stats["dss_candidates_fdr05"]))
_add("dss_candidates_n_hr_gt1", int((dss.loc[CANDIDATES, "HR_DSS"] > 1).sum()))
_xcheck("METABRIC DSS of the candidates (script 63): padj of the per-drug table = its summary, and the rounded "
        "keys give the same counts",
        all(np.isclose(meta_sum[f"metabric_{d}_padj_DSS"], dss.at[d, "padj_DSS"], rtol=1e-9) for d in CANDIDATES)
        and sum(stats[f"dss_{d}_padj"] < 0.05 for d in CANDIDATES) == stats["dss_candidates_n_fdr05"]
        and sum(stats[f"dss_{d}_hr"] > 1 for d in CANDIDATES) == stats["dss_candidates_n_hr_gt1"],
        f"DSS FDR < 0.05 {stats['dss_candidates_fdr05']}, HR > 1 in {stats['dss_candidates_n_hr_gt1']}")
_xcheck("METABRIC DSS (script 63): OS columns = survival_METABRIC.csv, counts = meta_dss_summary.csv",
        np.allclose(np.log(dss.loc[cohort_dfs["METABRIC"]["drug"], "HR_OS"].values),
                    np.log(cohort_dfs["METABRIC"]["HR"].values), rtol=0, atol=1e-9)
        and [stats["dss_metabric_n_sig"], stats["dss_metabric_os_n_sig"], stats["dss_metabric_os_dss_overlap"],
             stats["dss_metabric_n_sig_hr_gt1"]]
        == [meta_sum[k] for k in ("metabric_dss_n_sig", "metabric_os_n_sig", "metabric_os_dss_overlap",
                                  "metabric_dss_n_sig_hr_gt1")],
        "DSS table and summary of script 63 disagree")

# METABRIC overall survival of the twelve drugs for Supplementary Table S1. The 95% CI
# of the penalized fit is rebuilt from its Wald p, |z| = Phi^-1(1 - p/2) and
# SE = |log HR| / |z|, as script 64 does, and checked below against the fitted
# intervals of script 43 for the five controls.
_met_os = cohort_dfs["METABRIC"].set_index("drug")
_z_met = norm.isf(_met_os["pvalue"] / 2)
_se_met = np.abs(np.log(_met_os["HR"])) / _z_met
_met_lo = np.exp(np.log(_met_os["HR"]) - norm.isf(0.025) * _se_met)
_met_hi = np.exp(np.log(_met_os["HR"]) + norm.isf(0.025) * _se_met)
for d in TWELVE_DRUGS:
    _add(f"metabric_os_{d}_hr", _r2(_met_os.at[d, "HR"]))
    _add(f"metabric_os_{d}_ci_low", _r2(_met_lo[d]))
    _add(f"metabric_os_{d}_ci_high", _r2(_met_hi[d]))
    _add(f"metabric_os_{d}_padj", _p2(_met_os.at[d, "padj"]))
_fc_met = forest[forest["cohort_key"] == "METABRIC"].set_index("drug")
_xcheck("METABRIC 95% CI rebuilt from the Wald p = fitted CI of script 43 for the five controls",
        all(np.isclose(_met_lo[d], _fc_met.at[d, "CI_low"], rtol=1e-9)
            and np.isclose(_met_hi[d], _fc_met.at[d, "CI_high"], rtol=1e-9) for d in positive_controls),
        "the Wald reconstruction does not reproduce the fitted intervals")
print(f"Script 63: direction {stats['direction_all_n_same']}/{stats['direction_all_n']} "
      f"({stats['direction_all_pct_same']}% vs {stats['direction_all_pct_expected']}% expected), Stouffer "
      f"{stats['stouffer_n_sig']} ({stats['stouffer_fisher_overlap']} shared with Fisher), controls P = "
      f"{stats['controls_baserate_fisher_p_at_least_observed']}, METABRIC DSS {stats['dss_metabric_n_sig']} "
      f"vs OS {stats['dss_metabric_os_n_sig']}")

# --- script 60: AURORA background rates for TFs, sites and drugs ---
aur_sum = _summary(REAUDIT / "aurora_background_summary.csv", "aurora_background_summary")
aur_tf_bg = load_csv_required(REAUDIT / "aurora_tf_background.csv", "aurora_tf_background").set_index("tf")
_n_tf = len(aur_tf_bg)
_n0005 = int((aur_tf_bg["pvalue"] < 0.005).sum())
_core_bg = aur_tf_bg[aur_tf_bg["is_core"].astype(bool)].sort_values("rank")
_add("aurora_bg_n_paired_patients", int(_one(aur_tf_bg["n_paired"], "n_paired of aurora_tf_background.csv")))
_add("aurora_bg_tf_n_tested", _n_tf)
_add("aurora_bg_tf_n_p_lt_0005", _n0005)
_add("aurora_bonferroni_threshold_core10", round(0.05 / len(_core_bg), 6))
_add("tf_activity_min_targets",
     int(re.search(r"min_targets: int = (\d+)",
                   (REPO / "src" / "data" / "load_prior_knowledge.py").read_text(encoding="utf-8")).group(1)))
_add("aurora_bg_tf_pct_p_lt_0005", _pct(_n0005, _n_tf))
_add("aurora_bg_tf_n_p_lt_005", int((aur_tf_bg["pvalue"] < 0.05).sum()))
_add("aurora_bg_tf_n_bh_005", int((aur_tf_bg["padj_bh"] < 0.05).sum()))
_add("aurora_bg_tf_n_bonferroni_005", int((aur_tf_bg["bonferroni_p"] < 0.05).sum()))
_add("aurora_bg_core_n_p_lt_0005", int((_core_bg["pvalue"] < 0.005).sum()))
_add("aurora_bg_core_expected_p_lt_0005", round(len(_core_bg) * _n0005 / _n_tf, 1))
_add("aurora_bg_core_hypergeom_p", _p2(aur_sum["core_hypergeom_p_ge_observed"]))
for tf, r in _core_bg.iterrows():
    _add(f"aurora_bg_core_rank_{tf}", int(r["rank"]))
_add("aurora_bg_core_rank_min", int(_core_bg["rank"].min()))
_add("aurora_bg_core_rank_max", int(_core_bg["rank"].max()))
# The background must be the test of Table 4, so the ten core TFs have to come out
# of both files with the same p-values and deltas.
_t4_ref = load_csv_required(RESULTS / "v9_aurora" / "aurora_tf_ig_validation.csv",
                            "aurora_tf_ig_validation").set_index("tf")
_xcheck("AURORA background (script 60) = Table 4 test for the ten core TFs, and its counts = its summary",
        set(_core_bg.index) == set(_t4_ref.index)
        and float((_core_bg["pvalue"] - _t4_ref.loc[_core_bg.index, "pvalue"]).abs().max()) < 1e-12
        and float((_core_bg["delta"] - _t4_ref.loc[_core_bg.index, "delta"]).abs().max()) < 1e-12
        and _p2(hypergeom.sf(stats["aurora_bg_core_n_p_lt_0005"] - 1, _n_tf, _n0005, len(_core_bg)))
        == stats["aurora_bg_core_hypergeom_p"]
        and [stats["aurora_bg_tf_n_p_lt_0005"], stats["aurora_bg_tf_n_p_lt_005"], stats["aurora_bg_tf_n_bh_005"],
             stats["aurora_bg_tf_n_bonferroni_005"], stats["aurora_bg_core_n_p_lt_0005"]]
        == [aur_sum[k] for k in ("tf_n_p_lt_0005", "tf_n_p_lt_005", "tf_n_bh_005", "tf_n_bonferroni_all_005",
                                 "core_n_p_lt_0005")]
        and all(aur_sum[f"core_rank_{tf}"] == stats[f"aurora_bg_core_rank_{tf}"] for tf in _core_bg.index),
        "per-TF table, Table 4 and summary of script 60 disagree")

site_bg = load_csv_required(REAUDIT / "aurora_site_background.csv", "aurora_site_background").set_index("tf")
_site_all = site_bg["n_concordant"] == site_bg["n_sites"]
_site_ge5 = site_bg["n_concordant"] >= 5
_site_sig = site_bg["pooled_pvalue"] < 0.005
_add("aurora_site_bg_n_sites", int(_one(site_bg["n_sites"], "n_sites of aurora_site_background.csv")))
_add("aurora_site_bg_n_tfs", int(len(site_bg)))
_add("aurora_site_bg_n_concordant_all_sites", int(_site_all.sum()))
_add("aurora_site_bg_n_concordant_ge5", int(_site_ge5.sum()))
_add("aurora_site_bg_sig_n_tfs", int(_site_sig.sum()))
_add("aurora_site_bg_sig_n_concordant_all_sites", int((_site_sig & _site_all).sum()))
_add("aurora_site_bg_sig_n_concordant_ge5", int((_site_sig & _site_ge5).sum()))
# Whole percentages, as the text prints them
_add("aurora_site_bg_sig_pct_concordant_all_sites",
     int(round(100 * (_site_sig & _site_all).sum() / _site_sig.sum())))
_add("aurora_site_bg_sig_pct_concordant_ge5", int(round(100 * (_site_sig & _site_ge5).sum() / _site_sig.sum())))
_xcheck("AURORA site background (script 60): counts = its summary, core concordance = script 39",
        [stats["aurora_site_bg_n_concordant_all_sites"], stats["aurora_site_bg_n_concordant_ge5"],
         stats["aurora_site_bg_sig_n_tfs"], stats["aurora_site_bg_sig_n_concordant_all_sites"],
         stats["aurora_site_bg_sig_n_concordant_ge5"]]
        == [aur_sum[k] for k in ("site_all_n_concordant_all", "site_all_n_concordant_ge5", "site_pooled_p0005_n_tfs",
                                 "site_pooled_p0005_n_concordant_all", "site_pooled_p0005_n_concordant_ge5")]
        and all(int(site_bg.at[tf, "n_concordant"]) == stats.get(f"aurora_site_n_concordant_{tf}")
                for tf in _core_bg.index),
        "site table, summary of script 60 and script 39 disagree")

drug_pair = load_csv_required(REAUDIT / "aurora_drug_paired.csv", "aurora_drug_paired").set_index("drug")
_bh_pair = drug_pair["padj_bh_paired"] < 0.05
_add("aurora_drug_paired_n_patients", int(_one(drug_pair["n_paired"], "n_paired of aurora_drug_paired.csv")))
_add("aurora_drug_paired_n_tested", int(len(drug_pair)))
_add("aurora_drug_paired_n_bh005", int(_bh_pair.sum()))
# Low predicted AUC means predicted sensitive: a negative delta is more active in metastasis
_add("aurora_drug_paired_n_bh005_more_active", int((_bh_pair & (drug_pair["delta_auc"] < 0)).sum()))
_add("aurora_drug_paired_n_bh005_less_active", int((_bh_pair & (drug_pair["delta_auc"] > 0)).sum()))
for d in ("entinostat", "paclitaxel", "panobinostat"):
    _add(f"aurora_drug_paired_{d}_p", _p2(drug_pair.at[d, "pvalue_paired"]))
# Results: entinostat was predicted less active (higher AUC) in metastases
_add("aurora_drug_paired_entinostat_less_active_in_metastasis", bool(drug_pair.at["entinostat", "delta_auc"] > 0))
_xcheck("AURORA paired drug test (script 60): counts = its summary",
        [stats["aurora_drug_paired_n_bh005"], stats["aurora_drug_paired_n_bh005_more_active"],
         stats["aurora_drug_paired_n_bh005_less_active"]]
        == [aur_sum[k] for k in ("drug_patient_n_bh005", "drug_patient_n_bh005_more_active",
                                 "drug_patient_n_bh005_less_active")]
        and np.isclose(aur_sum["drug_entinostat_pvalue_paired"], drug_pair.at["entinostat", "pvalue_paired"],
                       rtol=1e-12),
        "drug table and summary of script 60 disagree")
print(f"AURORA background (script 60): {stats['aurora_bg_tf_n_p_lt_0005']}/{stats['aurora_bg_tf_n_tested']} TFs "
      f"below 0.005 ({stats['aurora_bg_tf_pct_p_lt_0005']}%), core {stats['aurora_bg_core_n_p_lt_0005']} vs "
      f"{stats['aurora_bg_core_expected_p_lt_0005']} expected (P = {stats['aurora_bg_core_hypergeom_p']}), paired "
      f"drugs {stats['aurora_drug_paired_n_bh005']} at FDR<0.05")

# --- script 61: TNBC and Basal subgroups, predicted AUC by PAM50 subtype ---
tnbc_sum = _summary(REAUDIT / "tnbc_subgroup_summary.csv", "tnbc_subgroup_summary")
tnbc_cox = load_csv_required(REAUDIT / "tnbc_subgroup_cox.csv", "tnbc_subgroup_cox").set_index(["subset", "drug"])
for subset, tag in (("TNBC_clinical", "tnbc"), ("PAM50_Basal", "basal")):
    sub = tnbc_cox.loc[subset]
    _add(f"{tag}_n_patients", int(_one(sub["n"], f"n of {subset}")))
    _add(f"{tag}_n_events", int(_one(sub["events"], f"events of {subset}")))
    for d in TWELVE_DRUGS:
        _add(f"{tag}_{d}_hr_per_sd", _r2(sub.at[d, "hr_per_sd"]))
        _add(f"{tag}_{d}_ci_low", _r2(sub.at[d, "ci_low"]))
        _add(f"{tag}_{d}_ci_high", _r2(sub.at[d, "ci_high"]))
        _add(f"{tag}_{d}_p", _p2(sub.at[d, "p"]))
        _add(f"{tag}_{d}_padj", _p2(sub.at[d, "p_adj_family"]))
    _add(f"{tag}_candidates_fdr05", [d for d in CANDIDATES if sub.at[d, "p_adj_family"] < 0.05])
    _add(f"{tag}_candidates_n_fdr05", len(stats[f"{tag}_candidates_fdr05"]))
    _add(f"{tag}_candidates_n_hr_gt1", int((sub.loc[CANDIDATES, "hr_per_sd"] > 1).sum()))
    _add(f"{tag}_controls_n_fdr05", int((sub.loc[positive_controls, "p_adj_family"] < 0.05).sum()))
    _xcheck(f"{subset} (script 61): counts from the per-drug table = its summary",
            [stats[f"{tag}_candidates_n_fdr05"], stats[f"{tag}_candidates_n_hr_gt1"],
             stats[f"{tag}_controls_n_fdr05"]]
            == [tnbc_sum[f"{subset}_candidate_n_padj05"], tnbc_sum[f"{subset}_candidate_n_hr_above1"],
                tnbc_sum[f"{subset}_positive_control_n_padj05"]],
            "per-drug table and summary of script 61 disagree")
_add("tnbc_n_tnbc_and_basal", int(tnbc_sum["n_tnbc_and_basal"]))
_add("tnbc_pct_tnbc_that_are_basal", _pct(tnbc_sum["n_tnbc_and_basal"], stats["tnbc_n_patients"]))
_add("tnbc_pct_basal_that_are_tnbc", _pct(tnbc_sum["n_tnbc_and_basal"], stats["basal_n_patients"]))
# Script 61 fits the full-cohort PAM50-stratified model as well, the same unpenalized
# model as script 64. The two scripts must agree.
_strat61 = tnbc_cox.loc["full_cohort_pam50_strata"]

pam_auc = load_csv_required(REAUDIT / "pam50_predicted_auc.csv", "pam50_predicted_auc").set_index("drug")
_PAM50 = ("LumA", "LumB", "Her2", "Basal", "Normal")
_n_pam = {s: int(_one(pam_auc[f"n_{s}"], f"n_{s} of pam50_predicted_auc.csv")) for s in _PAM50}
for s in _PAM50:
    _add(f"pam50_cohort_pct_{s}", _pct(_n_pam[s], sum(_n_pam.values())))
_less = pam_auc["basal_minus_luma_sd"] > 0   # higher predicted AUC in Basal = less sensitive
_add("pam50_candidates_basal_less_sensitive", [d for d in CANDIDATES if _less[d]])
_add("pam50_n_candidates_basal_less_sensitive", int(_less[CANDIDATES].sum()))
_add("pam50_n_controls_basal_less_sensitive", int(_less[positive_controls].sum()))
comp = load_csv_required(REAUDIT / "pam50_composition_sensitive_half.csv", "pam50_composition_sensitive_half")
_sens = comp[comp["half"] == "sensitive"].set_index("drug")
for d in TWELVE_DRUGS:
    _add(f"pam50_sensitive_half_pct_luma_{d}", round(float(_sens.at[d, "pct_LumA"]), 1))
    _add(f"pam50_sensitive_half_pct_basal_{d}", round(float(_sens.at[d, "pct_Basal"]), 1))
_add("pam50_sensitive_half_pct_luma_candidates_min", round(float(_sens.loc[CANDIDATES, "pct_LumA"].min()), 1))
_add("pam50_sensitive_half_pct_luma_candidates_max", round(float(_sens.loc[CANDIDATES, "pct_LumA"].max()), 1))
_xcheck("PAM50 composition (script 61): cohort = the 7,215-patient Cox cohort, flags and counts = its summary",
        sum(_n_pam.values()) == stats["SCANB_n_patients_cox"]
        and (_less == pam_auc["basal_less_sensitive_than_luma"].astype(bool)).all()
        and stats["pam50_n_candidates_basal_less_sensitive"] == tnbc_sum["n_candidates_less_sensitive_in_basal_than_luma"]
        and stats["pam50_n_controls_basal_less_sensitive"] == tnbc_sum["n_controls_less_sensitive_in_basal_than_luma"]
        and all(int(g["n"].sum()) == stats["SCANB_n_patients_cox"] for _, g in comp.groupby("drug")),
        f"cohort {sum(_n_pam.values())}")
print(f"TNBC subgroup (script 61): {stats['tnbc_n_patients']} patients ({stats['tnbc_n_events']} events), "
      f"FDR<0.05 {stats['tnbc_candidates_fdr05']}, Basal {stats['basal_candidates_fdr05']}, LumA share of the "
      f"sensitive half {stats['pam50_sensitive_half_pct_luma_candidates_min']}-"
      f"{stats['pam50_sensitive_half_pct_luma_candidates_max']}% vs {stats['pam50_cohort_pct_LumA']}%")

# --- script 62: treatment as confounder and as effect modifier ---
treat_sum = _summary(REAUDIT / "treatment_summary.csv", "treatment_summary")
for cohort in ("SCANB", "METABRIC"):
    for k in ("yes", "no", "missing"):
        _add(f"treat_{cohort.lower()}_chemo_{k}", int(treat_sum[f"{cohort}_chemo_{k}"]))
treat_adj = load_csv_required(REAUDIT / "treatment_adjusted_cox.csv", "treatment_adjusted_cox")
# clinical_pen01_dummies is the comparison fit in the form of script 37 and is not cited
_TREAT_MODELS = ("pam50_strata", "clinical", "clinical_treatment")
for cohort in ("SCANB", "METABRIC"):
    tag = cohort.lower()
    rows_c = treat_adj[treat_adj["cohort"] == cohort]
    _add(f"treat_{tag}_complete_n_patients", int(_one(rows_c["n_patients"], f"n_patients of {cohort}")))
    _add(f"treat_{tag}_complete_n_events", int(_one(rows_c["n_events"], f"n_events of {cohort}")))
    for model in _TREAT_MODELS:
        g = rows_c[rows_c["model"] == model].set_index("drug")
        for d in TWELVE_DRUGS:
            _add(f"treat_{tag}_{model}_{d}_hr_per_sd", _r2(g.at[d, "HR"]))
            _add(f"treat_{tag}_{model}_{d}_ci_low", _r2(g.at[d, "CI_low"]))
            _add(f"treat_{tag}_{model}_{d}_ci_high", _r2(g.at[d, "CI_high"]))
            _add(f"treat_{tag}_{model}_{d}_p", _p2(g.at[d, "p"]))
            _add(f"treat_{tag}_{model}_{d}_padj", _p2(g.at[d, "padj"]))
    g = rows_c[rows_c["model"] == "clinical_treatment"].set_index("drug")
    _add(f"treat_{tag}_clinical_treatment_candidates_hr_min", _r2(g.loc[CANDIDATES, "HR"].min()))
    _add(f"treat_{tag}_clinical_treatment_candidates_hr_max", _r2(g.loc[CANDIDATES, "HR"].max()))
    _add(f"treat_{tag}_clinical_treatment_fdr05_drugs", [d for d in TWELVE_DRUGS if g.at[d, "padj"] < 0.05])
    _add(f"treat_{tag}_clinical_treatment_fdr05_candidates", [d for d in CANDIDATES if g.at[d, "padj"] < 0.05])

treat_int = load_csv_required(REAUDIT / "treatment_interaction_cox.csv", "treatment_interaction_cox")
# Interaction of the standardized predicted AUC with chemotherapy, PAM50 as strata (the
# model the text reports). The drug-specific exposures are not recorded in either cohort.
_ti = treat_int[(treat_int["exposure"] == "chemo") & (treat_int["adjustment"] == "pam50_strata")]
for cohort in ("SCANB", "METABRIC"):
    tag = cohort.lower()
    g = _ti[_ti["cohort"] == cohort].set_index("drug")
    _add(f"treat_{tag}_chemo_interaction_n_patients", int(_one(g["n_patients"], f"interaction n of {cohort}")))
    _add(f"treat_{tag}_chemo_interaction_n_events", int(_one(g["n_events"], f"interaction events of {cohort}")))
    for d in TWELVE_DRUGS:
        _add(f"treat_{tag}_chemo_interaction_{d}_hr", _r2(g.at[d, "HR_interaction"]))
        _add(f"treat_{tag}_chemo_interaction_{d}_p", _p2(g.at[d, "p_interaction"]))
        _add(f"treat_{tag}_chemo_interaction_{d}_padj", _p2(g.at[d, "padj_interaction"]))
        _add(f"treat_{tag}_chemo_interaction_{d}_hr_untreated", _r2(g.at[d, "HR_within_unexposed"]))
        _add(f"treat_{tag}_chemo_interaction_{d}_hr_treated", _r2(g.at[d, "HR_within_exposed"]))
    _add(f"treat_{tag}_chemo_interaction_n_p05", int((g.loc[TWELVE_DRUGS, "p_interaction"] < 0.05).sum()))
    _add(f"treat_{tag}_chemo_interaction_n_padj05", int((g.loc[TWELVE_DRUGS, "padj_interaction"] < 0.05).sum()))
    _add(f"treat_{tag}_chemo_interaction_fdr05_drugs",
         [d for d in TWELVE_DRUGS if g.at[d, "padj_interaction"] < 0.05])
    _xcheck(f"Treatment (script 62, {cohort}): sizes and interaction counts = its summary",
            stats[f"treat_{tag}_complete_n_patients"] == treat_sum[f"{cohort}_A_n_patients"]
            and stats[f"treat_{tag}_complete_n_events"] == treat_sum[f"{cohort}_A_n_events"]
            and stats[f"treat_{tag}_chemo_interaction_n_p05"]
            == treat_sum[f"{cohort}_B_chemo_pam50_strata_n_interaction_p005"]
            and stats[f"treat_{tag}_chemo_interaction_n_padj05"]
            == treat_sum[f"{cohort}_B_chemo_pam50_strata_n_interaction_padj005"]
            and stats[f"treat_{tag}_chemo_interaction_n_patients"]
            == treat_sum[f"{cohort}_B_chemo_pam50_strata_n_patients"],
            "per-drug tables and summary of script 62 disagree")

# The same interaction refitted with the clinical covariates of the treatment-adjusted
# models and endocrine therapy as main effects (adjustment clinical_endo), on the
# complete cases with known chemotherapy, because chemotherapy is not randomized.
# padj is the BH of script 62 over the twelve drugs within each cohort and adjustment.
_ti_adj = treat_int[(treat_int["exposure"] == "chemo") & (treat_int["adjustment"] == "clinical_endo")]
for cohort in ("SCANB", "METABRIC"):
    tag = cohort.lower()
    pre = f"treat_{tag}_chemo_interaction_clinical_endo"
    g = _ti_adj[_ti_adj["cohort"] == cohort].set_index("drug")
    _add(f"{pre}_n_patients", int(_one(g["n_patients"], f"clinical_endo interaction n of {cohort}")))
    _add(f"{pre}_n_events", int(_one(g["n_events"], f"clinical_endo interaction events of {cohort}")))
    for d in TWELVE_DRUGS:
        _add(f"{pre}_{d}_hr", _r2(g.at[d, "HR_interaction"]))
        _add(f"{pre}_{d}_p", _p2(g.at[d, "p_interaction"]))
        _add(f"{pre}_{d}_padj", _p2(g.at[d, "padj_interaction"]))
        _add(f"{pre}_{d}_hr_untreated", _r2(g.at[d, "HR_within_unexposed"]))
        _add(f"{pre}_{d}_hr_treated", _r2(g.at[d, "HR_within_exposed"]))
    _add(f"{pre}_n_p05", int((g.loc[TWELVE_DRUGS, "p_interaction"] < 0.05).sum()))
    _add(f"{pre}_n_padj05", int((g.loc[TWELVE_DRUGS, "padj_interaction"] < 0.05).sum()))
    _add(f"{pre}_padj_min", _p2(g.loc[TWELVE_DRUGS, "padj_interaction"].min()))
    g_pam = _ti[_ti["cohort"] == cohort].set_index("drug")
    _s5 = ("erlotinib", "entinostat", "paclitaxel", "epirubicin", "talazoparib")
    _xcheck(f"Chemotherapy interaction (script 62, {cohort}): padj = BH over the twelve drugs in both adjustments, "
            "HR within treated / HR within untreated = HR of the interaction, clinical_endo cohort = the complete "
            "cases of the treatment-adjusted models with known chemotherapy, counts and five drugs = its summary",
            all(set(t.index) == set(TWELVE_DRUGS) and (t["status"] == "ok").all()
                and np.allclose(t["padj_interaction"], false_discovery_control(t["p_interaction"]), rtol=1e-9, atol=0)
                and np.allclose(t["HR_within_exposed"] / t["HR_within_unexposed"], t["HR_interaction"],
                                rtol=1e-9, atol=0)
                for t in (g, g_pam))
            and stats[f"{pre}_n_patients"] == treat_sum[f"{cohort}_B_chemo_clinical_endo_n_patients"]
            == treat_sum[f"{cohort}_A_chemo_yes"] + treat_sum[f"{cohort}_A_chemo_no"]
            and int(_one(g["n_exposed"], "n_exposed")) == treat_sum[f"{cohort}_A_chemo_yes"]
            and int(_one(g["n_cohort"], "n_cohort")) == stats[f"treat_{tag}_complete_n_patients"]
            and stats[f"{pre}_n_events"] == treat_sum[f"{cohort}_B_chemo_clinical_endo_n_events"]
            and stats[f"{pre}_n_p05"] == treat_sum[f"{cohort}_B_chemo_clinical_endo_n_interaction_p005"]
            and stats[f"{pre}_n_padj05"] == treat_sum[f"{cohort}_B_chemo_clinical_endo_n_interaction_padj005"]
            and all(round(float(g.at[d, "HR_interaction"]), 3)
                    == treat_sum[f"{cohort}_B_chemo_clinical_endo_{d}_HR_interaction"]
                    and np.isclose(g.at[d, "p_interaction"],
                                   treat_sum[f"{cohort}_B_chemo_clinical_endo_{d}_p_interaction"], rtol=1e-9)
                    for d in _s5),
            f"{cohort} clinical_endo n {stats[f'{pre}_n_patients']}, padj < 0.05 {stats[f'{pre}_n_padj05']}")

# Candidates whose HR per SD is larger in untreated than in treated SCAN-B patients
# (PAM50-strata interaction model), that is, whose interaction HR is below 1
_g_int = _ti[_ti["cohort"] == "SCANB"].set_index("drug").loc[CANDIDATES]
_untr_gt = _g_int["HR_within_unexposed"] > _g_int["HR_within_exposed"]
_add("treat_scanb_chemo_interaction_n_candidates_hr_untreated_gt_treated", int(_untr_gt.sum()))
_add("treat_scanb_chemo_interaction_candidates_hr_untreated_not_gt_treated",
     [d for d in CANDIDATES if not _untr_gt[d]])
_xcheck("Candidates with a larger HR per SD in untreated SCAN-B patients = candidates with interaction HR < 1, "
        "and the rounded keys order them the same way",
        (_untr_gt == (_g_int["HR_interaction"] < 1)).all()
        and all((stats[f"treat_scanb_chemo_interaction_{d}_hr_untreated"]
                 > stats[f"treat_scanb_chemo_interaction_{d}_hr_treated"]) == bool(_untr_gt[d]) for d in CANDIDATES),
        f"{stats['treat_scanb_chemo_interaction_candidates_hr_untreated_not_gt_treated']}")
print(f"Treatment (script 62): SCAN-B complete cases {stats['treat_scanb_complete_n_patients']} / "
      f"{stats['treat_scanb_complete_n_events']}, candidates HR per SD "
      f"{stats['treat_scanb_clinical_treatment_candidates_hr_min']}-"
      f"{stats['treat_scanb_clinical_treatment_candidates_hr_max']} after treatment adjustment, FDR<0.05 "
      f"{stats['treat_scanb_clinical_treatment_fdr05_candidates']}, chemotherapy interactions p<0.05: SCAN-B "
      f"{stats['treat_scanb_chemo_interaction_n_p05']}, METABRIC {stats['treat_metabric_chemo_interaction_n_p05']}")

# --- script 64: proportional hazards, PAM50-stratified Cox, hazard ratios per SD ---
ph_sum = _summary(REAUDIT / "ph_summary.csv", "ph_summary")
ph_tab = load_csv_required(REAUDIT / "ph_schoenfeld.csv", "ph_schoenfeld")
_add("ph_cox_n_patients", int(ph_sum["cox_n_patients"]))
_add("ph_cox_n_events", int(ph_sum["cox_n_events"]))
_ph_drug = ph_tab[ph_tab["term"] == "predicted_auc"].set_index("drug")
_ph_pam = ph_tab[ph_tab["term"] != "predicted_auc"]
# ph_p_exact is the score test of R survival 3.x cox.zph (rank time pooled across
# strata), the test Methods describes and on which script 64 rests its conclusions.
# ph_p is the lifelines approximation (pre-3.0 R), kept for reference.
for test, col in (("exact", "ph_p_exact"), ("lifelines", "ph_p")):
    violating = sorted(_ph_drug.index[_ph_drug[col] < 0.05])
    _add(f"ph_{test}_drug_terms_violating", violating)
    _add(f"ph_{test}_n_drug_terms_violating", len(violating))
    for term, g in _ph_pam.groupby("term"):
        _add(f"ph_{test}_{term}_n_models_violating", int((g[col] < 0.05).sum()))
        _add(f"ph_{test}_{term}_p_max", _p2(g[col].max()))
for d in TWELVE_DRUGS:
    _add(f"ph_{d}_p_exact", _p2(_ph_drug.at[d, "ph_p_exact"]))
_xcheck("Proportional hazards (script 64): counts from ph_schoenfeld.csv = ph_summary.csv",
        stats["ph_exact_drug_terms_violating"] == ph_sum["ph_drug_terms_violating_p05_exact"]
        and stats["ph_lifelines_drug_terms_violating"] == ph_sum["ph_drug_terms_violating_p05"]
        and all(stats[f"ph_exact_{t}_n_models_violating"] == ph_sum[f"ph_{t}_n_models_violating_p05_exact"]
                and stats[f"ph_lifelines_{t}_n_models_violating"] == ph_sum[f"ph_{t}_n_models_violating_p05"]
                for t in _ph_pam["term"].unique())
        and len(_ph_drug) == len(TWELVE_DRUGS),
        "Schoenfeld table and summary of script 64 disagree")

strat = load_csv_required(REAUDIT / "cox_pam50_stratified.csv", "cox_pam50_stratified").set_index("drug")
# ph_p_strat_* is the exact test with the rank of time pooled across strata, the
# primary test of script 64 for the stratified models.
_add("ph_strat_pen_n_drug_terms_violating", int((strat["ph_p_strat_pen"] < 0.05).sum()))
_add("ph_strat_unpen_n_drug_terms_violating", int((strat["ph_p_strat_unpen"] < 0.05).sum()))
_add("ph_strat_pen_p_min", _p2(strat["ph_p_strat_pen"].min()))
_add("ph_strat_unpen_p_min", _p2(strat["ph_p_strat_unpen"].min()))
for d in TWELVE_DRUGS:
    _add(f"strat_{d}_hr_per_sd", _r2(strat.at[d, "HR_per_sd_strat_unpen"]))
    _add(f"strat_{d}_ci_low", _r2(strat.at[d, "CI_low_per_sd_strat_unpen"]))
    _add(f"strat_{d}_ci_high", _r2(strat.at[d, "CI_high_per_sd_strat_unpen"]))
    _add(f"strat_{d}_p", _p2(strat.at[d, "p_strat_unpen"]))
_add("strat_candidates_hr_per_sd_min", _r2(strat.loc[CANDIDATES, "HR_per_sd_strat_unpen"].min()))
_add("strat_candidates_hr_per_sd_max", _r2(strat.loc[CANDIDATES, "HR_per_sd_strat_unpen"].max()))
_add("strat_n_p05", int((strat.loc[TWELVE_DRUGS, "p_strat_unpen"] < 0.05).sum()))
_add("strat_candidates_n_p05", int((strat.loc[CANDIDATES, "p_strat_unpen"] < 0.05).sum()))
# Penalized stratified fit (penalizer 0.1, as the screen), HR per SD of the candidates
_add("strat_pen_candidates_hr_per_sd_min", _r2(strat.loc[CANDIDATES, "HR_per_sd_strat_pen"].min()))
_add("strat_pen_candidates_hr_per_sd_max", _r2(strat.loc[CANDIDATES, "HR_per_sd_strat_pen"].max()))
_xcheck("PAM50-stratified fits (script 64): penalized HR per SD = HR per unit ** SD and HR per unit = ph_summary.csv, "
        "and the smallest PH p of the drug term (exact test, unrounded) is above 0.10 in the penalized and the "
        "unpenalized fits and = ph_summary.csv",
        np.allclose(strat["HR_per_sd_strat_pen"], strat["HR_strat_pen"] ** strat["sd_predicted_auc"], rtol=1e-9, atol=0)
        and all(np.isclose(strat.at[d, "HR_strat_pen"], ph_sum[f"{d}_HR_strat_pen"], rtol=1e-12) for d in TWELVE_DRUGS)
        and float(strat["ph_p_strat_pen"].min()) > 0.10 and float(strat["ph_p_strat_unpen"].min()) > 0.10
        and np.isclose(strat["ph_p_strat_pen"].min(), ph_sum["strat_pen_ph_p_min"], rtol=1e-12)
        and np.isclose(strat["ph_p_strat_unpen"].min(), ph_sum["strat_unpen_ph_p_min"], rtol=1e-12),
        f"PH p min {strat['ph_p_strat_pen'].min()} (penalized) and {strat['ph_p_strat_unpen'].min()} (unpenalized)")
_xcheck("PAM50-stratified unpenalized Cox: script 64 = full-cohort model of script 61, and its counts = ph_summary.csv",
        all(np.isclose(strat.at[d, "HR_per_sd_strat_unpen"], _strat61.at[d, "hr_per_sd"], rtol=1e-6)
            and np.isclose(strat.at[d, "p_strat_unpen"], _strat61.at[d, "p"], rtol=1e-4)
            for d in TWELVE_DRUGS)
        and stats["ph_strat_pen_n_drug_terms_violating"] == ph_sum["strat_pen_n_drug_terms_violating_ph_p05"]
        and stats["ph_strat_unpen_n_drug_terms_violating"] == ph_sum["strat_unpen_n_drug_terms_violating_ph_p05"]
        and stats["strat_n_p05"] == ph_sum["strat_unpen_n_p05"],
        "scripts 61 and 64 disagree on the stratified model")

hrsd = load_csv_required(REAUDIT / "hr_per_sd_all_drugs.csv", "hr_per_sd_all_drugs").set_index("drug")
for d in TWELVE_DRUGS:
    _add(f"hrsd_{d}", _r2(hrsd.at[d, "HR_per_sd"]))
_add("hrsd_candidates_min", _r2(hrsd.loc[CANDIDATES, "HR_per_sd"].min()))
_add("hrsd_candidates_max", _r2(hrsd.loc[CANDIDATES, "HR_per_sd"].max()))
_add("hrsd_candidates_sd_min", _r2(hrsd.loc[CANDIDATES, "sd_predicted_auc"].min()))
_add("hrsd_candidates_sd_max", _r2(hrsd.loc[CANDIDATES, "sd_predicted_auc"].max()))
_hrsd_fdr = hrsd[hrsd["padj"] < 0.05]
_add("hrsd_fdr_n_drugs", int(len(_hrsd_fdr)))
_add("hrsd_fdr_hr_per_sd_min", _r2(_hrsd_fdr["HR_per_sd"].min()))
_add("hrsd_fdr_hr_per_sd_max", _r2(_hrsd_fdr["HR_per_sd"].max()))
_add("hrsd_fdr_hr_per_unit_min", round(float(_hrsd_fdr["HR"].min()), 4))
_add("hrsd_fdr_hr_per_unit_max", round(float(_hrsd_fdr["HR"].max()), 1))
# The window [1.5, 10] of the selection rule is on the hazard ratio per unit of
# predicted AUC. On the per-SD scale it is [1.5 ** SD, 10 ** SD] for each drug, so it
# moves with the spread of that drug's predictions: the smallest lower bound and the
# largest upper bound over the seven candidates, and the per-SD range of the drugs of
# the statistical filter.
_lo, _hi = SELECTION_RULE["hr_range"]
_sd_c = hrsd.loc[CANDIDATES, "sd_predicted_auc"]
_add("window_candidates_per_sd_low_min", _r2(_lo ** _sd_c.min()))
_add("window_candidates_per_sd_high_max", _r2(_hi ** _sd_c.max()))
_win = hrsd.loc[step1["drug"]]
_add("window_filter_hr_per_sd_min", _r2(_win["HR_per_sd"].min()))
_add("window_filter_hr_per_sd_max", _r2(_win["HR_per_sd"].max()))
_xcheck("Selection window per SD (script 64): the 143 drugs of the statistical filter have HR per unit in the "
        "window and HR per SD = HR ** SD, their per-SD range and the SD of the candidates = ph_summary.csv, and "
        "each candidate lies inside its own per-SD window",
        len(_win) == stats["candidates_n_statistical_filter"] == int(ph_sum["window_in_n"])
        and _win["HR"].between(_lo, _hi).all()
        and np.allclose(_win["HR_per_sd"], _win["HR"] ** _win["sd_predicted_auc"], rtol=1e-9, atol=0)
        and np.isclose(_win["HR_per_sd"].min(), ph_sum["window_in_HR_per_sd_min"], rtol=1e-12)
        and np.isclose(_win["HR_per_sd"].max(), ph_sum["window_in_HR_per_sd_max"], rtol=1e-12)
        and np.isclose(_sd_c.min(), ph_sum["candidates_sd_min"], rtol=1e-12)
        and np.isclose(_sd_c.max(), ph_sum["candidates_sd_max"], rtol=1e-12)
        and all(_lo ** hrsd.at[d, "sd_predicted_auc"] <= hrsd.at[d, "HR_per_sd"] <= _hi ** hrsd.at[d, "sd_predicted_auc"]
                for d in CANDIDATES),
        f"{len(_win)} drugs, HR per SD {_win['HR_per_sd'].min()} to {_win['HR_per_sd'].max()}")
_top001 = hrsd.loc[hrsd["HR_pen001"].idxmax()]
_add("hrsd_pen001_max_hr_drug", str(_top001.name))
_add("hrsd_pen001_max_hr", round(float(_top001["HR_pen001"]), 1))
_add("hrsd_pen001_max_hr_sd_percentile", _r2(_top001["sd_percentile"]))
_xcheck("HR per SD (script 64): hazard ratios = survival_SCANB.csv, per-SD values = ph_summary.csv and the "
        "stratified-file conversion",
        np.allclose(hrsd.loc[_hr_s.index, "HR"].values, _hr_s["HR"].values, rtol=1e-12)
        and all(np.isclose(hrsd.at[d, "HR_per_sd"], ph_sum[f"{d}_HR_per_sd"], rtol=1e-12) for d in TWELVE_DRUGS)
        and all(np.isclose(hrsd.at[d, "HR_per_sd"], strat.at[d, "HR_per_sd_canonical"], rtol=1e-9)
                for d in TWELVE_DRUGS)
        and ph_sum["pen001_max_HR_drug"] == stats["hrsd_pen001_max_hr_drug"],
        "HR per SD sources disagree")
print(f"Script 64: PH violated by the drug term in {stats['ph_exact_n_drug_terms_violating']}/12 models "
      f"(exact test), {stats['ph_strat_unpen_n_drug_terms_violating']} once PAM50 is a stratum, stratified "
      f"candidates {stats['strat_candidates_hr_per_sd_min']}-{stats['strat_candidates_hr_per_sd_max']} per SD, "
      f"canonical {stats['hrsd_candidates_min']}-{stats['hrsd_candidates_max']}, FDR-significant drugs "
      f"{stats['hrsd_fdr_hr_per_sd_min']}-{stats['hrsd_fdr_hr_per_sd_max']} per SD")

# --- script 64, METABRIC: the same proportional hazards test on the canonical METABRIC model ---
# Methods states the check without exempting a cohort, so the twelve drugs are refitted
# on METABRIC exactly as on SCAN-B: every term of the penalized model with PAM50
# dummies, the unpenalized refit with the same covariates, and the PAM50-stratified
# fits, with the exact test (rank of time pooled across strata).
ph_m_sum = _summary(REAUDIT / "ph_summary_metabric.csv", "ph_summary_metabric")
ph_m_tab = load_csv_required(REAUDIT / "ph_schoenfeld_metabric.csv", "ph_schoenfeld_metabric")
strat_m = load_csv_required(REAUDIT / "cox_pam50_stratified_metabric.csv",
                            "cox_pam50_stratified_metabric").set_index("drug")
_ph_m_drug = ph_m_tab[ph_m_tab["term"] == "predicted_auc"].set_index("drug")
_ph_m_pam = ph_m_tab[ph_m_tab["term"] != "predicted_auc"]
_add("ph_metabric_cox_n_patients", int(ph_m_sum["cox_n_patients"]))
_add("ph_metabric_cox_n_events", int(ph_m_sum["cox_n_events"]))
for _tag, _col in (("exact", "ph_p_exact"), ("exact_unpenalized", "ph_p_exact_unpenalized")):
    _add(f"ph_metabric_{_tag}_drug_terms_violating", sorted(_ph_m_drug.index[_ph_m_drug[_col] < 0.05]))
    _add(f"ph_metabric_{_tag}_n_drug_terms_violating", int((_ph_m_drug[_col] < 0.05).sum()))
    _add(f"ph_metabric_{_tag}_n_pam50_terms_violating", int((_ph_m_pam[_col] < 0.05).sum()))
    for _term, _g in _ph_m_pam.groupby("term"):
        _add(f"ph_metabric_{_tag}_{_term}_n_models_violating", int((_g[_col] < 0.05).sum()))
_add("ph_metabric_n_pam50_terms", int(len(_ph_m_pam)))
for _model in ("strat_pen", "strat_unpen"):
    _add(f"ph_metabric_{_model}_drug_terms_violating", sorted(strat_m.index[strat_m[f"ph_p_{_model}"] < 0.05]))
    _add(f"ph_metabric_{_model}_n_drug_terms_violating", int((strat_m[f"ph_p_{_model}"] < 0.05).sum()))
_surv_m = load_csv_required(RESULTS / "v8_multicohort" / "survival_METABRIC.csv", "survival_METABRIC").set_index("drug")
_xcheck("Proportional hazards in METABRIC (script 64): the canonical fits reproduce survival_METABRIC.csv for the "
        "twelve drugs on the PAM50-adjusted Cox cohort of Table 2, the table holds the drug term and the four "
        "PAM50 terms of each, its p-values are probabilities, its counts are those of the summary and the "
        "information matrix of the test matches the lifelines variance",
        sorted(_ph_m_drug.index) == sorted(TWELVE_DRUGS) and sorted(strat_m.index) == sorted(TWELVE_DRUGS)
        and all(np.isclose(_ph_m_drug.at[d, "HR"], _surv_m.at[d, "HR"], rtol=1e-6)
                and np.isclose(strat_m.at[d, "HR_canonical"], _surv_m.at[d, "HR"], rtol=1e-6)
                and np.isclose(strat_m.at[d, "p_canonical"], _surv_m.at[d, "pvalue"], rtol=1e-6)
                for d in TWELVE_DRUGS)
        and (stats["ph_metabric_cox_n_patients"], stats["ph_metabric_cox_n_events"])
        == (stats["METABRIC_n_patients_cox"], stats["METABRIC_n_events_cox"])
        and (strat_m["n_patients"] == stats["METABRIC_n_patients_cox"]).all()
        and (strat_m["n_events"] == stats["METABRIC_n_events_cox"]).all()
        and stats["ph_metabric_n_pam50_terms"] == 4 * len(TWELVE_DRUGS) == int(ph_m_sum["ph_n_pam50_term_tests"])
        and _ph_m_pam.groupby("term").size().eq(len(TWELVE_DRUGS)).all() and _ph_m_pam["term"].nunique() == 4
        and ph_m_tab[["ph_p_exact", "ph_p_exact_unpenalized"]].gt(0).all().all()
        and ph_m_tab[["ph_p_exact", "ph_p_exact_unpenalized"]].le(1).all().all()
        and stats["ph_metabric_exact_drug_terms_violating"] == ph_m_sum["ph_drug_terms_violating_p05_exact"]
        and stats["ph_metabric_exact_unpenalized_drug_terms_violating"]
        == ph_m_sum["ph_drug_terms_violating_p05_exact_unpenalized"]
        and stats["ph_metabric_exact_n_pam50_terms_violating"] == ph_m_sum["ph_n_pam50_term_tests_violating_p05_exact"]
        and stats["ph_metabric_exact_unpenalized_n_pam50_terms_violating"]
        == ph_m_sum["ph_n_pam50_term_tests_violating_p05_exact_unpenalized"]
        and all(stats[f"ph_metabric_exact_{t}_n_models_violating"] == ph_m_sum[f"ph_{t}_n_models_violating_p05_exact"]
                for t in _ph_m_pam["term"].unique())
        and stats["ph_metabric_strat_pen_drug_terms_violating"] == ph_m_sum["strat_pen_drug_terms_violating_ph_p05"]
        and stats["ph_metabric_strat_unpen_drug_terms_violating"] == ph_m_sum["strat_unpen_drug_terms_violating_ph_p05"]
        and float(ph_m_sum["zph_exact_max_variance_rel_diff_vs_lifelines"]) <= 1e-6
        and int(ph_m_sum["zph_exact_n_models"]) == 4 * len(TWELVE_DRUGS),
        f"drug terms {stats['ph_metabric_exact_drug_terms_violating']}, PAM50 "
        f"{stats['ph_metabric_exact_n_pam50_terms_violating']} of {stats['ph_metabric_n_pam50_terms']}")
print(f"Script 64, METABRIC: PH violated by the drug term in {stats['ph_metabric_exact_n_drug_terms_violating']}/12 "
      f"canonical models (exact test), PAM50 terms {stats['ph_metabric_exact_n_pam50_terms_violating']}/"
      f"{stats['ph_metabric_n_pam50_terms']}, {stats['ph_metabric_strat_unpen_n_drug_terms_violating']} drug terms "
      f"once PAM50 is a stratum")

# --- Supplementary Tables S3 (robustness) and S1 (replication), built from the keys above ---
# Rows: the candidates in the order of Table 5, then the five positive controls.
_S_ORDER = selected.sort_values("HR", ascending=False)["drug"].tolist() + positive_controls


def _fmt_p(key):
    return f"{stats[key]:.1e}"


def _fmt_ci(prefix):
    return f"{stats[prefix + '_ci_low']:.2f}-{stats[prefix + '_ci_high']:.2f}"


s3_rows, s4_rows = [], []
for d in _S_ORDER:
    group = "candidate" if d in CANDIDATES else "positive_control"
    ta = f"treat_scanb_clinical_treatment_{d}"
    s3_rows.append({
        "Drug": d, "Group": group,
        "HR_per_SD_canonical": stats[f"hrsd_{d}"],
        "PH_p_drug_term": _fmt_p(f"ph_{d}_p_exact"),
        "HR_per_SD_PAM50_strata": stats[f"strat_{d}_hr_per_sd"],
        "CI95_PAM50_strata": _fmt_ci(f"strat_{d}"),
        "p_PAM50_strata": _fmt_p(f"strat_{d}_p"),
        "HR_per_SD_treatment_adjusted": stats[f"{ta}_hr_per_sd"],
        "CI95_treatment_adjusted": _fmt_ci(ta),
        "padj_treatment_adjusted": _fmt_p(f"{ta}_padj"),
        "HR_per_SD_TNBC": stats[f"tnbc_{d}_hr_per_sd"],
        "CI95_TNBC": _fmt_ci(f"tnbc_{d}"),
        "padj_TNBC": _fmt_p(f"tnbc_{d}_padj"),
        "HR_per_SD_Basal": stats[f"basal_{d}_hr_per_sd"],
        "CI95_Basal": _fmt_ci(f"basal_{d}"),
        "padj_Basal": _fmt_p(f"basal_{d}_padj"),
    })
    it = f"treat_scanb_chemo_interaction_{d}"
    ia = f"treat_scanb_chemo_interaction_clinical_endo_{d}"
    s4_rows.append({
        "Drug": d, "Group": group,
        "METABRIC_HR_OS": stats[f"metabric_os_{d}_hr"],
        "METABRIC_CI95_OS": _fmt_ci(f"metabric_os_{d}"),
        "METABRIC_padj_OS": _fmt_p(f"metabric_os_{d}_padj"),
        "METABRIC_HR_DSS": stats[f"dss_{d}_hr"],
        "METABRIC_padj_DSS": _fmt_p(f"dss_{d}_padj"),
        "Stouffer_padj": _fmt_p(f"stouffer_{d}_padj"),
        "SCANB_chemo_interaction_HR": stats[f"{it}_hr"],
        "SCANB_chemo_interaction_p": _fmt_p(f"{it}_p"),
        "SCANB_chemo_interaction_padj": _fmt_p(f"{it}_padj"),
        "SCANB_HR_per_SD_untreated": stats[f"{it}_hr_untreated"],
        "SCANB_HR_per_SD_treated": stats[f"{it}_hr_treated"],
        "SCANB_chemo_interaction_HR_adjusted": stats[f"{ia}_hr"],
        "SCANB_chemo_interaction_padj_adjusted": _fmt_p(f"{ia}_padj"),
        "METABRIC_HR_per_SD_treatment_adjusted": stats[f"treat_metabric_clinical_treatment_{d}_hr_per_sd"],
        "METABRIC_padj_treatment_adjusted": _fmt_p(f"treat_metabric_clinical_treatment_{d}_padj"),
    })
supp_s3 = pd.DataFrame(s3_rows)
supp_s4 = pd.DataFrame(s4_rows)
supp_s3.to_csv(OUT / "supp_table_S3_robustness.csv", index=False)
supp_s4.to_csv(OUT / "supp_table_S4_replication.csv", index=False)
print("\nSupplementary Table S3 (robustness in SCAN-B):")
print(supp_s3.to_string(index=False))
print("\nSupplementary Table S1 (METABRIC replication and chemotherapy interaction):")
print(supp_s4.to_string(index=False))


# ===================================================================
# RE-AUDIT, continued: scripts 65 and 66, the reruns of scripts 21, 37 and 40,
# and keys derived from tables that existed before the re-audit
# ===================================================================
# Registered with _add like the keys of scripts 59 to 64, so each one has to be
# pinned by a frozen-value check, and cross-checked through REAUDIT_XCHECKS. The
# keys of scripts 37 and 40 (rfs_*, clinical_adj_*) keep their section above and
# are pinned separately in the sanity block.
import ast  # noqa: E402

_COHORTS = ("SCANB", "METABRIC", "TCGA")
_surv = {c: cohort_dfs[c].set_index("drug") for c in _COHORTS}


def _list_cell(value):
    """A 'a, b, c' cell of a metric/value summary as a list ([] when empty)."""
    return [] if value is None or str(value).strip() in ("", "none") else [s.strip() for s in str(value).split(",")]


def _same(a, b):
    """Equal up to the last bit of a decimal string parsed by pandas."""
    return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0))


def _of(value):
    """'n of m' cell of a metric/value summary as the pair (n, m)."""
    n, m = str(value).split(" of ")
    return int(n), int(m)


def _module_constants(path):
    """Top-level NAME = literal assignments of a module, read with ast and never imported."""
    out = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return out


# --- script 21 rerun: Fisher meta-analysis with chi2.sf ---
# Until September 2026 script 21 took the Fisher p as 1 - chi2.cdf, which underflows
# to 0 below about 1e-16 (pardoprunox, whose fisher_p and fisher_padj were 0). With
# chi2.sf no p-value is 0, and the 551 drugs at FDR < 0.05 and the five controls do
# not change. The Fisher p of every drug is recomputed here from the three cohorts.
_fi = df_fisher.set_index("drug")
_pmat = pd.DataFrame({c: _surv[c].loc[_fi.index, "pvalue"] for c in _COHORTS})
_fisher_sf = pd.Series(_chi2.sf(-2 * np.log(_pmat).sum(axis=1), df=2 * len(_COHORTS)), index=_fi.index)
_add("fisher_n_p_equal_zero", int((_fi["fisher_p"] == 0).sum()))
_add("fisher_n_padj_equal_zero", int((_fi["fisher_padj"] == 0).sum()))
_add("fisher_min_p_drug", str(_fi["fisher_p"].idxmin()))
_add("fisher_pardoprunox_p", _p2(_fi.at["pardoprunox", "fisher_p"]))
_add("fisher_pardoprunox_padj", _p2(_fi.at["pardoprunox", "fisher_padj"]))
_xcheck("Fisher meta-analysis of script 21 = chi2.sf of the three cohort p-values for all 1,447 drugs, "
        "fisher_padj = its BH, no p-value equal to 0",
        len(_fi) == stats["SCANB_n_drugs_tested"] and int(_fi["n_cohorts"].min()) == len(_COHORTS)
        and np.allclose(_fi["fisher_p"], _fisher_sf, rtol=1e-9, atol=0)
        and np.allclose(_fi["fisher_padj"], false_discovery_control(_fi["fisher_p"]), rtol=1e-9, atol=0)
        and stats["fisher_n_p_equal_zero"] == 0,
        f"max relative difference {float(((_fi['fisher_p'] - _fisher_sf).abs() / _fisher_sf).max())}")
print(f"Fisher (script 21, chi2.sf): smallest p {stats['fisher_pardoprunox_p']} ({stats['fisher_min_p_drug']}), "
      f"padj {stats['fisher_pardoprunox_padj']}, {stats['Fisher_drugs_FDR005']} drugs at FDR<0.05")

# --- scripts 37 and 40 rerun with talazoparib (twelve drugs, BH over 12) ---
_fam = set(TWELVE_DRUGS)
_cl_tab = cl.set_index("drug")
_rfs_tab = rfs.set_index("drug")
_xcheck("Scripts 37 and 40 test the twelve drugs (talazoparib included) with BH over the twelve, and their "
        "comparison tables are the unrounded fits rounded (HR 2 decimals, padj 3 significant digits)",
        all(set(t.index) == _fam for t in (cl_fit["scanb"], cl_fit["metabric"], rfs_fit, _cl_tab, _rfs_tab))
        and all(np.allclose(t["padj"], false_discovery_control(t["pvalue"]), rtol=1e-9, atol=0)
                for t in (cl_fit["scanb"], cl_fit["metabric"], rfs_fit))
        and all(_same(round(float(cl_fit[coh].at[d, "HR_clinical"]), 2), _cl_tab.at[d, f"HR_clinical_{coh}"])
                and _same(_p3(cl_fit[coh].at[d, "padj"]), _cl_tab.at[d, f"padj_clinical_{coh}"])
                for coh in ("scanb", "metabric") for d in _fam)
        and all(_same(round(float(rfs_fit.at[d, "HR"]), 2), _rfs_tab.at[d, "HR_RFS"])
                and _same(_p3(rfs_fit.at[d, "padj"]), _rfs_tab.at[d, "padj_RFS"]) for d in _fam)
        and all(_same(round(float(_surv["SCANB"].at[d, "HR"]), 2), _cl_tab.at[d, "HR_pam50_scanb"])
                and _same(round(float(_surv["SCANB"].at[d, "HR"]), 2), _rfs_tab.at[d, "HR_OS"]) for d in _fam),
        "a table of script 37 or 40 is not the twelve-drug rerun")
_xcheck("RFS (script 40): the direction flag = sign of log HR of RFS against the PAM50-adjusted OS HR of "
        "survival_SCANB.csv",
        all((str(_rfs_tab.at[d, "same_direction"]).lower() == "yes")
            == (np.sign(np.log(rfs_fit.at[d, "HR"])) == np.sign(np.log(_surv["SCANB"].at[d, "HR"]))) for d in _fam),
        "same_direction of cox_rfs_comparison.csv disagrees with the hazard ratios")
# Inputs that no stage of pipeline.py produces. They are optional on purpose: each is
# cross-checked and copied only when present, its absence is reported with the reason,
# and no number of the manuscript is read from them.
OPTIONAL_INPUTS = {
    REAUDIT / "talazoparib_family12_summary.csv":
        "one-off check of the twelve-drug rerun of scripts 37 and 40, written by "
        "scripts/37_40_talazoparib_family_check.py from the eleven-drug outputs kept in "
        "results/v13_reaudit/_before/. Not a pipeline stage, and the rfs_* and clinical_adj_* "
        "keys come from the outputs of scripts 37 and 40 themselves",
}
# The one-off check of the rerun (scripts/37_40_talazoparib_family_check.py) is not a
# pipeline stage, so its summary is compared when present.
_fam12_path = REAUDIT / "talazoparib_family12_summary.csv"
if _fam12_path.exists():
    _f12 = pd.read_csv(_fam12_path).set_index("metric")["new_12_drugs"]

    def _semi(v):
        return [] if str(v) == "none" else str(v).split(";")

    _xcheck("Keys of scripts 37 and 40 = new_12_drugs column of the talazoparib family check",
            _semi(_f12["scanb_clin_candidates_fdr05"]) == stats["clinical_adj_scanb_candidates_fdr05"]
            and (float(_f12["scanb_clin_candidate_hr_min"]), float(_f12["scanb_clin_candidate_hr_max"]))
            == (stats["clinical_adj_scanb_hr_min"], stats["clinical_adj_scanb_hr_max"])
            and _semi(_f12["scanb_clin_controls_fdr05"]) == stats["clinical_adj_scanb_controls_fdr05"]
            and _semi(_f12["metabric_clin_candidates_fdr05"]) == stats["clinical_adj_metabric_candidates_fdr05"]
            and _semi(_f12["metabric_clin_controls_fdr05"]) == stats["clinical_adj_metabric_controls_fdr05"]
            and np.isclose(float(_f12["scanb_clin_trametinib_padj"]), cl_fit["scanb"].at["trametinib", "padj"], rtol=1e-9)
            and all(np.isclose(float(_f12[f"metabric_clin_{d}_padj"]), cl_fit["metabric"].at[d, "padj"], rtol=1e-9)
                    for d in ("brigatinib", "pelitinib"))
            and [int(_f12[k]) for k in ("rfs_n_drugs", "rfs_n_direction_concordant", "rfs_n_fdr05")]
            == [stats["rfs_n_drugs"], stats["rfs_n_direction_concordant"], stats["rfs_n_fdr05"]]
            and _semi(_f12["rfs_fdr05_drugs"]) == stats["rfs_fdr05_drugs"]
            and all(np.isclose(float(_f12[f"rfs_{d}_padj"]), rfs_fit.at[d, "padj"], rtol=1e-9)
                    for d in ("osimertinib", "saracatinib")),
            "the rerun of scripts 37 and 40 differs from the family check")
else:
    print(f"  [WARNING] Optional input {_fam12_path.name} not found, cross-check with the family "
          f"check skipped and the keys of scripts 37 and 40 checked without it: {OPTIONAL_INPUTS[_fam12_path]}")

# --- script 65: diagnostics of the Integrated Gradients attributions ---
ig_diag = _summary(REAUDIT / "ig_diagnostics_summary.csv", "ig_diagnostics_summary")
ig_sign = load_csv_required(REAUDIT / "ig_gradient_sign.csv", "ig_gradient_sign")
ig_sign["disagree"] = ig_sign["disagree"].astype(bool)
ig_by_tf = load_csv_required(REAUDIT / "ig_gradient_sign_by_tf.csv", "ig_gradient_sign_by_tf").set_index("tf")
_n_ig_drugs = int(ig_sign["drug"].nunique())
_dis_tf = ig_by_tf[ig_by_tf["n_drugs_disagree"] > 0]
_add("igdiag_n_cells", int(len(ig_sign)))
_add("igdiag_n_cells_attr_sign_ne_grad_sign", int(ig_sign["disagree"].sum()))
_add("igdiag_n_cells_disagree_negative_mean_dx", int((ig_sign["disagree"] & (ig_sign["mean_dx"] < 0)).sum()))
_add("igdiag_n_tfs_disagree", int(len(_dis_tf)))
_add("igdiag_n_tfs_disagree_all_drugs", int((_dis_tf["n_drugs_disagree"] == _n_ig_drugs).sum()))
# In the order of the IG top 20. A positive gradient means that higher activity of
# the TF raises the predicted AUC (resistance), whatever the sign of the attribution.
_add("igdiag_tfs_disagree", _dis_tf.index.tolist())
_add("igdiag_tfs_disagree_grad_positive", _dis_tf.index[_dis_tf["majority_grad_sign"] > 0].tolist())
_add("igdiag_tfs_disagree_grad_negative", _dis_tf.index[_dis_tf["majority_grad_sign"] < 0].tolist())
_add("igdiag_cross_drug_mean_r", round(float(ig_diag["cross_drug_mean_pairwise_pearson"]), 3))
_add("igdiag_cross_drug_min_r", round(float(ig_diag["cross_drug_min_pairwise_pearson"]), 3))
_add("igdiag_cross_drug_n_pairs", int(ig_diag["cross_drug_n_pairs"]))
_add("igdiag_n_ig_lines_train", int(ig_diag["n_ig_lines_train"]))
_add("igdiag_n_ig_lines_test", int(ig_diag["n_ig_lines_test"]))
_add("igdiag_n_tfs_top20_any_seed", int(ig_diag["n_tfs_top20_any_seed"]))
# Mean over the 24 (line, drug) pairs that script 65 recomputes with autograd
# (erlotinib and epirubicin on the 12 TNBC lines)
_add("igdiag_share_abs_attr_tf_dims", round(float(ig_diag["gradcheck_mean_share_abs_attr_tf_dims"]), 3))
for _m in ("SAGE", "HGT", "Ridge_expr_GNN_emb"):
    _add(f"igdiag_seed42_median_perdrug_{_m}", round(float(ig_diag[f"seed42_median_pearson_{_m}"]), 3))
# Expected top-20 overlap of two random rankings of the 771 TFs (Methods): 20 x 20 / 771
_add("igdiag_expected_top20_overlap_random", round(20 * 20 / stats["ig_multi_seed_n_tfs"], 2))

# Figure 3C matrix of script 65 = the one of script 32 the generator reads, so the
# diagnostics cannot be older than the attributions.
_sg65 = pd.read_csv(RESULTS / "v4_xai" / "tf_drug_ig_signed.csv", comment="#").set_index("tf")
_sg_long = _sg65.stack().rename("attr").reset_index().rename(columns={"level_1": "drug"})
_sg_cmp = ig_sign.merge(_sg_long, on=["tf", "drug"], how="outer")
_xcheck("IG diagnostics (script 65): the 220 cells are the Figure 3C matrix of script 32, the disagreements are "
        "the cells with negative mean x - x0, and the counts = its summary",
        len(_sg_cmp) == len(ig_sign) == _sg65.size
        and float((_sg_cmp["mean_attr"] - _sg_cmp["attr"]).abs().max()) < 1e-12
        and set(ig_by_tf.index) == set(stats["ig_top20_names"])
        and (ig_sign["attr_sign"] == np.sign(ig_sign["mean_attr"])).all()
        and (ig_sign["grad_sign"] == np.sign(ig_sign["mean_grad"])).all()
        and (ig_sign["disagree"] == (ig_sign["attr_sign"] != ig_sign["grad_sign"])).all()
        and stats["igdiag_n_cells_disagree_negative_mean_dx"] == stats["igdiag_n_cells_attr_sign_ne_grad_sign"]
        and int(_dis_tf["n_drugs_disagree"].sum()) == stats["igdiag_n_cells_attr_sign_ne_grad_sign"]
        and (_dis_tf["majority_grad_sign"] != _dis_tf["majority_attr_sign"]).all()
        and (_dis_tf["mean_dx"] < 0).all()
        and [stats["igdiag_n_cells"], stats["igdiag_n_cells_attr_sign_ne_grad_sign"],
             stats["igdiag_n_cells_disagree_negative_mean_dx"], stats["igdiag_n_tfs_disagree"],
             stats["igdiag_n_tfs_disagree_all_drugs"]]
        == [ig_diag[k] for k in ("n_fig3c_cells", "n_fig3c_cells_attr_sign_ne_grad_sign",
                                 "n_fig3c_cells_disagree_with_negative_mean_dx",
                                 "n_top20_tfs_with_any_disagreement", "n_top20_tfs_disagree_all_11_drugs")]
        and stats["igdiag_tfs_disagree"] == _list_cell(ig_diag["top20_tfs_with_disagreement"]),
        f"{stats['igdiag_n_cells_attr_sign_ne_grad_sign']} disagreements, TFs {stats['igdiag_tfs_disagree']}")
# Autograd check of script 65: its recomputed IG reproduces script 32 and its gradients
# agree in sign with the division attr / (x - x0) used for the table above.
_gchk = load_csv_required(REAUDIT / "ig_gradient_check.csv", "ig_gradient_check")
_xcheck("IG diagnostics (script 65): autograd reproduces the attributions of script 32 (relative difference "
        "< 1e-3) and the gradient signs of every recomputed (line, drug, TF)",
        float(ig_diag["gradcheck_attr_max_rel_diff"]) < 1e-3
        and _of(ig_diag["gradcheck_grad_sign_agree_all_tfs"])[0] == _of(ig_diag["gradcheck_grad_sign_agree_all_tfs"])[1]
        and _of(ig_diag["gradcheck_ZEB1_sign_agree_division_autograd"])[0]
        == _of(ig_diag["gradcheck_ZEB1_sign_agree_division_autograd"])[1] == len(_gchk)
        and np.isclose(_gchk["share_abs_attr_tf_dims"].mean(), float(ig_diag["gradcheck_mean_share_abs_attr_tf_dims"]),
                       rtol=1e-12),
        f"attribution difference {ig_diag['gradcheck_attr_max_rel_diff']}")
# Cross-drug similarity recomputed from the long-form attributions of script 32
_full_attr = pd.read_csv(RESULTS / "v4_xai" / "ig_full_attributions.csv",
                         usecols=["seed", "cell_depmap_id", "tf", "drug", "attribution"])
_vec = _full_attr.groupby(["tf", "drug"])["attribution"].mean().unstack("drug")
_cc = np.corrcoef(_vec.values.T)[np.triu_indices(_vec.shape[1], 1)]
_xcheck("IG cross-drug similarity recomputed from ig_full_attributions.csv (55 drug pairs over 771 TFs) = script 65",
        len(_cc) == stats["igdiag_cross_drug_n_pairs"] == stats["ig_n_drugs"] * (stats["ig_n_drugs"] - 1) // 2
        and _vec.shape[0] == stats["ig_multi_seed_n_tfs"] == stats["prism_n_tf_features"] == int(ig_diag["cross_drug_n_tfs"])
        and np.isclose(_cc.mean(), float(ig_diag["cross_drug_mean_pairwise_pearson"]), rtol=1e-9)
        and np.isclose(_cc.min(), float(ig_diag["cross_drug_min_pairwise_pearson"]), rtol=1e-9),
        f"mean {_cc.mean()}, min {_cc.min()}")
# Seed-42 split rebuilt as src/data/splits.cell_line_holdout_split does (sorted cell
# lines, RandomState(42).shuffle, the first 95 held out), on the 476 graph lines.
_split_cells = sorted(_graph_lines.tolist())
np.random.RandomState(42).shuffle(_split_cells)
_split_test = set(_split_cells[:stats["holdout_n_test_cells"]])
_ig_lines = set(ig_pairs["cell_depmap_id"])
_ig_test_65 = {s.split(" ")[0] for s in _list_cell(ig_diag["ig_lines_test"])}
_xcheck("IG lines in the seed-42 split: 8 train and 4 test, rebuilt from the split rule and = script 65",
        len(_split_cells) == stats["holdout_n_train_cells"] + stats["holdout_n_test_cells"]
        and _ig_lines & _split_test == _ig_test_65
        and (len(_ig_lines - _split_test), len(_ig_lines & _split_test))
        == (stats["igdiag_n_ig_lines_train"], stats["igdiag_n_ig_lines_test"])
        and stats["igdiag_n_ig_lines_train"] + stats["igdiag_n_ig_lines_test"] == stats["ig_n_cells"],
        f"test lines {sorted(_ig_lines & _split_test)} vs {sorted(_ig_test_65)}")
# Points of Figure 3a: for each TF of the seed-42 top 20, one per IG line, the
# |attribution| of the line averaged over its drugs. The bar is the mean of its points.
_ig_top20 = stats["ig_top20_names"]
_attr42 = _full_attr[(_full_attr["seed"] == 42) & _full_attr["tf"].isin(_ig_top20)].copy()
_attr42["abs_attr"] = _attr42["attribution"].abs()
_per_line = (_attr42.groupby(["tf", "cell_depmap_id"])
             .agg(n_drugs=("drug", "nunique"), abs_attr_mean=("abs_attr", "mean")).reset_index())
_ig_by_tf = df_ig.set_index("tf")
_per_line.insert(1, "rank_seed42", _per_line["tf"].map(_ig_by_tf["rank"]).astype(int))
_per_line.insert(3, "split42", np.where(_per_line["cell_depmap_id"].isin(_split_test), "test", "train"))
_per_line = _per_line.sort_values(["rank_seed42", "cell_depmap_id"]).reset_index(drop=True)
_per_line.to_csv(OUT / "tf_importance_ig_top20_per_line.csv", index=False)
# Panels a and c use the seed-42 top 20, panel b and Figure 7 the ten core TFs
_core10 = stats["ig_multi_seed_core10_names"]
_core_out = [t for t in _core10 if t not in _ig_top20]
_stab_by_tf = df_stab.set_index("tf")
_add("ig_core10_in_seed42_top20_n", len(_core10) - len(_core_out))
_add("ig_core10_outside_seed42_top20_names", _core_out)
_add("ig_core10_outside_seed42_top20_rank42", [int(_stab_by_tf.at[t, "rank_42"]) for t in _core_out])
_xcheck("Points of Figure 3a: TFs of the seed-42 top 20 in rank order x the IG lines, one row per pair, each pair "
        "from the attributions of all its drugs, no NaN and no negative value",
        len(_per_line) == len(_ig_top20) * stats["ig_n_cells"]
        and len(_attr42) == len(_per_line) * stats["ig_n_drugs"]
        and not _per_line.duplicated(["tf", "cell_depmap_id"]).any()
        and _per_line["tf"].drop_duplicates().tolist() == _ig_top20
        and (_per_line.groupby("tf")["cell_depmap_id"].nunique() == stats["ig_n_cells"]).all()
        and set(_per_line["cell_depmap_id"]) == _ig_lines
        and (_per_line["n_drugs"] == stats["ig_n_drugs"]).all()
        and _attr42["attribution"].notna().all()
        and _per_line.notna().all().all()
        and (_per_line["abs_attr_mean"] >= 0).all(),
        f"{len(_per_line)} rows, n_drugs {sorted(_per_line['n_drugs'].unique())}")
_bar_from_points = _per_line.groupby("tf")["abs_attr_mean"].mean().loc[_ig_top20].to_numpy()
_xcheck("Each bar of Figure 3a is the mean of its points: the per-line means of script 32 reproduce importance_mean "
        "of tf_importance_ig_seed42.csv (script 35, rtol 1e-12)",
        np.allclose(_bar_from_points, _ig_by_tf.loc[_ig_top20, "importance_mean"].to_numpy(), rtol=1e-12, atol=0),
        f"largest relative difference "
        f"{np.max(np.abs(_bar_from_points / _ig_by_tf.loc[_ig_top20, 'importance_mean'].to_numpy() - 1))}")
_xcheck("split42 of the per-line table of Figure 3a: 8 train and 4 test lines = the seed-42 split and script 65",
        _per_line.groupby("split42")["cell_depmap_id"].nunique().to_dict()
        == {"train": stats["igdiag_n_ig_lines_train"], "test": stats["igdiag_n_ig_lines_test"]}
        and set(_per_line.loc[_per_line["split42"] == "test", "cell_depmap_id"]) == _ig_test_65,
        f"{_per_line.groupby('split42')['cell_depmap_id'].nunique().to_dict()}")
# Share of the variance of |attribution| between the (line, drug) pairs of a TF that lies
# between lines. Not cited: it is what makes one point per line the right unit of the panel.
_attr42["tf_mean"] = _attr42.groupby("tf")["abs_attr"].transform("mean")
_attr42["line_mean"] = _attr42.groupby(["tf", "cell_depmap_id"])["abs_attr"].transform("mean")
_line_share = ((_attr42["line_mean"] - _attr42["tf_mean"]) ** 2).groupby(_attr42["tf"]).sum() \
    / ((_attr42["abs_attr"] - _attr42["tf_mean"]) ** 2).groupby(_attr42["tf"]).sum()
_xcheck("The cell line explains at least 80% of the variance of |attribution| between the pairs of each TF of the "
        "seed-42 top 20, so the points of Figure 3a stay one per line",
        len(_line_share) == len(_ig_top20) and (_line_share >= 0.80).all(),
        f"share {_line_share.min():.3f} to {_line_share.max():.3f}")
_xcheck("Core TFs against the seed-42 top 20 (Figure 3a): those outside have rank > 20 and those inside rank <= 20, "
        "with the same rank in both tables of seed 42",
        len(_core10) == 10 and len(_core_out) > 0
        and all(_ig_by_tf.at[t, "rank"] == _stab_by_tf.at[t, "rank_42"] for t in _core10)
        and all(_ig_by_tf.at[t, "rank"] > 20 for t in _core_out)
        and all(_ig_by_tf.at[t, "rank"] <= 20 for t in _core10 if t not in _core_out),
        f"{_core_out} at ranks {stats['ig_core10_outside_seed42_top20_rank42']}")
print(f"Figure 3a points: {len(_per_line)} rows, share of variance between lines {_line_share.min():.3f} to "
      f"{_line_share.max():.3f}, core TFs outside the seed-42 top 20 {_core_out} at ranks "
      f"{stats['ig_core10_outside_seed42_top20_rank42']}")
_seeds = [c.split("_", 1)[1] for c in df_stab.columns if c.startswith("rank_")]
_union20 = set().union(*(set(df_stab.nsmallest(20, f"rank_{s}")["tf"]) for s in _seeds))
_xcheck("TFs in the top 20 of at least one IG seed recomputed from the multi-seed table = script 65",
        len(_seeds) == stats["ig_n_seeds"] and len(_union20) == stats["igdiag_n_tfs_top20_any_seed"]
        and sorted(_union20) == _list_cell(ig_diag["tfs_top20_any_seed"]),
        f"{len(_union20)} TFs")
_per_drug_src = {"SAGE": RESULTS / "v2_fair" / "per_drug_SAGE_learned_emb.csv",
                 "HGT": RESULTS / "v2_fair" / "per_drug_HGT_learned_emb.csv",
                 "Ridge_expr_GNN_emb": RESULTS / "v3_embeddings" / "per_drug_Ridge(expression+GNN_emb).csv"}
_xcheck("Seed-42 per-drug medians (script 65) = the per-drug files of Table 1, and SAGE = the seed-42 full graph "
        "of the ablation (script 66)",
        all(np.isclose(pd.read_csv(p)["pearson_r"].median(), float(ig_diag[f"seed42_median_pearson_{m}"]),
                       rtol=0, atol=1e-9) for m, p in _per_drug_src.items())
        and np.isclose(float(ig_diag["seed42_median_pearson_SAGE"]),
                       float(load_csv_required(REAUDIT / "ablation_perdrug_median.csv", "ablation_perdrug_median")
                             .set_index("tag").at["full", "median_pearson_seed42"]), rtol=0, atol=1e-9),
        "seed-42 medians differ from the per-drug files")
print(f"IG diagnostics (script 65): {stats['igdiag_n_cells_attr_sign_ne_grad_sign']}/{stats['igdiag_n_cells']} "
      f"cells with attribution sign != gradient sign ({stats['igdiag_tfs_disagree']}), cross-drug r "
      f"{stats['igdiag_cross_drug_mean_r']} (min {stats['igdiag_cross_drug_min_r']}), IG lines "
      f"{stats['igdiag_n_ig_lines_train']} train / {stats['igdiag_n_ig_lines_test']} test")

# --- script 66: multiplicity and descriptive counts ---
mult = _summary(REAUDIT / "multiplicity_summary.csv", "multiplicity_summary")
# Benjamini-Yekutieli, valid under any dependence between the per-drug tests
for _c, _tag, _p in (("SCANB", "scanb", _surv["SCANB"]["pvalue"]), ("METABRIC", "metabric", _surv["METABRIC"]["pvalue"]),
                     ("TCGA", "tcga", _surv["TCGA"]["pvalue"]), ("Fisher", "fisher", _fi["fisher_p"])):
    _add(f"mult_by_{_tag}_fdr05", int((false_discovery_control(_p, method="by") < 0.05).sum()))
_s_fdr = _surv["SCANB"][_surv["SCANB"]["padj"] < 0.05]
_add("mult_scanb_fdr05_n_hr_lt1", int((_s_fdr["HR"] < 1).sum()))
_add("mult_scanb_fdr05_n_hr_gt1", int((_s_fdr["HR"] > 1).sum()))
_any_cohort = pd.concat([_surv[c].loc[_fi.index, "padj"] < 0.05 for c in _COHORTS], axis=1).any(axis=1)
_fisher_only = sorted(_fi.index[(_fi["fisher_padj"] < 0.05) & ~_any_cohort])
_add("mult_fisher_fdr05_no_single_cohort_n", len(_fisher_only))
_add("mult_fisher_fdr05_no_single_cohort_drugs", _fisher_only)
_xcheck("Multiplicity (script 66): BY counts, HR direction of the 623 and Fisher-only drugs recomputed from "
        "survival_*.csv and meta_analysis_fisher.csv = its summary, BH counts = Table 2",
        [stats[f"mult_by_{t}_fdr05"] for t in ("scanb", "metabric", "tcga", "fisher")]
        == [mult[k] for k in ("SCANB_by_fdr05", "METABRIC_by_fdr05", "TCGA_by_fdr05", "fisher_by_fdr05")]
        and [mult[k] for k in ("SCANB_bh_fdr05", "METABRIC_bh_fdr05", "TCGA_bh_fdr05", "fisher_bh_fdr05")]
        == [stats[k] for k in ("SCANB_drugs_FDR005", "METABRIC_drugs_FDR005", "TCGA_drugs_FDR005", "Fisher_drugs_FDR005")]
        and [stats["mult_scanb_fdr05_n_hr_lt1"], stats["mult_scanb_fdr05_n_hr_gt1"], 0]
        == [mult[k] for k in ("SCANB_bh_fdr05_hr_below_1", "SCANB_bh_fdr05_hr_above_1", "SCANB_bh_fdr05_hr_equal_1")]
        and stats["mult_scanb_fdr05_n_hr_lt1"] + stats["mult_scanb_fdr05_n_hr_gt1"] == stats["SCANB_drugs_FDR005"]
        and _fisher_only == _list_cell(mult["fisher_bh_fdr05_not_fdr05_in_any_cohort_drugs"])
        and len(_fisher_only) == mult["fisher_bh_fdr05_not_fdr05_in_any_cohort"]
        and (mult["fisher_script21_file_version"], mult["fisher_script21_n_p_equal_zero"]) == ("chi2.sf", 0)
        and _same(mult["fisher_min_p_chi2_sf"], f"{_fi['fisher_p'].min():.3e}"),
        "script 66 and the survival screen disagree")

# Penalizer sweep of script 45 at 0.01, 0.1 and 1.0 (0.5 has its keys in the sanity
# block). The hazard ratio range at 0.1 is the one the text prints, 0.009 to 54.4.
_psw = load_csv_required(RESULTS / "v8_multicohort" / "penalizer_sweep.csv", "penalizer_sweep")


def _pen_row(pen):
    return _psw.loc[(_psw["penalizer"] - pen).abs() < 1e-9].iloc[0]


for _pen, _tag in ((0.01, "pen001"), (0.1, "pen01"), (1.0, "pen10")):
    _add(f"pensweep_n_fdr_{_tag}", int(_pen_row(_pen)["n_fdr_all"]))
_add("pensweep_hr_min_sig_pen01", round(float(_pen_row(0.1)["hr_min_fdr"]), 4))
_add("pensweep_hr_max_sig_pen01", round(float(_pen_row(0.1)["hr_max_fdr"]), 1))
_xcheck("Penalizer sweep (script 45): 0.1 = survival_SCANB.csv (623, HR range of the significant drugs), counts "
        "= script 66",
        stats["pensweep_n_fdr_pen01"] == stats["SCANB_drugs_FDR005"]
        and np.isclose(_pen_row(0.1)["hr_min_fdr"], _s_fdr["HR"].min(), rtol=1e-9)
        and np.isclose(_pen_row(0.1)["hr_max_fdr"], _s_fdr["HR"].max(), rtol=1e-9)
        and (stats["pensweep_hr_min_sig_pen01"], stats["pensweep_hr_max_sig_pen01"])
        == (stats["hrsd_fdr_hr_per_unit_min"], stats["hrsd_fdr_hr_per_unit_max"])
        and [stats[f"pensweep_n_fdr_{t}"] for t in ("pen001", "pen01", "pen10")]
        == [mult[k] for k in ("SCANB_bh_fdr05_pen0.01", "SCANB_bh_fdr05_pen0.1", "SCANB_bh_fdr05_pen1")],
        f"{[stats[f'pensweep_n_fdr_{t}'] for t in ('pen001', 'pen01', 'pen10')]}")

# Graph ablation: mean and SD over the five seeds of the median per-drug Pearson
abl_med = load_csv_required(REAUDIT / "ablation_perdrug_median.csv", "ablation_perdrug_median").set_index("tag")
for _tag, _r in abl_med.iterrows():
    _add(f"ablation_median_perdrug_{_tag}_mean", round(float(_r["median_pearson_mean"]), 3))
    _add(f"ablation_median_perdrug_{_tag}_sd", round(float(_r["median_pearson_sd"]), 3))
_reduced = abl_med.drop(index=["full", "shuffled_drug_target"])["median_pearson_mean"]
_add("ablation_median_perdrug_reduced_min", round(float(_reduced.min()), 3))
_add("ablation_median_perdrug_reduced_max", round(float(_reduced.max()), 3))
_add("ablation_n_seeds_full_below_response_only_median",
     int(abl_med.at["full", "seeds_below_response_only_median"]))
_add("ablation_n_seeds_full_below_response_only_global",
     int(abl_med.at["full", "seeds_below_response_only_global"]))
_abl_seed = load_csv_required(RESULTS / "v6_strengthen" / "graph_ablation_per_seed.csv", "graph_ablation_per_seed")
_abl_g = _abl_seed.groupby("tag")["median_pearson"]
_abl_full = _abl_seed[_abl_seed["tag"] == "full"].set_index("init_seed")
_abl_ro = _abl_seed[_abl_seed["tag"] == "response_only"].set_index("init_seed").loc[_abl_full.index]
_xcheck("Ablation medians (script 66) = the per-seed table of script 27: mean and SD (ddof 1) over 5 seeds, full "
        "graph = Table 1 SAGE, seeds below response-only recomputed, global means = graph_ablation.csv",
        set(abl_med.index) == set(_abl_seed["tag"]) and (_abl_seed.groupby("tag").size() == 5).all()
        and np.allclose(_abl_g.mean().loc[abl_med.index], abl_med["median_pearson_mean"], rtol=0, atol=1e-6)
        and np.allclose(_abl_g.std(ddof=1).loc[abl_med.index], abl_med["median_pearson_sd"], rtol=0, atol=1e-6)
        and (stats["ablation_median_perdrug_full_mean"], stats["ablation_median_perdrug_full_sd"])
        == (stats["table1_SAGE_learned_emb_median_perdrug"], stats["table1_SAGE_learned_emb_median_perdrug_sd"])
        and int((_abl_full["median_pearson"] < _abl_ro["median_pearson"]).sum())
        == stats["ablation_n_seeds_full_below_response_only_median"]
        == _of(mult["ablation_seeds_full_below_response_only_median_perdrug"])[0]
        and int((_abl_full["test_pearson"] < _abl_ro["test_pearson"]).sum())
        == stats["ablation_n_seeds_full_below_response_only_global"]
        and all(np.isclose(abl_med.at[t, "global_pearson_mean"], r["test_pearson"], rtol=0, atol=1e-9)
                for t, r in df_abl.set_index(df_abl["config"].map(dict(zip(abl_med["config"], abl_med.index))))
                .iterrows()),
        "ablation tables disagree")
# Control that keeps the drug-target relation and rewires its edges by degree-preserving
# swaps, one rewiring per seed (SHUFFLED in src/models/training.py). Paired by seed with
# the full graph and with the graph without drug-target edges.
_abl_shuf = _abl_seed[_abl_seed["tag"] == "shuffled_drug_target"].set_index("init_seed").loc[_abl_full.index]
_abl_nodt = _abl_seed[_abl_seed["tag"] == "no_drug_target"].set_index("init_seed").loc[_abl_full.index]
_abl_kept = 100 * _abl_shuf["shuffled_edges_kept"] / _abl_shuf["shuffled_edges_total"]
_add("ablation_shuffled_global_mean", round(float(_abl_shuf["test_pearson"].mean()), 3))
_add("ablation_shuffled_global_sd", round(float(_abl_shuf["test_pearson"].std(ddof=1)), 3))
_add("ablation_shuffled_gain_kept_pct",
     round(100 * float((_abl_shuf["test_pearson"].mean() - _abl_nodt["test_pearson"].mean())
                       / (_abl_full["test_pearson"].mean() - _abl_nodt["test_pearson"].mean()))))
_add("ablation_shuffled_n_seeds_above_no_drug_target",
     int((_abl_shuf["test_pearson"] > _abl_nodt["test_pearson"]).sum()))
_add("ablation_shuffled_edges_kept_pct_min", round(float(_abl_kept.min()), 1))
_add("ablation_shuffled_edges_kept_pct_max", round(float(_abl_kept.max()), 1))
_abl_shuf_row = df_abl.set_index("config").loc["Drug-target rewired (degrees kept)"]
_xcheck("Drug-target control (script 27): five rewired runs, one per seed, over the 3,442 drug-target edges, "
        "the mean of the seeds = graph_ablation.csv, nothing excluded and only the drug-target relation rewired",
        len(_abl_shuf) == 5 and (_abl_shuf["shuffled_edges_total"] == stats["drug_target_edges"]).all()
        and np.isclose(_abl_shuf["test_pearson"].mean(), _abl_shuf_row["test_pearson"], rtol=0, atol=1e-9)
        and np.isclose(_abl_shuf["test_pearson"].std(ddof=1), _abl_shuf_row["test_pearson_sd"], rtol=0, atol=1e-9)
        and (_abl_shuf_row["excluded"], _abl_shuf_row["shuffled"]) == ("none", "targets")
        and (stats["ablation_shuffled_global_mean"], stats["ablation_shuffled_global_sd"])
        == (stats["ablation_Drug-target rewired (degrees kept)_pearson"], stats["ablation_Drug-target rewired (degrees kept)_pearson_sd"])
        and set(df_abl.loc[df_abl["config"] != "Drug-target rewired (degrees kept)", "shuffled"]) == {"none"},
        f"{len(_abl_shuf)} runs, kept {_abl_kept.round(2).tolist()}")

# METABRIC PAM50 coverage: genes of the genefu centroids present in the matrix
pam_cov = load_csv_required(REAUDIT / "metabric_pam50_coverage.csv", "metabric_pam50_coverage")
_present = pam_cov["present_by_entrez"].astype(bool)
_add("metabric_pam50_n_genes", int(len(pam_cov)))
_add("metabric_pam50_n_genes_present", int(_present.sum()))
_add("metabric_pam50_missing_genes", sorted(pam_cov.loc[~_present, "gene_current"]))
_xcheck("METABRIC PAM50 coverage (script 66): 50 distinct genes from genefu, count and missing genes = its summary, "
        "same count by current symbol",
        pam_cov["gene_genefu"].nunique() == len(pam_cov) == 50
        and str(mult["metabric_pam50_gene_list_source"]).startswith("genefu")
        and stats["metabric_pam50_n_genes_present"] == mult["metabric_pam50_present_by_entrez"]
        == mult["metabric_pam50_present_by_current_symbol"] == int(pam_cov["present_by_current_symbol"].sum())
        and stats["metabric_pam50_missing_genes"] == sorted(_list_cell(mult["metabric_pam50_missing_by_entrez"])),
        f"{stats['metabric_pam50_n_genes_present']} present")

# PRISM to GDSC cross-screen of script 05: cell lines evaluated and their PRISM edges
_cv = load_csv_required(RESULTS / "crossval_prism_to_gdsc.csv", "crossval_prism_to_gdsc")
_add("gdsc_crossval_n_cell_lines", int(mult["gdsc_eval_n_cell_lines"]))
_add("gdsc_crossval_n_cell_lines_with_prism_train_edges", _of(mult["gdsc_eval_cell_lines_with_train_edges"])[0])
_add("gdsc_crossval_n_drugs_with_prism_train_edges", _of(mult["gdsc_eval_drugs_with_train_edges"])[0])
_xcheck("GDSC cross-screen (script 66): 343 of 696 GDSC lines, 187 drugs = crossval_prism_to_gdsc.csv, pairs = the "
        "sum of its n_samples, every evaluated line and drug with PRISM training edges",
        mult["gdsc_n_cell_lines_total"] == stats["gdsc_n_cell_lines"]
        and mult["gdsc_eval_n_drugs"] == stats["crossval_n_drugs"] == len(_cv)
        and int(_cv["n_samples"].sum()) == mult["gdsc_eval_pairs"]
        == mult["gdsc_eval_pairs_prism_train_edge"] + mult["gdsc_eval_pairs_prism_test_edge"]
        + mult["gdsc_eval_pairs_not_screened_in_prism"]
        and _of(mult["gdsc_eval_cell_lines_with_train_edges"])[1] == stats["gdsc_crossval_n_cell_lines"]
        == stats["gdsc_crossval_n_cell_lines_with_prism_train_edges"]
        and _of(mult["gdsc_eval_drugs_with_train_edges"])[1] == stats["gdsc_crossval_n_drugs_with_prism_train_edges"]
        == stats["crossval_n_drugs"],
        f"{stats['gdsc_crossval_n_cell_lines']} lines")

# AURORA-US sample-level test of epirubicin (script 22), three significant digits
_aur_drug = load_csv_required(RESULTS / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",
                              "aurora_drug_primary_vs_meta").set_index("drug")
_add("aurora_drug_epirubicin_p", _p3(_aur_drug.at["epirubicin", "pvalue"]))
_xcheck("AURORA epirubicin (script 66) = aurora_drug_primary_vs_meta.csv, and its 782 nominal drugs",
        _same(mult["aurora_epirubicin_p"], round(float(_aur_drug.at["epirubicin", "pvalue"]), 4))
        and mult["aurora_drugs_nominal_p05"] == int((_aur_drug["pvalue"] < 0.05).sum()),
        f"{mult['aurora_epirubicin_p']}")
_xcheck("Clinical-adjusted olaparib (script 66) read the twelve-drug output of script 37",
        (mult["clinical_file_n_drugs"], str(mult["clinical_file_has_talazoparib"]))
        == (stats["clinical_adj_n_drugs"], "True")
        and _same(mult["clinical_olaparib_padj_scanb"], round(float(cl_fit["scanb"].at["olaparib", "padj"]), 3))
        and _same(mult["clinical_olaparib_padj_scanb"], stats["clinical_adj_scanb_olaparib_padj"]),
        f"{mult['clinical_olaparib_padj_scanb']} vs {stats['clinical_adj_scanb_olaparib_padj']}")
print(f"Multiplicity (script 66): BY {stats['mult_by_scanb_fdr05']} / {stats['mult_by_metabric_fdr05']} / "
      f"{stats['mult_by_fisher_fdr05']}, SCAN-B {stats['mult_scanb_fdr05_n_hr_lt1']} HR<1 and "
      f"{stats['mult_scanb_fdr05_n_hr_gt1']} HR>1, {stats['mult_fisher_fdr05_no_single_cohort_n']} Fisher-only, "
      f"PAM50 {stats['metabric_pam50_n_genes_present']}/{stats['metabric_pam50_n_genes']}, GDSC "
      f"{stats['gdsc_crossval_n_cell_lines']}/{stats['gdsc_n_cell_lines']} lines")

# --- Script 67: activity of the candidates in the TNBC cell lines of PRISM ---
# Keys are prefixed cltnbc_ (cell-line TNBC) so they never mix with the clinical
# TNBC subgroup of script 61, which owns the tnbc_ prefix. The analysis is an
# estimate with a declared bound: one set-level permutation test, and everything
# else descriptive, so the columns that carry multiplicity say so in their name.
cl_sum = _summary(REAUDIT / "tnbc_cell_lines_summary.csv", "tnbc_cell_lines_summary")
cl_primary = load_csv_required(REAUDIT / "tnbc_selectivity_primary.csv",
                               "tnbc_selectivity_primary").set_index("drug")
cl_bg = load_csv_required(REAUDIT / "tnbc_selectivity_background.csv",
                          "tnbc_selectivity_background").set_index("drug")
cl_set = load_csv_required(REAUDIT / "tnbc_set_test.csv", "tnbc_set_test").set_index("set_name")
cl_sens = load_csv_required(REAUDIT / "tnbc_sensitivity.csv", "tnbc_sensitivity")
cl_power = load_csv_required(REAUDIT / "tnbc_power_mde.csv", "tnbc_power_mde").set_index("drug")
cl_gate = load_csv_required(REAUDIT / "assay_sensitivity_gate.csv", "assay_sensitivity_gate")
cl_gdsc = load_csv_required(REAUDIT / "gdsc_replication.csv", "gdsc_replication")
cl_ortho = load_csv_required(REAUDIT / "tnbc_orthogonality.csv", "tnbc_orthogonality").set_index("key")
cl_excl = load_csv_required(REAUDIT / "tnbc_excluded_drugs.csv", "tnbc_excluded_drugs")
cl_universe = load_csv_required(RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv",
                                "per_drug_Ridge_expression")

_add("cltnbc_n_lines", int(cl_sum["n_tnbc_lines"]))
_add("cltnbc_n_comparison_lines", int(cl_sum["n_comparison_lines"]))
_add("cltnbc_n_breast_non_tnbc", int(cl_sum["n_breast_non_tnbc"]))
_add("cltnbc_ref_panel_n", int(cl_sum["n_ref_panel"]))
_add("cltnbc_n_evaluable", int(len(cl_bg)))
_add("cltnbc_n_excluded", int(len(cl_excl)))
_add("cltnbc_pct_censored_observed", float(cl_sum["pct_censored_observed"]))
_add("cltnbc_n_not_resolvable", int(cl_bg["not_resolvable"].sum()))
_add("cltnbc_missing_differential_n_padj05", int((cl_bg["missing_fisher_padj"] < 0.05).sum()))
_add("cltnbc_background_n_padj05", int((cl_bg["padj_bh_background_descriptive"] < 0.05).sum()))
_add("cltnbc_background_n_p05", int((cl_bg["p_perm"] < 0.05).sum()))
_add("cltnbc_background_pct_p05", _pct(stats["cltnbc_background_n_p05"], len(cl_bg)))
_add("cltnbc_storey_pi0_lambda05", float(cl_sum["storey_pi0_lambda05"]))
_add("cltnbc_storey_pi0_above_one", bool(cl_sum["storey_pi0_above_one_at_every_lambda"]))
_add("cltnbc_background_pct_delta_negative", float(cl_sum["background_pct_delta_negative"]))
_add("cltnbc_ref_cov_min", int(cl_sum["ref_cov_min"]))
_add("cltnbc_gdsc_lines_dropped", int(cl_sum["gdsc_lines_dropped_low_coverage"]))
_add("cltnbc_plan_sha256", str(cl_sum["plan_sha256"])[:12])
_add("cltnbc_variance_inflation", float(cl_sum["variance_inflation"]))
_add("cltnbc_set_delta", round(float(cl_set.at["candidates7", "cliffs_delta"]), 3))
_add("cltnbc_set_ci_low", round(float(cl_set.at["candidates7", "ci_low"]), 3))
_add("cltnbc_set_ci_high", round(float(cl_set.at["candidates7", "ci_high"]), 3))
# Permutation p-values are registered one decimal wide: their Monte Carlo error is
# about 0.005, so a finer value would pin the random stream and not the result.
_add("cltnbc_set_p", round(float(cl_set.at["candidates7", "p_perm"]), 1))
_add("cltnbc_set_mc_se", round(float(cl_set.at["candidates7", "mc_se"]), 3))
_add("cltnbc_set_n_perm", int(cl_set.at["candidates7", "n_perm"]))
_add("cltnbc_controls_set_delta", round(float(cl_set.at["controls5", "cliffs_delta"]), 3))
_add("cltnbc_controls_set_p", round(float(cl_set.at["controls5", "p_perm"]), 1))
_add("cltnbc_moa_pool_n", int(cl_sum["moa_pool_size"]))
_add("cltnbc_moa_p", round(float(cl_sum["moa_p_two_sided"]), 1))
_add("cltnbc_moa_observed_delta", round(float(cl_sum["moa_observed_mean_delta"]), 3))
_add("cltnbc_moa_null_mean", round(float(cl_sum["moa_null_mean"]), 3))
_add("cltnbc_moa_null_sd", round(float(cl_sum["moa_null_sd"]), 3))
_add("cltnbc_candidate_mean_percentile", round(float(cl_bg.loc[CANDIDATES, "background_percentile"].mean()), 1))
_add("cltnbc_n_candidates_expected_direction", int((cl_bg.loc[CANDIDATES, "cliffs_delta"] < 0).sum()))
_add("cltnbc_mde_delta_80", round(float(cl_sum["mde_delta_80"]), 2))
_add("cltnbc_mde_delta_80_3dp", round(float(cl_sum["mde_delta_80"]), 3))
_add("cltnbc_max_abs_delta_panel", round(float(cl_bg["cliffs_delta"].abs().max()), 2))
_add("cltnbc_n_panel_above_mde", int((cl_bg["cliffs_delta"].abs() >= float(cl_sum["mde_delta_80"])).sum()))
_xcheck("TNBC lines (script 67): the 6 drugs at the MDE are also the count at the printed 0.473",
        int((cl_bg["cliffs_delta"].abs() >= stats["cltnbc_mde_delta_80_3dp"]).sum()) == stats["cltnbc_n_panel_above_mde"],
        f"{stats['cltnbc_n_panel_above_mde']} at the MDE")
_add("cltnbc_pct_panel_above_mde", float(cl_sum["pct_panel_above_mde"]))
# The contrast of the primary test mixes the TNBC effect with a lineage effect, and
# the two screens differ in that term, so the three contrasts of paclitaxel are keyed
# separately instead of one number that folds them together.
_gate_pac = cl_gate[cl_gate["control"] == "paclitaxel"].set_index(["screen", "contrast"])
for _screen in ("PRISM", "GDSC"):
    for _contrast, _tag in (("tnbc_vs_other_breast", "within_breast"),
                            ("breast_vs_non_breast", "lineage"),
                            ("tnbc_vs_all_other_lines", "vs_all")):
        _add(f"cltnbc_gate_paclitaxel_{_screen.lower()}_{_tag}",
             round(float(_gate_pac.at[(_screen, _contrast), "cliffs_delta"]), 2))
_add("cltnbc_calibration_detect_rate", float(cl_sum["gate_calibration_detect_rate"]))
_cal = cl_gate[(cl_gate["control"] == "dabrafenib") &
               (cl_gate["contrast"] == "skin12_vs_rest_draws")]
_add("cltnbc_calibration_delta_dabrafenib", round(float(_cal["cliffs_delta"].iloc[0]), 2))
_add("cltnbc_power_ref_paclitaxel_within_breast",
     round(float(cl_sum["power_ref_paclitaxel_gdsc_within_breast"]), 2))
_add("cltnbc_power_at_ref_040_min",
     round(float(cl_power.loc[CANDIDATES, "power_at_ref_040"].min()), 2))
_add("cltnbc_power_at_ref_040_max",
     round(float(cl_power.loc[CANDIDATES, "power_at_ref_040"].max()), 2))
_add("cltnbc_loo_swing_max",
     round(max(float(cl_sum[f"{d}_loo_swing"]) for d in TWELVE_DRUGS), 2))
# The six lineage rows (all breast lines against the rest) have no expected direction,
# so script 67 writes them without a verdict and both counts of the gate leave them out
_gate_pass = cl_gate["passed"].eq(True)
_add("cltnbc_gate_n_passed", int(_gate_pass[cl_gate["passed"].notna()].sum()))
_add("cltnbc_gate_n_total", int(cl_gate["passed"].notna().sum()))
# The checks with an expected direction, chemotherapy, PARP and EGFR, without the skin
# calibration control. Each passing one is named as the tags of the summary of 67.
_GATE_DIRECTIONAL_CONTRASTS = ("tnbc_vs_other_breast", "tnbc_vs_all_other_lines",
                               "brca1_mutant_vs_other_tnbc", "egfr_driven_vs_other_tnbc")
_gate_dir = cl_gate[cl_gate["contrast"].isin(_GATE_DIRECTIONAL_CONTRASTS)]
_add("cltnbc_gate_n_directional_total", int(len(_gate_dir)))
_add("cltnbc_gate_n_directional_passed", int(_gate_pass[_gate_dir.index].sum()))
_add("cltnbc_gate_directional_passed",
     [f"{r.control}_{r.screen.lower()}_{r.contrast}" for r in _gate_dir[_gate_pass[_gate_dir.index]].itertuples()])
_add("cltnbc_gdsc_tnbc_lines", int(cl_sum["gdsc_tnbc_lines"]))
_add("cltnbc_gdsc_tnbc_shared_with_prism", int(cl_sum["gdsc_tnbc_shared_with_prism"]))
_add("cltnbc_gdsc_tnbc_disjoint", int(cl_sum["gdsc_tnbc_disjoint"]))
_add("cltnbc_gdsc_focal_missing", str(cl_sum["gdsc_focal_missing"]))
_add("cltnbc_gdsc_sign_agreement_pct",
     _pct(int(cl_gdsc.loc[cl_gdsc["arm"] == "technical_overlap", "sign_agree"].sum()),
          int(cl_gdsc.loc[cl_gdsc["arm"] == "technical_overlap", "sign_agree"].notna().sum())))
_add("cltnbc_ortho_rho", round(float(cl_ortho.at["delta_vs_scanb_loghr", "spearman_rho"]), 2))
_add("cltnbc_ortho_p", _p2(cl_ortho.at["delta_vs_scanb_loghr", "p"]))
_add("cltnbc_iclac_among_tnbc", int(cl_sum["iclac_n_among_tnbc"]))
_cl_grid = cl_sens[~cl_sens["arm"].isin(["S6"])]
for d in TWELVE_DRUGS:
    _add(f"{_role(d)}_{d}_cltnbc_delta", round(float(cl_primary.at[d, "cliffs_delta"]), 2))
    _add(f"{_role(d)}_{d}_cltnbc_p", round(float(cl_primary.at[d, "p_perm"]), 1))
    _add(f"{_role(d)}_{d}_cltnbc_ci_low", round(float(cl_primary.at[d, "delta_ci_low"]), 2))
    _add(f"{_role(d)}_{d}_cltnbc_ci_high", round(float(cl_primary.at[d, "delta_ci_high"]), 2))
    _add(f"{_role(d)}_{d}_cltnbc_percentile", round(float(cl_primary.at[d, "background_percentile"]), 1))
    _part = _cl_grid[_cl_grid["drug"] == d]["cliffs_delta"]
    _add(f"{_role(d)}_{d}_cltnbc_sens_min", round(float(_part.min()), 2))
    _add(f"{_role(d)}_{d}_cltnbc_sens_max", round(float(_part.max()), 2))

_xcheck("Cell-line TNBC (script 67): the twelve focal drugs are the same rows in the background table, "
        "and the evaluable plus the excluded drugs are the whole screen, all inside the 1,447 universe",
        all(np.isclose(cl_primary.at[d, "cliffs_delta"], cl_bg.at[d, "cliffs_delta"], rtol=0, atol=1e-12)
            for d in TWELVE_DRUGS)
        and len(cl_bg) + len(cl_excl) == 1448
        and set(cl_bg.index) <= set(cl_universe["drug"].str.lower())
        and stats["cltnbc_n_evaluable"] == int(cl_sum["n_evaluable_drugs"])
        and stats["cltnbc_n_excluded"] == int(cl_sum["n_excluded_drugs"]),
        f"{len(cl_bg)} evaluable and {len(cl_excl)} excluded")
_xcheck("Cell-line TNBC (script 67): the 12 TNBC lines and the 10 other breast lines of the screen "
        "match the subtype counts of Supplementary Figure 3",
        stats["cltnbc_n_lines"] == stats["breast_n_tnbc"] == 12
        and stats["cltnbc_n_breast_non_tnbc"] == stats["breast_n_erpos"] + stats["breast_n_her2pos"]
        and stats["cltnbc_n_lines"] + stats["cltnbc_n_comparison_lines"] == 476
        and stats["cltnbc_iclac_among_tnbc"] == 0,
        f"{stats['cltnbc_n_lines']} TNBC and {stats['cltnbc_n_breast_non_tnbc']} other breast lines")
_xcheck("Cell-line TNBC (script 67): the set-level test and the per-drug values of the summary CSV "
        "reproduce the per-drug tables, and the GDSC overlap matches script 66",
        _same(float(cl_sum["set_candidates7_delta"]), round(float(cl_set.at["candidates7", "cliffs_delta"]), 4))
        and _same(round(float(cl_sum["candidate_mean_percentile"]), 1),
                  stats["cltnbc_candidate_mean_percentile"])
        and int(cl_sum["n_candidates_expected_direction"]) == stats["cltnbc_n_candidates_expected_direction"]
        and int(cl_sum["gdsc_shared_lines"]) == stats["gdsc_crossval_n_cell_lines"]
        and int(cl_sum["gdsc_shared_drugs"]) == stats["crossval_n_drugs"],
        f"set delta {stats['cltnbc_set_delta']}, percentile {stats['cltnbc_candidate_mean_percentile']}")
# The assay gate of script 67 holds three kinds of rows: the checks with an expected
# direction (the three chemotherapies in two contrasts and two screens, PARP and EGFR
# inside the panel), the lineage term (no direction, so no verdict) and the skin
# calibration control. The design is spelled out here and the file must match it.
_gate_lineage = cl_gate[cl_gate["contrast"] == "breast_vs_non_breast"]
_gate_cal = cl_gate[cl_gate["contrast"].isin(["skin_vs_rest_all", "skin12_vs_rest_draws"])]
# The calibration rows with all the skin lines: a lineage effect beyond doubt, so their
# verdict is the same rule as the directional checks, the interval excluding zero below
_gate_skin_all = cl_gate[cl_gate["contrast"] == "skin_vs_rest_all"]
_gate_scored = cl_gate["passed"].notna()
_gate_has_delta = cl_gate["cliffs_delta"].notna()
_gate_dir_design = (
    {(d, s, c) for d in ("paclitaxel", "docetaxel", "epirubicin") for s in ("PRISM", "GDSC")
     for c in ("tnbc_vs_other_breast", "tnbc_vs_all_other_lines")}
    | {(d, "PRISM", "brca1_mutant_vs_other_tnbc") for d in ("olaparib", "talazoparib")}
    | {(d, "PRISM", "egfr_driven_vs_other_tnbc") for d in ("erlotinib", "osimertinib", "pelitinib")})
_xcheck("Cell-line TNBC (script 67): the assay gate is the declared checks with an expected direction "
        "(chemotherapy, PARP and EGFR), the lineage rows written without a verdict and the skin calibration "
        "control, each directional verdict and each verdict of the three calibration rows with all the skin "
        "lines is its interval excluding zero below, every sign column is the sign of its delta, and the "
        "three counts that leave the lineage rows out are the ones the summary of 67 writes",
        len(_gate_dir) + len(_gate_lineage) + len(_gate_cal) == len(cl_gate)
        and set(zip(_gate_dir["control"], _gate_dir["screen"], _gate_dir["contrast"])) == _gate_dir_design
        and len(_gate_dir) == len(_gate_dir_design) == stats["cltnbc_gate_n_directional_total"]
        and int((cl_gate["criterion"] == "lineage term, no direction expected").sum()) == len(_gate_lineage) == 6
        and (_gate_lineage["criterion"] == "lineage term, no direction expected").all()
        and _gate_lineage["passed"].isna().all()
        and cl_gate.drop(index=_gate_lineage.index)["passed"].map(lambda v: isinstance(v, (bool, np.bool_))).all()
        and all(bool(r.passed) == bool(r.ci_high < 0) for r in _gate_dir.itertuples())
        and all(r.cliffs_delta < 0 for r in _gate_dir.itertuples() if bool(r.passed))
        and sorted(_gate_skin_all["control"]) == ["dabrafenib", "trametinib", "vemurafenib"]
        and (_gate_skin_all["screen"] == "PRISM").all()
        and (_gate_skin_all["criterion"] == "delta < 0, a lineage effect that is beyond doubt").all()
        and all(bool(r.passed) == bool(r.ci_high < 0) for r in _gate_skin_all.itertuples())
        and _gate_has_delta.all()
        and all(bool(r.sign_negative) == bool(r.cliffs_delta < 0) for r in cl_gate.itertuples())
        and stats["cltnbc_gate_n_directional_passed"] == len(stats["cltnbc_gate_directional_passed"])
        and stats["cltnbc_gate_n_total"] == len(cl_gate) - len(_gate_lineage) == int(cl_sum["gate_n_total"])
        and stats["cltnbc_gate_n_passed"] == int(cl_sum["gate_n_passed"])
        == stats["cltnbc_gate_n_directional_passed"] + int(_gate_pass[_gate_cal.index].sum())
        and int(cl_sum["gate_n_sign_negative"]) == int(cl_gate.loc[_gate_scored, "sign_negative"].astype(bool).sum()),
        f"{stats['cltnbc_gate_n_directional_passed']} of {stats['cltnbc_gate_n_directional_total']} directional "
        f"({stats['cltnbc_gate_directional_passed']}), skin calibration with all lines "
        f"{dict(zip(_gate_skin_all['control'], _gate_skin_all['passed']))}, {stats['cltnbc_gate_n_passed']} of "
        f"{stats['cltnbc_gate_n_total']} with a verdict, {int(cl_sum['gate_n_sign_negative'])} of them with a "
        f"negative sign, {len(_gate_lineage)} lineage rows")
# Design constants of script 67 that Methods cites, read from the script with ast so
# the text cannot drift from the code, each checked against what the run wrote. The
# calibration rows give one detection rate per drug, exact in their detection_rate
# column, which is what the verdict compares, and rounded to two decimals in their note.
_c67 = _module_constants(REPO / "scripts" / "67_tnbc_cell_lines.py")
_cal_rows = cl_gate[cl_gate["contrast"] == "skin12_vs_rest_draws"]
_cal_rates = [float(r) for r in _cal_rows["detection_rate"]]
_cal_threshold = float(_c67["CALIBRATION_MIN_DETECT_RATE"])
_cal_min_verdict = bool(_cal_rows["passed"].iloc[int(np.argmin(_cal_rates))])
# Each exact rate is a count of detections over the draws, its note is that rate to two
# decimals, and no other row of the gate carries a rate
_cal_n_draws = int(_c67["N_CALIBRATION_DRAWS"])
_cal_rates_exact = (
    len(_cal_rates) == 3
    and all(np.isclose(r * _cal_n_draws, round(r * _cal_n_draws), rtol=0, atol=1e-9) for r in _cal_rates)
    and [n for n in _cal_rows["note"]] == [f"detection rate {r:.2f}" for r in _cal_rates]
    and cl_gate.drop(index=_cal_rows.index)["detection_rate"].isna().all())
_xcheck("Cell-line TNBC (script 67): nothing reaches FDR in the background, the panel maximum stays below "
        "the minimum detectable effect except for a handful of drugs, both screens agree on the "
        "within-breast taxane effect, and the lowest exact detection rate of the skin calibration control "
        f"is the one script 67 writes, with a written verdict that agrees with its threshold of {_cal_threshold} "
        "(the check does not require it to pass)",
        stats["cltnbc_background_n_padj05"] == 0
        and stats["cltnbc_n_panel_above_mde"] <= 10
        and stats["cltnbc_gate_paclitaxel_prism_within_breast"] < 0
        and stats["cltnbc_gate_paclitaxel_gdsc_within_breast"] < 0
        and _cal_rates_exact
        and np.isclose(min(_cal_rates), stats["cltnbc_calibration_detect_rate"], rtol=0, atol=1e-12)
        and _cal_min_verdict == (min(_cal_rates) >= _cal_threshold)
        and float(cl_power["mde_delta_80"].min()) >= 0.4,
        f"{stats['cltnbc_n_panel_above_mde']} drugs above the MDE, paclitaxel within breast "
        f"{stats['cltnbc_gate_paclitaxel_prism_within_breast']} in PRISM and "
        f"{stats['cltnbc_gate_paclitaxel_gdsc_within_breast']} in GDSC, lowest calibration detection "
        f"{stats['cltnbc_calibration_detect_rate']} against {_cal_threshold}, written verdict {_cal_min_verdict}")
_add("cltnbc_min_tnbc_lines", int(_c67["MIN_TNBC"]))
_add("cltnbc_min_comparison_lines", int(_c67["MIN_REST"]))
_add("cltnbc_calibration_n_draws", int(_c67["N_CALIBRATION_DRAWS"]))
_add("cltnbc_calibration_min_rate", float(_c67["CALIBRATION_MIN_DETECT_RATE"]))
_add("cltnbc_moa_n_sets", int(_c67["N_MOA_NULL"]))
_cl_excl_idx = cl_excl.set_index("drug")
_xcheck("Cell-line TNBC (script 67): every evaluable drug is measured in at least the minimum of TNBC and "
        "comparison lines of the script, and every excluded drug falls short of one of them for the reason "
        "it is written out with",
        int(cl_bg["n_tnbc"].min()) >= stats["cltnbc_min_tnbc_lines"]
        and int(cl_bg["n_rest"].min()) >= stats["cltnbc_min_comparison_lines"]
        and all((r["n_tnbc"] < stats["cltnbc_min_tnbc_lines"]
                 and r["reason"] == f"fewer than {stats['cltnbc_min_tnbc_lines']} TNBC lines measured")
                or (r["n_tnbc"] >= stats["cltnbc_min_tnbc_lines"]
                    and r["n_rest"] < stats["cltnbc_min_comparison_lines"]
                    and r["reason"] == f"fewer than {stats['cltnbc_min_comparison_lines']} comparison lines measured")
                for _, r in _cl_excl_idx.iterrows()),
        f"evaluable minimum {int(cl_bg['n_tnbc'].min())} TNBC and {int(cl_bg['n_rest'].min())} comparison lines")
# The calibration rows: the criterion names the number of draws, the verdict is the
# detection rate against the threshold, and the script compares with the constant
_m67 = ast.parse((REPO / "scripts" / "67_tnbc_cell_lines.py").read_text(encoding="utf-8"))
_cal_cmp = [n for n in ast.walk(_m67)
            if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "passed" for k in n.keys)
            and any(isinstance(k, ast.Constant) and k.value == "contrast" for k in n.keys)]
_cal_uses_constant = [v for d in _cal_cmp for k, v in zip(d.keys, d.values)
                      if isinstance(k, ast.Constant) and k.value == "passed" and isinstance(v, ast.Compare)
                      and any(isinstance(c, ast.Name) and c.id == "CALIBRATION_MIN_DETECT_RATE" for c in v.comparators)]
# The number of draws, from the code and not from the label 67 writes with the same
# constant: the loop over range(N_CALIBRATION_DRAWS) is the only one that fills a list,
# nothing else appends to that list, and the rate the verdict compares is computed from it
_cal_loops = [n for n in ast.walk(_m67)
              if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and getattr(n.iter.func, "id", "") == "range"
              and len(n.iter.args) == 1 and isinstance(n.iter.args[0], ast.Name)
              and n.iter.args[0].id == "N_CALIBRATION_DRAWS"]


def _appends(tree):
    """Names of the lists that a tree appends to, one entry per call."""
    return [c.func.value.id for c in ast.walk(tree)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "append"
            and isinstance(c.func.value, ast.Name)]


_cal_lists = sorted(set(_appends(_cal_loops[0]))) if len(_cal_loops) == 1 else []
_cal_rate_name = (_cal_uses_constant[0].left.id
                  if len(_cal_uses_constant) == 1 and isinstance(_cal_uses_constant[0].left, ast.Name) else None)
_cal_rate_from_list = [n for n in ast.walk(_m67)
                       if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == _cal_rate_name and len(_cal_lists) == 1
                       and any(isinstance(m, ast.Name) and m.id == _cal_lists[0] for m in ast.walk(n.value))]
# The detection_rate column is written, in the same row dict, from the variable the
# verdict compares, so the exact rate of the file is the one that decided
_cal_rate_column = [v for d in _cal_cmp
                    if any(v2 is u for v2 in _cal_uses_constant for k2, u in zip(d.keys, d.values))
                    for k, v in zip(d.keys, d.values)
                    if isinstance(k, ast.Constant) and k.value == "detection_rate"]
_xcheck("Cell-line TNBC (script 67): the three calibration controls were drawn as many times as the script says "
        "(its only loop over range(N_CALIBRATION_DRAWS) fills the list the detection rate is computed from, read "
        "with ast), each verdict is its exact detection rate against the threshold of the script, the script "
        "decides with that constant and writes the rate it compared in the detection_rate column, and the "
        "lowest exact rate is the cltnbc_calibration_detect_rate key",
        len(_cal_rows) == 3
        and all(c == f"detection rate over {stats['cltnbc_calibration_n_draws']} random draws of twelve skin lines"
                for c in _cal_rows["criterion"])
        and len(_cal_loops) == 1 and len(_cal_lists) == 1
        and _appends(_m67).count(_cal_lists[0]) == _appends(_cal_loops[0]).count(_cal_lists[0])
        and len(_cal_rate_from_list) == 1
        and _cal_rates_exact
        and [bool(p) for p in _cal_rows["passed"]] == [r >= stats["cltnbc_calibration_min_rate"] for r in _cal_rates]
        and len(_cal_uses_constant) == 1
        and len(_cal_rate_column) == 1 and isinstance(_cal_rate_column[0], ast.Name)
        and _cal_rate_column[0].id == _cal_rate_name
        and np.isclose(min(_cal_rates), stats["cltnbc_calibration_detect_rate"], rtol=0, atol=1e-12),
        f"exact rates {_cal_rates}, verdicts {list(_cal_rows['passed'])}, draw loops {len(_cal_loops)}, "
        f"lists {_cal_lists}, rate {_cal_rate_name}, column from "
        f"{[getattr(v, 'id', type(v).__name__) for v in _cal_rate_column]}")
_cl_moa_null = load_csv_required(REAUDIT / "tnbc_moa_matched_null.csv", "tnbc_moa_matched_null")
_xcheck("Cell-line TNBC (script 67): the mechanism-matched null holds as many sets as the constant of the "
        "script, which is the count its summary reports",
        len(_cl_moa_null) == _cl_moa_null["replicate"].nunique() == stats["cltnbc_moa_n_sets"]
        == int(cl_sum["n_moa_null"]),
        f"{len(_cl_moa_null)} sets against {stats['cltnbc_moa_n_sets']}")
print(f"Cell-line TNBC (script 67): set delta {stats['cltnbc_set_delta']} "
      f"[{stats['cltnbc_set_ci_low']}, {stats['cltnbc_set_ci_high']}], p {stats['cltnbc_set_p']}, "
      f"candidates at percentile {stats['cltnbc_candidate_mean_percentile']}, "
      f"{stats['cltnbc_background_n_padj05']} of {stats['cltnbc_n_evaluable']} drugs at FDR < 0.05, "
      f"MDE {stats['cltnbc_mde_delta_80']}")

# --- Script 69: does the ordering of the associations hold in a cohort that is all TNBC ---
# FUSCC-TNBC, 356 patients and 49 relapses. Keys are prefixed fuscc_. The analysis has a
# single decision, the permutation p of the correlation between the per-drug score z of
# FUSCC and of the SCAN-B TNBC subgroup, and everything else says descriptive in its name.
fu_sum = _summary(FUSCC / "fuscc_summary.csv", "fuscc_summary")
fu_corr = load_csv_required(FUSCC / "fuscc_primary_correlation.csv", "fuscc_primary_correlation")
fu_perdrug = load_csv_required(FUSCC / "fuscc_cox_perdrug.csv", "fuscc_cox_perdrug").set_index("drug")
fu_comp = load_csv_required(FUSCC / "fuscc_composite.csv", "fuscc_composite").set_index("set")
fu_gate = load_csv_required(FUSCC / "fuscc_cohort_gate.csv", "fuscc_cohort_gate").set_index("term")
fu_power = load_csv_required(FUSCC / "fuscc_power_mde.csv", "fuscc_power_mde")
# The exclusions are read as counts by reason, never as the per-patient file
fu_excl = load_csv_required(FUSCC / "fuscc_excluded_summary.csv", "fuscc_excluded_summary").set_index("reason")
fu_ph = load_csv_required(FUSCC / "fuscc_ph_tests.csv", "fuscc_ph_tests")
fu_sens = load_csv_required(FUSCC / "fuscc_sensitivity.csv", "fuscc_sensitivity")
_fu_primary = fu_corr[fu_corr["is_primary"]].iloc[0]

_add("fuscc_n_sequenced", int(fu_sum["n_sequenced"]))
_add("fuscc_n_patients", int(fu_sum["n_patients"]))
_add("fuscc_n_events", int(fu_sum["n_events"]))
_add("fuscc_n_excluded", int(fu_excl["n_patients"].sum()))
_add("fuscc_n_excluded_zero_time", int(fu_sum["n_excluded_zero_time"]))
_add("fuscc_n_excluded_predeclared", int(fu_sum["n_excluded_predeclared"]))
_add("fuscc_median_followup_months", float(fu_sum["median_followup_months_reverse_km"]))
_add("fuscc_n_tfs_shared_prism", int(fu_sum["n_tfs_shared_prism"]))
_add("fuscc_qt_clipped_pct", float(fu_sum["qt_clipped_fraction_pct"]))
_add("fuscc_qt_clipped_pct_scanb", float(fu_sum["qt_clipped_fraction_pct_scanb"]))
_add("fuscc_scanb_tnbc_n", int(fu_sum["scanb_tnbc_n"]))
_add("fuscc_scanb_tnbc_events", int(fu_sum["scanb_tnbc_events"]))
_add("fuscc_scanb_tnbc_pct_z_positive", float(fu_sum["scanb_tnbc_pct_z_positive_1447"]))
# The permutation p is registered two decimals wide, above its Monte Carlo error
_add("fuscc_r_primary", round(float(_fu_primary["r_pearson"]), 2))
_add("fuscc_r_primary_p_perm", round(float(_fu_primary["p_perm"]), 2))
_add("fuscc_r_primary_mc_se", round(float(_fu_primary["mc_se"]), 3))
_add("fuscc_r_primary_null_p95", round(float(_fu_primary["null_p95"]), 2))
_add("fuscc_effective_df", round(float(_fu_primary["effective_df"]), 1))
_add("fuscc_var_z_reference", float(fu_sum["var_z_reference_os"]))
_add("fuscc_var_z_fuscc", float(fu_sum["var_z_fuscc"]))
_add("fuscc_rel_reference", float(fu_sum["rel_reference_os"]))
_add("fuscc_ceiling_estimable", str(fu_sum["ceiling_attenuation"]) != "not estimable")
for _set, _key in (("filter143_descriptive", "143"), ("focal12_descriptive", "focal12")):
    _row = fu_corr[(fu_corr["drug_set"] == _set) & (fu_corr["arm"] == "primary")].iloc[0]
    _add(f"fuscc_r_{_key}_descriptive", round(float(_row["r_pearson"]), 2))
    _add(f"fuscc_r_{_key}_p_perm_descriptive", round(float(_row["p_perm"]), 2))
_add("fuscc_r_rfs_reference", round(float(fu_sum["r_rfs_reference"]), 2))
_add("fuscc_r_no_lar", round(float(fu_sum["r_no_lar"]), 2))
_add("fuscc_sign_agreement_pct_1447", float(fu_sum["sign_agreement_pct_1447"]))
_add("fuscc_sign_agreement_pct_focal12", float(fu_sum["sign_agreement_pct_focal12"]))
_add("fuscc_n_drugs_fdr05", int(fu_sum["n_drugs_fdr05_background"]))
_add("fuscc_n_drugs_tested", int(len(fu_perdrug)))
_add("fuscc_mde_hr_per_sd", round(float(fu_sum["mde_hr_per_sd_80"]), 2))
_add("fuscc_power_at_pelitinib", float(fu_sum["power_at_pelitinib_rescaled"]))
_add("fuscc_power_at_osimertinib", float(fu_sum["power_at_osimertinib_rescaled"]))
_add("fuscc_gate_nodes_hr", round(float(fu_gate.at["positive lymph nodes", "hr"]), 2))
_add("fuscc_gate_nodes_passed", bool(fu_gate.at["positive lymph nodes", "verdict"]))
_add("fuscc_composite_hr_per_sd", round(float(fu_comp.at["candidates7", "hr_per_sd"]), 2))
_add("fuscc_composite_p_perm", round(float(fu_comp.at["candidates7", "p_perm"]), 2))
_add("fuscc_composite_pct_background_descriptive",
     float(fu_comp.at["candidates7", "pct_background_matched_descriptive"]))
for d in TWELVE_DRUGS:
    _add(f"{_role(d)}_{d}_fuscc_hr_per_sd", round(float(fu_perdrug.at[d, "hr_per_sd_fuscc"]), 2))
    _add(f"{_role(d)}_{d}_fuscc_ci_low", round(float(fu_perdrug.at[d, "ci_low"]), 2))
    _add(f"{_role(d)}_{d}_fuscc_ci_high", round(float(fu_perdrug.at[d, "ci_high"]), 2))
    _add(f"{_role(d)}_{d}_fuscc_z", round(float(fu_perdrug.at[d, "score_z"]), 2))
    _add(f"{_role(d)}_{d}_fuscc_reference_z", round(float(fu_perdrug.at[d, "scanb_tnbc_score_z_os"]), 2))

# Reference effects of the power analysis (pelitinib and osimertinib in the SCAN-B TNBC
# subgroup, rescaled to the spread inside it) and the gate: nodes missing in FUSCC are
# counted as negative, and the complete cases are a sensitivity that does not decide
_fu_ref = fu_power[fu_power["statistic"].str.startswith("per drug")].set_index("reference_source")
_add("fuscc_reference_hr_pelitinib", _r2(_fu_ref.at["pelitinib_rescaled", "reference_effect"]))
_add("fuscc_reference_hr_osimertinib", _r2(_fu_ref.at["osimertinib_rescaled", "reference_effect"]))
_fu_cc = fu_gate.loc["positive lymph nodes, complete cases"]
_add("fuscc_gate_nodes_n_missing", int(fu_sum["gate_nodes_n_missing_counted_negative"]))
_add("fuscc_gate_nodes_hr_complete_cases", _r2(_fu_cc["hr"]))
_add("fuscc_gate_nodes_ci_low_complete_cases", _r2(_fu_cc["ci_low"]))
_add("fuscc_gate_nodes_ci_high_complete_cases", _r2(_fu_cc["ci_high"]))
# Sex of the analysed patients (Table S1 records it, script 00c copies it), the number
# of permutations of the primary, its p with three decimals next to the two-decimal key,
# and the range the cohort gate decides with, all written by script 69
_c69 = _module_constants(REPO / "scripts" / "69_fuscc_survival.py")
_add("fuscc_n_female", int(fu_sum["n_female"]))
_add("fuscc_n_perm", int(fu_sum["n_perm"]))
_add("fuscc_r_primary_p_perm_3dp", round(float(_fu_primary["p_perm"]), 3))
_add("fuscc_gate_nodes_range_low", float(fu_sum["gate_nodes_range_low"]))
_add("fuscc_gate_nodes_range_high", float(fu_sum["gate_nodes_range_high"]))
_fu_k = float(_fu_primary["p_perm"]) * (stats["fuscc_n_perm"] + 1) - 1
_xcheck("FUSCC (script 69): every analysed patient is recorded as female, the primary used the permutations of "
        "the script, its p is (1 + k) / (1 + n_perm) with a whole k and its Monte Carlo error, both p keys round "
        "the same value, and the correlation stays below the 95th percentile of its null",
        stats["fuscc_n_female"] == stats["fuscc_n_patients"]
        and stats["fuscc_n_perm"] == int(_c69["N_PERM"]) == int(_fu_primary["n_perm"])
        and (fu_corr["n_perm"] == stats["fuscc_n_perm"]).all()
        and abs(_fu_k - round(_fu_k)) < 1e-6 and 0 <= round(_fu_k) <= stats["fuscc_n_perm"]
        and np.isclose(float(_fu_primary["mc_se"]),
                       np.sqrt(float(_fu_primary["p_perm"]) * (1 - float(_fu_primary["p_perm"])) / stats["fuscc_n_perm"]),
                       rtol=1e-12)
        and round(stats["fuscc_r_primary_p_perm_3dp"], 2) == stats["fuscc_r_primary_p_perm"]
        and float(_fu_primary["p_perm"]) > 0.05
        and float(_fu_primary["r_pearson"]) < float(_fu_primary["null_p95"]),
        f"{stats['fuscc_n_female']} women of {stats['fuscc_n_patients']}, p {float(_fu_primary['p_perm'])} "
        f"with {stats['fuscc_n_perm']} permutations")
# Figure 8 draws these two saved nulls (panels b and d). A rerun of script 67 or 69 with
# another seed or N has to fail here too, and not only in the figure.
_cl_set_row = cl_set.loc["candidates7"]
_cl_null = load_csv_required(REAUDIT / "tnbc_permutation_null.csv", "tnbc_permutation_null")["delta_candidates7"].to_numpy()
_fu_null = load_csv_required(FUSCC / "fuscc_permutation_null.csv", "fuscc_permutation_null")["r_all1447"].to_numpy()
_cl_p_null = (1 + int((np.abs(_cl_null) >= abs(float(_cl_set_row["cliffs_delta"])) - 1e-12).sum())) / (1 + len(_cl_null))
_fu_p_null = (1 + int((_fu_null >= float(_fu_primary["r_pearson"])).sum())) / (1 + len(_fu_null))
_xcheck("TNBC lines and FUSCC: the saved permutation nulls of Figure 8 have the registered length and reproduce the p "
        "of tnbc_set_test.csv and fuscc_primary_correlation.csv, the 95th percentile and the spread of each null",
        len(_cl_null) == stats["cltnbc_set_n_perm"] and len(_fu_null) == stats["fuscc_n_perm"]
        and np.isclose(_cl_p_null, float(_cl_set_row["p_perm"]), rtol=0, atol=1e-12)
        and np.isclose(_fu_p_null, float(_fu_primary["p_perm"]), rtol=0, atol=1e-12)
        and np.isclose(np.percentile(_fu_null, 95), float(_fu_primary["null_p95"]), rtol=0, atol=1e-12)
        and np.isclose(np.percentile(np.abs(_cl_null), 95), float(_cl_set_row["null_p95_abs"]), rtol=0, atol=1e-12)
        and np.isclose(_cl_null.std(ddof=1), float(_cl_set_row["null_sd"]), rtol=0, atol=1e-12)
        and np.isclose(_fu_null.mean(), float(_fu_primary["null_mean"]), rtol=0, atol=1e-12)
        and np.isclose(_fu_null.std(ddof=1), float(_fu_primary["null_sd"]), rtol=0, atol=1e-12),
        f"composite p {_cl_p_null} over {len(_cl_null)} permutations, r p {_fu_p_null} over {len(_fu_null)}, "
        f"null p95 {np.percentile(_fu_null, 95)}")
_fu_gate_row = fu_gate.loc["positive lymph nodes"]


def _range_bound(node, i):
    """True when node is GATE_NODES_HR_RANGE[i]."""
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "GATE_NODES_HR_RANGE" and isinstance(node.slice, ast.Constant)
            and node.slice.value == i)


# The observed HR (3.11) lies inside the declared range and inside the 2.0 to 4.5 the
# verdict used before 30-oct, so the outputs cannot tell them apart. The code can: the
# only comparison under a "verdict" key of 69 must be GATE_NODES_HR_RANGE[0] <= the
# hazard ratio <= GATE_NODES_HR_RANGE[1]
_gate_verdict_cmp = [c for d in ast.walk(ast.parse((REPO / "scripts" / "69_fuscc_survival.py").read_text(encoding="utf-8")))
                     if isinstance(d, ast.Dict)
                     for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant) and k.value == "verdict"
                     for c in ast.walk(v) if isinstance(c, ast.Compare)]
_gate_uses_constant = (
    len(_gate_verdict_cmp) == 1
    and [type(o) for o in _gate_verdict_cmp[0].ops] == [ast.LtE, ast.LtE]
    and _range_bound(_gate_verdict_cmp[0].left, 0) and _range_bound(_gate_verdict_cmp[0].comparators[-1], 1)
    and any(isinstance(m, ast.Constant) and m.value == "exp(coef)" for m in ast.walk(_gate_verdict_cmp[0].comparators[0])))
_xcheck("FUSCC (script 69): the cohort gate decides with the range of the script (its only verdict comparison "
        "is GATE_NODES_HR_RANGE[0] <= HR <= GATE_NODES_HR_RANGE[1], read with ast), the one its table declares, "
        "and its verdict is the node hazard ratio inside that range",
        (stats["fuscc_gate_nodes_range_low"], stats["fuscc_gate_nodes_range_high"])
        == tuple(float(v) for v in _c69["GATE_NODES_HR_RANGE"])
        and _gate_uses_constant
        and str(_fu_gate_row["expected_range"])
        == f"{stats['fuscc_gate_nodes_range_low']} to {stats['fuscc_gate_nodes_range_high']}"
        and bool(_fu_gate_row["verdict"])
        == (stats["fuscc_gate_nodes_range_low"] <= float(_fu_gate_row["hr"]) <= stats["fuscc_gate_nodes_range_high"])
        and bool(fu_sum["gate_nodes_verdict"]) == bool(_fu_gate_row["verdict"]),
        f"range {stats['fuscc_gate_nodes_range_low']} to {stats['fuscc_gate_nodes_range_high']}, "
        f"HR {float(_fu_gate_row['hr'])}, verdict {_fu_gate_row['verdict']}, "
        f"{len(_gate_verdict_cmp)} verdict comparison(s), uses the constant {_gate_uses_constant}")

# Proportional hazards of every FUSCC fit (script 69 with the test of script 64), from
# the per-term table: the exact score test of R survival 3.x cox.zph (rank of time) is
# the primary one and the lifelines approximation is kept for reference
_fu_ph_key = fu_ph.set_index(["model", "label", "term"])
_add("fuscc_ph_n_fits", int(fu_ph[["model", "label"]].drop_duplicates().shape[0]))
_add("fuscc_ph_n_terms", len(fu_ph))
for _test in ("exact", "lifelines"):
    _add(f"fuscc_ph_{_test}_n_terms_violating", int(fu_ph[f"ph_violation_p05_{_test}"].sum()))
    _foc = fu_ph[(fu_ph["model"] == "focal12") & fu_ph[f"ph_violation_p05_{_test}"]]
    _add(f"fuscc_ph_{_test}_focal12_violating", sorted(_foc["label"]))
for _arm in ("S1_strata_mrna", "S1b_strata_intrinsic", "S2_clinical", "S3_no_LAR", "S4_surgery_year"):
    _add(f"fuscc_ph_exact_{_arm}_n_terms_violating",
         int(fu_ph.loc[fu_ph["model"] == _arm, "ph_violation_p05_exact"].sum()))
_add("fuscc_ph_exact_S2_size_n_violating",
     int(fu_ph.loc[(fu_ph["model"] == "S2_clinical") & (fu_ph["term"] == "size_cm"), "ph_violation_p05_exact"].sum()))
_add("fuscc_ph_composite_candidates7_p", _p2(_fu_ph_key.at[("composite", "candidates7", "z"), "ph_p_exact"]))
_add("fuscc_ph_composite_controls5_p", _p2(_fu_ph_key.at[("composite", "controls5", "z"), "ph_p_exact"]))
_add("fuscc_ph_gate_nodes_p", _p2(_fu_ph_key.at[("gate", "positive lymph nodes", "x"), "ph_p_exact"]))
_add("fuscc_ph_gate_nodes_complete_cases_p",
     _p2(_fu_ph_key.at[("gate", "positive lymph nodes, complete cases", "x"), "ph_p_exact"]))
_add("fuscc_ph_gate_size_p", _p2(_fu_ph_key.at[("gate", "tumour size (cm)", "x"), "ph_p_exact"]))
for d in TWELVE_DRUGS:
    _add(f"{_role(d)}_{d}_fuscc_ph_p", _p2(_fu_ph_key.at[("focal12", d, "z"), "ph_p_exact"]))

_xcheck("FUSCC (script 69): the analysed patients plus the excluded ones are the 360 sequenced, "
        "the exclusions by reason add up to the counts of the summary, and the reference is the "
        "TNBC subgroup of script 61",
        stats["fuscc_n_patients"] + stats["fuscc_n_excluded"] == stats["fuscc_n_sequenced"] == 360
        and stats["fuscc_n_excluded_zero_time"] + stats["fuscc_n_excluded_predeclared"] == stats["fuscc_n_excluded"]
        and int(fu_excl.at["relapse-free time of zero days", "n_patients"]) == stats["fuscc_n_excluded_zero_time"]
        and stats["fuscc_n_excluded"] == int(fu_sum["n_excluded_total"])
        and (stats["fuscc_scanb_tnbc_n"], stats["fuscc_scanb_tnbc_events"])
        == (stats["tnbc_n_patients"], stats["tnbc_n_events"]),
        f"{stats['fuscc_n_patients']} analysed and {stats['fuscc_n_excluded']} excluded")
# The reference effects rebuilt from their two sources, the TNBC HR per SD of script 61
# and the two spreads of the predicted AUC, and the power from the normal approximation
_tnbc61 = load_csv_required(REAUDIT / "tnbc_subgroup_cox.csv", "tnbc_subgroup_cox")
_tnbc61 = _tnbc61[_tnbc61["subset"] == "TNBC_clinical"].set_index("drug")
_fu_refz = load_csv_required(FUSCC / "fuscc_scanb_tnbc_reference_z.csv", "fuscc_scanb_tnbc_reference_z").set_index("drug")
_fu_ref_ok = []
for _d in ("pelitinib", "osimertinib"):
    _hr = float(np.exp(np.log(_tnbc61.at[_d, "hr_per_sd"])
                       * _fu_refz.at[_d, "sd_pred_auc_tnbc"] / _fu_refz.at[_d, "sd_pred_auc_full"]))
    _ev = stats["fuscc_n_events"]
    _pw = float(norm.sf(norm.ppf(0.975) - abs(np.log(_hr)) * np.sqrt(_ev)))
    _row = _fu_ref.loc[f"{_d}_rescaled"]
    _fu_ref_ok.append(np.isclose(_hr, float(_row["reference_effect"]), rtol=1e-9)
                      and np.isclose(_pw, float(_row["power_1.0x"]), rtol=1e-9)
                      and stats[f"fuscc_power_at_{_d}"] == round(_pw, 2)
                      and stats[f"fuscc_reference_hr_{_d}"] == _r2(_hr))
_xcheck("FUSCC (script 69): the reference effects are the TNBC HR per SD of script 61 rescaled by "
        "each drug's SD inside the subgroup over the SD of the Cox cohort, and the powers follow from "
        "them and the 49 events",
        all(_fu_ref_ok) and len(_fu_ref) == 2,
        f"pelitinib {stats['fuscc_reference_hr_pelitinib']}, osimertinib {stats['fuscc_reference_hr_osimertinib']}")
# Gate: the complete cases drop exactly the missing node counts, all of them on the
# negative side and without a relapse, so the imputation moves no event
_nodes_levels = [[int(v) for v in re.findall(r"\d+", fu_gate.at[t, "n_events_level"])]
                 for t in ("positive lymph nodes", "positive lymph nodes, complete cases")]
_xcheck("FUSCC (script 69): the gate on complete cases drops the patients with no node count, who were "
        "counted as node-negative and carry no relapse, and it does not decide",
        _nodes_levels[0][1] + _nodes_levels[0][3] == stats["fuscc_n_patients"]
        and _nodes_levels[0][:3] == _nodes_levels[1][:3]
        and _nodes_levels[0][3] - _nodes_levels[1][3] == stats["fuscc_gate_nodes_n_missing"]
        and not bool(fu_gate.at["positive lymph nodes, complete cases", "decides"])
        and bool(fu_gate.at["positive lymph nodes", "decides"]),
        f"levels {_nodes_levels}, {stats['fuscc_gate_nodes_n_missing']} missing")
# Proportional hazards: the table covers exactly the fits the other tables of script 69
# report (each hazard ratio is the one published there), and its counts are the summary's
_ph_cov = []
for d in TWELVE_DRUGS:
    _ph_cov.append(np.isclose(_fu_ph_key.at[("focal12", d, "z"), "hr"], fu_perdrug.at[d, "hr_per_sd_fuscc"], rtol=1e-10))
for _set in fu_comp.index:
    _ph_cov.append(np.isclose(_fu_ph_key.at[("composite", _set, "z"), "hr"], fu_comp.at[_set, "hr_per_sd"], rtol=1e-10))
for _t in fu_gate.index:
    _ph_cov.append(np.isclose(_fu_ph_key.at[("gate", _t, "x"), "hr"], fu_gate.at[_t, "hr"], rtol=1e-10))
for _, _r in fu_sens.iterrows():
    _ph_cov.append(np.isclose(_fu_ph_key.at[(_r["arm"], _r["drug"], "z"), "hr"], _r["hr_per_sd"], rtol=1e-10))
_xcheck("FUSCC (script 69): the proportional hazards table tests every fit of the focal, composite, gate and "
        "sensitivity tables plus the sequenced-against-not comparison, with their hazard ratios, its p-values "
        "are probabilities and its counts are those of the summary",
        all(_ph_cov)
        and stats["fuscc_ph_n_fits"] == len(TWELVE_DRUGS) + len(fu_comp) + len(fu_gate) + len(fu_sens) + 1
        and fu_ph[["ph_p_exact", "ph_p_lifelines"]].gt(0).all().all()
        and fu_ph[["ph_p_exact", "ph_p_lifelines"]].le(1).all().all()
        and stats["fuscc_ph_exact_n_terms_violating"] == int(fu_sum["ph_n_terms_violating_p05_exact"])
        and stats["fuscc_ph_lifelines_n_terms_violating"] == int(fu_sum["ph_n_terms_violating_p05_lifelines"])
        and stats["fuscc_ph_n_terms"] == int(fu_sum["ph_n_terms"])
        and float(fu_sum["ph_zph_exact_max_variance_rel_diff_vs_lifelines"]) <= 1e-6,
        f"{sum(_ph_cov)} of {len(_ph_cov)} fits matched, {stats['fuscc_ph_n_fits']} fits")
_xcheck("FUSCC (script 69): the primary correlation recomputed from the per-drug score z equals "
        "the one in the correlation table",
        np.isclose(float(np.corrcoef(fu_perdrug["score_z"], fu_perdrug["scanb_tnbc_score_z_os"])[0, 1]),
                   float(_fu_primary["r_pearson"]), rtol=1e-10)
        and int(_fu_primary["n_drugs"]) == len(fu_perdrug) == 1447,
        f"r = {stats['fuscc_r_primary']}")
_xcheck("FUSCC (script 69): the design detects less than the effects it is compared against, the "
        "cohort gate passes on positive nodes, and the transfer clips no more than SCAN-B does",
        stats["fuscc_mde_hr_per_sd"] > max(stats["fuscc_reference_hr_pelitinib"],
                                           stats["fuscc_reference_hr_osimertinib"])
        and stats["fuscc_gate_nodes_passed"]
        and stats["fuscc_qt_clipped_pct"] <= 1.5 * stats["fuscc_qt_clipped_pct_scanb"]
        and float(fu_power.loc[fu_power["statistic"].str.startswith("per drug"), "power_1.0x"].max()) < 0.8,
        f"MDE {stats['fuscc_mde_hr_per_sd']}, nodes HR {stats['fuscc_gate_nodes_hr']}, "
        f"clipping {stats['fuscc_qt_clipped_pct']}% against {stats['fuscc_qt_clipped_pct_scanb']}%")
_xcheck("FUSCC (script 69): the cohort tests the same drugs as the survival screen",
        stats["fuscc_n_drugs_tested"] == stats["SCANB_n_drugs_tested"]
        and set(fu_perdrug.index) == set(load_csv_required(RESULTS / "v8_multicohort" / "survival_SCANB.csv",
                                                           "survival_SCANB")["drug"]),
        f"{stats['fuscc_n_drugs_tested']} against {stats['SCANB_n_drugs_tested']}")
_xcheck("FUSCC (script 69): no drug reaches FDR in the cohort and the reliability of its z vector "
        "is not estimable, which is what the power statement says",
        stats["fuscc_n_drugs_fdr05"] == 0 and not stats["fuscc_ceiling_estimable"]
        and stats["fuscc_var_z_fuscc"] < stats["fuscc_var_z_reference"],
        f"var(z) {stats['fuscc_var_z_fuscc']} against {stats['fuscc_var_z_reference']}")
print(f"FUSCC (script 69): r = {stats['fuscc_r_primary']} (permutation p "
      f"{stats['fuscc_r_primary_p_perm']}, null p95 {stats['fuscc_r_primary_null_p95']}), "
      f"composite HR/SD {stats['fuscc_composite_hr_per_sd']}, sign agreement "
      f"{stats['fuscc_sign_agreement_pct_1447']}%, MDE {stats['fuscc_mde_hr_per_sd']} with "
      f"{stats['fuscc_n_events']} events")

# --- Scripts 70 and 71: what the gene universe of a cohort costs its transfer ---
# Supplementary Note 2 reports this as a sensitivity: the TCGA activities were
# quantified over the GDC gene set, the transfer clips 80% of them, and recomputing
# them over the PRISM gene universe changes neither the cohort result nor the
# conclusions. The published numbers stay as they are, so these keys describe a
# comparison, never a replacement.
gu = _summary(GENE_UNIVERSE / "gene_universe_summary.csv", "gene_universe_summary")
gu_clip = load_csv_required(GENE_UNIVERSE / "cohort_clipping.csv", "cohort_clipping").set_index("cohort")
# The R half (script 70): genes shared by the TCGA reference matrix and PRISM, and
# the FUSCC activities over the full transcriptome whose clipping made script 68
# restrict that cohort to the PRISM genes (Supplementary Note 2).
gu_r = _summary(GENE_UNIVERSE / "gene_universe_r_summary.csv", "gene_universe_r_summary")
_fuscc_r = _summary(DATA / "precision_processed" / "fuscc_r_summary.csv", "fuscc_r_summary")

for _cohort, _key in (("SCANB", "scanb"), ("METABRIC", "metabric"), ("FUSCC", "fuscc"),
                      ("FUSCC_full_transcriptome", "fuscc_full_transcriptome"),
                      ("TCGA_published", "tcga_published"),
                      ("TCGA_prism_universe", "tcga_prism_universe")):
    _add(f"geneuni_pct_outside_{_key}", float(gu_clip.at[_cohort, "pct_outside_train_range"]))
_add("geneuni_tcga_n_shared_genes", int(gu_r["n_shared_genes"]))
_xcheck("Gene universe (script 70): TCGA is recomputed over all 17,170 genes it shares with PRISM, and the "
        "full-transcriptome FUSCC activities come from the FPKM matrix of script 68 without its restriction",
        int(gu_r["n_genes_used"]) == stats["geneuni_tcga_n_shared_genes"]
        and int(gu_r["n_prism_genes"]) == int(_fuscc_r["n_prism_genes"])
        and str(gu_r["fuscc_expression_md5"]) == str(_fuscc_r["expression_md5"])
        and int(gu_r["fuscc_n_genes_input"]) == int(_fuscc_r["n_genes_input"])
        and int(gu_r["fuscc_n_genes_used"]) > int(_fuscc_r["n_genes_used"])
        and int(gu_r["fuscc_n_samples"]) == int(_fuscc_r["n_samples"])
        and stats["geneuni_pct_outside_fuscc_full_transcriptome"] > 50,
        f"{gu_r['n_genes_used']} genes used of {stats['geneuni_tcga_n_shared_genes']} shared, FUSCC md5 "
        f"{gu_r['fuscc_expression_md5']} against {_fuscc_r['expression_md5']}")
_add("geneuni_tcga_published_fdr05", int(gu["tcga_published_fdr05"]))
_add("geneuni_tcga_prism_universe_fdr05", int(gu["tcga_prism_universe_fdr05"]))
_add("geneuni_tcga_published_nominal05", int(gu["tcga_published_nominal05"]))
_add("geneuni_tcga_prism_universe_nominal05", int(gu["tcga_prism_universe_nominal05"]))
_add("geneuni_tcga_published_lambda_gc", float(gu["tcga_published_lambda_gc"]))
_add("geneuni_tcga_prism_universe_lambda_gc", float(gu["tcga_prism_universe_lambda_gc"]))
_add("geneuni_tcga_published_pred_sd", float(gu["tcga_published_pred_sd"]))
_add("geneuni_tcga_prism_universe_pred_sd", float(gu["tcga_prism_universe_pred_sd"]))
_add("geneuni_tcga_pvalue_spearman", float(gu["tcga_pvalue_spearman"]))
_add("geneuni_meta_published_fdr05", int(gu["meta_published_fdr05"]))
_add("geneuni_meta_prism_universe_fdr05", int(gu["meta_prism_universe_fdr05"]))
_add("geneuni_meta_shared", int(gu["meta_shared"]))
_add("geneuni_meta_lost", int(gu["meta_lost"]))
_add("geneuni_meta_gained", int(gu["meta_gained"]))
for _d in ("paclitaxel", "docetaxel", "olaparib"):
    _add(f"geneuni_meta_{_d}_padj_published", _p2(gu[f"meta_{_d}_padj_published"]))
    _add(f"geneuni_meta_{_d}_padj_prism_universe", _p2(gu[f"meta_{_d}_padj_prism_universe"]))
# Proportional hazards of the canonical TCGA model for the twelve focal drugs, both
# versions (script 71 with the test of script 64), from the per-term table
gu_ph = load_csv_required(GENE_UNIVERSE / "tcga_ph_schoenfeld.csv", "tcga_ph_schoenfeld")
_gu_cmp = load_csv_required(GENE_UNIVERSE / "tcga_screen_comparison.csv", "tcga_screen_comparison").set_index("drug")
for _v, _g in gu_ph.groupby("version"):
    _drug_terms = _g[_g["term"] == "predicted_auc"]
    _add(f"geneuni_ph_{_v}_exact_drug_terms_violating",
         sorted(_drug_terms.loc[_drug_terms["ph_violation_p05_exact"], "drug"]))
    _add(f"geneuni_ph_{_v}_exact_pam50_terms_violating",
         int(_g.loc[_g["term"] != "predicted_auc", "ph_violation_p05_exact"].sum()))
    _add(f"geneuni_ph_{_v}_exact_drug_term_p_min", _p2(_drug_terms["ph_p_exact"].min()))
_gu_ph_drug = gu_ph[gu_ph["term"] == "predicted_auc"]
_xcheck("Gene universe (script 71): proportional hazards tested on the screen's own fits, the twelve focal "
        "drugs of both TCGA versions with every term, their hazard ratios those of the comparison table, "
        "and the counts those of the summary",
        len(gu_ph) == 2 * len(TWELVE_DRUGS) * int(gu_ph.groupby(["version", "drug"]).size().iloc[0])
        and sorted(_gu_ph_drug["drug"].unique()) == sorted(TWELVE_DRUGS)
        and all(np.isclose(_r["HR"], _gu_cmp.at[_r["drug"], f"HR_{_r['version']}"], rtol=1e-10)
                for _, _r in _gu_ph_drug.iterrows())
        and all(len(stats[f"geneuni_ph_{_v}_exact_drug_terms_violating"])
                == int(gu[f"ph_{_v}_n_drug_terms_violating_p05_exact"])
                and stats[f"geneuni_ph_{_v}_exact_pam50_terms_violating"]
                == int(gu[f"ph_{_v}_n_pam50_term_tests_violating_p05_exact"])
                for _v in ("published", "prism_universe"))
        and float(gu["ph_zph_exact_max_variance_rel_diff_vs_lifelines"]) <= 1e-6,
        f"{len(gu_ph)} rows, violating {[stats[f'geneuni_ph_{_v}_exact_drug_terms_violating'] for _v in ('published', 'prism_universe')]}")

_xcheck("Gene universe (scripts 70 and 71): the published TCGA screen and the published Fisher "
        "count are the ones this comparison starts from, and the recomputation leaves the cohort "
        "at zero drugs",
        stats["geneuni_meta_published_fdr05"] == stats["Fisher_drugs_FDR005"]
        and stats["geneuni_tcga_published_fdr05"] == stats["geneuni_tcga_prism_universe_fdr05"] == 0
        and stats["geneuni_tcga_published_nominal05"] <= 0.05 * 1447
        and stats["geneuni_tcga_prism_universe_nominal05"] <= 0.05 * 1447,
        f"{stats['geneuni_meta_published_fdr05']} drugs in the published meta-analysis")
# Figure 5a: the drugs at nominal p < 0.05 of the same screens as the FDR counts of
# Table 2, drawn as outline bars around those counts.
for _c in _COHORTS:
    _add(f"{_c}_drugs_nominal005", int((cohort_dfs[_c]["pvalue"] < 0.05).sum()))
_xcheck("Figure 5a: in each cohort every drug has a p-value, the drugs at FDR < 0.05 are among those at nominal "
        "p < 0.05 and FDR count <= nominal count <= drugs tested, the three cohorts tested the same number of "
        "drugs, and the nominal count of TCGA is the one of scripts 70 and 71",
        all(cohort_dfs[_c]["pvalue"].notna().all()
            and not ((cohort_dfs[_c]["padj"] < 0.05) & (cohort_dfs[_c]["pvalue"] >= 0.05)).any()
            and stats[f"{_c}_drugs_FDR005"] <= stats[f"{_c}_drugs_nominal005"] <= stats[f"{_c}_n_drugs_tested"]
            for _c in _COHORTS)
        and len({stats[f"{_c}_n_drugs_tested"] for _c in _COHORTS}) == 1
        and stats["TCGA_drugs_nominal005"] == stats["geneuni_tcga_published_nominal05"],
        str({_c: (stats[f"{_c}_drugs_FDR005"], stats[f"{_c}_drugs_nominal005"], stats[f"{_c}_n_drugs_tested"])
             for _c in _COHORTS}))
_xcheck("Gene universe (scripts 70 and 71): the clipping of TCGA is the outlier among the cohorts "
        "and the recomputation brings it in line, while the meta-analysis barely moves",
        stats["geneuni_pct_outside_tcga_published"] > 50
        and stats["geneuni_pct_outside_tcga_prism_universe"] < 20
        and max(stats["geneuni_pct_outside_scanb"], stats["geneuni_pct_outside_metabric"],
                stats["geneuni_pct_outside_fuscc"]) < 20
        and abs(stats["geneuni_meta_prism_universe_fdr05"] - stats["geneuni_meta_published_fdr05"]) <= 20
        and stats["geneuni_meta_shared"] >= 0.9 * stats["geneuni_meta_published_fdr05"],
        f"{stats['geneuni_pct_outside_tcga_published']}% against "
        f"{stats['geneuni_pct_outside_tcga_prism_universe']}% after recomputing")
print(f"Gene universe (scripts 70 and 71): TCGA clipping "
      f"{stats['geneuni_pct_outside_tcga_published']}% -> "
      f"{stats['geneuni_pct_outside_tcga_prism_universe']}%, lambda "
      f"{stats['geneuni_tcga_published_lambda_gc']} -> {stats['geneuni_tcga_prism_universe_lambda_gc']}, "
      f"cohort still at {stats['geneuni_tcga_prism_universe_fdr05']} drugs, Fisher "
      f"{stats['geneuni_meta_published_fdr05']} -> {stats['geneuni_meta_prism_universe_fdr05']}")



# --- ComBat versus QuantileTransformer (script 41): which drugs change status ---
# The QT columns of the comparison are survival_SCANB.csv and the ComBat columns are
# combat_cox_scanb.csv, both checked below. The counts and the percentage are
# recomputed from the per-drug table rather than read from the summary.
cb_sum = load_csv_required(RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv",
                           "combat_vs_qt_summary").set_index("metric")["value"]
cb_cmp = load_csv_required(RESULTS / "v8_multicohort" / "combat_vs_qt_comparison.csv",
                           "combat_vs_qt_comparison").set_index("drug")
cb_cox = load_csv_required(RESULTS / "v8_multicohort" / "combat_cox_scanb.csv", "combat_cox_scanb").set_index("drug")
_cq, _cc_sig = cb_cmp["padj_qt"] < 0.05, cb_cmp["padj_combat"] < 0.05
_cb_same = np.sign(np.log(cb_cmp["HR_qt"])) == np.sign(np.log(cb_cmp["HR_combat"]))
_add("combat_qt_n_only_qt", int((_cq & ~_cc_sig).sum()))
_add("combat_qt_n_only_combat", int((~_cq & _cc_sig).sum()))
_add("combat_qt_n_both_same_direction", int((_cq & _cc_sig & _cb_same).sum()))
_cb_cand_sig = [d for d in CANDIDATES if _cc_sig[d]]
_add("combat_candidates_n_sig", len(_cb_cand_sig))
_add("combat_candidates_all_sig", len(_cb_cand_sig) == len(CANDIDATES))
_add("combat_candidates_not_sig", [d for d in CANDIDATES if d not in _cb_cand_sig])
for d in TWELVE_DRUGS:
    _add(f"combat_{d}_padj", _p2(cb_cmp.at[d, "padj_combat"]))
_add("combat_controls_fdr05", [d for d in positive_controls if _cc_sig[d]])
_add("combat_controls_n_sig", len(stats["combat_controls_fdr05"]))
_xcheck("ComBat (script 41): QT columns = survival_SCANB.csv, ComBat columns = combat_cox_scanb.csv, counts and "
        "98.6% recomputed = combat_vs_qt_summary.csv",
        len(cb_cmp) == len(cb_cox) == stats["SCANB_n_drugs_tested"]
        and np.allclose(np.log(cb_cmp.loc[_surv["SCANB"].index, "HR_qt"]), np.log(_surv["SCANB"]["HR"]), rtol=0, atol=1e-9)
        and np.allclose(cb_cmp.loc[_surv["SCANB"].index, "padj_qt"], _surv["SCANB"]["padj"], rtol=1e-9, atol=0)
        and np.allclose(np.log(cb_cmp.loc[cb_cox.index, "HR_combat"]), np.log(cb_cox["HR"]), rtol=0, atol=1e-12)
        and np.allclose(cb_cmp.loc[cb_cox.index, "padj_combat"], cb_cox["padj"], rtol=1e-12, atol=0)
        and (cb_cmp["sig_combat"].astype(bool) == _cc_sig).all() and (cb_cmp["same_direction"].astype(bool) == _cb_same).all()
        and [int(_cc_sig.sum()), int(_cq.sum()), int((_cq & _cc_sig).sum()), stats["combat_qt_n_only_qt"],
             stats["combat_qt_n_only_combat"], stats["combat_qt_n_both_same_direction"]]
        == [int(cb_sum[k]) for k in ("n_sig_combat", "n_sig_qt", "n_sig_both", "n_sig_only_qt", "n_sig_only_combat",
                                     "direction_concordant_among_both_sig")]
        and [int(_cc_sig.sum()), int(_cq.sum()), int((_cq & _cc_sig).sum())]
        == [stats["combat_n_sig"], stats["qt_n_sig"], stats["combat_qt_n_both"]]
        and _pct(stats["combat_qt_n_both_same_direction"], stats["combat_qt_n_both"]) == stats["combat_qt_direction_pct"],
        "ComBat comparison tables disagree")
print(f"ComBat (script 41): candidates {stats['combat_candidates_n_sig']}/7 still FDR<0.05, controls "
      f"{stats['combat_controls_fdr05']}, paclitaxel padj {stats['combat_paclitaxel_padj']}")

# --- Rank of each candidate in the statistical filter and in the whole screen ---
# The filter ranks by padj (src/analysis/candidate_selection.py) the PAM50-adjusted
# fit, which is survival_SCANB.csv, and the screen by p-value (ties share the lowest rank).
_scanb_rank = _surv["SCANB"]["pvalue"].rank(method="min").astype(int)
_s1_rank = step1.set_index("drug")["rank"]
for d in CANDIDATES:
    _add(f"candrank_{d}_in_filter", int(_s1_rank[d]))
    _add(f"candrank_{d}_in_screen", int(_scanb_rank[d]))
_add("candrank_filter_min", min(stats[f"candrank_{d}_in_filter"] for d in CANDIDATES))
_add("candrank_filter_max", max(stats[f"candrank_{d}_in_filter"] for d in CANDIDATES))
_add("candrank_screen_min", min(stats[f"candrank_{d}_in_screen"] for d in CANDIDATES))
_add("candrank_screen_max", max(stats[f"candrank_{d}_in_screen"] for d in CANDIDATES))
_add("candrank_n_drugs_filter", int(len(step1)))
_add("candrank_n_drugs_screen", int(len(_scanb_rank)))
_s1_file = pd.read_csv(OUT / "supp_table_S1_candidate_selection.csv").set_index("drug")
_pam_idx = df_pam.set_index("drug")
_xcheck("Candidate ranks: filter rank = Supplementary Data 1 = rank of p within the 143, the PAM50-adjusted fit = "
        "survival_SCANB.csv, no tie in p at the candidates",
        (_s1_file.loc[CANDIDATES, "rank"] == _s1_rank[CANDIDATES]).all()
        and (_surv["SCANB"].loc[_s1_rank.index, "pvalue"].rank(method="min").astype(int) == _s1_rank).all()
        and np.allclose(np.log(_pam_idx.loc[_surv["SCANB"].index, "HR"]), np.log(_surv["SCANB"]["HR"]), rtol=0, atol=1e-9)
        and np.allclose(_pam_idx.loc[_surv["SCANB"].index, "pvalue"], _surv["SCANB"]["pvalue"], rtol=1e-9, atol=0)
        and int(_surv["SCANB"]["pvalue"].isin(_surv["SCANB"].loc[CANDIDATES, "pvalue"]).sum()) == len(CANDIDATES)
        and stats["candrank_n_drugs_filter"] == stats["candidates_n_statistical_filter"]
        and stats["candrank_n_drugs_screen"] == stats["SCANB_n_drugs_tested"]
        and stats["candrank_filter_max"] <= SELECTION_RULE["rank_cutoff"],
        f"filter {[stats[f'candrank_{d}_in_filter'] for d in CANDIDATES]}")
print(f"Candidate ranks: {stats['candrank_filter_min']}-{stats['candrank_filter_max']} of "
      f"{stats['candrank_n_drugs_filter']} in the statistical filter, "
      f"{stats['candrank_screen_min']}-{stats['candrank_screen_max']} of "
      f"{stats['candrank_n_drugs_screen']} in SCAN-B")

# --- METABRIC replication of the seven candidates (script 21, not adjusted for
# clinical covariates, unlike the clinical_adj_* keys) ---
_met = _surv["METABRIC"]
for d in CANDIDATES:
    _add(f"metabric_repl_{d}_hr", _r2(_met.at[d, "HR"]))
    _add(f"metabric_repl_{d}_p", _p2(_met.at[d, "pvalue"]))
    _add(f"metabric_repl_{d}_padj", _p2(_met.at[d, "padj"]))
_add("metabric_repl_n_hr_above1", int(sum(float(_met.at[d, "HR"]) > 1 for d in CANDIDATES)))
_add("metabric_repl_fdr05_candidates", [d for d in CANDIDATES if float(_met.at[d, "padj"]) < 0.05])
_add("metabric_repl_n_fdr05", len(stats["metabric_repl_fdr05_candidates"]))
_xcheck("METABRIC replication keys = the METABRIC OS columns of Supplementary Table S1 (metabric_os_<drug>_*)",
        all(stats[f"metabric_repl_{d}_hr"] == stats[f"metabric_os_{d}_hr"]
            and stats[f"metabric_repl_{d}_padj"] == stats[f"metabric_os_{d}_padj"] for d in CANDIDATES),
        "two keys of the same METABRIC fit differ")
print(f"METABRIC replication: {stats['metabric_repl_n_hr_above1']}/7 candidates with HR>1, "
      f"FDR<0.05: {stats['metabric_repl_fdr05_candidates']}")

# --- Platinum salts, the reference for the choice of positive controls ---
# SCAN-B PAM50-adjusted fit (survival_SCANB.csv) and Fisher meta-analysis (script 21)
PLATINUM = ["cisplatin", "carboplatin", "oxaliplatin", "satraplatin"]
for d in PLATINUM:
    _add(f"platinum_{d}_scanb_hr", _r2(_surv["SCANB"].at[d, "HR"]))
    _add(f"platinum_{d}_scanb_padj", _p2(_surv["SCANB"].at[d, "padj"]))
    _add(f"platinum_{d}_fisher_padj", _p2(_fi.at[d, "fisher_padj"]))
_add("platinum_n_scanb_fdr05", int((_surv["SCANB"].loc[PLATINUM, "padj"] < 0.05).sum()))
_add("platinum_n_fisher_fdr05", int((_fi.loc[PLATINUM, "fisher_padj"] < 0.05).sum()))
_xcheck("Platinum salts: the four drugs of the screen named *platin, SCAN-B HR and padj = hr_per_sd_all_drugs.csv "
        "(script 64), Fisher padj = signed_meta_stouffer.csv (script 63)",
        sorted(d for d in _surv["SCANB"].index if d.endswith("platin")) == sorted(PLATINUM)
        and all(np.isclose(hrsd.at[d, "HR"], _surv["SCANB"].at[d, "HR"], rtol=1e-12)
                and np.isclose(hrsd.at[d, "padj"], _surv["SCANB"].at[d, "padj"], rtol=1e-12)
                and np.isclose(stouffer.at[d, "fisher_padj"], _fi.at[d, "fisher_padj"], rtol=1e-9) for d in PLATINUM),
        f"platinum drugs {sorted(d for d in _surv['SCANB'].index if 'platin' in d)}")
print(f"Platinum salts: SCAN-B padj {[stats[f'platinum_{d}_scanb_padj'] for d in PLATINUM]}, Fisher padj "
      f"{[stats[f'platinum_{d}_fisher_padj'] for d in PLATINUM]}")

# --- TF-level Cox (script 44): where the proliferation TFs sit among all TFs ---
_tfc = load_csv_required(RESULTS / "v8_multicohort" / "tf_level_survival_scanb.csv", "tf_level_survival_scanb")
_tfc_rank = _tfc.set_index("tf")["pvalue"].rank(method="min").astype(int)
_add("tfcox_n_tfs", int(len(_tfc)))
# GATA3 is named in Results next to TOX3 as a luminal regulator among the strongest
for g in ("MYC", "E2F1", "TP53", "GATA3"):
    _add(f"tfcox_{g}_rank", int(_tfc_rank[g]))
_add("tfcox_top5_names", _tfc.nsmallest(5, "pvalue")["tf"].tolist())
_xcheck("TF-level Cox (script 44): one row per SCAN-B TF (769), no tie in p",
        stats["tfcox_n_tfs"] == stats["SCANB_n_tfs"] == _tfc["tf"].nunique()
        and not _tfc["pvalue"].duplicated().any(),
        f"{stats['tfcox_n_tfs']} TFs")
print(f"TF-level Cox (script 44): top 5 {stats['tfcox_top5_names']}, MYC rank {stats['tfcox_MYC_rank']}, "
      f"E2F1 {stats['tfcox_E2F1_rank']}, TP53 {stats['tfcox_TP53_rank']} of {stats['tfcox_n_tfs']}")

# --- PC1-adjusted attenuation of the candidates ---
# Script 42 residualizes the 200 drugs with the most variable predictions and writes
# only p-values, so neither the change in log hazard ratio nor the direction of the
# seven candidates can be read from it: none of them is in the tested subset. The
# counts are registered so the gap is visible and pinned.
_pca_pd = load_csv_required(RESULTS / "v8_multicohort" / "pca_residualization_perdrug.csv",
                            "pca_residualization_perdrug")
_add("pca_perdrug_n_tested", int(len(_pca_pd)))
_add("pca_perdrug_candidates_present", sorted(set(_pca_pd["drug"]) & set(CANDIDATES)))
_add("pca_perdrug_n_candidates_present", len(stats["pca_perdrug_candidates_present"]))
_add("pca_perdrug_has_hazard_ratios", bool({"HR_raw", "HR_residualized"} <= set(_pca_pd.columns)))
_xcheck("PC1 residualization (script 42): nominal counts of the per-drug table = its summary",
        [int((_pca_pd["p_raw"] < 0.05).sum()), int((_pca_pd["p_residualized"] < 0.05).sum())]
        == [int(_pca_sum["n_nominal_sig_raw"]), int(_pca_sum["n_nominal_sig_residualized"])]
        and _pca_pd["drug"].nunique() == len(_pca_pd),
        "per-drug table and summary of script 42 disagree")

# --- Hyperparameters of Methods and their provenance ---
# Parsed from the modules that use them so the text cannot drift from the code: the
# constants RIDGE_ALPHA and RF_PARAMS of script 11 and the Ridge calls of scripts 12
# and 21 with ast, and the GNN dictionaries of src/models/training.py the same way,
# which avoids importing torch here. Where each value comes from is read from the
# outputs of the search of script 72 below and from script 66, which reads the git
# history.
_c11 = _module_constants(REPO / "scripts" / "11_rerun_fair_comparison.py")
_add("hp_ridge_expression_alpha", _c11["RIDGE_ALPHA"])
for _k, _v in _c11["RF_PARAMS"].items():
    _add(f"hp_rf_tf_{_k}", _v)


def _ridge_alphas(path):
    """alpha of every Ridge(...) call of a script."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Ridge":
            out.update(ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == "alpha")
    return out


# Ridge of the hybrid row of Table 1 (script 12) and of the clinical transfer (script 21)
for _key, _script in (("hp_hybrid_ridge_alpha", "12_gnn_embeddings_perdrug.py"),
                      ("hp_transfer_ridge_alpha", "21_multicohort_survival.py")):
    _alphas = _ridge_alphas(REPO / "scripts" / _script)
    if len(_alphas) != 1:
        raise AssertionError(f"scripts/{_script}: expected one Ridge alpha, found {_alphas}")
    _add(_key, _alphas.pop())

_mtr = ast.parse((REPO / "src" / "models" / "training.py").read_text(encoding="utf-8"))
_tr_consts = {t.id: ast.literal_eval(n.value)
              for n in _mtr.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("ARCHS", "OPTIM")}
for _k, _v in _tr_consts["OPTIM"].items():
    _add(f"hp_gnn_{_k}", _v)
for _arch, _cfg in _tr_consts["ARCHS"].items():
    for _k, _v in _cfg.items():
        if _k != "conv_type":
            _add(f"hp_{_arch}_{_k}", _v)

# The cell-line hold-out of Table 1: src/data/splits.cell_line_holdout_split itself, called
# as script 11 calls it, with the test fraction of src/models/training.py and the split
# seed of script 11 (both read with ast), on the lines of the PRISM response matrix.
# split_n_* repeat holdout_n_* (the sizes script 53 reports) and must equal them.
from src.data.splits import cell_line_holdout_split  # noqa: E402

_test_frac = float(_module_constants(REPO / "src" / "models" / "training.py")["TEST_FRAC"])
_split_seed = int(_module_constants(REPO / "scripts" / "11_rerun_fair_comparison.py")["SEED"])
_prism_resp = pd.read_csv(DATA / "PRISM" / "processed" / "drug_response_prism.csv", index_col=0)
_t1_train, _t1_test = cell_line_holdout_split(_prism_resp, test_frac=_test_frac, seed=_split_seed)
_add("split_n_train_lines", len(_t1_train))
_add("split_n_test_lines", len(_t1_test))
# The 95 test lines must carry exactly the test pairs of Table 1 over the 1,447 drugs
_prism_resp.columns = [normalize_drug_name(c) for c in _prism_resp.columns]
_t1_universe = sorted(set(normalize_drug_name(d) for d in cl_universe["drug"]) & set(_prism_resp.columns))
_xcheck("Split of Table 1: cell_line_holdout_split, called with the test fraction of training.py and the seed of "
        "script 11, has the sizes of script 53 (holdout_n_*), splits the 476 lines of the graph without overlap, "
        "and its test lines carry exactly the test pairs scored in Table 1",
        (stats["split_n_train_lines"], stats["split_n_test_lines"])
        == (stats["holdout_n_train_cells"], stats["holdout_n_test_cells"])
        and stats["split_n_train_lines"] + stats["split_n_test_lines"] == stats["prism_n_cell_lines"]
        and not set(_t1_train) & set(_t1_test)
        and set(_t1_train) | set(_t1_test) == set(_graph_lines)
        and len(_t1_universe) == 1447
        and int(_prism_resp.loc[_t1_test, _t1_universe].notna().sum().sum()) == stats["ref_n_test_pairs"],
        f"{stats['split_n_train_lines']} train and {stats['split_n_test_lines']} test lines, "
        f"{int(_prism_resp.loc[_t1_test, _t1_universe].notna().sum().sum())} test pairs against "
        f"{stats['ref_n_test_pairs']}")

# The hold-out is drawn by cell line, not by donor (DepMap PatientID), so a test line can
# share its donor with a training line. Only counted, the identifiers are never written.
_donor = pd.read_csv(DATA / "DepMap" / "Model.csv", usecols=["ModelID", "PatientID"]).set_index("ModelID")["PatientID"]
_train_donors = set(_donor.reindex(_t1_train).dropna())
_test_donor = _donor.reindex(_t1_test)
_add("split_n_lines_without_donor", int(_donor.reindex(list(_t1_train) + list(_t1_test)).isna().sum()))
_add("split_n_test_lines_sharing_donor_with_train", int(_test_donor.isin(_train_donors).sum()))
_xcheck("Split of Table 1: every line has a DepMap donor, and the test lines that share a donor with a training line "
        "are counted (split_n_test_lines_sharing_donor_with_train)",
        stats["split_n_lines_without_donor"] == 0
        and 0 < stats["split_n_test_lines_sharing_donor_with_train"] < stats["split_n_test_lines"],
        f"{stats['split_n_lines_without_donor']} lines without donor, "
        f"{stats['split_n_test_lines_sharing_donor_with_train']} test lines share a donor")

# Provenance. Ridge and random forest: the search of scripts/72_baseline_hp_search.py,
# run with the design of docs/plan_hp_baselines.md (plan and script versioned in 3d539a0
# before the search ran) on CESGA, job 11133811. Ten random trials per model
# (TPESampler with seed 42), each scored by 5-fold cross-validation inside the 381 train
# lines, on an input bundle without the test lines, with the median per-drug Pearson over
# the 200 most complete drugs. The values used before, chosen on the 95 test lines of
# Table 1 by scripts/06_optuna_tuning.py, and the Table 1 rows they gave are kept as
# hp_previous_* for the supplementary table. SAGE and the optimizer: best SAGE trial of
# scripts/07_optuna_gnn_extended.py, scored on a random edge-level split in which every
# Table 1 test line contributes training and scoring edges. HGT: defaults of script 05
# before any search.
import hashlib  # noqa: E402
import subprocess  # noqa: E402

HP_SEARCH = RESULTS / "v16_hp_search" / "cesga_11133811"
HP_INPUTS = RESULTS / "v16_hp_search" / "inputs"
HP_PLAN_COMMIT = "3d539a0"
HP_SEARCH_SCRIPT = "scripts/72_baseline_hp_search.py"
_HP_TAG = {"ridge": "ridge_expression", "rf": "rf_tf"}
_HP_MODEL = {"ridge": "Ridge_expression", "rf": "RF_TF"}
_RF_KEYS = ("n_estimators", "max_depth", "min_samples_leaf")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git_show(rev_path):
    """A file as versioned in git (git show <commit>:<path>), as raw bytes, or None where the
    commit is not in the history (the public code package does not carry it)."""
    try:
        res = subprocess.run(["git", "show", rev_path], cwd=REPO, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout if res.returncode == 0 else None


def _lf(data):
    """Bytes with CRLF turned into LF, so that a checkout with core.autocrlf hashes like the
    LF copy run on CESGA."""
    return data.replace(b"\r\n", b"\n")


_hp_best = {m: json.loads((HP_SEARCH / f"best_params_{m}.json").read_text()) for m in _HP_TAG}
# round_trip, so that the objective of each trial is the float written by script 72 and the
# best one can be compared exactly with best_params_*.json (the default parser can be one
# ulp off)
_hp_trials = {m: pd.read_csv(HP_SEARCH / f"trials_{m}.csv", float_precision="round_trip") for m in _HP_TAG}
_hp_manifest = json.loads((HP_INPUTS / "manifest.json").read_text())
# folds.csv carries the donor of each train line. In the private repository it is the DepMap
# PatientID, which the public package does not carry: there each donor is D0001, D0002, ...,
# numbered in the sorted order of the PatientID over every model of DepMap Model.csv, the same
# numbering as the Model.csv of the package. The donors are only counted here.
_hp_folds_raw = (HP_INPUTS / "folds.csv").read_bytes()
_hp_folds_is_pseudonymized = b",PT-" not in _hp_folds_raw


def _donor_codes(model_csv):
    """PatientID of DepMap -> D0001, D0002, ... in sorted order, or the identity if already coded."""
    ids = sorted(pd.read_csv(model_csv, usecols=["PatientID"])["PatientID"].dropna().unique())
    if all(str(i).startswith("D") for i in ids):
        return {i: i for i in ids}
    if not all(str(i).startswith("PT-") for i in ids):
        raise ValueError("Model.csv: PatientID is neither DepMap PT- codes nor pseudonyms")
    return {pid: f"D{n:04d}" for n, pid in enumerate(ids, 1)}


def _pseudonymize_folds(raw, codes):
    """folds.csv with the donor field replaced, every other byte kept."""
    lines = raw.decode("utf-8").split("\n")
    if not lines[0].startswith("cell_line,PatientID,"):
        raise ValueError("folds.csv: unexpected header")
    out = [lines[0]]
    for line in lines[1:]:
        if line:
            cell, pid, rest = line.split(",", 2)
            line = ",".join((cell, codes[pid], rest))
        out.append(line)
    return "\n".join(out).encode("utf-8")


_hp_donor_codes = _donor_codes(DATA / "DepMap" / "Model.csv")
_hp_folds_pseudo = _hp_folds_raw if _hp_folds_is_pseudonymized else _pseudonymize_folds(_hp_folds_raw, _hp_donor_codes)
_hp_folds = load_csv_required(HP_INPUTS / "folds.csv", "folds")
_hp_bundle_cells = set(_hp_folds["cell_line"])
_hp_n_split = int((_hp_folds.groupby("PatientID")["fold"].nunique() > 1).sum())
_hp_design = {m: b["design"] for m, b in _hp_best.items()}

_add("hp_ridge_rf_search_n_trials", int(_hp_design["ridge"]["n_trials"]))
_add("hp_ridge_rf_search_n_folds", int(_hp_folds["fold"].nunique()))
_add("hp_ridge_rf_search_n_train_lines", len(_hp_bundle_cells))
_add("hp_ridge_rf_search_n_objective_drugs", int(_hp_design["ridge"]["n_objective_drugs"]))
_add("hp_ridge_rf_search_sampler_seed", int(_hp_design["ridge"]["seed"]))
_add("hp_ridge_rf_search_patients_split_across_folds", _hp_n_split)
# SHA-256 of folds.csv with the donors pseudonymized, the file the public package carries
_add("hp_folds_pseudonymized_sha256", _sha256_bytes(_hp_folds_pseudo))
# Table 1 test lines in the input bundle of the search, which scores on folds of the train
# lines. Until October 2026 these keys described the development search, scored on the 95
# test lines (now hp_previous_search_*).
_add("hp_ridge_rf_search_n_test_lines", len(_hp_bundle_cells & set(_t1_test)))
_add("hp_ridge_rf_search_test_lines_in_table1_test", len(_hp_bundle_cells & set(_t1_test)))
_add("hp_ridge_rf_search_same_test_lines_as_table1", bool(_hp_bundle_cells & set(_t1_test)))
_add("hp_ridge_rf_search_has_validation_split",
     stats["hp_ridge_rf_search_n_folds"] > 1 and _hp_bundle_cells <= set(_t1_train))
_add("hp_ridge_rf_search_plan_commit", HP_PLAN_COMMIT)
# Search space, the constants of script 72 (that of scripts/06_optuna_tuning.py)
_c72 = _module_constants(REPO / HP_SEARCH_SCRIPT)
_add("hp_ridge_rf_search_ridge_alpha_min", float(_c72["RIDGE_ALPHA"][0]))
_add("hp_ridge_rf_search_ridge_alpha_max", float(_c72["RIDGE_ALPHA"][1]))
for _k, _c in (("n_estimators", "RF_N_ESTIMATORS"), ("max_depth", "RF_MAX_DEPTH"),
               ("min_samples_leaf", "RF_MIN_SAMPLES_LEAF")):
    _add(f"hp_ridge_rf_search_rf_{_k}_min", int(_c72[_c][0]))
    _add(f"hp_ridge_rf_search_rf_{_k}_max", int(_c72[_c][1]))
_add("hp_ridge_rf_search_rf_n_estimators_step", int(_c72["RF_N_ESTIMATORS"][2]))
for _m, _tag in _HP_TAG.items():
    _b = _hp_best[_m]
    _add(f"hp_{_tag}_best_trial", int(_b["best_trial"]))
    _add(f"hp_{_tag}_cv_mean", round(float(_b["chosen_cv_mean"]), 4))
    _add(f"hp_{_tag}_cv_sd", round(float(_b["chosen_cv_sd"]), 4))
    _add(f"hp_{_tag}_at_edge", bool(_b["edge_diagnostic"]))
# four significant figures, as in supp_table_hp_search.csv
_add("hp_ridge_expression_alpha_best_trial", float(f"{float(_hp_best['ridge']['best_params']['alpha']):.4g}"))
# Diagnostic of the plan (rule 3): the Ridge optimum lies in the outer 10% of the range
# (alpha > 251), so one point beyond it, alpha 10^3.5, was scored too. Table 1 keeps 506.
_edge = _hp_best["ridge"]["edge_diagnostic"]
if len(_edge) != 1 or _edge[0]["param"] != "alpha":
    raise AssertionError(f"Expected one Ridge edge diagnostic on alpha, got {_edge}")
_add("hp_ridge_expression_edge_alpha_log10", round(float(np.log10(_edge[0]["probe"])), 2))
_add("hp_ridge_expression_edge_cv_mean", round(float(_edge[0]["cv_mean"]), 4))
_add("hp_ridge_expression_edge_cv_sd", round(float(_edge[0]["cv_sd"]), 4))

# Values in use before the search, and the Table 1 rows they gave, as script 66 reads them
# from script 11 and table1_model_comparison.csv at 3d539a0. The development search scored
# its trials on the 95 test lines of Table 1 (optuna06_* of script 66).
_add("hp_previous_ridge_expression_alpha", int(mult["hp_previous_script11_ridge_alpha"]))
for _k in _RF_KEYS:
    _add(f"hp_previous_rf_tf_{_k}", int(mult[f"hp_previous_script11_rf_{_k}"]))
for _m, _tag in _HP_TAG.items():
    for _metric in ("global_pearson", "median_perdrug", "pct_gt_03"):
        _add(f"hp_previous_{_tag}_{_metric}", float(mult[f"previous_table1_{_HP_MODEL[_m]}_{_metric}"]))
_opt_test = _of(mult["optuna06_baseline_test_cells_in_table1_test"])
_add("hp_previous_search_n_trials", int(mult["optuna06_n_trials_baseline"]))
_add("hp_previous_search_n_test_lines", int(mult["optuna06_baseline_test_cell_lines"]))
_add("hp_previous_search_test_lines_in_table1_test", _opt_test[0])
_add("hp_previous_search_has_validation_split", "no validation split" not in str(mult["optuna06_baseline_objective"]))

# The random forest of Table 1 with random_state 42 alone, to separate the change of
# hyperparameters from the change of seed (plan, section on seeds)
_ps = load_csv_required(RESULTS / "v2_fair" / "fair_comparison_per_seed.csv", "fair_comparison_per_seed")
_rf_ps = _ps[_ps["model"] == "RF_TF"].sort_values("init_seed")
_rf42 = _rf_ps[_rf_ps["init_seed"] == 42]
if len(_rf42) != 1:
    raise AssertionError(f"{len(_rf42)} random forest rows with random_state 42 in fair_comparison_per_seed.csv")
_rf42 = _rf42.iloc[0]
_add("table1_RF_TF_seed42_global_pearson", round(float(_rf42["global_pearson"]), 3))
_add("table1_RF_TF_seed42_median_perdrug", round(float(_rf42["median_pearson"]), 3))
_add("table1_RF_TF_seed42_pct_gt_03", round(float(_rf42["pct_gt_0.3"]), 1))

_add("hp_gnn_search_n_trials", int(mult["optuna07_n_trials_per_model"]))
_add("hp_gnn_search_edge_level_split", "edge-level" in str(mult["optuna07_gnn_objective"]))
_add("hp_gnn_search_has_validation_split", "no validation split" not in str(mult["optuna07_gnn_objective"]))
_add("hp_gnn_search_table1_test_lines_with_training_edges",
     _of(mult["gnn_search_split_table1_test_lines_with_training_edges"])[0])
_add("hp_gnn_search_table1_test_lines_with_scoring_edges",
     _of(mult["gnn_search_split_table1_test_lines_with_objective_edges"])[0])
_add("hp_gnn_search_scoring_edges_from_table1_test_lines_pct",
     float(mult["gnn_search_split_objective_edges_from_table1_test_lines_pct"]))
_add("hp_hgt_from_search", str(mult["hgt_equals_optuna06_best"]).startswith("True")
     or str(mult["hgt_equals_optuna07_best"]).startswith("True"))
_xcheck("Hyperparameters parsed here = those script 66 read from scripts 11, 12, 21 and training.py (with the five "
        "random_state values of the random forest), and every Table 1 test line is in the GNN search",
        [stats["hp_ridge_expression_alpha"], stats["hp_rf_tf_n_estimators"], stats["hp_rf_tf_max_depth"],
         stats["hp_rf_tf_min_samples_leaf"], stats["hp_hybrid_ridge_alpha"], stats["hp_transfer_ridge_alpha"]]
        == [mult[k] for k in ("hp_script11_ridge_alpha", "hp_script11_rf_n_estimators", "hp_script11_rf_max_depth",
                              "hp_script11_rf_min_samples_leaf", "hp_script12_ridge_alpha", "hp_script21_ridge_alpha")]
        and mult["hp_script11_rf_seeds"] == ", ".join(str(s) for s in _c11["RF_SEEDS"])
        and all(stats[f"hp_gnn_{k}"] == mult[f"hp_gnn_optim_{k}"] for k in ("lr", "weight_decay", "epochs", "eta_min"))
        and mult["hgt_arch_in_use_hidden_layers_heads_dropout"]
        == ", ".join(str(stats[f"hp_hgt_{k}"]) for k in ("hidden_dim", "n_layers", "n_heads", "dropout"))
        and _of(mult["gnn_search_split_table1_test_lines_with_training_edges"])[1] == stats["holdout_n_test_cells"],
        "hyperparameters differ between the generator and script 66")

# Checks of the baseline search, against its own outputs, the input bundle, git and Table 1
_gnn_seeds = sorted(_ps.loc[_ps["model"] == "SAGE_learned_emb", "init_seed"])
_c11_seeds = list(_c11["RF_SEEDS"])
_xcheck("Random forest of Table 1: five fits with random_state 42, 123, 456, 789 and 1011 (RF_SEEDS of script 11, "
        "the init seeds of the GNNs), whose mean and sd give its Table 1 row and whose seed-42 fit gives "
        "per_drug_RF_TF.csv",
        _rf_ps["init_seed"].tolist() == _c11_seeds == _gnn_seeds == [42, 123, 456, 789, 1011]
        and _rf_ps["run"].is_unique
        and stats["table1_RF_TF_n_seeds"] == len(_rf_ps) == 5
        and round(_rf_ps["global_pearson"].mean(), 3) == stats["table1_RF_TF_global_pearson"]
        and round(_rf_ps["global_pearson"].std(ddof=1), 3) == stats["table1_RF_TF_global_pearson_sd"]
        and round(_rf_ps["median_pearson"].mean(), 3) == stats["table1_RF_TF_median_perdrug"]
        and round(_rf_ps["median_pearson"].std(ddof=1), 3) == stats["table1_RF_TF_median_perdrug_sd"]
        and round(_rf_ps["pct_gt_0.3"].mean(), 1) == stats["table1_RF_TF_pct_gt_03"]
        and (_rf_ps["n_drugs"] == 1447).all()
        and bool(np.isclose(pd.read_csv(RESULTS / "v2_fair" / "per_drug_RF_TF.csv")["pearson_r"].median(),
                            _rf42["median_pearson"], rtol=0, atol=1e-12)),
        f"seeds {_rf_ps['init_seed'].tolist()}, script 11 {_c11_seeds}, GNN {_gnn_seeds}")
_ridge_pd = pd.read_csv(RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv")["pearson_r"].dropna()
# script 59 stops when its refit differs from results/v2_fair, so it is enough that it
# takes the alpha from RIDGE_ALPHA of script 11 and has none of its own


def _script59_alpha_from_script11(src):
    """True when script 59 refits Ridge on expression with the alpha that script11_ridge_alpha()
    reads from RIDGE_ALPHA of scripts/11_rerun_fair_comparison.py, and with no other alpha."""
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    path_ok = any(isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "SCRIPT_11"
                  and ast.unparse(n.value).endswith("'scripts' / '11_rerun_fair_comparison.py'") for n in tree.body)
    reader, refit = funcs.get("script11_ridge_alpha"), funcs.get("reproduce_ridge_expression")
    if not path_ok or reader is None or refit is None:
        return False
    reader_ok = (any(isinstance(c, ast.Constant) and c.value == "RIDGE_ALPHA" for c in ast.walk(reader))
                 and any(isinstance(c, ast.Name) and c.id == "SCRIPT_11" for c in ast.walk(reader)))
    assigns = [n for n in ast.walk(refit) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
               and any(isinstance(t, ast.Name) and t.id == "alpha"
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    alpha_ok = (len(assigns) == 1 and isinstance(assigns[0], ast.Assign) and isinstance(assigns[0].value, ast.Call)
                and isinstance(assigns[0].value.func, ast.Name)
                and assigns[0].value.func.id == "script11_ridge_alpha" and not assigns[0].value.args)
    calls = [c for c in ast.walk(refit) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "run_baseline"]
    call_ok = (len(calls) == 1 and len(calls[0].args) >= 3 and isinstance(calls[0].args[0], ast.Constant)
               and calls[0].args[0].value == "Ridge_expression" and isinstance(calls[0].args[2], ast.Dict)
               and [(ast.unparse(k), ast.unparse(v)) for k, v in zip(calls[0].args[2].keys, calls[0].args[2].values)]
               == [("'alpha'", "alpha")])
    return reader_ok and alpha_ok and call_ok


_src59 = (REPO / "scripts" / "59_reference_baselines.py").read_text(encoding="utf-8")
_xcheck("Ridge row of Table 1 = per_drug_Ridge_expression.csv (median and share above 0.3), fitted once, "
        "and script 59 refits it with the alpha of script 11, read from its RIDGE_ALPHA (checked with ast)",
        _script59_alpha_from_script11(_src59)
        and round(float(_ridge_pd.median()), 3) == stats["table1_Ridge_expression_median_perdrug"]
        and round(100 * float((_ridge_pd > 0.3).mean()), 1) == stats["table1_Ridge_expression_pct_gt_03"]
        and len(_ridge_pd) == stats["table1_Ridge_expression_n_drugs"] == 1447
        and stats["table1_Ridge_expression_n_seeds"] == 1
        and "table1_Ridge_expression_global_pearson_sd" not in stats
        and "RIDGE_EXPR_ALPHA" not in _module_constants(REPO / "scripts" / "59_reference_baselines.py"),
        f"median {_ridge_pd.median()}")
_xcheck("Search of script 72: the chosen trial of each model is the best of its ten trials in trials_*.csv, "
        "and script 11 uses the chosen values (Ridge alpha rounded, random forest as found)",
        all(sorted(_hp_trials[m]["trial"]) == list(range(10))
            and int(_hp_trials[m].loc[_hp_trials[m]["cv_mean"].idxmax(), "trial"]) == _hp_best[m]["best_trial"]
            and float(_hp_trials[m]["cv_mean"].max()) == float(_hp_best[m]["best_cv_mean"])
            and stats[f"hp_{_HP_TAG[m]}_best_trial"] == _hp_best[m]["best_trial"]
            for m in _HP_TAG)
        and float(_hp_trials["ridge"].set_index("trial").at[_hp_best["ridge"]["best_trial"], "alpha"])
        == float(_hp_best["ridge"]["best_params"]["alpha"])
        and tuple(int(_hp_trials["rf"].set_index("trial").at[_hp_best["rf"]["best_trial"], k]) for k in _RF_KEYS)
        == tuple(_hp_best["rf"]["best_params"][k] for k in _RF_KEYS)
        == tuple(_hp_best["rf"]["chosen_params"][k] for k in _RF_KEYS)
        == tuple(stats[f"hp_rf_tf_{k}"] for k in _RF_KEYS)
        # the random forest is not rounded, so its chosen objective is the repeated best trial
        and float(_hp_best["rf"]["chosen_cv_mean"]) == float(_hp_best["rf"]["best_cv_mean"]),
        f"best trials {[_hp_best[m]['best_trial'] for m in _HP_TAG]}")
_xcheck("Search of script 72: the Ridge alpha of the best trial rounded to three significant figures, as the plan says, "
        "is the alpha of best_params_ridge.json and of script 11, and its objective was recomputed with it",
        float(f"{_hp_best['ridge']['best_params']['alpha']:.3g}") == float(_hp_best["ridge"]["chosen_params"]["alpha"])
        == float(stats["hp_ridge_expression_alpha"])
        and float(_hp_best["ridge"]["chosen_params"]["alpha"]) != float(_hp_best["ridge"]["best_params"]["alpha"])
        and float(_hp_best["ridge"]["chosen_cv_mean"]) != float(_hp_best["ridge"]["best_cv_mean"])
        and abs(float(_hp_best["ridge"]["chosen_cv_mean"]) - float(_hp_best["ridge"]["best_cv_mean"])) < 1e-4,
        f"best {_hp_best['ridge']['best_params']['alpha']}, chosen {_hp_best['ridge']['chosen_params']['alpha']}, "
        f"script 11 {stats['hp_ridge_expression_alpha']}")
_bundle_sha = {f: _sha256_bytes((HP_INPUTS / f).read_bytes()) for f in _hp_manifest["sha256"]}
# In the public package folds.csv is the pseudonymized copy, pinned by hp_folds_pseudonymized_sha256
_bundle_checked = {f: s for f, s in _hp_manifest["sha256"].items()
                   if not (f == "folds.csv" and _hp_folds_is_pseudonymized)}
_xcheck("Search of script 72: the input bundle of best_params_*.json is the one of results/v16_hp_search/inputs "
        "(sha256 of its four files in manifest.json, recomputed here), and it holds the 381 train lines and no "
        "Table 1 test line, with no patient split across folds",
        all(_hp_best[m]["bundle_sha256"] == _hp_manifest["sha256"] for m in _HP_TAG)
        and {f: _bundle_sha[f] for f in _bundle_checked} == _bundle_checked
        and (_hp_folds_is_pseudonymized or _bundle_sha["folds.csv"] == _hp_manifest["sha256"]["folds.csv"])
        and sorted(_hp_manifest["sha256"]) == sorted(["response_train.csv.gz", "expression_train.csv.gz",
                                                      "tf_train.csv.gz", "folds.csv"])
        and sorted(_hp_manifest["test_cell_ids"]) == sorted(_t1_test)
        and _hp_bundle_cells == set(_t1_train) and len(_hp_bundle_cells) == _hp_manifest["n_train"] == 381
        and stats["hp_ridge_rf_search_n_test_lines"] == 0 and _hp_n_split == 0,
        f"{sum(_hp_best[m]['bundle_sha256'] == _hp_manifest['sha256'] for m in _HP_TAG)} of 2 bundles match the "
        f"manifest, recomputed sha256 match {_bundle_sha == _hp_manifest['sha256']}")
_script72_sha = (HP_SEARCH / "script_sha256.txt").read_text().split()
_script72_wc = _lf((REPO / HP_SEARCH_SCRIPT).read_bytes())
_xcheck("Search of script 72: the sha256 of the script run on CESGA is that of the working copy of "
        "scripts/72_baseline_hp_search.py (line endings as LF)",
        _script72_sha[1] == HP_SEARCH_SCRIPT and _script72_sha[0] == _sha256_bytes(_script72_wc)
        and b"docs/plan_hp_baselines.md" in _script72_wc,
        f"CESGA {_script72_sha[0][:12]}, working copy {_sha256_bytes(_script72_wc)[:12]}")
# The commit that versioned plan and script before the search ran exists only in the
# private repository. Outside it the next check is skipped and the one of the previous
# Table 1 rows drops its comparison with the JSON at that commit, with a warning.
_script72_at_plan = _git_show(f"{HP_PLAN_COMMIT}:{HP_SEARCH_SCRIPT}")
_plan_at_commit = _git_show(f"{HP_PLAN_COMMIT}:docs/plan_hp_baselines.md")
_prev_json_raw = _git_show(f"{HP_PLAN_COMMIT}:paper/results/paper_statistics.json")
_hp_git = None not in (_script72_at_plan, _plan_at_commit, _prev_json_raw)
if _hp_git:
    _xcheck("Search of script 72: the script run on CESGA is scripts/72_baseline_hp_search.py at 3d539a0, the commit "
            "that versioned the plan with its rules before the search ran",
            _script72_sha[0] == _sha256_bytes(_lf(_script72_at_plan))
            and b"Reglas fijadas antes de ver resultados" in _plan_at_commit,
            f"CESGA {_script72_sha[0][:12]}, 3d539a0 {_sha256_bytes(_lf(_script72_at_plan))[:12]}")
else:
    print(f"  [WARNING] Commit {HP_PLAN_COMMIT} not in this git history: the checks of the script and the previous "
          "Table 1 rows against it are skipped")
_xcheck("Search of script 72: design of the plan in both models (official, 10 trials, 5 folds, 200 drugs, "
        "TPESampler with seed 42), and an edge diagnostic only where the best trial is in the outer 10% of the range "
        "(Ridge alpha > 10^2.4, probed at 10^3.5)",
        all((d["official"], d["n_trials"], d["n_folds"], d["n_objective_drugs"], d["seed"], d["sampler"])
            == (True, 10, 5, 200, 42, "TPESampler(seed=42)") for d in _hp_design.values())
        and stats["hp_ridge_rf_search_n_folds"] == 5 and sorted(_hp_folds["fold"].unique()) == list(range(5))
        and len(_hp_trials["ridge"]) == len(_hp_trials["rf"]) == stats["hp_ridge_rf_search_n_trials"]
        and stats["hp_ridge_expression_at_edge"] == (_hp_best["ridge"]["best_params"]["alpha"] > 10 ** 2.4
                                                    or _hp_best["ridge"]["best_params"]["alpha"] < 10 ** -2.4)
        and stats["hp_rf_tf_at_edge"] == (_hp_best["rf"]["best_params"]["max_depth"] >= 19
                                          or _hp_best["rf"]["best_params"]["min_samples_leaf"] <= 3
                                          or _hp_best["rf"]["best_params"]["n_estimators"] == 300)
        and bool(np.isclose(_edge[0]["probe"], 10 ** 3.5, rtol=1e-12)),
        f"designs {_hp_design}")
_tr_rf = _hp_trials["rf"]
_xcheck("Search of script 72: every trial lies in the search space of the hp_ridge_rf_search_* keys (alpha "
        "log-uniform in [1e-3, 1e3], n_estimators 50 to 300 by 50, max_depth 3 to 20, min_samples_leaf 2 to 20)",
        _hp_trials["ridge"]["alpha"].between(stats["hp_ridge_rf_search_ridge_alpha_min"],
                                             stats["hp_ridge_rf_search_ridge_alpha_max"]).all()
        and (stats["hp_ridge_rf_search_ridge_alpha_min"], stats["hp_ridge_rf_search_ridge_alpha_max"]) == (1e-3, 1e3)
        and all(_tr_rf[k].between(stats[f"hp_ridge_rf_search_rf_{k}_min"], stats[f"hp_ridge_rf_search_rf_{k}_max"]).all()
                for k in _RF_KEYS)
        and ((_tr_rf["n_estimators"] - stats["hp_ridge_rf_search_rf_n_estimators_min"])
             % stats["hp_ridge_rf_search_rf_n_estimators_step"] == 0).all()
        and [stats[f"hp_ridge_rf_search_rf_{k}"] for k in ("n_estimators_min", "n_estimators_max", "n_estimators_step",
                                                           "max_depth_min", "max_depth_max", "min_samples_leaf_min",
                                                           "min_samples_leaf_max")] == [50, 300, 50, 3, 20, 2, 20],
        f"alpha {_hp_trials['ridge']['alpha'].min()} to {_hp_trials['ridge']['alpha'].max()}")
_prev_json = json.loads(_prev_json_raw) if _hp_git else None
_xcheck("Previous Ridge and random forest (hp_previous_*): the values of script 11 and Table 1 at 3d539a0 (script 66) "
        "equal paper_statistics.json at that commit (where the commit is in the history) and the best trials of the "
        "development search, which scored them on the 95 Table 1 test lines without validation split",
        (_prev_json is None
         or (all(stats[f"hp_previous_{t}_{k}"] == _prev_json[f"table1_{_HP_MODEL[m]}_{k}"]
                 for m, t in _HP_TAG.items() for k in ("global_pearson", "median_perdrug", "pct_gt_03"))
             and _prev_json["hp_ridge_expression_alpha"] == stats["hp_previous_ridge_expression_alpha"]
             and all(_prev_json[f"hp_rf_tf_{k}"] == stats[f"hp_previous_rf_tf_{k}"] for k in _RF_KEYS)))
        and str(mult["optuna06_best_ridge_expression"]).startswith(f"alpha {stats['hp_previous_ridge_expression_alpha']}.")
        and str(mult["optuna06_best_rf_tf"]).startswith(", ".join(f"{k} {stats[f'hp_previous_rf_tf_{k}']}"
                                                                    for k in _RF_KEYS))
        and stats["hp_previous_search_n_test_lines"] == stats["hp_previous_search_test_lines_in_table1_test"]
        == stats["holdout_n_test_cells"],
        "previous values differ between script 66, the JSON at 3d539a0 and the development search")

# Supplementary table of the search: the ten trials of each model with their mean and sd
# over the five folds, the best one flagged, the value used in Table 1 (for Ridge the
# rounded alpha, with its objective recomputed), the edge diagnostic of the plan, and the
# previous values chosen on the test lines, with the Table 1 rows of both.
_HP_COLS = ["model", "row", "trial", "alpha", "n_estimators", "max_depth", "min_samples_leaf", "cv_mean", "cv_sd",
            "chosen", "test_global_pearson", "test_global_pearson_sd", "test_median_perdrug",
            "test_median_perdrug_sd", "test_pct_gt_03", "n_seeds"]


def _hp_params(model, src):
    if model == "ridge":
        return {"alpha": float(f"{float(src['alpha']):.4g}")}
    return {k: int(src[k]) for k in _RF_KEYS}


def _hp_test(prefix, model_label):
    return {"test_global_pearson": stats[f"{prefix}_global_pearson"],
            "test_global_pearson_sd": stats.get(f"{prefix}_global_pearson_sd"),
            "test_median_perdrug": stats[f"{prefix}_median_perdrug"],
            "test_median_perdrug_sd": stats.get(f"{prefix}_median_perdrug_sd"),
            "test_pct_gt_03": stats[f"{prefix}_pct_gt_03"],
            "n_seeds": stats.get(f"table1_{model_label}_n_seeds", 1) if prefix.startswith("table1") else 1}


# CV mean and sd with five decimals: with four, trials 0 and 6 of the random forest tie at
# 0.1794 and the table would not show why trial 6 was chosen.
_hp_rows = []
for _m, _tag in _HP_TAG.items():
    _b, _label = _hp_best[_m], _HP_MODEL[_m]
    for _, _r in _hp_trials[_m].sort_values("trial").iterrows():
        _hp_rows.append({"model": _label, "row": "trial", "trial": int(_r["trial"]), **_hp_params(_m, _r),
                         "cv_mean": round(float(_r["cv_mean"]), 5), "cv_sd": round(float(_r["cv_sd"]), 5),
                         "chosen": int(_r["trial"]) == _b["best_trial"]})
    _hp_rows.append({"model": _label, "row": "table1", "trial": int(_b["best_trial"]),
                     **_hp_params(_m, _b["chosen_params"]),
                     "cv_mean": round(float(_b["chosen_cv_mean"]), 5), "cv_sd": round(float(_b["chosen_cv_sd"]), 5),
                     "chosen": False, **_hp_test(f"table1_{_label}", _label)})
    for _e in _b["edge_diagnostic"]:
        _hp_rows.append({"model": _label, "row": "edge_diagnostic",
                         **_hp_params(_m, {**_b["chosen_params"], _e["param"]: _e["probe"]}),
                         "cv_mean": round(float(_e["cv_mean"]), 5), "cv_sd": round(float(_e["cv_sd"]), 5),
                         "chosen": False})
    _hp_rows.append({"model": _label, "row": "previous_selected_on_test",
                     **_hp_params(_m, {k.replace(f"hp_previous_{_tag}_", ""): v for k, v in stats.items()
                                       if k.startswith(f"hp_previous_{_tag}_")}),
                     "chosen": False, **_hp_test(f"hp_previous_{_tag}", _label)})
supp_hp = pd.DataFrame(_hp_rows).reindex(columns=_HP_COLS)
for _c in ("trial", "n_estimators", "max_depth", "min_samples_leaf", "n_seeds"):
    supp_hp[_c] = supp_hp[_c].astype("Int64")
supp_hp.to_csv(OUT / "supp_table_hp_search.csv", index=False)
_sh = pd.read_csv(OUT / "supp_table_hp_search.csv")
_xcheck("Supplementary table of the search (supp_table_hp_search.csv): ten trials per model with one chosen, the best "
        "by mean CV, a Table 1 row with the hp_* values and the Table 1 metrics, the Ridge edge diagnostic only, and "
        "the previous values with the metrics they gave on test",
        all(int(((_sh["model"] == _HP_MODEL[m]) & (_sh["row"] == "trial")).sum()) == 10
            and _sh.loc[(_sh["model"] == _HP_MODEL[m]) & _sh["chosen"], "trial"].tolist() == [_hp_best[m]["best_trial"]]
            and _sh.loc[(_sh["model"] == _HP_MODEL[m]) & (_sh["row"] == "trial"), "cv_mean"].max()
            == _sh.loc[(_sh["model"] == _HP_MODEL[m]) & _sh["chosen"], "cv_mean"].iloc[0]
            and (_sh.loc[(_sh["model"] == _HP_MODEL[m]) & (_sh["row"] == "trial"), "cv_mean"]
                 == _sh.loc[(_sh["model"] == _HP_MODEL[m]) & _sh["chosen"], "cv_mean"].iloc[0]).sum() == 1
            for m in _HP_TAG)
        and _sh.loc[_sh["row"] == "edge_diagnostic", "model"].tolist() == ["Ridge_expression"]
        and _sh.loc[(_sh["model"] == "Ridge_expression") & (_sh["row"] == "table1"), "alpha"].tolist()
        == [stats["hp_ridge_expression_alpha"]]
        and _sh.loc[(_sh["model"] == "Ridge_expression") & (_sh["row"] == "previous_selected_on_test"),
                    ["alpha", "test_global_pearson", "test_median_perdrug", "test_pct_gt_03"]].values.tolist()
        == [[stats["hp_previous_ridge_expression_alpha"], stats["hp_previous_ridge_expression_global_pearson"],
             stats["hp_previous_ridge_expression_median_perdrug"], stats["hp_previous_ridge_expression_pct_gt_03"]]]
        and _sh.loc[(_sh["model"] == "RF_TF") & (_sh["row"] == "table1"), list(_RF_KEYS)].values.tolist()
        == [[stats[f"hp_rf_tf_{k}"] for k in _RF_KEYS]]
        and _sh.loc[(_sh["model"] == "RF_TF") & (_sh["row"] == "previous_selected_on_test"), list(_RF_KEYS)].values.tolist()
        == [[stats[f"hp_previous_rf_tf_{k}"] for k in _RF_KEYS]]
        and _sh.loc[(_sh["model"] == "RF_TF") & (_sh["row"] == "table1"),
                    ["test_global_pearson", "test_global_pearson_sd", "n_seeds"]].values.tolist()
        == [[stats["table1_RF_TF_global_pearson"], stats["table1_RF_TF_global_pearson_sd"], 5]]
        and len(_sh) == 2 * (10 + 1 + 1) + 1,
        f"{len(_sh)} rows")
print(f"Hyperparameters: Ridge alpha {stats['hp_ridge_expression_alpha']} (hybrid and transfer "
      f"{stats['hp_transfer_ridge_alpha']}), RF {stats['hp_rf_tf_n_estimators']} trees / depth "
      f"{stats['hp_rf_tf_max_depth']} / min leaf {stats['hp_rf_tf_min_samples_leaf']}, GNN lr {stats['hp_gnn_lr']} / "
      f"dropout {stats['hp_sage_dropout']} / {stats['hp_gnn_epochs']} epochs. Ridge and RF searched by "
      f"{stats['hp_ridge_rf_search_n_folds']}-fold CV in the {stats['hp_ridge_rf_search_n_train_lines']} train lines "
      f"({stats['hp_ridge_rf_search_n_test_lines']} test lines), CV {stats['hp_ridge_expression_cv_mean']} and "
      f"{stats['hp_rf_tf_cv_mean']}, previously {stats['hp_previous_ridge_expression_alpha']} and "
      f"{stats['hp_previous_rf_tf_n_estimators']}/{stats['hp_previous_rf_tf_max_depth']}/"
      f"{stats['hp_previous_rf_tf_min_samples_leaf']} on the {stats['hp_previous_search_n_test_lines']} test lines")

# --- IG baseline test (script 33): number of random Gaussian baselines ---
# Parsed from the loop of script 33 that builds the random baselines, one draw per
# seed, so Methods cannot drift from the code.
_m33 = ast.parse((REPO / "scripts" / "33_ig_baseline_robustness.py").read_text(encoding="utf-8"))


def _is_random_baseline_call(node):
    return (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "build_baseline"
            and bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == "random")


_rand_loops = [n for n in ast.walk(_m33)
               if isinstance(n, ast.For) and any(_is_random_baseline_call(c) for c in ast.walk(n))]
if len(_rand_loops) != 1:
    raise AssertionError(f"scripts/33_ig_baseline_robustness.py: expected one loop of random baselines, "
                         f"found {len(_rand_loops)}")
_rand_seeds = ast.literal_eval(_rand_loops[0].iter)
_rand_calls = [c for c in ast.walk(_rand_loops[0]) if _is_random_baseline_call(c)]
_add("ig_baseline_n_random_runs", len(_rand_seeds))
_xcheck("IG baseline test (script 33): one call of the random baseline inside the loop, seeded with the loop "
        "variable, distinct seeds",
        len(_rand_calls) == 1 and isinstance(_rand_loops[0].target, ast.Name)
        and any(kw.arg == "random_seed" and isinstance(kw.value, ast.Name) and kw.value.id == _rand_loops[0].target.id
                for kw in _rand_calls[0].keywords)
        and len(set(_rand_seeds)) == len(_rand_seeds) > 1,
        f"seeds {_rand_seeds}")
print(f"IG baseline test (script 33): {stats['ig_baseline_n_random_runs']} random Gaussian baselines, seeds {_rand_seeds}")

# --- Coherence pass of 2026-11-10: numbers the text printed without a key ---
# L143: Stouffer-significant drugs whose SCAN-B and METABRIC hazard ratios point the same way
_add("stouffer_n_sig_same_direction", int((_st_sig & ~_opposite).sum()))
# Table 3, SCAN-B column of the positive controls, as printed
for d in positive_controls:
    _add(f"positive_control_{d}_SCANB_hr", round(float(_surv["SCANB"].at[d, "HR"]), 2))
    _add(f"positive_control_{d}_SCANB_padj", _p2(_surv["SCANB"].at[d, "padj"]))
# Caption of Table 5: fewest held-out lines with a measured response among the candidates
_add("candidate_heldout_n_test_min", int(tr_held.loc[CANDIDATES, "n_test"].min()))
# L292 and L335: TFs each survival cohort shares with the PRISM features (TCGA_n_tfs counts the cohort)
for _c in ("SCANB", "METABRIC", "TCGA"):
    _add(f"{_c}_n_tfs_shared_prism", int(cohort_summary.at[_c, "n_tfs_shared_with_prism"]))
# Table 4: the four printed columns of the ten core TFs
for _, _r in t4.iterrows():
    _add(f"aurora_core_{_r['TF']}_primary_mean", float(_r["Primary_mean"]))
    _add(f"aurora_core_{_r['TF']}_metastasis_mean", float(_r["Metastasis_mean"]))
    _add(f"aurora_core_{_r['TF']}_delta", float(_r["Delta"]))
    _add(f"aurora_core_{_r['TF']}_p", _p2(_r["pvalue"]))
# L268: the composite of the candidates in the five FUSCC-TNBC sensitivity analyses of Methods,
# and the relapse risk of the sequenced patients against the rest of the cohort
_fu_sens = load_csv_required(FUSCC / "fuscc_sensitivity.csv", "fuscc_sensitivity")
_fu_comp = _fu_sens[_fu_sens["drug"] == "composite_candidates7"]
_add("fuscc_sens_n_analyses", int(len(_fu_comp)))
_add("fuscc_sens_composite_hr_min", round(float(_fu_comp["hr_per_sd"].min()), 2))
_add("fuscc_sens_composite_hr_max", round(float(_fu_comp["hr_per_sd"].max()), 2))
_fu_seq = load_csv_required(FUSCC / "fuscc_rnaseq_vs_not.csv", "fuscc_rnaseq_vs_not").set_index("term")
_fu_seq_row = _fu_seq.loc["relapse risk of the sequenced patients"]
_add("fuscc_sequenced_relapse_hr", round(float(_fu_seq_row["value_sequenced"]), 2))
_add("fuscc_sequenced_relapse_p", round(float(_fu_seq_row["p"]), 2))
# Supplementary Figure 14: the cross-screen HGT of script 05, read with ast
_s05 = _module_constants(REPO / "scripts" / "05_train_hgt_and_crossval.py")
_add("crossval_hgt_epochs", int(_s05["EPOCHS"]))
_add("crossval_hgt_lr", float(_s05["LR"]))

# Tables 3 and 4 as typed in the main text, against these keys
_main_tex_path = REPO / "paper" / "latex" / "01_Article_MainText.tex"
# The public package has no LaTeX sources: there the frozen values below pin these keys
_main_tex = _main_tex_path.read_text(encoding="utf-8") if _main_tex_path.exists() else None


def _tex_block(label):
    start = _main_tex.index(f"\\label{{{label}}}")
    return _main_tex[start:_main_tex.index("\\end{tabular}", start)]


def _tex_num(cell):
    return float(cell.replace("$-$", "-").replace("+", "").strip())


_t4_tex = {}
for _line in (_tex_block("tab:aurora").splitlines() if _main_tex else []):
    _cells = [c.strip() for c in _line.split("&")]
    if _cells[0] in set(t4["TF"]):
        _t4_tex[_cells[0]] = [_tex_num(c) for c in _cells[1:5]]
_t3_tex = {}
for _m in re.finditer(r"^\s*([A-Z][a-z]+)\s*&\s*([0-9.]+) \[[0-9.]+--[0-9.]+\] \(([0-9.e-]+)\)",
                      _tex_block("tab:positive_controls") if _main_tex else "", re.M):
    _t3_tex[_m.group(1).lower()] = (float(_m.group(2)), float(_m.group(3)))
_xcheck("Coherence keys: Stouffer same plus opposite direction = significant (script 63), SCAN-B column of Table 3 "
        "and the four columns of Table 4 as typed in the main text, Table 5 lines at most the 95 test lines, shared "
        "TFs at most the TFs of each cohort, five FUSCC-TNBC sensitivity analyses on the 356 patients or a subset",
        stats["stouffer_n_sig_same_direction"] + stats["stouffer_n_sig_opposite_direction"] == stats["stouffer_n_sig"]
        and stats["stouffer_n_sig_same_direction"] == meta_sum["direction_stouffer_sig_n_same"]
        and (_main_tex is None
             or _t3_tex == {d: (stats[f"positive_control_{d}_SCANB_hr"], stats[f"positive_control_{d}_SCANB_padj"])
                            for d in positive_controls}
             and _t4_tex == {tf: [stats[f"aurora_core_{tf}_{k}"]
                                  for k in ("primary_mean", "metastasis_mean", "delta", "p")] for tf in t4["TF"]})
        and int(tr_held.loc[CANDIDATES, "n_test"].max()) == stats["holdout_n_test_cells"]
        and all(stats[f"{c}_n_tfs_shared_prism"] <= stats[f"{c}_n_tfs"] for c in ("SCANB", "METABRIC", "TCGA"))
        and stats["fuscc_sens_n_analyses"] == len(set(_fu_comp["arm"])) == 5
        and (_fu_comp["n"] <= stats["fuscc_n_patients"]).all(),
        f"Table 3 {_t3_tex}, Table 4 rows {len(_t4_tex)}")


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
    # Multi-cohort survival (Figure 5, Supplementary Figures 5 to 7)
    (RESULTS / "v8_multicohort" / "meta_analysis_fisher.csv",        OUT / "meta_analysis_fisher.csv"),
    (RESULTS / "v8_multicohort" / "survival_SCANB.csv",              OUT / "survival_SCANB.csv"),
    (RESULTS / "v8_multicohort" / "survival_METABRIC.csv",           OUT / "survival_METABRIC.csv"),
    (RESULTS / "v8_multicohort" / "survival_TCGA.csv",               OUT / "survival_TCGA.csv"),
    # Cohort sizes (Table 2, Methods dedup chains, AURORA sample counts)
    (RESULTS / "v8_multicohort" / "cohort_summary.csv",              OUT / "cohort_summary.csv"),
    (RESULTS / "v9_aurora" / "aurora_cohort_summary.csv",            OUT / "aurora_cohort_summary.csv"),
    (RESULTS / "graph_summary.csv",                                  OUT / "graph_summary.csv"),
    (RESULTS / "data_summary.csv",                                   OUT / "data_summary.csv"),
    # MOA and target enrichment (Supplementary Figures 8 and 9, script 52)
    (RESULTS / "v8_multicohort" / "moa_enrichment.csv",              OUT / "moa_enrichment.csv"),
    (RESULTS / "v8_multicohort" / "target_enrichment.csv",           OUT / "target_enrichment.csv"),
    # SCAN-B per-drug predictions (Fig6, Fig7): NOT copied here (220 MB).
    # generate_paper_figures.py reads it directly from RESULTS_DIR/v5_final/
    # (see data/README.md). It is regenerated by scripts/17_fix_all_critical.py.
    # AURORA TF data (Fig8)
    (RESULTS / "v9_aurora" / "aurora_tf_activities.csv",             OUT / "aurora_tf_activities.csv"),
    # Canonical IG attributions (Figure 3)
    (RESULTS / "v4_xai" / "tf_importance_ig_seed42.csv",            OUT / "tf_importance_ig_top20.csv"),
    (RESULTS / "v4_xai" / "tf_importance_ig_multi_seed_stability.csv",
                                                                     OUT / "tf_importance_ig_stability.csv"),
    (RESULTS / "v4_xai" / "ig_baseline_stability.csv",              OUT / "ig_baseline_stability.csv"),
    (RESULTS / "v4_xai" / "tf_drug_ig_signed.csv",                  OUT / "tf_drug_ig_signed.csv"),
    # Cell line embeddings (SFig03)
    (RESULTS / "v3_embeddings" / "cell_line_embeddings.csv",         OUT / "cell_line_embeddings.csv"),
    # Training history of the canonical SAGE model (Supplementary Figure 1)
    (RESULTS / "v12_gnn" / "history" / "sage_split42_init42_full.csv", OUT / "gnn_training_history.csv"),
    # Every GNN run with its test metrics (Table 1, Supplementary Figure 12, script 57)
    (RESULTS / "v12_gnn" / "runs.csv",                               OUT / "gnn_runs.csv"),
    (RESULTS / "v2_fair" / "fair_comparison_per_seed.csv",           OUT / "fair_comparison_per_seed.csv"),
    (RESULTS / "v3_embeddings" / "embedding_comparison_per_seed.csv", OUT / "embedding_comparison_per_seed.csv"),
    # Per-drug model results (Supplementary Figure 2)
    (RESULTS / "v2_fair" / "per_drug_RF_TF.csv",                    OUT / "per_drug_RF_TF.csv"),
    (RESULTS / "v2_fair" / "per_drug_Ridge_expression.csv",          OUT / "per_drug_Ridge_expression.csv"),
    (RESULTS / "v2_fair" / "per_drug_SAGE_learned_emb.csv",          OUT / "per_drug_SAGE_learned_emb.csv"),
    (RESULTS / "v2_fair" / "per_drug_HGT_learned_emb.csv",           OUT / "per_drug_HGT_learned_emb.csv"),
    (RESULTS / "v3_embeddings" / "per_drug_Ridge(expression+GNN_emb).csv",
     OUT / "per_drug_Ridge_expr_GNN_emb.csv"),
    # Cross-validation PRISM to GDSC (Supplementary Figure 14)
    (RESULTS / "crossval_prism_to_gdsc.csv",                         OUT / "crossval_prism_to_gdsc.csv"),
    # Basal+Her2 survival (Supplementary Figure 13)
    (RESULTS / "v7_weaknesses" / "basal_her2_survival.csv",          OUT / "basal_her2_survival.csv"),
    (RESULTS / "v7_weaknesses" / "basal_only_survival.csv",          OUT / "basal_only_survival.csv"),
    (RESULTS / "v7_weaknesses" / "subtype_pooling_summary.csv",      OUT / "subtype_pooling_summary.csv"),
    # AURORA drug sensitivity (Supplementary Figure 10)
    (RESULTS / "v9_aurora" / "aurora_drug_primary_vs_meta.csv",      OUT / "aurora_drug_primary_vs_meta.csv"),
    # Graph ablation (Supplementary Figure 12), mean and per seed
    (RESULTS / "v6_strengthen" / "graph_ablation.csv",               OUT / "graph_ablation.csv"),
    (RESULTS / "v6_strengthen" / "graph_ablation_per_seed.csv",      OUT / "graph_ablation_per_seed.csv"),
    # Pairwise agreement of the IG seeds (Results, Figure 3B)
    (RESULTS / "v4_xai" / "tf_importance_ig_seed_pairs.csv",         OUT / "tf_importance_ig_seed_pairs.csv"),
    # PCA / PC1 residualization (Discussion: shared prognostic axis)
    (RESULTS / "v8_multicohort" / "pca_residualization_summary.csv",  OUT / "pca_residualization_summary.csv"),
    (RESULTS / "v8_multicohort" / "pca_residualization_perdrug.csv",  OUT / "pca_residualization_perdrug.csv"),
    # Positive-control forest plot (Supplementary Figure 7): real Cox 95% CIs, SCAN-B vs METABRIC
    (RESULTS / "v8_multicohort" / "forest_positive_controls.csv",     OUT / "forest_positive_controls.csv"),
    # TF-level univariate Cox on SCAN-B OS (Results: MYC/TP53/E2F1 examples, script 44)
    (RESULTS / "v8_multicohort" / "tf_level_survival_scanb.csv",       OUT / "tf_level_survival_scanb.csv"),
    # Cox penalizer sweep (Supplementary Figure 11 + Discussion HR-range claims, script 45)
    (RESULTS / "v8_multicohort" / "penalizer_sweep.csv",              OUT / "penalizer_sweep.csv"),
    # Sensitivity analyses cited in Results and Discussion (scripts 37, 39, 40, 41)
    (RESULTS / "v8_multicohort" / "cox_clinical_comparison.csv",     OUT / "cox_clinical_comparison.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_by_site.csv",                OUT / "aurora_tf_by_site.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_by_site_summary.csv",        OUT / "aurora_tf_by_site_summary.csv"),
    (RESULTS / "v8_multicohort" / "cox_rfs_comparison.csv",          OUT / "cox_rfs_comparison.csv"),
    (RESULTS / "v8_multicohort" / "combat_vs_qt_summary.csv",        OUT / "combat_vs_qt_summary.csv"),
    # Supplementary Note 1: IG on the extended candidate set (scripts 32/35/36/49, --candidate-set extended)
    (RESULTS / "v4_xai" / "ig_extended_comparison.csv",              OUT / "ig_extended_comparison.csv"),
    # Reporting Summary: cohort demographics (script 54) and ICLAC cross-check (script 55)
    (RESULTS / "v10_reporting" / "cohort_demographics.csv",          OUT / "cohort_demographics.csv"),
    (RESULTS / "v10_reporting" / "iclac_summary.csv",                OUT / "iclac_summary.csv"),
    (RESULTS / "v10_reporting" / "iclac_matches.csv",                OUT / "iclac_matches.csv"),
    (RESULTS / "v11_collectri_snapshot" / "collectri_snapshot_sensitivity.csv",
     OUT / "collectri_snapshot_sensitivity.csv"),
    (RESULTS / "v5_final" / "scanb_km_logrank.csv",                  OUT / "scanb_km_logrank.csv"),
    (RESULTS / "v4_xai" / "ig_extended_core_tfs.csv",                OUT / "ig_extended_core_tfs.csv"),
    (RESULTS / "v9_aurora" / "aurora_tf_ig_validation_extended.csv",  OUT / "aurora_tf_ig_validation_extended.csv"),
    # Re-audit of September 2026 (scripts 59 to 64 and the extensions of 42 and 58):
    # every table the re-audit keys, Supplementary Tables S1 and S3 and the new
    # columns of Tables 3 and 5 are read from, so each number can be traced from git.
    (REAUDIT / "reference_baselines.csv",                            OUT / "reference_baselines.csv"),
    (REAUDIT / "transfer_ridge_heldout_perdrug.csv",                 OUT / "transfer_ridge_heldout_perdrug.csv"),
    (REAUDIT / "aurora_background_summary.csv",                      OUT / "aurora_background_summary.csv"),
    (REAUDIT / "aurora_tf_background.csv",                           OUT / "aurora_tf_background.csv"),
    (REAUDIT / "aurora_site_background.csv",                         OUT / "aurora_site_background.csv"),
    (REAUDIT / "aurora_drug_paired.csv",                             OUT / "aurora_drug_paired.csv"),
    (REAUDIT / "tnbc_subgroup_cox.csv",                              OUT / "tnbc_subgroup_cox.csv"),
    (REAUDIT / "tnbc_subgroup_summary.csv",                          OUT / "tnbc_subgroup_summary.csv"),
    (REAUDIT / "pam50_predicted_auc.csv",                            OUT / "pam50_predicted_auc.csv"),
    (REAUDIT / "pam50_composition_sensitive_half.csv",               OUT / "pam50_composition_sensitive_half.csv"),
    (REAUDIT / "treatment_adjusted_cox.csv",                         OUT / "treatment_adjusted_cox.csv"),
    (REAUDIT / "treatment_interaction_cox.csv",                      OUT / "treatment_interaction_cox.csv"),
    (REAUDIT / "treatment_summary.csv",                              OUT / "treatment_summary.csv"),
    (REAUDIT / "meta_dss_summary.csv",                               OUT / "meta_dss_summary.csv"),
    (REAUDIT / "signed_meta_stouffer.csv",                           OUT / "signed_meta_stouffer.csv"),
    (REAUDIT / "direction_agreement_summary.csv",                    OUT / "direction_agreement_summary.csv"),
    (REAUDIT / "positive_control_baserate.csv",                      OUT / "positive_control_baserate.csv"),
    (REAUDIT / "metabric_dss_survival.csv",                          OUT / "metabric_dss_survival.csv"),
    (REAUDIT / "ph_summary.csv",                                     OUT / "ph_summary.csv"),
    (REAUDIT / "ph_schoenfeld.csv",                                  OUT / "ph_schoenfeld.csv"),
    (REAUDIT / "cox_pam50_stratified.csv",                           OUT / "cox_pam50_stratified.csv"),
    (REAUDIT / "hr_per_sd_all_drugs.csv",                            OUT / "hr_per_sd_all_drugs.csv"),
    (REAUDIT / "ph_summary_metabric.csv",                            OUT / "ph_summary_metabric.csv"),
    (REAUDIT / "ph_schoenfeld_metabric.csv",                         OUT / "ph_schoenfeld_metabric.csv"),
    (REAUDIT / "cox_pam50_stratified_metabric.csv",                  OUT / "cox_pam50_stratified_metabric.csv"),
    (REAUDIT / "pc1_orientation.csv",                                OUT / "pc1_orientation.csv"),
    (REAUDIT / "scanb_km_extras.csv",                                OUT / "scanb_km_extras.csv"),
    (REAUDIT / "scanb_km_at_risk.csv",                               OUT / "scanb_km_at_risk.csv"),
    # Scripts 65 and 66, the unrounded fits of scripts 37 and 40 (twelve drugs) with the
    # check of that rerun, and the per-drug ComBat comparison of script 41
    (REAUDIT / "ig_diagnostics_summary.csv",                         OUT / "ig_diagnostics_summary.csv"),
    (REAUDIT / "ig_gradient_sign.csv",                               OUT / "ig_gradient_sign.csv"),
    (REAUDIT / "ig_gradient_sign_by_tf.csv",                         OUT / "ig_gradient_sign_by_tf.csv"),
    (REAUDIT / "multiplicity_summary.csv",                           OUT / "multiplicity_summary.csv"),
    (REAUDIT / "ablation_perdrug_median.csv",                        OUT / "ablation_perdrug_median.csv"),
    (REAUDIT / "metabric_pam50_coverage.csv",                        OUT / "metabric_pam50_coverage.csv"),
    # Optional, no pipeline stage writes it (OPTIONAL_INPUTS)
    (REAUDIT / "talazoparib_family12_summary.csv",                   OUT / "talazoparib_family12_summary.csv"),
    # Script 67, the TNBC cell lines of PRISM. The per-line table and the MOA-matched
    # null stay in results/. Everything the text or a supplementary table could quote
    # is copied here, and the null of the composite, which Figure 8b draws.
    (REAUDIT / "tnbc_selectivity_primary.csv",                       OUT / "tnbc_selectivity_primary.csv"),
    (REAUDIT / "tnbc_selectivity_background.csv",                    OUT / "tnbc_selectivity_background.csv"),
    (REAUDIT / "tnbc_set_test.csv",                                  OUT / "tnbc_set_test.csv"),
    (REAUDIT / "tnbc_permutation_null.csv",                          OUT / "tnbc_permutation_null.csv"),
    (REAUDIT / "tnbc_sensitivity.csv",                               OUT / "tnbc_sensitivity.csv"),
    (REAUDIT / "tnbc_power_mde.csv",                                 OUT / "tnbc_power_mde.csv"),
    (REAUDIT / "tnbc_excluded_drugs.csv",                            OUT / "tnbc_excluded_drugs.csv"),
    (REAUDIT / "tnbc_orthogonality.csv",                             OUT / "tnbc_orthogonality.csv"),
    (REAUDIT / "assay_sensitivity_gate.csv",                         OUT / "assay_sensitivity_gate.csv"),
    (REAUDIT / "gdsc_replication.csv",                               OUT / "gdsc_replication.csv"),
    (REAUDIT / "tnbc_cell_lines_summary.csv",                        OUT / "tnbc_cell_lines_summary.csv"),
    # Script 69, the FUSCC-TNBC cohort. Everything a table or the text could quote is
    # copied here, and the 10,000 permutation replicates, which Figure 8d draws.
    (FUSCC / "fuscc_primary_correlation.csv",                        OUT / "fuscc_primary_correlation.csv"),
    (FUSCC / "fuscc_permutation_null.csv",                           OUT / "fuscc_permutation_null.csv"),
    (FUSCC / "fuscc_cox_perdrug.csv",                                OUT / "fuscc_cox_perdrug.csv"),
    (FUSCC / "fuscc_composite.csv",                                  OUT / "fuscc_composite.csv"),
    (FUSCC / "fuscc_sensitivity.csv",                                OUT / "fuscc_sensitivity.csv"),
    (FUSCC / "fuscc_cohort_gate.csv",                                OUT / "fuscc_cohort_gate.csv"),
    (FUSCC / "fuscc_scanb_tnbc_reference_z.csv",                     OUT / "fuscc_scanb_tnbc_reference_z.csv"),
    # The exclusions travel as counts by reason. fuscc_excluded_patients.csv carries
    # patient identifiers and stays in results/v14_fuscc/.
    (FUSCC / "fuscc_excluded_summary.csv",                           OUT / "fuscc_excluded_summary.csv"),
    (FUSCC / "fuscc_rnaseq_vs_not.csv",                              OUT / "fuscc_rnaseq_vs_not.csv"),
    (FUSCC / "fuscc_power_mde.csv",                                  OUT / "fuscc_power_mde.csv"),
    (FUSCC / "fuscc_ph_tests.csv",                                   OUT / "fuscc_ph_tests.csv"),
    (FUSCC / "fuscc_covariate_coding.csv",                           OUT / "fuscc_covariate_coding.csv"),
    (FUSCC / "fuscc_summary.csv",                                    OUT / "fuscc_summary.csv"),
    # Scripts 70 and 71, the gene-universe sensitivity of Supplementary Note 2
    (GENE_UNIVERSE / "cohort_clipping.csv",                          OUT / "cohort_clipping.csv"),
    (GENE_UNIVERSE / "tcga_screen_comparison.csv",                   OUT / "tcga_screen_comparison.csv"),
    (GENE_UNIVERSE / "fisher_meta_comparison.csv",                   OUT / "fisher_meta_comparison.csv"),
    (GENE_UNIVERSE / "tcga_ph_schoenfeld.csv",                       OUT / "tcga_ph_schoenfeld.csv"),
    (GENE_UNIVERSE / "gene_universe_summary.csv",                    OUT / "gene_universe_summary.csv"),
    (GENE_UNIVERSE / "gene_universe_r_summary.csv",                  OUT / "gene_universe_r_summary.csv"),
    (RESULTS / "v8_multicohort" / "cox_clinical_adjusted_scanb.csv", OUT / "cox_clinical_adjusted_scanb.csv"),
    (RESULTS / "v8_multicohort" / "cox_clinical_adjusted_metabric.csv",
     OUT / "cox_clinical_adjusted_metabric.csv"),
    (RESULTS / "v8_multicohort" / "cox_rfs_scanb.csv",               OUT / "cox_rfs_scanb.csv"),
    (RESULTS / "v8_multicohort" / "combat_vs_qt_comparison.csv",     OUT / "combat_vs_qt_comparison.csv"),
]

# AURORA-US sample codes carry the patient identifier, so this one file is not
# copied byte for byte. Each sample becomes AUR-Pnn-<site>: the patients are
# numbered P01, P02, ... in the sorted order of their codes, the site code (TTP,
# TTM or NT and its number) is kept and the aliquot and plate fields are dropped.
# Figure 7 reads the patient and the site from these fields. Only the first column
# changes, so every value keeps its bytes. results/v9_aurora/ keeps the codes.
_AURORA_SAMPLE = re.compile(r"^AUR-([A-Z0-9]{4})-((?:TTP|TTM|NT)[0-9]+)-[A-Z0-9-]+$")


def _aurora_pseudonymize(src, dst):
    raw = src.read_bytes().decode("utf-8")
    # The public package carries the source already pseudonymized: copied as it is
    if all(re.fullmatch(r"AUR-P[0-9]{2}-(?:TTP|TTM|NT)[0-9]+", r.split(",", 1)[0])
           for r in raw.replace("\r\n", "\n").rstrip("\n").split("\n")[1:]):
        dst.write_bytes(src.read_bytes())
        return
    nl = "\r\n" if "\r\n" in raw else "\n"
    head, *rows = raw.rstrip(nl).split(nl)
    keys = [r.split(",", 1) for r in rows]
    parsed = [_AURORA_SAMPLE.match(k) for k, _ in keys]
    if not all(parsed):
        raise ValueError(f"{src.name}: {sum(m is None for m in parsed)} sample codes without the AURORA-US form")
    code = {p: f"P{i:02d}" for i, p in enumerate(sorted({m.group(1) for m in parsed}), 1)}
    new = [f"AUR-{code[m.group(1)]}-{m.group(2)}" for m in parsed]
    if len(set(new)) != len(new):
        raise ValueError(f"{src.name}: two samples of one patient share a site code")
    dst.write_bytes((nl.join([head] + [f"{n},{rest}" for n, (_, rest) in zip(new, keys)]) + nl).encode("utf-8"))


TRANSFORMS = {OUT / "aurora_tf_activities.csv": _aurora_pseudonymize}
n_copied = 0
for src, dst in copies:
    if src.exists():
        TRANSFORMS.get(dst, shutil.copy2)(src, dst)
        n_copied += 1
        print(f"  COPY: {src.name}" + (" (pseudonymized)" if dst in TRANSFORMS else ""))
    elif src in OPTIONAL_INPUTS:
        print(f"  [WARNING] Optional input not found, not copied: {src.name} ({OPTIONAL_INPUTS[src]})")
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


def _aurora_copy_ok():
    src = pd.read_csv(RESULTS / "v9_aurora" / "aurora_tf_activities.csv", index_col=0)
    cp = pd.read_csv(OUT / "aurora_tf_activities.csv", index_col=0)
    f_src, f_cp = src.index.to_series().str.split("-"), cp.index.to_series().str.split("-")
    return (cp.shape == src.shape and list(cp.columns) == list(src.columns)
            and np.array_equal(cp.to_numpy(), src.to_numpy(), equal_nan=True)
            and cp.index.str.fullmatch(r"AUR-P[0-9]{2}-(TTP|TTM|NT)[0-9]+").all() and cp.index.is_unique
            and (f_cp.str[2].to_numpy() == f_src.str[2].to_numpy()).all()
            and pd.crosstab(f_src.str[1].to_numpy(), f_cp.str[1].to_numpy()).gt(0).sum(axis=1).eq(1).all()
            and f_cp.str[1].nunique() == f_src.str[1].nunique() == stats_clean.get("aurora_cohort_n_patients")
            and len(cp) == stats_clean.get("aurora_cohort_n_samples"))




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
    "KG edges: PPI 84,587 interactions (169,174 directed, 166,352 distinct, over 83,176 gene pairs), "
    "drug-target 3,442, response 614,325, total ~830,000",
    stats_clean.get("ppi_interactions") == 84587 and stats_clean.get("ppi_edges_bidir") == 169174
    and stats_clean.get("ppi_edges_distinct_directed") == 166352
    and stats_clean.get("ppi_gene_pairs_distinct") == 83176
    and stats_clean.get("drug_target_edges") == 3442 and stats_clean.get("response_edges") == 614325
    and stats_clean.get("kg_edges_approx") == 830000,
    f"got ppi={stats_clean.get('ppi_interactions')} dt={stats_clean.get('drug_target_edges')} "
    f"resp={stats_clean.get('response_edges')} total={stats_clean.get('kg_edges_total')}",
)

# Cohort sizes (script 21) cited in Table 2 and Methods
_check(
    "Survival cohorts before the PAM50 filter (Methods chain): SCAN-B 7,397 / 1,188, METABRIC 1,979 / 1,143, TCGA 1,072 / 150",
    (stats_clean.get("SCANB_n_patients"), stats_clean.get("SCANB_n_events")) == (7397, 1188)
    and (stats_clean.get("METABRIC_n_patients"), stats_clean.get("METABRIC_n_events")) == (1979, 1143)
    and (stats_clean.get("TCGA_n_patients"), stats_clean.get("TCGA_n_events")) == (1072, 150),
    f"got SCANB {stats_clean.get('SCANB_n_patients')}/{stats_clean.get('SCANB_n_events')}, "
    f"METABRIC {stats_clean.get('METABRIC_n_patients')}/{stats_clean.get('METABRIC_n_events')}, "
    f"TCGA {stats_clean.get('TCGA_n_patients')}/{stats_clean.get('TCGA_n_events')}",
)
_check(
    "Table 2 cohorts = PAM50-adjusted Cox populations: SCAN-B 7,215 / 1,164, METABRIC 1,979 / 1,143, TCGA 1,060 / 149",
    (stats_clean.get("SCANB_n_patients_cox"), stats_clean.get("SCANB_n_events_cox")) == (7215, 1164)
    and (stats_clean.get("METABRIC_n_patients_cox"), stats_clean.get("METABRIC_n_events_cox")) == (1979, 1143)
    and (stats_clean.get("TCGA_n_patients_cox"), stats_clean.get("TCGA_n_events_cox")) == (1060, 149),
    f"got SCANB {stats_clean.get('SCANB_n_patients_cox')}/{stats_clean.get('SCANB_n_events_cox')}, "
    f"METABRIC {stats_clean.get('METABRIC_n_patients_cox')}/{stats_clean.get('METABRIC_n_events_cox')}, "
    f"TCGA {stats_clean.get('TCGA_n_patients_cox')}/{stats_clean.get('TCGA_n_events_cox')}",
)
_check(
    "Cox population never exceeds the survival cohort it is filtered from",
    all((stats_clean.get(f"{c}_n_patients_cox") or 0) <= (stats_clean.get(f"{c}_n_patients") or 0)
        and (stats_clean.get(f"{c}_n_events_cox") or 0) <= (stats_clean.get(f"{c}_n_events") or 0)
        for c in ("SCANB", "METABRIC", "TCGA")),
    "a Cox population is larger than its survival cohort",
)
_check(
    "ICLAC v14 by Cellosaurus accession: 13 of 476 graph lines (9 Table 1, 4 Table 2, 2 in the test split), none of the 12 IG lines",
    (stats_clean.get("iclac_n_graph_lines"), stats_clean.get("iclac_n_graph_matched"),
     stats_clean.get("iclac_n_graph_table1"), stats_clean.get("iclac_n_graph_table2"),
     stats_clean.get("iclac_n_graph_test"), stats_clean.get("iclac_n_ig_lines"),
     stats_clean.get("iclac_n_ig_matched")) == (476, 13, 9, 4, 2, 12, 0),
    f"got graph {stats_clean.get('iclac_n_graph_matched')}/{stats_clean.get('iclac_n_graph_lines')}, "
    f"T1 {stats_clean.get('iclac_n_graph_table1')}, T2 {stats_clean.get('iclac_n_graph_table2')}, "
    f"test {stats_clean.get('iclac_n_graph_test')}, IG {stats_clean.get('iclac_n_ig_matched')}/{stats_clean.get('iclac_n_ig_lines')}",
)
# Sex descriptors (script 54), tied to the cohort sizes of script 21 so the SAGER
# statement cannot drift from Table 2. A label with zero patients has no key at all.
_check(
    "Sex: METABRIC Cox population all female (1,979), SCAN-B has no sex variable",
    stats_clean.get("demo_METABRIC_cox_sex_Female") == stats_clean.get("METABRIC_n_patients_cox") == 1979
    and "demo_METABRIC_cox_sex_Male" not in stats_clean
    and stats_clean.get("demo_SCANB_cox_sex_variable_available") == 0,
    f"got METABRIC female {stats_clean.get('demo_METABRIC_cox_sex_Female')}, "
    f"SCAN-B sex flag {stats_clean.get('demo_SCANB_cox_sex_variable_available')}",
)
_check(
    "Sex: TCGA-BRCA 12 male patients in the survival cohort, none left after the PAM50 filter",
    stats_clean.get("demo_TCGA_survival_sex_male") == 12
    and (stats_clean.get("demo_TCGA_survival_sex_female") or 0) + (stats_clean.get("demo_TCGA_survival_sex_male") or 0)
        == stats_clean.get("TCGA_n_patients")
    and stats_clean.get("demo_TCGA_cox_sex_female") == stats_clean.get("TCGA_n_patients_cox") == 1060
    and "demo_TCGA_cox_sex_male" not in stats_clean,
    f"got survival male {stats_clean.get('demo_TCGA_survival_sex_male')}, "
    f"Cox female {stats_clean.get('demo_TCGA_cox_sex_female')}",
)
_check(
    "AURORA-US: 129 samples all female, 79 metastatic and 44 primary, 19 metastatic sites",
    (stats_clean.get("demo_AURORA_samples_n_samples"), stats_clean.get("demo_AURORA_samples_sex_Female"),
     stats_clean.get("demo_AURORA_samples_disease_Metastatic_tumor"),
     stats_clean.get("demo_AURORA_samples_disease_Primary_tumor"),
     stats_clean.get("demo_AURORA_metastatic_samples_n_distinct_sites")) == (129, 129, 79, 44, 19),
    f"got {stats_clean.get('demo_AURORA_samples_n_samples')} samples, "
    f"{stats_clean.get('demo_AURORA_samples_sex_Female')} female, "
    f"{stats_clean.get('demo_AURORA_metastatic_samples_n_distinct_sites')} sites",
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
    "Cross-screen PRISM->GDSC median per-drug Pearson = 0.052 (Discussion, Supplementary Figure 14)",
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
    "Enrichment: tubulin polymerization inhibitors 25/27 (93%, FDR 3.2e-7), 4 tubulin targets at FDR<0.05 (Supplementary Figures 8 and 9)",
    stats_clean.get("moa_enrichment_top_class") == "tubulin polymerization inhibitor"
    and (stats_clean.get("moa_enrichment_top_n_sig"), stats_clean.get("moa_enrichment_top_n_total")) == (25, 27)
    and 3.0e-7 <= (stats_clean.get("moa_enrichment_top_padj") or 0) <= 3.3e-7
    and stats_clean.get("moa_enrichment_n_classes_fdr05") == 1
    and set(stats_clean.get("target_enrichment_fdr05_targets") or []) == {"TUBB", "TUBB1", "TUBA4A", "TUBA1A"},
    f"got {stats_clean.get('moa_enrichment_top_class')} {stats_clean.get('moa_enrichment_top_n_sig')}/{stats_clean.get('moa_enrichment_top_n_total')} "
    f"padj {stats_clean.get('moa_enrichment_top_padj')}, classes {stats_clean.get('moa_enrichment_n_classes_fdr05')}, targets {stats_clean.get('target_enrichment_fdr05_targets')}",
)
_check(
    "Supplementary Figure 13 legend: 430 drugs at nominal p<0.05 and 22 after Bonferroni in the Basal+HER2 pool",
    (stats_clean.get("subtype_pooled_n_nominal_p05"), stats_clean.get("subtype_pooled_n_bonferroni")) == (430, 22),
    f"got {stats_clean.get('subtype_pooled_n_nominal_p05')}/{stats_clean.get('subtype_pooled_n_bonferroni')}",
)
_check(
    "Supplementary Figure 6: Spearman rho of -log10 p between SCAN-B and METABRIC = 0.198",
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
# Supplementary Table S2 and Supplementary Data 1) and its HRs must be biologically plausible.
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
    # 20, not 21: the class filter used to match 'kinase' inside 'urokinase'
    # (4-chlorophenylguanidine, rank 67, preclinical, so 16 and 7 do not change).
    _check(
        "Candidate rule counts match the manuscript (143 / 52 / 20 / 16 / 7)",
        (stats_clean.get("candidates_n_statistical_filter"),
         stats_clean.get("candidates_n_stable_pen05"),
         stats_clean.get("candidates_n_kinase_hdac"),
         stats_clean.get("candidates_n_extended_set"),
         stats_clean.get("candidates_n_selected")) == (143, 52, 20, 16, 7),
        f"got {[stats_clean.get(k) for k in ('candidates_n_statistical_filter', 'candidates_n_stable_pen05', 'candidates_n_kinase_hdac', 'candidates_n_extended_set', 'candidates_n_selected')]}",
    )
    # Table 5 printed brigatinib 3.5e-3 and entinostat 1.2e-2 at lambda 0.5 by
    # rounding 3.45e-3 and 1.25e-2 a second time.
    _check(
        "Table 5 p_adj at lambda 0.5 rounded once (brigatinib 3.4e-3, entinostat 1.3e-2)",
        (stats_clean.get("candidate_brigatinib_padj_pen05"),
         stats_clean.get("candidate_entinostat_padj_pen05")) == (3.4e-3, 1.3e-2),
        f"got {stats_clean.get('candidate_brigatinib_padj_pen05')}, "
        f"{stats_clean.get('candidate_entinostat_padj_pen05')}",
    )
    if "ig_extended_global_top20_overlap" in stats_clean:
        _check(
            "Supplementary Note 1: extended candidate set keeps 19 of the global top-20 TFs",
            stats_clean.get("ig_extended_global_top20_overlap") == 19,
            f"got {stats_clean.get('ig_extended_global_top20_overlap')}",
        )
        _check(
            "Supplementary Note 1: extended candidate set shares the 10 core TFs",
            stats_clean.get("ig_extended_core10_overlap", 0) == 10,
            f"got {stats_clean.get('ig_extended_core10_overlap')}",
        )
        _check(
            "Supplementary Note 1: global IG profiles Spearman >= 0.99 between candidate sets",
            stats_clean.get("ig_extended_global_spearman_all_tfs", 0) >= 0.99,
            f"got {stats_clean.get('ig_extended_global_spearman_all_tfs')}",
        )
        _check(
            "Supplementary Note 1: AURORA Bonferroni-significant core TFs equal in both runs (5)",
            stats_clean.get("ig_extended_aurora_extended_bonferroni") == 5
            and stats_clean.get("ig_extended_aurora_main_bonferroni") == 5,
            f"got main={stats_clean.get('ig_extended_aurora_main_bonferroni')} "
            f"extended={stats_clean.get('ig_extended_aurora_extended_bonferroni')}",
        )
    _check(
        "Candidate rule cutoff sensitivity matches Supplementary Table S2 (26:7, 35:10, 50:14)",
        (stats_clean.get("candidates_n_at_cutoff_26"),
         stats_clean.get("candidates_n_at_cutoff_35"),
         stats_clean.get("candidates_n_at_cutoff_50")) == (7, 10, 14),
        f"got {[stats_clean.get(f'candidates_n_at_cutoff_{n}') for n in (26, 35, 50)]}",
    )

# Within-subtype screens must match the manuscript (Results paragraph, Supplementary Figure 13 caption)
if "subtype_pooled_drugs_FDR005" in stats_clean:
    _check(
        "Within-subtype: Basal alone n/events/FDR<0.05 == 686/167/117",
        (stats_clean.get("subtype_basal_n_patients"), stats_clean.get("subtype_basal_n_events"),
         stats_clean.get("subtype_basal_drugs_FDR005")) == (686, 167, 117),
        f"got {[stats_clean.get(k) for k in ('subtype_basal_n_patients', 'subtype_basal_n_events', 'subtype_basal_drugs_FDR005')]}",
    )
    _check(
        "Within-subtype: Basal+HER2 pool n/events/FDR<0.05 == 1361/314/221",
        (stats_clean.get("subtype_pooled_n_patients"), stats_clean.get("subtype_pooled_n_events"),
         stats_clean.get("subtype_pooled_drugs_FDR005")) == (1361, 314, 221),
        f"got {[stats_clean.get(k) for k in ('subtype_pooled_n_patients', 'subtype_pooled_n_events', 'subtype_pooled_drugs_FDR005')]}",
    )

# Training history (Methods: 800 epochs, SFig04), sensitivity analyses (Results, Discussion)
if "training_epochs" in stats_clean:
    _check(
        "Training history of the canonical model: 800 epochs, final test Pearson in [0.74, 0.77]",
        stats_clean.get("training_epochs") == 800
        and 0.74 <= stats_clean.get("training_final_test_pearson", 0) <= 0.77,
        f"got epochs={stats_clean.get('training_epochs')} pearson={stats_clean.get('training_final_test_pearson')}",
    )
if "rfs_n_fdr05" in stats_clean:
    # 12 drugs and BH over 12 since talazoparib joined script 40 (September 2026). With
    # 11 drugs it was 11/11 concordant and 5 at FDR < 0.05, osimertinib and saracatinib
    # at 0.052, which the larger family moves just below 0.05.
    _check(
        "RFS sensitivity: 12/12 direction-concordant and 8 drugs FDR<0.05, osimertinib 0.0497 and saracatinib "
        "0.0488 (Sensitivity analyses)",
        (stats_clean.get("rfs_n_drugs"), stats_clean.get("rfs_n_direction_concordant"), stats_clean.get("rfs_n_fdr05"))
        == (12, 12, 8)
        and stats_clean.get("rfs_fdr05_drugs") == ["osimertinib", "saracatinib", "erlotinib", "brigatinib",
                                                   "entinostat", "docetaxel", "epirubicin", "talazoparib"]
        and (stats_clean.get("rfs_osimertinib_padj"), stats_clean.get("rfs_saracatinib_padj")) == (0.0497, 0.0488),
        f"got concordant={stats_clean.get('rfs_n_direction_concordant')} fdr05={stats_clean.get('rfs_fdr05_drugs')} "
        f"osimertinib {stats_clean.get('rfs_osimertinib_padj')} saracatinib {stats_clean.get('rfs_saracatinib_padj')}",
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
        # Values changed in September 2026 when script 37 started encoding ER and PR
        # correctly: before, both were constant and silently dropped, which gave 5 candidates
        # (HR 1.9-3.4) and brigatinib alone in METABRIC. Talazoparib joined the family the
        # same month (12 drugs, BH over 12): the candidates keep HR 2.26-3.72, talazoparib is
        # the fourth control at FDR < 0.05 in SCAN-B, and the padj move slightly (trametinib
        # 0.0813 to 0.081, METABRIC brigatinib 0.0039 to 0.0043 and pelitinib 0.028 to 0.031).
        "Clinical-covariate Cox (12 drugs): 6 candidates FDR<0.05 in SCAN-B (HR 2.26-3.72) with paclitaxel, docetaxel, "
        "epirubicin and talazoparib, trametinib 0.081, brigatinib (0.0043) and pelitinib (0.031) in METABRIC (Results)",
        stats_clean.get("clinical_adj_n_drugs") == 12
        and stats_clean.get("clinical_adj_scanb_n_candidates_fdr05") == 6
        and (stats_clean.get("clinical_adj_scanb_hr_min"), stats_clean.get("clinical_adj_scanb_hr_max")) == (2.26, 3.72)
        and stats_clean.get("clinical_adj_scanb_controls_fdr05") == ["paclitaxel", "docetaxel", "epirubicin", "talazoparib"]
        and stats_clean.get("clinical_adj_scanb_trametinib_padj") == 0.081
        and stats_clean.get("clinical_adj_metabric_candidates_fdr05") == ["brigatinib", "pelitinib"]
        and (stats_clean.get("clinical_adj_metabric_brigatinib_padj"),
             stats_clean.get("clinical_adj_metabric_pelitinib_padj")) == (0.0043, 0.031)
        and stats_clean.get("clinical_adj_metabric_controls_fdr05") == [],
        f"got scanb n={stats_clean.get('clinical_adj_scanb_n_candidates_fdr05')} "
        f"HR=[{stats_clean.get('clinical_adj_scanb_hr_min')}, {stats_clean.get('clinical_adj_scanb_hr_max')}] "
        f"controls={stats_clean.get('clinical_adj_scanb_controls_fdr05')} "
        f"metabric={stats_clean.get('clinical_adj_metabric_candidates_fdr05')}",
    )
if "aurora_site_n_sites" in stats_clean:
    _check(
        "AURORA by site: 7 sites, KLF8 concordant in every site (Results)",
        stats_clean.get("aurora_site_n_sites") == 7
        and stats_clean.get("aurora_site_fully_concordant_tfs") == ["KLF8"],
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
    stats_clean["aurora_drugs_n_tested"] = int(len(df_ad))
    stats_clean["aurora_drugs_sig_pct"] = round(100 * n_sig / len(df_ad), 1)
    stats_clean["aurora_drugs_more_effective_pct"] = round(
        100 * stats_clean["aurora_drugs_more_effective_meta"] / n_sig, 1)
    ad_idx = df_ad.set_index("drug")["pvalue"]
    for d in ("entinostat", "paclitaxel", "panobinostat"):
        stats_clean[f"aurora_drug_{d}_p"] = float(f"{ad_idx[d]:.2g}")
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
    stats_clean["pca_pc1_survival_z"] = float(pca.get("pc1_survival_z"))
    _check(
        "PC1 survival association below the 'p < 0.001' quoted in Results (unrounded p, Wald z)",
        stats_clean["pca_pc1_survival_pvalue"] < 1e-3 and abs(stats_clean["pca_pc1_survival_z"]) > 3.29,
        f"p={stats_clean['pca_pc1_survival_pvalue']} z={stats_clean['pca_pc1_survival_z']}",
    )
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

# Cox penalizer sweep (Supplementary Figure 11 + Discussion HR-range claims, script 45)
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
    # HR range among the FDR-significant drugs (Supplementary Figure 11b), which
    # at penalizer 1.0 is narrower than the range over every tested drug.
    for pen, tag in ((0.01, "pen001"), (0.5, "pen05"), (1.0, "pen10")):
        row = _prow(pen)
        stats_clean[f"pensweep_hr_min_sig_{tag}"] = round(float(row["hr_min_fdr"]), 4)
        stats_clean[f"pensweep_hr_max_sig_{tag}"] = round(float(row["hr_max_fdr"]), 2)
    # The seven candidates across the sweep (Discussion).
    psd = pd.read_csv(RESULTS / "v8_multicohort" / "penalizer_sweep_perdrug.csv")
    cands = stats_clean["candidates_selected"]
    for pen, tag in ((0.01, "pen001"), (0.1, "pen01"), (0.5, "pen05"), (1.0, "pen10")):
        sub = psd[((psd["penalizer"] - pen).abs() < 1e-6) & psd["drug"].isin(cands)]
        sig = sorted(sub.loc[sub["padj"] < 0.05, "drug"])
        stats_clean[f"pensweep_candidates_fdr05_{tag}"] = len(sig)
        stats_clean[f"pensweep_candidates_fdr05_{tag}_names"] = sig
    _check(
        "The seven candidates stay FDR-significant at penalizers 0.01, 0.1 and 0.5, only erlotinib at 1.0 (Discussion)",
        [stats_clean[f"pensweep_candidates_fdr05_{t}"] for t in ("pen001", "pen01", "pen05")] == [7, 7, 7]
        and stats_clean["pensweep_candidates_fdr05_pen10_names"] == ["erlotinib"],
        f"got {[stats_clean[f'pensweep_candidates_fdr05_{t}'] for t in ('pen001', 'pen01', 'pen05', 'pen10')]}",
    )
    _check(
        "HR range of FDR-significant drugs at penalizer 1.0 is [0.31, 1.58] (Supplementary Figure 11b)",
        (round(stats_clean["pensweep_hr_min_sig_pen10"], 2), round(stats_clean["pensweep_hr_max_sig_pen10"], 2))
        == (0.31, 1.58),
        f"got {stats_clean['pensweep_hr_min_sig_pen10']}, {stats_clean['pensweep_hr_max_sig_pen10']}",
    )

# Checks added with the September 2026 audit ---------------------------------
_t1_models = ("RF_TF", "Ridge_expression", "SAGE_learned_emb", "HGT_learned_emb", "Ridge_expr_GNN_emb")
# Since the search of script 72 (October 2026) the random forest row is also a mean over
# five seeds, so only Ridge is a single fit.
_check(
    "Table 1: every model evaluated on the same 1,447 drugs, random forest, GNN and hybrid rows over 5 seeds, "
    "Ridge fitted once",
    all(stats_clean.get(f"table1_{m}_n_drugs") == 1447 for m in _t1_models)
    and all(stats_clean.get(f"table1_{m}_n_seeds") == 5 for m in _t1_models if m != "Ridge_expression")
    and stats_clean.get("table1_Ridge_expression_n_seeds") == 1,
    f"got {[(m, stats_clean.get(f'table1_{m}_n_drugs'), stats_clean.get(f'table1_{m}_n_seeds')) for m in _t1_models]}",
)
_check(
    "Graph ablation: 5 seeds per configuration, full graph = Table 1 SAGE mean",
    all(v == 5 for k, v in stats_clean.items() if k.startswith("ablation_") and k.endswith("_n_seeds"))
    and stats_clean.get("ablation_Full graph_pearson") == stats_clean.get("table1_SAGE_learned_emb_global_pearson"),
    f"full {stats_clean.get('ablation_Full graph_pearson')} vs Table 1 {stats_clean.get('table1_SAGE_learned_emb_global_pearson')}",
)
_check(
    "Cell-line features = TF activities + 500 top-variable genes (1,271 = 771 + 500)",
    stats_clean.get("cell_line_n_features") == stats_clean.get("prism_n_tf_features", 0) + 500 == 1271,
    f"got {stats_clean.get('cell_line_n_features')} and {stats_clean.get('prism_n_tf_features')}",
)
_check(
    "METABRIC matrix 6,549 genes covering 6,519 of the 6,675 snapshot targets (Methods)",
    (stats_clean.get("metabric_n_genes_matrix"), stats_clean.get("metabric_n_collectri_targets_covered"),
     stats_clean.get("collectri_snapshot_n_targets")) == (6549, 6519, 6675)
    and stats_clean.get("collectri_snapshot_n_targets") == stats_clean.get("collectri_target_genes"),
    f"got {stats_clean.get('metabric_n_genes_matrix')}, {stats_clean.get('metabric_n_collectri_targets_covered')}",
)
_check(
    "Breast cell lines of the graph: 22 (12 TNBC, 4 ER+, 6 HER2+), TNBC = the IG lines",
    (stats_clean.get("breast_n_cell_lines"), stats_clean.get("breast_n_tnbc"), stats_clean.get("breast_n_erpos"),
     stats_clean.get("breast_n_her2pos")) == (22, 12, 4, 6)
    and stats_clean.get("breast_n_tnbc") == stats_clean.get("ig_n_cells"),
    f"got {[stats_clean.get(k) for k in ('breast_n_cell_lines', 'breast_n_tnbc', 'breast_n_erpos', 'breast_n_her2pos')]}",
)
_check(
    "Entinostat is the only HDAC inhibitor passing filters (i) and (ii) (Results)",
    stats_clean.get("candidates_hdac_stable") == ["entinostat"],
    f"got {stats_clean.get('candidates_hdac_stable')}",
)
_check(
    "Kaplan-Meier panels use the survival cohort (script 58 n = SCANB_n_patients)",
    (stats_clean.get("SCANB_km_n_patients"), stats_clean.get("SCANB_km_n_events"))
    == (stats_clean.get("SCANB_n_patients"), stats_clean.get("SCANB_n_events")),
    f"got {stats_clean.get('SCANB_km_n_patients')}/{stats_clean.get('SCANB_km_n_events')}",
)
_check(
    "Basal + HER2 pool = Basal + HER2-enriched patients of the Cox cohort (same patient rule as script 21)",
    stats_clean.get("subtype_pooled_n_patients")
    == stats_clean.get("demo_SCANB_cox_pam50_Basal", 0) + stats_clean.get("demo_SCANB_cox_pam50_Her2", 0),
    f"got {stats_clean.get('subtype_pooled_n_patients')}",
)
_t4 = pd.read_csv(RESULTS / "v9_aurora" / "aurora_tf_ig_validation.csv")
_check(
    "The ten TFs of Table 4 are the core set (lowest mean IG rank across seeds)",
    set(_t4["tf"]) == set(stats_clean.get("ig_multi_seed_core10_names", [])),
    f"Table 4 {sorted(_t4['tf'])} vs core {stats_clean.get('ig_multi_seed_core10_names')}",
)
_g32 = pd.read_csv(RESULTS / "v4_xai" / "tf_importance_ig_global.csv", comment="#")
_g35 = pd.read_csv(RESULTS / "v4_xai" / "tf_importance_ig_seed42.csv", comment="#")
_check(
    "IG of script 32 and seed 42 of script 35 come from the same canonical model (identical top 20)",
    _g32.nsmallest(20, "rank")["tf"].tolist() == _g35.nsmallest(20, "rank")["tf"].tolist(),
    "the Figure 3A and 3C rankings come from different checkpoints",
)
_snap = {k[len("collectri_snap_"):]: v for k, v in stats_clean.items() if k.startswith("collectri_snap_")}
_check(
    "Supplementary Note 2 counts (script 56): TFs 125/119/137 and 26/30/39, drugs 604/87/0/558, 59 of 59 concordant",
    [_snap.get(f"{c}_n_tfs_changed") for c in ("SCANB", "METABRIC", "TCGA")] == [125, 119, 137]
    and [_snap.get(f"{c}_n_tfs_r099") for c in ("SCANB", "METABRIC", "TCGA")] == [26, 30, 39]
    and [_snap.get(f"{c}_drugs_FDR005") for c in ("SCANB", "METABRIC", "TCGA", "Fisher")] == [604, 87, 0, 558]
    and [_snap.get(f"{c}_drugs_FDR005_shared") for c in ("SCANB", "METABRIC", "Fisher")] == [574, 71, 525]
    and (_snap.get("shared_drugs_both_cohorts"), _snap.get("shared_drugs_direction_consistent")) == (59, 59),
    f"got {_snap}",
)
_check(
    "Supplementary Note 2 rule: 154/56/18/15/7, indirubin (HR 8.8) replaces trametinib (rank 35)",
    [_snap.get(f"candidates_n_{s}") for s in ("statistical_filter", "stable_pen05", "kinase_hdac", "extended_set", "selected")]
    == [154, 56, 18, 15, 7]
    and _snap.get("candidates_added") == ["indirubin"] and _snap.get("candidates_dropped") == ["trametinib"]
    and round(_snap.get("added_indirubin_hr", 0), 1) == 8.8 and _snap.get("dropped_trametinib_rank") == 35
    and _snap.get("positive_control_docetaxel_fisher_padj", 1) < 0.05
    and _snap.get("positive_control_olaparib_fisher_padj", 0) >= 0.05,
    f"got {_snap}",
)

# Checks of the September 2026 re-audit (scripts 59 to 64, 42 and 58) -----------
# (1) The cross-checks collected in the re-audit block: every key against the
# summary its script wrote, and the scripts against each other and against the
# tables they were built from.
for _name, _cond, _detail in REAUDIT_XCHECKS:
    _check(_name, _cond, _detail)

# (2) Cross-checks with the keys and tables of the rest of the paper.
_s = stats_clean
_check(
    "Table 1 reference rows: drug mean 0.764 with no per-drug metric, transfer Ridge 0.700 / 0.086 / 6.2%, "
    "1,447 drugs, flagged in Row_type after the five models",
    (_s.get("table1_Drug_mean_reference_global_pearson"), _s.get("table1_Drug_mean_reference_median_perdrug"),
     _s.get("table1_Drug_mean_reference_pct_gt_03"), _s.get("table1_Drug_mean_reference_row_type")) == (0.764, None, None, "reference")
    and (_s.get("table1_Transfer_Ridge_TF_global_pearson"), _s.get("table1_Transfer_Ridge_TF_median_perdrug"),
         _s.get("table1_Transfer_Ridge_TF_pct_gt_03"), _s.get("table1_Transfer_Ridge_TF_row_type")) == (0.7, 0.086, 6.2, "reference")
    and _s.get("table1_Drug_mean_reference_n_drugs") == _s.get("table1_Transfer_Ridge_TF_n_drugs")
    == _s.get("transfer_heldout_n_drugs") == 1447
    and all(_s.get(f"table1_{m}_row_type") == "model" for m in _t1_models)
    and pd.read_csv(OUT / "table1_model_comparison.csv")["Row_type"].tolist() == ["model"] * 5 + ["reference"] * 2,
    f"got drug mean {_s.get('table1_Drug_mean_reference_global_pearson')}, transfer "
    f"{_s.get('table1_Transfer_Ridge_TF_global_pearson')}/{_s.get('table1_Transfer_Ridge_TF_median_perdrug')}",
)
_n_scanb_drugs = _s.get("SCANB_n_drugs_tested", 0)
_check(
    "Re-audit cohorts = the cohorts of Table 2 and Figure 4 (SCAN-B Cox 7,215 / 1,164, Kaplan-Meier 7,397 / 1,188, "
    "METABRIC 1,979 / 1,143, Basal 686 / 167, AURORA 39 pairs, chemotherapy yes + no + unknown = cohort)",
    (_s["pc1_orient_n_patients"], _s["pc1_orient_n_events"]) == (_s["SCANB_n_patients_cox"], _s["SCANB_n_events_cox"])
    and (_s["ph_cox_n_patients"], _s["ph_cox_n_events"]) == (_s["SCANB_n_patients_cox"], _s["SCANB_n_events_cox"])
    and (_s["km_extra_pam50_n_patients"], _s["km_extra_pam50_n_events"])
    == (_s["SCANB_n_patients_cox"], _s["SCANB_n_events_cox"])
    and (_s["km_extra_n_patients"], _s["km_extra_n_events"]) == (_s["SCANB_km_n_patients"], _s["SCANB_km_n_events"])
    and (_s["dss_metabric_n_patients"], _s["dss_metabric_n_deaths"])
    == (_s["METABRIC_n_patients_cox"], _s["METABRIC_n_events_cox"])
    and _s["dss_metabric_n_dss_deaths"] + _s["dss_metabric_n_other_cause_deaths"]
    + _s["dss_metabric_n_deaths_unknown_cause_censored"] == _s["dss_metabric_n_deaths"]
    and (_s["basal_n_patients"], _s["basal_n_events"]) == (_s["subtype_basal_n_patients"], _s["subtype_basal_n_events"])
    and _s["treat_scanb_chemo_yes"] + _s["treat_scanb_chemo_no"] + _s["treat_scanb_chemo_missing"]
    == _s["SCANB_n_patients_cox"]
    and _s["treat_metabric_chemo_yes"] + _s["treat_metabric_chemo_no"] + _s["treat_metabric_chemo_missing"]
    == _s["METABRIC_n_patients_cox"]
    and _s["treat_scanb_chemo_interaction_n_patients"] == _s["treat_scanb_chemo_yes"] + _s["treat_scanb_chemo_no"]
    and _s["aurora_bg_n_paired_patients"] == _s["aurora_drug_paired_n_patients"] == _s["aurora_cohort_n_paired_patients"]
    and _s["pc1_orient_n_drug_pairs"] == _n_scanb_drugs * (_n_scanb_drugs - 1) // 2,
    "a re-audit cohort differs from the cohort it reanalyses",
)
_check(
    "Re-audit counts = survival screen: 623 SCAN-B, 74 METABRIC, 551 Fisher (502 + 49), 632 Stouffer (502 + 130), "
    "54 shared with 53 concordant, 1,447 drugs",
    _s["direction_scanb_sig_n"] == _s["hrsd_fdr_n_drugs"] == _s["SCANB_drugs_FDR005"]
    and _s["dss_metabric_os_n_sig"] == _s["METABRIC_drugs_FDR005"]
    and _s["stouffer_fisher_overlap"] + _s["stouffer_fisher_sig_lost"] == _s["Fisher_drugs_FDR005"]
    and _s["stouffer_fisher_overlap"] + _s["stouffer_sig_not_fisher"] == _s["stouffer_n_sig"]
    and _s["stouffer_n_sig_hr_gt1"] + _s["stouffer_n_sig_hr_lt1"] == _s["stouffer_n_sig"]
    and (_s["direction_both_sig_n"], _s["direction_both_sig_n_same"])
    == (_s["shared_drugs_both_cohorts"], _s["shared_drugs_direction_consistent"])
    and _s["direction_all_n"] == _s["stouffer_n_drugs"] == _n_scanb_drugs == 1447,
    "a re-audit count differs from the screen of Table 2",
)
_check(
    "AURORA background agrees with Table 4, script 39 and script 22: 5 core TFs at p < 0.005, 7 sites, "
    "782 / 633 / 149 drugs at sample level",
    _s["aurora_bg_core_n_p_lt_0005"] == _s["aurora_bonferroni_sig"]
    and _s["aurora_site_bg_n_sites"] == _s["aurora_site_n_sites"]
    and [aur_sum["drug_sample_n_p005"], aur_sum["drug_sample_n_more_active_p005"], aur_sum["drug_sample_n_less_active_p005"]]
    == [_s.get("aurora_drugs_sig"), _s.get("aurora_drugs_more_effective_meta"), _s.get("aurora_drugs_less_effective_meta")],
    f"core {_s['aurora_bg_core_n_p_lt_0005']} vs {_s.get('aurora_bonferroni_sig')}, sample-level drugs "
    f"{aur_sum['drug_sample_n_p005']} vs {_s.get('aurora_drugs_sig')}",
)
_check(
    "Penalizer 0.01: the largest hazard ratio (245.5) is that of the penalizer sweep (script 45)",
    _s["hrsd_pen001_max_hr"] == _s.get("pensweep_hr_max_pen001"),
    f"got {_s['hrsd_pen001_max_hr']} vs {_s.get('pensweep_hr_max_pen001')}",
)
_t5c = pd.read_csv(OUT / "table5_candidates.csv").set_index("Drug")
_check(
    "Table 5 HR/SD (1.11 to 1.14) and r_test (0.04 to 0.50) columns = hrsd_<drug> and candidate_<drug>_heldout_r",
    list(_t5c.columns) == ["MOA", "Phase", "HR_PAM50adj", "CI95_PAM50adj", "padj_PAM50adj", "HR_per_SD",
                           "HR_pen05", "padj_pen05", "r_test"]
    and all(float(_t5c.at[d, "HR_per_SD"]) == _s[f"hrsd_{d}"]
            and float(_t5c.at[d, "r_test"]) == _s[f"candidate_{d}_heldout_r"] for d in CANDIDATES),
    f"got {_t5c[['HR_per_SD', 'r_test']].to_dict()}",
)
_t3c = pd.read_csv(OUT / "table3_positive_controls.csv", dtype=str).set_index("Drug")
_check(
    "Table 3 Stouffer_padj column = stouffer_<drug>_padj, two significant digits (paclitaxel 7.0e-02)",
    all(_t3c.at[d, "Stouffer_padj"] == f"{_s[f'stouffer_{d}_padj']:.1e}" for d in positive_controls),
    f"got {_t3c['Stouffer_padj'].to_dict()}",
)
# The labels are the numbers of the PDF, the file names keep their historical ones
for _path, _rows, _label in ((OUT / "supp_table_S3_robustness.csv", s3_rows, "S3"),
                             (OUT / "supp_table_S4_replication.csv", s4_rows, "S1")):
    _tab = pd.read_csv(_path, dtype=str, keep_default_na=False)
    _check(
        f"Supplementary Table {_label}: 12 rows (the 7 candidates in Table 5 order, then the 5 controls), "
        "no missing value, every cell = its re-audit key",
        len(_tab) == 12 and _tab["Drug"].tolist() == _S_ORDER
        and _tab["Group"].tolist() == ["candidate"] * 7 + ["positive_control"] * 5
        and not (_tab == "").any().any()
        and _tab.equals(pd.DataFrame(_rows).astype(str)),
        f"got {len(_tab)} rows, order {_tab['Drug'].tolist()}",
    )
_s4_cols = pd.read_csv(OUT / "supp_table_S4_replication.csv", nrows=0).columns.tolist()
_check(
    "Supplementary Table S1 columns: METABRIC OS and DSS, Stouffer, SCAN-B chemotherapy interaction with PAM50 strata "
    "(HR, p, padj, HR per SD untreated and treated) and clinically adjusted (HR, padj), METABRIC treatment-adjusted",
    _s4_cols == ["Drug", "Group", "METABRIC_HR_OS", "METABRIC_CI95_OS", "METABRIC_padj_OS", "METABRIC_HR_DSS",
                 "METABRIC_padj_DSS", "Stouffer_padj", "SCANB_chemo_interaction_HR", "SCANB_chemo_interaction_p",
                 "SCANB_chemo_interaction_padj", "SCANB_HR_per_SD_untreated", "SCANB_HR_per_SD_treated",
                 "SCANB_chemo_interaction_HR_adjusted", "SCANB_chemo_interaction_padj_adjusted",
                 "METABRIC_HR_per_SD_treatment_adjusted", "METABRIC_padj_treatment_adjusted"]
    and sum(float(v) < 0.05 for v in supp_s4["SCANB_chemo_interaction_padj"]) == _s["treat_scanb_chemo_interaction_n_padj05"]
    and sum(float(v) < 0.05 for v in supp_s4["SCANB_chemo_interaction_padj_adjusted"])
    == _s["treat_scanb_chemo_interaction_clinical_endo_n_padj05"],
    f"got {_s4_cols}",
)

# (3) Frozen values. Each re-audit key is pinned to the value the manuscript and
# the supplementary tables print, so any material change in results/v13_reaudit/
# fails here. Per-drug values are listed in the order of TWELVE_DRUGS (the seven
# candidates, then the five controls), Kaplan-Meier values in the order of Figure 4
# (paclitaxel, epirubicin, erlotinib, entinostat).
_FROZEN_KEYS = set()


def _expand(patterns, drugs):
    out = {}
    for pattern, values in patterns.items():
        if len(values) != len(drugs):
            raise AssertionError(f"{pattern}: {len(values)} frozen values for {len(drugs)} drugs")
        for d, v in zip(drugs, values):
            out[pattern.format(d=d, role=_role(d))] = v
    return out


def _check_frozen(name, expected):
    _FROZEN_KEYS.update(expected)
    diff = {k: (stats_clean.get(k, "<missing>"), v) for k, v in expected.items()
            if k not in stats_clean or stats_clean[k] != v or isinstance(stats_clean[k], bool) != isinstance(v, bool)}
    _check(name, not diff, f"{len(diff)} keys differ (got, expected): {dict(list(diff.items())[:10])}")


_KM_ORDER = ["paclitaxel", "epirubicin", "erlotinib", "entinostat"]
_check_frozen(
    "Re-audit keys of script 59 frozen: between-drug variance 0.59, 122,784 test pairs, held-out r of the"
    " candidates 0.06 (erlotinib) to 0.50 (trametinib)",
    {
        "ref_drug_mean_between_drug_variance_share": 0.59,
        "ref_n_test_pairs": 122784,
        "transfer_heldout_n_drugs": 1447,
        **_expand({
            "{role}_{d}_heldout_r": (0.06, 0.1, 0.29, 0.36, 0.47, 0.5, 0.04, 0.22, 0.27, 0.34, 0.14, 0.46),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of script 42 frozen: oriented PC1 vs E2F1 -0.55, MYC -0.37, TP53 0.45, ESR1 0.37, "
    "talazoparib AUC 0.76, PC2 11%, mean |r| 0.16, 34% negative pairs",
    {
        "pc1_orient_n_patients": 7215,
        "pc1_orient_n_events": 1164,
        "pc1_orient_sign_flipped": False,
        "pc1_orient_pc2_explained_var_pct": 11,
        "pc1_orient_corr_E2F1": -0.55,
        "pc1_orient_corr_MYC": -0.37,
        "pc1_orient_corr_TP53": 0.45,
        "pc1_orient_corr_ESR1": 0.37,
        "pc1_orient_corr_paclitaxel_auc": 0.34,
        "pc1_orient_corr_talazoparib_auc": 0.76,
        "pc1_orient_corr_olaparib_auc": 0.51,
        "pc1_orient_survival_z": -6.04,
        "pc1_orient_inter_drug_abs_corr_mean": 0.16,
        "pc1_orient_pct_drug_pairs_negative": 34,
        "pc1_orient_n_drug_pairs": 1046181,
    },
)
_check_frozen(
    "Re-audit keys of script 58 frozen: 10-year OS difference -4.8 / -8.8 / 5.3 / 7.5 points, "
    "PAM50-stratified p 7.0e-3 / 1.9e-4 / 2.1e-2 / 4.9e-7, trend p paclitaxel 8.1e-8 and entinostat "
    "4.9e-12, 747 patients at risk at 10 years",
    {
        "km_extra_n_patients": 7397,
        "km_extra_n_events": 1188,
        "km_extra_pam50_n_patients": 7215,
        "km_extra_pam50_n_events": 1164,
        "km_at_risk_10y_total": 747,
        "km_at_risk_step_years": 2,
        **_expand({
            "km_{d}_os10_sensitive_pct": (74.7, 72.7, 79.7, 80.9),
            "km_{d}_os10_resistant_pct": (79.5, 81.6, 74.4, 73.4),
            "km_{d}_os10_diff_pp": (-4.8, -8.8, 5.3, 7.5),
            "km_{d}_pam50_strat_p": (0.007, 0.00019, 0.021, 4.9e-07),
            "km_{d}_trend_p": (8.1e-08, 8.6e-18, 9.3e-15, 4.9e-12),
            "km_{d}_trend_hazard_rises_with_auc": (False, False, True, True),
            "km_{d}_pct_luma_sensitive": (41.5, 31.4, 66.6, 59.1),
            "km_{d}_pct_luma_resistant": (57.5, 67.6, 32.2, 39.7),
            "km_{d}_pct_basal_sensitive": (15.0, 16.5, 3.4, 5.0),
            "km_{d}_pct_basal_resistant": (4.1, 2.5, 15.7, 14.0),
        }, _KM_ORDER),
    },
)
_check_frozen(
    "Re-audit keys of script 63 frozen: direction 1,011/1,447 (69.9% vs 51.9%), Stouffer 632 (502 with "
    "Fisher, 49 lost, 75 opposite), controls P = 0.073 (ranks 82 to 682), METABRIC DSS 184 vs 74, 646 / "
    "496 deaths",
    {
        "direction_all_n": 1447,
        "direction_all_n_same": 1011,
        "direction_all_pct_same": 69.9,
        "direction_all_pct_expected": 51.9,
        "direction_scanb_sig_n": 623,
        "direction_scanb_sig_n_same": 504,
        "direction_scanb_sig_pct_same": 80.9,
        "direction_scanb_sig_pct_expected": 56.4,
        "direction_both_sig_n": 54,
        "direction_both_sig_n_same": 53,
        "direction_both_sig_pct_same": 98.1,
        "direction_both_sig_pct_expected": 64.4,
        "stouffer_n_drugs": 1447,
        "stouffer_n_sig": 632,
        "stouffer_pct_sig": 43.7,
        "stouffer_n_sig_hr_gt1": 176,
        "stouffer_n_sig_hr_lt1": 456,
        "stouffer_fisher_overlap": 502,
        "stouffer_fisher_sig_lost": 49,
        "stouffer_sig_not_fisher": 130,
        "stouffer_fisher_sig_opposite_direction": 75,
        "stouffer_fisher_sig_opposite_lost": 48,
        "stouffer_controls_fdr05": ["docetaxel", "epirubicin", "talazoparib"],
        "stouffer_candidates_n_fdr05": 7,
        "controls_baserate_fisher_frac_sig": 0.381,
        "controls_baserate_fisher_n_sig": 4,
        "controls_baserate_fisher_p_at_least_observed": 0.073,
        "controls_baserate_fisher_paclitaxel_rank": 361,
        "controls_baserate_fisher_docetaxel_rank": 487,
        "controls_baserate_fisher_epirubicin_rank": 82,
        "controls_baserate_fisher_olaparib_rank": 682,
        "controls_baserate_fisher_talazoparib_rank": 158,
        "controls_baserate_fisher_rank_min": 82,
        "controls_baserate_fisher_rank_max": 682,
        "controls_baserate_stouffer_frac_sig": 0.437,
        "controls_baserate_stouffer_n_sig": 3,
        "controls_baserate_stouffer_p_at_least_observed": 0.38,
        "controls_baserate_stouffer_paclitaxel_rank": 678,
        "controls_baserate_stouffer_docetaxel_rank": 536,
        "controls_baserate_stouffer_epirubicin_rank": 51,
        "controls_baserate_stouffer_olaparib_rank": 652,
        "controls_baserate_stouffer_talazoparib_rank": 251,
        "controls_baserate_stouffer_rank_min": 51,
        "controls_baserate_stouffer_rank_max": 678,
        "dss_metabric_n_patients": 1979,
        "dss_metabric_n_deaths": 1143,
        "dss_metabric_n_dss_deaths": 646,
        "dss_metabric_n_other_cause_deaths": 496,
        "dss_metabric_n_deaths_unknown_cause_censored": 1,
        "dss_metabric_pct_deaths_other_causes": 43.4,
        "dss_metabric_n_sig": 184,
        "dss_metabric_os_n_sig": 74,
        "dss_metabric_os_dss_overlap": 30,
        "dss_metabric_n_sig_hr_gt1": 23,
        **_expand({
            "stouffer_{d}_padj": (5.6e-07, 1.9e-09, 1.5e-05, 4.8e-06, 5.9e-07, 0.0013, 0.0023, 0.07, 0.023, 2.6e-07, 0.057, 0.00051),
            "dss_{d}_hr": (2.04, 3.34, 1.98, 1.95, 1.85, 0.85, 0.88, 0.82, 0.84, 0.09, 0.51, 0.55),
            "dss_{d}_padj": (0.2, 0.034, 0.41, 0.34, 0.33, 0.68, 0.88, 0.72, 0.74, 0.026, 0.09, 0.17),
            "metabric_os_{d}_hr": (2.13, 4.4, 2.31, 2.9, 3.08, 1.18, 1.22, 1.25, 0.98, 0.12, 0.85, 0.93),
            "metabric_os_{d}_ci_low": (1.17, 2.3, 1.02, 1.42, 1.61, 0.83, 0.66, 0.77, 0.64, 0.04, 0.55, 0.57),
            "metabric_os_{d}_ci_high": (3.87, 8.44, 5.25, 5.92, 5.88, 1.67, 2.24, 2.03, 1.5, 0.42, 1.33, 1.51),
            "metabric_os_{d}_padj": (0.12, 0.0015, 0.22, 0.056, 0.022, 0.6, 0.74, 0.61, 0.97, 0.025, 0.7, 0.89),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of script 60 frozen: 231 of 770 TFs at p < 0.005 (30.0%), 3.0 expected, P = 0.15, core"
    " ranks 69 to 704, sites 136 / 517 and 47% / 96%, paired drugs 711 (597 / 114), entinostat p = 6.2e-3",
    {
        "aurora_bg_n_paired_patients": 39,
        "aurora_bg_tf_n_tested": 770,
        "aurora_bg_tf_n_p_lt_0005": 231,
        "aurora_bonferroni_threshold_core10": 0.005,
        "tf_activity_min_targets": 5,
        "aurora_bg_tf_pct_p_lt_0005": 30.0,
        "aurora_bg_tf_n_p_lt_005": 367,
        "aurora_bg_tf_n_bh_005": 298,
        "aurora_bg_tf_n_bonferroni_005": 77,
        "aurora_bg_core_n_p_lt_0005": 5,
        "aurora_bg_core_expected_p_lt_0005": 3.0,
        "aurora_bg_core_hypergeom_p": 0.15,
        "aurora_bg_core_rank_KLF8": 69,
        "aurora_bg_core_rank_STAT2": 117,
        "aurora_bg_core_rank_AEBP1": 137,
        "aurora_bg_core_rank_SPDEF": 185,
        "aurora_bg_core_rank_SETBP1": 189,
        "aurora_bg_core_rank_ZEB1": 381,
        "aurora_bg_core_rank_MZF1": 424,
        "aurora_bg_core_rank_RELA": 469,
        "aurora_bg_core_rank_IRX1": 562,
        "aurora_bg_core_rank_BARX1": 704,
        "aurora_bg_core_rank_min": 69,
        "aurora_bg_core_rank_max": 704,
        "aurora_site_bg_n_sites": 7,
        "aurora_site_bg_n_tfs": 770,
        "aurora_site_bg_n_concordant_all_sites": 136,
        "aurora_site_bg_n_concordant_ge5": 517,
        "aurora_site_bg_sig_n_tfs": 231,
        "aurora_site_bg_sig_n_concordant_all_sites": 108,
        "aurora_site_bg_sig_n_concordant_ge5": 221,
        "aurora_site_bg_sig_pct_concordant_all_sites": 47,
        "aurora_site_bg_sig_pct_concordant_ge5": 96,
        "aurora_drug_paired_n_patients": 39,
        "aurora_drug_paired_n_tested": 1447,
        "aurora_drug_paired_n_bh005": 711,
        "aurora_drug_paired_n_bh005_more_active": 597,
        "aurora_drug_paired_n_bh005_less_active": 114,
        "aurora_drug_paired_entinostat_p": 0.0062,
        "aurora_drug_paired_paclitaxel_p": 0.33,
        "aurora_drug_paired_panobinostat_p": 0.88,
        "aurora_drug_paired_entinostat_less_active_in_metastasis": True,
    },
)
_check_frozen(
    "Re-audit keys of script 61 frozen: TNBC 676 / 183 (osimertinib 1.24, pelitinib 1.43), Basal 686 / "
    "167 (pelitinib, trametinib), 5 candidates less sensitive in Basal, LumA 45.7-66.7% vs 49.5%, Basal "
    "9.5%",
    {
        "tnbc_n_patients": 676,
        "tnbc_n_events": 183,
        "tnbc_candidates_fdr05": ["osimertinib", "pelitinib"],
        "tnbc_candidates_n_fdr05": 2,
        "tnbc_candidates_n_hr_gt1": 7,
        "tnbc_controls_n_fdr05": 0,
        "basal_n_patients": 686,
        "basal_n_events": 167,
        "basal_candidates_fdr05": ["pelitinib", "trametinib"],
        "basal_candidates_n_fdr05": 2,
        "basal_candidates_n_hr_gt1": 7,
        "basal_controls_n_fdr05": 0,
        "tnbc_n_tnbc_and_basal": 493,
        "tnbc_pct_tnbc_that_are_basal": 72.9,
        "tnbc_pct_basal_that_are_tnbc": 71.9,
        "pam50_cohort_pct_LumA": 49.5,
        "pam50_cohort_pct_LumB": 24.2,
        "pam50_cohort_pct_Her2": 9.4,
        "pam50_cohort_pct_Basal": 9.5,
        "pam50_cohort_pct_Normal": 7.4,
        "pam50_candidates_basal_less_sensitive": ["erlotinib", "brigatinib", "saracatinib", "trametinib", "entinostat"],
        "pam50_n_candidates_basal_less_sensitive": 5,
        "pam50_n_controls_basal_less_sensitive": 0,
        "pam50_sensitive_half_pct_luma_candidates_min": 45.7,
        "pam50_sensitive_half_pct_luma_candidates_max": 66.7,
        **_expand({
            "tnbc_{d}_hr_per_sd": (1.16, 1.09, 1.24, 1.12, 1.43, 1.19, 1.07, 1.03, 1.05, 1.04, 1.02, 1.02),
            "tnbc_{d}_ci_low": (1.01, 0.95, 1.08, 0.97, 1.25, 1.02, 0.94, 0.9, 0.92, 0.89, 0.9, 0.87),
            "tnbc_{d}_ci_high": (1.33, 1.25, 1.44, 1.29, 1.64, 1.4, 1.21, 1.19, 1.2, 1.21, 1.15, 1.18),
            "tnbc_{d}_p": (0.031, 0.23, 0.0032, 0.13, 2.7e-07, 0.03, 0.33, 0.67, 0.45, 0.63, 0.75, 0.84),
            "tnbc_{d}_padj": (0.054, 0.27, 0.011, 0.18, 1.9e-06, 0.054, 0.33, 0.84, 0.84, 0.84, 0.84, 0.84),
            "basal_{d}_hr_per_sd": (1.17, 1.18, 1.14, 1.07, 1.28, 1.3, 1.05, 1.03, 1.03, 1.15, 1.02, 1.01),
            "basal_{d}_ci_low": (1.01, 1.02, 0.98, 0.92, 1.12, 1.11, 0.92, 0.89, 0.9, 0.97, 0.91, 0.85),
            "basal_{d}_ci_high": (1.35, 1.36, 1.32, 1.24, 1.46, 1.53, 1.2, 1.19, 1.18, 1.36, 1.16, 1.19),
            "basal_{d}_p": (0.032, 0.025, 0.089, 0.41, 0.00021, 0.0015, 0.49, 0.74, 0.66, 0.12, 0.7, 0.92),
            "basal_{d}_padj": (0.056, 0.056, 0.12, 0.47, 0.0015, 0.0054, 0.49, 0.92, 0.92, 0.59, 0.92, 0.92),
            "pam50_sensitive_half_pct_luma_{d}": (66.7, 58.9, 53.0, 55.2, 45.7, 56.6, 59.1, 41.4, 42.8, 31.3, 40.5, 25.1),
            "pam50_sensitive_half_pct_basal_{d}": (3.4, 6.5, 10.4, 7.5, 14.5, 9.7, 5.0, 15.0, 13.3, 16.5, 14.4, 18.1),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of script 62 frozen: chemotherapy 2,802 / 4,041 / 372, 6,231 / 959 complete cases, "
    "candidates 1.01-1.17 per SD after treatment adjustment, entinostat 1.24 untreated and 0.93 treated, "
    "METABRIC 412 treated",
    {
        "treat_scanb_chemo_yes": 2802,
        "treat_scanb_chemo_no": 4041,
        "treat_scanb_chemo_missing": 372,
        "treat_metabric_chemo_yes": 412,
        "treat_metabric_chemo_no": 1567,
        "treat_metabric_chemo_missing": 0,
        "treat_scanb_complete_n_patients": 6231,
        "treat_scanb_complete_n_events": 959,
        "treat_scanb_clinical_treatment_candidates_hr_min": 1.01,
        "treat_scanb_clinical_treatment_candidates_hr_max": 1.17,
        "treat_scanb_clinical_treatment_fdr05_drugs": ["brigatinib", "saracatinib", "talazoparib"],
        "treat_scanb_clinical_treatment_fdr05_candidates": ["brigatinib", "saracatinib"],
        "treat_metabric_complete_n_patients": 1903,
        "treat_metabric_complete_n_events": 1103,
        "treat_metabric_clinical_treatment_candidates_hr_min": 0.98,
        "treat_metabric_clinical_treatment_candidates_hr_max": 1.12,
        "treat_metabric_clinical_treatment_fdr05_drugs": ["brigatinib", "pelitinib"],
        "treat_metabric_clinical_treatment_fdr05_candidates": ["brigatinib", "pelitinib"],
        "treat_scanb_chemo_interaction_n_patients": 6843,
        "treat_scanb_chemo_interaction_n_events": 1069,
        "treat_scanb_chemo_interaction_n_p05": 6,
        "treat_scanb_chemo_interaction_n_padj05": 3,
        "treat_metabric_chemo_interaction_n_patients": 1979,
        "treat_metabric_chemo_interaction_n_events": 1143,
        "treat_metabric_chemo_interaction_n_p05": 0,
        "treat_metabric_chemo_interaction_n_padj05": 0,
        **_expand({
            "treat_scanb_pam50_strata_{d}_hr_per_sd": (1.19, 1.25, 1.2, 1.23, 1.24, 1.2, 1.16, 0.89, 0.9, 0.86, 0.94, 0.88),
            "treat_scanb_pam50_strata_{d}_ci_low": (1.11, 1.16, 1.12, 1.15, 1.16, 1.12, 1.08, 0.83, 0.84, 0.79, 0.88, 0.81),
            "treat_scanb_pam50_strata_{d}_ci_high": (1.28, 1.34, 1.28, 1.31, 1.33, 1.28, 1.24, 0.95, 0.96, 0.93, 1.0, 0.95),
            "treat_scanb_pam50_strata_{d}_p": (6.1e-07, 2e-09, 9.2e-08, 4.2e-09, 3.4e-10, 5.4e-08, 1.2e-05, 0.00086, 0.00091, 0.0002, 0.054, 0.002),
            "treat_scanb_pam50_strata_{d}_padj": (1.2e-06, 1.2e-08, 2.2e-07, 1.7e-08, 4e-09, 1.6e-07, 2.1e-05, 0.0011, 0.0011, 0.00031, 0.054, 0.0022),
            "treat_scanb_clinical_{d}_hr_per_sd": (1.06, 1.17, 1.05, 1.1, 1.08, 1.01, 1.07, 0.93, 0.95, 0.92, 0.97, 0.89),
            "treat_scanb_clinical_{d}_ci_low": (0.99, 1.08, 0.98, 1.03, 1.01, 0.95, 1.0, 0.87, 0.89, 0.85, 0.91, 0.82),
            "treat_scanb_clinical_{d}_ci_high": (1.14, 1.25, 1.12, 1.18, 1.16, 1.08, 1.15, 0.99, 1.01, 1.0, 1.04, 0.98),
            "treat_scanb_clinical_{d}_p": (0.085, 4e-05, 0.17, 0.0074, 0.022, 0.74, 0.045, 0.026, 0.095, 0.043, 0.38, 0.012),
            "treat_scanb_clinical_{d}_padj": (0.13, 0.00048, 0.2, 0.044, 0.062, 0.74, 0.077, 0.062, 0.13, 0.077, 0.42, 0.048),
            "treat_scanb_clinical_treatment_{d}_hr_per_sd": (1.06, 1.17, 1.05, 1.1, 1.08, 1.01, 1.07, 0.93, 0.95, 0.92, 0.97, 0.89),
            "treat_scanb_clinical_treatment_{d}_ci_low": (0.99, 1.08, 0.98, 1.03, 1.01, 0.95, 1.0, 0.87, 0.89, 0.85, 0.91, 0.82),
            "treat_scanb_clinical_treatment_{d}_ci_high": (1.14, 1.26, 1.12, 1.18, 1.16, 1.08, 1.15, 1.0, 1.01, 1.0, 1.04, 0.97),
            "treat_scanb_clinical_treatment_{d}_p": (0.093, 3.9e-05, 0.16, 0.0052, 0.024, 0.74, 0.036, 0.036, 0.12, 0.045, 0.42, 0.011),
            "treat_scanb_clinical_treatment_{d}_padj": (0.14, 0.00047, 0.19, 0.031, 0.072, 0.74, 0.072, 0.072, 0.16, 0.076, 0.46, 0.043),
            "treat_metabric_pam50_strata_{d}_hr_per_sd": (1.09, 1.15, 1.06, 1.1, 1.12, 1.01, 1.02, 1.02, 1.0, 0.88, 0.96, 1.0),
            "treat_metabric_pam50_strata_{d}_ci_low": (1.03, 1.08, 1.0, 1.03, 1.05, 0.95, 0.96, 0.96, 0.94, 0.82, 0.9, 0.93),
            "treat_metabric_pam50_strata_{d}_ci_high": (1.16, 1.22, 1.12, 1.17, 1.19, 1.07, 1.08, 1.09, 1.06, 0.94, 1.02, 1.08),
            "treat_metabric_pam50_strata_{d}_p": (0.0064, 8.6e-06, 0.054, 0.0037, 0.00052, 0.73, 0.55, 0.45, 0.99, 0.00029, 0.18, 0.97),
            "treat_metabric_pam50_strata_{d}_padj": (0.015, 0.0001, 0.11, 0.011, 0.0021, 0.88, 0.74, 0.67, 0.99, 0.0017, 0.31, 0.99),
            "treat_metabric_clinical_{d}_hr_per_sd": (1.08, 1.12, 1.05, 1.07, 1.1, 0.98, 0.98, 1.0, 0.99, 0.94, 0.98, 1.01),
            "treat_metabric_clinical_{d}_ci_low": (1.01, 1.06, 0.99, 1.0, 1.03, 0.93, 0.92, 0.94, 0.94, 0.88, 0.92, 0.94),
            "treat_metabric_clinical_{d}_ci_high": (1.16, 1.19, 1.11, 1.14, 1.17, 1.05, 1.04, 1.06, 1.06, 1.01, 1.04, 1.08),
            "treat_metabric_clinical_{d}_p": (0.017, 0.00025, 0.12, 0.043, 0.0055, 0.6, 0.52, 0.96, 0.86, 0.087, 0.55, 0.83),
            "treat_metabric_clinical_{d}_padj": (0.069, 0.0029, 0.24, 0.13, 0.033, 0.79, 0.79, 0.96, 0.94, 0.21, 0.79, 0.94),
            "treat_metabric_clinical_treatment_{d}_hr_per_sd": (1.08, 1.12, 1.04, 1.06, 1.09, 0.98, 0.98, 1.0, 0.99, 0.94, 0.98, 1.0),
            "treat_metabric_clinical_treatment_{d}_ci_low": (1.01, 1.05, 0.98, 1.0, 1.02, 0.93, 0.92, 0.94, 0.93, 0.88, 0.92, 0.93),
            "treat_metabric_clinical_treatment_{d}_ci_high": (1.16, 1.19, 1.11, 1.13, 1.17, 1.05, 1.04, 1.06, 1.05, 1.01, 1.04, 1.08),
            "treat_metabric_clinical_treatment_{d}_p": (0.019, 0.00031, 0.19, 0.065, 0.0072, 0.61, 0.42, 0.96, 0.76, 0.08, 0.44, 0.91),
            "treat_metabric_clinical_treatment_{d}_padj": (0.075, 0.0037, 0.37, 0.19, 0.043, 0.81, 0.66, 0.96, 0.92, 0.19, 0.66, 0.96),
            "treat_scanb_chemo_interaction_{d}_hr": (0.87, 0.86, 1.01, 0.87, 1.04, 0.92, 0.75, 1.02, 1.05, 1.37, 0.99, 1.3),
            "treat_scanb_chemo_interaction_{d}_p": (0.029, 0.026, 0.84, 0.045, 0.5, 0.23, 9.5e-06, 0.8, 0.45, 5.8e-06, 0.86, 0.00016),
            "treat_scanb_chemo_interaction_{d}_hr_untreated": (1.25, 1.29, 1.17, 1.24, 1.18, 1.19, 1.24, 0.89, 0.89, 0.76, 0.93, 0.74),
            "treat_scanb_chemo_interaction_{d}_hr_treated": (1.09, 1.11, 1.19, 1.08, 1.23, 1.1, 0.93, 0.91, 0.94, 1.04, 0.92, 0.96),
            "treat_metabric_chemo_interaction_{d}_hr": (1.0, 0.94, 1.03, 1.05, 1.07, 0.99, 1.01, 0.87, 0.87, 1.02, 0.99, 0.89),
            "treat_metabric_chemo_interaction_{d}_p": (0.95, 0.36, 0.75, 0.53, 0.39, 0.89, 0.92, 0.054, 0.075, 0.78, 0.88, 0.11),
            "treat_metabric_chemo_interaction_{d}_hr_untreated": (1.1, 1.18, 1.06, 1.09, 1.11, 1.03, 1.01, 1.05, 1.03, 0.88, 0.97, 1.03),
            "treat_metabric_chemo_interaction_{d}_hr_treated": (1.1, 1.11, 1.09, 1.14, 1.19, 1.01, 1.02, 0.91, 0.9, 0.9, 0.96, 0.91),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of script 64 frozen: exact PH test 6 drug terms, HER2 and LumA in 12/12 models, none "
    "stratified, stratified candidates 1.14-1.26 per SD, canonical 1.11-1.14, FDR-significant 0.81-1.17 "
    "per SD",
    {
        "ph_cox_n_patients": 7215,
        "ph_cox_n_events": 1164,
        "ph_exact_drug_terms_violating": ["entinostat", "epirubicin", "erlotinib", "olaparib", "paclitaxel", "talazoparib"],
        "ph_exact_n_drug_terms_violating": 6,
        "ph_exact_sub_Her2_n_models_violating": 12,
        "ph_exact_sub_Her2_p_max": 0.02,
        "ph_exact_sub_LumA_n_models_violating": 12,
        "ph_exact_sub_LumA_p_max": 7.9e-05,
        "ph_exact_sub_LumB_n_models_violating": 0,
        "ph_exact_sub_LumB_p_max": 0.088,
        "ph_exact_sub_Normal_n_models_violating": 0,
        "ph_exact_sub_Normal_p_max": 0.96,
        "ph_lifelines_drug_terms_violating": ["erlotinib", "olaparib", "talazoparib"],
        "ph_lifelines_n_drug_terms_violating": 3,
        "ph_lifelines_sub_Her2_n_models_violating": 0,
        "ph_lifelines_sub_Her2_p_max": 0.74,
        "ph_lifelines_sub_LumA_n_models_violating": 12,
        "ph_lifelines_sub_LumA_p_max": 0.0057,
        "ph_lifelines_sub_LumB_n_models_violating": 12,
        "ph_lifelines_sub_LumB_p_max": 0.0018,
        "ph_lifelines_sub_Normal_n_models_violating": 0,
        "ph_lifelines_sub_Normal_p_max": 0.4,
        "ph_strat_pen_n_drug_terms_violating": 0,
        "ph_strat_unpen_n_drug_terms_violating": 0,
        "ph_strat_pen_p_min": 0.11,
        "ph_strat_unpen_p_min": 0.13,
        "strat_candidates_hr_per_sd_min": 1.14,
        "strat_candidates_hr_per_sd_max": 1.26,
        "strat_n_p05": 11,
        "strat_candidates_n_p05": 7,
        "hrsd_candidates_min": 1.11,
        "hrsd_candidates_max": 1.14,
        "hrsd_candidates_sd_min": 0.06,
        "hrsd_candidates_sd_max": 0.15,
        "hrsd_fdr_n_drugs": 623,
        "hrsd_fdr_hr_per_sd_min": 0.81,
        "hrsd_fdr_hr_per_sd_max": 1.17,
        "hrsd_fdr_hr_per_unit_min": 0.0087,
        "hrsd_fdr_hr_per_unit_max": 54.4,
        "hrsd_pen001_max_hr_drug": "wp1066",
        "hrsd_pen001_max_hr": 245.5,
        "hrsd_pen001_max_hr_sd_percentile": 0.48,
        **_expand({
            "ph_{d}_p_exact": (0.00055, 0.13, 0.38, 0.43, 0.17, 0.75, 0.025, 0.035, 0.22, 0.0092, 0.0025, 1.2e-07),
            "strat_{d}_hr_per_sd": (1.17, 1.2, 1.22, 1.19, 1.26, 1.19, 1.14, 0.9, 0.91, 0.88, 0.94, 0.87),
            "strat_{d}_ci_low": (1.1, 1.12, 1.15, 1.12, 1.19, 1.12, 1.07, 0.85, 0.86, 0.82, 0.89, 0.81),
            "strat_{d}_ci_high": (1.25, 1.28, 1.29, 1.27, 1.34, 1.26, 1.21, 0.96, 0.96, 0.94, 1.0, 0.94),
            "strat_{d}_p": (9.7e-07, 2.8e-08, 9.5e-11, 2.5e-08, 6.7e-14, 9.9e-09, 1.9e-05, 0.00062, 0.0013, 0.00046, 0.063, 0.00033),
            "hrsd_{d}": (1.14, 1.13, 1.12, 1.12, 1.11, 1.11, 1.11, 0.92, 0.92, 0.89, 0.94, 0.88),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of script 64 in METABRIC frozen: exact PH test 6 of 12 drug terms in the canonical and the "
    "unpenalized model, 36 of 48 PAM50 terms (HER2, LumA and LumB in 12/12), 3 drug terms once PAM50 is a "
    "stratum",
    {
        "ph_metabric_cox_n_patients": 1979,
        "ph_metabric_cox_n_events": 1143,
        "ph_metabric_n_pam50_terms": 48,
        "ph_metabric_exact_drug_terms_violating": ["docetaxel", "entinostat", "epirubicin", "erlotinib",
                                                   "pelitinib", "talazoparib"],
        "ph_metabric_exact_n_drug_terms_violating": 6,
        "ph_metabric_exact_n_pam50_terms_violating": 36,
        "ph_metabric_exact_sub_Her2_n_models_violating": 12,
        "ph_metabric_exact_sub_LumA_n_models_violating": 12,
        "ph_metabric_exact_sub_LumB_n_models_violating": 12,
        "ph_metabric_exact_sub_Normal_n_models_violating": 0,
        "ph_metabric_exact_unpenalized_drug_terms_violating": ["docetaxel", "entinostat", "epirubicin", "erlotinib",
                                                               "pelitinib", "talazoparib"],
        "ph_metabric_exact_unpenalized_n_drug_terms_violating": 6,
        "ph_metabric_exact_unpenalized_n_pam50_terms_violating": 36,
        "ph_metabric_exact_unpenalized_sub_Her2_n_models_violating": 12,
        "ph_metabric_exact_unpenalized_sub_LumA_n_models_violating": 12,
        "ph_metabric_exact_unpenalized_sub_LumB_n_models_violating": 12,
        "ph_metabric_exact_unpenalized_sub_Normal_n_models_violating": 0,
        "ph_metabric_strat_pen_drug_terms_violating": ["docetaxel", "paclitaxel", "talazoparib"],
        "ph_metabric_strat_pen_n_drug_terms_violating": 3,
        "ph_metabric_strat_unpen_drug_terms_violating": ["docetaxel", "paclitaxel", "talazoparib"],
        "ph_metabric_strat_unpen_n_drug_terms_violating": 3,
    },
)
_check_frozen(
    "Re-audit keys of script 62 frozen, interaction padj and clinically adjusted interaction: PAM50 strata padj "
    "5.7e-5 (entinostat, epirubicin), 6.5e-4 (talazoparib), 0.069 (erlotinib, brigatinib), 0.089 (saracatinib), "
    "clinical_endo 6,173 / 946 SCAN-B and 1,903 / 1,103 METABRIC patients with no padj < 0.05 (smallest 0.26 and "
    "0.41), 5 candidates with a larger HR per SD untreated (not osimertinib, pelitinib)",
    {
        "treat_scanb_chemo_interaction_fdr05_drugs": ["entinostat", "epirubicin", "talazoparib"],
        "treat_metabric_chemo_interaction_fdr05_drugs": [],
        "treat_scanb_chemo_interaction_clinical_endo_n_patients": 6173,
        "treat_scanb_chemo_interaction_clinical_endo_n_events": 946,
        "treat_scanb_chemo_interaction_clinical_endo_n_p05": 2,
        "treat_scanb_chemo_interaction_clinical_endo_n_padj05": 0,
        "treat_scanb_chemo_interaction_clinical_endo_padj_min": 0.26,
        "treat_metabric_chemo_interaction_clinical_endo_n_patients": 1903,
        "treat_metabric_chemo_interaction_clinical_endo_n_events": 1103,
        "treat_metabric_chemo_interaction_clinical_endo_n_p05": 0,
        "treat_metabric_chemo_interaction_clinical_endo_n_padj05": 0,
        "treat_metabric_chemo_interaction_clinical_endo_padj_min": 0.41,
        "treat_scanb_chemo_interaction_n_candidates_hr_untreated_gt_treated": 5,
        "treat_scanb_chemo_interaction_candidates_hr_untreated_not_gt_treated": ["osimertinib", "pelitinib"],
        **_expand({
            "treat_scanb_chemo_interaction_{d}_padj": (0.069, 0.069, 0.86, 0.089, 0.67, 0.4, 5.7e-05, 0.86, 0.67, 5.7e-05, 0.86, 0.00065),
            "treat_metabric_chemo_interaction_{d}_padj": (0.95, 0.93, 0.95, 0.95, 0.93, 0.95, 0.95, 0.45, 0.45, 0.95, 0.95, 0.45),
            "treat_scanb_chemo_interaction_clinical_endo_{d}_hr": (1.03, 1.09, 1.1, 1.0, 1.14, 1.07, 0.86, 0.94, 0.98, 1.11, 0.87, 1.05),
            "treat_scanb_chemo_interaction_clinical_endo_{d}_p": (0.64, 0.23, 0.18, 0.97, 0.066, 0.33, 0.034, 0.37, 0.82, 0.18, 0.046, 0.5),
            "treat_scanb_chemo_interaction_clinical_endo_{d}_padj": (0.77, 0.47, 0.43, 0.97, 0.26, 0.56, 0.26, 0.56, 0.89, 0.43, 0.26, 0.66),
            "treat_scanb_chemo_interaction_clinical_endo_{d}_hr_untreated": (1.06, 1.14, 1.02, 1.09, 1.04, 1.0, 1.12, 0.94, 0.95, 0.89, 1.01, 0.88),
            "treat_scanb_chemo_interaction_clinical_endo_{d}_hr_treated": (1.09, 1.24, 1.12, 1.09, 1.19, 1.07, 0.96, 0.88, 0.94, 0.98, 0.88, 0.93),
            "treat_metabric_chemo_interaction_clinical_endo_{d}_hr": (1.01, 0.99, 1.0, 1.13, 1.07, 0.99, 1.06, 0.87, 0.86, 0.95, 1.02, 0.91),
            "treat_metabric_chemo_interaction_clinical_endo_{d}_p": (0.94, 0.84, 0.99, 0.1, 0.44, 0.87, 0.48, 0.075, 0.053, 0.5, 0.79, 0.2),
            "treat_metabric_chemo_interaction_clinical_endo_{d}_padj": (0.99, 0.99, 0.99, 0.41, 0.86, 0.99, 0.86, 0.41, 0.41, 0.86, 0.99, 0.59),
            "treat_metabric_chemo_interaction_clinical_endo_{d}_hr_untreated": (1.08, 1.12, 1.04, 1.04, 1.08, 0.99, 0.97, 1.03, 1.02, 0.95, 0.97, 1.03),
            "treat_metabric_chemo_interaction_clinical_endo_{d}_hr_treated": (1.09, 1.11, 1.04, 1.17, 1.15, 0.97, 1.02, 0.9, 0.87, 0.9, 0.99, 0.93),
        }, TWELVE_DRUGS),
    },
)
_check_frozen(
    "Re-audit keys of scripts 63 and 64 frozen: 28 of the 632 Stouffer-significant drugs with opposite SCAN-B and "
    "METABRIC directions, METABRIC DSS of the candidates 1 at FDR < 0.05 (brigatinib) and 5 with HR > 1, penalized "
    "stratified candidates 1.08-1.15 per SD, window [1.5, 10] per unit = 1.03 to 1.42 per SD for the candidates, "
    "1.06-1.17 per SD over the 143 drugs of the statistical filter",
    {
        "stouffer_n_sig_opposite_direction": 28,
        "dss_candidates_fdr05": ["brigatinib"],
        "dss_candidates_n_fdr05": 1,
        "dss_candidates_n_hr_gt1": 5,
        "strat_pen_candidates_hr_per_sd_min": 1.08,
        "strat_pen_candidates_hr_per_sd_max": 1.15,
        "window_candidates_per_sd_low_min": 1.03,
        "window_candidates_per_sd_high_max": 1.42,
        "window_filter_hr_per_sd_min": 1.06,
        "window_filter_hr_per_sd_max": 1.17,
    },
)

_check_frozen(
    "Re-audit keys of script 65 frozen: 77 of 220 Figure 3C cells with attribution sign != gradient sign (7 TFs, "
    "ZEB1, RUNX2, ARID1A, HOXB9, HIVEP1 with positive gradient, ZNF362 and LHX1 negative), cross-drug r 0.968 (min "
    "0.836), IG lines 8 train / 4 test, 44 TFs in any top 20, TF share 0.262, seed-42 medians 0.036 / 0.049 / 0.111, "
    "7 of the 10 core TFs in the seed-42 top 20 and KLF8, STAT2 and SETBP1 outside it at ranks 24, 37 and 39",
    {
        "igdiag_n_cells": 220,
        "igdiag_n_cells_attr_sign_ne_grad_sign": 77,
        "igdiag_n_cells_disagree_negative_mean_dx": 77,
        "igdiag_n_tfs_disagree": 7,
        "igdiag_n_tfs_disagree_all_drugs": 7,
        "igdiag_tfs_disagree": ["ZEB1", "ZNF362", "RUNX2", "HOXB9", "LHX1", "ARID1A", "HIVEP1"],
        "igdiag_tfs_disagree_grad_positive": ["ZEB1", "RUNX2", "HOXB9", "ARID1A", "HIVEP1"],
        "igdiag_tfs_disagree_grad_negative": ["ZNF362", "LHX1"],
        "igdiag_cross_drug_mean_r": 0.968,
        "igdiag_cross_drug_min_r": 0.836,
        "igdiag_cross_drug_n_pairs": 55,
        "igdiag_n_ig_lines_train": 8,
        "igdiag_n_ig_lines_test": 4,
        "igdiag_n_tfs_top20_any_seed": 44,
        "igdiag_share_abs_attr_tf_dims": 0.262,
        "igdiag_seed42_median_perdrug_SAGE": 0.036,
        "igdiag_seed42_median_perdrug_HGT": 0.049,
        "igdiag_seed42_median_perdrug_Ridge_expr_GNN_emb": 0.111,
        "igdiag_expected_top20_overlap_random": 0.52,
        "ig_core10_in_seed42_top20_n": 7,
        "ig_core10_outside_seed42_top20_names": ["KLF8", "STAT2", "SETBP1"],
        "ig_core10_outside_seed42_top20_rank42": [24, 37, 39],
    },
)
_check_frozen(
    "Re-audit keys of script 66 frozen: BY 391 / 15 / 0 / 338, 464 HR < 1 and 159 HR > 1, 34 Fisher-only drugs, "
    "678 / 623 / 87 at penalizer 0.01 / 0.1 / 1.0 (HR 0.0087 to 54.4 at 0.1), ablation medians 0.032 vs 0.037-0.052 "
    "(full below response-only in 3 of 5 seeds), rewired drug-target edges 0.741 keeping 86% of the gain, PAM50 39 of 50, GDSC 343 lines, AURORA epirubicin p = 0.0172",
    {
        "mult_by_scanb_fdr05": 391,
        "mult_by_metabric_fdr05": 15,
        "mult_by_tcga_fdr05": 0,
        "mult_by_fisher_fdr05": 338,
        "mult_scanb_fdr05_n_hr_lt1": 464,
        "mult_scanb_fdr05_n_hr_gt1": 159,
        "mult_fisher_fdr05_no_single_cohort_n": 34,
        "mult_fisher_fdr05_no_single_cohort_drugs": [
            "afatinib", "alisertib", "amg-232", "av-412", "bay-11-7082", "belinostat", "cephalomannine",
            "cobicistat", "defactinib", "dovitinib", "ecamsule-triethanolamine", "fleroxacin", "gw-501516",
            "idoxuridine", "jib04", "kd-023", "lgx818", "ly456236", "mk-8245", "monensin", "nomegestrol-acetate",
            "oxonic-acid", "pevonedistat", "phenelzine", "ramifenazone", "rheochrysidin", "salinomycin",
            "sdz-wag-994", "spiradoline", "tamibarotene", "tyloxapol", "venetoclax", "vidarabine", "xl-647"],
        "pensweep_n_fdr_pen001": 678,
        "pensweep_n_fdr_pen01": 623,
        "pensweep_n_fdr_pen10": 87,
        "pensweep_hr_min_sig_pen01": 0.0087,
        "pensweep_hr_max_sig_pen01": 54.4,
        "ablation_median_perdrug_full_mean": 0.032,
        "ablation_median_perdrug_full_sd": 0.009,
        "ablation_median_perdrug_no_ppi_mean": 0.042,
        "ablation_median_perdrug_no_ppi_sd": 0.015,
        "ablation_median_perdrug_no_collectri_mean": 0.047,
        "ablation_median_perdrug_no_collectri_sd": 0.014,
        "ablation_median_perdrug_no_drug_target_mean": 0.037,
        "ablation_median_perdrug_no_drug_target_sd": 0.009,
        "ablation_median_perdrug_no_ppi_no_drug_target_mean": 0.052,
        "ablation_median_perdrug_no_ppi_no_drug_target_sd": 0.01,
        "ablation_median_perdrug_response_only_mean": 0.04,
        "ablation_median_perdrug_response_only_sd": 0.02,
        "ablation_median_perdrug_shuffled_drug_target_mean": 0.033,
        "ablation_median_perdrug_shuffled_drug_target_sd": 0.007,
        "ablation_shuffled_global_mean": 0.741,
        "ablation_shuffled_global_sd": 0.001,
        "ablation_shuffled_gain_kept_pct": 86,
        "ablation_shuffled_n_seeds_above_no_drug_target": 5,
        "ablation_shuffled_edges_kept_pct_min": 1.6,
        "ablation_shuffled_edges_kept_pct_max": 1.9,
        "ablation_median_perdrug_reduced_min": 0.037,
        "ablation_median_perdrug_reduced_max": 0.052,
        "ablation_n_seeds_full_below_response_only_median": 3,
        "ablation_n_seeds_full_below_response_only_global": 0,
        "metabric_pam50_n_genes": 50,
        "metabric_pam50_n_genes_present": 39,
        "metabric_pam50_missing_genes": ["ACTR3B", "ANLN", "CENPF", "GPR160", "GRB7", "MELK", "MLPH", "NUF2",
                                         "ORC6", "TMEM45B", "UBE2T"],
        "gdsc_crossval_n_cell_lines": 343,
        "gdsc_crossval_n_cell_lines_with_prism_train_edges": 343,
        "gdsc_crossval_n_drugs_with_prism_train_edges": 187,
        "aurora_drug_epirubicin_p": 0.0172,
    },
)
# Changed with the search of script 72 (October 2026, punto 3.16): Ridge alpha 730 -> 506 and
# RF 200 / 19 / 9 -> 150 / 8 / 13, the values of best_params_*.json. The hp_ridge_rf_search_*
# keys now describe that search: 0 Table 1 test lines instead of 95 (n_test_lines and
# test_lines_in_table1_test), same_test_lines_as_table1 False instead of True and
# has_validation_split True instead of False, because each trial is scored by 5-fold CV inside
# the train lines. The development search and the Table 1 rows it gave move to hp_previous_*.
_check_frozen(
    "Hyperparameter keys frozen: Ridge alpha 506 (Table 1) and 100 (hybrid and transfer), RF 150 / 8 / 13, SAGE "
    "128 / 3 / 0.33, HGT 128 / 2 / 4 / 0.2, lr 0.0013, 800 epochs, Ridge and RF searched by 5-fold CV inside the "
    "381 train lines with no Table 1 test line (CV 0.1622 +/- 0.0155 and 0.1794 +/- 0.0287, Ridge alpha 506.2 "
    "rounded, edge diagnostic at 10^3.5 0.1769, space of scripts/06), previously 730 and 200 / 19 / 9 chosen on the "
    "95 test lines "
    "(Table 1 0.728 / 0.114 / 10.6% and 0.769 / 0.128 / 14.2%), GNN search on an edge-level split that uses all 95",
    {
        "hp_ridge_expression_alpha": 506,
        "hp_rf_tf_n_estimators": 150,
        "hp_rf_tf_max_depth": 8,
        "hp_rf_tf_min_samples_leaf": 13,
        "hp_ridge_rf_search_n_folds": 5,
        "hp_ridge_rf_search_n_train_lines": 381,
        "hp_ridge_rf_search_n_objective_drugs": 200,
        "hp_ridge_rf_search_sampler_seed": 42,
        "hp_ridge_rf_search_patients_split_across_folds": 0,
        # added 4-nov-2026: the Table 1 hold-out is drawn by line, not by donor
        "split_n_lines_without_donor": 0,
        "split_n_test_lines_sharing_donor_with_train": 2,
        "hp_ridge_rf_search_plan_commit": "3d539a0",
        "hp_ridge_expression_best_trial": 1,
        "hp_ridge_expression_cv_mean": 0.1622,
        "hp_ridge_expression_cv_sd": 0.0155,
        "hp_ridge_expression_at_edge": True,
        # 506.16 until 31-oct-2026, now with the four significant figures of the supplementary table
        "hp_ridge_expression_alpha_best_trial": 506.2,
        "hp_ridge_rf_search_ridge_alpha_min": 0.001,
        "hp_ridge_rf_search_ridge_alpha_max": 1000.0,
        "hp_ridge_rf_search_rf_n_estimators_min": 50,
        "hp_ridge_rf_search_rf_n_estimators_max": 300,
        "hp_ridge_rf_search_rf_n_estimators_step": 50,
        "hp_ridge_rf_search_rf_max_depth_min": 3,
        "hp_ridge_rf_search_rf_max_depth_max": 20,
        "hp_ridge_rf_search_rf_min_samples_leaf_min": 2,
        "hp_ridge_rf_search_rf_min_samples_leaf_max": 20,
        "hp_ridge_expression_edge_alpha_log10": 3.5,
        "hp_ridge_expression_edge_cv_mean": 0.1769,
        "hp_ridge_expression_edge_cv_sd": 0.0211,
        "hp_rf_tf_best_trial": 6,
        "hp_rf_tf_cv_mean": 0.1794,
        "hp_rf_tf_cv_sd": 0.0287,
        "hp_rf_tf_at_edge": False,
        "hp_previous_ridge_expression_alpha": 730,
        "hp_previous_rf_tf_n_estimators": 200,
        "hp_previous_rf_tf_max_depth": 19,
        "hp_previous_rf_tf_min_samples_leaf": 9,
        "hp_previous_ridge_expression_global_pearson": 0.728,
        "hp_previous_ridge_expression_median_perdrug": 0.114,
        "hp_previous_ridge_expression_pct_gt_03": 10.6,
        "hp_previous_rf_tf_global_pearson": 0.769,
        "hp_previous_rf_tf_median_perdrug": 0.128,
        "hp_previous_rf_tf_pct_gt_03": 14.2,
        "hp_previous_search_n_trials": 10,
        "hp_previous_search_n_test_lines": 95,
        "hp_previous_search_test_lines_in_table1_test": 95,
        "hp_previous_search_has_validation_split": False,
        "hp_hybrid_ridge_alpha": 100,
        "hp_transfer_ridge_alpha": 100,
        "hp_gnn_lr": 0.0013,
        "hp_gnn_weight_decay": 0.0001,
        "hp_gnn_epochs": 800,
        "hp_gnn_eta_min": 1e-05,
        "hp_sage_hidden_dim": 128,
        "hp_sage_n_layers": 3,
        "hp_sage_dropout": 0.33,
        "hp_hgt_hidden_dim": 128,
        "hp_hgt_n_layers": 2,
        "hp_hgt_n_heads": 4,
        "hp_hgt_dropout": 0.2,
        "hp_ridge_rf_search_n_trials": 10,
        "hp_ridge_rf_search_n_test_lines": 0,
        "hp_ridge_rf_search_test_lines_in_table1_test": 0,
        "hp_ridge_rf_search_same_test_lines_as_table1": False,
        "hp_ridge_rf_search_has_validation_split": True,
        "hp_gnn_search_n_trials": 35,
        "hp_gnn_search_edge_level_split": True,
        "hp_gnn_search_has_validation_split": False,
        "hp_gnn_search_table1_test_lines_with_training_edges": 95,
        "hp_gnn_search_table1_test_lines_with_scoring_edges": 95,
        "hp_gnn_search_scoring_edges_from_table1_test_lines_pct": 20.0,
        "hp_hgt_from_search": False,
    },
)
_check_frozen(
    "Derived re-audit keys frozen: ComBat 256 only QT / 267 only ComBat / 362 concordant, all 7 candidates "
    "significant (paclitaxel 0.24, docetaxel 0.063, talazoparib 0.17), candidate ranks 14-26 of 143 and 85-161 "
    "of 1,447, METABRIC replication 7 with HR > 1 (brigatinib, pelitinib), TF ranks MYC 262 / E2F1 267 / TP53 "
    "295 of 769, no candidate among the 200 residualized drugs",
    {
        "combat_qt_n_only_qt": 256,
        "combat_qt_n_only_combat": 267,
        "combat_qt_n_both_same_direction": 362,
        "combat_candidates_n_sig": 7,
        "combat_candidates_all_sig": True,
        "combat_candidates_not_sig": [],
        "combat_controls_fdr05": ["epirubicin", "olaparib"],
        "combat_controls_n_sig": 2,
        **_expand({
            "combat_{d}_padj": (4.2e-07, 0.00043, 1.2e-07, 0.0001, 9.6e-06, 5.7e-05, 4.8e-07,
                                0.24, 0.063, 0.026, 0.025, 0.17),
        }, TWELVE_DRUGS),
        **_expand({
            "candrank_{d}_in_filter": (14, 17, 20, 21, 23, 24, 26),
            "candrank_{d}_in_screen": (85, 96, 127, 132, 146, 148, 161),
            "metabric_repl_{d}_hr": (2.13, 4.4, 2.31, 2.9, 3.08, 1.18, 1.22),
            "metabric_repl_{d}_p": (0.013, 8e-06, 0.045, 0.0035, 0.00065, 0.36, 0.53),
            "metabric_repl_{d}_padj": (0.12, 0.0015, 0.22, 0.056, 0.022, 0.6, 0.74),
        }, CANDIDATES),
        "candrank_filter_min": 14,
        "candrank_filter_max": 26,
        "candrank_screen_min": 85,
        "candrank_screen_max": 161,
        "candrank_n_drugs_filter": 143,
        "candrank_n_drugs_screen": 1447,
        "metabric_repl_n_hr_above1": 7,
        "metabric_repl_fdr05_candidates": ["brigatinib", "pelitinib"],
        "metabric_repl_n_fdr05": 2,
        "tfcox_n_tfs": 769,
        "tfcox_MYC_rank": 262,
        "tfcox_E2F1_rank": 267,
        "tfcox_TP53_rank": 295,
        "tfcox_GATA3_rank": 7,
        "tfcox_top5_names": ["TOX3", "NCOA6", "BRD4", "ZNF804A", "HIC1"],
        "pca_perdrug_n_tested": 200,
        "pca_perdrug_candidates_present": [],
        "pca_perdrug_n_candidates_present": 0,
        "pca_perdrug_has_hazard_ratios": False,
    },
)
_check_frozen(
    "Platinum salts and IG baseline runs frozen: SCAN-B padj cisplatin 0.65, carboplatin 0.86, oxaliplatin 0.33, "
    "satraplatin 0.36, none significant in SCAN-B or in the Fisher meta-analysis, 3 random Gaussian baselines in "
    "script 33",
    {
        **_expand({
            "platinum_{d}_scanb_hr": (0.89, 0.96, 0.77, 0.83),
            "platinum_{d}_scanb_padj": (0.65, 0.86, 0.33, 0.36),
            "platinum_{d}_fisher_padj": (0.75, 0.63, 0.45, 0.84),
        }, PLATINUM),
        "platinum_n_scanb_fdr05": 0,
        "platinum_n_fisher_fdr05": 0,
        "ig_baseline_n_random_runs": 3,
    },
)
_check_frozen(
    "Fisher rerun of script 21 (chi2.sf) frozen: no p-value equal to 0, pardoprunox p 6.3e-21 and padj 9.1e-18, "
    "551 drugs at FDR < 0.05 and the five controls unchanged",
    {
        "fisher_n_p_equal_zero": 0,
        "fisher_n_padj_equal_zero": 0,
        "fisher_min_p_drug": "pardoprunox",
        "fisher_pardoprunox_p": 6.3e-21,
        "fisher_pardoprunox_padj": 9.1e-18,
        "Fisher_drugs_FDR005": 551,
        "positive_control_paclitaxel_fisher_padj": "8.63e-03",
        "positive_control_docetaxel_fisher_padj": "3.06e-02",
        "positive_control_epirubicin_fisher_padj": "3.22e-06",
        "positive_control_olaparib_fisher_padj": "1.21e-01",
        "positive_control_talazoparib_fisher_padj": "1.36e-04",
    },
)
_fisher_copy = pd.read_csv(OUT / "meta_analysis_fisher.csv")
_check(
    "paper/results/meta_analysis_fisher.csv is the chi2.sf version of results/v8_multicohort (no p equal to 0)",
    _fisher_copy.equals(pd.read_csv(RESULTS / "v8_multicohort" / "meta_analysis_fisher.csv"))
    and int((_fisher_copy["fisher_p"] == 0).sum()) == 0,
    "the copy of the Fisher table is stale",
)
_check_frozen(
    "Scripts 37 and 40 rerun with talazoparib (12 drugs, BH over 12) frozen: RFS 12/12 concordant and 8 at FDR < "
    "0.05, clinical-adjusted SCAN-B 6 candidates (HR 2.26-3.72) and 4 controls, trametinib 0.081, olaparib 0.141, "
    "METABRIC brigatinib 3.35 (0.0043) and pelitinib 2.63 (0.031)",
    {
        "rfs_n_drugs": 12,
        "rfs_n_direction_concordant": 12,
        "rfs_n_fdr05": 8,
        "rfs_fdr05_drugs": ["osimertinib", "saracatinib", "erlotinib", "brigatinib", "entinostat", "docetaxel",
                            "epirubicin", "talazoparib"],
        "rfs_osimertinib_padj": 0.0497,
        "rfs_saracatinib_padj": 0.0488,
        "clinical_adj_n_drugs": 12,
        "clinical_adj_n_candidates": 7,
        "clinical_adj_scanb_n_candidates_fdr05": 6,
        "clinical_adj_scanb_candidates_fdr05": ["osimertinib", "saracatinib", "erlotinib", "brigatinib", "pelitinib",
                                                "entinostat"],
        "clinical_adj_scanb_hr_min": 2.26,
        "clinical_adj_scanb_hr_max": 3.72,
        "clinical_adj_scanb_controls_fdr05": ["paclitaxel", "docetaxel", "epirubicin", "talazoparib"],
        "clinical_adj_scanb_trametinib_padj": 0.081,
        "clinical_adj_scanb_olaparib_padj": 0.141,
        "clinical_adj_metabric_n_candidates_fdr05": 2,
        "clinical_adj_metabric_candidates_fdr05": ["brigatinib", "pelitinib"],
        "clinical_adj_metabric_controls_fdr05": [],
        "clinical_adj_metabric_brigatinib_hr": 3.35,
        "clinical_adj_metabric_brigatinib_padj": 0.0043,
        "clinical_adj_metabric_pelitinib_hr": 2.63,
        "clinical_adj_metabric_pelitinib_padj": 0.031,
    },
)
# Keys this sanity block writes itself (penalizer sweep, TF-level Cox, AURORA sample-level
# drugs), pinned next to the re-audit keys that share their sources
_check_frozen(
    "Keys written in the sanity block frozen: HR range of the significant drugs [0.0017, 245.51] / [0.1314, 5.53] / "
    "[0.3053, 1.58] at penalizer 0.01 / 0.5 / 1.0, TF-level HR MYC 1.13, E2F1 1.12, TP53 0.90, AURORA 782 drugs "
    "(633 / 149) with entinostat p 0.0028",
    {
        "pensweep_hr_min_sig_pen001": 0.0017,
        "pensweep_hr_max_sig_pen001": 245.51,
        "pensweep_hr_min_sig_pen05": 0.1314,
        "pensweep_hr_max_sig_pen05": 5.53,
        "pensweep_hr_min_sig_pen10": 0.3053,
        "pensweep_hr_max_sig_pen10": 1.58,
        "pensweep_hr_min_pen05": 0.13,
        "pensweep_hr_max_pen05": 5.53,
        "pensweep_hr_max_pen001": 245.5,
        "pensweep_n_fdr_pen05": 306,
        "tfcox_MYC_HR": 1.13,
        "tfcox_E2F1_HR": 1.12,
        "tfcox_TP53_HR": 0.9,
        "aurora_drugs_n_tested": 1447,
        "aurora_drugs_sig": 782,
        "aurora_drugs_more_effective_meta": 633,
        "aurora_drugs_less_effective_meta": 149,
        "aurora_drugs_sig_pct": 54.0,
        "aurora_drugs_more_effective_pct": 80.9,
        "aurora_drug_entinostat_p": 0.0028,
        "aurora_drug_paclitaxel_p": 0.56,
        "aurora_drug_panobinostat_p": 0.72,
    },
)

_check_frozen(
    "Re-audit keys of script 67 frozen: the composite of the candidates gives Cliff's delta"
    " 0.095 (permutation p 0.6) over the 12 TNBC lines, no drug of the 1,409 reaches FDR, the"
    " minimum detectable delta is 0.47, both screens agree on the within-breast taxane effect"
    " (-0.50 in PRISM, -0.33 in GDSC) and differ only in the lineage term, and of the 17 gate checks"
    " with an expected direction only the two of paclitaxel pass (7 of the 23 rows with a verdict)",
    {
        'cltnbc_background_n_p05': 53,
        'cltnbc_background_n_padj05': 0,
        'cltnbc_background_pct_delta_negative': 46.13,
        'cltnbc_background_pct_p05': 3.8,
        'cltnbc_calibration_delta_dabrafenib': -0.71,
        'cltnbc_calibration_detect_rate': 0.79,
        'cltnbc_candidate_mean_percentile': 52.1,
        'cltnbc_controls_set_delta': -0.149,
        'cltnbc_controls_set_p': 0.4,
        'cltnbc_gate_n_passed': 7,
        'cltnbc_gate_n_total': 23,
        'cltnbc_gate_n_directional_total': 17,
        'cltnbc_gate_n_directional_passed': 2,
        'cltnbc_gate_directional_passed': ['paclitaxel_prism_tnbc_vs_other_breast',
                                           'paclitaxel_gdsc_tnbc_vs_all_other_lines'],
        'cltnbc_gate_paclitaxel_gdsc_lineage': -0.86,
        'cltnbc_gate_paclitaxel_gdsc_vs_all': -0.89,
        'cltnbc_gate_paclitaxel_gdsc_within_breast': -0.33,
        'cltnbc_gate_paclitaxel_prism_lineage': 0.21,
        'cltnbc_gate_paclitaxel_prism_vs_all': 0.02,
        'cltnbc_gate_paclitaxel_prism_within_breast': -0.5,
        'cltnbc_gdsc_focal_missing': 'brigatinib',
        'cltnbc_gdsc_lines_dropped': 23,
        'cltnbc_gdsc_sign_agreement_pct': 53.3,
        'cltnbc_gdsc_tnbc_disjoint': 10,
        'cltnbc_gdsc_tnbc_lines': 21,
        'cltnbc_gdsc_tnbc_shared_with_prism': 11,
        'cltnbc_iclac_among_tnbc': 0,
        'cltnbc_loo_swing_max': 0.27,
        'cltnbc_max_abs_delta_panel': 0.6,
        'cltnbc_mde_delta_80': 0.47,
        'cltnbc_missing_differential_n_padj05': 0,
        'cltnbc_min_tnbc_lines': 8,
        'cltnbc_min_comparison_lines': 30,
        'cltnbc_calibration_n_draws': 200,
        'cltnbc_calibration_min_rate': 0.8,
        'cltnbc_moa_n_sets': 10000,
        'cltnbc_moa_null_mean': 0.059,
        'cltnbc_moa_null_sd': 0.056,
        'cltnbc_moa_observed_delta': 0.034,
        'cltnbc_moa_p': 0.6,
        'cltnbc_moa_pool_n': 348,
        'cltnbc_n_breast_non_tnbc': 10,
        'cltnbc_n_candidates_expected_direction': 4,
        'cltnbc_n_comparison_lines': 464,
        'cltnbc_n_evaluable': 1409,
        'cltnbc_n_excluded': 39,
        'cltnbc_n_lines': 12,
        'cltnbc_n_not_resolvable': 90,
        'cltnbc_n_panel_above_mde': 6,
        'cltnbc_mde_delta_80_3dp': 0.473,
        'cltnbc_ortho_p': 0.00034,
        'cltnbc_ortho_rho': 0.1,
        'cltnbc_pct_censored_observed': 2.48,
        'cltnbc_pct_panel_above_mde': 0.43,
        'cltnbc_plan_sha256': '974f41ce4b69',
        'cltnbc_power_at_ref_040_max': 0.69,
        'cltnbc_power_at_ref_040_min': 0.68,
        'cltnbc_power_ref_paclitaxel_within_breast': -0.33,
        'cltnbc_ref_cov_min': 243,
        'cltnbc_ref_panel_n': 402,
        'cltnbc_set_ci_high': 0.426,
        'cltnbc_set_ci_low': -0.236,
        'cltnbc_set_delta': 0.095,
        'cltnbc_set_mc_se': 0.005,
        'cltnbc_set_n_perm': 10000,
        'cltnbc_set_p': 0.6,
        'cltnbc_storey_pi0_above_one': True,
        'cltnbc_storey_pi0_lambda05': 1.057,
        'cltnbc_variance_inflation': 9.87,
        **_expand({
            "{role}_{d}_cltnbc_ci_high": (0.52, 0.24, 0.23, 0.16, 0.28, 0.59, 0.53, 0.35, 0.27, 0.31, 0.24, 0.33),
            "{role}_{d}_cltnbc_ci_low": (-0.14, -0.43, -0.43, -0.51, -0.38, -0.07, -0.13, -0.31, -0.4, -0.5, -0.42, -0.39),
            "{role}_{d}_cltnbc_delta": (0.19, -0.09, -0.1, -0.17, -0.05, 0.26, 0.2, 0.02, -0.07, -0.1, -0.09, -0.03),
            "{role}_{d}_cltnbc_p": (0.3, 0.6, 0.6, 0.3, 0.8, 0.1, 0.2, 0.9, 0.7, 0.6, 0.6, 0.9),
            "{role}_{d}_cltnbc_percentile": (85.5, 26.5, 24.6, 12.3, 35.5, 92.8, 87.4, 50.0, 31.7, 26.1, 26.7, 38.6),
            "{role}_{d}_cltnbc_sens_max": (0.43, -0.08, 0.41, 0.03, 0.12, 0.29, 0.35, 0.06, -0.01, 0.42, -0.08, -0.02),
            "{role}_{d}_cltnbc_sens_min": (0.16, -0.4, -0.15, -0.23, -0.09, -0.02, 0.19, -0.5, -0.42, -0.1, -0.2, -0.27),
        }, TWELVE_DRUGS),
    },
)

_check_frozen(
    "Re-audit keys of script 69 frozen: over 1,447 drugs the ordering of the FUSCC"
    " associations correlates 0.32 with the SCAN-B TNBC ordering (permutation p 0.05,"
    " null p95 0.33), the composite of the candidates is flat at 1.02 per SD, no drug"
    " reaches FDR with 49 events and the cohort gate passes on positive nodes",
    {
        'fuscc_ceiling_estimable': False,
        'fuscc_composite_hr_per_sd': 1.02,
        'fuscc_composite_p_perm': 0.91,
        'fuscc_composite_pct_background_descriptive': 76.3,
        'fuscc_effective_df': 24.4,
        'fuscc_gate_nodes_hr': 3.11,
        'fuscc_gate_nodes_passed': True,
        'fuscc_mde_hr_per_sd': 1.49,
        'fuscc_median_followup_months': 47.6,
        'fuscc_n_drugs_fdr05': 0,
        'fuscc_n_drugs_tested': 1447,
        'fuscc_n_events': 49,
        'fuscc_n_excluded': 4,
        'fuscc_n_excluded_predeclared': 2,
        'fuscc_n_excluded_zero_time': 2,
        'fuscc_n_patients': 356,
        'fuscc_n_sequenced': 360,
        'fuscc_n_tfs_shared_prism': 769,
        'fuscc_power_at_osimertinib': 0.33,
        'fuscc_power_at_pelitinib': 0.76,
        'fuscc_reference_hr_pelitinib': 1.46,
        'fuscc_reference_hr_osimertinib': 1.24,
        'fuscc_gate_nodes_n_missing': 2,
        'fuscc_gate_nodes_hr_complete_cases': 3.08,
        'fuscc_gate_nodes_ci_low_complete_cases': 1.71,
        'fuscc_gate_nodes_ci_high_complete_cases': 5.55,
        'fuscc_gate_nodes_range_low': 2.5,
        'fuscc_gate_nodes_range_high': 3.5,
        'fuscc_n_female': 356,
        'fuscc_n_perm': 10000,
        'fuscc_r_primary_p_perm_3dp': 0.053,
        'fuscc_ph_n_fits': 84,
        'fuscc_ph_n_terms': 136,
        'fuscc_ph_exact_n_terms_violating': 22,
        'fuscc_ph_lifelines_n_terms_violating': 12,
        'fuscc_ph_exact_focal12_violating': ['brigatinib', 'trametinib'],
        'fuscc_ph_lifelines_focal12_violating': ['erlotinib', 'olaparib'],
        'fuscc_ph_exact_S1_strata_mrna_n_terms_violating': 0,
        'fuscc_ph_exact_S1b_strata_intrinsic_n_terms_violating': 1,
        'fuscc_ph_exact_S2_clinical_n_terms_violating': 14,
        'fuscc_ph_exact_S3_no_LAR_n_terms_violating': 2,
        'fuscc_ph_exact_S4_surgery_year_n_terms_violating': 2,
        'fuscc_ph_exact_S2_size_n_violating': 13,
        'fuscc_ph_composite_candidates7_p': 0.099,
        'fuscc_ph_composite_controls5_p': 0.64,
        'fuscc_ph_gate_nodes_p': 0.084,
        'fuscc_ph_gate_nodes_complete_cases_p': 0.085,
        'fuscc_ph_gate_size_p': 0.035,
        'fuscc_qt_clipped_pct': 13.92,
        'fuscc_qt_clipped_pct_scanb': 12.7,
        'fuscc_r_143_descriptive': 0.35,
        'fuscc_r_143_p_perm_descriptive': 0.03,
        'fuscc_r_focal12_descriptive': -0.07,
        'fuscc_r_focal12_p_perm_descriptive': 0.57,
        'fuscc_r_no_lar': 0.32,
        'fuscc_r_primary': 0.32,
        'fuscc_r_primary_mc_se': 0.002,
        'fuscc_r_primary_null_p95': 0.33,
        'fuscc_r_primary_p_perm': 0.05,
        'fuscc_r_rfs_reference': 0.21,
        'fuscc_rel_reference': 0.595,
        'fuscc_scanb_tnbc_events': 183,
        'fuscc_scanb_tnbc_n': 676,
        'fuscc_scanb_tnbc_pct_z_positive': 43.3,
        'fuscc_sign_agreement_pct_1447': 62.3,
        'fuscc_sign_agreement_pct_focal12': 75.0,
        'fuscc_var_z_fuscc': 0.911,
        'fuscc_var_z_reference': 2.471,
        **_expand({
            "{role}_{d}_fuscc_ci_high": (1.22, 1.26, 1.54, 1.34, 1.45, 1.33, 1.27, 1.37, 1.35, 1.44, 1.49, 1.76),
            "{role}_{d}_fuscc_ci_low": (0.69, 0.72, 0.88, 0.76, 0.82, 0.76, 0.71, 0.77, 0.77, 0.83, 0.84, 0.99),
            "{role}_{d}_fuscc_hr_per_sd": (0.92, 0.95, 1.16, 1.01, 1.09, 1.0, 0.95, 1.03, 1.02, 1.09, 1.12, 1.32),
            "{role}_{d}_fuscc_reference_z": (2.16, 1.19, 2.94, 1.53, 5.16, 2.17, 0.98, 0.43, 0.75, 0.48, 0.31, 0.2),
            "{role}_{d}_fuscc_z": (-0.58, -0.36, 1.04, 0.06, 0.6, 0.03, -0.35, 0.2, 0.13, 0.64, 0.75, 1.89),
            "{role}_{d}_fuscc_ph_p": (0.12, 0.037, 0.46, 0.3, 0.58, 0.045, 0.2, 0.61, 0.46, 0.41, 0.11, 0.7),
        }, TWELVE_DRUGS),
    },
)

_check_frozen(
    "Split of Table 1 frozen: 381 training and 95 test lines",
    {"split_n_train_lines": 381, "split_n_test_lines": 95},
)

# Ridge and random forest rows of Table 1 since the search of script 72 (October 2026).
# Before it they were 0.728 / 0.114 / 10.6% and 0.769 / 0.128 / 14.2%, single fits with
# the values chosen on the test lines (now hp_previous_*). The random forest row is the mean
# and sd over five random_state values, whose global sd (3e-5) rounds to 0.000, and its
# seed-42 fit, which gives per_drug_RF_TF.csv, is pinned apart.
_check_frozen(
    "Table 1 rows of Ridge and the random forest frozen: Ridge 0.726 / 0.114 / 10.6% (one fit), random forest "
    "0.769 +/- 0.000 / 0.126 +/- 0.003 / 14.5% over five seeds, seed 42 alone 0.769 / 0.129 / 14.1%",
    {
        "table1_Ridge_expression_global_pearson": 0.726,
        "table1_Ridge_expression_median_perdrug": 0.114,
        "table1_Ridge_expression_pct_gt_03": 10.6,
        "table1_Ridge_expression_n_drugs": 1447,
        "table1_Ridge_expression_n_seeds": 1,
        "table1_Ridge_expression_row_type": "model",
        "table1_RF_TF_global_pearson": 0.769,
        "table1_RF_TF_global_pearson_sd": 0.0,
        "table1_RF_TF_median_perdrug": 0.126,
        "table1_RF_TF_median_perdrug_sd": 0.003,
        "table1_RF_TF_pct_gt_03": 14.5,
        "table1_RF_TF_n_drugs": 1447,
        "table1_RF_TF_n_seeds": 5,
        "table1_RF_TF_row_type": "model",
        "table1_RF_TF_seed42_global_pearson": 0.769,
        "table1_RF_TF_seed42_median_perdrug": 0.129,
        "table1_RF_TF_seed42_pct_gt_03": 14.1,
    },
)

_check_frozen(
    "Keys of scripts 70 and 71 frozen: the transfer clips 80% of the published TCGA"
    " activities and 12% after recomputing them over the 17,170 genes shared with PRISM,"
    " FUSCC 71.4% over its full transcriptome, the cohort stays at zero drugs either way"
    " and the Fisher meta-analysis moves 551 to 559",
    {
        'geneuni_meta_docetaxel_padj_prism_universe': 0.035,
        'geneuni_meta_docetaxel_padj_published': 0.031,
        'geneuni_meta_gained': 34,
        'geneuni_meta_lost': 26,
        'geneuni_meta_olaparib_padj_prism_universe': 0.12,
        'geneuni_meta_olaparib_padj_published': 0.12,
        'geneuni_meta_paclitaxel_padj_prism_universe': 0.012,
        'geneuni_meta_paclitaxel_padj_published': 0.0086,
        'geneuni_meta_prism_universe_fdr05': 559,
        'geneuni_meta_published_fdr05': 551,
        'geneuni_meta_shared': 525,
        'geneuni_pct_outside_fuscc': 13.93,
        'geneuni_pct_outside_fuscc_full_transcriptome': 71.4,
        'geneuni_pct_outside_metabric': 12.3,
        'geneuni_pct_outside_scanb': 12.7,
        'geneuni_pct_outside_tcga_prism_universe': 11.79,
        'geneuni_pct_outside_tcga_published': 80.27,
        'geneuni_tcga_prism_universe_fdr05': 0,
        'geneuni_tcga_prism_universe_lambda_gc': 1.03,
        'geneuni_tcga_prism_universe_nominal05': 46,
        'geneuni_tcga_prism_universe_pred_sd': 0.1315,
        'geneuni_tcga_published_fdr05': 0,
        'geneuni_tcga_published_lambda_gc': 0.84,
        'geneuni_tcga_published_nominal05': 34,
        'geneuni_tcga_published_pred_sd': 0.0824,
        'geneuni_tcga_pvalue_spearman': 0.211,
        'geneuni_tcga_n_shared_genes': 17170,
        'geneuni_ph_published_exact_drug_terms_violating': ['olaparib', 'osimertinib'],
        'geneuni_ph_published_exact_pam50_terms_violating': 0,
        'geneuni_ph_published_exact_drug_term_p_min': 0.012,
        'geneuni_ph_prism_universe_exact_drug_terms_violating': ['brigatinib', 'talazoparib'],
        'geneuni_ph_prism_universe_exact_pam50_terms_violating': 0,
        'geneuni_ph_prism_universe_exact_drug_term_p_min': 0.00089,
    },
)

_check_frozen(
    "Nominal counts of Figure 5a frozen: 714 drugs at p < 0.05 in SCAN-B, 322 in METABRIC and 34 in TCGA-BRCA",
    {
        "SCANB_drugs_nominal005": 714,
        "METABRIC_drugs_nominal005": 322,
        "TCGA_drugs_nominal005": 34,
    },
)

# Table 5 as printed: HR_0.1 [95% CI], p_adj,0.1, HR_0.5 and p_adj,0.5 of the seven
# candidates, in the order of CANDIDATES. The keys come from the same rows that write
# table5_candidates.csv, so pinning them here is what stops the table from drifting.
_check_frozen(
    "Table 5 frozen as printed: HR_0.1 2.00 (trametinib) to 6.38 (osimertinib) with its 95% CI and p_adj,0.1 "
    "1.5e-6 (erlotinib) to 8.4e-5 (entinostat), HR_0.5 and p_adj,0.5 of the stability filter",
    _expand({
        "candidate_{d}_hr": (4.28, 3.63, 6.38, 5.5, 3.42, 2.0, 3.28),
        "candidate_{d}_ci_low": (2.51, 2.24, 3.02, 2.74, 2.03, 1.48, 1.94),
        "candidate_{d}_ci_high": (7.3, 5.88, 13.48, 11.02, 5.77, 2.69, 5.54),
        "candidate_{d}_padj": (1.5e-6, 2.6e-6, 1.3e-5, 1.7e-5, 4.0e-5, 4.7e-5, 8.4e-5),
        "candidate_{d}_hr_pen05": (1.94, 1.74, 2.09, 2.01, 1.56, 1.34, 1.71),
        "candidate_{d}_padj_pen05": (1.7e-3, 3.4e-3, 1.5e-2, 1.2e-2, 3.9e-2, 1.4e-2, 1.3e-2),
    }, CANDIDATES),
)
# Script 31 applies BH over the drugs whose Cox fit converged, so a failed fit would
# shrink the denominator of the FDR behind Table 5 and Supplementary Data 1.
_c31_fit = pd.read_csv(RESULTS / "v6_strengthen" / "cox_multivar_pam50_fit_summary.csv").set_index("tag")
_c31_fail = pd.read_csv(RESULTS / "v6_strengthen" / "cox_multivar_pam50_failures.csv")
_check(
    "Script 31: 0 failed Cox fits at the three penalizers, 1,447 drugs fitted and written for each, BH over all "
    "of them, on the 7,215 / 1,164 PAM50-adjusted cohort",
    list(_c31_fit.index) == ["adjusted", "pen05", "pen001"]
    and (_c31_fit["n_failed"] == 0).all() and len(_c31_fail) == 0
    and (_c31_fit["n_fitted"] == _c31_fit["n_drugs_predicted"]).all()
    and all(int(_c31_fit.at[t, "n_fitted"]) == len(pd.read_csv(RESULTS / "v6_strengthen" / f"cox_multivar_pam50_{t}.csv"))
            == stats_clean.get("SCANB_n_drugs_tested") == 1447 for t in _c31_fit.index)
    and (_c31_fit["n_patients"] == stats_clean.get("SCANB_n_patients_cox")).all()
    and (_c31_fit["n_events"] == stats_clean.get("SCANB_n_events_cox")).all(),
    f"failed {_c31_fit['n_failed'].to_dict()}, fitted {_c31_fit['n_fitted'].to_dict()}, "
    f"first failures {_c31_fail.head(5).to_dict('records')}",
)

_check_frozen(
    "Coherence keys of 2026-11-10 frozen: 604 Stouffer drugs with concordant directions, SCAN-B column of Table 3, "
    "83 held-out lines, 769 / 767 / 771 shared TFs, Table 4, FUSCC sensitivity 0.99 to 1.09 and sequenced HR 0.92, "
    "cross-screen HGT 300 epochs at 1e-3",
    {
        "stouffer_n_sig_same_direction": 604,
        "positive_control_paclitaxel_SCANB_hr": 0.47, "positive_control_paclitaxel_SCANB_padj": 0.0011,
        "positive_control_docetaxel_SCANB_hr": 0.5, "positive_control_docetaxel_SCANB_padj": 0.0024,
        "positive_control_epirubicin_SCANB_hr": 0.15, "positive_control_epirubicin_SCANB_padj": 5.7e-05,
        "positive_control_olaparib_SCANB_hr": 0.6, "positive_control_olaparib_SCANB_padj": 0.019,
        "positive_control_talazoparib_SCANB_hr": 0.4, "positive_control_talazoparib_SCANB_padj": 4.9e-06,
        "candidate_heldout_n_test_min": 83,
        "SCANB_n_tfs_shared_prism": 769, "METABRIC_n_tfs_shared_prism": 767, "TCGA_n_tfs_shared_prism": 771,
        "aurora_core_KLF8_primary_mean": 3.48, "aurora_core_KLF8_metastasis_mean": 3.22,
        "aurora_core_KLF8_delta": -0.26, "aurora_core_KLF8_p": 3.8e-05,
        "aurora_core_STAT2_primary_mean": 6.91, "aurora_core_STAT2_metastasis_mean": 6.38,
        "aurora_core_STAT2_delta": -0.54, "aurora_core_STAT2_p": 0.00039,
        "aurora_core_AEBP1_primary_mean": 1.87, "aurora_core_AEBP1_metastasis_mean": 2.03,
        "aurora_core_AEBP1_delta": 0.16, "aurora_core_AEBP1_p": 0.00084,
        "aurora_core_SPDEF_primary_mean": 1.77, "aurora_core_SPDEF_metastasis_mean": 2.11,
        "aurora_core_SPDEF_delta": 0.34, "aurora_core_SPDEF_p": 0.0022,
        "aurora_core_SETBP1_primary_mean": 0.21, "aurora_core_SETBP1_metastasis_mean": -0.09,
        "aurora_core_SETBP1_delta": -0.31, "aurora_core_SETBP1_p": 0.0024,
        "aurora_core_ZEB1_primary_mean": -3.39, "aurora_core_ZEB1_metastasis_mean": -3.56,
        "aurora_core_ZEB1_delta": -0.17, "aurora_core_ZEB1_p": 0.06,
        "aurora_core_MZF1_primary_mean": 3.15, "aurora_core_MZF1_metastasis_mean": 3.23,
        "aurora_core_MZF1_delta": 0.08, "aurora_core_MZF1_p": 0.1,
        "aurora_core_RELA_primary_mean": 19.02, "aurora_core_RELA_metastasis_mean": 18.47,
        "aurora_core_RELA_delta": -0.55, "aurora_core_RELA_p": 0.15,
        "aurora_core_IRX1_primary_mean": 0.52, "aurora_core_IRX1_metastasis_mean": 0.47,
        "aurora_core_IRX1_delta": -0.05, "aurora_core_IRX1_p": 0.34,
        "aurora_core_BARX1_primary_mean": 3.66, "aurora_core_BARX1_metastasis_mean": 3.63,
        "aurora_core_BARX1_delta": -0.03, "aurora_core_BARX1_p": 0.81,
        "fuscc_sens_n_analyses": 5, "fuscc_sens_composite_hr_min": 0.99, "fuscc_sens_composite_hr_max": 1.09,
        "fuscc_sequenced_relapse_hr": 0.92, "fuscc_sequenced_relapse_p": 0.76,
        "crossval_hgt_epochs": 300, "crossval_hgt_lr": 0.001,
    },
)
_check_frozen(
    "Donors of the hyperparameter search pseudonymized: SHA-256 of folds.csv with D0001... in place of the DepMap "
    "PatientID, the file of the public package",
    {"hp_folds_pseudonymized_sha256": "657ebac8d16dc31013af0753d434804802c4c3f90e1ceb4be6885ae0bbb6e999"},
)
_check(
    "AURORA-US copy without sample codes: the TF activities of results/v9_aurora/ unchanged, each sample as "
    "AUR-Pnn-<site> with its site kept, one pseudonym per patient, 129 samples of 53 patients",
    _aurora_copy_ok(),
    "paper/results/aurora_tf_activities.csv differs from its source or keeps a sample code",
)

# Every clinical_adj_* and rfs_* key written by the section of scripts 37 and 40 is
# in the frozen check above, so a new key there cannot go unchecked.
_check(
    "Every key of scripts 37 and 40 (clinical_adj_*, rfs_*) is pinned by a frozen-value check",
    {k for k in stats_clean if k.startswith(("clinical_adj_", "rfs_"))} <= _FROZEN_KEYS,
    f"not frozen: {sorted(k for k in stats_clean if k.startswith(('clinical_adj_', 'rfs_')) and k not in _FROZEN_KEYS)}",
)

_check(
    "Every re-audit key is pinned by a frozen-value check",
    set(REAUDIT_KEYS) <= _FROZEN_KEYS and len(REAUDIT_KEYS) == len(set(REAUDIT_KEYS)),
    f"not frozen: {sorted(set(REAUDIT_KEYS) - _FROZEN_KEYS)[:20]}",
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
