#!/usr/bin/env python3
"""
PRECISION: AURORA-US complementary analysis.

1. Compute TF activities for AURORA primary and metastatic tumors
2. Compare TF profiles: primary vs metastasis (are GNN-key TFs altered?)
3. Predict drug sensitivity for metastatic TNBC samples
4. Compare drug sensitivity predictions: primary vs metastasis (same patient)

Run from project root:
    python scripts/22_aurora_complementary.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root

import numpy as np
import pandas as pd
import gzip
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from config import DATA_DIR, RESULTS_DIR
from src.data.splits import set_all_seeds, cell_line_holdout_split
from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aurora")

RESULTS = RESULTS_DIR / "v9_aurora"
DATA = DATA_DIR
SEED = 42


def load_aurora_expression():
    """Load AURORA UQN expression matrix."""
    path = DATA / "AURORA-US/GSE209998_AUR_129_UQN.txt.gz"
    logger.info("Loading AURORA expression: %s", path)
    with gzip.open(path, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    logger.info("  Expression: %d genes x %d samples", *expr.shape)
    # AURORA UQN is linear scale (median ~6, a few genes reach tens of millions,
    # max 26.2 after log2). Apply log2(x+1) for ULM.
    logger.info("  Raw scale: max=%.1f, median=%.1f", expr.values.max(),
                np.median(expr.values))
    expr = np.log2(expr + 1)
    logger.info("  After log2(x+1): max=%.1f, median=%.1f", expr.values.max(),
                np.median(expr.values))
    return expr


def load_aurora_metadata():
    """Parse AURORA sample metadata from series matrix."""
    path = DATA / "AURORA-US/GSE209998_series_matrix.txt.gz"
    logger.info("Loading AURORA metadata: %s", path)

    # Parse series matrix (GEO format)
    # Note: GEO matrices can have duplicate keys (e.g. multiple
    # !Sample_characteristics_ch1 rows). We append a numeric suffix
    # to avoid overwriting previous rows.
    meta_lines = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_"):
                key = line.split("\t")[0].replace("!Sample_", "")
                vals = [v.strip().strip('"') for v in line.strip().split("\t")[1:]]
                if key in meta_lines:
                    n = 1
                    while f"{key}_{n}" in meta_lines:
                        n += 1
                    key = f"{key}_{n}"
                meta_lines[key] = vals
            if line.startswith("!series_matrix_table_begin"):
                break

    meta = pd.DataFrame(meta_lines)
    if "geo_accession" in meta.columns:
        meta = meta.set_index("geo_accession")

    logger.info("  Metadata: %d samples, columns: %s",
                len(meta), meta.columns[:10].tolist())
    return meta


def compute_aurora_tf_activities(expr):
    """Compute TF activities for AURORA samples."""
    import decoupler as dc

    logger.info("Computing TF activities for AURORA (%d samples)...", expr.shape[1])
    # Same CollecTRI snapshot as the graph (data/prior_knowledge/collectri_edges.csv)
    from src.data.load_prior_knowledge import load_collectri
    collectri = load_collectri()

    # Expression is genes x samples: decoupler needs genes in rows
    tf_acts, _ = dc.mt.ulm(data=expr.T, net=collectri, tmin=5)
    logger.info("  TF activities: %d samples x %d TFs", *tf_acts.shape)
    return tf_acts


def classify_samples(meta):
    """Classify AURORA samples as primary, metastasis, TNBC, etc."""
    # Try to find tissue type and subtype columns
    info = {}

    for col in meta.columns:
        col_lower = col.lower()
        vals = meta[col].astype(str).str.lower()

        if any(k in col_lower for k in ["tissue", "source", "type", "characteristics"]):
            if vals.str.contains("primary|metast|tumor|normal").any():
                info["tissue_col"] = col
                logger.info("  Tissue column: %s", col)
                logger.info("    Values: %s", meta[col].value_counts().head(5).to_dict())

        if any(k in col_lower for k in ["subtype", "er", "her2", "tnbc", "receptor"]):
            info["subtype_col"] = col
            logger.info("  Subtype column: %s", col)
            logger.info("    Values: %s", meta[col].value_counts().head(5).to_dict())

    # Parse characteristics columns (GEO format: "key: value")
    char_cols = [c for c in meta.columns if "characteristics" in c.lower() or "ch1" in c.lower()]
    parsed = {}
    for col in char_cols:
        for idx, val in meta[col].items():
            if ":" in str(val):
                key, value = str(val).split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if idx not in parsed:
                    parsed[idx] = {}
                parsed[idx][key] = value

    if parsed:
        parsed_df = pd.DataFrame(parsed).T
        logger.info("  Parsed characteristics columns: %s", parsed_df.columns.tolist())
        for col in parsed_df.columns:
            logger.info("    %s: %s", col, parsed_df[col].value_counts().head(3).to_dict())

        # Map GSM IDs to AURORA sample IDs (from title column: "... [AUR-...]")
        if "title" in meta.columns:
            import re
            gsm_to_aurora = {}
            for gsm_id, title in meta["title"].items():
                match = re.search(r"\[([^\]]+)\]", str(title))
                if match:
                    gsm_to_aurora[gsm_id] = match.group(1)
            if gsm_to_aurora:
                parsed_df.index = [gsm_to_aurora.get(idx, idx) for idx in parsed_df.index]
                logger.info("  Mapped %d GSM IDs to AURORA sample IDs", len(gsm_to_aurora))

        return parsed_df

    return meta


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Load AURORA data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Load AURORA-US data")
    logger.info("=" * 60)

    expr = load_aurora_expression()
    meta = load_aurora_metadata()
    parsed = classify_samples(meta)

    # Compute TF activities
    tf_acts = compute_aurora_tf_activities(expr)
    tf_acts.to_csv(RESULTS / "aurora_tf_activities.csv")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Compare TF profiles primary vs metastasis
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Primary vs Metastasis TF comparison")
    logger.info("=" * 60)

    # Identify primary vs metastasis from parsed metadata
    if parsed is not None and len(parsed) > 0:
        # Find tissue type column
        tissue_col = None
        for col in parsed.columns:
            vals = parsed[col].astype(str).str.lower()
            if vals.str.contains("primary|metast").any():
                tissue_col = col
                break

        if tissue_col:
            merged = tf_acts.join(parsed[[tissue_col]], how="inner")
            merged[tissue_col] = merged[tissue_col].astype(str).str.lower()

            is_primary = merged[tissue_col].str.contains("primary")
            is_metastasis = merged[tissue_col].str.contains("metast")

            primary_tf = tf_acts.loc[merged.index[is_primary]]
            metastasis_tf = tf_acts.loc[merged.index[is_metastasis]]

            logger.info("  Primary: %d, Metastasis: %d", len(primary_tf), len(metastasis_tf))

            # CRITICAL: AURORA samples are not independent - many patients
            # contribute multiple metastatic biopsies. Extract patient ID from
            # sample ID (format: AUR-PATIENT-TT(P|M)N-...) and run a paired
            # Wilcoxon signed-rank test on patients with both primary and
            # metastatic samples (averaging metastatic samples per patient).
            import re

            def extract_patient(sample_id):
                m = re.match(r"AUR-([A-Z0-9]+)-", str(sample_id))
                return m.group(1) if m else None

            primary_pat = pd.Series(
                [extract_patient(s) for s in primary_tf.index],
                index=primary_tf.index, name="patient"
            )
            meta_pat = pd.Series(
                [extract_patient(s) for s in metastasis_tf.index],
                index=metastasis_tf.index, name="patient"
            )

            # Average metastatic samples per patient (some patients have up to 8)
            primary_by_pat = primary_tf.groupby(primary_pat).mean()
            meta_by_pat = metastasis_tf.groupby(meta_pat).mean()

            # Patients with BOTH primary and metastatic (paired)
            paired_patients = sorted(set(primary_by_pat.index) & set(meta_by_pat.index))
            logger.info("  Paired patients (primary + metastatic): %d", len(paired_patients))

            # Cohort sizes cited in the manuscript (Methods, Results): written
            # to aurora_cohort_summary.csv so that the numbers come from here.
            all_pat = pd.Series([extract_patient(s) for s in tf_acts.index],
                                index=tf_acts.index).dropna()
            samples_per_patient = all_pat.value_counts()
            cohort_summary = {
                "n_samples": int(len(tf_acts)),
                "n_patients": int(all_pat.nunique()),
                "n_primary": int(is_primary.sum()),
                "n_metastasis": int(is_metastasis.sum()),
                "n_unclassified": int(len(tf_acts) - (is_primary | is_metastasis).sum()),
                "n_paired_patients": int(len(paired_patients)),
                "n_patients_multiple_samples": int((samples_per_patient >= 2).sum()),
                "max_metastatic_samples_per_patient": int(meta_pat.value_counts().max()),
            }
            pd.DataFrame([cohort_summary]).to_csv(
                RESULTS / "aurora_cohort_summary.csv", index=False
            )
            logger.info("  Cohort summary: %s", cohort_summary)

            primary_paired = primary_by_pat.loc[paired_patients]
            meta_paired = meta_by_pat.loc[paired_patients]

            # GNN key TFs from XAI
            gnn_key_tfs = ["TP53", "MYC", "DNMT3A", "HIF1A", "AR", "SPI1",
                           "E2F1", "SRSF2", "IRF4", "KLF8"]
            available_tfs = [tf for tf in gnn_key_tfs if tf in tf_acts.columns]

            if len(paired_patients) >= 10:
                from scipy.stats import wilcoxon
                comparison = []
                for tf in available_tfs:
                    p_vals = primary_paired[tf].dropna()
                    m_vals = meta_paired[tf].dropna()
                    common = sorted(set(p_vals.index) & set(m_vals.index))
                    if len(common) >= 10:
                        p_arr = p_vals.loc[common].values
                        m_arr = m_vals.loc[common].values
                        try:
                            stat, pval = wilcoxon(p_arr, m_arr, alternative="two-sided")
                        except ValueError:
                            pval = 1.0
                        comparison.append({
                            "tf": tf,
                            "primary_mean": p_arr.mean(),
                            "metastasis_mean": m_arr.mean(),
                            "delta": m_arr.mean() - p_arr.mean(),
                            "pvalue": pval,
                            "n_paired": len(common),
                        })

                comp_df = pd.DataFrame(comparison).sort_values("pvalue")
                comp_df.to_csv(RESULTS / "aurora_primary_vs_meta_tfs.csv", index=False)

                logger.info("  TF changes primary→metastasis:")
                for _, r in comp_df.iterrows():
                    sig = "*" if r["pvalue"] < 0.05 else ""
                    direction = "↑" if r["delta"] > 0 else "↓"
                    logger.info("    %s: delta=%.3f (%s) p=%.4f %s",
                                r["tf"], r["delta"], direction, r["pvalue"], sig)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Drug sensitivity predictions for AURORA
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: Drug sensitivity predictions for AURORA")
    logger.info("=" * 60)

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    tf_prism = pd.read_csv(DATA / "precision_processed/tf_activities_all_prism.csv", index_col=0)
    train_cells, _ = cell_line_holdout_split(drug_response, 0.2, SEED)

    # Align TFs
    shared_tfs = sorted(set(tf_prism.columns) & set(tf_acts.columns))
    logger.info("  Shared TFs PRISM-AURORA: %d", len(shared_tfs))

    # Batch correction: QT fitted on TRAIN PRISM only (no test leakage)
    train_prism_tf = tf_prism.loc[[c for c in train_cells if c in tf_prism.index], shared_tfs]
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    qt.fit(train_prism_tf)
    prism_qt = pd.DataFrame(qt.transform(tf_prism[shared_tfs]),
                            index=tf_prism.index, columns=shared_tfs)
    aurora_qt = pd.DataFrame(qt.transform(tf_acts[shared_tfs]),
                             index=tf_acts.index, columns=shared_tfs)

    logger.info("  Post-QT: PRISM mean=%.3f std=%.3f, AURORA mean=%.3f std=%.3f",
                prism_qt.mean().mean(), prism_qt.std().mean(),
                aurora_qt.mean().mean(), aurora_qt.std().mean())

    # Per-drug Ridge predictions
    train_feat = prism_qt.loc[[c for c in train_cells if c in prism_qt.index]]
    train_resp = drug_response.loc[train_feat.index]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_aurora = scaler.transform(aurora_qt)

    predictions = {}
    for drug in drug_response.columns:
        y = train_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X_train[mask], y[mask])
        predictions[drug] = ridge.predict(X_aurora)

    pred_df = pd.DataFrame(predictions, index=aurora_qt.index)
    pred_df.to_csv(RESULTS / "aurora_drug_predictions.csv")
    logger.info("  Predictions: %d samples x %d drugs", *pred_df.shape)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Primary vs Metastasis drug sensitivity comparison
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: Drug sensitivity primary vs metastasis")
    logger.info("=" * 60)

    if parsed is not None and tissue_col:
        merged_pred = pred_df.join(parsed[[tissue_col]], how="inner")
        merged_pred[tissue_col] = merged_pred[tissue_col].astype(str).str.lower()

        is_p = merged_pred[tissue_col].str.contains("primary")
        is_m = merged_pred[tissue_col].str.contains("metast")

        primary_pred = pred_df.loc[merged_pred.index[is_p]]
        meta_pred = pred_df.loc[merged_pred.index[is_m]]

        logger.info("  Primary predictions: %d, Metastasis: %d", len(primary_pred), len(meta_pred))

        # For each drug: compare mean predicted AUC in primary vs metastasis
        drug_changes = []
        for drug in pred_df.columns:
            p_auc = primary_pred[drug].dropna()
            m_auc = meta_pred[drug].dropna()
            if len(p_auc) > 3 and len(m_auc) > 3:
                stat, pval = mannwhitneyu(p_auc, m_auc, alternative="two-sided")
                drug_changes.append({
                    "drug": drug,
                    "primary_mean_auc": p_auc.mean(),
                    "metastasis_mean_auc": m_auc.mean(),
                    "delta_auc": m_auc.mean() - p_auc.mean(),
                    "pvalue": pval,
                })

        changes_df = pd.DataFrame(drug_changes).sort_values("pvalue")
        changes_df.to_csv(RESULTS / "aurora_drug_primary_vs_meta.csv", index=False)

        n_sig = (changes_df["pvalue"] < 0.05).sum()
        n_more_sensitive = (changes_df[changes_df["pvalue"] < 0.05]["delta_auc"] < 0).sum()
        n_more_resistant = (changes_df[changes_df["pvalue"] < 0.05]["delta_auc"] > 0).sum()

        logger.info("  Drugs with significant change primary→metastasis: %d/%d",
                    n_sig, len(changes_df))
        logger.info("  Metastasis more SENSITIVE: %d drugs", n_more_sensitive)
        logger.info("  Metastasis more RESISTANT: %d drugs", n_more_resistant)

        logger.info("\n  Top drugs with INCREASED sensitivity in metastasis:")
        more_sens = changes_df[changes_df["delta_auc"] < 0].head(10)
        for _, r in more_sens.iterrows():
            sig = "*" if r["pvalue"] < 0.05 else ""
            logger.info("    %s: delta=%.4f p=%.4f %s", r["drug"], r["delta_auc"], r["pvalue"], sig)

        logger.info("\n  Top drugs with INCREASED resistance in metastasis:")
        more_res = changes_df[changes_df["delta_auc"] > 0].head(10)
        for _, r in more_res.iterrows():
            sig = "*" if r["pvalue"] < 0.05 else ""
            logger.info("    %s: delta=%.4f p=%.4f %s", r["drug"], r["delta_auc"], r["pvalue"], sig)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("AURORA-US SUMMARY")
    logger.info("=" * 60)
    logger.info("  Samples: %d (%d primary, %d metastasis)",
                len(tf_acts), len(primary_tf) if 'primary_tf' in dir() else 0,
                len(metastasis_tf) if 'metastasis_tf' in dir() else 0)
    logger.info("  TF activities: %d TFs", tf_acts.shape[1])
    logger.info("  Drug predictions: %d drugs", pred_df.shape[1])
    logger.info("DONE. Results in %s/", RESULTS)


if __name__ == "__main__":
    main()
