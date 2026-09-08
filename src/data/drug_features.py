"""
Compute drug node features from PRISM annotations.

Strategy: multi-hot encoding of MOA (mechanism of action) categories.
Morgan fingerprints require RDKit, which was not available in the
development environment. MOA encoding is a strong alternative that captures
pharmacological class information.

Where RDKit is available, Morgan fingerprints could be added as an
additional feature block (not implemented in this release).

Data location is resolved through config.DATA_DIR (see data/README.md).
"""

import pandas as pd
import numpy as np
import logging

from config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "precision_processed"


def compute_moa_features(cache: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute multi-hot MOA encoding for PRISM drugs.

    Each drug gets a binary vector indicating its mechanism(s) of action.

    Returns:
        Tuple of (feature_matrix, drug_names, moa_categories).
    """
    cache_path = CACHE_DIR / "drug_moa_features.npy"
    cache_names = CACHE_DIR / "drug_moa_names.csv"
    cache_cats = CACHE_DIR / "drug_moa_categories.csv"

    if cache and cache_path.exists():
        logger.info("Loading cached MOA features")
        feat = np.load(cache_path)
        names = pd.read_csv(cache_names)["name"].tolist()
        cats = pd.read_csv(cache_cats)["moa"].tolist()
        return feat, names, cats

    logger.info("Computing MOA features from PRISM annotations")

    ti = pd.read_csv(DATA_DIR / "PRISM" / "raw" / "secondary-screen-replicate-treatment-info.csv")
    drug_moa = (
        ti[["name", "moa"]]
        .dropna()
        .drop_duplicates()
    )

    # Explode comma-separated MOAs
    drug_moa["moa"] = drug_moa["moa"].str.split(",")
    drug_moa = drug_moa.explode("moa")
    drug_moa["moa"] = drug_moa["moa"].str.strip().str.lower()

    # Filter rare MOAs (< 3 drugs)
    moa_counts = drug_moa["moa"].value_counts()
    valid_moas = moa_counts[moa_counts >= 3].index.tolist()
    drug_moa = drug_moa[drug_moa["moa"].isin(valid_moas)]

    # Build multi-hot matrix
    all_drugs = sorted(drug_moa["name"].unique())
    all_moas = sorted(valid_moas)
    moa_idx = {m: i for i, m in enumerate(all_moas)}

    feat = np.zeros((len(all_drugs), len(all_moas)), dtype=np.float32)
    for i, drug in enumerate(all_drugs):
        drug_moas = drug_moa[drug_moa["name"] == drug]["moa"]
        for m in drug_moas:
            if m in moa_idx:
                feat[i, moa_idx[m]] = 1.0

    logger.info("MOA features: %d drugs x %d MOA categories", len(all_drugs), len(all_moas))

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, feat)
        pd.DataFrame({"name": all_drugs}).to_csv(cache_names, index=False)
        pd.DataFrame({"moa": all_moas}).to_csv(cache_cats, index=False)

    return feat, all_drugs, all_moas


def get_drug_feature_matrix(drug_names: list[str]) -> np.ndarray:
    """Get MOA feature matrix aligned to a specific drug name list.

    Args:
        drug_names: Ordered list of drug names (from graph node_maps).

    Returns:
        (n_drugs, n_features) matrix. Drugs without MOA get zero vectors.
    """
    feat, fp_names, moa_cats = compute_moa_features()
    fp_dict = dict(zip(fp_names, feat))

    result = np.zeros((len(drug_names), feat.shape[1]), dtype=np.float32)
    matched = 0
    for i, name in enumerate(drug_names):
        if name in fp_dict:
            result[i] = fp_dict[name]
            matched += 1

    logger.info("Drug features: %d/%d drugs matched with MOA encoding (%.0f%%)",
                matched, len(drug_names), 100 * matched / len(drug_names))
    return result
