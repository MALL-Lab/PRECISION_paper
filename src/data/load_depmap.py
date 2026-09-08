"""
Load and preprocess DepMap data (expression, cell line metadata, CRISPR).

Provides unified access to DepMap data files (release pinned in
config.DEPMAP_VERSION). File locations are resolved through
config.depmap_file(kind), which maps each data kind to the file name used by
that release (see data/README.md).
"""

import pandas as pd
import logging

from config import DEPMAP_VERSION, depmap_file

logger = logging.getLogger(__name__)


def load_expression() -> pd.DataFrame:
    """Load DepMap expression matrix (TPM log+1, protein-coding genes).

    Returns:
        DataFrame with depmap_id as index, gene symbols as columns.
        Column names are cleaned to gene symbol only (no Entrez ID).
    """
    path = depmap_file("expression")
    logger.info("Loading DepMap expression (%s): %s", DEPMAP_VERSION, path)
    df = pd.read_csv(path, index_col=0)
    df.columns = [col.split(" ")[0] for col in df.columns]
    logger.info("Expression matrix: %d cell lines x %d genes", *df.shape)
    return df


def load_model_info() -> pd.DataFrame:
    """Load cell line metadata (lineage, subtype, etc.)."""
    path = depmap_file("model")
    logger.info("Loading cell line metadata: %s", path)
    return pd.read_csv(path, index_col=0)


def load_crispr_effect() -> pd.DataFrame:
    """Load CRISPR gene effect scores.

    Returns:
        DataFrame with depmap_id as index, gene symbols as columns.
    """
    path = depmap_file("crispr_effect")
    logger.info("Loading CRISPR gene effect: %s", path)
    df = pd.read_csv(path, index_col=0)
    df.columns = [col.split(" ")[0] for col in df.columns]
    logger.info("CRISPR matrix: %d cell lines x %d genes", *df.shape)
    return df


def filter_breast_lines(model_info: pd.DataFrame) -> list[str]:
    """Return depmap_ids for breast cancer cell lines."""
    mask = model_info["OncotreeLineage"].str.lower() == "breast"
    ids = model_info.index[mask].tolist()
    logger.info("Found %d breast cancer cell lines", len(ids))
    return ids


def filter_tnbc_lines(model_info: pd.DataFrame) -> list[str]:
    """Return depmap_ids for TNBC cell lines.

    Uses ModelSubtypeFeatures (current DepMap column with molecular subtypes).
    Falls back to legacy columns if not available.
    """
    if "ModelSubtypeFeatures" in model_info.columns:
        mask = model_info["ModelSubtypeFeatures"].fillna("").str.contains(
            "TNBC", case=False
        )
    elif "LegacySubSubtype" in model_info.columns:
        mask = model_info["LegacySubSubtype"].fillna("") == "ERneg_HER2neg"
    else:
        mask = model_info["OncotreeSubtype"].fillna("").str.contains(
            "TNBC|Triple Negative", case=False
        )
    ids = model_info.index[mask].tolist()
    logger.info("Found %d TNBC (ER-/HER2-) cell lines", len(ids))
    return ids
