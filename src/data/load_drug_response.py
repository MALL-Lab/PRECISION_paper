"""
Load and preprocess drug response data from PRISM and GDSC.

Preprocessing adapted from an earlier lab pipeline.
Matches drug response matrices with DepMap expression data.
Data location is resolved through config.DATA_DIR (see data/README.md).
"""

import pandas as pd
import logging

from config import DATA_DIR

logger = logging.getLogger(__name__)

PRISM_DIR = DATA_DIR / "PRISM"
GDSC_DIR = DATA_DIR / "GDSC"

PRISM_FILES = {
    "dose_response": PRISM_DIR / "raw" / "secondary-screen-dose-response-curve-parameters.csv",
    "treatment_info": PRISM_DIR / "raw" / "secondary-screen-replicate-treatment-info.csv",
    "response_processed": PRISM_DIR / "processed" / "drug_response_prism.csv",
    "drug_net": PRISM_DIR / "processed" / "drug_net_prism.csv",
    "metadata": PRISM_DIR / "processed" / "metadata_ccle_prism.csv",
    "counts_matched": PRISM_DIR / "processed" / "counts_matched_prism.csv",
}

GDSC_FILES = {
    "dose_response": GDSC_DIR / "raw" / "sanger-dose-response.csv",
    "compounds": GDSC_DIR / "raw" / "screened_compounds_rel_8.5.csv",
    "response_processed": GDSC_DIR / "processed" / "drug_response_sanger.csv",
    "drug_net": GDSC_DIR / "processed" / "drug_net_sanger.csv",
    "metadata": GDSC_DIR / "processed" / "metadata_ccle_sanger.csv",
    "counts_matched": GDSC_DIR / "processed" / "counts_matched_sanger.csv",
}

# PRISM screen priority order
PRISM_SCREEN_PRIORITY = ["MTS010", "MTS006", "MTS005", "HTS002"]


def load_prism_response(use_processed: bool = True) -> pd.DataFrame:
    """Load PRISM drug response matrix (cell lines x drugs, AUC values).

    Args:
        use_processed: If True, load pre-processed file. If False, rebuild from raw.

    Returns:
        DataFrame with depmap_id as index, drug names as columns, AUC as values.
    """
    if use_processed:
        logger.info("Loading processed PRISM response")
        df = pd.read_csv(PRISM_FILES["response_processed"], index_col=0)
        logger.info("PRISM response: %d cell lines x %d drugs", *df.shape)
        return df

    return _preprocess_prism_from_raw()


def _preprocess_prism_from_raw() -> pd.DataFrame:
    """Rebuild PRISM response from raw dose-response data.

    Reproduces the original preprocessing logic:
    1. Match cell lines with DepMap expression
    2. Deduplicate by screen priority
    3. Pivot to cell line x drug AUC matrix
    """
    from .load_depmap import load_expression

    logger.info("Preprocessing PRISM from raw data")
    expression = load_expression()
    dose_resp = pd.read_csv(PRISM_FILES["dose_response"])

    shared_cells = set(dose_resp["depmap_id"]) & set(expression.index)
    dose_resp = dose_resp[dose_resp["depmap_id"].isin(shared_cells)]
    logger.info("Shared cell lines PRISM-DepMap: %d", len(shared_cells))

    dose_resp["screen_id"] = pd.Categorical(
        dose_resp["screen_id"], categories=PRISM_SCREEN_PRIORITY, ordered=True
    )
    dose_resp = dose_resp.sort_values("screen_id")

    seen_broad_ids = set()
    result_rows = []
    for screen_id in PRISM_SCREEN_PRIORITY:
        current = dose_resp[dose_resp["screen_id"] == screen_id]
        new = current[~current["broad_id"].isin(seen_broad_ids)]
        seen_broad_ids.update(new["broad_id"])
        result_rows.append(new)

    filtered = pd.concat(result_rows, ignore_index=True)
    pivot = filtered.groupby(["depmap_id", "name"])["auc"].max().unstack()

    logger.info("PRISM response (rebuilt): %d cell lines x %d drugs", *pivot.shape)
    return pivot


def load_gdsc_response(use_processed: bool = True) -> pd.DataFrame:
    """Load GDSC drug response matrix (cell lines x drugs, AUC values).

    Args:
        use_processed: If True, load pre-processed file. If False, rebuild from raw.

    Returns:
        DataFrame with depmap_id as index, drug names as columns, AUC as values.
    """
    if use_processed:
        logger.info("Loading processed GDSC response")
        df = pd.read_csv(GDSC_FILES["response_processed"], index_col=0)
        logger.info("GDSC response: %d cell lines x %d drugs", *df.shape)
        return df

    return _preprocess_gdsc_from_raw()


def _preprocess_gdsc_from_raw() -> pd.DataFrame:
    """Rebuild GDSC response from raw data."""
    from .load_depmap import load_expression

    logger.info("Preprocessing GDSC from raw data")
    expression = load_expression()
    dose_resp = pd.read_csv(GDSC_FILES["dose_response"])

    dose_resp["DRUG_NAME"] = dose_resp["DRUG_NAME"].str.lower()

    seen_drugs = set()
    result_rows = []
    for dataset in ["GDSC2", "GDSC1"]:
        current = dose_resp[dose_resp["DATASET"] == dataset]
        new = current[~current["DRUG_NAME"].isin(seen_drugs)]
        seen_drugs.update(new["DRUG_NAME"])
        result_rows.append(new)

    filtered = pd.concat(result_rows, ignore_index=True)
    filtered["BROAD_ID"] = filtered["BROAD_ID"].str.split(",", n=1).str[0]

    shared_cells = set(filtered["ARXSPAN_ID"]) & set(expression.index)
    filtered = filtered[filtered["ARXSPAN_ID"].isin(shared_cells)]

    pivot = filtered.groupby(["ARXSPAN_ID", "DRUG_NAME"])["auc"].max().unstack()
    pivot.index.name = "depmap_id"

    logger.info("GDSC response (rebuilt): %d cell lines x %d drugs", *pivot.shape)
    return pivot


def load_prism_drug_annotations() -> pd.DataFrame:
    """Load PRISM drug annotations (broad_id, name, MOA, targets).

    Returns:
        DataFrame with columns: broad_id, name, moa (one row per drug-MOA pair).
    """
    df = pd.read_csv(PRISM_FILES["drug_net"])
    logger.info("PRISM drug annotations: %d entries", len(df))
    return df


def load_gdsc_drug_annotations() -> pd.DataFrame:
    """Load GDSC drug annotations (drug name, target pathway)."""
    df = pd.read_csv(GDSC_FILES["drug_net"])
    logger.info("GDSC drug annotations: %d entries", len(df))
    return df


def get_shared_cell_lines(prism_response: pd.DataFrame,
                          gdsc_response: pd.DataFrame) -> list[str]:
    """Return cell lines present in both PRISM and GDSC."""
    shared = sorted(set(prism_response.index) & set(gdsc_response.index))
    logger.info("Shared cell lines PRISM-GDSC: %d", len(shared))
    return shared


def get_shared_drugs(prism_response: pd.DataFrame,
                     gdsc_response: pd.DataFrame) -> list[str]:
    """Return drugs present in both PRISM and GDSC (name-matched)."""
    prism_drugs = set(prism_response.columns.str.lower())
    gdsc_drugs = set(gdsc_response.columns.str.lower())
    shared = sorted(prism_drugs & gdsc_drugs)
    logger.info("Shared drugs PRISM-GDSC: %d", len(shared))
    return shared
