"""
Centralized drug name normalization.

All drug names are lowercased and stripped to ensure consistent
matching across PRISM, GDSC, MOA features, and graph node maps.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name to lowercase stripped."""
    return str(name).strip().lower()


def normalize_drug_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize drug column names in a response matrix."""
    df.columns = [normalize_drug_name(c) for c in df.columns]
    # Remove duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def normalize_drug_series(series: pd.Series, col: str = "name") -> pd.Series:
    """Normalize drug names in a Series or DataFrame column."""
    return series.str.strip().str.lower()


def build_drug_index(drug_map: dict) -> dict:
    """Map normalized drug names to graph node indices.

    The graph pickle keeps the original PRISM spelling of some drug names
    (for example '1-naphthyl-PP1'), while response matrices are normalized
    to lowercase, so exact lookups against node_maps['drug'] miss them.
    """
    index, seen = {}, {}
    for name, idx in drug_map.items():
        key = normalize_drug_name(name)
        if key in index:
            raise ValueError(f"Drug nodes {seen[key]!r} and {name!r} collide after normalization")
        index[key], seen[key] = idx, name
    return index
