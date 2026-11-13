"""Receptor subtype of breast cell lines from the DepMap model table."""

import pandas as pd


def classify_breast_subtype(features) -> str:
    """TNBC, HER2+, ER+ or Other from ModelSubtypeFeatures (for example 'basal_A TNBC').

    Precedence is TNBC, then HER2+, then ER+, so ER+ HER2+ lines count as HER2+.
    """
    st = str(features).upper()
    if "TNBC" in st or ("ER-" in st and "HER2-" in st) or "ERNEG" in st:
        return "TNBC"
    if "HER2+" in st or "HER2POS" in st or "HER2 AMP" in st:
        return "HER2+"
    if "ER+" in st or "ERPOS" in st or "LUMINAL" in st:
        return "ER+"
    return "Other"


def breast_subtypes(model: pd.DataFrame, cell_lines) -> pd.Series:
    """Subtype of every breast line among cell_lines (index: ModelID)."""
    meta = model.loc[sorted(set(cell_lines) & set(model.index))]
    breast = meta[meta["OncotreeLineage"].fillna("").str.lower() == "breast"]
    col = "ModelSubtypeFeatures" if "ModelSubtypeFeatures" in breast.columns else "OncotreeSubtype"
    return breast[col].fillna("").map(classify_breast_subtype)
