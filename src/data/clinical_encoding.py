"""Encoding helpers for the clinical covariates of the survival models."""
from __future__ import annotations

import numpy as np
import pandas as pd

_POSITIVE = {"1", "1.0", "pos", "positive", "positve"}
_NEGATIVE = {"0", "0.0", "neg", "negative"}
_MISSING = {"", "nan", "none", "na", "<na>", "null"}


def encode_binary_receptor(values: pd.Series) -> pd.Series:
    """Map a receptor-status column to 1.0 (positive), 0.0 (negative) or NaN (missing).

    Accepts the encodings found in the source cohorts: numeric 1 and 0 (SCAN-B er and pr)
    and text labels, including the "Positve" misspelling that METABRIC's ER_IHC carries
    from cBioPortal. Missing values come back as NaN and are never coded as negative, so
    the caller has to decide explicitly how to handle them. Any other value raises,
    because a silent miscoding is exactly how the receptor covariates were lost before.
    """
    def as_text(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).strip().lower()

    text = values.map(as_text)
    known = text.isin(_POSITIVE | _NEGATIVE | _MISSING)
    if not known.all():
        raise ValueError(f"Unrecognised receptor values: {sorted(set(text[~known]))}")

    out = pd.Series(np.nan, index=values.index, dtype=float)
    out[text.isin(_POSITIVE)] = 1.0
    out[text.isin(_NEGATIVE)] = 0.0
    return out
