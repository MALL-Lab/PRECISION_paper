"""Regression tests for the receptor encoding used by the clinical-covariate Cox models.

scripts/37_cox_clinical_covariates.py encoded hormone receptors with string tests that
never matched the real values: SCAN-B stores er and pr as 1.0 and 0.0, and METABRIC stores
ER_IHC as "Positve" (the misspelling comes from cBioPortal). Every patient was coded as
negative, the columns were constant, and the nunique() filter dropped them silently, so
the models described as adjusted for ER and PR never were.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.clinical_encoding import encode_binary_receptor


def test_scanb_numeric_receptor():
    out = encode_binary_receptor(pd.Series([1.0, 0.0, np.nan, 1.0]))
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == 1.0


def test_metabric_misspelled_positive():
    out = encode_binary_receptor(pd.Series(["Positve", "Negative", np.nan, "Positive"]))
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == 1.0


def test_real_encodings_are_not_constant():
    scanb = encode_binary_receptor(pd.Series([1.0, 0.0, 1.0, np.nan]))
    metabric = encode_binary_receptor(pd.Series(["Positve", "Negative", "Positve"]))
    assert scanb.dropna().nunique() == 2
    assert metabric.dropna().nunique() == 2


def test_missing_is_returned_as_nan_not_negative():
    out = encode_binary_receptor(pd.Series([np.nan, "nan", None, ""]))
    assert out.isna().all()


def test_unrecognised_value_raises():
    with pytest.raises(ValueError):
        encode_binary_receptor(pd.Series(["borderline"]))


def test_previous_script37_expressions_were_constant():
    """Documents the defect: the old expressions produce a single value on real data."""
    scanb = pd.Series([1.0, 0.0, 1.0])
    assert (scanb.astype(str).str.upper().str.startswith("P")).astype(int).nunique() == 1
    metabric = pd.Series(["Positve", "Negative"])
    assert (metabric.astype(str).str.lower() == "pos").astype(int).nunique() == 1
