"""The class filter of the candidate rule must match protein kinase and HDAC inhibitors only.

It used to match the substring 'kinase' anywhere in the PRISM MOA string, so a
urokinase inhibitor (a serine protease) passed filter (iii).
"""

import pandas as pd
import pytest

from src.analysis.candidate_selection import apply_selection_rule, is_kinase_or_hdac_inhibitor


@pytest.mark.parametrize("moa", [
    "EGFR inhibitor", "HDAC inhibitor", "MEK inhibitor", "src inhibitor",
    "ALK tyrosine kinase receptor inhibitor", "Aurora kinase inhibitor",
    "urokinase inhibitor, EGFR inhibitor",
])
def test_kinase_and_hdac_inhibitors_pass(moa):
    assert is_kinase_or_hdac_inhibitor(moa)


@pytest.mark.parametrize("moa", [
    "urokinase inhibitor", "glucokinase activator", "adenosine kinase inhibitor",
    "sphingosine kinase inhibitor", "phosphofructokinase inhibitor",
    "tyrosine phosphatase inhibitor", "MAP kinase phosphatase inhibitor",
    "MAP kinase activator", "tubulin polymerization inhibitor", "", None,
])
def test_other_mechanisms_fail(moa):
    assert not is_kinase_or_hdac_inhibitor(moa)


def _cox(rows):
    return pd.DataFrame(rows, columns=["drug", "HR", "padj"])


def test_rule_funnel_and_rank():
    pam = _cox([
        ("a", 3.0, 1e-5), ("b", 2.0, 1e-4), ("c", 12.0, 1e-6), ("d", 1.2, 1e-6),
        ("e", 2.5, 1e-3), ("f", 4.0, 0.2),
    ])
    pam05 = _cox([("a", 2.0, 1e-3), ("b", 1.8, 0.2), ("e", 2.0, 1e-2), ("f", 2.0, 0.5)])
    annot = pd.DataFrame({
        "name": ["A", "b", "e", "c", "d", "f"],
        "moa": ["EGFR inhibitor", "HDAC inhibitor", "urokinase inhibitor",
                "EGFR inhibitor", "EGFR inhibitor", "EGFR inhibitor"],
        "target": [""] * 6,
        "phase": ["Launched", "Phase 2", "Phase 3", "Launched", "Launched", "Launched"],
        "disease_area": [""] * 6,
    })
    step1 = apply_selection_rule(pam, pam05, annot)
    # c (HR above 10), d (HR below 1.5) and f (not significant) fail filter (i).
    assert step1["drug"].tolist() == ["a", "b", "e"]
    assert step1["rank"].tolist() == [1, 2, 3]
    assert step1["pass_stability"].tolist() == [True, False, True]
    assert step1["pass_class"].tolist() == [True, True, False]
    assert step1["selected"].tolist() == [True, False, False]
