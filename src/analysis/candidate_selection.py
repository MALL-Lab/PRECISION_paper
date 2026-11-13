"""
Rule-based selection of repurposing candidates (Methods, "Rule-based candidate selection").

Four filters and one cutoff, applied in this order to the PAM50-adjusted
penalized Cox results. No expert scoring: the mechanism class and the
development stage come from the PRISM annotations as distributed (19Q4), and
the rank cutoff is an explicit choice whose sensitivity is reported in
Supplementary Table S1. Shared by generate_paper_results.py (Table 5, Table S1)
and scripts/56_collectri_snapshot_sensitivity.py (Supplementary Note 2), so
that both apply the same rule.
"""

import re

import pandas as pd

SELECTION_RULE = {
    "padj_max": 0.05,            # filter (i), penalizer 0.1
    "hr_range": (1.5, 10.0),     # filter (i), penalizer 0.1
    "stability_padj_max": 0.05,  # filter (ii), penalizer 0.5
    "min_phase": 2,              # filter (iv), PRISM phase annotation
    "rank_cutoff": 30,           # cutoff (v), rank in the filter (i) ranking
}

# PRISM MOA terms denoting protein kinase or HDAC inhibitors (filter iii).
# Each comma-separated term of the MOA string is tested on its own.
KINASE_HDAC_MOA = re.compile(
    r"kinase|egfr|\balk\b|mek|\bsrc\b|hdac|mtor|pi3k|cdk|braf|vegfr|jak|fak|"
    r"aurora|plk|bcr-abl|\babl\b|flt3|btk|igf|fgfr|pdgfr|\bkit\b|\braf\b|erk|"
    r"akt|syk|lck|her2|erbb|ros1|\bret\b|\bmet\b|trk|axl|tyrosine",
    re.IGNORECASE,
)
# Terms that the pattern above matches but that are not protein kinase
# inhibitors: other enzymes whose name contains 'kinase' (urokinase is a
# serine protease), metabolic and lipid kinases, phosphatases and activators.
NOT_KINASE_INHIBITOR = re.compile(
    r"urokinase|glucokinase|phosphofructokinase|adenosine kinase|sphingosine kinase|"
    r"phosphatase|activator",
    re.IGNORECASE,
)

# PRISM phase labels mapped to a development stage number. Phase 1/Phase 2
# agents count as phase 2.
PHASE_RANK = {
    "Preclinical": 0, "Phase 1": 1, "Phase 1/Phase 2": 2, "Phase 2": 2,
    "Phase 2/Phase 3": 3, "Phase 3": 3, "Launched": 4, "Withdrawn": 4,
}


def is_kinase_or_hdac_inhibitor(moa) -> bool:
    """True if any term of a PRISM MOA string is a protein kinase or HDAC inhibitor."""
    if not isinstance(moa, str):
        return False
    terms = (t.strip() for t in moa.split(","))
    return any(KINASE_HDAC_MOA.search(t) and not NOT_KINASE_INHIBITOR.search(t) for t in terms)


def apply_selection_rule(df_pam: pd.DataFrame, df_pam05: pd.DataFrame,
                         annot: pd.DataFrame, rule: dict = SELECTION_RULE) -> pd.DataFrame:
    """Return the drugs passing filter (i), ranked, with one flag per filter.

    Args:
        df_pam: PAM50-adjusted Cox results at penalizer 0.1 (drug, HR, padj, ...).
        df_pam05: the same at penalizer 0.5.
        annot: PRISM drug annotations (name, moa, target, phase, disease_area).
    """
    annot = annot.drop_duplicates(subset="name").copy()
    annot["name_l"] = annot["name"].str.lower()

    lo, hi = rule["hr_range"]
    step1 = df_pam[(df_pam["padj"] < rule["padj_max"])
                   & (df_pam["HR"] >= lo) & (df_pam["HR"] <= hi)].copy()
    step1 = step1.sort_values("padj").reset_index(drop=True)
    step1["rank"] = step1.index + 1
    step1["name_l"] = step1["drug"].str.lower()
    step1 = step1.merge(annot[["name_l", "moa", "target", "phase", "disease_area"]],
                        on="name_l", how="left").drop(columns="name_l")
    pam05_idx = df_pam05.set_index("drug")
    step1["HR_pen05"] = step1["drug"].map(pam05_idx["HR"])
    step1["padj_pen05"] = step1["drug"].map(pam05_idx["padj"])
    step1["pass_stability"] = step1["padj_pen05"] < rule["stability_padj_max"]
    step1["pass_class"] = step1["moa"].map(is_kinase_or_hdac_inhibitor)
    step1["pass_phase"] = (step1["phase"].map(lambda p: PHASE_RANK.get(str(p).strip(), 0))
                           >= rule["min_phase"])
    step1["pass_rank"] = step1["rank"] <= rule["rank_cutoff"]
    extended_mask = step1[["pass_stability", "pass_class", "pass_phase"]].all(axis=1)
    step1["selected"] = extended_mask & step1["pass_rank"]
    return step1


def funnel_counts(step1: pd.DataFrame) -> tuple[int, int, int, int, int]:
    """Drugs left after each step: filter (i), (ii), (iii), (iv) and the rank cutoff."""
    stable = step1["pass_stability"]
    stable_class = stable & step1["pass_class"]
    extended = stable_class & step1["pass_phase"]
    return (int(len(step1)), int(stable.sum()), int(stable_class.sum()),
            int(extended.sum()), int(step1["selected"].sum()))
