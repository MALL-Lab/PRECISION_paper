"""Supplementary Note 2: effect of the CollecTRI network version on the clinical results.

The published SCAN-B, METABRIC and TCGA-BRCA TF activities
(data/precision_processed/tf_activities_{scanb,metabric,tcga}.csv) were computed
in April 2026 with CollecTRI downloaded live through decoupleR. The graph, the
cell-line and AURORA-US TF activities use the snapshot cached on 28 March 2026,
which scripts 09 and 20 now read as well. Their output for that snapshot is kept
in data/precision_processed/collectri_snapshot/ (first produced on 7 September
2026, provenance/ holds the R scripts and logs of that run).

This script reruns scripts 21 and 31 on the snapshot-based matrices in a
temporary data tree, so the published inputs are never touched, compares each
result of the Note with the published one and writes
results/v11_collectri_snapshot/collectri_snapshot_sensitivity.csv (metric, value),
read by generate_paper_results.py. If the outputs of the 7 September run are
present under results/v11_collectri_snapshot/reference_2026-09-07/, the rerun
must reproduce them.
"""

import importlib.util
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.analysis.candidate_selection import apply_selection_rule, funnel_counts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA = DATA_DIR
PROCESSED = DATA / "precision_processed"
SNAPSHOT = PROCESSED / "collectri_snapshot"
RESULTS = RESULTS_DIR
OUT = RESULTS / "v11_collectri_snapshot"
REFERENCE = OUT / "reference_2026-09-07"

COHORTS = {"SCANB": "scanb", "METABRIC": "metabric", "TCGA": "tcga"}
# The only files of precision_processed that scripts 21 and 31 read.
INPUTS = ["tf_activities_all_prism.csv"] + [f"{c}_clinical.csv" for c in COHORTS.values()]
POSITIVE_CONTROLS = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
CHANGE_TOL = 1e-6   # a TF counts as changed if any sample moves by more than this
LOW_R = 0.99        # changed TFs whose activities correlate below this across samples


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tf_changes(published: pd.DataFrame, snapshot: pd.DataFrame) -> tuple[int, int, int]:
    """TFs whose activities change with the snapshot, and how many correlate below LOW_R."""
    samples = published.index.intersection(snapshot.index)
    tfs = published.columns.intersection(snapshot.columns)
    if len(tfs) != len(published.columns) or len(tfs) != len(snapshot.columns):
        raise ValueError("Published and snapshot matrices do not share the same TFs")
    a, b = published.loc[samples, tfs], snapshot.loc[samples, tfs]
    changed = (a - b).abs().max() > CHANGE_TOL
    r = a.loc[:, changed].corrwith(b.loc[:, changed])
    return int(len(tfs)), int(changed.sum()), int((r < LOW_R).sum())


def rerun(tmp: Path) -> tuple[Path, Path]:
    """Scripts 21 and 31 on a data tree whose clinical TF matrices come from the snapshot."""
    pp = tmp / "data" / "precision_processed"
    pp.mkdir(parents=True)
    for name in INPUTS:
        shutil.copy(PROCESSED / name, pp / name)
    for c in COHORTS.values():
        shutil.copy(SNAPSHOT / f"tf_activities_{c}.csv", pp / f"tf_activities_{c}.csv")

    out8, out6 = tmp / "v8_multicohort", tmp / "v6_strengthen"
    out8.mkdir()
    out6.mkdir()
    m21 = load_module("s21_snapshot", ROOT / "scripts" / "21_multicohort_survival.py")
    m21.DATA, m21.RESULTS = tmp / "data", out8
    m21.main()
    argv = sys.argv
    sys.argv = ["31_cox_multivar_pam50.py"]
    try:
        m31 = load_module("s31_snapshot", ROOT / "scripts" / "31_cox_multivar_pam50.py")
        m31.DATA, m31.RESULTS = tmp / "data", out6
        m31.main()
    finally:
        sys.argv = argv
    return out8, out6


def check_reference(out8: Path, out6: Path) -> None:
    """The rerun must reproduce the outputs of the 7 September run, if they are kept."""
    if not REFERENCE.exists():
        logger.warning("No reference run under %s, reproduction not checked", REFERENCE)
        return
    pairs = [(out8 / f"survival_{c}.csv", REFERENCE / "v8_multicohort" / f"survival_{c}.csv")
             for c in COHORTS]
    pairs += [(out6 / f"cox_multivar_pam50_{t}.csv", REFERENCE / "v6_strengthen" / f"cox_multivar_pam50_{t}.csv")
              for t in ("adjusted", "pen05", "pen001")]
    for new, ref in pairs:
        a = pd.read_csv(new).set_index("drug")
        b = pd.read_csv(ref).set_index("drug")
        if set(a.index) != set(b.index):
            raise ValueError(f"{new.name}: drug sets differ from the reference run")
        for col in ("HR", "padj"):
            if not np.allclose(a.loc[b.index, col], b[col], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{new.name}: {col} differs from the reference run")
    fa = pd.read_csv(out8 / "meta_analysis_fisher.csv").set_index("drug")["fisher_padj"]
    fb = pd.read_csv(REFERENCE / "v8_multicohort" / "meta_analysis_fisher.csv").set_index("drug")["fisher_padj"]
    if not np.allclose(fa.loc[fb.index], fb, rtol=1e-9, atol=1e-12):
        raise ValueError("meta_analysis_fisher.csv differs from the reference run")
    logger.info("Rerun reproduces the 7 September run (%d files)", len(pairs) + 1)


def significant(path: Path, col: str = "padj") -> set:
    df = pd.read_csv(path)
    return set(df.loc[df[col] < 0.05, "drug"])


def both_cohorts(dir8: Path) -> tuple[int, int]:
    s = pd.read_csv(dir8 / "survival_SCANB.csv")
    m = pd.read_csv(dir8 / "survival_METABRIC.csv")
    both = s[s["padj"] < 0.05].merge(m[m["padj"] < 0.05], on="drug", suffixes=("_s", "_m"))
    return int(len(both)), int(((both["HR_s"] > 1) == (both["HR_m"] > 1)).sum())


def main():
    rows = []

    def put(metric, value):
        rows.append({"metric": metric, "value": json.dumps(value) if isinstance(value, list) else value})

    for label, c in COHORTS.items():
        pub = pd.read_csv(PROCESSED / f"tf_activities_{c}.csv", index_col=0)
        snap = pd.read_csv(SNAPSHOT / f"tf_activities_{c}.csv", index_col=0)
        n_tfs, n_changed, n_low = tf_changes(pub, snap)
        put(f"{label}_n_tfs", n_tfs)
        put(f"{label}_n_tfs_changed", n_changed)
        put(f"{label}_n_tfs_r099", n_low)
        logger.info("%s: %d of %d TFs change, %d below r = %.2f", label, n_changed, n_tfs, n_low, LOW_R)

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="precision_snapshot_") as tmp:
        out8, out6 = rerun(Path(tmp))
        check_reference(out8, out6)
        for sub, src in (("v8_multicohort", out8), ("v6_strengthen", out6)):
            (OUT / sub).mkdir(exist_ok=True)
            for f in src.glob("*.csv"):
                shutil.copy(f, OUT / sub / f.name)
    snap8, snap6 = OUT / "v8_multicohort", OUT / "v6_strengthen"
    pub8, pub6 = RESULTS / "v8_multicohort", RESULTS / "v6_strengthen"

    for label in COHORTS:
        s = significant(snap8 / f"survival_{label}.csv")
        p = significant(pub8 / f"survival_{label}.csv")
        put(f"{label}_drugs_FDR005", len(s))
        put(f"{label}_drugs_FDR005_shared", len(s & p))
    s = significant(snap8 / "meta_analysis_fisher.csv", "fisher_padj")
    p = significant(pub8 / "meta_analysis_fisher.csv", "fisher_padj")
    put("Fisher_drugs_FDR005", len(s))
    put("Fisher_drugs_FDR005_shared", len(s & p))
    n_both, n_same = both_cohorts(snap8)
    put("shared_drugs_both_cohorts", n_both)
    put("shared_drugs_direction_consistent", n_same)
    fisher = pd.read_csv(snap8 / "meta_analysis_fisher.csv").set_index("drug")["fisher_padj"]
    for d in POSITIVE_CONTROLS:
        put(f"positive_control_{d}_fisher_padj", float(fisher[d]))

    annot = pd.read_csv(PROCESSED / "prism_drug_annotations.csv")
    rule = {}
    for tag, d6 in (("snapshot", snap6), ("published", pub6)):
        rule[tag] = apply_selection_rule(pd.read_csv(d6 / "cox_multivar_pam50_adjusted.csv"),
                                         pd.read_csv(d6 / "cox_multivar_pam50_pen05.csv"), annot)
    for step, n in zip(("statistical_filter", "stable_pen05", "kinase_hdac", "extended_set", "selected"),
                       funnel_counts(rule["snapshot"])):
        put(f"candidates_n_{step}", n)
    sel = {t: set(r.loc[r["selected"], "drug"]) for t, r in rule.items()}
    snap_rule = rule["snapshot"].set_index("drug")
    put("candidates_selected", sorted(sel["snapshot"]))
    put("candidates_n_unchanged", len(sel["snapshot"] & sel["published"]))
    put("candidates_added", sorted(sel["snapshot"] - sel["published"]))
    put("candidates_dropped", sorted(sel["published"] - sel["snapshot"]))
    for d in sorted(sel["snapshot"] - sel["published"]):
        put(f"added_{d}_hr", round(float(snap_rule.loc[d, "HR"]), 2))
        put(f"added_{d}_moa", str(snap_rule.loc[d, "moa"]))
    for d in sorted(sel["published"] - sel["snapshot"]):
        put(f"dropped_{d}_rank", int(snap_rule.loc[d, "rank"]) if d in snap_rule.index else -1)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "collectri_snapshot_sensitivity.csv", index=False)
    logger.info("Wrote %s (%d metrics)", OUT / "collectri_snapshot_sensitivity.csv", len(out))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
