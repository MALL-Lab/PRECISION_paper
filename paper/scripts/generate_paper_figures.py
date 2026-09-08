#!/usr/bin/env python3
"""
PRECISION -- Unified figure generation with Okabe-Ito palette.

Generates ALL figures referenced in the LaTeX manuscript.

Supplementary assets match the manuscript one to one: FigureSN.png IS
Supplementary Figure N, for N = 1 to 14.

Main-figure assets keep their historical identifiers, which do NOT match the
rendered numbers, because several rendered figures are composed of more than
one asset. The mapping is:
  Figure 1 = Figure1.png (A) + Figure2.png (B)
  Figure 2 = Figure3.png
  Figure 3 = Figure4.png (A) + Figure5.png (B) + Figure11.png (C)
  Figure 4 = Figure10.png
  Figure 5 = Figure6.png
  Figure 6 = Figure7.png (A) + Figure8.png (B)
  Figure 7 = Figure9.png

Run from the package root (any cwd works, paths are anchored to config.py):
    python paper/scripts/generate_paper_figures.py
"""

import sys
import os
import logging
from pathlib import Path

# paper/scripts/ -> package root, so `config` and `src` are importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import pearsonr, spearmanr, fisher_exact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_figs")

# ============================================================
# Paths
# ============================================================
from config import (  # noqa: E402
    ROOT, DATA_DIR, RESULTS_DIR, PAPER_RESULTS, FIG_DIR, SFIG_DIR, depmap_file,
)

# Large intermediate outputs live under RESULTS_DIR (see data/README.md). The
# 220 MB SCAN-B per-drug prediction matrix is read directly from there instead
# of being duplicated into paper/results/.
RESULTS_V5 = RESULTS_DIR / "v5_final"
DPI = 300

# ============================================================
# Okabe-Ito Palette (single source of truth)
# ============================================================
PALETTE = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#999999",
    "skyblue":    "#56B4E9",
}

# Semantic aliases (all reference PALETTE values)
C_PRIMARY    = PALETTE["blue"]
C_HIGHLIGHT  = PALETTE["orange"]
C_ACCENT     = PALETTE["vermillion"]
C_SECONDARY  = PALETTE["purple"]
C_NEUTRAL    = PALETTE["grey"]
C_GREEN      = PALETTE["green"]
C_SKYBLUE    = PALETTE["skyblue"]
C_YELLOW     = PALETTE["yellow"]

# ============================================================
# Global style
# ============================================================
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": C_PRIMARY,
    "axes.labelcolor": C_PRIMARY,
    "text.color": C_PRIMARY,
    "xtick.color": C_NEUTRAL,
    "ytick.color": C_NEUTRAL,
    "grid.color": "#E8E9EB",
    "savefig.dpi": DPI,
    "font.size": 11,
})


# ============================================================
# Helpers
# ============================================================
generated = []
failed = []


def _save(fig, directory, name, pdf=True):
    """Save figure and track result."""
    path_png = directory / f"{name}.png"
    fig.savefig(path_png, bbox_inches="tight")
    if pdf:
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path_png.relative_to(ROOT)))
    logger.info("  OK: %s", name)


def _skip(name, reason):
    failed.append((name, reason))
    logger.warning("  SKIP %s: %s", name, reason)


def _csv(path):
    """Load CSV, skip if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


_STATS_CACHE = {}


def _stat(key, default=None):
    """Value of paper_statistics.json (generate_paper_results.py), so that
    counts printed inside figures come from the same source as the text."""
    if not _STATS_CACHE:
        p = PAPER_RESULTS / "paper_statistics.json"
        if p.exists():
            import json
            with open(p, encoding="utf-8") as fh:
                _STATS_CACHE.update(json.load(fh))
    return _STATS_CACHE.get(key, default)


def _fmt_n(key, default=None):
    v = _stat(key, default)
    return f"{int(v):,}" if v is not None else "?"


def _scanb_patient_cohort():
    """SCAN-B clinical table with the cohort definition of script 21
    (multi-cohort Cox): indexed by sample id, sorted, one sample per patient
    (the first), overall survival present and positive. Joined with the
    per-drug predictions this gives the 7,397 patients of Table 2."""
    clin = pd.read_csv(DATA_DIR / "precision_processed" / "scanb_clinical.csv")
    clin = clin.set_index("sample_id").sort_index()
    if "patient_id" in clin.columns:
        clin = clin[~clin["patient_id"].duplicated(keep="first")]
    clin["os_time"] = pd.to_numeric(clin["os_time"], errors="coerce")
    clin["os_event"] = pd.to_numeric(clin["os_event"], errors="coerce")
    clin = clin.dropna(subset=["os_time", "os_event"])
    return clin[clin["os_time"] > 0]


def _fmt_p(p):
    return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"


# ============================================================
# MAIN FIGURES
# ============================================================

def fig1_graph_schema():
    """Fig 1: Heterogeneous graph schema."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.6, 6.9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node and edge counts come from paper_statistics.json (script 51 summary
    # of the prebuilt graph), never typed here.
    nodes = {
        f"Gene\n({_fmt_n('kg_genes')})":          (2.5, 4.5, C_PRIMARY,   1.0),
        f"TF\n({_fmt_n('kg_tfs')})":              (2.5, 1.5, C_SECONDARY, 0.85),
        f"Drug\n({_fmt_n('kg_drugs')})":          (7.5, 4.5, C_HIGHLIGHT,  0.85),
        f"Cell Line\n({_fmt_n('kg_cell_lines')})": (7.5, 1.5, C_NEUTRAL,    0.85),
    }

    for label, (x, y, color, radius) in nodes.items():
        circle = plt.Circle((x, y), radius, facecolor=color,
                             edgecolor="white", linewidth=3, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=6)

    edges_spec = [
        ((2.5, 2.4), (2.5, 3.5), f"regulates\n{_fmt_n('collectri_edges')}",  C_SECONDARY, "->", -1.3, 0),
        ((7.5, 3.5), (7.5, 2.4), f"response\n{_fmt_n('response_edges')}",    C_NEUTRAL,   "-",   1.3, 0),
        ((6.5, 4.5), (3.5, 4.5), f"targets\n{_fmt_n('drug_target_edges')}",  C_HIGHLIGHT, "->",  0,  -0.4),  # drug -> gene
    ]

    for start, end, label, color, style, off_x, off_y in edges_spec:
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle=style, color=color, lw=2.5,
                                     connectionstyle="arc3,rad=0"))
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx + off_x, my + off_y, label, ha="center", va="center",
                fontsize=8, color=color, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.8))

    # Gene-gene PPI edges are a self-loop on the gene node (they never reach
    # the drug node): drawn as a small loop sitting on top of the gene circle.
    loop = plt.Circle((2.5, 5.85), 0.42, fill=False, edgecolor=C_PRIMARY,
                      linewidth=2.5, zorder=4)
    ax.add_patch(loop)
    ax.text(2.5, 6.5, f"PPI  {_fmt_n('ppi_edges_bidir')}", ha="center", va="center",
            fontsize=8, color=C_PRIMARY, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))

    ax.text(5, -0.35, f"Heterogeneous knowledge graph ({_fmt_n('kg_nodes')} nodes)",
            ha="center", fontsize=12, fontweight="bold", color=C_PRIMARY)
    _save(fig, FIG_DIR, "Figure1")


def fig1b_pipeline():
    """Fig 1b: Pipeline workflow diagram."""
    # Two parallel branches share the TF-activity representation, as in
    # Methods: the GNN + Integrated Gradients branch explains drug response in
    # cell lines, the per-drug Ridge transfer branch carries the predictions to
    # the clinical cohorts. The GNN does not feed the transfer.
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    n_cells = _fmt_n("prism_n_cell_lines", 476)
    n_tfs = _fmt_n("tf_total", 771)
    boxes = [
        (1.3, 3.0, f"PRISM + DepMap\n({n_cells} cell lines)",       C_PRIMARY,   1.1, 0.75),
        (3.8, 3.0, f"TF activities\n({n_tfs} TFs, ULM)",             C_SECONDARY, 1.1, 0.75),
        (6.6, 4.4, "Knowledge graph\n+ GNN (GraphSAGE)",             C_ACCENT,    1.2, 0.75),
        (9.4, 4.4, "Integrated\nGradients (XAI)",                    C_HIGHLIGHT, 1.1, 0.75),
        (6.6, 1.6, "Per-drug Ridge\ntransfer (QT)",                  C_PRIMARY,   1.2, 0.75),
        (9.4, 1.6, "Survival validation\nSCAN-B, METABRIC, TCGA",    C_ACCENT,    1.25, 0.75),
        (12.3, 4.4, "AURORA-US\nprimary vs metastasis",              C_NEUTRAL,   1.2, 0.75),
        (12.3, 1.6, "Rule-based\ncandidate selection",               C_HIGHLIGHT, 1.2, 0.75),
    ]
    for x, y, label, color, hw, hh in boxes:
        rect = plt.Rectangle((x - hw, y - hh), 2 * hw, 2 * hh, facecolor=color,
                              edgecolor="white", linewidth=2, alpha=0.85,
                              zorder=5, joinstyle="round")
        rect.set_clip_on(False)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=6)

    arrows = [
        ((2.4, 3.0), (2.7, 3.0)),      # data -> TF activities
        ((4.9, 3.3), (5.4, 4.2)),      # TF activities -> graph/GNN
        ((4.9, 2.7), (5.4, 1.8)),      # TF activities -> Ridge transfer
        ((7.8, 4.4), (8.3, 4.4)),      # GNN -> IG
        ((7.8, 1.6), (8.15, 1.6)),     # Ridge -> survival validation
        ((10.5, 4.4), (11.1, 4.4)),    # IG TFs -> AURORA
        ((10.65, 1.6), (11.1, 1.6)),   # survival -> selection rule
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2))

    labels = [
        (1.3, 4.2, "CollecTRI, OmniPath,\ndrug targets", 7.5),
        (3.8, 4.2, "shared representation\n(cell lines and patients)", 7.5),
        (8.0, 5.55, "explanation branch (cell lines)", 7.5),
        (8.0, 0.45, "clinical transfer branch (patients)", 7.5),
        (12.3, 5.55, "core TFs", 7.5),
        (12.3, 0.45, "seven candidates", 7.5),
    ]
    for x, y, text, fs in labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=C_NEUTRAL, fontstyle="italic")

    ax.set_title("PRECISION pipeline: from pharmacogenomic data to clinical drug candidates",
                 fontweight="bold", color=C_PRIMARY, fontsize=12, pad=12)
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure2", pdf=False)


def fig2_model_comparison():
    """Fig 2: Model comparison barplot.

    Reads the canonical Table 1 values (table1_model_comparison.csv) so the
    figure never drifts from the table, and flags GNN vs baseline explicitly
    (the previous version hard-coded stale SAGE values, an extra Ridge model
    absent from the table, and mis-coloured the Ridge+GNN-embedding baseline as
    a GNN via a substring match).
    """
    t1 = _csv(PAPER_RESULTS / "table1_model_comparison.csv")
    if t1 is None:
        return _skip("fig2_model_comparison", "table1_model_comparison.csv missing")
    # display name + is-GNN flag, keyed by the CSV Model column
    model_meta = {
        "RF_TF":              ("RF\n(TF activities)",     False),
        "Ridge_expression":   ("Ridge\n(expression)",     False),
        "SAGE_learned_emb":   ("SAGE GNN\n(multitask)",   True),
        "HGT_learned_emb":    ("HGT GNN",                 True),
        "Ridge_expr_GNN_emb": ("Ridge\n(expr + GNN emb)", False),
    }
    rows = []
    for _, r in t1.iterrows():
        if r["Model"] not in model_meta:
            continue
        disp, is_gnn = model_meta[r["Model"]]
        rows.append((disp, float(r["Global_Pearson"]),
                     float(r["Median_PerDrug_Pearson"]), is_gnn))
    rows.sort(key=lambda x: x[1])  # ascending global Pearson (barh: best at top)
    names = [r[0] for r in rows]
    global_pr = [r[1] for r in rows]
    per_drug = [r[2] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [C_HIGHLIGHT if r[3] else C_PRIMARY for r in rows]

    for ax, vals, xlabel, title, xlim in [
        (axes[0], global_pr, "Global Pearson correlation",
         "A) Global prediction", 0.85),
        (axes[1], per_drug, "Median per-drug Pearson",
         "B) Per-drug prediction", 0.18),
    ]:
        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.55)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontweight="bold", color=C_PRIMARY, fontsize=13)
        ax.set_xlim(0, xlim)
        for bar, val in zip(bars, vals):
            ax.text(val + xlim * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, color=C_PRIMARY,
                    fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=C_PRIMARY, label="Traditional ML"),
        mpatches.Patch(facecolor=C_HIGHLIGHT, label="Graph Neural Network"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure3")


def fig3_tf_importance():
    """Fig 4 (panel A of the composed XAI figure): IG top-20 TFs (seed 42).

    Canonical IG ranking on the seed-42 GNN with mean-train baseline.
    Highlighted in accent colour: TFs that also appear in the multi-seed
    robust core (top 20 across all 3 seeds).
    """
    tf = _csv(PAPER_RESULTS / "tf_importance_ig_top20.csv")
    if tf is None:
        return _skip("Fig4_tf_importance_ig", "tf_importance_ig_top20.csv missing")

    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    robust = set(stab[stab["in_top20_all_seeds"]]["tf"]) if stab is not None else set()

    top = tf.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [C_ACCENT if t in robust else C_NEUTRAL for t in top["tf"]]

    ax.barh(top["tf"], top["importance_mean"], color=colors,
            edgecolor="white", height=0.65, alpha=0.9)
    ax.axvline(x=0, color=C_NEUTRAL, linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Mean $|$attribution$|$ (Integrated Gradients)", fontsize=12)
    ax.set_title("TF importance via Integrated Gradients\n(seed 42, mean-train baseline, top 20)",
                 fontweight="bold", color=C_PRIMARY, fontsize=13)

    legend_elements = [
        mpatches.Patch(facecolor=C_ACCENT, alpha=0.9,
                       label="Robust across 3 seeds"),
        mpatches.Patch(facecolor=C_NEUTRAL, alpha=0.9,
                       label="Not in the top 20 of all three seeds"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure4")


def fig4_gnn_vs_rf():
    """Fig 5 (panel B of the composed XAI figure): IG multi-seed stability.

    For each TF that enters the top 20 IG ranking of at least one seed,
    plot the rank per seed (lower = more important). Points are connected
    by a light line to show stability across seeds. TFs in the top 20 of
    all three seeds are highlighted.
    """
    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    if stab is None:
        return _skip("Fig5_ig_stability", "tf_importance_ig_stability.csv missing")

    # Keep only TFs that enter the top 20 of at least one seed.
    cols = ["rank_42", "rank_123", "rank_456"]
    top20_any = stab[(stab[cols] <= 20).any(axis=1)].copy()
    top20_any = top20_any.sort_values("mean_rank")

    # Plot: per-TF three ranks (1 = best), lower y is better.
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in top20_any.iterrows():
        ranks = [row["rank_42"], row["rank_123"], row["rank_456"]]
        robust = bool(row.get("in_top20_all_seeds", False))
        color = C_ACCENT if robust else C_NEUTRAL
        alpha = 0.95 if robust else 0.45
        lw = 1.6 if robust else 0.9
        ax.plot([0, 1, 2], ranks, marker="o", markersize=7 if robust else 5,
                color=color, alpha=alpha, linewidth=lw)
        if robust:
            ax.text(2.05, ranks[-1], row["tf"], va="center", fontsize=9,
                    fontweight="bold", color=C_PRIMARY)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Seed 42", "Seed 123", "Seed 456"], fontsize=11)
    ax.set_ylabel("IG rank (1 = most important)", fontsize=12)
    ax.invert_yaxis()
    ax.axhline(y=20.5, linestyle="--", color=C_NEUTRAL, lw=0.8, alpha=0.6)
    ax.text(0, 20.5, " top-20 threshold", va="bottom", ha="left",
            fontsize=8, color=C_NEUTRAL)
    ax.set_title("IG multi-seed stability\n(TFs in top 20 of at least one seed)",
                 fontweight="bold", color=C_PRIMARY, fontsize=13)

    n_robust = int(stab["in_top20_all_seeds"].sum())
    legend_elements = [
        mpatches.Patch(facecolor=C_ACCENT, label=f"Top-20 in all 3 seeds (n={n_robust})"),
        mpatches.Patch(facecolor=C_NEUTRAL, label="Top-20 in only 1-2 seeds"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure5", pdf=False)


def fig5_drug_ranking_meta():
    """Fig 5: Drug ranking by Fisher meta-analysis."""
    fisher = _csv(PAPER_RESULTS / "meta_analysis_fisher.csv")
    if fisher is None:
        return _skip("Fig5_drug_ranking_meta", "meta_analysis_fisher.csv missing in paper/results/")

    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    tcga = _csv(PAPER_RESULTS / "survival_TCGA.csv")

    fig5_data = fisher[fisher["fisher_padj"] < 0.05].copy()
    fig5_data["fisher_padj_capped"] = fig5_data["fisher_padj"].clip(lower=1e-15)
    fig5_data["neg_log_padj"] = -np.log10(fig5_data["fisher_padj_capped"])

    # TCGA-BRCA contributes zero individually significant drugs (underpowered),
    # so we classify each Fisher-significant drug by whether it is individually
    # FDR<0.05 in SCAN-B, METABRIC, or both (a "3-cohort" category is impossible
    # and would be misleading).
    def _sig(df, drug):
        if df is None:
            return False
        r = df[df["drug"] == drug]
        return len(r) > 0 and r.iloc[0]["padj"] < 0.05

    def cohort_category(drug):
        sc, me = _sig(scanb, drug), _sig(metabric, drug)
        if sc and me:
            return "both"
        if sc:
            return "scanb"
        if me:
            return "metabric"
        return "meta_only"

    fig5_data["cohort_cat"] = fig5_data["drug"].map(cohort_category)
    fig5_top = fig5_data.nlargest(30, "neg_log_padj")

    # Cohort colours consistent with the multi-cohort figure (Figure 4):
    # SCAN-B = blue, METABRIC = orange, both = green.
    cat_color = {"both": C_GREEN, "scanb": C_PRIMARY,
                 "metabric": C_HIGHLIGHT, "meta_only": C_NEUTRAL}
    cat_label = {"both": "SCAN-B and METABRIC", "scanb": "SCAN-B only",
                 "metabric": "METABRIC only", "meta_only": "Fisher meta only"}

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        range(len(fig5_top)), fig5_top["neg_log_padj"],
        color=[cat_color[c] for c in fig5_top["cohort_cat"]],
        alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(fig5_top)))
    ax.set_yticklabels(fig5_top["drug"], fontsize=9)
    ax.set_xlabel("-log10(Fisher padj)", fontsize=11)
    ax.set_title(f"Top 30 drugs by Fisher meta-analysis significance\n(of {len(fig5_data)} at FDR<0.05)",
                 fontsize=13, fontweight="bold", color=C_PRIMARY)
    ax.invert_yaxis()

    bc_drugs = ["paclitaxel", "docetaxel", "epirubicin", "olaparib", "talazoparib"]
    for i, drug in enumerate(fig5_top["drug"]):
        if drug in bc_drugs:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(C_ACCENT)

    # Legend shows only the categories actually present in the top-30.
    present = [c for c in ["both", "scanb", "metabric", "meta_only"]
               if c in set(fig5_top["cohort_cat"])]
    legend_elements = [mpatches.Patch(facecolor=cat_color[c], label=cat_label[c])
                       for c in present]
    ax.legend(handles=legend_elements, title="Individually FDR$<$0.05 in:",
              loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure6")


def fig6_scanb_survival():
    """Fig 6: SCAN-B KM survival for meta-analysis top drugs."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        return _skip("Fig6_scanb_survival", "lifelines not installed")

    pred = _csv(RESULTS_V5 / "scanb_perdrug_predictions.csv")
    if pred is None:
        return _skip("Fig6_scanb_survival", f"scanb_perdrug_predictions.csv missing in {RESULTS_V5}")
    pred = pd.read_csv(RESULTS_V5 / "scanb_perdrug_predictions.csv", index_col=0)
    clin = _scanb_patient_cohort()  # one sample per patient, as in Table 2 (n = 7,397)

    drugs = ["paclitaxel", "epirubicin", "erlotinib", "entinostat"]
    drugs = [d for d in drugs if d in pred.columns]
    if len(drugs) < 4:
        fisher = _csv(PAPER_RESULTS / "meta_analysis_fisher.csv")
        if fisher is not None:
            for d in fisher["drug"]:
                if d not in drugs and d in pred.columns:
                    drugs.append(d)
                if len(drugs) >= 4:
                    break

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    kmf = KaplanMeierFitter()

    for idx, drug in enumerate(drugs[:4]):
        ax = axes[idx]
        p = pred[drug].rename("auc")
        m = clin.join(p, how="inner")
        m = m.dropna(subset=["os_time", "os_event", "auc"])
        n_expected = _stat("SCANB_n_patients")
        if n_expected is not None and len(m) != int(n_expected):
            logger.warning("  KM cohort for %s has %d patients, Table 2 reports %s", drug, len(m), n_expected)
        med = m["auc"].median()
        s, r = m[m["auc"] < med], m[m["auc"] >= med]

        kmf.fit(s["os_time"] / 365.25, s["os_event"], label="Predicted sensitive")
        kmf.plot_survival_function(ax=ax, color=C_HIGHLIGHT, lw=2.5, ci_show=False)
        kmf.fit(r["os_time"] / 365.25, r["os_event"], label="Predicted resistant")
        kmf.plot_survival_function(ax=ax, color=C_PRIMARY, lw=2.5, ci_show=False)

        lr = logrank_test(s["os_time"], r["os_time"],
                          event_observed_A=s["os_event"],
                          event_observed_B=r["os_event"])
        name = drug.replace("-", " ").title()
        ax.set_title(f"{name}\n(log-rank {_fmt_p(lr.p_value)}, n = {len(m):,})",
                     fontweight="bold", color=C_PRIMARY, fontsize=11)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Overall survival")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower left")

    for idx in range(len(drugs[:4]), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure7")


def fig7_dose_response():
    """Fig 7: Dose-response quintiles of predicted AUC vs survival."""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return _skip("Fig7_dose_response", "lifelines not installed")

    pred_path = RESULTS_V5 / "scanb_perdrug_predictions.csv"
    if not pred_path.exists():
        return _skip("Fig7_dose_response", f"scanb_perdrug_predictions.csv missing in {RESULTS_V5}")
    pred = pd.read_csv(pred_path, index_col=0)
    clin = _scanb_patient_cohort()  # one sample per patient, as in Table 2 (n = 7,397)

    drugs = ["paclitaxel", "entinostat"]
    drugs = [d for d in drugs if d in pred.columns]
    if not drugs:
        return _skip("Fig7_dose_response", "no target drugs in predictions")

    fig, axes = plt.subplots(1, len(drugs), figsize=(7 * len(drugs), 6))
    if len(drugs) == 1:
        axes = [axes]
    kmf = KaplanMeierFitter()

    # Q1 (most sensitive) = orange, Q5 (most resistant) = blue, matching the
    # median-split panel (A); vermillion fills the second warm step.
    quintile_colors = [C_HIGHLIGHT, C_ACCENT, C_NEUTRAL, C_SKYBLUE, C_PRIMARY]
    labels_q = ["Q1 (most sensitive)", "Q2", "Q3", "Q4", "Q5 (most resistant)"]

    for idx, drug in enumerate(drugs):
        ax = axes[idx]
        p = pred[drug].rename("auc")
        m = clin.join(p, how="inner")
        m = m.dropna(subset=["os_time", "os_event", "auc"])
        m["quintile"] = pd.qcut(m["auc"], 5, labels=False)

        for q in range(5):
            subset = m[m["quintile"] == q]
            kmf.fit(subset["os_time"] / 365.25, subset["os_event"],
                    label=labels_q[q])
            kmf.plot_survival_function(ax=ax, color=quintile_colors[q],
                                       lw=2, ci_show=False)

        name = drug.replace("-", " ").title()
        ax.set_title(f"{name}\n(n = {len(m):,})",
                     fontweight="bold", color=C_PRIMARY, fontsize=12)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Overall survival")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower left")

    plt.suptitle("Dose-response: Survival by quintiles of predicted drug sensitivity",
                 fontweight="bold", color=C_PRIMARY, fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure8", pdf=False)


def fig8_aurora_tf_boxplots():
    """Fig 9: AURORA TF activity boxplots for the ten seed-stable IG TFs
    (primary vs metastasis).

    Reads the IG-based validation (table4_aurora_tfs.csv, the same source as
    Table 4) instead of the legacy ablation-TF comparison, so the panels match
    the table exactly. Bonferroni-significant TFs are highlighted (green tint,
    green border, starred title) so they read at a glance (internal review).
    """
    stats_path = PAPER_RESULTS / "table4_aurora_tfs.csv"
    tf_act_path = PAPER_RESULTS / "aurora_tf_activities.csv"
    if not stats_path.exists() or not tf_act_path.exists():
        return _skip("Fig9_aurora_tf_boxplots", "AURORA IG data missing in paper/results/")

    # Same ten IG TFs as Table 4, ordered by significance
    tf_stats = pd.read_csv(stats_path).sort_values("pvalue").reset_index(drop=True)
    tf_act = pd.read_csv(tf_act_path, index_col=0)
    ig_tfs = tf_stats["TF"].tolist()

    # Determine sample type from sample IDs (primary has TTP, metastasis has TTM)
    is_primary = tf_act.index.str.contains("TTP")
    is_meta = tf_act.index.str.contains("TTM")

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for idx, tf in enumerate(ig_tfs):
        ax = axes[idx]
        row = tf_stats.iloc[idx]
        sig = bool(row["Bonferroni_sig"])
        if tf not in tf_act.columns:
            ax.axis("off")
            continue

        primary_vals = tf_act.loc[is_primary, tf].dropna()
        meta_vals = tf_act.loc[is_meta, tf].dropna()

        bp = ax.boxplot([primary_vals, meta_vals],
                        labels=["Primary", "Metastasis"],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", lw=2))
        bp["boxes"][0].set_facecolor(C_PRIMARY)
        bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(C_ACCENT)
        bp["boxes"][1].set_alpha(0.8)
        for element in ["whiskers", "caps"]:
            for item in bp[element]:
                item.set_color(C_NEUTRAL)

        # Highlight Bonferroni-significant TFs at a glance
        if sig:
            ax.set_facecolor(C_GREEN)
            ax.patch.set_alpha(0.10)
            for spine in ax.spines.values():
                spine.set_edgecolor(C_GREEN)
                spine.set_linewidth(2.2)

        pval = row["pvalue"]
        delta = row["Delta"]
        p_str = f"p = {pval:.1e}" if pval < 0.001 else f"p = {pval:.3f}"
        ax.set_title(f"{tf}{' *' if sig else ''}\n({p_str}, $\\Delta$ = {delta:+.2f})",
                     fontweight="bold",
                     color=C_GREEN if sig else C_NEUTRAL, fontsize=10)
        ax.set_ylabel("TF activity")

    for idx in range(len(ig_tfs), len(axes)):
        axes[idx].axis("off")

    n_sig = int(tf_stats["Bonferroni_sig"].sum())
    plt.suptitle("AURORA-US: IG-identified TF activity in primary vs metastatic breast tumors "
                 f"({n_sig}/10 significant after Bonferroni, highlighted)",
                 fontweight="bold", color=C_PRIMARY, fontsize=14, y=1.02)
    legend_elements = [
        mpatches.Patch(facecolor=C_PRIMARY, alpha=0.8, label="Primary"),
        mpatches.Patch(facecolor=C_ACCENT, alpha=0.8, label="Metastasis"),
        mpatches.Patch(facecolor=C_GREEN, alpha=0.25,
                       label="Bonferroni-significant TF (green, *)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure9", pdf=False)


# ============================================================
# SUPPLEMENTARY FIGURES
# ============================================================

def sfig04_tf_drug_heatmap():
    """SFig 4: TF-Drug interaction heatmap using per-drug Ridge coefficients.

    TFs are the ten seed-stable IG core regulators (lowest mean IG rank across
    the three training seeds), read from tf_importance_ig_stability.csv, i.e.
    the same ten reported in Table 4 and Figure 9. Reading them from the
    stability table keeps this panel from drifting away from the table (the
    previous version hard-coded the pre-IG ablation TFs). Drugs are the positive
    controls + data-driven candidates, grouped by MOA.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from collections import OrderedDict

    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    if stab is None:
        return _skip("SFig04_tf_drug_heatmap", "tf_importance_ig_stability.csv missing")

    # Load PRISM data
    tf_path = DATA_DIR / "precision_processed" / "tf_activities_all_prism.csv"
    if not tf_path.exists():
        return _skip("SFig04_tf_drug_heatmap", "PRISM TF activities missing")
    tf_prism = pd.read_csv(tf_path, index_col=0)

    try:
        from src.data.load_drug_response import load_prism_response
        from src.data.drug_names import normalize_drug_columns
        drug_resp = normalize_drug_columns(load_prism_response(use_processed=True))
    except Exception as e:
        return _skip("SFig04_tf_drug_heatmap", f"cannot load PRISM response: {e}")

    shared_cells = sorted(set(tf_prism.index) & set(drug_resp.index))
    tf_prism = tf_prism.loc[shared_cells]
    drug_resp = drug_resp.loc[shared_cells]

    # Curated drugs: positive controls + data-driven candidates (Table 5,
    # pen=0.1), ordered by MOA. Reflects the EGFR-axis convergence narrative.
    drug_moa = OrderedDict([
        ("paclitaxel",    "Tubulin"),
        ("docetaxel",     "Tubulin"),
        ("epirubicin",    "Topo II"),
        ("olaparib",      "PARP"),
        ("osimertinib",   "EGFR"),
        ("erlotinib",     "EGFR"),
        ("pelitinib",     "EGFR"),
        ("brigatinib",    "ALK/EGFR"),
        ("saracatinib",   "Src"),
        ("trametinib",    "MEK"),
        ("entinostat",    "HDAC"),
    ])
    sel_drugs = [d for d in drug_moa if d in drug_resp.columns]

    # Ten seed-stable IG core TFs (same source as Table 4 / Figure 9), read
    # from the stability table so this panel tracks the table automatically.
    # Biological function (for the y-axis labels) follows the programme
    # annotations used in the abstract and Discussion, ordered by function.
    tf_function = OrderedDict([
        ("CREB3L1", "Stress response"),
        ("KLF8",    "EMT"),
        ("AEBP1",   "Stromal / TNBC"),
        ("MZF1",    "Stromal / TNBC"),
        ("SPDEF",   "Epithelial"),
        ("HIVEP1",  "NF-$\\kappa$B"),
        ("REL",     "NF-$\\kappa$B"),
        ("IRX1",    "HOX / developmental"),
        ("ASXL1",   "Epigenetic"),
        ("BCOR",    "Epigenetic"),
    ])
    core10 = stab.sort_values("mean_rank").head(10)["tf"].tolist()
    # Keep the ten core TFs present in PRISM, in the functional order above;
    # append (with a generic label) any core TF not yet annotated, so the panel
    # never silently drops a table TF if the stability ranking shifts.
    sel_tfs = [t for t in tf_function if t in core10 and t in tf_prism.columns]
    for t in core10:
        if t not in sel_tfs and t in tf_prism.columns:
            tf_function.setdefault(t, "IG core")
            sel_tfs.append(t)

    # Train per-drug Ridge and extract coefficients
    scaler = StandardScaler()
    X = scaler.fit_transform(tf_prism)
    tf_cols = list(tf_prism.columns)
    tf_idx = {t: tf_cols.index(t) for t in sel_tfs}

    coef_mat = pd.DataFrame(np.nan, index=sel_tfs, columns=sel_drugs)
    for drug in sel_drugs:
        y = drug_resp[drug].values
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        ridge = Ridge(alpha=100)
        ridge.fit(X[mask], y[mask])
        for tf in sel_tfs:
            coef_mat.loc[tf, drug] = ridge.coef_[tf_idx[tf]]

    # Winsorize at 2nd/98th percentile
    vals = coef_mat.values[~np.isnan(coef_mat.values)]
    if len(vals) > 0:
        lo, hi = np.percentile(vals, [2, 98])
        coef_mat = coef_mat.clip(lower=lo, upper=hi)

    # Clean labels
    drug_labels = [d.replace("-", " ").replace("mesylate", "").strip().title()
                   for d in sel_drugs]
    tf_labels = [f"{tf}  ({tf_function[tf]})" for tf in sel_tfs]

    # Build MOA group positions for x-axis brackets
    moa_groups = OrderedDict()
    for i, d in enumerate(sel_drugs):
        moa = drug_moa[d]
        moa_groups.setdefault(moa, []).append(i)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    # Blue = sensitivity (negative coef), vermillion = resistance (positive),
    # matching the signed IG panel (Figure \ref{fig:xai}C, RdBu_r).
    cmap = LinearSegmentedColormap.from_list("p", [C_PRIMARY, "white", C_ACCENT])
    vmax = np.nanmax(np.abs(coef_mat.values))
    if vmax == 0:
        vmax = 1

    sns.heatmap(coef_mat.values, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                annot=False, linewidths=0.8, linecolor="white",
                mask=np.isnan(coef_mat.values),
                cbar_kws={"label": "Ridge coefficient (blue = sensitivity, "
                                   "vermillion = resistance)", "shrink": 0.65},
                ax=ax, xticklabels=drug_labels, yticklabels=tf_labels)

    ax.set_title("TF-Drug Interaction Map\n(per-drug Ridge coefficients on TF activities)",
                 fontweight="bold", color=C_PRIMARY, fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    # Draw MOA group brackets below x-axis
    for moa, indices in moa_groups.items():
        x_start = indices[0] + 0.1
        x_end = indices[-1] + 0.9
        x_mid = (x_start + x_end) / 2
        y_pos = len(sel_tfs) + 0.8
        if len(indices) > 1:
            ax.plot([x_start, x_end], [y_pos, y_pos],
                    color=C_NEUTRAL, lw=1.5, clip_on=False)
            for idx in indices:
                ax.plot([idx + 0.5, idx + 0.5], [y_pos - 0.15, y_pos],
                        color=C_NEUTRAL, lw=0.8, clip_on=False)
        ax.text(x_mid, y_pos + 0.35, moa, ha="center", va="top",
                fontsize=7.5, color=C_NEUTRAL, fontweight="bold", clip_on=False)

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS4", pdf=False)


def sfig03_umap_embeddings():
    """SFig 3: UMAP of GNN cell line embeddings."""
    try:
        import umap
    except ImportError:
        return _skip("SFig03_umap_embeddings", "umap not installed")

    emb_path = PAPER_RESULTS / "cell_line_embeddings.csv"
    model_path = depmap_file("model")
    if not emb_path.exists() or not model_path.exists():
        return _skip("SFig03_umap_embeddings", "embedding or model data missing")

    cell_emb = pd.read_csv(emb_path, index_col=0)
    mi = pd.read_csv(model_path, index_col=0)
    shared = sorted(set(cell_emb.index) & set(mi.index))
    emb = cell_emb.loc[shared]
    meta = mi.loc[shared]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    emb2d = reducer.fit_transform(emb.values)

    lineage = meta["OncotreeLineage"].fillna("Other")
    # Seven most frequent lineages plus Breast, which the caption highlights
    top_lineages = lineage.value_counts().head(8).index.tolist()
    breast_names = [l for l in lineage.unique() if str(l).lower() == "breast"]
    if breast_names and breast_names[0] not in top_lineages:
        top_lineages = lineage.value_counts().head(7).index.tolist() + breast_names

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: top 8 lineages
    ax = axes[0]
    lineage_colors = {}
    oi_list = [C_PRIMARY, C_HIGHLIGHT, C_GREEN, C_ACCENT, C_SECONDARY,
               C_SKYBLUE, C_YELLOW, C_NEUTRAL]
    for i, lin in enumerate(top_lineages):
        lineage_colors[lin] = oi_list[i % len(oi_list)]

    for lin in top_lineages:
        mask = lineage == lin
        c = lineage_colors[lin]
        s = 50 if lin.lower() == "breast" else 15
        z = 10 if lin.lower() == "breast" else 3
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=c, s=s, alpha=0.7,
                   label=f"{lin} ({mask.sum()})", zorder=z, rasterized=True)
    other = ~lineage.isin(top_lineages)
    ax.scatter(emb2d[other, 0], emb2d[other, 1], c="#DDDDDD", s=5, alpha=0.2,
               label=f"Other ({other.sum()})", rasterized=True, zorder=1)
    ax.set_title("A) GNN embeddings by cancer lineage",
                 fontweight="bold", color=C_PRIMARY)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False)

    # Panel B: breast subtypes (using ModelSubtypeFeatures for molecular subtypes)
    ax = axes[1]
    is_breast = lineage.str.lower() == "breast"
    breast_idx = np.where(is_breast)[0]
    breast_meta = meta[is_breast]

    # ModelSubtypeFeatures has values like "basal_A TNBC", "luminal ER+", "HER2+"
    if "ModelSubtypeFeatures" in breast_meta.columns:
        subtype = breast_meta["ModelSubtypeFeatures"].fillna("")
    elif "LegacySubSubtype" in breast_meta.columns:
        subtype = breast_meta["LegacySubSubtype"].fillna("")
    else:
        subtype = breast_meta["OncotreeSubtype"].fillna("")

    def _classify_subtype(st):
        st = str(st).upper()
        if "TNBC" in st or ("ER-" in st and "HER2-" in st) or "ERNEG" in st:
            return "TNBC"
        if "HER2+" in st or "HER2POS" in st or "HER2 AMP" in st:
            return "HER2+"
        if "ER+" in st or "ERPOS" in st or "LUMINAL" in st:
            return "ER+"
        return "Other"

    subtype_class = subtype.apply(_classify_subtype)
    subtype_colors = {"TNBC": C_ACCENT, "HER2+": C_PRIMARY, "ER+": C_HIGHLIGHT,
                      "Other": C_NEUTRAL}

    for i_b, (orig_i, _) in enumerate(breast_meta.iterrows()):
        c = subtype_colors[subtype_class.loc[orig_i]]
        ax.scatter(emb2d[breast_idx[i_b], 0], emb2d[breast_idx[i_b], 1],
                   c=c, s=80, alpha=0.85, edgecolors="white", linewidths=0.8,
                   zorder=5)

    # Count subtypes for legend
    n_tnbc = (subtype_class == "TNBC").sum()
    n_erpos = (subtype_class == "ER+").sum()
    n_her2 = (subtype_class == "HER2+").sum()

    ax.set_title("B) Breast cancer receptor subtypes",
                 fontweight="bold", color=C_PRIMARY)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # Legend colours taken from the same mapping as the points (the previous
    # version had ER+ and HER2+ swapped in the legend)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=subtype_colors["TNBC"],
                   markersize=10, label=f"TNBC (n={n_tnbc})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=subtype_colors["ER+"],
                   markersize=10, label=f"ER+ (n={n_erpos})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=subtype_colors["HER2+"],
                   markersize=10, label=f"HER2+ (n={n_her2})"),
    ], fontsize=9)

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS3", pdf=False)


def sfig01_training_curves():
    """SFig 1: GNN training curves."""
    history = _csv(PAPER_RESULTS / "gnn_training_history.csv")
    if history is None:
        return _skip("SFig01_training_curves", "training history missing")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(history["epoch"], history["train_loss"], color=C_PRIMARY, lw=2,
             label="Train loss")
    ax1.plot(history["epoch"], history["test_loss"], color=C_ACCENT, lw=2,
             label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber loss")
    ax1.set_title("A) Loss curves", fontweight="bold", color=C_PRIMARY)
    ax1.legend(fontsize=10)

    ax2.plot(history["epoch"], history["test_pearson"], color=C_ACCENT, lw=2.5,
             label="Test Pearson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pearson correlation")
    ax2.set_title("B) Test set correlation", fontweight="bold", color=C_PRIMARY)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS1", pdf=False)


def sfig02_perdrug_distribution():
    """SFig 2: Per-drug Pearson distribution across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = [
        ("RF (TF activities)",  PAPER_RESULTS / "per_drug_RF_TF.csv",            C_PRIMARY),
        ("Ridge (expression)",  PAPER_RESULTS / "per_drug_Ridge_expression.csv",  C_NEUTRAL),
        ("SAGE GNN",            PAPER_RESULTS / "per_drug_SAGE_learned_emb.csv",  C_HIGHLIGHT),
    ]
    any_data = False
    for name, path, color in datasets:
        if path.exists():
            df = pd.read_csv(path)
            ax.hist(df["pearson_r"].dropna(), bins=40, alpha=0.5, color=color,
                    label=name, edgecolor="white")
            any_data = True

    if not any_data:
        plt.close()
        return _skip("SFig02_perdrug_distribution", "no per-drug CSVs found")

    ax.axvline(x=0, color=C_ACCENT, lw=1.5, ls="--")
    ax.set_xlabel("Per-drug Pearson correlation")
    ax.set_ylabel("Number of drugs")
    ax.set_title("Per-drug prediction quality\n(cell-line hold-out)",
                 fontweight="bold", color=C_PRIMARY)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS2", pdf=False)


def sfig14_crossval():
    """SFig 14: Cross-validation PRISM to GDSC."""
    cv = _csv(PAPER_RESULTS / "crossval_prism_to_gdsc.csv")
    if cv is None:
        return _skip("SFig14_crossval", "crossval CSV missing")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cv["pearson_r"].dropna(), bins=30, color=C_PRIMARY,
            edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color=C_ACCENT, lw=2, ls="--", label="r = 0")
    ax.axvline(x=cv["pearson_r"].median(), color=C_HIGHLIGHT, lw=2,
               label=f"Median = {cv['pearson_r'].median():.3f}")
    ax.set_xlabel("Pearson correlation per drug")
    ax.set_ylabel("Number of drugs")
    ax.set_title("Cross-screen validation: PRISM to GDSC",
                 fontweight="bold", color=C_PRIMARY)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS14", pdf=False)


def fig_multicohort():
    """Main figure: multi-cohort clinical validation (bar + overlap).

    Promoted from the supplementary set (internal review): the per-cohort
    significance counts and the SCAN-B/METABRIC overlap are a headline result,
    so they belong in the main text rather than buried in the supplement.
    """
    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    tcga = _csv(PAPER_RESULTS / "survival_TCGA.csv")
    if any(x is None for x in [scanb, metabric, tcga]):
        return _skip("Fig4_multicohort", "survival CSVs missing in paper/results/")

    s_sig = set(scanb[scanb["padj"] < 0.05]["drug"])
    m_sig = set(metabric[metabric["padj"] < 0.05]["drug"])
    shared = s_sig & m_sig

    consistent = 0
    for d in shared:
        s_hr = scanb[scanb["drug"] == d].iloc[0]["HR"]
        m_hr = metabric[metabric["drug"] == d].iloc[0]["HR"]
        if (s_hr > 1 and m_hr > 1) or (s_hr < 1 and m_hr < 1):
            consistent += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: bar chart
    cohorts = ["SCAN-B\n(RNA-seq)", "METABRIC\n(Microarray)", "TCGA\n(RNA-seq)"]
    n_sig = [len(s_sig), len(m_sig), (tcga["padj"] < 0.05).sum()]
    total = [len(scanb), len(metabric), len(tcga)]
    pcts = [f"{n}/{t}\n({n/t*100:.0f}%)" for n, t in zip(n_sig, total)]

    bars = ax1.bar(cohorts, n_sig, color=[C_PRIMARY, C_HIGHLIGHT, C_GREEN],
                   alpha=0.85, edgecolor="white", width=0.6)
    for bar, pct in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 pct, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Drugs FDR < 0.05", fontsize=11)
    ax1.set_title("A. Drugs with significant survival association per cohort",
                  fontsize=12, fontweight="bold", color=C_PRIMARY)
    ax1.set_ylim(0, max(n_sig) * 1.2)

    # Panel B: overlap (fallback bar chart, no venn dependency)
    try:
        from matplotlib_venn import venn2
        v = venn2([s_sig, m_sig], set_labels=("SCAN-B", "METABRIC"), ax=ax2)
        v.get_label_by_id("10").set_text(f"{len(s_sig - shared)}")
        v.get_label_by_id("01").set_text(f"{len(m_sig - shared)}")
        v.get_label_by_id("11").set_text(
            f"{len(shared)}\n({consistent}/{len(shared)}\nsame direction)")
        v.get_label_by_id("11").set_fontsize(8)
        v.get_label_by_id("01").set_fontsize(9)
        for pid in ["10", "01", "11"]:
            patch = v.get_patch_by_id(pid)
            if patch:
                patch.set_alpha(0.6)
        v.get_patch_by_id("10").set_facecolor(C_PRIMARY)
        v.get_patch_by_id("01").set_facecolor(C_HIGHLIGHT)
        v.get_patch_by_id("11").set_facecolor(C_GREEN)
    except ImportError:
        wedge_labels = [f"SCAN-B only\n({len(s_sig - shared)})",
                        f"Both\n({len(shared)})",
                        f"METABRIC only\n({len(m_sig - shared)})"]
        wedge_data = [len(s_sig - shared), len(shared), len(m_sig - shared)]
        ax2.bar(wedge_labels, wedge_data,
                color=[C_PRIMARY, C_GREEN, C_HIGHLIGHT],
                edgecolor="white", width=0.5)
        for i, v in enumerate(wedge_data):
            ax2.text(i, v + 3, str(v), ha="center", fontweight="bold",
                     fontsize=11, color=C_PRIMARY)
        ax2.set_ylabel("Number of drugs")

    ax2.set_title(f"B. Cross-cohort overlap: {len(shared)} drugs, "
                  f"{consistent}/{len(shared)} direction-consistent",
                  fontsize=12, fontweight="bold", color=C_PRIMARY)

    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure10")


def sfig05_qqplot():
    """SFig 5: QQ-plots of survival p-values per cohort."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cohorts = [
        ("SCAN-B (RNA-seq)",       PAPER_RESULTS / "survival_SCANB.csv",    C_PRIMARY),
        ("METABRIC (microarray)",  PAPER_RESULTS / "survival_METABRIC.csv",  C_HIGHLIGHT),
        ("TCGA-BRCA (RNA-seq)",    PAPER_RESULTS / "survival_TCGA.csv",      C_NEUTRAL),
    ]

    any_data = False
    for idx, (name, path, color) in enumerate(cohorts):
        ax = axes[idx]
        df = _csv(path)
        if df is None:
            ax.set_title(f"{name}\n(data missing)", color=C_NEUTRAL)
            continue
        any_data = True

        observed = -np.log10(np.sort(df["pvalue"].clip(1e-50)))
        n = len(observed)
        expected = -np.log10(np.arange(1, n + 1) / (n + 1))

        ax.scatter(expected, observed, c=color, s=8, alpha=0.5, rasterized=True)
        ax.plot([0, max(expected)], [0, max(expected)], "--", color=C_NEUTRAL, lw=1)

        median_chi2 = np.median(stats.chi2.ppf(1 - df["pvalue"].clip(1e-300), df=1))
        lam = median_chi2 / stats.chi2.ppf(0.5, df=1)
        n_sig = (df["padj"] < 0.05).sum()

        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
        ax.set_title(f"{name}\n(lambda={lam:.2f}, {n_sig} FDR<0.05)",
                     fontweight="bold", color=C_PRIMARY, fontsize=10)

    if not any_data:
        plt.close()
        return _skip("SFig05_qqplot", "no survival CSVs found")

    plt.suptitle("QQ-plots of survival p-values per cohort",
                 fontweight="bold", color=C_PRIMARY, fontsize=12)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS5", pdf=False)


def sfig08_forest_plot():
    """SFig 8: Forest plot of the five clinically established BC drugs,
    SCAN-B vs METABRIC, with real penalized-Cox 95% CIs (script 43).

    Focuses on the positive controls and the honest cross-cohort discordance:
    all five reach FDR < 0.05 in SCAN-B, only epirubicin in METABRIC. TCGA is
    omitted as underpowered (QQ lambda = 0.62, SFig 5). CIs come directly from
    the Cox model, not reconstructed from p-values.
    """
    df = _csv(PAPER_RESULTS / "forest_positive_controls.csv")
    if df is None:
        return _skip("SFig08_forest_plot", "forest_positive_controls.csv missing")

    # Order drugs by SCAN-B HR (strongest protective effect at top)
    order = (df[df["cohort"] == "SCAN-B"]
             .sort_values("HR", ascending=False)["drug"].tolist())
    cohorts = [("SCAN-B", C_PRIMARY, 0.16), ("METABRIC", C_HIGHLIGHT, -0.16)]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y_labels = []
    for row, drug in enumerate(order):
        y_labels.append(drug.title())
        for cohort, color, dy in cohorts:
            r = df[(df["drug"] == drug) & (df["cohort"] == cohort)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = row + dy
            sig = r["padj_in_cohort"] < 0.05
            ax.plot([r["CI_low"], r["CI_high"]], [y, y], "-",
                    color=color, lw=2.0, zorder=4, alpha=0.9)
            ax.plot(r["HR"], y, "o" if sig else "o",
                    color=color if sig else "white",
                    markeredgecolor=color, markeredgewidth=1.8,
                    markersize=9, zorder=5)
            # star significant estimates
            if sig:
                ax.text(r["CI_high"] * 1.06, y, "*", color=color,
                        fontsize=15, va="center", ha="left", fontweight="bold")

    ax.axvline(x=1, color=C_NEUTRAL, ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 3.2)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Hazard ratio (log scale, per unit predicted AUC)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_title("Clinically established breast-cancer drugs:\n"
                 "cross-cohort hazard ratios",
                 fontweight="bold", color=C_PRIMARY, fontsize=12)
    # Cohort sizes from paper_statistics.json (Table 2), never typed here
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color=C_PRIMARY, lw=0, markersize=9,
                   label=f"SCAN-B (n = {_fmt_n('SCANB_n_patients')})"),
        plt.Line2D([0], [0], marker="o", color=C_HIGHLIGHT, lw=0, markersize=9,
                   label=f"METABRIC (n = {_fmt_n('METABRIC_n_patients')})"),
        plt.Line2D([0], [0], marker="*", color=C_NEUTRAL, lw=0, markersize=12,
                   label="FDR < 0.05 (filled = significant)"),
    ], fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS8", pdf=False)


def sfig10_penalizer():
    """SFig 10: Penalizer sensitivity analysis (real values from SCAN-B Cox).

    All 1,447 drugs, one sample per patient, SCAN-B Cox with PAM50-adjusted
    covariates. Values read from penalizer_sweep.csv (script 45), the same
    columns the Discussion cites (FDR counts and HR range among significant
    drugs), not the top-200 subset.
    """
    sw = _csv(PAPER_RESULTS / "penalizer_sweep.csv")
    if sw is None:
        return _skip("SFig10_penalizer", "penalizer_sweep.csv missing")
    sw = sw.sort_values("penalizer")
    pens = sw["penalizer"].tolist()
    n_drugs = int(sw["n_drugs"].iloc[0])
    n_fdr = sw["n_fdr_all"].tolist()
    n_nominal = sw["n_nominal_all"].tolist()
    hr_max = sw["hr_max_all"].tolist()
    hr_min = sw["hr_min_all"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(pens, n_nominal, "o-", color=C_NEUTRAL, lw=2, markersize=7,
            label="nominal p < 0.05")
    ax.plot(pens, n_fdr, "o-", color=C_ACCENT, lw=2.5, markersize=8,
            label="FDR < 0.05")
    for x, y in zip(pens, n_fdr):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=C_ACCENT)
    ax.set_xlabel("Cox penalizer")
    ax.set_ylabel(f"Drugs (of {n_drugs:,} tested)")
    ax.set_title("A) Regularization vs significance",
                 fontweight="bold", color=C_PRIMARY)
    ax.set_xscale("log")
    ax.axhline(y=0.05 * n_drugs, color=C_NEUTRAL, ls="--", alpha=0.5,
               label="5% expected by chance")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.fill_between(pens, hr_min, hr_max, alpha=0.2, color=C_ACCENT)
    ax.plot(pens, hr_max, "o-", color=C_ACCENT, lw=2, label="Max HR (FDR < 0.05 drugs)")
    ax.plot(pens, hr_min, "o-", color=C_PRIMARY, lw=2, label="Min HR (FDR < 0.05 drugs)")
    ax.axhline(y=1, color=C_NEUTRAL, ls="--", alpha=0.5)
    ax.set_xlabel("Cox penalizer")
    ax.set_ylabel("Hazard ratio range (log scale)")
    ax.set_title("B) HR regularization", fontweight="bold", color=C_PRIMARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS10", pdf=False)


def sfig12_basal_her2():
    """SFig: Basal+HER2 pooled survival histogram."""
    basal = _csv(PAPER_RESULTS / "basal_her2_survival.csv")
    if basal is None:
        return _skip("SFig12_basal_her2", "basal_her2_survival.csv missing")

    fig, ax = plt.subplots(figsize=(8, 6))
    logp = -np.log10(basal["pvalue"].clip(lower=1e-20))

    ax.hist(logp, bins=50, color=C_PRIMARY, edgecolor="white", alpha=0.8)
    ax.axvline(x=-np.log10(0.05), color=C_ACCENT, ls="--", lw=2,
               label=f"p = 0.05 ({(basal['pvalue'] < 0.05).sum()} drugs)")
    ax.axvline(x=-np.log10(0.05 / len(basal)), color=C_HIGHLIGHT, ls="--", lw=2,
               label=f"Bonferroni ({(basal['pvalue'] < 0.05/len(basal)).sum()} drugs)")

    ax.set_xlabel("-log10(p-value)")
    ax.set_ylabel("Number of drugs")
    n_fdr = (basal["padj"] < 0.05).sum() if "padj" in basal.columns else 0
    n_fdr10 = (basal["padj"] < 0.10).sum() if "padj" in basal.columns else 0
    ax.set_title(f"Basal + HER2 subtype survival analysis\n"
                 f"({n_fdr} FDR < 0.05, {n_fdr10} FDR < 0.10)",
                 fontweight="bold", color=C_PRIMARY)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS12", pdf=False)


def sfig09_aurora_drugs():
    """SFig 9: AURORA drug sensitivity volcano."""
    dr = _csv(PAPER_RESULTS / "aurora_drug_primary_vs_meta.csv")
    if dr is None:
        return _skip("SFig09_aurora_drugs", "aurora drug results missing")

    fig, ax = plt.subplots(figsize=(9, 6))
    logp = -np.log10(dr["pvalue"].clip(lower=1e-20))
    delta_col = "delta" if "delta" in dr.columns else "delta_auc"
    delta = dr[delta_col]
    sig = dr["pvalue"] < 0.05

    ax.scatter(delta[~sig], logp[~sig], c=C_NEUTRAL, s=8, alpha=0.2,
               rasterized=True)

    mask_sens = sig & (delta < 0)
    ax.scatter(delta[mask_sens], logp[mask_sens], c=C_HIGHLIGHT, s=15, alpha=0.6,
               label=f"More sensitive in metastasis ({mask_sens.sum()})")

    mask_res = sig & (delta > 0)
    ax.scatter(delta[mask_res], logp[mask_res], c=C_PRIMARY, s=15, alpha=0.6,
               label=f"More resistant in metastasis ({mask_res.sum()})")

    key_drugs = ["entinostat", "panobinostat", "epirubicin", "paclitaxel"]
    offsets = [(8, 8), (8, -14), (-40, 8), (-40, -14)]  # spread labels apart
    for d, off in zip(key_drugs, offsets):
        row = dr[dr["drug"] == d]
        if len(row) > 0:
            r = row.iloc[0]
            lp = -np.log10(max(r["pvalue"], 1e-20))
            ax.annotate(d.title(), (r[delta_col], lp), fontsize=7, color=C_PRIMARY,
                        xytext=off, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=C_NEUTRAL, lw=0.5))

    ax.axhline(y=-np.log10(0.05), color=C_NEUTRAL, ls="--", lw=1, alpha=0.5)
    ax.axvline(x=0, color=C_NEUTRAL, ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Delta predicted AUC (metastasis - primary)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("AURORA-US: Drug sensitivity changes\n"
                 "primary to metastatic breast tumors",
                 fontweight="bold", color=C_PRIMARY, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS9", pdf=False)


def sfig07_moa_enrichment():
    """SFig 7: MOA enrichment among significant drugs.

    Reads moa_enrichment.csv (script 52, copied to paper/results/ by
    generate_paper_results.py) instead of recomputing from the raw PRISM
    metadata, so the figure, the caption and paper_statistics.json share one
    computation.
    """
    res_df = _csv(PAPER_RESULTS / "moa_enrichment.csv")
    if res_df is None:
        return _skip("SFig07_moa_enrichment", "moa_enrichment.csv missing in paper/results/")
    res_df = res_df.sort_values("pvalue")

    top = res_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [C_ACCENT if p < 0.05 else C_NEUTRAL for p in top["padj"]]
    ax.barh(top["moa"], -np.log10(top["pvalue"]), color=colors,
            edgecolor="white", height=0.6)
    ax.axvline(x=-np.log10(0.05), color=C_NEUTRAL, ls="--", lw=1, alpha=0.5,
               label="p = 0.05")

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(-np.log10(r["pvalue"]) + 0.1, i,
                f"{r['n_sig']}/{r['n_total']} ({r['pct_sig']:.0f}%)",
                va="center", fontsize=8, color=C_PRIMARY)

    ax.set_xlabel("-log10(p-value, Fisher exact test)")
    ax.set_title("MOA Enrichment among drugs with significant\n"
                 "survival association (Fisher meta FDR<0.05)",
                 fontweight="bold", color=C_PRIMARY, fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_ACCENT, label="FDR < 0.05"),
        mpatches.Patch(facecolor=C_NEUTRAL, label="Not significant"),
    ], fontsize=9)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS7", pdf=False)


def sfig13_scanb_metabric_scatter():
    """SFig 13: SCAN-B vs METABRIC p-value scatter."""
    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    if scanb is None or metabric is None:
        return _skip("SFig13_scanb_metabric_scatter", "survival CSVs missing in paper/results/")

    merged = scanb[["drug", "pvalue", "HR"]].merge(
        metabric[["drug", "pvalue", "HR"]],
        on="drug", suffixes=("_scanb", "_metabric")
    )
    merged["neg_log_s"] = -np.log10(merged["pvalue_scanb"].clip(lower=1e-30))
    merged["neg_log_m"] = -np.log10(merged["pvalue_metabric"].clip(lower=1e-30))

    both_sig = ((scanb.set_index("drug")["padj"].reindex(merged["drug"]).values < 0.05) &
                (metabric.set_index("drug")["padj"].reindex(merged["drug"]).values < 0.05))
    consistent_dir = (((merged["HR_scanb"] > 1) & (merged["HR_metabric"] > 1)) |
                      ((merged["HR_scanb"] < 1) & (merged["HR_metabric"] < 1)))
    merged["category"] = "Not significant"
    merged.loc[both_sig & consistent_dir, "category"] = "Both FDR<0.05, consistent"
    merged.loc[both_sig & ~consistent_dir, "category"] = "Both FDR<0.05, inconsistent"

    fig, ax = plt.subplots(figsize=(8, 8))
    cat_colors = {
        "Not significant": C_NEUTRAL,
        "Both FDR<0.05, consistent": C_PRIMARY,
        "Both FDR<0.05, inconsistent": C_ACCENT,
    }
    for cat, color in cat_colors.items():
        mask = merged["category"] == cat
        ax.scatter(merged.loc[mask, "neg_log_s"], merged.loc[mask, "neg_log_m"],
                   c=color, s=15,
                   alpha=0.5 if cat == "Not significant" else 0.8,
                   label=f"{cat} (n={mask.sum()})",
                   zorder=2 if cat == "Not significant" else 3)

    ax.set_xlabel("-log10(p) SCAN-B", fontsize=11)
    ax.set_ylabel("-log10(p) METABRIC", fontsize=11)
    ax.set_title("Cross-cohort p-value concordance",
                 fontsize=13, fontweight="bold", color=C_PRIMARY)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False)

    r, p = spearmanr(merged["neg_log_s"], merged["neg_log_m"])
    ax.text(0.95, 0.05, f"Spearman rho = {r:.3f}",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=10, color=C_PRIMARY, fontweight="bold")

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS13", pdf=False)


def sfig06_target_enrichment():
    """SFig 6: Drug target enrichment.

    Reads target_enrichment.csv (script 52, copied to paper/results/ by
    generate_paper_results.py) instead of recomputing from the raw PRISM
    metadata, so the figure, the caption and paper_statistics.json share one
    computation.
    """
    res_df = _csv(PAPER_RESULTS / "target_enrichment.csv")
    if res_df is None:
        return _skip("SFig06_target_enrichment", "target_enrichment.csv missing in paper/results/")
    res_df = res_df.sort_values("pvalue")

    top = res_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [C_ACCENT if p < 0.05 else C_NEUTRAL for p in top["padj"]]
    ax.barh(top["target"], -np.log10(top["pvalue"]), color=colors,
            edgecolor="white", height=0.6)
    ax.axvline(x=-np.log10(0.05), color=C_NEUTRAL, ls="--", lw=1, alpha=0.5)

    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(-np.log10(r["pvalue"]) + 0.1, i,
                f"{r['n_sig']}/{r['n_total']}",
                va="center", fontsize=8, color=C_PRIMARY)

    n_sig = (res_df["padj"] < 0.05).sum()
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Drug Target Enrichment\n"
                 "(targets of drugs with significant survival association)",
                 fontweight="bold", color=C_PRIMARY, fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_ACCENT, label=f"FDR < 0.05 ({n_sig})"),
        mpatches.Patch(facecolor=C_NEUTRAL, label="Not significant"),
    ], fontsize=9)
    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS6", pdf=False)


def sfig11_graph_ablation():
    """SFig 11: Graph component ablation study."""
    abl = _csv(PAPER_RESULTS / "graph_ablation.csv")
    if abl is None:
        return _skip("SFig11_graph_ablation", "graph_ablation.csv missing")

    abl = abl.sort_values("test_pearson", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = []
    for cfg in abl["config"]:
        if cfg == "Full graph":
            colors.append(C_PRIMARY)
        elif "Only response" in cfg or "No PPI + No" in cfg:
            colors.append(C_ACCENT)
        else:
            colors.append(C_HIGHLIGHT)

    bars = ax.barh(abl["config"], abl["test_pearson"], color=colors,
                   edgecolor="white", height=0.5, alpha=0.85)

    for bar, val in zip(bars, abl["test_pearson"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, color=C_PRIMARY,
                fontweight="bold")

    ax.set_xlabel("Global Pearson correlation (test set)")
    ax.set_title("Graph Component Ablation Study\n"
                 "(removing edge types from the knowledge graph)",
                 fontweight="bold", color=C_PRIMARY, fontsize=13)
    ax.set_xlim(0.7, 0.77)

    ax.legend(handles=[
        mpatches.Patch(facecolor=C_PRIMARY, label="Full graph"),
        mpatches.Patch(facecolor=C_HIGHLIGHT, label="Single edge type removed"),
        mpatches.Patch(facecolor=C_ACCENT, label="Multiple / all removed"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    _save(fig, SFIG_DIR, "FigureS11", pdf=False)


def fig3c_ig_heatmap():
    """Figure 3C: signed Integrated Gradients attributions, top 20 TFs x 11 drugs.

    Positive attribution (red) = resistance driver (higher TF activity -> higher
    predicted AUC). Negative (blue) = sensitivity driver.
    """
    ig_path = PAPER_RESULTS / "tf_drug_ig_signed.csv"
    if not ig_path.exists():
        return _skip("Fig3C_ig_heatmap", "tf_drug_ig_signed.csv missing")
    ig = pd.read_csv(ig_path, comment="#").set_index("tf")
    # Reorder drugs: controls + EGFR cluster + Src + MEK + HDAC
    drug_order = ["paclitaxel", "docetaxel", "epirubicin", "olaparib",
                  "osimertinib", "erlotinib", "brigatinib", "pelitinib",
                  "saracatinib", "trametinib", "entinostat"]
    drug_order = [d for d in drug_order if d in ig.columns]
    ig = ig[drug_order]

    # Diverging colormap, symmetric scale
    vmax = max(abs(ig.values.min()), abs(ig.values.max()))

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ig.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(ig.columns)))
    ax.set_xticklabels(
        [d.replace("-", "\n") for d in ig.columns],
        rotation=35, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(ig.index)))
    ax.set_yticklabels(ig.index, fontsize=10)
    ax.set_xlabel("Drug", fontsize=11)
    ax.set_ylabel("Top 20 TFs by IG importance", fontsize=11)
    ax.set_title(
        "Signed Integrated Gradients attributions (top 20 TFs x candidates + controls)",
        fontsize=11, pad=12
    )

    # Separator between positive controls (4) and candidates (7)
    ax.axvline(x=3.5, color="black", linewidth=1.2, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("Signed IG attribution\n(-) sensitivity driver   |   (+) resistance driver",
                   fontsize=9)

    plt.tight_layout()
    _save(fig, FIG_DIR, "Figure11", pdf=False)


# ============================================================
# Main
# ============================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SFIG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PRECISION: Generating ALL paper figures (Okabe-Ito palette)")
    logger.info("=" * 60)

    # Main figures
    fig1_graph_schema()
    fig1b_pipeline()
    fig2_model_comparison()
    fig3_tf_importance()
    fig4_gnn_vs_rf()
    fig5_drug_ranking_meta()
    fig_multicohort()
    fig6_scanb_survival()
    fig7_dose_response()
    fig8_aurora_tf_boxplots()

    # Supplementary figures
    sfig04_tf_drug_heatmap()
    sfig03_umap_embeddings()
    sfig01_training_curves()
    sfig02_perdrug_distribution()
    sfig14_crossval()
    sfig05_qqplot()
    sfig08_forest_plot()
    sfig10_penalizer()
    sfig12_basal_her2()
    sfig09_aurora_drugs()
    sfig07_moa_enrichment()
    sfig13_scanb_metabric_scatter()
    sfig06_target_enrichment()
    sfig11_graph_ablation()
    fig3c_ig_heatmap()

    # Summary
    logger.info("=" * 60)
    logger.info("GENERATED: %d figures", len(generated))
    for g in generated:
        logger.info("  OK  %s", g)

    if failed:
        logger.warning("SKIPPED: %d figures", len(failed))
        for name, reason in failed:
            logger.warning("  SKIP  %s: %s", name, reason)
    else:
        logger.info("All figures generated successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
