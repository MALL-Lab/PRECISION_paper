#!/usr/bin/env python3
"""
PRECISION -- Unified figure generation with Okabe-Ito palette.

Generates ALL figures referenced in the LaTeX manuscript.

Every figure is one PNG and the file names match the manuscript one to one,
main and supplementary: FigureN.png IS Figure N and FigureSN.png IS
Supplementary Figure N. Each PNG is RGB at 300 dpi, and every main figure is
170 mm wide (the text width), so LaTeX includes it at width=\\linewidth
without scaling and its 7 pt text prints at 7 pt.

Figures 1, 3 and 4 have panels drawn by separate functions. Those panels are
rendered and checked as before, kept in memory and composed pixel for pixel
into the single PNG of their figure by compose_main_figures():
  Figure 1 = fig1a (graph schema) over fig1b (pipeline)
  Figure 3 = fig3a and fig3b side by side over fig3c
  Figure 4 = fig4a (median split) over fig4b (quintiles)

Panel letters are drawn only by _letter, with one rule for every figure:
8 pt bold lowercase Arial in black, at the top left of the panel, with the
panel content starting LETTER_BAND_PT below the top of the letter. LaTeX
draws no letters. Run with --letters-json PATH to write the letters drawn in
each PNG.

Run from the repository root:
    python paper/scripts/generate_paper_figures.py
"""

import sys
import os
import io
import re
import json
import logging
import argparse
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
import matplotlib.legend_handler
import matplotlib.transforms as mtransforms
import matplotlib.font_manager
import matplotlib.axis
import matplotlib.spines
import seaborn as sns
from PIL import Image
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
DATA = DATA_DIR
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
    # TrueType fonts in any PDF or PS output, as the Nature figure guide asks
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Panel letters and main-figure geometry. Every panel letter is 8 pt bold, and
# the panel content starts LETTER_BAND_PT below the top of its letter. Main
# figures are MAIN_W_MM wide (\linewidth of the bioRxiv class) and at most
# MAIN_MAX_H_MM tall, so that each fits with its caption in the 247 mm text
# block. The composed figures leave COMPOSE_TOP_MM above the first row,
# COMPOSE_ROW_GAP_MM between rows and COMPOSE_BOTTOM_MM under the last one.
PANEL_LETTER_FS = 8
LETTER_BAND_PT = PANEL_LETTER_FS + 3
MAIN_W_MM = 170.0
MAIN_MAX_H_MM = 210.0
COMPOSE_TOP_MM = 2.5
COMPOSE_ROW_GAP_MM = 3.0
COMPOSE_BOTTOM_MM = 1.0
COMPOSE_MIN_GAP_MM = 2.0


# ============================================================
# Helpers
# ============================================================
generated = []
failed = []
PANELS = {}    # panel key -> RGB array of a panel kept in memory for _compose
LETTERS = {}   # PNG name -> panel letters drawn in it by _letter


def _check_arial():
    """The manuscript figures are set in Arial (npj asks for Arial or
    Helvetica). On a machine without it, matplotlib falls back to another
    sans-serif without warning, so the run stops instead."""
    fm = matplotlib.font_manager
    for weight, expected in (("normal", "arial.ttf"), ("bold", "arialbd.ttf")):
        path = fm.findfont(fm.FontProperties(family="sans-serif", weight=weight),
                           fallback_to_default=False)
        if Path(path).name.lower() != expected:
            raise RuntimeError(f"sans-serif {weight} resolves to {path}, not Arial: install Arial "
                               f"so that the figures keep their typeface")


def _letter(fig, x, y, letter):
    """Draw a panel letter, the only place where one is drawn: 8 pt bold
    upright lowercase Arial in black, its top left corner at (x, y) in figure
    coordinates. The letter is recorded on the figure, and _save files it
    under the PNG name for the check against the captions."""
    if len(letter) != 1 or not letter.islower():
        raise ValueError(f"Panel letters are single lowercase letters, got {letter!r}")
    t = fig.text(x, y, letter, ha="left", va="top", fontsize=PANEL_LETTER_FS,
                 fontweight="bold", fontstyle="normal", color="black")
    fig.__dict__.setdefault("_panel_letter_list", []).append(letter)
    return t


def _drawn_bbox(ax, renderer):
    """Display box of what an axes draws: its tight box (frame, ticks, labels
    and artists), or, when its axis is off and neither frame nor background
    is drawn (a Venn diagram), the union of its visible artists."""
    if ax.axison:
        return ax.get_tightbbox(renderer)
    boxes = [a.get_window_extent(renderer) for a in ax.get_children()
             if a is not ax.patch and a.get_visible()
             and not isinstance(a, (matplotlib.axis.Axis, matplotlib.spines.Spine))]
    return mtransforms.Bbox.union([b for b in boxes if b.width > 0 or b.height > 0])


def _panel_letters(fig, pairs, top=1.0, left=None):
    """Letters at the top of a row of panels, each at the left edge of what
    its axes draws (tick and axis labels included). pairs is a sequence of
    (letter, axes) and top the top of the row in figure coordinates: the top
    of the canvas, or lower for a row under another one on the same canvas.
    left, in figure coordinates, moves a letter out to that column when the
    panel starts further right, so that stacked panels share one column.
    Call after the layout, which must leave LETTER_BAND_PT free under top.
    Raises when the content of an axes reaches into the band of its letter."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tol = 0.5 * fig.dpi / 72   # half a point
    band_bottom = fig.transFigure.transform((0, top))[1] - LETTER_BAND_PT * fig.dpi / 72
    for letter, ax in pairs:
        bb = _drawn_bbox(ax, renderer)
        x = inv.transform(bb.p0)[0]
        t = _letter(fig, max(x if left is None else min(x, left), 0.0), top, letter)
        if bb.y1 > band_bottom + tol or t.get_window_extent(renderer).y0 < band_bottom - tol:
            raise ValueError(f"Panel {letter}: the content or the letter leaves the letter band")


def _write_png(rgb, directory, name, letters):
    """Write an RGB array as directory/name.png with 300 dpi in its metadata
    and check the file. A main figure must be MAIN_W_MM wide and at most
    MAIN_MAX_H_MM tall."""
    path = directory / f"{name}.png"
    Image.fromarray(rgb, "RGB").save(path, dpi=(DPI, DPI))
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (0, 0))
        if im.mode != "RGB" or any(abs(d - DPI) > 0.01 for d in dpi):
            raise ValueError(f"{name}.png is {im.mode} at {dpi} dpi, expected RGB at {DPI}")
        w_px, h_px = im.size
    if directory == FIG_DIR:
        target = MAIN_W_MM / 25.4 * DPI
        if abs(w_px - target) > 1 or h_px / DPI * 25.4 > MAIN_MAX_H_MM:
            raise ValueError(f"{name}.png is {w_px / DPI * 25.4:.1f} x {h_px / DPI * 25.4:.1f} mm, "
                             f"a main figure is {MAIN_W_MM:g} mm wide and at most {MAIN_MAX_H_MM:g} mm tall")
    LETTERS[name] = "".join(letters)
    generated.append(str(path.relative_to(ROOT)))
    logger.info("  OK: %s", name)


def _save(fig, directory, name):
    """Render the figure at 300 dpi with the tight bounding box. With a
    directory, write it there as an RGB PNG. With directory=None, keep it in
    PANELS[name] for _compose, which draws its letters, so a panel kept in
    memory must carry none."""
    letters = list(getattr(fig, "_panel_letter_list", []))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    arr = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"{name}: unexpected image shape {arr.shape}")
    if arr.shape[2] == 4 and (arr[..., 3] != 255).any():
        raise ValueError(f"{name}: the figure has transparent pixels")
    rgb = np.ascontiguousarray(arr[..., :3])
    if directory is None:
        if letters:
            raise ValueError(f"{name}: a panel composed by _compose must not draw letters")
        PANELS[name] = rgb
        logger.info("  OK: %s (panel kept for composition)", name)
        return
    _write_png(rgb, directory, name, letters)


def _compose(name, rows):
    """Compose panels kept in PANELS into FIG_DIR/name.png, pixel for pixel.

    rows is a list of rows, each a list of (letter, panel key, x_mm), where
    x_mm is the left edge of the panel PNG (its white padding included)
    measured from the left edge of the MAIN_W_MM canvas. Each panel is
    cropped to its ink (any channel below 255) and pasted with figimage, so
    nothing is resampled. Its letter goes at the top of the row, at the left
    edge of the cropped panel, and the panel content starts LETTER_BAND_PT
    below it. The first letter of every row takes the leftmost of those edges,
    so that stacked panels share one column, as in Figures 7 and 8. Raises when
    a panel does not fit, when two panels of a row come closer than
    COMPOSE_MIN_GAP_MM, when a letter leaves its band or when the saved PNG
    does not have the expected size."""
    mm, pt = DPI / 25.4, DPI / 72
    W = int(MAIN_W_MM * mm)   # the same pixel width as the figures drawn at 170 mm
    band, gap = int(round(LETTER_BAND_PT * pt)), int(round(COMPOSE_ROW_GAP_MM * mm))
    placed, firsts, y = [], {}, int(round(COMPOSE_TOP_MM * mm))
    for row in rows:
        crops = []
        for letter, key, x_mm in row:
            arr = PANELS[key]
            ink = (arr < 255).any(axis=2)
            r_ink, c_ink = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
            crop = arr[r_ink[0]:r_ink[-1] + 1, c_ink[0]:c_ink[-1] + 1]
            x_px = int(round(x_mm * mm)) + int(c_ink[0])
            if x_px < 0 or x_px + crop.shape[1] > W:
                raise ValueError(f"{name}: panel {letter} ({key}) does not fit in {MAIN_W_MM:g} mm")
            crops.append((letter, key, crop, x_px))
        crops.sort(key=lambda c: c[3])
        for (_, k0, c0, x0), (_, k1, _, x1) in zip(crops, crops[1:]):
            if x1 - (x0 + c0.shape[1]) < COMPOSE_MIN_GAP_MM * mm:
                raise ValueError(f"{name}: {k0} and {k1} are closer than {COMPOSE_MIN_GAP_MM:g} mm")
        top = y + band
        placed += [(letter, crop, x_px, y, top) for letter, _, crop, x_px in crops]
        firsts[crops[0][0]] = crops[0][3]
        y = top + max(c[2].shape[0] for c in crops) + gap
    H = y - gap + int(round(COMPOSE_BOTTOM_MM * mm))
    column = min(firsts.values())

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for _, crop, x_px, _, top in placed:
        canvas[top:top + crop.shape[0], x_px:x_px + crop.shape[1]] = crop
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.figimage(canvas, xo=0, yo=0, origin="upper")
    texts = []
    for letter, _, x_px, y_row, top in placed:
        x_letter = column if letter in firsts else x_px
        texts.append((_letter(fig, x_letter / W, 1 - y_row / H, letter), x_letter, top))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, x_letter, top in texts:
        bb = t.get_window_extent(renderer)   # display pixels, origin at the bottom
        if bb.y0 < H - top or bb.y1 > H + 0.5 or bb.x0 < x_letter - 0.5:
            raise ValueError(f"{name}: letter {t.get_text()} leaves its band")
    letters = list(fig._panel_letter_list)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    rgb = np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(buf.getvalue())))[..., :3])
    if rgb.shape[:2] != (H, W):
        raise ValueError(f"{name}: rendered {rgb.shape[1]} x {rgb.shape[0]} px, expected {W} x {H}")
    for letter, crop, x_px, _, top in placed:
        if not np.array_equal(rgb[top:top + crop.shape[0], x_px:x_px + crop.shape[1]], crop):
            raise ValueError(f"{name}: panel {letter} was not pasted pixel for pixel")
    _write_png(rgb, FIG_DIR, name, letters)


def compose_main_figures():
    """Figures 1, 3 and 4 as one PNG each, with the horizontal layout that
    LaTeX gave their panels when they were separate files: 1a centred over
    1b, 3a and 3b side by side over 3c, 4a over 4b."""
    plan = {
        "Figure1": ([("a", "fig1a", (MAIN_W_MM - FIG1A_W_MM) / 2)],
                    [("b", "fig1b", (MAIN_W_MM - FIG1B_W_MM) / 2)]),
        "Figure3": ([("a", "fig3a", 0.0), ("b", "fig3b", MAIN_W_MM - FIG3_HALF_W_IN * 25.4)],
                    [("c", "fig3c", 0.0)]),
        "Figure4": ([("a", "fig4a", 0.0)],
                    [("b", "fig4b", 0.0)]),
    }
    for name, rows in plan.items():
        missing = [key for row in rows for _, key, _ in row if key not in PANELS]
        if missing:
            _skip(name, f"panels {missing} were not drawn")
            continue
        _compose(name, list(rows))


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


def _stat(key):
    """Value of paper_statistics.json (generate_paper_results.py), so that
    counts printed inside figures come from the same source as the text.
    A missing key raises KeyError: no figure falls back to a typed number."""
    if not _STATS_CACHE:
        import json
        with open(PAPER_RESULTS / "paper_statistics.json", encoding="utf-8") as fh:
            _STATS_CACHE.update(json.load(fh))
    if key not in _STATS_CACHE:
        raise KeyError(f"'{key}' is not in paper_statistics.json, run generate_paper_results.py")
    return _STATS_CACHE[key]


def _fmt_n(key):
    return f"{int(_stat(key)):,}"


def _scanb_patient_cohort():
    """SCAN-B survival cohort of script 21 (src/data/scanb_cohort.py). Joined
    with the per-drug predictions it gives the 7,397 patients with overall
    survival, which include the 182 without a PAM50 subtype that the Cox
    models of Table 2 leave out."""
    from src.data.scanb_cohort import scanb_patient_cohort
    return scanb_patient_cohort(DATA)


# ============================================================
# MAIN FIGURES
# ============================================================

FIG1_TEXT = "#222222"   # dark text for labels and for nodes and boxes with a light fill
# Widths of the two panels of Figure 1 (mm), 1a centred over 1b by _compose
FIG1A_W_MM = 99.0
FIG1B_W_MM = 161.0


def _fill_text_color(fill):
    """White text on the dark fills (blue, vermillion), dark text on the light
    ones (orange, purple, grey), where white text is too faint at 7 pt."""
    return "white" if fill in (C_PRIMARY, C_ACCENT) else FIG1_TEXT


def fig1_graph_schema():
    """Figure 1a (panel fig1a of Figure1.png): heterogeneous graph schema.

    Drawn at its printed size (99 mm wide, centred over 1b), with data units
    in inches, so every label prints at 7 to 7.5 pt.
    The canvas is 0.2 in narrower than the placed width because the tight
    bounding box of _save adds 0.1 in of padding on each side.
    The cell line to drug response edges are dashed because they are the
    prediction targets and take no part in message passing (Methods). Every
    edge type is counted once per relation, so the PPI label gives the
    84,587 interactions (ppi_interactions), each stored in the graph as two
    directed edges (ppi_edges_bidir).
    """
    width_in, height_in = FIG1A_W_MM / 25.4 - 0.2, 2.5
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node and edge counts come from paper_statistics.json (script 51 summary
    # of the prebuilt graph), never typed here.
    x_left, x_right, y_top, y_bot = 1.0, 2.85, 1.72, 0.47
    nodes = [
        (f"Gene\n({_fmt_n('kg_genes')})",           x_left,  y_top, C_PRIMARY,   0.40),
        (f"TF\n({_fmt_n('kg_tfs')})",               x_left,  y_bot, C_SECONDARY, 0.36),
        (f"Drug\n({_fmt_n('kg_drugs')})",           x_right, y_top, C_HIGHLIGHT, 0.36),
        (f"Cell line\n({_fmt_n('kg_cell_lines')})", x_right, y_bot, C_NEUTRAL,   0.36),
    ]
    radius = {}
    for label, x, y, color, r in nodes:
        ax.add_patch(plt.Circle((x, y), r, facecolor=color, edgecolor="white",
                                linewidth=1.5, zorder=5))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=_fill_text_color(color), zorder=6)
        radius[(x, y)] = r
    gap = 0.05

    def _arrow(start, end, color, style="-|>", ls="-"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6, ls=ls,
                                    mutation_scale=9, shrinkA=0, shrinkB=0))

    # TF -> gene (CollecTRI regulation)
    _arrow((x_left, y_bot + radius[(x_left, y_bot)] + gap),
           (x_left, y_top - radius[(x_left, y_top)] - gap), C_SECONDARY)
    ax.text(x_left - 0.12, (y_top + y_bot) / 2, f"regulates\n{_fmt_n('collectri_edges')}",
            ha="right", va="center", fontsize=7, color=FIG1_TEXT, fontstyle="italic")

    # drug -> gene (drug-target annotations)
    _arrow((x_right - radius[(x_right, y_top)] - gap, y_top),
           (x_left + radius[(x_left, y_top)] + gap, y_top), C_HIGHLIGHT)
    ax.text((x_left + x_right) / 2, y_top + 0.07, f"targets {_fmt_n('drug_target_edges')}",
            ha="center", va="bottom", fontsize=7, color=FIG1_TEXT, fontstyle="italic")

    # cell line - drug response: dashed, it is the prediction target and is
    # excluded from message passing
    _arrow((x_right, y_bot + radius[(x_right, y_bot)] + gap),
           (x_right, y_top - radius[(x_right, y_top)] - gap), C_NEUTRAL, style="-", ls=(0, (3, 2)))
    ax.text(x_right - 0.12, (y_top + y_bot) / 2,
            f"response {_fmt_n('response_edges')}\n(prediction target,\nnot used in\nmessage passing)",
            ha="right", va="center", fontsize=7, color=FIG1_TEXT, fontstyle="italic",
            linespacing=1.15)

    # Gene-gene PPI edges are a self-loop on the gene node (they never reach
    # the drug node): drawn as a small loop sitting on top of the gene circle.
    # One count per interaction, like the other three edge types.
    if int(_stat("ppi_edges_bidir")) != 2 * int(_stat("ppi_interactions")):
        raise ValueError(f"ppi_edges_bidir ({_stat('ppi_edges_bidir')}) is not twice "
                         f"ppi_interactions ({_stat('ppi_interactions')})")
    r_gene = radius[(x_left, y_top)]
    loop_r = 0.15
    ax.add_patch(plt.Circle((x_left, y_top + r_gene + loop_r - 0.03), loop_r, fill=False,
                            edgecolor=C_PRIMARY, linewidth=1.6, zorder=4))
    ax.text(x_left + loop_r + 0.08, y_top + r_gene + loop_r - 0.03, f"PPI {_fmt_n('ppi_interactions')}",
            ha="left", va="center", fontsize=7, color=FIG1_TEXT, fontstyle="italic")

    ax.text(width_in - 0.05, height_in - 0.08, f"{_fmt_n('kg_nodes')} nodes in total",
            ha="right", va="top", fontsize=7, color=FIG1_TEXT)
    _save(fig, None, "fig1a")


def fig1b_pipeline():
    """Figure 1b (panel fig1b of Figure1.png): pipeline workflow diagram."""
    # Two parallel branches share the TF-activity representation, as in
    # Methods: the GNN + Integrated Gradients branch explains drug response in
    # cell lines, the per-drug Ridge transfer branch carries the predictions to
    # the clinical cohorts. The GNN does not feed the transfer.
    # Drawn at its printed size (161 mm, centred under 1a), with data units in
    # inches, so the box text prints at 7 pt. _placed_save checks that nothing
    # leaves the canvas, that no text is below 6 pt and that no texts overlap.
    # The prior-knowledge resources feed the knowledge graph, not the PRISM box.
    # AURORA-US is reached twice: the core TFs from Integrated Gradients are
    # tested there, and the Ridge transfer also predicts drug response there.
    # The selection rule uses SCAN-B alone (Methods). The fourth row holds the
    # two checks within TNBC, which take no part in the selection: FUSCC-TNBC
    # (scripts 00c, 68 and 69) receives the Ridge predictions of the whole
    # screen, and the TNBC lines of PRISM (script 67) use the measured AUC. An
    # arrow from the PRISM box to them would cross the whole figure, so their
    # box names the source.
    fig = _placed_canvas(FIG1B_W_MM / 25.4, 3.25 + 2 * plt.rcParams["savefig.pad_inches"])
    width_in, height_in = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.set_aspect("equal")
    ax.axis("off")

    n_cells = _fmt_n("prism_n_cell_lines")
    n_tfs = _fmt_n("prism_n_tf_features")
    n_cand = int(_stat("candidates_n_selected"))
    n_fu, n_rel = _fmt_n("fuscc_n_patients"), _fmt_n("fuscc_n_events")
    n_cl, n_cmp = _fmt_n("cltnbc_n_lines"), _fmt_n("cltnbc_n_comparison_lines")
    # The PRISM box and the box of the TNBC lines count the same cell lines
    _check_stat("prism_n_cell_lines", _stat("cltnbc_n_lines") + _stat("cltnbc_n_comparison_lines"))
    hh = 0.25
    dy = height_in - 2.5   # the three upper rows keep their spacing above the fourth
    y_top, y_mid, y_bot = 1.85 + dy, 1.2 + dy, 0.55 + dy
    y_tnbc, y_pk = y_bot - 0.8, 2.2 + dy
    c1, c2, c3, c4, c5 = 0.47, 1.61, 2.87, 4.21, 5.54
    boxes = {
        "data":   (c1, y_mid, f"PRISM + DepMap\n({n_cells} cell lines)",                 C_PRIMARY,   0.46),
        "tf":     (c2, y_mid, f"TF activities\n({n_tfs} TFs, ULM)",                       C_SECONDARY, 0.46),
        "gnn":    (c3, y_top, "Knowledge graph\n+ GNN (GraphSAGE)",                       C_ACCENT,    0.56),
        "ig":     (c4, y_top, "Integrated\nGradients (XAI)",                              C_HIGHLIGHT, 0.56),
        "ridge":  (c3, y_bot, "Per-drug Ridge\ntransfer",                           C_PRIMARY,   0.56),
        "surv":   (c4, y_bot, "Survival association\nSCAN-B, METABRIC,\nTCGA-BRCA",        C_ACCENT,    0.56),
        "aurora": (c5, y_top, "AURORA-US\nprimary vs\nmetastasis",                         C_NEUTRAL,   0.53),
        "select": (c5, y_bot, "Rule-based\ncandidate selection\n(SCAN-B)",                C_HIGHLIGHT, 0.53),
        "fuscc":  (c4, y_tnbc, f"FUSCC-TNBC\n{n_fu} patients,\n{n_rel} relapses",           C_NEUTRAL,   0.56),
        "cltnbc": (c5, y_tnbc, f"PRISM cell lines\n{n_cl} TNBC vs {n_cmp}\n(measured AUC)", C_NEUTRAL,   0.53),
    }
    for x, y, label, color, hw in boxes.values():
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - hw, y - hh), 2 * hw, 2 * hh, boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=color, edgecolor="none", zorder=5))
        ax.text(x, y, label, ha="center", va="center", fontsize=7, linespacing=1.15,
                fontweight="bold", color=_fill_text_color(color), zorder=6)

    def _edge(name, side, frac=0.0):
        """Point on a box border: side l, r, t or b, frac along it from the centre (-1 to 1)."""
        x, y, _, _, hw = boxes[name]
        return {"l": (x - hw, y + frac * hh), "r": (x + hw, y + frac * hh),
                "t": (x + frac * hw, y + hh), "b": (x + frac * hw, y - hh)}[side]

    def _arrow(start, end, lw=1.2, connectionstyle="arc3"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=C_PRIMARY, lw=lw,
                                    mutation_scale=8, shrinkA=1, shrinkB=1,
                                    connectionstyle=connectionstyle))

    _arrow(_edge("data", "r"), _edge("tf", "l"))
    _arrow(_edge("tf", "r", 0.4), _edge("gnn", "l", -0.5))
    _arrow(_edge("tf", "r", -0.4), _edge("ridge", "l", 0.5))
    _arrow(_edge("gnn", "r"), _edge("ig", "l"))
    _arrow(_edge("ridge", "r"), _edge("surv", "l"))
    _arrow(_edge("ig", "r"), _edge("aurora", "l"))
    _arrow(_edge("surv", "r"), _edge("select", "l"))
    # thin route: the Ridge transfer also predicts drug response in AURORA-US
    r_start, r_end = _edge("ridge", "t", 0.7), _edge("aurora", "b", -0.6)
    _arrow(r_start, r_end, lw=0.7)
    angle = np.degrees(np.arctan2(r_end[1] - r_start[1], r_end[0] - r_start[0]))
    nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
    ax.text((r_start[0] + r_end[0]) / 2 + 0.07 * nx, (r_start[1] + r_end[1]) / 2 + 0.07 * ny,
            "drug predictions", rotation=angle, rotation_mode="anchor", ha="center", va="bottom",
            fontsize=7, color=FIG1_TEXT, fontstyle="italic")

    # prior knowledge enters the knowledge graph
    pk_x = 0.88
    ax.text(pk_x, y_pk, "CollecTRI, OmniPath,\ndrug targets", ha="center", va="center",
            fontsize=7, color=FIG1_TEXT, fontstyle="italic", linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=C_NEUTRAL, lw=0.8))
    _arrow((pk_x + 0.58, y_pk), _edge("gnn", "l", 0.4))

    # The candidates go down to both checks within TNBC
    _arrow(_edge("select", "b", -0.75), _edge("fuscc", "t", 0.65))
    _arrow(_edge("select", "b", 0.75), _edge("cltnbc", "t", 0.75))
    # thin route, as to AURORA-US: the Ridge transfer also predicts every drug
    # in FUSCC-TNBC. It leaves the Ridge box downward and enters FUSCC-TNBC from
    # the left, under the label of the clinical branch.
    f_start, f_end = _edge("ridge", "b", -0.6), _edge("fuscc", "l")
    _arrow(f_start, f_end, lw=0.7, connectionstyle="angle,angleA=-90,angleB=180,rad=0")

    labels = [
        (c2 - 0.06, y_mid + hh + 0.05, "bottom", "shared representation\n(cell lines and patients)"),
        ((c3 + c4) / 2, y_pk, "center", "explanation branch (cell lines)"),
        ((c3 + c4) / 2, y_bot - hh - 0.1, "center", "clinical transfer branch (patients)"),
        ((c4 + c5) / 2, y_pk, "center", "core TFs"),
        (c5 - 0.02, (y_bot + y_tnbc) / 2, "center", f"{n_cand} candidates"),
        ((f_start[0] + f_end[0]) / 2, y_tnbc + 0.03, "bottom", "drug predictions"),
        ((c4 + c5) / 2, y_tnbc - hh - 0.12, "center", "TNBC checks (patients and cell lines)"),
    ]
    for x, y, va, text in labels:
        ax.text(x, y, text, ha="center", va=va, fontsize=7, linespacing=1.15,
                color=FIG1_TEXT, fontstyle="italic")

    _placed_save(fig, "fig1b", directory=None)


def fig2_model_comparison():
    """Fig 2: model comparison of Table 1, drawn at its placed width (170 mm).

    Panel a gives the global test Pearson correlation: one point per seed for
    the Random Forest (random_state), the GNNs and the hybrid (initialization),
    with the mean and standard deviation (diamond and bar), single fits for
    Ridge and the clinical transfer model, and the drug-mean reference as a
    dashed line. Panel b gives the per-drug Pearson correlations over the 1,447
    drugs as box plots (seed-42 models for the Random Forest, the GNNs and the
    hybrid). Whiskers reach 1.5
    times the interquartile range and values beyond them are not drawn, which
    the caption states. Values come from table1_model_comparison.csv and the
    per-seed and per-drug tables, so the figure never drifts from the table.
    No text crosses the drug-mean line, the zero line or the axes frame, which
    the function checks on the rendered layout and raises otherwise.
    Figure2.png is drawn on _placed_canvas at the full 170 mm, with the band
    of the panel letters at the top, and saved through _placed_save.
    """
    t1 = _csv(PAPER_RESULTS / "table1_model_comparison.csv")
    if t1 is None:
        return _skip("fig2_model_comparison", "table1_model_comparison.csv missing")
    t1i = t1.set_index("Model")
    # Blue baselines, orange GNNs, green hybrid (it uses GNN embeddings, so it
    # is neither), grey references
    cat_color = {"baseline": C_PRIMARY, "gnn": C_HIGHLIGHT, "hybrid": C_GREEN, "reference": C_NEUTRAL}
    cat_label = {"baseline": "Traditional ML", "gnn": "GNN", "hybrid": "Hybrid", "reference": "Reference"}
    # display name as in Table 1, category and per-drug table, keyed by the CSV Model column
    model_meta = {
        "RF_TF":              ("RF (TF activities)",     "baseline",  "per_drug_RF_TF.csv"),
        "Ridge_expression":   ("Ridge (expression)",     "baseline",  "per_drug_Ridge_expression.csv"),
        "SAGE_learned_emb":   ("GraphSAGE",              "gnn",       "per_drug_SAGE_learned_emb.csv"),
        "HGT_learned_emb":    ("HGT",                    "gnn",       "per_drug_HGT_learned_emb.csv"),
        "Ridge_expr_GNN_emb": ("Ridge (expr + GNN emb)", "hybrid",    "per_drug_Ridge_expr_GNN_emb.csv"),
        "Transfer_Ridge_TF":  ("Ridge (TF, transfer)",   "reference", "transfer_ridge_heldout_perdrug.csv"),
    }
    drug_mean_key = "Drug_mean_reference"
    missing = (set(model_meta) | {drug_mean_key}) - set(t1i.index)
    if missing:
        raise KeyError(f"table1_model_comparison.csv lacks the rows {sorted(missing)}")
    n_drugs = t1i.loc[list(model_meta) + [drug_mean_key], "N_drugs"].astype(int)
    if n_drugs.nunique() != 1:
        raise ValueError(f"Table 1 rows differ in the number of drugs: {n_drugs.to_dict()}")
    n_drugs = int(n_drugs.iloc[0])

    # Model rows by ascending global Pearson (best at the top), the reference
    # row under a divider, as in Table 1
    models = sorted((m for m in model_meta if t1i.loc[m, "Row_type"] == "model"),
                    key=lambda m: float(t1i.loc[m, "Global_Pearson"]))
    refs = [m for m in model_meta if t1i.loc[m, "Row_type"] == "reference"]
    rows = refs + models
    ypos = {m: float(i) for i, m in enumerate(refs)}
    ypos.update({m: len(refs) + 0.3 + i for i, m in enumerate(models)})
    divider = len(refs) - 0.35

    # Per-seed global Pearson of the rows fitted with five seeds (scripts 11, 12
    # and 57), whose means must give the Table 1 values
    per_seed = {}
    for fn, rename in (("fair_comparison_per_seed.csv", {}),
                       ("embedding_comparison_per_seed.csv",
                        {"Ridge(expression+GNN_emb)": "Ridge_expr_GNN_emb"})):
        ps = _csv(PAPER_RESULTS / fn)
        if ps is not None:
            for m, grp in ps.groupby("model"):
                if rename.get(m, m) in model_meta:
                    per_seed[rename.get(m, m)] = grp["global_pearson"].to_numpy()
    for k in rows:
        n_seeds = int(t1i.loc[k, "N_seeds"])
        if n_seeds > 1:
            v = per_seed.get(k)
            if v is None or len(v) != n_seeds or abs(v.mean() - float(t1i.loc[k, "Global_Pearson"])) > 5e-4:
                raise ValueError(f"Per-seed global Pearson of {k} does not reproduce Table 1")

    # Per-drug Pearson. Single fits must reproduce the Table 1 median, the Random
    # Forest, the GNNs and the hybrid show the seed-42 model (the caption says so).
    dists = {}
    for k in rows:
        pdf = _csv(PAPER_RESULTS / model_meta[k][2])
        if pdf is None:
            return _skip("fig2_model_comparison", f"{model_meta[k][2]} missing")
        dists[k] = pdf["pearson_r"].dropna().to_numpy()
        if len(dists[k]) != n_drugs:
            raise ValueError(f"{model_meta[k][2]} has {len(dists[k])} drugs, Table 1 has {n_drugs}")
        if int(t1i.loc[k, "N_seeds"]) == 1 and \
                abs(np.median(dists[k]) - float(t1i.loc[k, "Median_PerDrug_Pearson"])) > 5e-4:
            raise ValueError(f"Per-drug median of {k} does not reproduce Table 1")

    fs = 7
    marker_pt = 4.5
    fig = _placed_canvas(PLACED_W_IN, 2.9 + LETTER_BAND_PT / 72)
    ax_a, ax_b = fig.subplots(1, 2, sharey=True)

    # a) global Pearson: seeds as a small strip under the mean and SD. The
    # value labels are placed after the layout (see below).
    drug_mean = float(t1i.loc[drug_mean_key, "Global_Pearson"])
    ax_a.axvline(drug_mean, color=C_NEUTRAL, lw=1, ls=(0, (4, 2)), zorder=1)
    value_labels = {}
    for k in rows:
        yi, c = ypos[k], cat_color[model_meta[k][1]]
        mean, sd = float(t1i.loc[k, "Global_Pearson"]), t1i.loc[k, "Global_Pearson_SD"]
        if k in per_seed and int(t1i.loc[k, "N_seeds"]) > 1:
            v = per_seed[k]
            ax_a.scatter(v, yi - 0.24 + np.linspace(-0.07, 0.07, len(v)), s=7, color=c,
                         alpha=0.8, linewidths=0, zorder=3)
        ax_a.errorbar(mean, yi, xerr=None if pd.isna(sd) else float(sd), fmt="D", color=c,
                      markersize=marker_pt, capsize=2, elinewidth=1, capthick=1, zorder=4)
        value_labels[k] = ax_a.text(mean, yi, f"{mean:.3f}", ha="right", va="center",
                                    fontsize=fs, color=FIG1_TEXT, zorder=5)
    ax_a.text(drug_mean - 0.006, ypos[models[0]],
              str(t1i.loc[drug_mean_key, "Label"]).replace(" (", "\n("),
              ha="right", va="center", fontsize=fs, color=FIG1_TEXT, linespacing=1.15)
    lo = min([float(t1i.loc[k, "Global_Pearson"] - (0 if pd.isna(t1i.loc[k, "Global_Pearson_SD"])
                                                     else t1i.loc[k, "Global_Pearson_SD"]))
              for k in rows] + [v.min() for v in per_seed.values()])
    hi = max([float(t1i.loc[k, "Global_Pearson"]) for k in rows] + [drug_mean])
    # room for the value labels beside the leftmost and rightmost markers
    ax_a.set_xlim(lo - 0.055, hi + 0.07)
    ax_a.set_xlabel("Global Pearson correlation (test cell lines)", fontsize=fs, color=FIG1_TEXT)
    handles = [plt.Line2D([], [], marker="D", ls="", color=cat_color[c], markersize=4.5,
                          label=cat_label[c]) for c in cat_color]
    ax_a.legend(handles=handles, loc="upper left", fontsize=fs, handletextpad=0.3,
                borderpad=0.4, labelspacing=0.35, frameon=True, framealpha=1,
                edgecolor="#DDDDDD", labelcolor=FIG1_TEXT)

    # b) per-drug Pearson over the drugs, outliers not drawn
    bp = ax_b.boxplot([dists[k] for k in rows], vert=False, positions=[ypos[k] for k in rows],
                      widths=0.5, patch_artist=True, showfliers=False,
                      medianprops={"color": "black", "lw": 1.2},
                      whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, boxprops={"lw": 0.8})
    for patch, k in zip(bp["boxes"], rows):
        patch.set_facecolor(cat_color[model_meta[k][1]])
        patch.set_alpha(0.6)
    # Median labels above the boxes, in one column that starts just right of
    # the zero line so that no label crosses it (all medians are positive)
    ax_b.axvline(0, color=C_NEUTRAL, lw=0.6, zorder=0)
    median_labels = []
    for k in rows:
        med = np.median(dists[k])
        median_labels.append(ax_b.text(
            0, ypos[k] + 0.29, f"median {med:.3f}", ha="left", va="bottom", fontsize=fs,
            color=FIG1_TEXT,
            transform=mtransforms.offset_copy(ax_b.transData, fig=fig, x=3, units="points")))
    ax_b.set_xlabel(f"Per-drug Pearson correlation ({n_drugs:,} drugs)", fontsize=fs, color=FIG1_TEXT)

    # No vertical grid: every value is printed, and a grid line would run
    # through the value labels
    for ax in (ax_a, ax_b):
        ax.axhline(divider, color=C_NEUTRAL, lw=0.6)
        ax.tick_params(labelsize=fs, labelcolor=FIG1_TEXT, length=2, pad=2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_edgecolor(C_FRAME)
    ax_a.set_yticks([ypos[k] for k in rows])
    ax_a.set_yticklabels([model_meta[k][0] for k in rows])
    ax_a.set_ylim(-0.55, max(ypos.values()) + 0.75)

    band = LETTER_BAND_PT / 72 / fig.get_size_inches()[1]
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 1 - band), w_pad=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Value labels of panel a beside the marker (or the end of its SD bar), on
    # the side away from the drug-mean line, so that no label crosses that
    # line. Placed from the rendered layout, with 2 pt of clearance.
    pt_px = fig.dpi / 72
    line_px = ax_a.transData.transform((drug_mean, 0))[0]
    ax_px = ax_a.get_window_extent(renderer)
    for k, t in value_labels.items():
        mean, sd = float(t1i.loc[k, "Global_Pearson"]), t1i.loc[k, "Global_Pearson_SD"]
        sd = 0.0 if pd.isna(sd) else float(sd)
        side = 1 if mean > drug_mean else -1
        x_mark = ax_a.transData.transform((mean, 0))[0]
        # half-width of the diamond marker is markersize / sqrt(2)
        reach = max(marker_pt / np.sqrt(2) * pt_px, abs(ax_a.transData.transform((mean + sd, 0))[0] - x_mark))
        x_lab = ax_a.transData.inverted().transform((x_mark + side * (reach + 2 * pt_px), 0))[0]
        t.set_x(x_lab)
        t.set_ha("left" if side > 0 else "right")
        bb = t.get_window_extent(renderer)
        if (bb.x0 - 2 * pt_px < line_px < bb.x1 + 2 * pt_px
                or bb.x0 < ax_px.x0 + 2 * pt_px or bb.x1 > ax_px.x1 - 2 * pt_px):
            raise ValueError(f"Value label of {k} crosses the drug-mean line or the axes")
    zero_px = ax_b.transData.transform((0, 0))[0]
    for t in median_labels:
        if t.get_window_extent(renderer).x0 < zero_px + 2 * pt_px:
            raise ValueError("A median label of panel b crosses the zero line")
    _panel_letters(fig, (("a", ax_a), ("b", ax_b)))
    _placed_save(fig, "Figure2")


# Figure 3 has panels a and b side by side, each 0.49 of the 170 mm text
# width, with panel c at the full width underneath. The three panels are drawn
# at exactly those sizes, kept in memory and composed into Figure3.png, so
# their 7 pt text prints at 7 pt.
FIG3_FS = 7
FIG3_HALF_W_IN = 0.49 * 170 / 25.4
FIG3_FULL_W_IN = 170 / 25.4
FIG3_AB_H_IN = 2.95
FIG3_C_H_IN = 3.3
FIG3_CORE_TEXT = "#555555"   # bars (panel a) and lines (panel b) of the core TFs not in the top 20 of every seed


def _fig3_rc():
    """Style of the Figure 3 panels: 7 pt text, dark labels and thin lines."""
    return plt.rc_context({
        "font.size": FIG3_FS, "axes.labelsize": FIG3_FS, "axes.titlesize": FIG3_FS,
        "xtick.labelsize": FIG3_FS, "ytick.labelsize": FIG3_FS, "legend.fontsize": FIG3_FS,
        "legend.title_fontsize": FIG3_FS,
        "axes.edgecolor": C_FRAME,
        "text.color": FIG1_TEXT, "axes.labelcolor": FIG1_TEXT,
        "xtick.labelcolor": FIG1_TEXT, "ytick.labelcolor": FIG1_TEXT,
        "axes.linewidth": 0.8, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3,
    })


def _fig3_canvas(width_in, height_in):
    """Figure whose saved PNG measures width_in x height_in. The canvas is
    narrower and shorter by the padding that bbox_inches="tight" adds on each
    side, and an invisible rectangle over the whole canvas makes the tight
    bounding box equal to the canvas, so the size does not depend on the
    content. _fig3_save checks that nothing spills out of the canvas."""
    pad = plt.rcParams["savefig.pad_inches"]
    fig = plt.figure(figsize=(width_in - 2 * pad, height_in - 2 * pad))
    fig.patches.append(mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                          fill=False, linewidth=0, alpha=0))
    return fig


def _fig3_save(fig, name, directory=None):
    """Save a Figure 3 panel after checking that every artist lies inside the
    canvas (so the PNG has its placed size), that no text is below 7 pt and
    that no tick label is a number of four digits or more without thousands
    separators. The panels are kept in memory for _compose by default."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > w + 1e-3 or bb.y1 > h + 1e-3:
        raise ValueError(f"{name}: an artist spills out of the canvas, the PNG would not have its placed size")
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FIG3_FS - 1e-6]
    if small:
        raise ValueError(f"{name}: text below {FIG3_FS} pt: {small[:5]}")
    _check_tick_commas(fig, name)
    _save(fig, directory, name)


def _fig3_legend_above(ax, handles):
    """Legend in two rows above the axes, left-aligned with the frame, the
    same in panels a and b so that their frames line up side by side."""
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=(len(handles) + 1) // 2,
              frameon=False, borderaxespad=0, borderpad=0.2, handlelength=1.4,
              handletextpad=0.4, columnspacing=1.0, labelspacing=0.3)


def _fig3_robust_and_core(stab):
    """Top 20 in all three seeds and the ten core TFs from paper_statistics.json,
    checked against the per-seed ranks of tf_importance_ig_stability.csv."""
    robust = list(_stat("ig_multi_seed_top20_all_names"))
    core = list(_stat("ig_multi_seed_core10_names"))
    if set(robust) != set(stab.loc[stab["in_top20_all_seeds"].astype(bool), "tf"]):
        raise ValueError("ig_multi_seed_top20_all_names does not match tf_importance_ig_stability.csv")
    in_core = stab["tf"].isin(core)
    if in_core.sum() != len(core) or \
            stab.loc[in_core, "mean_rank"].max() >= stab.loc[~in_core, "mean_rank"].min():
        raise ValueError("ig_multi_seed_core10_names are not the TFs with the lowest mean rank")
    if not set(robust) <= set(core):
        raise ValueError("A TF in the top 20 of all seeds is not in the core set")
    return robust, core


def fig3_tf_importance():
    """Figure 3a (panel fig3a of Figure3.png): top 20 TFs by global IG importance, seed 42.

    Canonical IG ranking of the seed-42 GNN with the mean-train baseline. Each
    point is one TNBC cell line, its |attribution| averaged over the drugs, and
    each bar is the mean of its points. A line keeps the same vertical offset in
    every row, so no random jitter is needed. The bars follow the colors of panel
    b: vermillion for the TFs in the top 20 of all three seeds and dark grey for
    the other core TFs. The bars must be the ig_top20_names of
    paper_statistics.json.
    """
    tf = _csv(PAPER_RESULTS / "tf_importance_ig_top20.csv")
    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    per_line = _csv(PAPER_RESULTS / "tf_importance_ig_top20_per_line.csv")
    if tf is None or stab is None or per_line is None:
        return _skip("Fig3a_tf_importance_ig", "tf_importance_ig_top20.csv, tf_importance_ig_stability.csv or "
                                               "tf_importance_ig_top20_per_line.csv missing")
    top = tf.sort_values("rank").head(20)
    top20 = top["tf"].tolist()
    if top20 != list(_stat("ig_top20_names")):
        raise ValueError("tf_importance_ig_top20.csv does not match ig_top20_names")
    robust, core = _fig3_robust_and_core(stab)

    # The points are the top 20 x the IG lines, and each bar is the mean of its points
    n_cells = _stat("ig_n_cells")
    lines = sorted(per_line["cell_depmap_id"].unique())
    n_per_tf = per_line.groupby("tf")["cell_depmap_id"].nunique()
    if set(per_line["tf"]) != set(top20) or len(lines) != n_cells or len(per_line) != len(top20) * n_cells \
            or (n_per_tf != n_cells).any() or (per_line["n_drugs"] != _stat("ig_n_drugs")).any():
        raise ValueError("tf_importance_ig_top20_per_line.csv is not the top 20 TFs x the IG cell lines over all drugs")
    bar_from_points = per_line.groupby("tf")["abs_attr_mean"].mean().loc[top20].to_numpy()
    if not np.allclose(bar_from_points, top["importance_mean"].to_numpy(), rtol=1e-9, atol=0):
        raise ValueError("The bars of Figure 3a are not the mean of their points")
    # The core TFs missing from this panel are named in the caption
    outside = [t for t in core if t not in top20]
    if outside != list(_stat("ig_core10_outside_seed42_top20_names")):
        raise ValueError(f"Core TFs outside the seed-42 top 20: {outside}, "
                         f"paper_statistics.json has {_stat('ig_core10_outside_seed42_top20_names')}")
    _check_stat("ig_core10_in_seed42_top20_n", len(core) - len(outside))

    def bar_color(name):
        if name in robust:
            return C_ACCENT
        return FIG3_CORE_TEXT if name in core else C_NEUTRAL

    with _fig3_rc():
        fig = _fig3_canvas(FIG3_HALF_W_IN, FIG3_AB_H_IN)
        ax = fig.add_subplot()
        top = top.iloc[::-1]
        row = {t: i for i, t in enumerate(top["tf"])}
        ax.barh(range(len(top)), top["importance_mean"], height=0.7, linewidth=0, zorder=1,
                color=[bar_color(t) for t in top["tf"]])
        offset = dict(zip(lines, np.linspace(-0.24, 0.24, n_cells)))
        ax.scatter(per_line["abs_attr_mean"],
                   per_line["tf"].map(row) + per_line["cell_depmap_id"].map(offset),
                   s=5, facecolors="white", edgecolors=FIG1_TEXT, linewidths=0.45, zorder=3, clip_on=False)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["tf"])
        ax.set_ylim(-0.6, len(top) - 0.4)
        ax.set_xlim(0, per_line["abs_attr_mean"].max() * 1.04)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.002))
        ax.set_xlabel("Mean |attribution| (Integrated Gradients)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", linewidth=0.5)
        _fig3_legend_above(ax, [
            mpatches.Patch(facecolor=C_ACCENT, label="Top 20 in all three seeds"),
            mpatches.Patch(facecolor=FIG3_CORE_TEXT, label="Other core TFs"),
            mpatches.Patch(facecolor=C_NEUTRAL, label="Other TFs"),
            plt.Line2D([], [], ls="", marker="o", markersize=2.5, markerfacecolor="white",
                       markeredgecolor=FIG1_TEXT, markeredgewidth=0.45, label="One TNBC cell line"),
        ])
        fig.tight_layout(pad=0.15)
        # clip_on=False hides a point that leaves the axis, so it has to be refused here
        lo, hi = ax.get_xlim()
        if per_line["abs_attr_mean"].min() < lo or per_line["abs_attr_mean"].max() > hi:
            raise ValueError("A point of Figure 3a falls outside the x axis")
        _fig3_save(fig, "fig3a")


def _spread_labels(targets, gap, lo, hi):
    """Label positions (display units) as close as possible to targets, at
    least gap apart and within [lo, hi]. Overlapping labels are merged into
    groups centred on the mean of their targets until no two groups overlap."""
    order = np.argsort(targets)
    t = np.asarray(targets, dtype=float)[order]
    groups = [[i] for i in range(len(t))]

    def place(g):
        c = t[g].mean() - gap * (len(g) - 1) / 2
        c = min(max(c, lo), hi - gap * (len(g) - 1))
        return [c + gap * k for k in range(len(g))]

    merged = True
    while merged:
        merged = False
        for j in range(len(groups) - 1):
            if place(groups[j])[-1] + gap > place(groups[j + 1])[0] + 1e-9:
                groups[j:j + 2] = [groups[j] + groups[j + 1]]
                merged = True
                break
    pos = np.empty(len(t))
    for g in groups:
        pos[order[g]] = place(g)
    return pos


def fig3b_ig_stability():
    """Figure 3b (panel fig3b of Figure3.png): IG rank of each TF under the three seeds.

    One line per TF in the top 20 of at least one seed, on a logarithmic rank
    axis. The ten core TFs of paper_statistics.json (lowest mean rank) are
    labelled at the right in a column that holds no data, with leader lines,
    and the TFs in the top 20 of all three seeds are drawn in vermillion. The
    top-20 threshold is labelled in an empty strip left of seed 42, so no line
    runs over the label. The function checks both placements on the rendered
    layout and raises if a label overlaps another label or the data.
    """
    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    if stab is None:
        return _skip("Fig3b_ig_stability", "tf_importance_ig_stability.csv missing")
    robust, core = _fig3_robust_and_core(stab)
    # one rank column per training seed, named rank_<seed>
    cols = [c for c in stab.columns if c.startswith("rank_")]
    if len(cols) != int(_stat("ig_n_seeds")):
        raise ValueError(f"tf_importance_ig_stability.csv has {len(cols)} seeds, ig_n_seeds is {_stat('ig_n_seeds')}")
    top20_any = stab[(stab[cols] <= 20).any(axis=1)].sort_values("mean_rank")
    xs = np.arange(len(cols))
    rank_max = float(top20_any[cols].max().max())

    with _fig3_rc():
        fig = _fig3_canvas(FIG3_HALF_W_IN, FIG3_AB_H_IN)
        ax = fig.add_subplot()
        ax.set_yscale("log")
        # others first, then the core TFs, then the top 20 of all seeds on top
        for _, row in top20_any.iterrows():
            tf_name = row["tf"]
            if tf_name in robust:
                style = dict(color=C_ACCENT, lw=1.3, markersize=3.2, alpha=1, zorder=4)
            elif tf_name in core:
                style = dict(color=FIG3_CORE_TEXT, lw=0.9, markersize=2.6, alpha=1, zorder=3)
            else:
                style = dict(color=C_NEUTRAL, lw=0.6, markersize=2.0, alpha=0.5, zorder=2)
            ax.plot(xs, row[cols].to_numpy(dtype=float), marker="o", markeredgewidth=0, **style)

        # threshold drawn up to the last seed only, so it stays out of the label column
        x_lim = (-0.55, xs[-1] + 0.95)
        thr_end = xs[-1] + 0.08
        ax.plot([x_lim[0], thr_end], [20.5, 20.5], color=FIG1_TEXT, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(*x_lim)
        ax.set_ylim(10 ** np.ceil(np.log10(rank_max)), 0.75)   # inverted: rank 1 at the top
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if t <= 10 ** np.ceil(np.log10(rank_max))]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("rank_", "") for c in cols])
        ax.set_xlabel("Training seed")
        ax.set_ylabel("IG rank (1 = most important)")
        ax.grid(False)
        _fig3_legend_above(ax, [
            plt.Line2D([], [], color=C_ACCENT, lw=1.3, marker="o", markersize=3.2,
                       markeredgewidth=0, label="Top 20 in all three seeds"),
            plt.Line2D([], [], color=FIG3_CORE_TEXT, lw=0.9, marker="o", markersize=2.6,
                       markeredgewidth=0, label="Other core TFs"),
            plt.Line2D([], [], color=C_NEUTRAL, lw=0.6, marker="o", markersize=2.0,
                       markeredgewidth=0, alpha=0.5, label="Other TFs"),
        ])
        fig.tight_layout(pad=0.15)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pt_px = fig.dpi / 72
        to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform

        # top-20 threshold label in the empty strip left of the first seed
        thr = ax.text(-0.5, 20.5, "top 20", ha="left", va="bottom", color=FIG1_TEXT,
                      transform=mtransforms.offset_copy(ax.transData, fig=fig, y=1, units="points"))
        bb = thr.get_window_extent(renderer)
        if bb.x1 > to_disp((xs[0], 1))[0] - 3 * pt_px:
            raise ValueError("The top-20 label reaches the seed-42 points")

        # core TF labels in the column right of the last seed
        last = cols[-1]
        lab = top20_any[top20_any["tf"].isin(core)]
        y_disp = np.array([to_disp((xs[-1], r))[1] for r in lab[last]])
        ax_bb = ax.get_window_extent(renderer)
        gap = FIG3_FS * 1.2 * pt_px
        y_lab = _spread_labels(y_disp, gap, ax_bb.y0 + gap / 2, ax_bb.y1 - gap / 2)
        x_text = xs[-1] + 0.32
        texts = []
        for (_, row), yl in zip(lab.iterrows(), y_lab):
            r_lab = to_data((0, yl))[1]
            is_robust = row["tf"] in robust
            ax.plot([xs[-1] + 0.06, x_text - 0.04], [row[last], r_lab], color=FIG3_CORE_TEXT,
                    lw=0.4, zorder=5, clip_on=False)
            texts.append(ax.text(x_text, r_lab, row["tf"], ha="left", va="center", zorder=6,
                                 color=C_ACCENT if is_robust else FIG1_TEXT,
                                 fontweight="bold" if is_robust else "normal"))
        fig.canvas.draw()
        boxes = sorted((t.get_window_extent(renderer) for t in texts), key=lambda b: b.y0)
        if any(b1.y0 < b0.y1 for b0, b1 in zip(boxes, boxes[1:])):
            raise ValueError("Two core TF labels overlap")
        if any(b.x1 > ax_bb.x1 for b in boxes):
            raise ValueError("A core TF label runs past the axes frame")
        if min(b.x0 for b in boxes) < to_disp((thr_end, 1))[0] + 2 * pt_px:
            raise ValueError("The core TF labels reach the top-20 threshold line")
        _fig3_save(fig, "fig3b")


# Figures 2 and 4 to 8 (Figure2.png, the panels of Figure4.png, Figure5.png,
# Figure6.png, Figure7.png and Figure8.png) are drawn at their placed size, 170 mm wide,
# with 7 pt text, so LaTeX includes each PNG at width=\linewidth without
# scaling it and every text prints at 7 pt. _placed_save refuses a figure with
# text below 6 pt or with overlapping texts.
PLACED_W_IN = MAIN_W_MM / 25.4
PLACED_FS = 7
PLACED_MIN_FS = 6
C_LIGHT_ORANGE = "#F2C96E"   # C_HIGHLIGHT lightened toward white (Q4 of Figure 4b)
C_FRAME = "#555555"          # neutral axes frame, so that no frame color reads as a data category


def _placed_rc():
    """7 pt dark text, thin neutral frames and ticks, for the figures drawn at
    their placed size. The frames are dark grey instead of the global blue,
    which in these figures is a data color (predicted sensitive, primary
    tumor, SCAN-B)."""
    return plt.rc_context({
        "font.size": PLACED_FS, "axes.labelsize": PLACED_FS, "axes.titlesize": PLACED_FS,
        "xtick.labelsize": PLACED_FS, "ytick.labelsize": PLACED_FS,
        "legend.fontsize": PLACED_FS, "legend.title_fontsize": PLACED_FS,
        "axes.edgecolor": C_FRAME,
        "text.color": FIG1_TEXT, "axes.labelcolor": FIG1_TEXT,
        "xtick.labelcolor": FIG1_TEXT, "ytick.labelcolor": FIG1_TEXT,
        "axes.linewidth": 0.6, "grid.linewidth": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2, "axes.titlepad": 3,
    })


def _placed_canvas(width_in, height_in):
    """Figure whose saved PNG measures width_in x height_in. The canvas is
    smaller by the padding that bbox_inches="tight" adds on each side, and an
    invisible rectangle over the whole canvas makes the tight bounding box
    equal to the canvas, so the PNG size does not depend on the content."""
    pad = plt.rcParams["savefig.pad_inches"]
    fig = plt.figure(figsize=(width_in - 2 * pad, height_in - 2 * pad))
    fig.patches.append(mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                          fill=False, linewidth=0, alpha=0))
    return fig


def _inch_rect(fig, left, bottom, width, height):
    """Axes rectangle given in inches from the lower left corner of the canvas."""
    w, h = fig.get_size_inches()
    return [left / w, bottom / h, width / w, height / h]


def _placed_texts(fig):
    """Texts drawn in the figure: free texts, legends, titles, axis labels and
    the tick labels that fall inside the view limits of visible axes."""
    texts = list(fig.texts)
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]
    # Legends kept with ax.add_artist when an axes carries two of them
    legends += [a for ax in fig.axes for a in ax.artists if isinstance(a, matplotlib.legend.Legend)]
    for leg in legends:
        texts += [*leg.get_texts(), leg.get_title()]
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, *ax.texts]
        if not ax.axison:
            continue
        texts += [ax.xaxis.label, ax.yaxis.label]
    texts += _visible_tick_labels(fig)
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def _visible_tick_labels(fig):
    """Tick labels that fall inside the view limits of the visible axes."""
    labels = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            eps = 1e-9 * (hi - lo)
            labels += [lab for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels())
                       if lo - eps <= loc <= hi + eps]
    return [t for t in labels if t.get_visible() and t.get_text().strip()]


NO_THOUSANDS_SEP = re.compile("^[-−]?[0-9]{4,}$")


def _check_tick_commas(fig, name):
    """Raise when a visible tick label is a number of four digits or more
    written without thousands separators (1000 instead of 1,000). Call after
    the figure is drawn."""
    bare = [t.get_text() for t in _visible_tick_labels(fig) if NO_THOUSANDS_SEP.match(t.get_text())]
    if bare:
        raise ValueError(f"{name}: tick labels without thousands separators: {bare[:5]}")


def _placed_save(fig, name, directory=FIG_DIR):
    """Save a figure drawn on _placed_canvas after checking that every artist
    lies inside the canvas (so the PNG keeps its placed size), that no text is
    below 6 pt (math text included), that no two texts overlap and that every
    tick label of four digits or more has thousands separators. The
    supplementary figures pass directory=SFIG_DIR, and the panels 1b, 4a and 4b
    directory=None, which keeps them in memory for _compose."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(renderer)
    w, h = fig.get_size_inches()
    texts = _placed_texts(fig)
    if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > w + 1e-3 or bb.y1 > h + 1e-3:
        canvas = fig.bbox
        out = [t.get_text() for t in texts
               if not canvas.contains(*t.get_window_extent(renderer).p0)
               or not canvas.contains(*t.get_window_extent(renderer).p1)]
        raise ValueError(f"{name}: an artist spills out of the canvas, the PNG would not have "
                         f"its placed size (texts outside: {out[:5]})")
    small = [t.get_text() for t in texts if t.get_fontsize() < PLACED_MIN_FS]
    if small:
        raise ValueError(f"{name}: text below {PLACED_MIN_FS} pt: {small[:5]}")
    # Math-text sub- and superscripts shrink to 0.7 of the font size
    math = [t.get_text() for t in texts if t.get_text().count("$") >= 2]
    if math:
        raise ValueError(f"{name}: math text would print below {PLACED_MIN_FS} pt: {math[:5]}")
    # Text.get_window_extent leaves out the leader line of an annotation
    boxes = [matplotlib.text.Text.get_window_extent(t, renderer) for t in texts]
    tol = 0.5 * fig.dpi / 72   # half a point
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a.x0 + tol < b.x1 and b.x0 + tol < a.x1 and a.y0 + tol < b.y1 and b.y0 + tol < a.y1:
                raise ValueError(f"{name}: texts overlap: {texts[i].get_text()!r} and {texts[j].get_text()!r}")
    _check_tick_commas(fig, name)
    _save(fig, directory, name)


def _fmt_p_2sig(p):
    """p-value with two significant digits, written as in Tables 3 and 4:
    e-notation below 0.01 (3.8e-5), otherwise decimal keeping the trailing
    zero (0.060, 0.10). Plain text, because math-text exponents would print
    at 0.7 of the font size, below 6 pt."""
    if not 0 < p <= 1:
        raise ValueError(f"p-value out of range: {p}")
    if p < 0.01:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}e{int(exp)}"
    return f"{p:#.2g}"


def _fmt_signed(x, digits=2):
    """Signed number with a typographic minus sign."""
    return f"{x:+.{digits}f}".replace("-", "\u2212")


def _drug_display(name):
    """PRISM drug identifier as printed in a figure. Compound codes go in
    upper case (kw-2478 to KW-2478, bnc105 to BNC105, cpp to CPP), generic
    names are capitalized as in Figures 3C and 4 (Brigatinib), with short
    letter groups in upper case (Tyrphostin-AG-1296), and systematic chemical
    names, which start with a digit, stay as they are."""
    if name[:1].isdigit():
        return name
    parts = name.split("-")
    if all(len(p) <= 3 or any(ch.isdigit() for ch in p) for p in parts):
        return name.upper()
    return "-".join([parts[0].capitalize()] +
                    [p.upper() if p.isalpha() and len(p) <= 3 else p for p in parts[1:]])


def fig6_drug_ranking_meta():
    """Figure 6 (Figure6.png): the 30 drugs with the lowest Fisher p-value.

    Bar length is -log10 of the Benjamini-Hochberg adjusted Fisher p-value of
    meta_analysis_fisher.csv. Script 21 combines the p-values with chi2.sf, so
    no p-value underflows to zero and no bar is capped, which the function
    checks. Bar color gives the cohorts in which the drug is individually
    FDR < 0.05, with the colors of Figure 5 (SCAN-B blue, METABRIC orange, both
    green), and the legend lists only the categories present (TCGA-BRCA has no
    significant drug). Drug names go through _drug_display, so compound codes
    print in upper case and generic names capitalized. Drawn at 170 x 100 mm.
    """
    fisher = _csv(PAPER_RESULTS / "meta_analysis_fisher.csv")
    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    if fisher is None or scanb is None or metabric is None:
        return _skip("Figure6", "meta_analysis_fisher.csv or the survival CSVs missing in paper/results/")
    if (fisher[["fisher_p", "fisher_padj"]] <= 0).any().any():
        raise ValueError("meta_analysis_fisher.csv has a Fisher p-value of 0, rerun script 21 (chi2.sf)")

    n_top = 30
    sig = fisher[fisher["fisher_padj"] < 0.05]
    if len(sig) != int(_stat("Fisher_drugs_FDR005")):
        raise ValueError(f"{len(sig)} Fisher-significant drugs, paper_statistics.json has "
                         f"{_stat('Fisher_drugs_FDR005')}")
    ranked = sig.sort_values(["fisher_p", "drug"], kind="mergesort").reset_index(drop=True)
    if len(ranked) > n_top and ranked.loc[n_top, "fisher_p"] == ranked.loc[n_top - 1, "fisher_p"]:
        raise ValueError(f"Tie in the Fisher p-value at the top-{n_top} cut")
    top = ranked.head(n_top).copy()
    first = top.iloc[0]
    if first["drug"] != _stat("fisher_min_p_drug") or \
            not np.isclose(first["fisher_padj"], float(_stat(f"fisher_{first['drug']}_padj")), rtol=0.05):
        raise ValueError(f"The top drug {first['drug']} does not match fisher_min_p_drug and its padj key")
    top["neg_log_padj"] = -np.log10(top["fisher_padj"])

    s_sig = set(scanb.loc[scanb["padj"] < 0.05, "drug"])
    m_sig = set(metabric.loc[metabric["padj"] < 0.05, "drug"])
    top["cat"] = ["both" if d in s_sig and d in m_sig else "scanb" if d in s_sig
                  else "metabric" if d in m_sig else "neither" for d in top["drug"]]
    cat_color = {"both": C_GREEN, "scanb": C_PRIMARY, "metabric": C_HIGHLIGHT, "neither": C_NEUTRAL}
    cat_label = {"both": "SCAN-B and METABRIC", "scanb": "SCAN-B only",
                 "metabric": "METABRIC only", "neither": "Neither (Fisher only)"}

    with _placed_rc():
        fig = _placed_canvas(PLACED_W_IN, 100 / 25.4)
        w, h = fig.get_size_inches()
        ax = fig.add_axes(_inch_rect(fig, 2.6, 0.32, w - 2.6 - 0.12, h - 0.32 - 0.04))
        y = np.arange(len(top))
        ax.barh(y, top["neg_log_padj"], height=0.72, color=[cat_color[c] for c in top["cat"]],
                edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels([_drug_display(d) for d in top["drug"]])
        ax.set_ylim(len(top) - 0.5, -0.5)
        ax.set_xlim(0, np.ceil(top["neg_log_padj"].max()) + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.set_xlabel("\u2212log10 of the Benjamini-Hochberg adjusted Fisher p-value")
        ax.grid(True, axis="x", color="#E8E9EB")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        present = [c for c in cat_color if c in set(top["cat"])]
        ax.legend(handles=[mpatches.Patch(facecolor=cat_color[c], label=cat_label[c]) for c in present],
                  title="Individually FDR < 0.05 in", loc="lower right", frameon=False,
                  alignment="left", handlelength=1.2, handleheight=0.8, borderaxespad=0.4)
        # The drug names set the left margin: move the axes so that the longest
        # name ends just inside the canvas.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_w = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels()) / fig.dpi
        left = label_w + 0.08
        ax.set_position(_inch_rect(fig, left, 0.32, w - left - 0.12, h - 0.32 - 0.04))
        _placed_save(fig, "Figure6")


KM_DRUGS_SPLIT = ["paclitaxel", "epirubicin", "erlotinib", "entinostat"]
KM_DRUGS_QUINTILE = ["paclitaxel", "entinostat"]
# Blue for predicted sensitive and vermillion for predicted resistant, as in
# Figure 3C and Supplementary Figure 4. Q4 is a light orange because orange
# and vermillion come close under deuteranopia.
KM_SENSITIVE, KM_RESISTANT = C_PRIMARY, C_ACCENT
KM_QUINTILE_COLORS = [C_PRIMARY, C_SKYBLUE, C_NEUTRAL, C_LIGHT_ORANGE, C_ACCENT]
# Geometry shared by panels 4a and 4b, so that their axes and their tables of
# numbers at risk line up when _compose stacks them in Figure4.png (inches).
# Under each row of axes go the tick labels and "Time (years)" (KM_TICK_BLOCK),
# a gap and the table, one KM_RISK_PITCH per line, and above the next row its
# title (KM_TITLE_BLOCK). KM_NUM_HALF leaves room left of x = 0 for half of the
# widest number, and the key of a table row is a line KM_KEY_LINE long.
KM_LEFT, KM_RIGHT, KM_WSPACE = 0.50, 0.06, 0.50
KM_TOP, KM_AX_H = 0.40, 1.10
KM_TICK_BLOCK, KM_RISK_GAP, KM_RISK_PITCH, KM_TITLE_BLOCK = 0.27, 0.03, 0.115, 0.25
KM_NUM_HALF, KM_KEY_LINE = 0.18, 0.12


def _km_risk_block(nrisk):
    """Height under a row of axes with a table of nrisk rows and its header."""
    return KM_TICK_BLOCK + KM_RISK_GAP + (nrisk + 1) * KM_RISK_PITCH


def _km_canvas_h(nrows, nrisk):
    """Height of the PNG of nrows rows of axes, each with a table of nrisk rows."""
    block = _km_risk_block(nrisk)
    return 0.2 + KM_TOP + block + 0.02 + nrows * KM_AX_H + (nrows - 1) * (block + KM_TITLE_BLOCK)


def _km_axes(fig, nrows, nrisk):
    """Two columns of Kaplan-Meier axes with the fixed geometry above. Returns
    the axes and the bottom edge of each one in inches."""
    w, _ = fig.get_size_inches()
    ax_w = (w - KM_LEFT - KM_RIGHT - KM_WSPACE) / 2
    block = _km_risk_block(nrisk)
    axes, bottoms = [], []
    for r in range(nrows):
        bottom = block + 0.02 + (nrows - 1 - r) * (KM_AX_H + block + KM_TITLE_BLOCK)
        for c in range(2):
            axes.append(fig.add_axes(_inch_rect(fig, KM_LEFT + c * (ax_w + KM_WSPACE), bottom,
                                                ax_w, KM_AX_H)))
            bottoms.append(bottom)
    return axes, bottoms


def _km_style(ax, drug, stat_text, xmax, left):
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 0.5, 2))
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.round(np.arange(0, 1.0 + 1e-9, 0.2), 1))
    ax.grid(True, axis="y", color="#E8E9EB")
    ax.grid(False, axis="x")
    ax.set_title(drug.capitalize(), loc="left", fontweight="bold")
    ax.set_title(stat_text, loc="right")
    ax.set_xlabel("Time (years)")
    if left:
        ax.set_ylabel("Overall survival")


def _km_risk_table(fig, ax, bottom_in, times, rows, colors, labels):
    """Numbers at risk under ax: a header and one row per group, with the
    numbers centred on the x ticks in dark text, because the lighter curve
    colors (Q2, Q4) are not legible at 7 pt. Each row is keyed in the margin
    left of x = 0 by a line of the group color, followed by its label when it
    has one. The header starts where the key lines start."""
    w, h = fig.get_size_inches()
    mixed = mtransforms.blended_transform_factory(ax.transData, fig.transFigure)
    key_right = ax.get_position().x0 * w - KM_NUM_HALF
    y_head = bottom_in - KM_TICK_BLOCK - KM_RISK_GAP - KM_RISK_PITCH / 2
    row_y = [(y_head - (k + 1) * KM_RISK_PITCH) / h for k in range(len(rows))]
    renderer = fig.canvas.get_renderer()
    label_w = 0.0
    for y, row, label in zip(row_y, rows, labels):
        for t, n in zip(times, row):
            ax.text(t, y, f"{int(n):,}", transform=mixed, ha="center", va="center", clip_on=False)
        if label:
            text = fig.text(key_right / w, y, label, ha="right", va="center")
            label_w = max(label_w, text.get_window_extent(renderer).width / fig.dpi)
    line_right = key_right - (label_w + 0.03 if label_w else 0)
    for y, color in zip(row_y, colors):
        fig.add_artist(plt.Line2D([(line_right - KM_KEY_LINE) / w, line_right / w], [y, y],
                                  transform=fig.transFigure, color=color, lw=1.6, solid_capstyle="butt"))
    fig.text((line_right - KM_KEY_LINE) / w, y_head / h, "Number at risk", ha="left", va="center")


def fig_scanb_km():
    """Figure 4: SCAN-B Kaplan-Meier curves by predicted drug sensitivity.

    Panel a (fig4a) splits the survival cohort at the median predicted
    AUC for two cytotoxic positive controls and two candidates, and
    panel b (fig4b) splits it into quintiles of the predicted AUC for
    paclitaxel and entinostat. Predicted sensitive (low AUC) is blue and
    predicted resistant vermillion, as in Figure 3C, and the quintiles run
    from blue (Q1) through sky blue, grey and light orange to vermillion (Q5).
    The two panels are 170 mm wide with the same column geometry and the same
    0 to 1 survival axis, so their axes and tables line up when _compose stacks
    them in Figure4.png.

    The cohort is the survival cohort of script 58 (7,397 patients with
    overall survival, including the 182 without a PAM50 call). The log-rank p
    of panel a (SCANB_km_<drug>_logrank_p) is recomputed here to make sure the
    figure shows the same split, and the trend p of panel b (km_<drug>_trend_p,
    log-rank test for trend over the quintile index) is checked against
    scanb_km_extras.csv.

    A table under each axes gives the numbers at risk (patients with os_time
    at or above each time) at the x ticks with a patient still at risk, keyed
    by a colored line in panel a and by a colored line and Q1 to Q5 in panel b.
    The tick of 12 years has no column because nobody reaches it. The numbers
    printed come from scanb_km_at_risk.csv (script 58). They are recomputed
    here over the same groups as the curves, and any difference raises, as does
    a time that is not such a tick, a sum at 0 years other than
    SCANB_km_n_patients or a total at 10 years other than km_at_risk_10y_total.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        return _skip("Figure4", "lifelines not installed")
    pred_path = RESULTS_V5 / "scanb_perdrug_predictions.csv"
    if not pred_path.exists():
        return _skip("Figure4", "scanb_perdrug_predictions.csv missing in results/v5_final/")
    drugs = list(dict.fromkeys(KM_DRUGS_SPLIT + KM_DRUGS_QUINTILE))
    header = pd.read_csv(pred_path, nrows=0).columns
    missing = [d for d in drugs if d not in header]
    if missing:
        raise KeyError(f"{missing} are not in {pred_path.name}")
    pred = pd.read_csv(pred_path, index_col=0, usecols=[header[0], *drugs])
    clin = _scanb_patient_cohort()  # survival cohort (n = 7,397), not the Cox cohort of Table 2
    n_pat, n_ev = int(_stat("SCANB_km_n_patients")), int(_stat("SCANB_km_n_events"))
    data = {}
    for d in drugs:
        m = clin.join(pred[d].rename("auc"), how="inner").dropna(subset=["os_time", "os_event", "auc"])
        if (len(m), int(m["os_event"].sum())) != (n_pat, n_ev):
            raise ValueError(f"KM cohort for {d}: {len(m)} patients and {int(m['os_event'].sum())} "
                             f"events, script 58 has {n_pat} and {n_ev}")
        data[d] = m.assign(years=m["os_time"] / 365.25)

    def km(part):
        sf = KaplanMeierFitter().fit(part["years"], part["os_event"]).survival_function_.iloc[:, 0]
        return sf.index.to_numpy(), sf.to_numpy()

    # The groups of each drug, the same ones for the curves and for the counts
    half_groups, q_groups = ["sensitive", "resistant"], [f"Q{k + 1}" for k in range(5)]
    groups = {}
    for d, m in data.items():
        sens = m["auc"] < m["auc"].median()
        q = pd.qcut(m["auc"], 5, labels=False)  # 0 = lowest predicted AUC (Q1), as in script 58
        groups[d] = {"sensitive": m[sens], "resistant": m[~sens],
                     **{g: m[q == k] for k, g in enumerate(q_groups)}}

    split = {}
    for d in KM_DRUGS_SPLIT:
        s, r = groups[d]["sensitive"], groups[d]["resistant"]
        lr = logrank_test(s["os_time"], r["os_time"], event_observed_A=s["os_event"],
                          event_observed_B=r["os_event"])
        p_key = float(_stat(f"SCANB_km_{d}_logrank_p"))
        if not np.isclose(lr.p_value, p_key, rtol=0.01, atol=0):
            raise ValueError(f"KM log-rank p for {d}: figure {lr.p_value:.3g}, script 58 {p_key}")
        split[d] = (km(s), km(r), p_key)

    extras = _csv(PAPER_RESULTS / "scanb_km_extras.csv")
    if extras is None:
        raise FileNotFoundError("scanb_km_extras.csv missing in paper/results/ (script 58)")
    extras = extras.set_index("drug")
    quint = {}
    for d in KM_DRUGS_QUINTILE:
        p_key = float(_stat(f"km_{d}_trend_p"))
        if not np.isclose(float(extras.loc[d, "trend_p"]), p_key, rtol=0.05, atol=0):
            raise ValueError(f"Trend p for {d}: key {p_key}, scanb_km_extras.csv {extras.loc[d, 'trend_p']}")
        quint[d] = ([km(groups[d][g]) for g in q_groups], p_key)

    # Shared x axis, whole years in steps of 2, the same in both PNGs
    xmax = 2 * np.ceil(max(m["years"].max() for m in data.values()) / 2)
    lw = 1.1

    # Numbers at risk: printed from the CSV of script 58, which must equal the
    # counts recomputed on the groups of the curves
    at_risk = _csv(PAPER_RESULTS / "scanb_km_at_risk.csv")
    if at_risk is None:
        raise FileNotFoundError("scanb_km_at_risk.csv missing in paper/results/ (script 58)")
    n_risk = at_risk.set_index(["drug", "group", "time_years"])["n_at_risk"]
    times = sorted(int(t) for t in at_risk["time_years"].unique())
    ticks_with_patients = [int(t) for t in np.arange(0, xmax + 0.5, 2)
                           if any((m["os_time"] >= t * 365.25).any() for m in data.values())]
    if times != ticks_with_patients:
        raise ValueError(f"scanb_km_at_risk.csv has times {times}, the x ticks with a patient at risk "
                         f"are {ticks_with_patients}")
    recount = {(d, g, t): int((part["os_time"] >= t * 365.25).sum())
               for d, parts in groups.items() for g, part in parts.items() for t in times}
    csv_counts = n_risk.to_dict()
    if len(n_risk) != len(recount) or csv_counts != recount:
        diff = [(k, csv_counts.get(k), v) for k, v in recount.items() if csv_counts.get(k) != v]
        raise ValueError(f"scanb_km_at_risk.csv differs from the counts of the curves "
                         f"(rows {len(n_risk)} and {len(recount)}, CSV and recount): {diff[:5]}")
    total_10y = int(_stat("km_at_risk_10y_total"))
    for d in data:
        n0, n10 = (sum(n_risk[d, g, t] for g in half_groups) for t in (0, 10))
        if (n0, n10) != (n_pat, total_10y):
            raise ValueError(f"Numbers at risk of {d}: {n0} at 0 years and {n10} at 10 years, the keys "
                             f"have {n_pat} and {total_10y}")

    def risk_rows(d, group_names):
        return [[n_risk[d, g, t] for t in times] for g in group_names]

    with _placed_rc():
        # Panel a: median split, 2 x 2
        fig = _placed_canvas(PLACED_W_IN, _km_canvas_h(2, len(half_groups)))
        axes, bottoms = _km_axes(fig, 2, len(half_groups))
        for i, (ax, d) in enumerate(zip(axes, KM_DRUGS_SPLIT)):
            (ts, ss), (tr, sr), p = split[d]
            ax.step(ts, ss, where="post", color=KM_SENSITIVE, lw=lw)
            ax.step(tr, sr, where="post", color=KM_RESISTANT, lw=lw)
            _km_style(ax, d, f"log-rank p = {_fmt_p_2sig(p)}", xmax, left=i % 2 == 0)
            _km_risk_table(fig, ax, bottoms[i], times, risk_rows(d, half_groups),
                           [KM_SENSITIVE, KM_RESISTANT], ["", ""])
        fig.legend(handles=[plt.Line2D([], [], color=KM_SENSITIVE, lw=lw,
                                       label="Predicted sensitive (AUC below the median)"),
                            plt.Line2D([], [], color=KM_RESISTANT, lw=lw,
                                       label="Predicted resistant (AUC at or above the median)")],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                   borderaxespad=0, borderpad=0.1, handlelength=1.8, columnspacing=2.0)
        _placed_save(fig, "fig4a", directory=None)

        # Panel b: quintiles, 1 x 2
        fig = _placed_canvas(PLACED_W_IN, _km_canvas_h(1, len(q_groups)))
        axes, bottoms = _km_axes(fig, 1, len(q_groups))
        for i, (ax, d) in enumerate(zip(axes, KM_DRUGS_QUINTILE)):
            curves, p = quint[d]
            for (t, s), col in zip(curves, KM_QUINTILE_COLORS):
                ax.step(t, s, where="post", color=col, lw=lw)
            _km_style(ax, d, f"trend p = {_fmt_p_2sig(p)}", xmax, left=i == 0)
            _km_risk_table(fig, ax, bottoms[i], times, risk_rows(d, q_groups), KM_QUINTILE_COLORS, q_groups)
        labels =["Q1 (lowest AUC, most sensitive)", "Q2", "Q3", "Q4", "Q5 (highest AUC, most resistant)"]
        fig.legend(handles=[plt.Line2D([], [], color=c, lw=lw, label=l)
                            for c, l in zip(KM_QUINTILE_COLORS, labels)],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
                   borderaxespad=0, borderpad=0.1, handlelength=1.8, columnspacing=1.6)
        _placed_save(fig, "fig4b", directory=None)


def fig7_aurora_tf():
    """Figure 7 (Figure7.png, drawn at 170 x 140 mm): activity of the ten core
    TFs in the AURORA-US patients with both a primary and a metastatic sample
    (a), against the same test over every TF of the activity matrix (b).

    The pairs are built as in scripts/36_aurora_ig_validation.py: the patient
    is the second field of the sample identifier, TTP marks primary and TTM
    metastatic samples, and each patient contributes the mean of their primary
    samples and the mean of their metastatic samples. In panel a, one thin
    grey line per patient joins the two values, over light boxes of the
    per-patient values (median line, whiskers to 1.5 times the interquartile
    range, no outlier markers because every patient is drawn). The paired
    Wilcoxon p and the mean paired difference (metastasis minus primary) in
    the titles are recomputed from these pairs and checked against
    table4_aurora_tfs.csv, the source of Table 4, and the p-values are
    formatted as in that table. All ten panels have the same neutral style.

    Panel b plots -log10 of the paired p of all the TFs against their rank by
    p (method min, as in aurora_tf_background.csv of script 60, so tied TFs
    share a point). The p-values and differences are recomputed from the pairs
    of panel a and checked against that CSV. The core TFs are circled and
    labeled, in the style of Supplementary Figure 10, each label on the side
    of the threshold where its TF lies. The only threshold is the Bonferroni
    of Table 4, 0.05 over the ten core TFs. Its value and every count printed
    come from paper_statistics.json and are checked against the CSV.
    """
    from scipy.stats import hypergeom, wilcoxon
    from matplotlib.colors import to_rgba
    import matplotlib.patheffects as pe
    stats_path = PAPER_RESULTS / "table4_aurora_tfs.csv"
    tf_act_path = PAPER_RESULTS / "aurora_tf_activities.csv"
    bg_path = PAPER_RESULTS / "aurora_tf_background.csv"
    if not all(p.exists() for p in (stats_path, tf_act_path, bg_path)):
        return _skip("Figure7", "table4_aurora_tfs.csv, aurora_tf_activities.csv or "
                                "aurora_tf_background.csv missing in paper/results/")

    t4 = pd.read_csv(stats_path).sort_values("pvalue", kind="mergesort").reset_index(drop=True)
    act = pd.read_csv(tf_act_path, index_col=0)
    bg = pd.read_csv(bg_path)
    fields = act.index.to_series().str.split("-")
    patient, code = fields.str[1], fields.str[2]
    is_prim, is_meta = code.str.startswith("TTP"), code.str.startswith("TTM")
    prim = act[is_prim.to_numpy()].groupby(patient[is_prim].to_numpy()).mean()
    meta = act[is_meta.to_numpy()].groupby(patient[is_meta].to_numpy()).mean()
    both = prim.index.intersection(meta.index).sort_values()
    if len(both) != int(_stat("aurora_cohort_n_paired_patients")):
        raise ValueError(f"{len(both)} paired patients, paper_statistics.json has "
                         f"{_stat('aurora_cohort_n_paired_patients')}")
    prim, meta = prim.loc[both], meta.loc[both]

    n_tests = int(_stat("aurora_total_tfs_tested"))
    if len(t4) != n_tests or set(t4.loc[t4["Bonferroni_sig"], "TF"]) != set(_stat("aurora_bonferroni_tfs")):
        raise ValueError("table4_aurora_tfs.csv does not match the aurora_* keys of paper_statistics.json")
    bonf_alpha = 0.05 / n_tests
    rows = []
    for _, r in t4.iterrows():
        tf = r["TF"]
        diff = meta[tf] - prim[tf]
        p = float(wilcoxon(prim[tf], meta[tf], alternative="two-sided").pvalue)
        delta = float(diff.mean())
        if not np.isclose(p, r["pvalue"], rtol=1e-4, atol=0) or abs(delta - r["Delta"]) > 0.005 + 1e-9 \
                or abs(prim[tf].mean() - r["Primary_mean"]) > 0.005 + 1e-9 \
                or abs(meta[tf].mean() - r["Metastasis_mean"]) > 0.005 + 1e-9 \
                or bool(r["Bonferroni_sig"]) != (p < bonf_alpha):
            raise ValueError(f"{tf}: the recomputed pairs do not reproduce Table 4")
        rows.append((tf, p, delta))

    # Panel b: the paired test over every TF, from the pairs of panel a
    n_tf = int(_stat("aurora_bg_tf_n_tested"))
    if len(bg) != n_tf or act.shape[1] != n_tf or set(bg["tf"]) != set(act.columns):
        raise ValueError("Figure 7: aurora_tf_background.csv and the activity matrix have different TFs")
    if int(_stat("aurora_bg_n_paired_patients")) != len(both) or not (bg["n_paired"] == len(both)).all():
        raise ValueError(f"Figure 7: aurora_tf_background.csv is not on the {len(both)} paired patients of panel a")
    is_core = bg["is_core"].to_numpy(dtype=bool)
    core_bg = bg[is_core].sort_values("rank", kind="mergesort")
    if not set(core_bg["tf"]) == set(t4["TF"]) == set(_stat("ig_multi_seed_core10_names")):
        raise ValueError("Figure 7: the core TFs of aurora_tf_background.csv are not those of Table 4")
    p_all = np.array([wilcoxon(prim[tf], meta[tf], alternative="two-sided").pvalue for tf in bg["tf"]])
    delta_all = (meta[bg["tf"]] - prim[bg["tf"]]).mean().to_numpy()
    if not np.allclose(p_all, bg["pvalue"], rtol=1e-9, atol=0) or (np.abs(delta_all - bg["delta"]) > 1e-12).any():
        raise ValueError("Figure 7: the recomputed pairs do not reproduce aurora_tf_background.csv")
    if not (bg["pvalue"].rank(method="min") == bg["rank"]).all():
        raise ValueError("Figure 7: the rank column of aurora_tf_background.csv is not the rank of its p-values")
    n_below = int((bg["pvalue"] < bonf_alpha).sum())
    core_below = set(core_bg.loc[core_bg["pvalue"] < bonf_alpha, "tf"])
    _check_stat("aurora_bg_tf_n_p_lt_0005", n_below)
    _check_stat("aurora_bg_tf_pct_p_lt_0005", round(100 * n_below / n_tf, 1))
    _check_stat("aurora_bg_core_n_p_lt_0005", len(core_below))
    _check_stat("aurora_bonferroni_sig", len(core_below))
    if core_below != set(_stat("aurora_bonferroni_tfs")):
        raise ValueError("Figure 7: the core TFs below the threshold are not aurora_bonferroni_tfs")
    _check_stat("aurora_bg_core_expected_p_lt_0005", round(n_tests * n_below / n_tf, 1))
    _check_stat("aurora_bg_core_hypergeom_p",
                float(f"{hypergeom.sf(len(core_below) - 1, n_tf, n_below, n_tests):.1e}"))
    t4_p = t4.set_index("TF")["pvalue"]
    for _, r in core_bg.iterrows():
        _check_stat(f"aurora_bg_core_rank_{r['tf']}", int(r["rank"]))
        if not np.isclose(r["pvalue"], t4_p[r["tf"]], rtol=1e-9, atol=0):
            raise ValueError(f"{r['tf']}: its p in aurora_tf_background.csv is not the one of Table 4")

    ncol, nrow = 5, 2
    left, right, wspace = 0.40, 0.04, 0.33
    band = LETTER_BAND_PT / 72   # letter and legend at the top, letter b over its panel
    top_a, hspace = band + 0.43, 0.58   # three-line titles above the first row
    b_bottom, b_h = 0.36, 1.42
    b_gap = 0.20 + band   # x tick labels of the second row and the band of letter b
    with _placed_rc():
        fig = _placed_canvas(PLACED_W_IN, 140 / 25.4)
        w, h = fig.get_size_inches()
        ax_w = (w - left - right - (ncol - 1) * wspace) / ncol
        a_bottom = b_bottom + b_h + b_gap
        ax_h = (h - top_a - a_bottom - (nrow - 1) * hspace) / nrow
        names = []
        for i, (tf, p, delta) in enumerate(rows):
            r, c = divmod(i, ncol)
            ax = fig.add_axes(_inch_rect(fig, left + c * (ax_w + wspace),
                                         a_bottom + (nrow - 1 - r) * (ax_h + hspace), ax_w, ax_h))
            bp = ax.boxplot([prim[tf], meta[tf]], positions=[0, 1], widths=0.5, patch_artist=True,
                            showfliers=False, medianprops={"color": FIG1_TEXT, "lw": 0.9},
                            boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5, "color": C_NEUTRAL},
                            capprops={"lw": 0.5, "color": C_NEUTRAL}, zorder=1)
            for box, col in zip(bp["boxes"], (C_PRIMARY, C_ACCENT)):
                box.set_facecolor(to_rgba(col, 0.22))
                box.set_edgecolor(col)
            ax.plot(np.vstack([np.zeros(len(both)), np.ones(len(both))]),
                    np.vstack([prim[tf].to_numpy(), meta[tf].to_numpy()]),
                    color="#9A9A9A", lw=0.4, alpha=0.8, zorder=2)
            ax.scatter(np.zeros(len(both)), prim[tf], s=3, color=C_PRIMARY, linewidths=0, zorder=3)
            ax.scatter(np.ones(len(both)), meta[tf], s=3, color=C_ACCENT, linewidths=0, zorder=3)
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Primary", "Metastasis"])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.grid(True, axis="y", color="#E8E9EB")
            ax.grid(False, axis="x")
            ax.set_title(f"p = {_fmt_p_2sig(p)}\n\u0394 = {_fmt_signed(delta)}", linespacing=1.25)
            names.append((ax, tf))
            if c == 0:
                ax.set_ylabel("TF activity (ULM score)")
        fig.legend(handles=[
            mpatches.Patch(facecolor=to_rgba(C_PRIMARY, 0.22), edgecolor=C_PRIMARY,
                           label="Primary tumor"),
            mpatches.Patch(facecolor=to_rgba(C_ACCENT, 0.22), edgecolor=C_ACCENT,
                           label="Metastasis (mean per patient)"),
        ], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
            borderaxespad=0, borderpad=0.1, handlelength=1.4, columnspacing=2.0)

        ax_b = fig.add_axes(_inch_rect(fig, left, b_bottom, w - left - right, b_h))
        x = bg["rank"].to_numpy(dtype=float)
        y = -np.log10(bg["pvalue"].to_numpy())
        y_thr = -np.log10(bonf_alpha)
        ax_b.scatter(x[~is_core], y[~is_core], s=4, color=C_LIGHT, lw=0, zorder=2)
        ax_b.scatter(x[is_core], y[is_core], s=14, facecolors="none", edgecolors="black",
                     linewidths=0.7, zorder=5)
        ax_b.axhline(y_thr, color=FIG1_TEXT, ls="--", lw=0.6, zorder=1)
        ax_b.set_xlim(-8, n_tf + 8)
        ax_b.set_xticks([1, *range(100, n_tf, 100), n_tf])
        ax_b.set_ylim(-0.02 * y.max(), 1.05 * y.max())
        ax_b.grid(True, color="#E8E9EB")
        ax_b.set_xlabel(f"Rank of the paired Wilcoxon p-value among the {n_tf} TFs")
        ax_b.set_ylabel("\u2212log10(paired p)")
        ax_b.text(0.995, 0.97,
                  f"{_stat('aurora_bg_tf_n_p_lt_0005')} of {n_tf} TFs "
                  f"({_stat('aurora_bg_tf_pct_p_lt_0005'):g}%) with p < {bonf_alpha:g}\n"
                  f"Core TFs: {_stat('aurora_bg_core_n_p_lt_0005')} of {n_tests}, "
                  f"{_stat('aurora_bg_core_expected_p_lt_0005')} expected for {n_tests} random TFs, "
                  f"hypergeometric p = {_stat('aurora_bg_core_hypergeom_p'):g}",
                  transform=ax_b.transAxes, ha="right", va="top", linespacing=1.35, zorder=6,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        ax_b.text(0.995, y_thr + 0.12, f"Bonferroni threshold over the {n_tests} core TFs, p = {bonf_alpha:g}",
                  transform=mtransforms.blended_transform_factory(ax_b.transAxes, ax_b.transData),
                  ha="right", va="bottom", zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

        # TF name in bold above the two-line title, 1 pt above its rendered top
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, tf in names:
            gap_pt = (ax.title.get_window_extent(renderer).y1 - ax.bbox.y1) * 72 / fig.dpi
            ax.annotate(tf, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, gap_pt + 1),
                        textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        # One column for the y labels of both panels, so that letters a and b,
        # each at the left edge of its panel, share it too
        label_axes = [names[0][0], names[ncol][0], ax_b]
        label_x = min(ax.yaxis.label.get_window_extent(renderer).x1 for ax in label_axes)
        for ax in label_axes:
            ax.yaxis.set_label_coords(fig.transFigure.inverted().transform((label_x, 0))[0], 0.5,
                                      mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes))

        # In rank order, except SETBP1 before SPDEF: their points almost coincide,
        # and SETBP1, the one on the right, takes the free ground on that side
        label_order = ["KLF8", "STAT2", "AEBP1", "SETBP1", "SPDEF", "ZEB1", "MZF1", "RELA", "IRX1", "BARX1"]
        core_lab = core_bg.set_index("tf").loc[label_order]
        stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
        labels = _place_leader_labels(
            fig, ax_b, np.column_stack([core_lab["rank"], -np.log10(core_lab["pvalue"])]), label_order,
            np.column_stack([x, y]), point_r_pt=np.sqrt(14) / 2,
            lines=(("h", y_thr),), keep_side=True, color="black", path_effects=stroke, zorder=7,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "shrinkA": 1, "shrinkB": 2.5})
        # The rendered labels against the threshold: a TF with p below it lies
        # above the line, and so must its label, and the reverse
        fig.canvas.draw()
        thr_px = ax_b.transData.transform((0, y_thr))[1]
        for t, p in zip(labels, core_lab["pvalue"]):
            box = matplotlib.text.Text.get_window_extent(t, renderer)
            if box.y0 <= thr_px if p < bonf_alpha else box.y1 >= thr_px:
                raise ValueError(f"Figure 7: the label of {t.get_text()} is on the wrong side of the threshold")
        _panel_letters(fig, (("a", names[0][0]),))
        _panel_letters(fig, (("b", ax_b),), top=(b_bottom + b_h + band) / h)
        _placed_save(fig, "Figure7")


C_TNBC_CONTROL = "#6E6E6E"   # positive controls of Figure 8
C_TNBC_MDE = "#E3E3E3"       # bars behind the rows of Figure 8, effects below the MDE
C_TNBC_NULL = "#BDBDBD"      # permutation nulls of Figure 8
C_TNBC_TAIL = "#7A7A7A"      # upper 5% of the null of panel d of Figure 8


def _mde_delta_80(n1, n2):
    """Minimum detectable Cliff's delta at 80% power and two-sided 5% for
    groups of n1 and n2, from the null standard error (script 67, mde_delta)."""
    return (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))


def _forest(ax, items):
    """Rows of a forest panel of Figure 8, one per item from y = 0 down. An
    item is ("header", text) or ("row", label, p), and p holds est, lo and hi
    (estimate and interval), color, marker, ms and, when they apply, mec
    (marker edge), mde (span of the grey bar behind the row), range (span of
    the dashed box) and note (text at the right edge of the axes). Headers go
    over the reference line, drawn at zorder 1.5, and under the spines."""
    ticks, labels = [], []
    for y, item in enumerate(items):
        if item[0] == "header":
            ax.text(0.004, y, item[1], transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontstyle="italic", zorder=2.2,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
            continue
        _, label, p = item
        ticks.append(y)
        labels.append(label)
        if "mde" in p:
            lo, hi = p["mde"]
            ax.add_patch(mpatches.Rectangle((lo, y - 0.36), hi - lo, 0.72, facecolor=C_TNBC_MDE,
                                            edgecolor="none", zorder=1))
        if "range" in p:
            lo, hi = p["range"]
            ax.add_patch(mpatches.Rectangle((lo, y - 0.36), hi - lo, 0.72, facecolor="none",
                                            edgecolor=C_FRAME, lw=0.5, ls=(0, (2, 1.5)), zorder=1))
        ax.plot([p["lo"], p["hi"]], [y, y], color=p["color"], lw=0.9, solid_capstyle="butt", zorder=3)
        ax.plot(p["est"], y, marker=p["marker"], color=p["color"], markersize=p["ms"],
                markeredgecolor=p.get("mec", p["color"]), markeredgewidth=0.6, lw=0, zorder=4)
        if "note" in p:
            ax.annotate(p["note"], xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                        textcoords="offset points", ha="right", va="center")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(items) - 0.5, -0.8)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color="#E8E9EB")
    ax.tick_params(axis="y", length=0)


def fig_tnbc_direct():
    """Figure 8 (Figure8.png, drawn at 170 x 147 mm): the candidates in the
    TNBC cell lines of PRISM (script 67, top row) and in the FUSCC-TNBC cohort
    (script 69, bottom row). The left column shows the estimates over the
    effect each comparison can detect, the right column the primary statistic
    of the analysis against its permutation null.

    a: Cliff's delta of the TNBC lines against the other lines for the
    composite of the candidates, each candidate (Table 5 order) and each
    positive control (Table 3 order), with intervals from the null standard
    error. The assay checks compare paclitaxel within breast in PRISM and GDSC,
    and the calibration rows draw skin lines against the non-skin lines, with
    the median and the central 95% of the draws instead of an interval. Behind
    each row, a grey bar from -MDE to +MDE of that comparison, recomputed from
    its group sizes and checked against the CSV.

    b: the composite against its 10,000 label permutations, with the MDE.

    c: hazard ratio per SD of the predicted AUC in FUSCC-TNBC for the same
    rows, the MDE from the events, and the cohort check (nodes, a binary term
    and not per SD, so with no bar) with its declared range as a dashed box.

    d: the correlation over all the drugs between the Cox z of FUSCC-TNBC and
    of the SCAN-B TNBC subgroup, against its 10,000 outcome permutations. The
    bin grid is anchored at the 95th percentile of the null, so that the upper
    5% takes whole bars and no bar is drawn twice.

    Every number printed comes from paper_statistics.json, and each drawn value
    is checked against its key. A missing CSV raises FileNotFoundError."""
    tol = 0.005 + 1e-9   # keys stored with two decimals

    def read(name):
        return pd.read_csv(PAPER_RESULTS / name)

    # The rows of the two forests, from Tables 5 and 3
    candidates = read("table5_candidates.csv")["Drug"].tolist()
    controls = read("table3_positive_controls.csv")["Drug"].tolist()
    # The printed Table 3 has talazoparib before olaparib, its CSV the other way round
    table3_order = ["paclitaxel", "docetaxel", "epirubicin", "talazoparib", "olaparib"]
    if set(controls) != set(table3_order):
        raise ValueError("Figure 8: table3_positive_controls.csv is not the five drugs of the printed Table 3")
    controls = table3_order
    if len(candidates) != int(_stat("candidates_n_selected")) or set(candidates) != set(_stat("candidates_selected")):
        raise ValueError("Figure 8: table5_candidates.csv does not match candidates_selected")
    if int(_stat("candidates_n_selected")) != 7:
        raise ValueError("Figure 8: the legend says seven candidates")
    focal = [(d, "candidate") for d in candidates] + [(d, "control") for d in controls]

    # Top row: the TNBC cell lines of PRISM
    prim = read("tnbc_selectivity_primary.csv").set_index("drug")
    background = read("tnbc_selectivity_background.csv")
    set_test = read("tnbc_set_test.csv").set_index("set_name").loc["candidates7"]
    power = read("tnbc_power_mde.csv").set_index("drug")
    gate = read("assay_sensitivity_gate.csv")
    summary = read("tnbc_cell_lines_summary.csv").set_index("metric")["value"]
    null_delta = read("tnbc_permutation_null.csv")["delta_candidates7"].to_numpy()
    if set(prim.index[prim["family"] == "candidate"]) != set(candidates) \
            or set(prim.index[prim["family"] == "positive_control"]) != set(controls) or len(prim) != len(focal):
        raise ValueError("Figure 8: tnbc_selectivity_primary.csv is not the drugs of Tables 5 and 3")
    if set(power.index) != set(prim.index):
        raise ValueError("Figure 8: tnbc_power_mde.csv is not the drugs of Tables 5 and 3")
    n_tnbc, n_other = int(_stat("cltnbc_n_lines")), int(_stat("cltnbc_n_comparison_lines"))
    if int(set_test["n_tnbc"]) != n_tnbc or int(set_test["n_lines"]) != n_tnbc + n_other:
        raise ValueError("Figure 8: the composite of tnbc_set_test.csv is not on the lines of the keys")
    _check_stat("cltnbc_set_delta", set_test["cliffs_delta"], 5e-4 + 1e-9)
    _check_stat("cltnbc_set_ci_low", set_test["ci_low"], 5e-4 + 1e-9)
    _check_stat("cltnbc_set_ci_high", set_test["ci_high"], 5e-4 + 1e-9)
    _check_stat("cltnbc_set_n_perm", len(null_delta))
    p_null = (1 + int((np.abs(null_delta) >= abs(set_test["cliffs_delta"]) - 1e-12).sum())) / (1 + len(null_delta))
    if not np.isclose(p_null, set_test["p_perm"], rtol=0, atol=1e-12):
        raise ValueError("Figure 8: the saved null does not give the p of tnbc_set_test.csv")
    _check_stat("cltnbc_set_p", round(p_null, 1))
    mde_set = float(_mde_delta_80(n_tnbc, n_other))
    _check_stat("cltnbc_mde_delta_80", mde_set, tol)
    if abs(mde_set - float(summary["mde_delta_80"])) > 5e-5 + 1e-9:
        raise ValueError("Figure 8: the MDE of the composite is not the one of tnbc_cell_lines_summary.csv")
    if not np.allclose(_mde_delta_80(power["n_tnbc"], power["n_rest"]), power["mde_delta_80"], rtol=0, atol=1e-9):
        raise ValueError("Figure 8: an MDE of tnbc_power_mde.csv is not the one of its group sizes")
    for d, role in focal:
        for key, col in (("delta", "cliffs_delta"), ("ci_low", "delta_ci_low"), ("ci_high", "delta_ci_high")):
            _check_stat(f"{role}_{d}_cltnbc_{key}", prim.at[d, col], tol)
    _check_stat("cltnbc_n_evaluable", len(background))
    _check_stat("cltnbc_n_panel_above_mde",
                int((background["cliffs_delta"].abs() >= float(summary["mde_delta_80"])).sum()))
    _check_stat("cltnbc_background_n_padj05", int((background["padj_bh_background_descriptive"] < 0.05).sum()))
    # Checks of the assay: the two rows of paclitaxel within breast and the calibration draws
    by_contrast = gate.set_index(["control", "screen", "contrast"])
    pac_prism = by_contrast.loc[("paclitaxel", "PRISM", "tnbc_vs_other_breast")]
    pac_gdsc = by_contrast.loc[("paclitaxel", "GDSC", "tnbc_vs_other_breast")]
    _check_stat("cltnbc_gate_paclitaxel_prism_within_breast", pac_prism["cliffs_delta"], tol)
    _check_stat("cltnbc_gate_paclitaxel_gdsc_within_breast", pac_gdsc["cliffs_delta"], tol)
    passed = _stat("cltnbc_gate_directional_passed")
    if "paclitaxel_prism_tnbc_vs_other_breast" not in passed or "paclitaxel_gdsc_tnbc_vs_other_breast" in passed:
        raise ValueError("Figure 8: of the two rows of paclitaxel only the one of PRISM is in "
                         "cltnbc_gate_directional_passed, as the caption says")
    calibration = gate[gate["contrast"] == "skin12_vs_rest_draws"].sort_values("cliffs_delta")
    if len(calibration) != 3 or (calibration["n_group"] != n_tnbc).any() or (calibration["screen"] != "PRISM").any():
        raise ValueError(f"Figure 8: expected three calibration rows of {n_tnbc} PRISM lines")
    _check_stat("cltnbc_calibration_detect_rate", calibration["detection_rate"].min(), 1e-9)
    _check_stat("cltnbc_calibration_delta_dabrafenib",
                calibration.set_index("control").at["dabrafenib", "cliffs_delta"], tol)
    assay_rows = pd.DataFrame([pac_prism, pac_gdsc])
    for rows in (assay_rows, calibration):
        if not np.allclose(_mde_delta_80(rows["n_group"].astype(float), rows["n_rest"].astype(float)),
                           rows["mde_delta_80"].astype(float), rtol=0, atol=1e-9):
            raise ValueError("Figure 8: an MDE of assay_sensitivity_gate.csv is not the one of its group sizes")

    # Bottom row: FUSCC-TNBC
    perdrug = read("fuscc_cox_perdrug.csv").set_index("drug")
    comp = read("fuscc_composite.csv").set_index("set").loc["candidates7"]
    nodes = read("fuscc_cohort_gate.csv").set_index("term").loc["positive lymph nodes"]
    corr = read("fuscc_primary_correlation.csv")
    corr = corr[corr["is_primary"]]
    if len(corr) != 1:
        raise ValueError("Figure 8: fuscc_primary_correlation.csv does not have one primary row")
    corr = corr.iloc[0]
    null_r = read("fuscc_permutation_null.csv")["r_all1447"].to_numpy()
    _check_stat("fuscc_n_drugs_tested", len(perdrug))
    r_obs = float(np.corrcoef(perdrug["score_z"], perdrug["scanb_tnbc_score_z_os"])[0, 1])
    if not np.isclose(r_obs, corr["r_pearson"], rtol=0, atol=1e-10):
        raise ValueError("Figure 8: r recomputed from fuscc_cox_perdrug.csv differs from the primary row")
    _check_stat("fuscc_r_primary", r_obs, tol)
    _check_stat("fuscc_n_perm", len(null_r))
    p_null_r = (1 + int((null_r >= r_obs).sum())) / (1 + len(null_r))
    if not np.isclose(p_null_r, corr["p_perm"], rtol=0, atol=1e-12):
        raise ValueError("Figure 8: the saved null does not give the p of fuscc_primary_correlation.csv")
    _check_stat("fuscc_r_primary_p_perm_3dp", round(p_null_r, 3))
    p95_r = float(np.percentile(null_r, 95))
    if not np.isclose(p95_r, corr["null_p95"], rtol=0, atol=1e-12):
        raise ValueError("Figure 8: the 95th percentile of the saved null differs from the primary row")
    _check_stat("fuscc_r_primary_null_p95", p95_r, tol)
    for d, role in focal:
        for key, col in (("hr_per_sd", "hr_per_sd_fuscc"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
            _check_stat(f"{role}_{d}_fuscc_{key}", perdrug.at[d, col], tol)
    _check_stat("fuscc_composite_hr_per_sd", comp["hr_per_sd"], tol)
    _check_stat("fuscc_composite_p_perm", comp["p_perm"], tol)
    _check_stat("fuscc_n_patients", int(comp["n"]))
    _check_stat("fuscc_n_events", int(comp["events"]))
    mde_hr = float(np.exp((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(comp["events"])))
    _check_stat("fuscc_mde_hr_per_sd", mde_hr, tol)
    if not np.isclose(mde_hr, comp["mde_hr_per_sd_80"], rtol=0, atol=1e-9):
        raise ValueError("Figure 8: the MDE of fuscc_composite.csv is not the one of its events")
    _check_stat("fuscc_gate_nodes_hr", nodes["hr"], tol)
    range_low, range_high = float(_stat("fuscc_gate_nodes_range_low")), float(_stat("fuscc_gate_nodes_range_high"))
    if not range_low <= nodes["hr"] <= range_high or bool(nodes["verdict"]) != bool(_stat("fuscc_gate_nodes_passed")):
        raise ValueError("Figure 8: the hazard ratio of the nodes against its declared range or its verdict")
    _check_stat("fuscc_n_drugs_fdr05", int((perdrug["padj_bh1447_descriptive"] < 0.05).sum()))

    def row(label, est, lo, hi, color, marker="o", ms=3.6, **extra):
        return ("row", label, {"est": float(est), "lo": float(lo), "hi": float(hi), "color": color,
                               "marker": marker, "ms": ms, **extra})

    def delta_row(d, color):
        return row(_drug_display(d), prim.at[d, "cliffs_delta"], prim.at[d, "delta_ci_low"],
                   prim.at[d, "delta_ci_high"], color, mde=symmetric(power.at[d, "mde_delta_80"]))

    def hr_row(d, color):
        return row(_drug_display(d), perdrug.at[d, "hr_per_sd_fuscc"], perdrug.at[d, "ci_low"],
                   perdrug.at[d, "ci_high"], color, mde=(1 / mde_hr, mde_hr))

    def symmetric(value):
        return (-float(value), float(value))

    composite_style = {"marker": "D", "ms": 4.2, "mec": FIG1_TEXT}
    check_style = {"color": FIG1_TEXT, "marker": "s", "ms": 3.2}
    a_items = [("header", "Candidates"),
               row("Composite", set_test["cliffs_delta"], set_test["ci_low"], set_test["ci_high"], C_HIGHLIGHT,
                   mde=symmetric(mde_set), **composite_style),
               *[delta_row(d, C_HIGHLIGHT) for d in candidates],
               ("header", "Positive controls"),
               *[delta_row(d, C_TNBC_CONTROL) for d in controls],
               ("header", "Assay check: paclitaxel, TNBC vs the other breast lines"),
               *[row(screen, r["cliffs_delta"], r["ci_low"], r["ci_high"], mde=symmetric(r["mde_delta_80"]),
                     **check_style) for screen, r in (("PRISM", pac_prism), ("GDSC", pac_gdsc))],
               ("header", f"Calibration: {n_tnbc} skin lines vs the non-skin lines"),
               *[row(_drug_display(r.control), r.cliffs_delta, r.ci_low, r.ci_high, mde=symmetric(r.mde_delta_80),
                     **check_style) for r in calibration.itertuples()]]
    c_items = [("header", "Candidates, per SD of predicted AUC"),
               row("Composite", comp["hr_per_sd"], comp["ci_low"], comp["ci_high"], C_HIGHLIGHT,
                   mde=(1 / mde_hr, mde_hr), note=f"p = {_stat('fuscc_composite_p_perm'):g}", **composite_style),
               *[hr_row(d, C_HIGHLIGHT) for d in candidates],
               ("header", "Positive controls, per SD of predicted AUC"),
               *[hr_row(d, C_TNBC_CONTROL) for d in controls],
               ("header", "Cohort check: node-positive vs node-negative"),
               row("Nodes", nodes["hr"], nodes["ci_low"], nodes["ci_high"], range=(range_low, range_high),
                   **check_style)]

    a_xlim, c_xlim, b_xlim, d_xlim = (-1, 1), (0.45, 7.5), (-1, 1), (-0.8, 0.8)
    for items, (x0, x1) in ((a_items, a_xlim), (c_items, c_xlim)):
        for item in items:
            if item[0] == "row" and not x0 <= item[2]["lo"] <= item[2]["est"] <= item[2]["hi"] <= x1:
                raise ValueError(f"Figure 8: the row {item[1]} does not fit in the axis {x0} to {x1}")
    if not (b_xlim[0] < set_test["cliffs_delta"] < b_xlim[1] and d_xlim[0] < r_obs < d_xlim[1]):
        raise ValueError("Figure 8: an observed statistic does not fit in its axis")

    fmt_minus = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
    fmt_count = matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    row_pitch = 0.098   # inches per forest row (2.5 mm)
    band = LETTER_BAND_PT / 72
    legend_h, xlab, title_h, row_gap = 0.34, 0.30, 0.16, 0.30
    h_a, h_c = (len(a_items) + 0.3) * row_pitch, (len(c_items) + 0.3) * row_pitch
    y_c = legend_h + xlab
    top_c = y_c + h_c + title_h + band          # top of the letters of the bottom row
    y_a = top_c - band + row_gap + xlab         # the gap between the rows holds the band
    canvas_h = y_a + h_a + title_h + band
    pad = plt.rcParams["savefig.pad_inches"]
    with _placed_rc():
        fig = _placed_canvas(PLACED_W_IN, canvas_h + 2 * pad)
        w, h = fig.get_size_inches()
        left, wa, gap, right = 0.66, 3.10, 0.52, 0.10
        wb = w - left - wa - gap - right
        ax_a = fig.add_axes(_inch_rect(fig, left, y_a, wa, h_a))
        ax_b = fig.add_axes(_inch_rect(fig, left + wa + gap, y_a, wb, h_a))
        ax_c = fig.add_axes(_inch_rect(fig, left, y_c, wa, h_c))
        ax_d = fig.add_axes(_inch_rect(fig, left + wa + gap, y_c, wb, h_c))

        _forest(ax_a, a_items)
        ax_a.axvline(0, color=FIG1_TEXT, lw=0.6, ls="--", zorder=1.5)
        ax_a.set_xlim(*a_xlim)
        ax_a.set_xticks(np.arange(-1, 1.01, 0.5))
        ax_a.xaxis.set_major_formatter(fmt_minus)
        ax_a.set_xlabel("Cliff’s δ (negative: more active in the first group)")
        ax_a.set_title("PRISM: TNBC lines vs the other lines", loc="left")

        counts_b, _, _ = ax_b.hist(null_delta, bins=np.linspace(*b_xlim, 81), color=C_TNBC_NULL,
                                   edgecolor="none", zorder=2)
        if counts_b.sum() != len(null_delta):
            raise ValueError("Figure 8: the bins of panel b do not hold the whole null")
        ymax_b, top_mde = counts_b.max() * 1.85, counts_b.max() * 1.12
        for sign in (-1, 1):
            ax_b.plot([sign * mde_set] * 2, [0, top_mde], color=FIG1_TEXT, lw=0.6, ls=(0, (2, 1.5)), zorder=4)
            ax_b.text(sign * mde_set, top_mde * 1.02, "MDE", ha="center", va="bottom")
        ax_b.axvline(set_test["cliffs_delta"], color=C_HIGHLIGHT, lw=1.4, zorder=5)
        ax_b.text(set_test["cliffs_delta"] + 0.06, ymax_b * 0.97,
                  f"composite\nδ = {_stat('cltnbc_set_delta'):.3f}\np = {_stat('cltnbc_set_p'):g}",
                  ha="left", va="top", linespacing=1.15)
        ax_b.set_xlim(*b_xlim)
        ax_b.set_ylim(0, ymax_b)
        ax_b.set_xticks(np.arange(-1, 1.01, 0.5))
        ax_b.xaxis.set_major_formatter(fmt_minus)
        ax_b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax_b.yaxis.set_major_formatter(fmt_count)
        ax_b.set_xlabel("Cliff’s δ of the composite")
        ax_b.set_ylabel("Label permutations")
        ax_b.grid(True, axis="y", color="#E8E9EB")
        ax_b.grid(False, axis="x")
        ax_b.set_title("Null of the composite", loc="left")

        _forest(ax_c, c_items)
        ax_c.set_xscale("log")
        ax_c.axvline(1, color=FIG1_TEXT, lw=0.6, ls="--", zorder=1.5)
        ax_c.set_xlim(*c_xlim)
        _plain_log_ticks(ax_c.xaxis, [0.5, 1, 1.5, 2, 3, 5])
        ax_c.set_xlabel("Hazard ratio for relapse-free survival (log scale)")
        ax_c.set_title(f"FUSCC-TNBC: {_fmt_n('fuscc_n_patients')} patients, "
                       f"{_fmt_n('fuscc_n_events')} relapses", loc="left")

        edges_d = p95_r + np.arange(-45, 20) * 0.025   # 64 bins of 0.025 over the axis, one edge at p95
        counts_d, _, bars_d = ax_d.hist(null_r, bins=edges_d, color=C_TNBC_NULL, edgecolor="none", zorder=2)
        in_tail = edges_d[:-1] >= p95_r
        for bar, tail in zip(bars_d, in_tail):
            if tail:
                bar.set_facecolor(C_TNBC_TAIL)
        if counts_d.sum() != len(null_r) or counts_d[in_tail].sum() != (null_r >= p95_r).sum():
            raise ValueError("Figure 8: the bins of panel d do not hold the whole null or its upper 5%")
        ymax_d = counts_d.max() * 1.6
        ax_d.axvline(r_obs, color=C_HIGHLIGHT, lw=1.4, zorder=5)
        ax_d.text(r_obs + 0.05, ymax_d * 0.97,
                  f"r = {_stat('fuscc_r_primary'):g}\np = {_stat('fuscc_r_primary_p_perm_3dp'):g}",
                  ha="left", va="top", linespacing=1.15)
        tail_x = 0.45
        ax_d.annotate("upper 5%\nof the null", xy=(tail_x, counts_d[np.searchsorted(edges_d, tail_x) - 1]),
                      xytext=(0.47, ymax_d * 0.52), ha="left", va="center", linespacing=1.15,
                      arrowprops={"arrowstyle": "-", "lw": 0.5, "color": FIG1_TEXT, "shrinkA": 1, "shrinkB": 0})
        ax_d.set_xlim(*d_xlim)
        ax_d.set_ylim(0, ymax_d)
        ax_d.set_xticks(np.arange(-0.8, 0.81, 0.4))
        ax_d.xaxis.set_major_formatter(fmt_minus)
        ax_d.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax_d.yaxis.set_major_formatter(fmt_count)
        ax_d.set_xlabel("Pearson r, FUSCC-TNBC vs SCAN-B TNBC")
        ax_d.set_ylabel("Outcome permutations")
        ax_d.grid(True, axis="y", color="#E8E9EB")
        ax_d.grid(False, axis="x")
        ax_d.set_title(f"Per-drug Cox z over {_fmt_n('SCANB_n_drugs_tested')} drugs", loc="left")

        fig.legend(handles=[
            plt.Line2D([], [], color=C_HIGHLIGHT, marker="D", markeredgecolor=FIG1_TEXT, markersize=4.2,
                       markeredgewidth=0.6, lw=0.9, label="Composite of the seven candidates"),
            plt.Line2D([], [], color=C_HIGHLIGHT, marker="o", markersize=3.6, lw=0.9, label="Candidate"),
            plt.Line2D([], [], color=C_TNBC_CONTROL, marker="o", markersize=3.6, lw=0.9, label="Positive control"),
            plt.Line2D([], [], color=FIG1_TEXT, marker="s", markersize=3.2, lw=0.9,
                       label="Assay, calibration or cohort check"),
            mpatches.Patch(facecolor=C_TNBC_MDE, edgecolor="none",
                           label="Smaller than the minimum detectable effect (80% power)"),
            mpatches.Patch(facecolor="none", edgecolor=C_FRAME, lw=0.5, ls=(0, (2, 1.5)),
                           label="Declared range of the cohort check"),
        ], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, borderaxespad=0, borderpad=0.1,
            handlelength=1.6, columnspacing=1.4, handletextpad=0.4, labelspacing=0.3)

        # Letters: one column for each column of panels
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()

        def column(*axes):
            return min(to_fig.transform(_drawn_bbox(ax, renderer).p0)[0] for ax in axes)

        col_left, col_right = column(ax_a, ax_c), column(ax_b, ax_d)
        for letter, ax, top, col in (("a", ax_a, 1.0, col_left), ("b", ax_b, 1.0, col_right),
                                     ("c", ax_c, top_c / h, col_left), ("d", ax_d, top_c / h, col_right)):
            _panel_letters(fig, ((letter, ax),), top=top, left=col)
        _placed_save(fig, "Figure8")


# ============================================================
# SUPPLEMENTARY FIGURES
# ============================================================
# Each supplementary figure is drawn at the size at which LaTeX places it
# (placed width x height in millimetres in its docstring, the width to be set
# in 02_Article_Supplementary.tex) with the 7 pt text of the main figures, and
# saved through _placed_save, which refuses text below 6 pt, overlapping texts
# and artists outside the canvas. Panels carry bold lowercase letters at the
# top left, and no PNG carries a title that repeats the caption.

C_SIG_BAR = "#4D4D4D"   # enrichment bars at FDR < 0.05 (Supplementary Figures 8 and 9)
C_LIGHT = "#C8C8C8"     # bars and points that are not significant


def _sup_canvas(width_mm, height_mm):
    """Canvas of a supplementary figure at its placed size in millimetres."""
    return _placed_canvas(width_mm / 25.4, height_mm / 25.4)


def _sup_layout(fig, letters=(), pad=0.3, bottom=0.0, w_pad=None):
    """tight_layout, keeping a band of LETTER_BAND_PT at the top for the panel
    letters when there are any, then the letters through _panel_letters.
    letters is a sequence of (letter, axes) pairs. bottom keeps a strip free
    under the axes for a figure legend, in figure coordinates, and w_pad is
    the width padding of tight_layout."""
    band = LETTER_BAND_PT / 72 / fig.get_size_inches()[1] if letters else 0.0
    fig.tight_layout(pad=pad, rect=(0, bottom, 1, 1 - band), w_pad=w_pad)
    if letters:
        _panel_letters(fig, letters)


def _sup_save(fig, name):
    """Save FigureSN.png, the file the LaTeX includes, after the checks of
    _placed_save."""
    _placed_save(fig, name, directory=SFIG_DIR)


def _check_stat(key, value, tol=0.0):
    """Raise when a number computed from a CSV differs from its key in
    paper_statistics.json by more than tol (for keys stored rounded)."""
    expected = _stat(key)
    if abs(float(value) - float(expected)) > tol:
        raise ValueError(f"{key}: the figure data give {value}, paper_statistics.json has {expected}")


def _top_rows_without_split_tie(df, n, col="pvalue"):
    """The first n rows of df, already sorted by col, or fewer when rows n and
    n + 1 tie on col, so that the cut never keeps part of a tie and drops the
    rest by name order."""
    if len(df) <= n:
        return df
    vals = df[col].to_numpy()
    k = n
    while k > 0 and np.isclose(vals[k - 1], vals[k], rtol=1e-12, atol=0.0):
        k -= 1
    return df.head(k)


def _plain_log_ticks(axis, ticks):
    """Ticks of a log axis written as plain numbers (0.01, 0.5, 1, 1,000),
    without minor tick labels. Math-text powers of ten would print their
    exponents at 0.7 of the font size, below 6 pt."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _segment_hits_box(p, q, box):
    """Whether the segment from p to q touches the box (x0, y0, x1, y1)."""
    t0, t1 = 0.0, 1.0
    for start, delta, lo, hi in ((p[0], q[0] - p[0], box[0], box[2]), (p[1], q[1] - p[1], box[1], box[3])):
        if delta == 0:
            if start < lo or start > hi:
                return False
            continue
        a, b = sorted(((lo - start) / delta, (hi - start) / delta))
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def _segments_cross(p1, q1, p2, q2):
    """Whether the segments p1-q1 and p2-q2 cross each other."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return side(p1, q1, p2) * side(p1, q1, q2) < 0 and side(p2, q2, p1) * side(p2, q2, q1) < 0


def _place_leader_labels(fig, ax, anchors, names, points, point_r_pt, lines=(), arrowprops=None,
                         keep_side=False, **text_kw):
    """Label each anchor (data coordinates) with a leader line, the label at a
    position computed from the drawn data instead of typed. The candidates
    lie on rings around the anchor (8 to 88 pt, every 15 degrees). A
    candidate is kept only when its text box stays inside the axes and covers
    no point, no line of lines, the legend, an earlier label or an earlier
    leader line, and its own leader line crosses no earlier label or leader.
    With keep_side the label must also stay on the same side as its anchor of
    every line of lines. Among those, the one whose leader line crosses the
    fewest points wins, then the closest. Raises when a label has no free
    position, so a change in the data can never put a label over the points
    without an error.

    Call after the layout. points is an (n, 2) array of every point drawn,
    point_r_pt the largest marker radius in points and lines a sequence of
    ("h", y) or ("v", x) reference lines in data coordinates."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    pts = ax.transData.transform(np.asarray(points, dtype=float))
    r_px = point_r_pt * px
    margin = 1.0 * px
    axb = ax.get_window_extent(renderer)
    blocked = []
    if ax.get_legend() is not None:
        lb = ax.get_legend().get_window_extent(renderer)
        blocked.append((lb.x0, lb.y0, lb.x1, lb.y1))
    line_px = [(k, ax.transData.transform((0, v))[1] if k == "h" else ax.transData.transform((v, 0))[0])
               for k, v in lines]
    placed = []
    leaders = []   # (anchor, center of the label) of the labels already placed
    for (x0, y0), name in zip(anchors, names):
        a = ax.transData.transform((x0, y0))
        probe = ax.text(0, 0, name, **text_kw)
        ext = probe.get_window_extent(renderer)
        probe.remove()
        w, h = ext.width + 2 * margin, ext.height + 2 * margin
        # Points under the anchor marker are crossed by every leader line
        away = np.hypot(*(pts - a).T) > 2 * r_px
        best = None
        for rad in range(8, 92, 4):
            for ang in range(0, 360, 15):
                dx, dy = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
                cx, cy = a[0] + rad * px * dx, a[1] + rad * px * dy
                ha = "left" if dx > 0.3 else "right" if dx < -0.3 else "center"
                va = "bottom" if dy > 0.3 else "top" if dy < -0.3 else "center"
                bx0 = cx - margin if ha == "left" else cx - w + margin if ha == "right" else cx - w / 2
                by0 = cy - margin if va == "bottom" else cy - h + margin if va == "top" else cy - h / 2
                box = (bx0, by0, bx0 + w, by0 + h)
                if box[0] < axb.x0 or box[2] > axb.x1 or box[1] < axb.y0 or box[3] > axb.y1:
                    continue
                if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in blocked):
                    continue
                if ((pts[:, 0] > box[0] - r_px) & (pts[:, 0] < box[2] + r_px)
                        & (pts[:, 1] > box[1] - r_px) & (pts[:, 1] < box[3] + r_px)).any():
                    continue
                if any((k == "h" and box[1] < v < box[3]) or (k == "v" and box[0] < v < box[2])
                       for k, v in line_px):
                    continue
                if keep_side and any((box[1] > v) != (a[1] > v) if k == "h" else (box[0] > v) != (a[0] > v)
                                     for k, v in line_px):
                    continue
                # The leader runs from the anchor toward the center of the box
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                if any(_segment_hits_box(a, center, b) for b in blocked):
                    continue
                if any(_segment_hits_box(p, q, box) or _segments_cross(a, center, p, q) for p, q in leaders):
                    continue
                seg = center - a
                u = np.clip(((pts - a) @ seg) / (seg @ seg), 0.0, 1.0)
                dist = np.hypot(*(pts - a - np.outer(u, seg)).T)
                score = (int(((dist < r_px) & away).sum()), rad)
                if best is None or score < best[0]:
                    best = (score, cx, cy, ha, va, box)
        if best is None:
            raise ValueError(f"No free position for the label {name!r}")
        _, cx, cy, ha, va, box = best
        blocked.append(box)
        leaders.append((a, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])))
        placed.append(ax.annotate(name, (x0, y0), xytext=((cx - a[0]) / px, (cy - a[1]) / px),
                                  textcoords="offset points", ha=ha, va=va, arrowprops=arrowprops,
                                  **text_kw))
    return placed


def sfig04_tf_drug_heatmap():
    """Supplementary Figure 4 (FigureS4.png, placed 170 x 80 mm): coefficients
    of the per-drug Ridge models that transfer the predictions to SCAN-B, on
    the ten core TFs.

    The models are refit with the protocol of the clinical transfer (scripts
    17 and 21): the TFs shared by PRISM and SCAN-B, a QuantileTransformer with
    normal output and a StandardScaler fitted on the training cell lines of
    cell_line_holdout_split (seed 42), and a Ridge (alpha 100) fitted on the
    training cell lines of each drug. With this protocol the refit models
    reproduce the SCAN-B prediction matrix of script 17. The TFs are the ten
    with the lowest mean IG rank across the three seeds
    (tf_importance_ig_stability.csv), the same ten as Table 4 and Figure 7
    (Figure7.png). The drugs are four positive controls and the seven
    candidates of Table 5, grouped by mechanism of action, with the groups
    named above the heatmap so that no bracket crosses a drug label.
    Coefficients are winsorized at the 2nd and 98th percentiles for display,
    as the caption says. Blue marks a negative coefficient (toward
    sensitivity) and vermillion a positive one (toward resistance), the colors
    of Figure 3c.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from collections import OrderedDict
    from src.data.splits import cell_line_holdout_split

    stab = _csv(PAPER_RESULTS / "tf_importance_ig_stability.csv")
    t3 = _csv(PAPER_RESULTS / "table3_positive_controls.csv")
    t5 = _csv(PAPER_RESULTS / "table5_candidates.csv")
    if stab is None or t3 is None or t5 is None:
        return _skip("SFig04_tf_drug_heatmap", "IG stability table, table3 or table5 missing")
    tf_path = DATA / "precision_processed/tf_activities_all_prism.csv"
    scanb_path = DATA / "precision_processed/tf_activities_scanb.csv"
    if not tf_path.exists() or not scanb_path.exists():
        return _skip("SFig04_tf_drug_heatmap", "PRISM or SCAN-B TF activities missing")
    tf_prism = pd.read_csv(tf_path, index_col=0)
    scanb_tfs = pd.read_csv(scanb_path, index_col=0, nrows=0).columns
    try:
        from src.data.load_drug_response import load_prism_response
        from src.data.drug_names import normalize_drug_columns
        drug_resp = normalize_drug_columns(load_prism_response(use_processed=True))
    except Exception as e:
        return _skip("SFig04_tf_drug_heatmap", f"cannot load PRISM response: {e}")

    # Transfer protocol of scripts 17 and 21
    train_cells, _ = cell_line_holdout_split(drug_resp, 0.2, 42)
    shared_tfs = sorted(set(tf_prism.columns) & set(scanb_tfs))
    train_tf = tf_prism.loc[[c for c in train_cells if c in tf_prism.index], shared_tfs]
    _check_stat("SCANB_n_tfs", len(shared_tfs))
    _check_stat("holdout_n_train_cells", len(train_tf))
    qt = QuantileTransformer(output_distribution="normal", random_state=42).fit(train_tf)
    X = StandardScaler().fit_transform(qt.transform(train_tf))
    train_resp = drug_resp.loc[train_tf.index]

    # Four positive controls of Table 3 and the seven candidates of Table 5,
    # ordered by mechanism of action
    controls = OrderedDict([
        ("paclitaxel", "Tubulin"),
        ("docetaxel",  "Tubulin"),
        ("epirubicin", "Topo II"),
        ("olaparib",   "PARP"),
    ])
    candidates = OrderedDict([
        ("osimertinib", "EGFR"),
        ("erlotinib",   "EGFR"),
        ("pelitinib",   "EGFR"),
        ("brigatinib",  "ALK/\nEGFR"),   # two lines, one column is narrower than "ALK/EGFR"
        ("saracatinib", "Src"),
        ("trametinib",  "MEK"),
        ("entinostat",  "HDAC"),
    ])
    if not set(controls) <= set(t3["Drug"]) or set(candidates) != set(t5["Drug"]):
        raise ValueError("Supplementary Figure 4: drugs do not match Tables 3 and 5")
    drug_moa = OrderedDict([*controls.items(), *candidates.items()])

    # Ten core TFs, in the functional order below. A core TF without an
    # annotation is appended as "IG core", so the panel never drops a TF of
    # Table 4 if the stability ranking shifts.
    tf_function = OrderedDict([
        ("ZEB1",    "EMT"),
        ("KLF8",    "EMT"),
        ("AEBP1",   "Stromal / TNBC"),
        ("MZF1",    "Stromal / TNBC"),
        ("SPDEF",   "Epithelial"),
        ("RELA",    "NF-κB"),
        ("STAT2",   "Interferon"),
        ("IRX1",    "Homeobox / developmental"),
        ("BARX1",   "Homeobox / developmental"),
        ("SETBP1",  "SET binding"),
        ("CREB3L1", "Stress response"),
        ("HIVEP1",  "NF-κB"),
        ("REL",     "NF-κB"),
        ("ASXL1",   "Epigenetic"),
        ("BCOR",    "Epigenetic"),
    ])
    core10 = stab.sort_values("mean_rank").head(10)["tf"].tolist()
    if set(core10) != set(_stat("ig_multi_seed_core10_names")):
        raise ValueError("Supplementary Figure 4: core TFs differ from ig_multi_seed_core10_names")
    if not set(core10) <= set(shared_tfs):
        raise ValueError("Supplementary Figure 4: a core TF is not shared by PRISM and SCAN-B")
    sel_tfs = [t for t in tf_function if t in core10]
    for t in core10:
        if t not in sel_tfs:
            tf_function[t] = "IG core"
            sel_tfs.append(t)

    tf_idx = [shared_tfs.index(t) for t in sel_tfs]
    coef = pd.DataFrame(np.nan, index=sel_tfs, columns=list(drug_moa))
    for drug in drug_moa:
        y = train_resp[drug].to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            raise ValueError(f"Supplementary Figure 4: {drug} has no transfer model (fewer than 30 cell lines)")
        coef[drug] = Ridge(alpha=100).fit(X[mask], y[mask]).coef_[tf_idx]

    # Winsorized at the 2nd and 98th percentiles for display (declared in the caption)
    lo, hi = np.percentile(coef.to_numpy(), [2, 98])
    coef = coef.clip(lower=lo, upper=hi)
    vmax = float(np.abs(coef.to_numpy()).max())
    cmap = LinearSegmentedColormap.from_list("sens_res", [C_PRIMARY, "white", C_ACCENT])

    with _placed_rc():
        fig = _sup_canvas(170, 80)
        ax = fig.add_subplot()
        im = ax.imshow(coef.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(np.arange(coef.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(coef.shape[0] + 1) - 0.5, minor=True)
        ax.grid(True, which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(range(coef.shape[1]))
        # Vertical, because the boxes of neighbouring slanted labels overlap
        ax.set_xticklabels([d.capitalize() for d in coef.columns], rotation=90)
        ax.set_yticks(range(coef.shape[0]))
        ax.set_yticklabels([f"{t} ({tf_function[t]})" for t in sel_tfs])
        ax.tick_params(axis="both", which="major", length=0)

        # Mechanism-of-action groups above the heatmap: a line over the
        # columns of the group and its name above the line
        groups = OrderedDict()
        for i, d in enumerate(coef.columns):
            groups.setdefault(drug_moa[d], []).append(i)
        trans = ax.get_xaxis_transform()
        for moa, idx in groups.items():
            ax.plot([idx[0] - 0.42, idx[-1] + 0.42], [1.025, 1.025], color=FIG1_TEXT, lw=0.8,
                    transform=trans, clip_on=False, solid_capstyle="butt")
            ax.text((idx[0] + idx[-1]) / 2, 1.045, moa, ha="center", va="bottom", transform=trans)

        # Shrunk so that its end labels stay below the MOA names, and set off
        # from the heatmap so that they do not reach its last column
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.07, aspect=12, shrink=0.68)
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-12 else f"{v:.3f}".replace("-", "−")))
        cbar.ax.tick_params(length=2, width=0.5)
        cbar.outline.set_linewidth(0.5)
        cbar.set_label("Ridge coefficient")
        cbar.ax.text(0.5, 1.03, "Toward\nresistance", ha="center", va="bottom",
                     transform=cbar.ax.transAxes, linespacing=1.1)
        cbar.ax.text(0.5, -0.03, "Toward\nsensitivity", ha="center", va="top",
                     transform=cbar.ax.transAxes, linespacing=1.1)
        _sup_layout(fig)
        _sup_save(fig, "FigureS4")


def sfig03_umap_embeddings():
    """Supplementary Figure 3 (FigureS3.png, placed 170 x 75 mm): UMAP of the
    cell-line embeddings of the seed-42 GNN.

    (a) The seven most frequent lineages other than breast in muted tones and
    greys, breast in black and larger, the rest in light grey. The palette
    leaves out blue, orange and vermillion, which (b) uses for the breast
    receptor subtypes (HER2+, ER+, TNBC), so no color has two meanings in the
    figure. The subtype counts must equal the breast_n_* keys of the caption,
    and the lineage counts of (a) must add up to the kg_cell_lines key.

    In (b) some lines lie about 2 pt apart at the placed size, so each subtype
    has its own marker and they are stacked so that none hides another: TNBC
    as filled circles at the bottom, HER2+ as open rings over them (a TNBC
    point under a ring shows through it) and ER+, the smallest group, as
    filled diamonds with a white edge on top.
    """
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
    # Every cell line of the graph has a lineage, so the counts of (a) add up
    # to the cell lines of the graph
    _check_stat("kg_cell_lines", len(shared))
    emb = cell_emb.loc[shared]
    meta = mi.loc[shared]

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    emb2d = reducer.fit_transform(emb.values)

    lineage = meta["OncotreeLineage"].fillna("Other")
    is_breast = (lineage.str.lower() == "breast").to_numpy()
    top7 = lineage[~is_breast].value_counts().head(7).index.tolist()
    # Muted tones and greys (Paul Tol's muted scheme without its blue and red)
    muted = ["#44AA99", "#117733", "#999933", "#DDCC77", "#AA4499", "#882255", "#777777"]

    # Same classification as the breast_n_* keys of generate_paper_results.py
    from src.data.breast_subtypes import breast_subtypes
    breast_meta = meta[is_breast]
    subtype_class = breast_subtypes(meta, breast_meta.index)
    subtypes = ["TNBC", "ER+", "HER2+"]   # order of the caption and the legend
    counts = {s: int((subtype_class == s).sum()) for s in subtypes}
    keys = {"TNBC": "breast_n_tnbc", "ER+": "breast_n_erpos", "HER2+": "breast_n_her2pos"}
    for s, key in keys.items():
        _check_stat(key, counts[s])
    _check_stat("breast_n_other", int((subtype_class == "Other").sum()))
    _check_stat("breast_n_cell_lines", int(is_breast.sum()))

    with _placed_rc():
        fig = _sup_canvas(170, 75)
        ax_a, ax_b = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1.0]})

        other = ~lineage.isin(top7).to_numpy() & ~is_breast
        ax_a.scatter(emb2d[other, 0], emb2d[other, 1], color="#E3E3E3", s=3, lw=0,
                     label=f"Other lineages ({other.sum()})", zorder=1)
        for lin, color in zip(top7, muted):
            mask = (lineage == lin).to_numpy()
            ax_a.scatter(emb2d[mask, 0], emb2d[mask, 1], color=color, s=5, lw=0, alpha=0.85,
                         label=f"{lin} ({mask.sum()})", zorder=2)
        ax_a.scatter(emb2d[is_breast, 0], emb2d[is_breast, 1], color="black", s=11, lw=0.3,
                     edgecolors="white", label=f"Breast ({is_breast.sum()})", zorder=3)
        handles, labels = ax_a.get_legend_handles_labels()
        order = [len(handles) - 1, *range(1, len(handles) - 1), 0]   # breast first, other last
        ax_a.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), frameon=False, handletextpad=0.2,
                    borderaxespad=0.0, markerscale=1.6, title="Lineage (cell lines)")

        # Drawn bottom to top: TNBC, HER2+ rings, ER+ (see the docstring)
        breast_xy = emb2d[is_breast]
        style = {
            "TNBC":  dict(marker="o", s=16, facecolors=C_ACCENT, edgecolors="white", linewidths=0.4, zorder=3),
            "HER2+": dict(marker="o", s=30, facecolors="none", edgecolors=C_PRIMARY, linewidths=1.0, zorder=4),
            "ER+":   dict(marker="D", s=13, facecolors=C_HIGHLIGHT, edgecolors="white", linewidths=0.4, zorder=5),
        }
        handles = {}
        for s in style:
            mask = (subtype_class.loc[breast_meta.index] == s).to_numpy()
            handles[s] = ax_b.scatter(breast_xy[mask, 0], breast_xy[mask, 1],
                                      label=f"{s} (n = {counts[s]})", **style[s])
        # Legend in the order of the caption
        ax_b.legend([handles[s] for s in subtypes], [handles[s].get_label() for s in subtypes],
                    loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                    handletextpad=0.2, borderaxespad=0.0, markerscale=1.3, title="Breast subtype")

        for ax in (ax_a, ax_b):
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        _sup_layout(fig, (("a", ax_a), ("b", ax_b)))
        _sup_save(fig, "FigureS3")


def sfig01_training_curves():
    """Supplementary Figure 1 (FigureS1.png, placed 170 x 60 mm): training of
    the seed-42 GraphSAGE model, (a) train and test Huber loss and (b) test
    Pearson correlation per epoch, from gnn_training_history.csv."""
    history = _csv(PAPER_RESULTS / "gnn_training_history.csv")
    if history is None:
        return _skip("SFig01_training_curves", "training history missing")

    with _placed_rc():
        fig = _sup_canvas(170, 60)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(history["epoch"], history["train_loss"], color=C_PRIMARY, lw=1.0, label="Train")
        ax1.plot(history["epoch"], history["test_loss"], color=C_ACCENT, lw=1.0, label="Test")
        ax1.set_yscale("log")
        losses = history[["train_loss", "test_loss"]].to_numpy()
        _plain_log_ticks(ax1.yaxis, [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
                                     if losses.min() / 1.2 <= t <= losses.max() * 1.2])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Huber loss (log scale)")
        ax1.legend(frameon=False, loc="upper right")

        ax2.plot(history["epoch"], history["test_pearson"], color=C_ACCENT, lw=1.0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Pearson correlation")
        for ax in (ax1, ax2):
            ax.set_xlim(0, history["epoch"].max())
        _sup_layout(fig, (("a", ax1), ("b", ax2)))
        _sup_save(fig, "FigureS1")


def sfig02_perdrug_distribution():
    """Supplementary Figure 2 (FigureS2.png, placed 120 x 70 mm): per-drug test
    Pearson correlation of the seed-42 Random Forest on TF activities, Ridge on
    expression and the seed-42 GraphSAGE model over the same drugs, as step
    histograms on common bins (np.histogram_bin_edges over the three models),
    so that the heights compare across models. The Random Forest file must be
    its random_state 42 fit, whose median is table1_RF_TF_seed42_median_perdrug,
    and the GNN file the seed-42 model, whose median is
    igdiag_seed42_median_perdrug_SAGE."""
    datasets = [
        ("Random Forest (seed 42)",       "per_drug_RF_TF.csv",            C_PRIMARY),
        ("Ridge (expression)",            "per_drug_Ridge_expression.csv", C_NEUTRAL),
        ("GraphSAGE (seed 42)",           "per_drug_SAGE_learned_emb.csv", C_HIGHLIGHT),
    ]
    series = {}
    for name, fname, _ in datasets:
        df = _csv(PAPER_RESULTS / fname)
        if df is None:
            return _skip("SFig02_perdrug_distribution", f"{fname} missing")
        series[name] = df.set_index("drug")["pearson_r"]
    if len({frozenset(s.dropna().index) for s in series.values()}) != 1:
        raise ValueError("Supplementary Figure 2: the per-drug CSVs cover different drugs")
    _check_stat("igdiag_seed42_median_perdrug_SAGE",
                series["GraphSAGE (seed 42)"].median(), tol=0.0005)
    _check_stat("table1_RF_TF_seed42_median_perdrug",
                series["Random Forest (seed 42)"].median(), tol=0.0005)
    values = {k: s.dropna().to_numpy() for k, s in series.items()}
    bins = np.histogram_bin_edges(np.concatenate(list(values.values())), bins=40)

    with _placed_rc():
        fig = _sup_canvas(120, 70)
        ax = fig.add_subplot()
        for name, _, color in datasets:
            ax.hist(values[name], bins=bins, histtype="step", color=color, lw=1.1, label=name)
        ax.axvline(0, color=FIG1_TEXT, lw=0.6, ls="--")
        ax.set_xlabel("Per-drug Pearson correlation (test cell lines)")
        ax.set_ylabel("Number of drugs")
        # White box without an edge, so the grid does not run behind the text
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
        _sup_layout(fig)
        _sup_save(fig, "FigureS2")


def sfig14_crossval():
    """Supplementary Figure 14 (FigureS14.png, placed 100 x 62 mm): per-drug
    Pearson correlation of the PRISM-to-GDSC cross-screen evaluation of the
    HGT variant, with its median, checked against crossval_n_drugs and
    crossval_median_pearson."""
    cv = _csv(PAPER_RESULTS / "crossval_prism_to_gdsc.csv")
    if cv is None:
        return _skip("SFig14_crossval", "crossval CSV missing")
    r = cv["pearson_r"].dropna()
    _check_stat("crossval_n_drugs", len(r))
    _check_stat("crossval_median_pearson", r.median(), tol=0.0005)

    with _placed_rc():
        fig = _sup_canvas(100, 62)
        ax = fig.add_subplot()
        ax.hist(r, bins=30, color=C_PRIMARY, edgecolor="white", lw=0.3)
        ax.axvline(0, color=FIG1_TEXT, lw=0.7, ls="--", label="r = 0")
        ax.axvline(r.median(), color=C_HIGHLIGHT, lw=1.2,
                   label=f"Median = {_stat('crossval_median_pearson'):.3f}")
        ax.set_xlabel("Per-drug Pearson correlation, PRISM to GDSC")
        ax.set_ylabel("Number of drugs")
        ax.legend(frameon=False, loc="upper right")
        _sup_layout(fig)
        _sup_save(fig, "FigureS14")


def fig_multicohort():
    """Figure 5 (Figure5.png): survival associations in the three cohorts.

    Panel a counts the drugs of each cohort at FDR < 0.05 (filled bars) inside
    those at nominal p < 0.05 (outline bars of the same color), with a dotted
    line at 5% of the drugs tested, the number expected by chance at the
    nominal threshold. Panel b is the area-proportional overlap of SCAN-B and
    METABRIC, with the METABRIC-only count and the overlap (shared drugs and
    how many of them agree in the direction of the hazard ratio) in callouts
    outside the circles, whose leader lines end in a dot inside the region they
    label. Counts come from the survival CSVs and are checked against
    paper_statistics.json.
    Cohort colors are those of Figure 6: SCAN-B blue, METABRIC orange, both
    green, and TCGA-BRCA grey. Drawn at 170 x 55 mm, with the band of the
    panel letters at the top.
    """
    try:
        from matplotlib_venn import venn2, venn2_circles
    except ImportError:
        return _skip("Figure5", "matplotlib_venn not installed")
    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    tcga = _csv(PAPER_RESULTS / "survival_TCGA.csv")
    if any(x is None for x in [scanb, metabric, tcga]):
        return _skip("Figure5", "survival CSVs missing in paper/results/")

    sig = {k: set(df.loc[df["padj"] < 0.05, "drug"]) for k, df in
           (("SCANB", scanb), ("METABRIC", metabric), ("TCGA", tcga))}
    total = {"SCANB": len(scanb), "METABRIC": len(metabric), "TCGA": len(tcga)}
    nominal = {k: int((df["pvalue"] < 0.05).sum()) for k, df in
               (("SCANB", scanb), ("METABRIC", metabric), ("TCGA", tcga))}
    for k in sig:
        if len(sig[k]) != int(_stat(f"{k}_drugs_FDR005")) or total[k] != int(_stat(f"{k}_n_drugs_tested")):
            raise ValueError(f"{k}: survival CSV and paper_statistics.json disagree")
        _check_stat(f"{k}_drugs_nominal005", nominal[k])
        if not len(sig[k]) <= nominal[k] <= total[k]:
            raise ValueError(f"{k}: {nominal[k]} drugs at nominal p < 0.05 is not between the "
                             f"{len(sig[k])} at FDR < 0.05 and the {total[k]} tested")
    # One chance line for the three bars, so the cohorts must share the denominator
    if len(set(total.values())) != 1:
        raise ValueError(f"Figure 5a: the cohorts tested different numbers of drugs: {total}")
    chance = 0.05 * total["SCANB"]
    s_sig, m_sig = sig["SCANB"], sig["METABRIC"]
    shared = sorted(s_sig & m_sig)
    hr_s = scanb.set_index("drug").loc[shared, "HR"]
    hr_m = metabric.set_index("drug").loc[shared, "HR"]
    n_same = int((((hr_s > 1) & (hr_m > 1)) | ((hr_s < 1) & (hr_m < 1))).sum())
    if len(shared) != int(_stat("shared_drugs_both_cohorts")) or \
            n_same != int(_stat("shared_drugs_direction_consistent")):
        raise ValueError("SCAN-B and METABRIC overlap does not match the shared_drugs_* keys")
    n10, n01, n11 = len(s_sig) - len(shared), len(m_sig) - len(shared), len(shared)

    with _placed_rc():
        fig = _placed_canvas(PLACED_W_IN, 55 / 25.4)
        w, h = fig.get_size_inches()
        band = LETTER_BAND_PT / 72   # free strip at the top for the panel letters

        # a: drugs per cohort, FDR < 0.05 filled inside nominal p < 0.05 in outline
        ax_a = fig.add_axes(_inch_rect(fig, 0.46, 0.34, 2.30, h - 0.34 - band))
        cohorts = [("SCANB", "SCAN-B\n(RNA-seq)", C_PRIMARY),
                   ("METABRIC", "METABRIC\n(microarray)", C_HIGHLIGHT),
                   ("TCGA", "TCGA-BRCA\n(RNA-seq)", C_NEUTRAL)]
        colors = [c for _, _, c in cohorts]
        bar_w = 0.6
        ax_a.bar(range(3), [len(sig[k]) for k, _, _ in cohorts], width=bar_w, color=colors, edgecolor="none")
        ax_a.bar(range(3), [nominal[k] for k, _, _ in cohorts], width=bar_w, facecolor="none",
                 edgecolor=colors, linewidth=0.8)
        ax_a.axhline(y=chance, color=FIG1_TEXT, ls=":", lw=0.8, zorder=3)
        # White behind the labels above the bars, which the grid would cross.
        # TCGA-BRCA stays below the chance line, so its label goes above the line.
        for i, (k, _, _) in enumerate(cohorts):
            n, t = len(sig[k]), total[k]
            ax_a.annotate(f"{n:,}/{t:,} ({100 * n / t:.0f}%)\nat FDR < 0.05", xy=(i, max(nominal[k], chance)), xytext=(0, 5),
                          textcoords="offset points", ha="center", va="bottom", linespacing=1.15,
                          bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"})
            ax_a.annotate(f"{nominal[k]:,}", xy=(i + bar_w / 2, nominal[k]), xytext=(1.5, 0),
                          textcoords="offset points", ha="left", va="center")
        ax_a.set_xticks(range(3))
        ax_a.set_xticklabels([lab for _, lab, _ in cohorts])
        ax_a.set_xlim(-0.55, 2.55)
        ax_a.set_ylim(0, max(nominal.values()) * 1.30)
        ax_a.set_ylabel(f"Drugs (of {total['SCANB']:,} tested)")
        ax_a.grid(True, axis="y", color="#E8E9EB")
        ax_a.grid(False, axis="x")
        ax_a.tick_params(axis="x", length=0)
        ax_a.legend(handles=[
            mpatches.Patch(facecolor=C_FRAME, edgecolor="none", label="FDR < 0.05"),
            mpatches.Patch(facecolor="none", edgecolor=C_FRAME, linewidth=0.8, label="Nominal p < 0.05"),
            plt.Line2D([], [], color=FIG1_TEXT, ls=":", lw=0.8, label="5% of drugs tested"),
        ], frameon=False, loc="upper right", handlelength=1.5)
        # No printed count may sit on the chance line
        fig.canvas.draw()
        y_chance = ax_a.transData.transform((0, chance))[1]
        crossed = [t.get_text() for t in ax_a.texts
                   if matplotlib.text.Text.get_window_extent(t).y0 <= y_chance
                   <= matplotlib.text.Text.get_window_extent(t).y1]
        if crossed:
            raise ValueError(f"Figure 5a: the chance line crosses the labels {crossed}")

        # b: SCAN-B and METABRIC overlap
        b_left = 3.0
        ax_b = fig.add_axes(_inch_rect(fig, b_left, 0.02, w - b_left - 0.02, h - 0.04))
        v = venn2(subsets=(n10, n01, n11), set_labels=None, ax=ax_b)
        for rid, col, alpha in (("10", C_PRIMARY, 0.40), ("01", C_HIGHLIGHT, 0.75), ("11", C_GREEN, 0.75)):
            patch = v.get_patch_by_id(rid)
            patch.set_facecolor(col)
            patch.set_alpha(alpha)
            patch.set_edgecolor("none")
        venn2_circles(subsets=(n10, n01, n11), ax=ax_b, linewidth=0.6, color=FIG1_TEXT)
        pos = {}
        for rid in ("10", "01", "11"):
            lab = v.get_label_by_id(rid)
            pos[rid] = lab.get_position()
            lab.set_text("")
        (c1, c2), (r1, r2) = v.centers, v.radii
        # Data limits that give the SCAN-B circle a radius of 0.62 in, with its
        # label above it and the whole centered vertically, in the aspect of
        # the axes so that the circles stay round
        bw, bh = ax_b.get_position().width * w, ax_b.get_position().height * h
        s = 0.62 / r1   # inches per data unit
        label_h = 0.18  # height of the SCAN-B label above the circle, inches
        y1 = c1.y + r1 + (label_h + (bh - 2 * 0.62 - label_h) / 2) / s
        x0 = c1.x - r1 - 0.04 / s
        ax_b.set_xlim(x0, x0 + bw / s)
        ax_b.set_ylim(y1 - bh / s, y1)
        ax_b.set_aspect("equal", adjustable="box")
        ax_b.set_axis_off()

        ax_b.text(pos["10"][0], pos["10"][1], f"{n10:,}", ha="center", va="center")
        ax_b.text(c1.x, c1.y + r1 + 0.04 / s, f"SCAN-B ({len(s_sig):,})", ha="center", va="bottom")
        x_call = c2.x + r2 + 0.08 / s
        ax_b.text(x_call, c2.y, f"METABRIC ({len(m_sig):,})", ha="left", va="center")
        lead = {"arrowstyle": "-", "lw": 0.5, "color": FIG1_TEXT, "shrinkA": 1, "shrinkB": 0}
        # The overlap leader leaves the lens straight up through the upper
        # crossing point of the two circles, so it crosses neither crescent,
        # and turns right above the METABRIC circle toward its label
        d_c = c2.x - c1.x
        x_cross = c1.x + (d_c ** 2 - r2 ** 2 + r1 ** 2) / (2 * d_c)
        h_cross = np.sqrt(r1 ** 2 - (x_cross - c1.x) ** 2)
        anchor11 = (x_cross, c1.y + 0.45 * h_cross)
        y_top = max(c1.y + h_cross, c2.y + r2) + 0.14 / s
        ax_b.annotate(f"{n11:,} shared, {n_same:,} same direction", xy=anchor11,
                      xytext=(x_call, y_top), ha="left", va="center",
                      arrowprops={**lead, "connectionstyle": "angle,angleA=0,angleB=90,rad=0"})
        ax_b.annotate(f"{n01:,} METABRIC only", xy=pos["01"],
                      xytext=(x_call, c2.y - r2 - 0.14 / s), ha="left", va="center", arrowprops=lead)
        # A dot marks the region each leader points at
        for xy in (anchor11, pos["01"]):
            ax_b.plot(*xy, marker="o", markersize=2.2, color=FIG1_TEXT, markeredgewidth=0, zorder=5)

        _panel_letters(fig, (("a", ax_a), ("b", ax_b)))
        _placed_save(fig, "Figure5")


def sfig05_qqplot():
    """Supplementary Figure 5 (FigureS5.png, placed 170 x 58 mm): QQ-plots of
    the per-drug survival p-values in (a) SCAN-B, (b) METABRIC and (c)
    TCGA-BRCA, in the cohort colors of Figures 5 and 6. The genomic-control
    lambda and the number of drugs at FDR < 0.05 printed in each panel are the
    keys of paper_statistics.json, checked against the survival CSVs."""
    cohorts = [
        ("SCANB",    "SCAN-B (RNA-seq)",      "survival_SCANB.csv",    C_PRIMARY),
        ("METABRIC", "METABRIC (microarray)", "survival_METABRIC.csv", C_HIGHLIGHT),
        ("TCGA",     "TCGA-BRCA (RNA-seq)",   "survival_TCGA.csv",     C_NEUTRAL),
    ]
    frames = {k: _csv(PAPER_RESULTS / f) for k, _, f, _ in cohorts}
    if any(df is None for df in frames.values()):
        return _skip("SFig05_qqplot", "survival CSVs missing in paper/results/")

    with _placed_rc():
        fig = _sup_canvas(170, 58)
        axes = fig.subplots(1, 3)
        for ax, (k, name, _, color) in zip(axes, cohorts):
            df = frames[k]
            p = df["pvalue"].to_numpy()
            if not (p > 0).all():
                raise ValueError(f"Supplementary Figure 5: {k} has a p-value of 0")
            lam = np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1)
            _check_stat(f"{k}_qq_lambda", lam, tol=0.005)
            _check_stat(f"{k}_drugs_FDR005", int((df["padj"] < 0.05).sum()))
            _check_stat(f"{k}_n_drugs_tested", len(p))

            observed = -np.log10(np.sort(p))
            expected = -np.log10(np.arange(1, len(p) + 1) / (len(p) + 1))
            ax.scatter(expected, observed, color=color, s=4, lw=0, alpha=0.7, zorder=3)
            ax.plot([0, expected.max()], [0, expected.max()], ls="--", color=FIG1_TEXT, lw=0.6, zorder=2)
            ax.set_title(name)
            ax.set_xlabel("Expected −log10(p)")
            ax.set_ylabel("Observed −log10(p)")
            ax.text(0.04, 0.96,
                    f"λ = {_stat(f'{k}_qq_lambda'):.2f}\n"
                    f"{_fmt_n(f'{k}_drugs_FDR005')} of {_fmt_n(f'{k}_n_drugs_tested')} drugs\n"
                    f"at FDR < 0.05",
                    transform=ax.transAxes, ha="left", va="top", linespacing=1.2, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        _sup_layout(fig, tuple(zip("abc", axes)))
        _sup_save(fig, "FigureS5")


def sfig07_forest_plot():
    """Supplementary Figure 7 (FigureS7.png, placed 130 x 62 mm): hazard ratios
    with the 95% CIs of the penalized Cox models (script 43) for the five
    positive controls in SCAN-B and METABRIC.

    All five reach FDR < 0.05 in SCAN-B and only epirubicin in METABRIC. TCGA
    is omitted as underpowered (QQ lambda = 0.84, Supplementary Figure 5).
    Drugs are ordered by SCAN-B HR, weakest effect (highest HR) at top. CIs
    come directly from the Cox model, not reconstructed from p-values. Filled
    markers and an asterisk in a column at the right edge mark FDR < 0.05
    within the cohort, and the legend shows the same asterisk glyph.
    """
    df = _csv(PAPER_RESULTS / "forest_positive_controls.csv")
    if df is None:
        return _skip("SFig07_forest_plot", "forest_positive_controls.csv missing")

    order = (df[df["cohort"] == "SCAN-B"]
             .sort_values("HR", ascending=False)["drug"].tolist())
    cohorts = [("SCAN-B", C_PRIMARY, -0.17), ("METABRIC", C_HIGHLIGHT, 0.17)]
    asterisk = "$*$"   # a marker, not a text, so the check of math text does not apply

    with _placed_rc():
        fig = _sup_canvas(130, 62)
        ax = fig.add_subplot()
        for row, drug in enumerate(order):
            for cohort, color, dy in cohorts:
                r = df[(df["drug"] == drug) & (df["cohort"] == cohort)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = row + dy
                sig = r["padj_in_cohort"] < 0.05
                ax.plot([r["CI_low"], r["CI_high"]], [y, y], "-", color=color, lw=1.1, zorder=4)
                ax.plot(r["HR"], y, "o", color=color if sig else "white", markeredgecolor=color,
                        markeredgewidth=1.0, markersize=5, zorder=5)
                if sig:
                    # In a column at the right edge, clear of the HR = 1 line
                    ax.plot(0.965, y, marker=asterisk, color=color, markersize=6, lw=0, zorder=5,
                            transform=ax.get_yaxis_transform())

        ax.axvline(x=1, color=FIG1_TEXT, ls="--", lw=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(df["CI_low"].min() / 1.25, df["CI_high"].max() * 1.6)
        lo, hi = ax.get_xlim()
        _plain_log_ticks(ax.xaxis, [t for t in (0.05, 0.1, 0.25, 0.5, 1, 2) if lo <= t <= hi])
        ax.set_xlabel("Hazard ratio per unit of predicted AUC (log scale)")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([d.capitalize() for d in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.grid(False, axis="y")
        # Two legends: the cohorts, with their sizes from paper_statistics.json
        # (Table 2, never typed here), and below them the FDR encoding, whose
        # title covers only its own two entries
        leg_cohort = ax.legend(handles=[
            plt.Line2D([0], [0], marker="o", color=C_PRIMARY, lw=0, markersize=5,
                       label=f"SCAN-B (n = {_fmt_n('SCANB_n_patients_cox')})"),
            plt.Line2D([0], [0], marker="o", color=C_HIGHLIGHT, lw=0, markersize=5,
                       label=f"METABRIC (n = {_fmt_n('METABRIC_n_patients_cox')})"),
        ], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0,
            title="Cohort", alignment="left")
        ax.add_artist(leg_cohort)
        open_marker = plt.Line2D([0], [0], marker="o", color="white", markeredgecolor=FIG1_TEXT,
                                 markeredgewidth=1.0, lw=0, markersize=5)
        filled_marker = plt.Line2D([0], [0], marker="o", color=FIG1_TEXT, lw=0, markersize=5)
        star = plt.Line2D([0], [0], marker=asterisk, color=FIG1_TEXT, lw=0, markersize=6)
        ax.legend([open_marker, (filled_marker, star)],
                  ["FDR ≥ 0.05: open marker", "FDR < 0.05: filled marker\nand asterisk"],
                  handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=0.6)},
                  loc="upper left", bbox_to_anchor=(1.02, 0.52), frameon=False, borderaxespad=0.0,
                  title="Within-cohort FDR", alignment="left")
        _sup_layout(fig)
        _sup_save(fig, "FigureS7")


def sfig11_penalizer():
    """Supplementary Figure 11 (FigureS11.png, placed 170 x 62 mm): penalizer
    sensitivity analysis on all 1,447 drugs in SCAN-B (PAM50-adjusted Cox, one
    sample per patient), read from penalizer_sweep.csv (script 45). (a) Drugs
    at FDR < 0.05 and at nominal p < 0.05 per penalizer, with the FDR counts
    checked against the pensweep_n_fdr_* keys. (b) Range of the hazard ratio
    among the FDR-significant drugs. Both x axes mark the four penalizers
    (0.01, 0.1, 0.5 and 1) as plain numbers.
    """
    sw = _csv(PAPER_RESULTS / "penalizer_sweep.csv")
    if sw is None:
        return _skip("SFig11_penalizer", "penalizer_sweep.csv missing")
    sw = sw.sort_values("penalizer")
    pens = sw["penalizer"].tolist()
    n_drugs = int(sw["n_drugs"].iloc[0])
    n_fdr = sw["n_fdr_all"].tolist()
    n_nominal = sw["n_nominal_all"].tolist()
    hr_max = sw["hr_max_fdr"].tolist()
    hr_min = sw["hr_min_fdr"].tolist()
    keys = {0.01: "pensweep_n_fdr_pen001", 0.1: "pensweep_n_fdr_pen01",
            0.5: "pensweep_n_fdr_pen05", 1.0: "pensweep_n_fdr_pen10"}
    if sorted(keys) != pens:
        raise ValueError(f"Supplementary Figure 11: penalizers {pens} differ from {sorted(keys)}")
    for pen, n in zip(pens, n_fdr):
        _check_stat(keys[pen], n)

    with _placed_rc():
        fig = _sup_canvas(170, 62)
        ax_a, ax_b = fig.subplots(1, 2)

        ax_a.axhline(y=0.05 * n_drugs, color=FIG1_TEXT, ls=":", lw=0.8,
                     label="5% expected by chance")
        ax_a.plot(pens, n_nominal, "o-", color=C_NEUTRAL, lw=1.0, markersize=3.5,
                  label="Nominal p < 0.05")
        ax_a.plot(pens, n_fdr, "o-", color=C_ACCENT, lw=1.2, markersize=4, label="FDR < 0.05")
        # Each count on the side of its point that no segment of the FDR line
        # leaves from: below and to the left, where the line arrives from the
        # upper left and leaves to the lower right, and to the upper right of
        # the last point, clear of the 5% line
        for i, (x, y) in enumerate(zip(pens, n_fdr)):
            last = i == len(pens) - 1
            ax_a.annotate(f"{y:,}", (x, y), textcoords="offset points",
                          xytext=(3.5, 2.5) if last else (-3.5, -2.5),
                          ha="left" if last else "right", va="bottom" if last else "top",
                          color=C_ACCENT, zorder=5)
        ax_a.set_ylabel(f"Drugs (of {n_drugs:,} tested)")
        ax_a.set_ylim(0, max(n_nominal) * 1.12)
        handles, labels = ax_a.get_legend_handles_labels()
        # Lower left, above the 5% line, where no line of the panel passes
        chance = 0.05 * n_drugs / ax_a.get_ylim()[1]
        ax_a.legend(handles[::-1], labels[::-1], frameon=False, loc="lower left",
                    bbox_to_anchor=(0.0, chance + 0.04), borderaxespad=0.3)

        ax_b.fill_between(pens, hr_min, hr_max, alpha=0.15, color=C_NEUTRAL, lw=0)
        ax_b.plot(pens, hr_max, "o-", color=C_ACCENT, lw=1.1, markersize=3.5,
                  label="Maximum HR (FDR < 0.05 drugs)")
        ax_b.plot(pens, hr_min, "o-", color=C_PRIMARY, lw=1.1, markersize=3.5,
                  label="Minimum HR (FDR < 0.05 drugs)")
        ax_b.axhline(y=1, color=FIG1_TEXT, ls="--", lw=0.6)
        ax_b.set_yscale("log")
        _plain_log_ticks(ax_b.yaxis, [0.001, 0.01, 0.1, 1, 10, 100, 1000])
        ax_b.set_ylim(0.0007, 1500)
        ax_b.set_ylabel("Hazard ratio (log scale)")
        ax_b.legend(frameon=False, loc="upper right")

        for ax in (ax_a, ax_b):
            ax.set_xscale("log")
            _plain_log_ticks(ax.xaxis, pens)
            ax.set_xlim(pens[0] / 1.6, pens[-1] * 1.6)
            ax.set_xlabel("Cox penalizer λ")
        _sup_layout(fig, (("a", ax_a), ("b", ax_b)))
        _sup_save(fig, "FigureS11")


def sfig13_basal_her2():
    """Supplementary Figure 13 (FigureS13.png, placed 110 x 70 mm): per-drug
    log-rank p-values of the within-subtype survival screen in the pooled
    basal-like and HER2-enriched (PAM50) SCAN-B patients (script 19b). Every
    count in the legend and the annotation is a subtype_pooled_* key,
    checked against basal_her2_survival.csv."""
    basal = _csv(PAPER_RESULTS / "basal_her2_survival.csv")
    if basal is None:
        return _skip("SFig13_basal_her2", "basal_her2_survival.csv missing")
    p = basal["pvalue"]
    n = len(basal)
    if not (p > 0).all():
        raise ValueError("Supplementary Figure 13: a log-rank p-value of 0")
    _check_stat("subtype_pooled_n_tested", n)
    _check_stat("subtype_pooled_n_nominal_p05", int((p < 0.05).sum()))
    _check_stat("subtype_pooled_n_bonferroni", int((p < 0.05 / n).sum()))
    _check_stat("subtype_pooled_drugs_FDR005", int((basal["padj"] < 0.05).sum()))

    with _placed_rc():
        fig = _sup_canvas(110, 70)
        ax = fig.add_subplot()
        ax.hist(-np.log10(p), bins=50, color=C_PRIMARY, edgecolor="white", lw=0.3)
        ax.axvline(x=-np.log10(0.05), color=FIG1_TEXT, ls="--", lw=0.7,
                   label=f"p = 0.05 ({_fmt_n('subtype_pooled_n_nominal_p05')} drugs pass)")
        ax.axvline(x=-np.log10(0.05 / n), color=FIG1_TEXT, ls=":", lw=1.0,
                   label=f"Bonferroni ({_fmt_n('subtype_pooled_n_bonferroni')} drugs pass)")
        ax.set_xlabel("−log10(log-rank p)")
        ax.set_ylabel("Number of drugs")
        # Legend and annotation on white, so the grid does not run behind the text
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
        ax.text(0.97, 0.68,
                f"{_fmt_n('subtype_pooled_n_patients')} patients, "
                f"{_fmt_n('subtype_pooled_n_events')} events\n"
                f"{_fmt_n('subtype_pooled_drugs_FDR005')} drugs at FDR < 0.05",
                transform=ax.transAxes, ha="right", va="top", linespacing=1.2, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        _sup_layout(fig)
        _sup_save(fig, "FigureS13")


def sfig10_aurora_drugs():
    """Supplementary Figure 10 (FigureS10.png, placed 120 x 85 mm): paired,
    patient-level change in the predicted drug response between primary and
    metastatic tumors in AURORA-US (aurora_drug_paired.csv: the 39 patients
    of the TF analysis, metastatic samples averaged per patient, paired
    Wilcoxon test with Benjamini-Hochberg correction over the drugs).

    x is the paired difference in predicted AUC, metastasis minus primary, and
    y is -log10 of the BH-adjusted p. Blue marks the drugs predicted more
    active in metastasis (lower AUC) at FDR < 0.05 and vermillion those
    predicted less active, as blue marks sensitivity and vermillion resistance
    in Figure 3c and Supplementary Figure 4. Only the three drugs of the
    caption are labeled, in black over a white stroke. The counts in the
    legend are checked against the aurora_drug_paired_* keys.
    """
    import matplotlib.patheffects as pe

    dr = _csv(PAPER_RESULTS / "aurora_drug_paired.csv")
    if dr is None:
        return _skip("SFig10_aurora_drugs", "aurora_drug_paired.csv missing in paper/results/")
    delta = dr["delta_auc"]
    padj = dr["padj_bh_paired"]
    if not np.allclose(delta, dr["metastasis_mean_auc"] - dr["primary_mean_auc"]):
        raise ValueError("Supplementary Figure 10: delta_auc is not metastasis minus primary")
    if not (padj > 0).all():
        raise ValueError("Supplementary Figure 10: an adjusted p-value of 0")
    if not (dr["n_paired"] == int(_stat("aurora_drug_paired_n_patients"))).all():
        raise ValueError("Supplementary Figure 10: n_paired differs from aurora_drug_paired_n_patients")
    sig = padj < 0.05
    more = sig & (delta < 0)
    less = sig & (delta > 0)
    if not (dr.loc[sig, "more_active_in_metastasis"] == (delta[sig] < 0)).all():
        raise ValueError("Supplementary Figure 10: more_active_in_metastasis disagrees with delta_auc")
    _check_stat("aurora_drug_paired_n_tested", len(dr))
    _check_stat("aurora_drug_paired_n_bh005", int(sig.sum()))
    _check_stat("aurora_drug_paired_n_bh005_more_active", int(more.sum()))
    _check_stat("aurora_drug_paired_n_bh005_less_active", int(less.sum()))
    y = -np.log10(padj)

    with _placed_rc():
        fig = _sup_canvas(120, 85)
        ax = fig.add_subplot()
        ax.scatter(delta[~sig], y[~sig], color=C_LIGHT, s=5, lw=0, zorder=2,
                   label=f"FDR ≥ 0.05 ({int((~sig).sum()):,})")
        # Opaque, so blue and vermillion keep the hues of sensitivity and
        # resistance instead of fading toward the orange of other figures
        ax.scatter(delta[more], y[more], color=C_PRIMARY, s=6, lw=0, zorder=3,
                   label=f"More active in metastasis ({int(more.sum()):,})")
        ax.scatter(delta[less], y[less], color=C_ACCENT, s=6, lw=0, zorder=3,
                   label=f"Less active in metastasis ({int(less.sum()):,})")
        ax.axhline(y=-np.log10(0.05), color=FIG1_TEXT, ls="--", lw=0.6, zorder=1, label="FDR = 0.05")
        ax.axvline(x=0, color=FIG1_TEXT, ls=":", lw=0.6, zorder=1)

        anchors = []
        for drug in ("entinostat", "paclitaxel", "panobinostat"):
            r = dr[dr["drug"] == drug]
            if len(r) != 1:
                raise ValueError(f"Supplementary Figure 10: {drug} missing from aurora_drug_paired.csv")
            anchors.append((drug, float(r["delta_auc"].iloc[0]),
                            float(-np.log10(r["padj_bh_paired"].iloc[0]))))
            ax.scatter([anchors[-1][1]], [anchors[-1][2]], s=14, facecolors="none", edgecolors="black",
                       linewidths=0.7, zorder=5)
        ax.set_xlabel("Paired difference in predicted AUC (metastasis − primary)")
        ax.set_ylabel("−log10(BH-adjusted paired p)")
        ax.set_ylim(-0.02 * y.max(), y.max() * 1.05)
        # White box, so the zero line does not run through the legend
        ax.legend(loc="upper right", markerscale=1.8, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=1.0)
        _sup_layout(fig)

        # Labels placed after the layout on free ground, joined to their drug
        # by a leader line: no label covers a point, a reference line, the
        # legend or another label (checked, not typed, see _place_leader_labels)
        stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
        _place_leader_labels(
            fig, ax, [(x0, y0) for _, x0, y0 in anchors], [d.capitalize() for d, _, _ in anchors],
            np.column_stack([delta, y]), point_r_pt=np.sqrt(14) / 2,
            lines=(("h", -np.log10(0.05)), ("v", 0.0)),
            color="black", path_effects=stroke, zorder=6,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "shrinkA": 1, "shrinkB": 2.5})
        _sup_save(fig, "FigureS10")


# Src (the kinase, not an acronym) as in Table 5, the text and Supplementary Figure 4
_MOA_ACRONYMS = {
    "akt": "AKT", "mtor": "mTOR", "hsp": "HSP", "plk": "PLK", "ppar": "PPAR", "chk": "CHK",
    "dna": "DNA", "rna": "RNA", "src": "Src", "egfr": "EGFR", "ikk": "IKK", "atpase": "ATPase",
    "nfkb": "NF-κB", "flt3": "FLT3", "vegfr": "VEGFR", "bcl": "BCL", "cdk": "CDK",
    "hdac": "HDAC", "parp": "PARP", "pi3k": "PI3K", "mek": "MEK", "jak": "JAK", "abl": "ABL",
    "bcr-abl": "BCR-ABL", "kit": "KIT", "pdgfr": "PDGFR", "raf": "RAF", "alk": "ALK",
    "igf-1": "IGF-1", "hmgcr": "HMGCR", "gaba": "GABA", "erk": "ERK", "fgfr": "FGFR",
}


def _moa_display(moa):
    """PRISM mechanism-of-action class as printed: acronyms in their usual
    case (mTOR inhibitor, DNA methyltransferase inhibitor) and the first
    letter of the class capitalized."""
    words = [_MOA_ACRONYMS.get(w, w) for w in moa.split(" ")]
    first = words[0]
    if first == moa.split(" ")[0]:
        words[0] = first[:1].upper() + first[1:]
    return " ".join(words)


def _enrichment_bars(res_df, label_col, name, width_mm, height_mm, n_top, count_label,
                     display, n_sig_key, n_tested_key):
    """Horizontal bars of -log10(p) of the n_top classes or targets with the
    lowest p (cut before a tie that crosses n_top), dark for FDR < 0.05 and
    light otherwise, with a dashed line at p = 0.05 in the legend and the
    counts drawn on a white box so the line does not cross them."""
    if len(res_df) != int(_stat(n_tested_key)):
        raise ValueError(f"{name}: {len(res_df)} rows, {n_tested_key} = {_stat(n_tested_key)}")
    n_sig = int((res_df["padj"] < 0.05).sum())
    _check_stat(n_sig_key, n_sig)
    res_df = res_df.sort_values(["pvalue", label_col], kind="mergesort").reset_index(drop=True)
    top = _top_rows_without_split_tie(res_df, n_top).iloc[::-1]
    logp = -np.log10(top["pvalue"].to_numpy())

    with _placed_rc():
        fig = _sup_canvas(width_mm, height_mm)
        ax = fig.add_subplot()
        ypos = np.arange(len(top))
        colors = [C_SIG_BAR if q < 0.05 else C_LIGHT for q in top["padj"]]
        ax.barh(ypos, logp, color=colors, edgecolor="none", height=0.65, zorder=2)
        ax.axvline(x=-np.log10(0.05), color=FIG1_TEXT, ls="--", lw=0.7, zorder=3)
        for yi, x, (_, r) in zip(ypos, logp, top.iterrows()):
            # A white box 1.5 pt wider than the glyphs on every side (bbox pad
            # in points), so the dashed line stops short of the count
            ax.annotate(count_label(r), (x, yi), xytext=(3.5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=PLACED_MIN_FS + 0.5, zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_yticks(ypos)
        ax.set_yticklabels([display(v) for v in top[label_col]])
        ax.set_ylim(-0.6, len(top) - 0.4)
        ax.set_xlim(0, logp.max() * 1.22)
        ax.grid(False, axis="y")
        ax.set_xlabel("−log10(p), one-sided Fisher's exact test")
        ax.legend(handles=[
            mpatches.Patch(facecolor=C_SIG_BAR, label=f"FDR < 0.05 ({n_sig})"),
            mpatches.Patch(facecolor=C_LIGHT, label="FDR ≥ 0.05"),
            plt.Line2D([0], [0], color=FIG1_TEXT, ls="--", lw=0.7, label="p = 0.05"),
        ], frameon=False, loc="lower right")
        _sup_layout(fig)
        _sup_save(fig, name)
    return len(top)


def sfig09_moa_enrichment():
    """Supplementary Figure 9 (FigureS9.png, placed 150 x 90 mm): enrichment of
    mechanism-of-action classes among the drugs with a significant survival
    association (Fisher meta-analysis, FDR < 0.05).

    Reads moa_enrichment.csv (script 52, copied to paper/results/ by
    generate_paper_results.py), so the figure, the caption and
    paper_statistics.json share one computation. The 20 classes with the
    lowest p are shown (fewer if the cut would split a tie), sorted stably by
    p and then by name.
    """
    res_df = _csv(PAPER_RESULTS / "moa_enrichment.csv")
    if res_df is None:
        return _skip("SFig09_moa_enrichment", "moa_enrichment.csv missing in paper/results/")
    if not np.allclose(res_df["pct_sig"], 100 * res_df["n_sig"] / res_df["n_total"], atol=0.05):
        raise ValueError("Supplementary Figure 9: pct_sig is not n_sig / n_total")

    def pct_half_up(r):
        # Integer rounding half up, so 5/8 prints 63%, not the 62% of the
        # round-half-to-even of f"{62.5:.0f}"
        n_sig, n_total = int(r["n_sig"]), int(r["n_total"])
        return (200 * n_sig + n_total) // (2 * n_total)

    n = _enrichment_bars(res_df, "moa", "FigureS9", 150, 90, 20,
                         lambda r: f"{r['n_sig']}/{r['n_total']} ({pct_half_up(r)}%)",
                         _moa_display, "moa_enrichment_n_classes_fdr05",
                         "moa_enrichment_n_classes_tested")
    logger.info("  FigureS9 shows %d of %d classes", n, len(res_df))


def sfig06_scanb_metabric_scatter():
    """Supplementary Figure 6 (FigureS6.png, placed 100 x 100 mm): per-drug
    survival p-values in SCAN-B against METABRIC.

    Colors follow Figures 5 and 6: FDR < 0.05 in SCAN-B only blue, in METABRIC
    only orange, in both with the same direction of the hazard ratio green, in
    both with opposite directions purple, and in neither light grey. Counts
    come from the survival CSVs and are checked against paper_statistics.json,
    and so is the Spearman correlation printed in the panel.
    """
    scanb = _csv(PAPER_RESULTS / "survival_SCANB.csv")
    metabric = _csv(PAPER_RESULTS / "survival_METABRIC.csv")
    if scanb is None or metabric is None:
        return _skip("SFig06_scanb_metabric_scatter", "survival CSVs missing in paper/results/")

    cols = ["drug", "pvalue", "padj", "HR"]
    merged = scanb[cols].merge(metabric[cols], on="drug", suffixes=("_s", "_m"))
    if len(merged) != len(scanb) or len(merged) != len(metabric):
        raise ValueError("Supplementary Figure 6: SCAN-B and METABRIC cover different drugs")
    if not ((merged["pvalue_s"] > 0) & (merged["pvalue_m"] > 0)).all():
        raise ValueError("Supplementary Figure 6: a survival p-value of 0")
    x = -np.log10(merged["pvalue_s"])
    y = -np.log10(merged["pvalue_m"])
    s_sig = merged["padj_s"] < 0.05
    m_sig = merged["padj_m"] < 0.05
    same = (((merged["HR_s"] > 1) & (merged["HR_m"] > 1)) |
            ((merged["HR_s"] < 1) & (merged["HR_m"] < 1)))
    _check_stat("SCANB_drugs_FDR005", int(s_sig.sum()))
    _check_stat("METABRIC_drugs_FDR005", int(m_sig.sum()))
    _check_stat("shared_drugs_both_cohorts", int((s_sig & m_sig).sum()))
    _check_stat("shared_drugs_direction_consistent", int((s_sig & m_sig & same).sum()))
    rho = spearmanr(x, y)[0]
    _check_stat("scanb_metabric_neglogp_spearman", rho, tol=0.0005)

    cats = [
        ("Neither",                       ~s_sig & ~m_sig,       C_LIGHT,     4, 1),
        ("SCAN-B only",                   s_sig & ~m_sig,        C_PRIMARY,   6, 2),
        ("METABRIC only",                 ~s_sig & m_sig,        C_HIGHLIGHT, 9, 3),
        ("Both, same HR direction",       s_sig & m_sig & same,  C_GREEN,     9, 4),
        ("Both, opposite HR direction",   s_sig & m_sig & ~same, C_SECONDARY, 14, 5),
    ]
    with _placed_rc():
        fig = _sup_canvas(100, 100)
        ax = fig.add_subplot()
        for label, mask, color, size, z in cats:
            ax.scatter(x[mask], y[mask], color=color, s=size, lw=0.3 if z == 5 else 0,
                       edgecolors="black" if z == 5 else "none", alpha=0.85, zorder=z,
                       label=f"{label} ({int(mask.sum()):,})")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], frameon=False,
                  loc="upper right", title="FDR < 0.05 in", markerscale=1.5)
        ax.set_xlabel("−log10(p), SCAN-B")
        ax.set_ylabel("−log10(p), METABRIC")
        ax.set_xlim(-0.015 * x.max(), x.max() * 1.04)
        ax.set_ylim(-0.015 * y.max(), y.max() * 1.25)
        ax.text(0.97, 0.03, f"Spearman ρ = {_stat('scanb_metabric_neglogp_spearman'):.2f}",
                transform=ax.transAxes, ha="right", va="bottom")
        _sup_layout(fig)
        _sup_save(fig, "FigureS6")


def sfig08_target_enrichment():
    """Supplementary Figure 8 (FigureS8.png, placed 110 x 85 mm): enrichment of
    drug targets among the drugs with a significant survival association.

    Reads target_enrichment.csv (script 52, copied to paper/results/ by
    generate_paper_results.py), so the figure, the caption and
    paper_statistics.json share one computation. The targets are sorted
    stably by p and then by name, and the list stops before a tie that
    crosses the 20th row, since which of the tied targets would be shown
    would otherwise depend on their names. The number shown is logged.
    """
    res_df = _csv(PAPER_RESULTS / "target_enrichment.csv")
    if res_df is None:
        return _skip("SFig08_target_enrichment", "target_enrichment.csv missing in paper/results/")
    n = _enrichment_bars(res_df, "target", "FigureS8", 110, 85, 20,
                         lambda r: f"{r['n_sig']}/{r['n_total']}", str,
                         "target_enrichment_n_fdr05", "target_enrichment_n_targets_tested")
    logger.info("  FigureS8 shows %d of %d targets", n, len(res_df))


def sfig12_graph_ablation():
    """Supplementary Figure 12 (FigureS12.png, placed 170 x 82 mm): graph
    component ablation. One row per configuration in a fixed structural order,
    the four that keep the drug-target relation (the control with its edges
    rewired among them) above a divider and the three without it below it.
    The mean and SD over the seeds as an open diamond
    with error bar on the row, and one point per initialization seed in a strip
    under it, spread vertically by a fixed offset that keeps them off the
    diamond. The offset spans less than one marker, so seeds with nearly equal
    correlation overlap, as three seeds of the rewired control do in a.

    a) Global Pearson on a fixed 0.02 grid, with the drug-mean reference of
    Table 1 (dashed, the style of Figure 2a). b) Median per-drug Pearson from
    zero, with the Random Forest on TF activities of Table 1 (dotted). The
    drug-mean reference predicts one value per drug, so it has no per-drug
    correlation and is not drawn in b.

    Positions come from the CSVs and printed numbers from the keys, checked
    against each other. The Full graph row must be the GraphSAGE of Table 1
    seed by seed. The figure raises when a seed reaches the drug-mean reference
    or the Random Forest median, because the caption says that none does, and
    when the configurations differ from the fixed order."""
    abl = _csv(PAPER_RESULTS / "graph_ablation.csv")
    per_seed = _csv(PAPER_RESULTS / "graph_ablation_per_seed.csv")
    fair = _csv(PAPER_RESULTS / "fair_comparison_per_seed.csv")
    t1 = _csv(PAPER_RESULTS / "table1_model_comparison.csv")
    if abl is None or per_seed is None or fair is None or t1 is None:
        return _skip("SFig12_graph_ablation", "graph_ablation.csv, graph_ablation_per_seed.csv, "
                     "fair_comparison_per_seed.csv or table1_model_comparison.csv missing")

    # Fixed order, top to bottom. A configuration renamed by script 27 raises
    # instead of reordering the rows without notice.
    order = ("full", "shuffled_drug_target", "no_collectri", "no_ppi", "no_drug_target",
             "no_ppi_no_drug_target", "response_only")
    n_with_dt = 4
    pairs = per_seed[["config", "tag"]].drop_duplicates()
    if (len(pairs) != len(order) or set(pairs["tag"]) != set(order)
            or sorted(abl["config"]) != sorted(pairs["config"])):
        raise ValueError(f"Supplementary Figure 12: configurations {sorted(abl['config'])} and tags "
                         f"{sorted(pairs['tag'])} differ from the fixed order {order}")
    abl = abl.assign(tag=abl["config"].map(dict(zip(pairs["config"], pairs["tag"]))))
    abl = abl.set_index("tag").loc[list(order)]

    seeds = {tag: per_seed[per_seed["tag"] == tag].sort_values("init_seed") for tag in order}
    for tag, r in abl.iterrows():
        cfg, s = r["config"], seeds[tag]
        _check_stat(f"ablation_{cfg}_pearson", r["test_pearson"], tol=0.0005)
        _check_stat(f"ablation_{cfg}_pearson_sd", r["test_pearson_sd"], tol=0.0005)
        _check_stat(f"ablation_{cfg}_n_seeds", len(s))
        _check_stat(f"ablation_median_perdrug_{tag}_mean", s["median_pearson"].mean(), tol=0.0005)
        _check_stat(f"ablation_median_perdrug_{tag}_sd", s["median_pearson"].std(ddof=1), tol=0.0005)
        # The diamond comes from the summary table and the points from the per-seed one
        if not np.allclose([s["test_pearson"].mean(), s["test_pearson"].std(ddof=1)],
                           [r["test_pearson"], r["test_pearson_sd"]], rtol=0, atol=1e-6):
            raise ValueError(f"Supplementary Figure 12: {cfg}, the seeds do not give the mean and SD "
                             f"of graph_ablation.csv")
    n_seeds = {len(s) for s in seeds.values()}
    if len(n_seeds) != 1:
        raise ValueError(f"Supplementary Figure 12: configurations with different seed counts {n_seeds}")

    # The Full graph row is the GraphSAGE of Table 1, seed by seed
    sage = fair[fair["model"] == "SAGE_learned_emb"].set_index("run")
    full = seeds["full"].set_index("run")
    if sorted(sage.index) != sorted(full.index) or not (
            np.allclose(sage.loc[full.index, "global_pearson"], full["test_pearson"], rtol=0, atol=1e-6)
            and np.allclose(sage.loc[full.index, "median_pearson"], full["median_pearson"], rtol=0, atol=1e-6)):
        raise ValueError("Supplementary Figure 12: the Full graph runs differ from the GraphSAGE runs of Table 1")

    t1 = t1.set_index("Model")
    drug_mean = float(_stat("table1_Drug_mean_reference_global_pearson"))
    rf_median = float(_stat("table1_RF_TF_median_perdrug"))
    _check_stat("table1_Drug_mean_reference_global_pearson", t1.at["Drug_mean_reference", "Global_Pearson"])
    _check_stat("table1_RF_TF_median_perdrug", t1.at["RF_TF", "Median_PerDrug_Pearson"])
    # The caption says that every seed of every configuration stays below both
    if not (per_seed["test_pearson"] < drug_mean).all():
        raise ValueError("Supplementary Figure 12: a seed reaches the drug-mean reference of Table 1, "
                         "the caption no longer holds")
    if not (per_seed["median_pearson"] < rf_median).all():
        raise ValueError("Supplementary Figure 12: a seed reaches the Random Forest median of Table 1, "
                         "the caption no longer holds")
    n_drugs = per_seed["n_drugs"].unique()
    if len(n_drugs) != 1 or int(n_drugs[0]) != int(_stat("table1_SAGE_learned_emb_n_drugs")):
        raise ValueError(f"Supplementary Figure 12: drug counts {n_drugs} differ from Table 1")

    def color_of(tag):
        if tag == "full":
            return C_PRIMARY
        if tag == "shuffled_drug_target":
            return C_SKYBLUE
        if tag in ("no_ppi_no_drug_target", "response_only"):
            return C_ACCENT
        return C_HIGHLIGHT

    # A gap between the two groups, and the divider in the gap, raised to clear
    # the value label of the row below it
    gap = 0.35
    ypos = {tag: (len(order) - 1 - i) + (gap if i < n_with_dt else 0.0) for i, tag in enumerate(order)}
    divider = ypos[order[n_with_dt]] + 0.5 + gap / 2 + 0.12
    top = max(ypos.values())

    with _placed_rc():
        fig = _sup_canvas(170, 82)
        ax_a, ax_b = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
        # a) on a 0.02 grid, from below the lowest seed to past the drug-mean line
        ax_a.set_xlim(np.floor(per_seed["test_pearson"].min() * 50) / 50, np.ceil((drug_mean + 0.008) * 50) / 50)
        ax_a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        # b) from zero to past the Random Forest line
        ax_b.set_xlim(-0.005, np.ceil((rf_median + 0.008) * 50) / 50)
        ax_b.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))

        def strip(ax, tag, vals, mean, sd, label):
            lo, hi = ax.get_xlim()
            if min(vals.min(), mean - sd) < lo or max(vals.max(), mean + sd) > hi:
                raise ValueError(f"Supplementary Figure 12: {tag} falls outside the axis, {lo:g} to {hi:g}")
            yi, c = ypos[tag], color_of(tag)
            # The seeds go in a strip under the mean, so that seeds and diamond
            # do not hide each other in the rows with little spread.
            jitter = np.linspace(-0.06, 0.06, len(vals))
            ax.errorbar(mean, yi + 0.04, xerr=sd, fmt="none", ecolor=c, capsize=2.5, elinewidth=0.9,
                        capthick=0.9, zorder=3)
            ax.scatter([mean], [yi + 0.04], marker="D", s=26, facecolors="white", edgecolors=c,
                       linewidths=1.0, zorder=4)
            ax.scatter(vals, yi - 0.2 + jitter, color=c, s=10, edgecolors="white", linewidths=0.4, zorder=5)
            # High enough for the white box to clear the top seed
            ax.text(mean, yi + 0.29, label, ha="center", va="bottom", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})

        # Positions from the CSVs, printed numbers from the keys (checked equal above)
        for tag, r in abl.iterrows():
            cfg, s = r["config"], seeds[tag]
            strip(ax_a, tag, s["test_pearson"].to_numpy(), r["test_pearson"], r["test_pearson_sd"],
                  f"{_stat(f'ablation_{cfg}_pearson'):.3f} ± {_stat(f'ablation_{cfg}_pearson_sd'):.3f}")
            strip(ax_b, tag, s["median_pearson"].to_numpy(), s["median_pearson"].mean(),
                  s["median_pearson"].std(ddof=1),
                  f"{_stat(f'ablation_median_perdrug_{tag}_mean'):.3f} ± "
                  f"{_stat(f'ablation_median_perdrug_{tag}_sd'):.3f}")

        def ref_label(ax, x, text):
            # Along the reference line, 1.5 pt to its right, centred on the rows
            return ax.text(x, top / 2, text, rotation=90, ha="left", va="center", color=FIG1_TEXT, zorder=2,
                           bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                           transform=mtransforms.offset_copy(ax.transData, fig=fig, x=1.5, units="points"))

        ax_a.axvline(drug_mean, color=C_NEUTRAL, lw=1, ls=(0, (4, 2)), zorder=1)
        label_a = ref_label(ax_a, drug_mean, str(t1.at["Drug_mean_reference", "Label"]))
        ax_a.set_xlabel("Global Pearson correlation (test cell lines)")
        ax_b.axvline(0, color=C_NEUTRAL, lw=0.6, zorder=0)
        ax_b.axvline(rf_median, color=FIG1_TEXT, lw=0.8, ls=":", zorder=1)
        label_b = ref_label(ax_b, rf_median, "RF (TF activities)")
        ax_b.set_xlabel(f"Median per-drug Pearson correlation ({_fmt_n('table1_SAGE_learned_emb_n_drugs')} drugs)")

        # No vertical grid: every value is printed, as in Figure 2
        for ax in (ax_a, ax_b):
            ax.axhline(divider, color=C_NEUTRAL, lw=0.6, zorder=0)
            ax.grid(False)
        ax_a.set_yticks([ypos[t] for t in order])
        # The configuration keeps the TF activities computed with CollecTRI as
        # cell-line features, so the label names the edges that are removed.
        if (abl["config"].str.count(r" \(no prior knowledge\)") != (abl.index == "response_only")).any():
            raise ValueError("Supplementary Figure 12: the response-only label of graph_ablation.csv changed")
        ax_a.set_yticklabels([abl.at[t, "config"].replace(" (no prior knowledge)", "\n(no prior-knowledge edges)")
                              for t in order])
        ax_a.set_ylim(-0.6, top + 0.8)
        fig.legend(handles=[
            mpatches.Patch(facecolor=C_PRIMARY, label="Full graph"),
            mpatches.Patch(facecolor=C_SKYBLUE, label="Drug-target edges rewired"),
            mpatches.Patch(facecolor=C_HIGHLIGHT, label="One edge type removed"),
            mpatches.Patch(facecolor=C_ACCENT, label="Two or all edge types removed"),
        ], frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.2)
        _sup_layout(fig, (("a", ax_a), ("b", ax_b)), bottom=0.07, w_pad=1.6)
        renderer = fig.canvas.get_renderer()
        for ax, t in ((ax_a, label_a), (ax_b, label_b)):
            box, frame = t.get_window_extent(renderer), ax.get_window_extent(renderer)
            if box.x1 > frame.x1 or box.y0 < frame.y0 or box.y1 > frame.y1:
                raise ValueError(f"Supplementary Figure 12: the label {t.get_text()!r} leaves its axes")
        _sup_save(fig, "FigureS12")


def fig3c_ig_heatmap():
    """Figure 3c (panel fig3c of Figure3.png): signed IG attributions, top 20 TFs x 11 drugs.

    The attribution of a TF is its contribution to the predicted AUC relative
    to the mean-train baseline: red raises and blue lowers the predicted AUC
    of these cell lines. Because the sign is relative to the baseline, it is
    not the direction in which higher activity of the TF moves the prediction,
    so the figure does not call the TFs resistance or sensitivity drivers.
    Rows must be the ig_top20_names of paper_statistics.json. The positive
    controls of Table 3 come first and the candidates of Table 5 after the
    dashed separator.
    """
    ig_path = PAPER_RESULTS / "tf_drug_ig_signed.csv"
    t3 = _csv(PAPER_RESULTS / "table3_positive_controls.csv")
    t5 = _csv(PAPER_RESULTS / "table5_candidates.csv")
    if not ig_path.exists() or t3 is None or t5 is None:
        return _skip("Fig3c_ig_heatmap", "tf_drug_ig_signed.csv, table3 or table5 missing")
    ig = pd.read_csv(ig_path, comment="#").set_index("tf")
    if ig.index.tolist() != list(_stat("ig_top20_names")):
        raise ValueError("tf_drug_ig_signed.csv rows do not match ig_top20_names")
    if ig.size != int(_stat("ig_signed_n_cells")):
        raise ValueError("tf_drug_ig_signed.csv size does not match ig_signed_n_cells")
    # Positive controls in the order of Table 3, then the candidates grouped by
    # mechanism (EGFR, Src, MEK, HDAC), which must be the drugs of Table 5
    controls = [d for d in t3["Drug"] if d in ig.columns]
    candidates = ["osimertinib", "erlotinib", "brigatinib", "pelitinib",
                  "saracatinib", "trametinib", "entinostat"]
    if set(candidates) != set(t5["Drug"]) or len(candidates) != int(_stat("candidates_n_selected")):
        raise ValueError("The candidate columns of Figure 3c do not match table5_candidates.csv")
    if set(ig.columns) != set(controls) | set(candidates):
        raise ValueError("tf_drug_ig_signed.csv columns are not the controls and candidates")
    ig = ig[controls + candidates]

    # Diverging colormap on a symmetric scale
    vmax = float(np.abs(ig.to_numpy()).max())

    with _fig3_rc():
        fig = _fig3_canvas(FIG3_FULL_W_IN, FIG3_C_H_IN)
        ax = fig.add_subplot()
        im = ax.imshow(ig.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.grid(False)  # the style grid falls on cell centres and splits every row in two
        for sp in ax.spines.values():
            sp.set_color(C_FRAME)
            sp.set_linewidth(0.6)
        ax.set_xticks(range(ig.shape[1]))
        ax.set_xticklabels([d.capitalize() for d in ig.columns], rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_yticks(range(ig.shape[0]))
        ax.set_yticklabels(ig.index)
        ax.set_ylabel("Top 20 TFs by IG importance")

        # Separator between the positive controls and the candidates, with the
        # two groups named above the columns
        sep = len(controls) - 0.5
        ax.axvline(x=sep, color="black", linewidth=1.0, linestyle="--")
        for x0, x1, name in ((-0.5, sep, "Positive controls"), (sep, ig.shape[1] - 0.5, "Candidates")):
            ax.text((x0 + x1) / 2, 1.015, name, ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035, aspect=22)
        step = 0.0005
        cbar.set_ticks(np.arange(-np.floor(vmax / step), np.floor(vmax / step) + 1) * step)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-12 else f"{v:.4f}".replace("-", "−")))
        cbar.ax.tick_params(length=2, width=0.6)
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor(C_FRAME)
        cbar.set_label("IG attribution\n(relative to mean-train baseline)")
        ends = [cbar.ax.text(0.5, 1.02, "Raises\npredicted AUC", ha="center", va="bottom",
                             transform=cbar.ax.transAxes, linespacing=1.1),
                cbar.ax.text(0.5, -0.02, "Lowers\npredicted AUC", ha="center", va="top",
                             transform=cbar.ax.transAxes, linespacing=1.1)]
        fig.tight_layout(pad=0.15)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        min_gap = 2 * fig.dpi / 72
        tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        for e in ends:
            eb = e.get_window_extent(renderer)
            if any(eb.x0 < tb.x1 + min_gap and tb.x0 < eb.x1 and eb.y0 < tb.y1 and tb.y0 < eb.y1
                   for tb in tick_boxes):
                raise ValueError("A colorbar end label runs into a drug label")
        _fig3_save(fig, "fig3c")


# ============================================================
# Main
# ============================================================

def main(letters_json=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SFIG_DIR.mkdir(parents=True, exist_ok=True)
    _check_arial()

    logger.info("=" * 60)
    logger.info("PRECISION: Generating ALL paper figures (Okabe-Ito palette)")
    logger.info("=" * 60)

    # Main figures (the panels of Figures 1, 3 and 4 stay in memory until
    # compose_main_figures)
    fig1_graph_schema()
    fig1b_pipeline()
    fig2_model_comparison()
    fig3_tf_importance()
    fig3b_ig_stability()
    fig6_drug_ranking_meta()
    fig_multicohort()
    fig_scanb_km()
    fig7_aurora_tf()
    fig_tnbc_direct()

    # Supplementary figures
    sfig04_tf_drug_heatmap()
    sfig03_umap_embeddings()
    sfig01_training_curves()
    sfig02_perdrug_distribution()
    sfig14_crossval()
    sfig05_qqplot()
    sfig07_forest_plot()
    sfig11_penalizer()
    sfig13_basal_her2()
    sfig10_aurora_drugs()
    sfig09_moa_enrichment()
    sfig06_scanb_metabric_scatter()
    sfig08_target_enrichment()
    sfig12_graph_ablation()
    fig3c_ig_heatmap()

    # One PNG per main figure for the figures whose panels come from
    # separate functions
    compose_main_figures()

    if letters_json:
        Path(letters_json).write_text(json.dumps(LETTERS, indent=1, sort_keys=True), encoding="utf-8")
        logger.info("Panel letters per PNG written to %s", letters_json)

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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--letters-json", metavar="PATH",
                        help="write the panel letters drawn in each PNG to this JSON file")
    main(parser.parse_args().letters_json)
