#!/bin/bash
# PRECISION Paper 1: full paper build (results, figures and, when the LaTeX
# sources are present, the PDF).
#
# Usage, from the package root:
#   bash paper/build_paper.sh
# The package root is derived from this script's own location, so it also
# works when invoked from any other directory.
#
# Both steps go through pipeline.py so that the same configuration (config.py)
# is used as in the rest of the pipeline.
# Data and results locations can be overridden with PRECISION_DATA and
# PRECISION_RESULTS (see config.py). Set PYTHON to pick the interpreter.

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$PAPER_DIR")"
PYTHON="${PYTHON:-python}"

echo "=== PRECISION Paper Build Pipeline ==="
cd "$PROJECT_ROOT"

echo "1. Generating paper results (tables, paper_statistics.json, sanity checks)..."
"$PYTHON" pipeline.py --force --stage paper

echo "2. Generating all figures (matplotlib, Okabe-Ito)..."
"$PYTHON" pipeline.py --force --stage figures

LATEX_DIR="$PAPER_DIR/latex"
if [ -f "$LATEX_DIR/00_Article_Merge.tex" ]; then
    echo "3. Compiling LaTeX..."
    cd "$LATEX_DIR"
    pdflatex -interaction=nonstopmode 00_Article_Merge.tex > /dev/null
    bibtex 00_Article_Merge > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode 00_Article_Merge.tex > /dev/null
    pdflatex -interaction=nonstopmode 00_Article_Merge.tex > /dev/null

    echo "4. Copying final PDF..."
    cp 00_Article_Merge.pdf ../precision_manuscript_bioarxiv.pdf
    echo "=== Done: paper/precision_manuscript_bioarxiv.pdf ($(ls -lh ../precision_manuscript_bioarxiv.pdf | awk '{print $5}')) ==="
else
    echo "3. LaTeX sources not found ($LATEX_DIR): skipping PDF compilation."
    echo "=== Done: paper/results/, paper/figures/, paper/supplementary/ ==="
fi
