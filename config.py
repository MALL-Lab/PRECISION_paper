"""
Central path configuration for the PRECISION reproducibility package.

Every script in ``scripts/`` and ``paper/scripts/`` and every module in
``src/`` resolves its input and output locations through this module, so the
pipeline can be run from any working directory and the data can live outside
the repository.

Usage from a script (the ``sys.path`` insert makes ``config`` importable from
any working directory; use ``parents[2]`` for ``paper/scripts/``)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_DIR, RESULTS_DIR, PAPER_RESULTS, FIG_DIR

Variables
---------
ROOT
    Root directory of the package. Anchored to the location of this file,
    never to the current working directory.
DATA_DIR
    Directory with the raw and processed inputs (``PRISM/``, ``GDSC/``,
    ``DepMap/``, ``SCANB/``, ``METABRIC/``, ``TCGA/``, ``prior_knowledge/``,
    ``precision_processed/``, ``precision_graph/``). Defaults to
    ``ROOT/data``. Override with the environment variable ``PRECISION_DATA``.
RESULTS_DIR
    Directory where the pipeline writes its intermediate and final outputs
    (models, prediction matrices, per-drug tables), organised in versioned
    subfolders (``v5_final/``, ...). Defaults to ``ROOT/results``. Override
    with the environment variable ``PRECISION_RESULTS``.
PAPER_DIR
    ``ROOT/paper``: manuscript sources and everything derived from the
    results for the paper.
PAPER_RESULTS
    ``PAPER_DIR/results``: canonical small CSV tables and
    ``paper_statistics.json`` that the manuscript cites.
FIG_DIR
    ``PAPER_DIR/figures``: main figures.
SFIG_DIR
    ``PAPER_DIR/supplementary``: supplementary figures and tables.
DEPMAP_VERSION
    DepMap release the package is pinned to. Read from the first line of
    ``DATA_DIR/DepMap/ACTIVE_VERSION`` when that file exists, otherwise
    ``"24Q4"``. See ``data/README.md`` for how to download a release.

Functions
---------
depmap_file(kind)
    Path of a DepMap file of the given kind for the active release. See the
    function docstring for the accepted kinds and the release-dependent
    naming of the expression matrix.

Overriding the data and results locations
-----------------------------------------
Set the environment variables before running any script or the pipeline::

    # POSIX shells
    export PRECISION_DATA=/mnt/storage/precision_inputs
    export PRECISION_RESULTS=/mnt/storage/precision_outputs
    python pipeline.py

    # PowerShell
    $env:PRECISION_DATA = "D:\\precision_inputs"
    $env:PRECISION_RESULTS = "D:\\precision_outputs"
    python pipeline.py

Running ``python config.py`` prints every resolved path together with
whether it exists, which is the quickest way to check an installation.
"""

import os
import re
from pathlib import Path

# ── Package root and top-level directories ────────────────────────────────
ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("PRECISION_DATA", ROOT / "data")).expanduser()
RESULTS_DIR = Path(os.environ.get("PRECISION_RESULTS", ROOT / "results")).expanduser()

PAPER_DIR = ROOT / "paper"
PAPER_RESULTS = PAPER_DIR / "results"
FIG_DIR = PAPER_DIR / "figures"
SFIG_DIR = PAPER_DIR / "supplementary"

# ── DepMap release ────────────────────────────────────────────────────────
_DEPMAP_DIR = DATA_DIR / "DepMap"
_DEPMAP_DEFAULT_VERSION = "24Q4"


def _read_depmap_version() -> str:
    """Return the release name written in DepMap/ACTIVE_VERSION, or the default."""
    marker = _DEPMAP_DIR / "ACTIVE_VERSION"
    if marker.is_file():
        try:
            first_line = marker.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            first_line = []
        if first_line and first_line[0].strip():
            return first_line[0].strip()
    return _DEPMAP_DEFAULT_VERSION


DEPMAP_VERSION = _read_depmap_version()

# The expression matrix was renamed by DepMap between 24Q4 and 25Q3. Every
# other file kept its name across releases.
_EXPRESSION_NAME_OLD = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"  # up to 24Q4
_EXPRESSION_NAME_NEW = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"  # 25Q3 onwards
_EXPRESSION_RENAME_RELEASE = (25, 3)

_DEPMAP_FIXED_NAMES = {
    "model": "Model.csv",
    "crispr_effect": "CRISPRGeneEffect.csv",
}

DEPMAP_KINDS = ("expression",) + tuple(_DEPMAP_FIXED_NAMES)


def _parse_release(version: str):
    """Turn a release string such as '24Q4' into (24, 4). None if unparseable."""
    match = re.search(r"(\d{2})Q([1-4])", version.upper())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _expression_name_for_release(version: str) -> str:
    """Expression file name that DepMap used for the given release."""
    release = _parse_release(version)
    # An unrecognised release string is assumed to be more recent than the
    # rename, so the current naming applies.
    if release is None or release >= _EXPRESSION_RENAME_RELEASE:
        return _EXPRESSION_NAME_NEW
    return _EXPRESSION_NAME_OLD


def depmap_file(kind: str) -> Path:
    """Return the path of a DepMap file for the active release.

    Args:
        kind: one of ``"expression"`` (log2(TPM+1) protein-coding
            expression matrix), ``"model"`` (``Model.csv`` cell line
            metadata) or ``"crispr_effect"`` (``CRISPRGeneEffect.csv``).
            These are the only DepMap files the pipeline reads.

    Returns:
        ``DATA_DIR/DepMap/<file name>``. For ``"expression"`` the file name
        depends on the release: ``OmicsExpressionProteinCodingGenesTPMLogp1.csv``
        up to 24Q4 and ``OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv``
        from 25Q3. Both names are looked up on disk and the one that exists
        is returned, whatever ``DEPMAP_VERSION`` says. When neither exists the
        name expected for ``DEPMAP_VERSION`` is returned, so the resulting
        "file not found" error names the file that should be downloaded.

    Raises:
        ValueError: if ``kind`` is not one of the accepted kinds.
    """
    if kind == "expression":
        expected = _DEPMAP_DIR / _expression_name_for_release(DEPMAP_VERSION)
        for name in (_EXPRESSION_NAME_OLD, _EXPRESSION_NAME_NEW):
            candidate = _DEPMAP_DIR / name
            if candidate.is_file():
                return candidate
        return expected
    try:
        return _DEPMAP_DIR / _DEPMAP_FIXED_NAMES[kind]
    except KeyError:
        raise ValueError(
            f"Unknown DepMap file kind {kind!r}. "
            f"Accepted kinds: {', '.join(DEPMAP_KINDS)}"
        ) from None


# ── Diagnostics ───────────────────────────────────────────────────────────
def _describe(path: Path) -> str:
    if path.is_dir():
        return "exists (dir)"
    if path.is_file():
        size_mb = path.stat().st_size / 1e6
        return f"exists (file, {size_mb:.1f} MB)"
    return "MISSING"


def main() -> None:
    """Print every resolved path and whether it exists."""
    print(f"PRECISION config ({Path(__file__).resolve()})")
    print(f"  cwd:               {Path.cwd()}")
    print(f"  PRECISION_DATA:    {os.environ.get('PRECISION_DATA', '<unset>')}")
    print(f"  PRECISION_RESULTS: {os.environ.get('PRECISION_RESULTS', '<unset>')}")
    print()
    print("Directories")
    for name in ("ROOT", "DATA_DIR", "RESULTS_DIR", "PAPER_DIR",
                 "PAPER_RESULTS", "FIG_DIR", "SFIG_DIR"):
        path = globals()[name]
        print(f"  {name:<14} {path}  [{_describe(path)}]")
    print()
    marker = _DEPMAP_DIR / "ACTIVE_VERSION"
    source = "ACTIVE_VERSION file" if marker.is_file() else "default"
    print(f"DepMap release: {DEPMAP_VERSION} ({source})")
    for kind in DEPMAP_KINDS:
        path = depmap_file(kind)
        print(f"  {kind:<18} {path.name:<55} [{_describe(path)}]")


if __name__ == "__main__":
    main()
