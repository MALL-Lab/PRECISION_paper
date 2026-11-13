#!/usr/bin/env python3
"""
PRECISION - Cross the cell lines used in the study against the ICLAC Register of
Misidentified Cell Lines.

Matching is by Cellosaurus accession: the DepMap RRID of each cell line against the
accession the register gives for each misidentified line. Matching by name is not used,
because splitting register entries such as "MCF-7/AdrR (NCI/ADR-RES)" or
"3AB-OS (MG-63 derivative)" flags the parent line (MCF7, MG-63), which is not itself
listed.

The register is pinned by a SHA-256 of its parsed cell content, not of the file. Cloud
sync clients can rewrite the customXml and docProps parts of Office files: the file
downloaded from iclac.org hashed to
9fb224380124193cc84ab191d725224a25838f49a7367c2672eda931304194a5 and changed to a
different file hash minutes after being copied, with every cell identical. A content hash
survives that rewrite and still changes if the register itself changes.

Inputs:
  data/reference/iclac_register_v14.xlsx   ICLAC register, version 14 (15 February 2026)
  data/DepMap/Model.csv                    RRID, names and lineage (DepMap 25Q3)
  data/precision_graph/hetero_graph_prism.pkl
  results/v4_xai/ig_full_attributions.csv  the 12 TNBC lines actually used for IG
  data/GDSC/processed/drug_response_sanger.csv

Outputs in results/v10_reporting/:
  iclac_matches.csv    one row per matched cell line, with register table and split
  iclac_summary.csv    counts read by generate_paper_results.py

Run from project root:
    python scripts/55_iclac_check.py
"""
from __future__ import annotations

import hashlib
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> package root
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

import openpyxl
import pandas as pd

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split

REGISTER = DATA_DIR / "reference/iclac_register_v14.xlsx"
REGISTER_CONTENT_SHA256 = "25c0051a8a86769d2e2f60a4bca8ffbf9939b65f09a11274af951e9512fe66a5"
REGISTER_VERSION = "14 (2026-02-15)"


def register_content_hash(path: Path) -> str:
    """SHA-256 of every cell of every worksheet, in order, as text."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    cells = [tuple("" if c is None else str(c) for c in row)
             for ws in wb.worksheets for row in ws.iter_rows(values_only=True)]
    return hashlib.sha256(repr(cells).encode()).hexdigest()
OUT = RESULTS_DIR / "v10_reporting"
SEED = 42


def load_register() -> pd.DataFrame:
    digest = register_content_hash(REGISTER)
    if digest != REGISTER_CONTENT_SHA256:
        raise ValueError(f"ICLAC register content hash mismatch: {digest}")
    wb = openpyxl.load_workbook(REGISTER, read_only=True, data_only=True)
    rows = []
    for ws in wb.worksheets:
        table = re.search(r"Table\s*([12])", ws.title)
        if not table:
            continue
        values = list(ws.iter_rows(values_only=True))
        header_i = next(i for i, r in enumerate(values[:100])
                        if r and any(isinstance(c, str) and c.strip().lower()
                                     .startswith("misidentified cell line") for c in r[:2]))
        header = [str(c).strip() if c is not None else f"c{j}"
                  for j, c in enumerate(values[header_i])]
        name_col = next(c for c in header if c.lower() == "misidentified cell line")
        ac_col = next(c for c in header if "misidentified" in c.lower() and "ac" in c.lower())
        id_col = next(c for c in header if "registration" in c.lower())
        for r in values[header_i + 1:]:
            if not r or all(c is None or str(c).strip() == "" for c in r):
                continue
            rec = dict(zip(header, r))
            for acc in re.findall(r"CVCL_[A-Z0-9]{4}", str(rec.get(ac_col))):
                rows.append({"accession": acc, "register_table": int(table.group(1)),
                             "registration_id": rec.get(id_col),
                             "register_name": rec.get(name_col)})
    reg = pd.DataFrame(rows).drop_duplicates()
    if reg.empty:
        raise ValueError("No accessions parsed from the ICLAC register")
    return reg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = load_register()
    model = pd.read_csv(DATA_DIR / "DepMap/Model.csv", index_col="ModelID")

    with open(DATA_DIR / "precision_graph/hetero_graph_prism.pkl", "rb") as f:
        graph_lines = list(pickle.load(f)["node_maps"]["cell_line"])
    ig_lines = (pd.read_csv(RESULTS_DIR / "v4_xai/ig_full_attributions.csv",
                            usecols=["cell_depmap_id"])["cell_depmap_id"].unique().tolist())
    gdsc_lines = (pd.read_csv(DATA_DIR / "GDSC/processed/drug_response_sanger.csv",
                              index_col=0, usecols=[0]).index.tolist())
    assert len(graph_lines) == 476, len(graph_lines)
    assert len(ig_lines) == 12, len(ig_lines)

    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    _, test_cells = cell_line_holdout_split(drug_response, 0.2, SEED)
    test_cells = set(test_cells)

    by_acc = reg.groupby("accession")
    matches = []
    for mid in graph_lines:
        rrid = model.at[mid, "RRID"] if mid in model.index else None
        if not isinstance(rrid, str) or rrid not in by_acc.groups:
            continue
        for _, e in by_acc.get_group(rrid).iterrows():
            matches.append({"ModelID": mid, "CellLineName": model.at[mid, "CellLineName"],
                            "RRID": rrid, "OncotreeLineage": model.at[mid, "OncotreeLineage"],
                            "register_table": e["register_table"],
                            "registration_id": e["registration_id"],
                            "register_name": e["register_name"],
                            "split_seed42": "test" if mid in test_cells else "train",
                            "in_ig": mid in ig_lines})
    m = pd.DataFrame(matches)
    m.to_csv(OUT / "iclac_matches.csv", index=False)

    matched = set(m["ModelID"]) if len(m) else set()
    per_line = m.drop_duplicates("ModelID") if len(m) else m
    gdsc_known = [x for x in gdsc_lines if x in model.index]
    gdsc_matched = {x for x in gdsc_known
                    if isinstance(model.at[x, "RRID"], str) and model.at[x, "RRID"] in by_acc.groups}

    summary = {
        "iclac_version": REGISTER_VERSION,
        "iclac_register_content_sha256": REGISTER_CONTENT_SHA256,
        "iclac_n_graph_lines": len(graph_lines),
        "iclac_n_graph_without_rrid": int(model.reindex(graph_lines)["RRID"].isna().sum()),
        "iclac_n_graph_matched": len(matched),
        "iclac_n_graph_table1": int((per_line["register_table"] == 1).sum()) if len(m) else 0,
        "iclac_n_graph_table2": int((per_line["register_table"] == 2).sum()) if len(m) else 0,
        "iclac_n_graph_train": int((per_line["split_seed42"] == "train").sum()) if len(m) else 0,
        "iclac_n_graph_test": int((per_line["split_seed42"] == "test").sum()) if len(m) else 0,
        "iclac_n_ig_lines": len(ig_lines),
        "iclac_n_ig_matched": len(matched & set(ig_lines)),
        "iclac_n_gdsc_lines": len(gdsc_known),
        "iclac_n_gdsc_matched": len(gdsc_matched),
        "iclac_n_gdsc_matched_not_in_graph": len(gdsc_matched - set(graph_lines)),
    }
    pd.DataFrame([summary]).to_csv(OUT / "iclac_summary.csv", index=False)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if len(m):
        print(per_line[["ModelID", "CellLineName", "register_table", "split_seed42"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
