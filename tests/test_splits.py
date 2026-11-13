"""Response edges must reach every drug node, whatever the case of the graph node names.

The graph pickle keeps the original PRISM spelling of 501 drug names (for example
'1-naphthyl-PP1'), while normalize_drug_columns lowercases the response matrix.
split_response_edges used to drop those drugs silently, so the GNN trained on 947
of the 1,448 drugs.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.drug_names import build_drug_index
from src.data.splits import split_response_edges

NODE_MAPS = {
    "cell_line": {"ACH-1": 0, "ACH-2": 1, "ACH-3": 2},
    "drug": {"erlotinib": 0, "1-naphthyl-PP1": 1, "AZD-8055": 2},
}


def _response(columns):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((3, len(columns))), index=["ACH-1", "ACH-2", "ACH-3"],
                        columns=columns)


def test_mixed_case_drug_nodes_receive_edges():
    resp = _response(["erlotinib", "1-naphthyl-pp1", "azd-8055"])
    edges = split_response_edges(resp, NODE_MAPS, ["ACH-1", "ACH-2"], ["ACH-3"])
    assert sorted(edges["train_drug"].unique().tolist()) == [0, 1, 2]
    assert sorted(edges["test_drug"].unique().tolist()) == [0, 1, 2]
    assert len(edges["train_auc"]) == 6
    assert len(edges["test_auc"]) == 3


def test_edge_values_follow_the_matrix():
    resp = _response(["erlotinib", "1-naphthyl-pp1", "azd-8055"])
    edges = split_response_edges(resp, NODE_MAPS, ["ACH-1"], ["ACH-3"])
    got = dict(zip(edges["train_drug"].tolist(), edges["train_auc"].tolist()))
    assert got[1] == pytest.approx(resp.loc["ACH-1", "1-naphthyl-pp1"])
    assert got[2] == pytest.approx(resp.loc["ACH-1", "azd-8055"])


def test_missing_values_are_skipped():
    resp = _response(["erlotinib", "1-naphthyl-pp1", "azd-8055"])
    resp.loc["ACH-1", "azd-8055"] = np.nan
    edges = split_response_edges(resp, NODE_MAPS, ["ACH-1"], ["ACH-3"])
    assert sorted(edges["train_drug"].tolist()) == [0, 1]


def test_unknown_drug_raises_instead_of_being_dropped():
    resp = _response(["erlotinib", "not-a-graph-drug"])
    with pytest.raises(ValueError, match="not-a-graph-drug"):
        split_response_edges(resp, NODE_MAPS, ["ACH-1"], ["ACH-3"])


def test_build_drug_index_is_case_insensitive():
    idx = build_drug_index(NODE_MAPS["drug"])
    assert idx["1-naphthyl-pp1"] == 1
    assert idx["azd-8055"] == 2


def test_build_drug_index_rejects_collisions():
    with pytest.raises(ValueError, match="Erlotinib"):
        build_drug_index({"erlotinib": 0, "Erlotinib": 1})
