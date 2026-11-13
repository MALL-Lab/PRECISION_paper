"""Smoke test of the canonical GNN trainer on the real graph, with a few epochs.

Skipped when the graph pickle is not available (for example in the public
package without Level 3 data).
"""

from pathlib import Path

import pytest
from config import DATA_DIR

GRAPH = DATA_DIR / "precision_graph/hetero_graph_prism.pkl"
pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason="graph pickle not available")


@pytest.fixture(scope="module")
def bundle():
    from src.models import training as T
    dev = T.device()
    graph = T.load_graph(dev)
    return T, graph, T.prepare_split(42, graph, dev)


def test_every_drug_is_trained(bundle):
    T, graph, split = bundle
    assert graph.n_drugs == 1448
    assert split.edges["train_drug"].unique().numel() == 1448
    assert len(split.edges["train_auc"]) + len(split.edges["test_auc"]) == 614325


def test_ablation_removes_exact_relations(bundle):
    T, graph, _ = bundle
    full = {et[1] for et in T.build_mp_edges(graph.data)}
    no_ppi = {et[1] for et in T.build_mp_edges(graph.data, ("interacts",))}
    assert full == {"regulates", "rev_regulates", "interacts", "rev_interacts", "targets", "rev_targets"}
    assert no_ppi == full - {"interacts", "rev_interacts"}
    assert T.build_mp_edges(graph.data, ("interacts", "regulates", "targets")) == {}


def test_shuffle_keeps_degrees_and_rewires(bundle):
    from collections import Counter
    T, graph, _ = bundle
    et = ("drug", "targets", "gene")
    orig = graph.data[et].edge_index
    mp = T.build_mp_edges(graph.data, shuffle="targets", shuffle_seed=42)
    new = mp[et]
    assert set(mp) == set(T.build_mp_edges(graph.data))
    assert Counter(new[0].tolist()) == Counter(orig[0].tolist())
    assert Counter(new[1].tolist()) == Counter(orig[1].tolist())
    assert len(set(zip(*new.tolist()))) == new.shape[1]
    assert T.shuffle_record(graph.data, mp, "targets")["shuffled_edges_kept"] < 0.1 * new.shape[1]
    assert torch_equal(mp[("gene", "rev_targets", "drug")], new.flip(0))
    again = T.build_mp_edges(graph.data, shuffle="targets", shuffle_seed=42)[et]
    other = T.build_mp_edges(graph.data, shuffle="targets", shuffle_seed=123)[et]
    assert torch_equal(new, again) and not torch_equal(new, other)
    for rel in ("interacts", "regulates"):
        key = next(k for k in mp if k[1] == rel)
        assert torch_equal(mp[key], graph.data[key].edge_index)


def torch_equal(a, b) -> bool:
    import torch
    return a.shape == b.shape and bool(torch.equal(a, b))


def test_train_save_load_roundtrip(bundle, tmp_path, monkeypatch):
    T, graph, split = bundle
    monkeypatch.setattr(T, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(T, "OPTIM", dict(T.OPTIM, epochs=3))
    for arch, tag in (("sage", "full"), ("hgt", "full"), ("sage", "response_only"),
                      ("sage", "shuffled_drug_target")):
        side = T.train_and_save(arch, graph, split, 7, tag, history_every=1)
        assert side["checkpoint_selection"] == "final epoch"
        assert side["n_train_drugs"] == 1448
        model, mp, loaded = T.load_run(arch, 42, 7, graph, tag)
        metrics, _ = T.evaluate(model, graph, mp, split)
        assert metrics["global_pearson"] == pytest.approx(side["metrics"]["global_pearson"], abs=1e-6)
    hist = (tmp_path / "history" / "sage_split42_init7_full.csv").read_text().splitlines()
    assert len(hist) == 4  # header plus epochs 1 to 3
