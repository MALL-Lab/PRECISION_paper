"""
Canonical training and evaluation of the GNNs of Paper 1.

Every GNN of the paper is trained here, once, by scripts/57_train_gnns.py, and
downstream scripts load the saved checkpoints instead of retraining:

- Table 1 and Supplementary Figures 1 and 2: SAGE and HGT on the seed-42
  cell-line split with five initialization seeds.
- Integrated Gradients (Figure 3, Table 4, Supplementary Note 1): the
  canonical SAGE model (split 42, init 42) and the SAGE models of seeds 123
  and 456, each on its own split.
- Graph ablation (Supplementary Figure 12): SAGE on the seed-42 split with
  the same five seeds and the same protocol as Table 1, plus a control that
  keeps every relation and rewires the drug-target edges with
  degree-preserving swaps.

Protocol: 800 epochs, full batch, Adam (lr 1.3e-3, weight decay 1e-4), cosine
annealing to 1e-5 and Huber loss, as in the original script 11. The weights of
the final epoch are kept. The test cell lines are never used to choose a
checkpoint. Response edges never take part in message passing.
"""

import hashlib
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from src.data.drug_names import normalize_drug_columns
from src.data.load_drug_response import load_prism_response
from src.data.splits import cell_line_holdout_split, set_all_seeds, split_response_edges
from src.graph.hetero_data import load_hetero_data
from src.models.hetero_gnn import HeteroGNNDrugResponse, HGTDrugResponse
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

RUNS_DIR = RESULTS_DIR / "v12_gnn"
RESPONSE = ("cell_line", "responds_to", "drug")

ARCHS = {
    "sage": {"hidden_dim": 128, "n_layers": 3, "dropout": 0.33, "conv_type": "sage"},
    "hgt": {"hidden_dim": 128, "n_layers": 2, "n_heads": 4, "dropout": 0.20},
}
OPTIM = {"epochs": 800, "lr": 1.3e-3, "weight_decay": 1e-4, "eta_min": 1e-5}
TEST_FRAC = 0.2
# Drugs entering the evaluation, as for the Ridge and random forest baselines
# of script 11: at least 30 training and 5 test cell lines with an AUC.
MIN_TRAIN_CELLS, MIN_TEST_CELLS = 30, 5

# Prior-knowledge relations removed in the graph ablation. Matching is on the
# exact relation name, and the reverse relation goes with it.
ABLATIONS = {
    "full": ("Full graph", ()),
    "no_ppi": ("No PPI (gene-gene)", ("interacts",)),
    "no_collectri": ("No CollecTRI (TF-gene)", ("regulates",)),
    "no_drug_target": ("No drug-target", ("targets",)),
    "no_ppi_no_drug_target": ("No PPI + No drug-target", ("interacts", "targets")),
    "response_only": ("Only response edges (no prior knowledge)", ("interacts", "regulates", "targets")),
}
# Control of the drug-target ablation. Removing the relation also removes the
# GraphSAGE layers of every drug node, so this configuration keeps the relation
# and rewires its edges instead: each drug keeps its number of targets and each
# gene the number of drugs that target it, but the targets are no longer the
# annotated ones. Each initialization seed gets its own rewiring.
SHUFFLED = {
    "shuffled_drug_target": ("Drug-target rewired (degrees kept)", "targets"),
}
SHUFFLE_ATTEMPTS_PER_EDGE = 100


def ablation_config(tag: str) -> tuple[tuple, str | None]:
    """(excluded relations, shuffled relation) of a configuration tag."""
    if tag in SHUFFLED:
        return (), SHUFFLED[tag][1]
    return ABLATIONS[tag][1], None


def config_label(tag: str) -> str:
    return (SHUFFLED[tag] if tag in SHUFFLED else ABLATIONS[tag])[0]


@dataclass
class Graph:
    data: object
    node_maps: dict
    x_dict: dict
    nfd: dict
    n_drugs: int
    inv_drug: dict


@dataclass
class Split:
    seed: int
    train_cells: list
    test_cells: list
    edges: dict
    eval_mask: torch.Tensor   # test edges of the drugs in the evaluation universe
    n_eval_drugs: int


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_graph(dev: torch.device) -> Graph:
    data, node_maps = load_hetero_data()
    data = data.to(dev)
    x_dict = {nt: data[nt].x for nt in data.node_types}
    nfd = {nt: data[nt].x.shape[1] for nt in data.node_types}
    inv_drug = {v: k for k, v in node_maps["drug"].items()}
    return Graph(data, node_maps, x_dict, nfd, len(node_maps["drug"]), inv_drug)


def degree_preserving_shuffle(edge_index: torch.Tensor, seed: int) -> torch.Tensor:
    """Rewire a bipartite relation with edge swaps (Maslov and Sneppen 2002).

    A swap exchanges the destinations of two edges, so (a, b) and (c, d) become
    (a, d) and (c, b), and it is rejected when either new edge already exists.
    Every source keeps its out-degree, every destination its in-degree, and the
    relation keeps its number of distinct edges. The sources stay in place.
    """
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    present = set(zip(src, dst))
    if len(present) != len(src):
        raise ValueError("The relation to shuffle has duplicated edges")
    rng = np.random.default_rng(seed)
    for i, j in rng.integers(0, len(src), size=(SHUFFLE_ATTEMPTS_PER_EDGE * len(src), 2)).tolist():
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == c or b == d or (a, d) in present or (c, b) in present:
            continue
        present.difference_update(((a, b), (c, d)))
        present.update(((a, d), (c, b)))
        dst[i], dst[j] = d, b
    return torch.stack([edge_index[0], torch.tensor(dst, dtype=edge_index.dtype, device=edge_index.device)])


def build_mp_edges(data, exclude: tuple = (), shuffle: str | None = None,
                   shuffle_seed: int | None = None) -> dict:
    """Message-passing edges: every relation except responses and the excluded ones, plus reverses.

    With shuffle, the edges of that relation are rewired by
    degree_preserving_shuffle with shuffle_seed, and its reverse follows them.
    """
    if shuffle is not None and sum(et[1] == shuffle for et in data.edge_types) != 1:
        raise ValueError(f"The graph does not have exactly one {shuffle!r} relation")
    mp = {}
    for et in data.edge_types:
        if et == RESPONSE or et[1] in exclude:
            continue
        src, rel, dst = et
        ei = data[et].edge_index
        if rel == shuffle:
            ei = degree_preserving_shuffle(ei, shuffle_seed)
        mp[et] = ei
        mp[(dst, f"rev_{rel}", src)] = ei.flip(0)
    return mp


def shuffle_record(data, mp: dict, shuffle: str) -> dict:
    """Hash of the rewired edges and how many of the original edges they keep."""
    et = next(et for et in data.edge_types if et[1] == shuffle)
    new = mp[et].cpu()
    kept = set(zip(*data[et].edge_index.cpu().tolist())) & set(zip(*new.tolist()))
    return {"shuffled_edges_sha256": hashlib.sha256(np.ascontiguousarray(new.numpy(), dtype=np.int64).tobytes()).hexdigest(),
            "shuffled_edges_total": int(new.shape[1]), "shuffled_edges_kept": len(kept)}


def prepare_split(split_seed: int, graph: Graph, dev: torch.device) -> Split:
    drug_response = normalize_drug_columns(load_prism_response(use_processed=True))
    train_cells, test_cells = cell_line_holdout_split(drug_response, TEST_FRAC, split_seed)
    edges = split_response_edges(drug_response, graph.node_maps, train_cells, test_cells, dev)
    cells = drug_response.index.isin(list(graph.node_maps["cell_line"]))
    n_train = drug_response[cells & drug_response.index.isin(train_cells)].notna().sum()
    n_test = drug_response[cells & drug_response.index.isin(test_cells)].notna().sum()
    universe = {graph.node_maps["drug"][d] for d in drug_response.columns
                if n_train[d] >= MIN_TRAIN_CELLS and n_test[d] >= MIN_TEST_CELLS}
    idx = torch.tensor(sorted(universe), device=dev)
    eval_mask = torch.isin(edges["test_drug"], idx)
    logger.info("Split %d: %d train / %d test edges, %d drugs in the evaluation universe",
                split_seed, len(edges["train_auc"]), len(edges["test_auc"]), len(universe))
    return Split(split_seed, train_cells, test_cells, edges, eval_mask, len(universe))


def build_model(arch: str, graph: Graph, mp: dict):
    kw = dict(ARCHS[arch], node_feature_dims=graph.nfd, edge_types=list(mp) or None,
              n_drugs=graph.n_drugs)
    if arch == "sage":
        return HeteroGNNDrugResponse(**kw)
    if arch == "hgt":
        return HGTDrugResponse(node_types=list(graph.nfd), **kw)
    raise ValueError(f"Unknown architecture {arch!r}")


def _test_metrics(model, graph: Graph, mp: dict, split: Split) -> tuple[float, float, np.ndarray]:
    e = split.edges
    model.eval()
    with torch.no_grad():
        pred = model(graph.x_dict, mp, e["test_cell"], e["test_drug"])
        loss = F.huber_loss(pred[split.eval_mask], e["test_auc"][split.eval_mask]).item()
    yt = e["test_auc"][split.eval_mask].cpu().numpy()
    yp = pred[split.eval_mask].cpu().numpy()
    return loss, float(pearsonr(yt, yp)[0]), yp


def train_gnn(arch: str, graph: Graph, split: Split, init_seed: int, exclude: tuple = (),
              history_every: int = 50, shuffle: str | None = None, shuffle_seed: int | None = None):
    """Train for OPTIM['epochs'] epochs and return the final-epoch model and its history.

    The test metrics recorded in the history never feed back into training.
    """
    mp = build_mp_edges(graph.data, exclude, shuffle, shuffle_seed)
    set_all_seeds(init_seed)
    model = build_model(arch, graph, mp).to(split.edges["train_auc"].device)
    opt = torch.optim.Adam(model.parameters(), lr=OPTIM["lr"], weight_decay=OPTIM["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OPTIM["epochs"], eta_min=OPTIM["eta_min"])
    e = split.edges
    history = []
    for ep in range(1, OPTIM["epochs"] + 1):
        model.train()
        opt.zero_grad()
        loss = F.huber_loss(model(graph.x_dict, mp, e["train_cell"], e["train_drug"]), e["train_auc"])
        loss.backward()
        opt.step()
        sch.step()
        if ep % history_every == 0 or ep == OPTIM["epochs"]:
            test_loss, test_r, _ = _test_metrics(model, graph, mp, split)
            history.append({"epoch": ep, "train_loss": loss.item(), "test_loss": test_loss,
                            "test_pearson": test_r, "lr": sch.get_last_lr()[0]})
            if ep % 100 == 0:
                logger.info("  %s split %d init %d %s | epoch %d | train %.4f | test %.4f | r %.4f",
                            arch, split.seed, init_seed, exclude or (f"{shuffle} rewired" if shuffle else "full"), ep, loss.item(),
                            test_loss, test_r)
    return model, mp, pd.DataFrame(history)


def evaluate(model, graph: Graph, mp: dict, split: Split) -> tuple[dict, pd.DataFrame]:
    """Global and per-drug test metrics over the evaluation universe, drug names in lowercase."""
    _, global_r, yp = _test_metrics(model, graph, mp, split)
    e = split.edges
    yt = e["test_auc"][split.eval_mask].cpu().numpy()
    drugs = np.array([graph.inv_drug[int(d)] for d in e["test_drug"][split.eval_mask].cpu()])
    rows = []
    for drug in sorted(set(drugs)):
        m = drugs == drug
        if m.sum() < MIN_TEST_CELLS:
            continue
        rows.append({"drug": drug, "n_samples": int(m.sum()),
                     "pearson_r": pearsonr(yt[m], yp[m])[0], "spearman_r": spearmanr(yt[m], yp[m])[0],
                     "rmse": float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))})
    per_drug = pd.DataFrame(rows)
    valid = per_drug.dropna(subset=["pearson_r"])
    metrics = {"global_pearson": global_r, "n_drugs": int(len(valid)),
               "median_pearson": float(valid["pearson_r"].median()),
               "mean_pearson": float(valid["pearson_r"].mean()),
               "pct_gt_0.3": float((valid["pearson_r"] > 0.3).mean() * 100)}
    return metrics, per_drug


def run_name(arch: str, split_seed: int, init_seed: int, tag: str = "full") -> str:
    return f"{arch}_split{split_seed}_init{init_seed}_{tag}"


def _git_state() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except OSError:
        return {"git_sha": None, "git_dirty": None}


def save_run(model, arch: str, split: Split, init_seed: int, tag: str, exclude: tuple,
             history: pd.DataFrame, metrics: dict, per_drug: pd.DataFrame, seconds: float,
             extra: dict | None = None) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "per_drug").mkdir(exist_ok=True)
    (RUNS_DIR / "history").mkdir(exist_ok=True)
    name = run_name(arch, split.seed, init_seed, tag)
    ckpt = RUNS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), ckpt)
    history.to_csv(RUNS_DIR / "history" / f"{name}.csv", index=False)
    per_drug.to_csv(RUNS_DIR / "per_drug" / f"{name}.csv", index=False)
    e = split.edges
    import torch_geometric
    sidecar = {
        "run": name, "arch": arch, "split_seed": split.seed, "init_seed": init_seed, "tag": tag,
        "excluded_relations": list(exclude), "hyperparameters": ARCHS[arch], "optimizer": OPTIM,
        "checkpoint_selection": "final epoch",
        "n_train_edges": int(len(e["train_auc"])), "n_test_edges": int(len(e["test_auc"])),
        "n_train_drugs": int(e["train_drug"].unique().numel()),
        "n_eval_test_edges": int(split.eval_mask.sum()), "n_eval_drugs": split.n_eval_drugs,
        "metrics": metrics, "seconds": round(seconds, 1),
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        **_git_state(), **(extra or {}),
    }
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def load_run(arch: str, split_seed: int, init_seed: int, graph: Graph, tag: str = "full"):
    """Rebuild a trained model from its checkpoint. Returns (model, mp_edges, sidecar)."""
    name = run_name(arch, split_seed, init_seed, tag)
    sidecar = json.loads((RUNS_DIR / f"{name}.json").read_text())
    ckpt = RUNS_DIR / f"{name}.pt"
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != sidecar["sha256"]:
        raise ValueError(f"{ckpt} does not match the hash recorded when it was trained")
    shuffle = sidecar.get("shuffled_relation")
    mp = build_mp_edges(graph.data, tuple(sidecar["excluded_relations"]), shuffle, sidecar.get("shuffle_seed"))
    if shuffle is not None and (shuffle_record(graph.data, mp, shuffle)["shuffled_edges_sha256"]
                                != sidecar["shuffled_edges_sha256"]):
        raise ValueError(f"The rewired {shuffle!r} edges of {name} differ from those it was trained with")
    model = build_model(arch, graph, mp).to(graph.data[RESPONSE].edge_index.device)
    model.load_state_dict(torch.load(ckpt, map_location=graph.data[RESPONSE].edge_index.device))
    model.eval()
    return model, mp, sidecar


def train_and_save(arch: str, graph: Graph, split: Split, init_seed: int, tag: str = "full",
                   history_every: int = 50) -> dict:
    exclude, shuffle = ablation_config(tag)
    shuffle_seed = init_seed if shuffle else None
    t0 = time.time()
    model, mp, history = train_gnn(arch, graph, split, init_seed, exclude, history_every, shuffle, shuffle_seed)
    metrics, per_drug = evaluate(model, graph, mp, split)
    extra = ({"shuffled_relation": shuffle, "shuffle_seed": shuffle_seed,
              "shuffle_attempts_per_edge": SHUFFLE_ATTEMPTS_PER_EDGE, **shuffle_record(graph.data, mp, shuffle)}
             if shuffle else None)
    sidecar = save_run(model, arch, split, init_seed, tag, exclude, history, metrics, per_drug,
                       time.time() - t0, extra)
    logger.info("%s: global r %.4f | median per-drug %.4f | %d drugs | %.0f s", sidecar["run"],
                metrics["global_pearson"], metrics["median_pearson"], metrics["n_drugs"],
                sidecar["seconds"])
    return sidecar
