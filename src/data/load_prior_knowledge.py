"""
Load biological prior knowledge networks for graph construction.

- CollecTRI: TF-target regulatory interactions (via decoupler/OmniPath)
- STRING: Protein-protein interactions
- Drug-target: from PRISM/GDSC annotations
"""

import pandas as pd
import numpy as np
import logging

from config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "prior_knowledge"


def load_collectri(organism: str = "human",
                   cache: bool = True) -> pd.DataFrame:
    """Load CollecTRI TF-target regulatory network.

    Args:
        organism: 'human' or 'mouse'.
        cache: If True, cache to disk after first download.

    Returns:
        DataFrame with columns: source (TF), target (gene), weight (+1/-1).
    """
    cache_path = CACHE_DIR / "collectri_edges.csv"

    if cache and cache_path.exists():
        logger.info("Loading cached CollecTRI: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Downloading CollecTRI via decoupler (organism=%s)", organism)
    import decoupler as dc
    net = dc.op.collectri(organism=organism)

    net = net.rename(columns={"mor": "weight"})
    net = net[["source", "target", "weight"]].drop_duplicates()

    n_tfs = net["source"].nunique()
    n_targets = net["target"].nunique()
    logger.info("CollecTRI: %d TFs, %d targets, %d edges", n_tfs, n_targets, len(net))

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        net.to_csv(cache_path, index=False)
        logger.info("Cached to %s", cache_path)

    return net


def compute_tf_activities(expression: pd.DataFrame,
                          collectri: pd.DataFrame | None = None,
                          min_targets: int = 5) -> pd.DataFrame:
    """Compute TF activity scores for ALL TFs using ULM (decoupleR).

    Unlike the earlier lab pipeline, which filtered to 5 TFs, we compute all
    ~700 TFs to use as features in the heterogeneous graph.

    Args:
        expression: Gene expression matrix (cell lines x genes).
        collectri: CollecTRI network. If None, loads automatically.
        min_targets: Minimum number of targets for a TF to be included.

    Returns:
        DataFrame with cell lines as index, TF names as columns, ULM scores as values.
    """
    import decoupler as dc

    if collectri is None:
        collectri = load_collectri()

    logger.info(
        "Computing TF activities via ULM for %d cell lines x %d genes",
        *expression.shape,
    )

    tf_acts, _ = dc.mt.ulm(
        data=expression,
        net=collectri,
        tmin=min_targets,
    )

    logger.info("TF activities: %d cell lines x %d TFs", *tf_acts.shape)
    return tf_acts


def load_omnipath_ppi(cache: bool = True) -> pd.DataFrame:
    """Load protein-protein interactions from OmniPath (curated, includes STRING).

    Faster and more reliable than direct STRING download. Returns curated
    interactions from multiple sources including STRING, BioGRID, IntAct, etc.

    Returns:
        DataFrame with columns: protein1, protein2, combined_score (set to 1.0).
    """
    cache_path = CACHE_DIR / "omnipath_ppi.csv"

    if cache and cache_path.exists():
        logger.info("Loading cached OmniPath PPI: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Downloading PPI from OmniPath...")
    from omnipath.interactions import OmniPath
    # license="commercial" does NOT filter this endpoint; see README (OmniPath mixed licensing)
    ppi = OmniPath.get(genesymbols=True, organisms="human")

    result = pd.DataFrame({
        "protein1": ppi["source_genesymbol"],
        "protein2": ppi["target_genesymbol"],
        "combined_score": 1000,  # OmniPath interactions are curated
    }).drop_duplicates()

    # Remove self-loops
    result = result[result["protein1"] != result["protein2"]]

    logger.info("OmniPath PPI: %d edges, %d unique genes",
                len(result), len(set(result["protein1"]) | set(result["protein2"])))

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        logger.info("Cached to %s", cache_path)

    return result


def load_string_ppi(confidence_threshold: int = 700,
                    cache: bool = True) -> pd.DataFrame:
    """Load STRING protein-protein interactions for human.

    Downloads from STRING database if not cached.

    Args:
        confidence_threshold: Minimum combined score (0-1000).
        cache: Cache to disk.

    Returns:
        DataFrame with columns: protein1, protein2, combined_score.
        Protein names are gene symbols (mapped from ENSP IDs).
    """
    cache_path = CACHE_DIR / f"string_ppi_{confidence_threshold}.csv"

    if cache and cache_path.exists():
        logger.info("Loading cached STRING PPI: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Downloading STRING PPI (confidence >= %d)", confidence_threshold)

    import requests, gzip, io

    # STRING v12.0 human protein links
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    info_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"

    # Download protein info for ENSP -> gene symbol mapping
    logger.info("Downloading STRING protein info for ID mapping...")
    r = requests.get(info_url, stream=True)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        info = pd.read_csv(f, sep="\t")
    ensp_to_symbol = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    logger.info("Mapped %d STRING protein IDs to symbols", len(ensp_to_symbol))

    # Download interactions
    logger.info("Downloading STRING interactions (~200 MB compressed)...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        links = pd.read_csv(f, sep=" ")

    # Filter by confidence
    links = links[links["combined_score"] >= confidence_threshold]
    logger.info("Edges with confidence >= %d: %d", confidence_threshold, len(links))

    # Map to gene symbols
    links["protein1"] = links["protein1"].map(ensp_to_symbol)
    links["protein2"] = links["protein2"].map(ensp_to_symbol)
    links = links.dropna(subset=["protein1", "protein2"])

    # Remove self-loops and duplicates (keep higher score)
    links = links[links["protein1"] != links["protein2"]]
    links = links.sort_values("combined_score", ascending=False)
    links["edge_key"] = links.apply(
        lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1
    )
    links = links.drop_duplicates(subset="edge_key").drop(columns="edge_key")

    logger.info("STRING PPI final: %d edges, %d unique genes",
                len(links), len(set(links["protein1"]) | set(links["protein2"])))

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        links.to_csv(cache_path, index=False)
        logger.info("Cached to %s", cache_path)

    return links


def build_drug_target_edges(prism_drug_net: pd.DataFrame,
                            gdsc_drug_net: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build unified drug-target edge list from PRISM and GDSC annotations.

    Args:
        prism_drug_net: PRISM drug annotations (from load_prism_drug_annotations).
        gdsc_drug_net: GDSC drug annotations (optional, for additional coverage).

    Returns:
        DataFrame with columns: drug, target_gene, source.
    """
    edges = []

    # PRISM: extract target genes from treatment info if available
    try:
        treatment_info = pd.read_csv(
            DATA_DIR / "PRISM" / "raw" / "secondary-screen-replicate-treatment-info.csv"
        )
        if "target" in treatment_info.columns:
            targets = treatment_info[["name", "target"]].dropna().drop_duplicates()
            targets = targets.rename(columns={"name": "drug", "target": "target_gene"})
            # Explode comma-separated targets
            targets["target_gene"] = targets["target_gene"].str.split(",")
            targets = targets.explode("target_gene")
            targets["target_gene"] = targets["target_gene"].str.strip()
            targets["source"] = "PRISM"
            edges.append(targets)
            logger.info("PRISM drug-target edges: %d", len(targets))
    except Exception as e:
        logger.warning("Could not parse PRISM targets: %s", e)

    if gdsc_drug_net is not None and "target" in gdsc_drug_net.columns:
        targets = gdsc_drug_net[["name", "target"]].dropna().drop_duplicates()
        targets = targets.rename(columns={"name": "drug", "target": "target_gene"})
        targets["target_gene"] = targets["target_gene"].str.split(",")
        targets = targets.explode("target_gene")
        targets["target_gene"] = targets["target_gene"].str.strip()
        targets["source"] = "GDSC"
        edges.append(targets)
        logger.info("GDSC drug-target edges: %d", len(targets))

    if not edges:
        logger.warning("No drug-target edges found")
        return pd.DataFrame(columns=["drug", "target_gene", "source"])

    result = pd.concat(edges, ignore_index=True).drop_duplicates(
        subset=["drug", "target_gene"]
    )
    logger.info("Combined drug-target edges: %d (drugs: %d, genes: %d)",
                len(result), result["drug"].nunique(), result["target_gene"].nunique())
    return result
