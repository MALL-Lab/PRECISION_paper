"""
Heterogeneous GNN for drug response prediction.

Architecture:
1. Per-node-type linear projections to shared hidden dim
2. Heterogeneous message passing (SAGEConv or GATConv per edge type)
3. Cell line and drug embeddings are concatenated
4. MLP head predicts AUC

The model predicts drug response (AUC) for (cell_line, drug) pairs
by propagating information through the biological knowledge graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, HGTConv, Linear
import logging

logger = logging.getLogger(__name__)


class HeteroGNNDrugResponse(nn.Module):
    """Heterogeneous GNN for drug response prediction.

    Args:
        node_feature_dims: Dict of node_type -> input feature dimension.
        hidden_dim: Hidden dimension for all node types after projection.
        n_layers: Number of message passing layers.
        conv_type: 'sage' or 'gat'.
        dropout: Dropout rate.
        edge_types: List of edge type tuples in the graph.
    """

    def __init__(self,
                 node_feature_dims: dict[str, int],
                 hidden_dim: int = 128,
                 n_layers: int = 2,
                 conv_type: str = "sage",
                 dropout: float = 0.2,
                 edge_types: list[tuple] | None = None,
                 n_drugs: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Per-node-type input projections
        self.input_projections = nn.ModuleDict()
        for ntype, in_dim in node_feature_dims.items():
            self.input_projections[ntype] = Linear(in_dim, hidden_dim)

        # Learned drug embeddings (added to projected MOA features)
        # This breaks MOA clustering: each drug gets a unique learned component
        self.drug_embedding = None
        if n_drugs > 0:
            self.drug_embedding = nn.Embedding(n_drugs, hidden_dim)

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            if edge_types is not None:
                for etype in edge_types:
                    if conv_type == "sage":
                        conv_dict[etype] = SAGEConv(hidden_dim, hidden_dim)
                    elif conv_type == "gat":
                        conv_dict[etype] = GATConv(
                            hidden_dim, hidden_dim // 4, heads=4, concat=True
                        )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # MLP prediction head: concat(cell_line_emb, drug_emb) → AUC
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x_dict, edge_index_dict):
        """Encode all node types through projection + message passing.

        Returns:
            Dict of node_type -> embeddings (n_nodes, hidden_dim).
        """
        # Project all node types to shared hidden dim
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_projections:
                h_dict[ntype] = self.input_projections[ntype](x).relu()
            else:
                h_dict[ntype] = x

        # Add learned per-drug embeddings to break MOA clustering
        if self.drug_embedding is not None and "drug" in h_dict:
            drug_ids = torch.arange(h_dict["drug"].shape[0], device=h_dict["drug"].device)
            h_dict["drug"] = h_dict["drug"] + self.drug_embedding(drug_ids)

        # Message passing: preserve embeddings for nodes not updated by convolution
        for conv in self.convs:
            h_update = conv(h_dict, edge_index_dict)
            # Merge: use updated embeddings where available, keep original otherwise
            for ntype in h_dict:
                if ntype in h_update:
                    h_dict[ntype] = F.relu(F.dropout(
                        h_update[ntype], p=self.dropout, training=self.training
                    ))
                # else: h_dict[ntype] stays as-is (projected but not message-passed)

        return h_dict

    def decode(self, h_dict, cell_line_idx, drug_idx):
        """Predict AUC for (cell_line, drug) pairs.

        Args:
            h_dict: Node embeddings from encode().
            cell_line_idx: Tensor of cell line indices.
            drug_idx: Tensor of drug indices.

        Returns:
            Predicted AUC values (n_pairs,).
        """
        cell_emb = h_dict["cell_line"][cell_line_idx]
        drug_emb = h_dict["drug"][drug_idx]
        pair_emb = torch.cat([cell_emb, drug_emb], dim=-1)
        return self.predictor(pair_emb).squeeze(-1)

    def forward(self, x_dict, edge_index_dict, cell_line_idx, drug_idx):
        """Full forward pass: encode graph + predict AUC."""
        h_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(h_dict, cell_line_idx, drug_idx)


class HGTDrugResponse(nn.Module):
    """Heterogeneous Graph Transformer for drug response prediction.

    Uses HGTConv which applies multi-head attention with type-specific
    parameters for each node type and edge type. This is the Level 2
    model in the PRECISION execution plan.

    Args:
        node_feature_dims: Dict of node_type -> input feature dimension.
        hidden_dim: Hidden dimension.
        n_layers: Number of HGT layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        node_types: List of node type names.
        edge_types: List of edge type tuples (src, rel, dst).
    """

    def __init__(self,
                 node_feature_dims: dict[str, int],
                 hidden_dim: int = 128,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.2,
                 node_types: list[str] | None = None,
                 edge_types: list[tuple] | None = None,
                 n_drugs: int = 0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Per-node-type input projections
        self.input_projections = nn.ModuleDict()
        for ntype, in_dim in node_feature_dims.items():
            self.input_projections[ntype] = Linear(in_dim, hidden_dim)

        # Learned drug embeddings
        self.drug_embedding = None
        if n_drugs > 0:
            self.drug_embedding = nn.Embedding(n_drugs, hidden_dim)

        # HGT convolution layers
        metadata = (node_types or list(node_feature_dims.keys()), edge_types or [])
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                HGTConv(hidden_dim, hidden_dim, metadata, heads=n_heads)
            )

        # MLP prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x_dict, edge_index_dict):
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_projections:
                h_dict[ntype] = self.input_projections[ntype](x).relu()
            else:
                h_dict[ntype] = x

        if self.drug_embedding is not None and "drug" in h_dict:
            drug_ids = torch.arange(h_dict["drug"].shape[0], device=h_dict["drug"].device)
            h_dict["drug"] = h_dict["drug"] + self.drug_embedding(drug_ids)

        for conv in self.convs:
            h_update = conv(h_dict, edge_index_dict)
            for ntype in h_dict:
                if ntype in h_update:
                    h_dict[ntype] = F.relu(F.dropout(
                        h_update[ntype], p=self.dropout, training=self.training
                    ))

        return h_dict

    def decode(self, h_dict, cell_line_idx, drug_idx):
        cell_emb = h_dict["cell_line"][cell_line_idx]
        drug_emb = h_dict["drug"][drug_idx]
        pair_emb = torch.cat([cell_emb, drug_emb], dim=-1)
        return self.predictor(pair_emb).squeeze(-1)

    def forward(self, x_dict, edge_index_dict, cell_line_idx, drug_idx):
        h_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(h_dict, cell_line_idx, drug_idx)
