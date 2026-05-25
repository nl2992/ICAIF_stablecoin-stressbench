"""Graph Neural Network models for stablecoin settlement network analysis.

Graph construction:
    Nodes: venues, stablecoins, chains, base assets (BTC/ETH)
    Edges: tradable pair, transfer rail, DEX pool, issuer reserve

Prediction tasks:
    edge-level: profitable route yes/no
    node-level: venue/stablecoin stress state
    graph-level: market-wide stress regime

Requires: torch, torch_geometric
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def _check_pyg() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        raise ImportError(
            "torch_geometric is required for graph models. "
            "Install with: pip install torch_geometric"
        )


class HeteroGNNWrapper:
    """Heterogeneous GNN for the stablecoin settlement network.

    Uses PyTorch Geometric's HeteroData and HGTConv (Heterogeneous Graph
    Transformer) for node-level and graph-level predictions.

    This is the 'ambitious' model and should not block the benchmark.
    """

    def __init__(
        self,
        node_types: list[str] | None = None,
        edge_types: list[tuple[str, str, str]] | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        task: str = "node_classification",
    ) -> None:
        _check_pyg()
        self.node_types = node_types or ["venue", "stablecoin", "chain", "base_asset"]
        self.edge_types = edge_types or [
            ("venue", "tradable_pair", "stablecoin"),
            ("stablecoin", "transfer_rail", "chain"),
            ("venue", "cross_quote_basis", "stablecoin"),
        ]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.task = task
        self._model = None

    def build_hetero_data(self, graph_snapshot: dict[str, Any]):
        """Convert a graph snapshot dict to a PyG HeteroData object.

        Args:
            graph_snapshot: Dict from
                :func:`~stressbench.features.graph_features.build_graph_snapshot`.

        Returns:
            A :class:`torch_geometric.data.HeteroData` object.
        """
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        nodes = graph_snapshot.get("nodes", {})
        edges = graph_snapshot.get("edges", [])

        # Assign node features
        node_id_map: dict[str, dict[str, int]] = {}
        for node_type, node_list in nodes.items():
            node_id_map[node_type] = {n["id"]: i for i, n in enumerate(node_list)}
            # Simple one-hot encoding as placeholder; replace with real features
            n = len(node_list)
            data[node_type].x = torch.zeros(n, self.hidden_dim)

        # Assign edge indices
        for edge in edges:
            etype = edge.get("type", "unknown")
            src_type = edge.get("source_type", "venue")
            dst_type = edge.get("target_type", "stablecoin")
            src_id = edge.get("source", "")
            dst_id = edge.get("target", "")

            src_idx = node_id_map.get(src_type, {}).get(src_id, 0)
            dst_idx = node_id_map.get(dst_type, {}).get(dst_id, 0)

            key = (src_type, etype, dst_type)
            if not hasattr(data[key], "edge_index"):
                data[key].edge_index = torch.zeros(2, 0, dtype=torch.long)

            new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
            data[key].edge_index = torch.cat([data[key].edge_index, new_edge], dim=1)

        return data

    def fit(self, graph_snapshots: list[dict], labels: np.ndarray) -> "HeteroGNNWrapper":
        logger.info(
            "HeteroGNN training is a placeholder. "
            "Implement full training loop after baseline tables are stable."
        )
        return self

    def predict(self, graph_snapshots: list[dict]) -> np.ndarray:
        logger.info("HeteroGNN predict: returning zeros (placeholder).")
        return np.zeros(len(graph_snapshots))
