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
    """

    def __init__(
        self,
        node_types: list[str] | None = None,
        edge_types: list[tuple[str, str, str]] | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        task: str = "node_classification",
        num_epochs: int = 20,
        lr: float = 1e-3,
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
        self.num_epochs = num_epochs
        self.lr = lr
        self._model = None
        self._trained_node_types: list[str] = []
        self._trained_edge_types: list[tuple] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

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

        node_id_map: dict[str, dict[str, int]] = {}
        for node_type, node_list in nodes.items():
            node_id_map[node_type] = {n["id"]: i for i, n in enumerate(node_list)}
            n = len(node_list)
            data[node_type].x = torch.zeros(n, self.hidden_dim)

        for edge in edges:
            etype = edge.get("type", "unknown")
            src_type = edge.get("source_type", "venue")
            dst_type = edge.get("target_type", "stablecoin")
            src_id = edge.get("source", "")
            dst_id = edge.get("target", "")

            src_idx = node_id_map.get(src_type, {}).get(src_id, 0)
            dst_idx = node_id_map.get(dst_type, {}).get(dst_id, 0)

            key = (src_type, etype, dst_type)
            if not hasattr(data[key], "edge_index") or data[key].edge_index is None:
                data[key].edge_index = torch.zeros(2, 0, dtype=torch.long)

            new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
            data[key].edge_index = torch.cat([data[key].edge_index, new_edge], dim=1)

        return data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self, graph_snapshots: list[dict], labels: np.ndarray
    ) -> "HeteroGNNWrapper":
        """Train the HGT model on a sequence of graph snapshots.

        Args:
            graph_snapshots: List of graph snapshot dicts from
                :func:`~stressbench.features.graph_features.build_graph_snapshot`.
            labels: 1-D array of target labels (binary or continuous).

        Returns:
            ``self`` for chaining.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch_geometric.nn import HGTConv

        if not graph_snapshots:
            logger.warning("No graph snapshots provided; skipping HeteroGNN training.")
            return self

        device = torch.device("cpu")

        # Build PyG data objects
        data_list = []
        for snap in graph_snapshots:
            try:
                data_list.append(self.build_hetero_data(snap))
            except Exception as exc:
                logger.warning("Failed to build HeteroData: %s", exc)

        if not data_list:
            logger.warning("All snapshots failed to convert; skipping training.")
            return self

        # Collect all node types and edge types across the dataset
        all_node_types: set[str] = set(self.node_types)
        all_edge_types_set: set[tuple] = set()
        for data in data_list:
            all_node_types.update(data.node_types)
            for et in data.edge_types:
                all_edge_types_set.add(et)

        node_types_list = sorted(all_node_types)
        edge_types_list = sorted(all_edge_types_set)

        if not edge_types_list:
            logger.warning(
                "No edges found across snapshots; skipping HeteroGNN training."
            )
            return self

        self._trained_node_types = node_types_list
        self._trained_edge_types = edge_types_list
        metadata = (node_types_list, edge_types_list)

        # Build the HGT model
        hidden = self.hidden_dim
        out_dim = self.output_dim

        class _HGTModel(nn.Module):
            def __init__(
                self, hidden_dim: int, output_dim: int, num_layers: int, metadata
            ):
                super().__init__()
                self.convs = nn.ModuleList(
                    [
                        HGTConv(hidden_dim, hidden_dim, metadata, heads=2, group="sum")
                        for _ in range(num_layers)
                    ]
                )
                self.head = nn.Linear(hidden_dim, output_dim)

            def forward(self, x_dict, edge_index_dict):
                for conv in self.convs:
                    x_dict = conv(x_dict, edge_index_dict)
                    x_dict = {k: v.relu() for k, v in x_dict.items() if v is not None}
                # Global mean pool across all node types → graph embedding
                embeds = [
                    v.mean(dim=0)
                    for v in x_dict.values()
                    if v is not None and v.numel() > 0
                ]
                if not embeds:
                    return torch.zeros(output_dim)
                return self.head(torch.stack(embeds).mean(dim=0))

        self._model = _HGTModel(hidden, out_dim, self.num_layers, metadata).to(device)

        labels_t = torch.tensor(labels, dtype=torch.float32).to(device)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        is_clf = self.task in ("node_classification", "graph_classification")
        loss_fn = nn.BCEWithLogitsLoss() if is_clf else nn.MSELoss()

        self._model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            n_steps = 0
            for data, label in zip(data_list, labels_t):
                # Fill in any missing node types with zero tensors
                for nt in node_types_list:
                    if nt not in data.node_types or data[nt].x is None:
                        data[nt].x = torch.zeros(1, hidden, device=device)

                x_dict = {nt: data[nt].x.to(device) for nt in node_types_list}
                edge_index_dict = {
                    et: data[et].edge_index.to(device)
                    for et in edge_types_list
                    if et in data.edge_types and data[et].edge_index is not None
                }
                if not edge_index_dict:
                    continue

                optimizer.zero_grad()
                out = self._model(x_dict, edge_index_dict)
                loss = loss_fn(out.squeeze(), label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

            if n_steps > 0 and (epoch + 1) % 5 == 0:
                logger.info(
                    "HeteroGNN epoch %d/%d  avg_loss=%.4f",
                    epoch + 1,
                    self.num_epochs,
                    epoch_loss / n_steps,
                )

        logger.info(
            "HeteroGNN training complete: %d epochs, %d snapshots.",
            self.num_epochs,
            len(data_list),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, graph_snapshots: list[dict]) -> np.ndarray:
        """Run inference on a list of graph snapshots.

        Args:
            graph_snapshots: List of graph snapshot dicts.

        Returns:
            1-D numpy array of predictions (one per snapshot).
        """
        if self._model is None:
            logger.warning("HeteroGNN not trained; returning zeros.")
            return np.zeros(len(graph_snapshots))

        import torch

        device = torch.device("cpu")
        hidden = self.hidden_dim
        node_types_list = self._trained_node_types
        edge_types_list = self._trained_edge_types

        self._model.eval()
        predictions: list[float] = []

        with torch.no_grad():
            for snap in graph_snapshots:
                try:
                    data = self.build_hetero_data(snap)
                except Exception as exc:
                    logger.warning("Failed to build HeteroData for predict: %s", exc)
                    predictions.append(0.0)
                    continue

                for nt in node_types_list:
                    if nt not in data.node_types or data[nt].x is None:
                        data[nt].x = torch.zeros(1, hidden, device=device)

                x_dict = {nt: data[nt].x.to(device) for nt in node_types_list}
                edge_index_dict = {
                    et: data[et].edge_index.to(device)
                    for et in edge_types_list
                    if et in data.edge_types and data[et].edge_index is not None
                }
                if not edge_index_dict:
                    predictions.append(0.0)
                    continue

                out = self._model(x_dict, edge_index_dict)
                predictions.append(float(out.squeeze().item()))

        return np.array(predictions)
