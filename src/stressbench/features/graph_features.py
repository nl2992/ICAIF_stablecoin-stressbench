"""Graph-based features for the stablecoin settlement network.

Constructs a heterogeneous graph with:
    - Nodes: venues, stablecoins, chains, base assets (BTC/ETH)
    - Edges: tradable pair, transfer rail, DEX pool, issuer reserve

Edge features: spread_bps, depth, basis_bps, fee_bps, transfer_status,
               settlement_delay_proxy, volume

This module provides graph construction utilities. The actual GNN training
is implemented in :mod:`stressbench.models.graph_models`.
"""

from __future__ import annotations

from typing import Any


def build_graph_snapshot(
    book_features: list[dict],
    basis_features: list[dict],
    settlement_features: list[dict],
    fee_schedules: dict[str, Any],
    ts_ns: int,
) -> dict[str, Any]:
    """Build a heterogeneous graph snapshot for a given timestamp.

    Args:
        book_features: List of microstructure feature dicts (one per instrument).
        basis_features: List of basis feature dicts.
        settlement_features: List of settlement feature dicts.
        fee_schedules: Fee schedule config dict.
        ts_ns: Timestamp in nanoseconds.

    Returns:
        Dict with ``nodes`` and ``edges`` keys describing the graph.
    """
    nodes: dict[str, list[dict]] = {
        "venue": [],
        "stablecoin": [],
        "chain": [],
        "base_asset": [],
    }
    edges: list[dict] = []

    # Build venue nodes from book features
    seen_venues = set()
    for feat in book_features:
        venue_id = feat.get("venue_id", "")
        if venue_id and venue_id not in seen_venues:
            nodes["venue"].append(
                {
                    "id": venue_id,
                    "spread_bps": feat.get("spread_bps"),
                    "depth_bid_10bp": feat.get("depth_bid_10bp"),
                    "depth_ask_10bp": feat.get("depth_ask_10bp"),
                    "imbalance_1bp": feat.get("imbalance_1bp"),
                    "data_quality_score": feat.get("data_quality_score"),
                }
            )
            seen_venues.add(venue_id)

    # Build stablecoin nodes
    for sc in ["USDC", "USDT", "DAI"]:
        nodes["stablecoin"].append({"id": sc})

    # Build chain nodes
    for chain in ["ethereum", "polygon", "solana"]:
        nodes["chain"].append({"id": chain})

    # Build base asset nodes
    for asset in ["BTC", "ETH"]:
        nodes["base_asset"].append({"id": asset})

    # Build tradable pair edges from book features
    for feat in book_features:
        instrument_id = feat.get("instrument_id", "")
        if ":" in instrument_id:
            venue, symbol = instrument_id.split(":", 1)
            edges.append(
                {
                    "type": "tradable_pair",
                    "source": venue,
                    "target": symbol,
                    "spread_bps": feat.get("spread_bps"),
                    "depth": feat.get("depth_bid_10bp"),
                    "volume": feat.get("trade_volume"),
                    "fee_bps": fee_schedules.get(venue, {}).get("taker_bps"),
                }
            )

    # Build basis edges from cross-quote features
    for basis in basis_features:
        edges.append(
            {
                "type": "cross_quote_basis",
                "source": basis.get("venue_id"),
                "target": basis.get("stablecoin"),
                "basis_bps": basis.get("cross_quote_basis_bps"),
            }
        )

    # Build settlement edges from on-chain features
    for settle in settlement_features:
        edges.append(
            {
                "type": "transfer_rail",
                "source": settle.get("chain"),
                "target": settle.get("stablecoin"),
                "transfer_count": settle.get("transfer_count_1m"),
                "gas_proxy": settle.get("gas_proxy"),
                "dex_volume": settle.get("dex_swap_volume_1m"),
            }
        )

    return {
        "ts_ns": ts_ns,
        "nodes": nodes,
        "edges": edges,
    }
