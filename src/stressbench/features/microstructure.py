"""Microstructure feature computation.

Produces 1-second and 1-minute book state snapshots with spread, depth,
imbalance, trade volume, and data-quality scores.

Output tables: ``feat_book_1s``, ``feat_book_1m``
"""

from __future__ import annotations

import polars as pl

from stressbench.book.order_book import OrderBook
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_BPS_WINDOWS = [1.0, 5.0, 10.0]


def compute_book_snapshot_features(
    book: OrderBook,
    ts_ns: int,
    venue_id: str,
    instrument_id: str,
    trade_count: int = 0,
    trade_volume: float = 0.0,
    quote_update_count: int = 0,
    is_resync: bool = False,
    is_checksum_failed: bool = False,
) -> dict:
    """Compute microstructure features from a single order-book state.

    Args:
        book: Reconstructed :class:`~stressbench.book.order_book.OrderBook`.
        ts_ns: Timestamp in nanoseconds.
        venue_id: Venue identifier.
        instrument_id: Canonical instrument identifier.
        trade_count: Number of trades in the current interval.
        trade_volume: Total trade volume in the current interval.
        quote_update_count: Number of quote updates in the current interval.
        is_resync: Whether the book is in a resync window.
        is_checksum_failed: Whether a checksum failure was detected.

    Returns:
        Dict of microstructure features conforming to ``feat_book_1s`` schema.
    """
    mid = book.mid()
    best_bid = book.best_bid()
    best_ask = book.best_ask()
    spread_bps = book.spread_bps()

    depths_bid = {bps: book.depth_within_bps("bid", bps) for bps in _BPS_WINDOWS}
    depths_ask = {bps: book.depth_within_bps("ask", bps) for bps in _BPS_WINDOWS}
    imbalances = {
        bps: book.imbalance(bps) for bps in _BPS_WINDOWS
    }

    # Data quality score: 1.0 = perfect, 0.0 = unusable
    dq_score = 1.0
    if is_resync:
        dq_score -= 0.5
    if is_checksum_failed:
        dq_score -= 0.3
    if book.is_crossed():
        dq_score -= 0.4
    dq_score = max(0.0, dq_score)

    return {
        "ts_ns": ts_ns,
        "venue_id": venue_id,
        "instrument_id": instrument_id,
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread_bps,
        "depth_bid_1bp": depths_bid[1.0],
        "depth_ask_1bp": depths_ask[1.0],
        "depth_bid_5bp": depths_bid[5.0],
        "depth_ask_5bp": depths_ask[5.0],
        "depth_bid_10bp": depths_bid[10.0],
        "depth_ask_10bp": depths_ask[10.0],
        "imbalance_1bp": imbalances[1.0],
        "imbalance_5bp": imbalances[5.0],
        "trade_count": trade_count,
        "trade_volume": trade_volume,
        "quote_update_count": quote_update_count,
        "data_quality_score": dq_score,
    }


def aggregate_to_1m(df_1s: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 1-second book snapshots to 1-minute intervals.

    Args:
        df_1s: DataFrame of 1-second book snapshot features.

    Returns:
        DataFrame of 1-minute aggregated features.
    """
    if df_1s.is_empty():
        return df_1s

    # Floor timestamp to minute
    df = df_1s.with_columns(
        ((pl.col("ts_ns") // 60_000_000_000) * 60_000_000_000).alias("ts_1m_ns")
    )

    agg = df.group_by(["ts_1m_ns", "venue_id", "instrument_id"]).agg(
        pl.col("mid").mean().alias("mid_mean"),
        pl.col("mid").first().alias("mid_open"),
        pl.col("mid").last().alias("mid_close"),
        pl.col("mid").max().alias("mid_high"),
        pl.col("mid").min().alias("mid_low"),
        pl.col("spread_bps").mean().alias("spread_bps_mean"),
        pl.col("spread_bps").max().alias("spread_bps_max"),
        pl.col("depth_bid_10bp").mean().alias("depth_bid_10bp_mean"),
        pl.col("depth_ask_10bp").mean().alias("depth_ask_10bp_mean"),
        pl.col("imbalance_1bp").mean().alias("imbalance_1bp_mean"),
        pl.col("trade_count").sum().alias("trade_count_1m"),
        pl.col("trade_volume").sum().alias("trade_volume_1m"),
        pl.col("quote_update_count").sum().alias("quote_update_count_1m"),
        pl.col("data_quality_score").min().alias("data_quality_score_min"),
    )
    return agg.sort(["venue_id", "instrument_id", "ts_1m_ns"])
