"""Rule-based stablecoin market regime labels.

Regimes:
    normal
    peg_pressure
    cross_venue_fragmentation
    liquidity_vacuum
    issuer_event_window
    settlement_congestion
    recovery

Rules:
    peg_pressure: abs(stablecoin deviation) > 25 bps
    liquidity_vacuum: spread_bps > 95th percentile AND depth_10bp < 5th percentile
    issuer_event_window: within ±6h of issuer/reserve event
    settlement_congestion: gas/block/transfer stress above threshold
    recovery: deviation has started mean-reverting from event peak
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_PEG_PRESSURE_THRESHOLD_BPS = 25.0
_ISSUER_WINDOW_HOURS = 6
_SETTLEMENT_CONGESTION_TRANSFER_THRESHOLD = 500  # transfers/minute


def add_regime_labels(
    df: pl.DataFrame,
    deviation_col: str = "deviation_from_1_usd_bps",
    spread_col: str = "spread_bps",
    depth_col: str = "depth_bid_10bp",
    gas_col: str = "gas_proxy",
    transfer_count_col: str = "transfer_count_1m",
    issuer_event_col: str = "is_issuer_event_window",
    ts_col: str = "ts_ns",
) -> pl.DataFrame:
    """Add rule-based regime labels to a feature DataFrame.

    Args:
        df: Feature DataFrame with microstructure and settlement columns.
        deviation_col: Column with stablecoin USD deviation in basis points.
        spread_col: Column with bid-ask spread in basis points.
        depth_col: Column with depth within 10bp.
        gas_col: Column with gas price proxy (gwei).
        transfer_count_col: Column with on-chain transfer count per minute.
        issuer_event_col: Column with issuer event window flag.
        ts_col: Timestamp column name (nanoseconds).

    Returns:
        DataFrame with ``label_regime`` column appended.
    """
    if df.is_empty():
        return df

    # Compute percentile thresholds for liquidity vacuum detection
    spread_95th = df[spread_col].quantile(0.95) if spread_col in df.columns else None
    depth_5th = df[depth_col].quantile(0.05) if depth_col in df.columns else None

    conditions = []

    # Peg pressure
    if deviation_col in df.columns:
        conditions.append(
            (pl.col(deviation_col).abs() > _PEG_PRESSURE_THRESHOLD_BPS).alias(
                "_is_peg_pressure"
            )
        )
    else:
        conditions.append(pl.lit(False).alias("_is_peg_pressure"))

    # Liquidity vacuum
    if (
        spread_col in df.columns
        and depth_col in df.columns
        and spread_95th
        and depth_5th
    ):
        conditions.append(
            (
                (pl.col(spread_col) > spread_95th) & (pl.col(depth_col) < depth_5th)
            ).alias("_is_liquidity_vacuum")
        )
    else:
        conditions.append(pl.lit(False).alias("_is_liquidity_vacuum"))

    # Issuer event window
    if issuer_event_col in df.columns:
        conditions.append(
            pl.col(issuer_event_col).cast(pl.Boolean).alias("_is_issuer_event")
        )
    else:
        conditions.append(pl.lit(False).alias("_is_issuer_event"))

    # Settlement congestion
    if transfer_count_col in df.columns:
        conditions.append(
            (
                pl.col(transfer_count_col) > _SETTLEMENT_CONGESTION_TRANSFER_THRESHOLD
            ).alias("_is_settlement_congestion")
        )
    else:
        conditions.append(pl.lit(False).alias("_is_settlement_congestion"))

    df = df.with_columns(conditions)

    # Assign regime label (priority order)
    df = df.with_columns(
        pl.when(pl.col("_is_issuer_event"))
        .then(pl.lit("issuer_event_window"))
        .when(pl.col("_is_peg_pressure") & pl.col("_is_liquidity_vacuum"))
        .then(pl.lit("liquidity_vacuum"))
        .when(pl.col("_is_peg_pressure"))
        .then(pl.lit("peg_pressure"))
        .when(pl.col("_is_settlement_congestion"))
        .then(pl.lit("settlement_congestion"))
        .when(pl.col("_is_liquidity_vacuum"))
        .then(pl.lit("cross_venue_fragmentation"))
        .otherwise(pl.lit("normal"))
        .alias("label_regime")
    )

    # Drop intermediate columns
    df = df.drop(
        [
            "_is_peg_pressure",
            "_is_liquidity_vacuum",
            "_is_issuer_event",
            "_is_settlement_congestion",
        ]
    )
    return df
