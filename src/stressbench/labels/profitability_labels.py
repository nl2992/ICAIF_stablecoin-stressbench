"""Net profitability ranking labels.

Ranks model predictions by net economic value after transaction costs.
Used for the economic evaluation leaderboard.
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def add_profitability_rank_label(
    df: pl.DataFrame,
    net_profit_col: str = "net_profit_bps_q50000",
    ts_col: str = "ts_ns",
    rank_col: str = "label_profitability_rank",
) -> pl.DataFrame:
    """Add a profitability rank label based on net profit per timestamp.

    Ranks rows by net profit in descending order within each time window.
    Rank 1 = highest net profit opportunity.

    Args:
        df: Feature DataFrame with net profit column.
        net_profit_col: Column with net profit in basis points.
        ts_col: Timestamp column name.
        rank_col: Output rank column name.

    Returns:
        DataFrame with profitability rank column appended.
    """
    if df.is_empty() or net_profit_col not in df.columns:
        return df

    df = df.with_columns(
        pl.col(net_profit_col)
        .rank(method="dense", descending=True)
        .over(ts_col)
        .alias(rank_col)
    )
    return df


def add_is_profitable_label(
    df: pl.DataFrame,
    net_profit_col: str = "net_profit_bps_q50000",
    threshold_bps: float = 0.0,
    label_col: str = "label_is_profitable",
) -> pl.DataFrame:
    """Add a binary label indicating whether the net profit exceeds the threshold.

    Args:
        df: Feature DataFrame with net profit column.
        net_profit_col: Column with net profit in basis points.
        threshold_bps: Minimum net profit threshold.
        label_col: Output binary label column name.

    Returns:
        DataFrame with binary profitability label appended.
    """
    if df.is_empty() or net_profit_col not in df.columns:
        return df

    df = df.with_columns(
        (pl.col(net_profit_col) > threshold_bps).cast(pl.Int8).alias(label_col)
    )
    return df
