"""Recovery half-life labels.

For each stress event:
    peak_deviation_time = time of max abs(deviation)
    half_life = first time after peak where abs(deviation) <= 0.5 × peak_deviation

Label: label_recovery_halflife_minutes
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def compute_recovery_halflife(
    df: pl.DataFrame,
    deviation_col: str = "deviation_from_1_usd_bps",
    ts_col: str = "ts_ns",
    event_start_ns: int | None = None,
    event_end_ns: int | None = None,
) -> float | None:
    """Compute the recovery half-life in minutes for a stress event.

    Args:
        df: Feature DataFrame sorted by ``ts_col`` ascending.
        deviation_col: Column with stablecoin USD deviation in basis points.
        ts_col: Timestamp column name (nanoseconds).
        event_start_ns: Start of the stress event window (nanoseconds).
        event_end_ns: End of the stress event window (nanoseconds).

    Returns:
        Recovery half-life in minutes, or ``None`` if not recoverable within
        the event window.
    """
    if df.is_empty() or deviation_col not in df.columns:
        return None

    df = df.sort(ts_col)

    if event_start_ns is not None:
        df = df.filter(pl.col(ts_col) >= event_start_ns)
    if event_end_ns is not None:
        df = df.filter(pl.col(ts_col) <= event_end_ns)

    if df.is_empty():
        return None

    abs_dev = df[deviation_col].abs()
    peak_idx = abs_dev.arg_max()
    peak_deviation = float(abs_dev[peak_idx])
    peak_ts = int(df[ts_col][peak_idx])

    if peak_deviation == 0:
        return 0.0

    half_threshold = peak_deviation * 0.5

    # Find first timestamp after peak where abs(deviation) <= half_threshold
    post_peak = df.filter(pl.col(ts_col) > peak_ts)
    if post_peak.is_empty():
        return None

    recovery_rows = post_peak.filter(pl.col(deviation_col).abs() <= half_threshold)
    if recovery_rows.is_empty():
        return None

    recovery_ts = int(recovery_rows[ts_col][0])
    halflife_minutes = (recovery_ts - peak_ts) / 60_000_000_000
    return halflife_minutes


def add_recovery_labels(
    df: pl.DataFrame,
    event_windows: list[dict],
    deviation_col: str = "deviation_from_1_usd_bps",
    ts_col: str = "ts_ns",
) -> pl.DataFrame:
    """Add recovery half-life labels to a feature DataFrame.

    For each row, the label is the recovery half-life of the stress event
    that the row belongs to, or ``None`` if the row is not in a stress event.

    Args:
        df: Feature DataFrame.
        event_windows: List of event window dicts with ``start_ns`` and ``end_ns``.
        deviation_col: Deviation column name.
        ts_col: Timestamp column name.

    Returns:
        DataFrame with ``label_recovery_halflife_minutes`` column appended.
    """
    if df.is_empty():
        return df

    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("label_recovery_halflife_minutes")
    )

    for event in event_windows:
        start_ns = event.get("start_ns")
        end_ns = event.get("end_ns")
        if start_ns is None or end_ns is None:
            continue

        halflife = compute_recovery_halflife(
            df=df,
            deviation_col=deviation_col,
            ts_col=ts_col,
            event_start_ns=start_ns,
            event_end_ns=end_ns,
        )
        if halflife is not None:
            df = df.with_columns(
                pl.when((pl.col(ts_col) >= start_ns) & (pl.col(ts_col) <= end_ns))
                .then(pl.lit(halflife))
                .otherwise(pl.col("label_recovery_halflife_minutes"))
                .alias("label_recovery_halflife_minutes")
            )

    return df
