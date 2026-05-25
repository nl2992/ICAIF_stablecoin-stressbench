"""Arbitrage window labels.

For each timestamp, computes the maximum net profit (after all transaction costs)
achievable over the next H minutes for a given notional size Q.

Label: arb_window(Q, H) = 1 if max net_profit_bps over next H minutes > threshold

Sizes Q: $10k, $50k, $100k, $500k
Horizons H: 1m, 5m, 15m
Thresholds: 0, 5, 10 bps

IMPORTANT: Labels are generated from future net profit values.
No future information is used at feature time t.
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_NOTIONAL_SIZES = [10_000, 50_000, 100_000, 500_000]
_HORIZONS_NS = {
    "1m": 60_000_000_000,
    "5m": 300_000_000_000,
    "15m": 900_000_000_000,
}
_THRESHOLDS_BPS = [0.0, 5.0, 10.0]


def add_arbitrage_labels(
    df: pl.DataFrame,
    net_profit_col_template: str = "net_profit_bps_q{q}",
    ts_col: str = "ts_ns",
) -> pl.DataFrame:
    """Add forward-looking arbitrage window labels to a feature DataFrame.

    The DataFrame must contain pre-computed ``net_profit_bps`` columns for
    each notional size Q, named according to ``net_profit_col_template``.

    Args:
        df: Feature DataFrame with net profit columns and ``ts_col``.
        net_profit_col_template: Template for net profit column names.
            ``{q}`` is replaced with the notional size (e.g. ``10000``).
        ts_col: Timestamp column name (nanoseconds).

    Returns:
        DataFrame with arbitrage window label columns appended.
    """
    if df.is_empty():
        return df

    df = df.sort(ts_col)

    for q in _NOTIONAL_SIZES:
        net_col = net_profit_col_template.format(q=q)
        if net_col not in df.columns:
            logger.warning("Column '%s' not found; skipping Q=%d labels.", net_col, q)
            continue

        for horizon_name, horizon_ns in _HORIZONS_NS.items():
            # Rolling max of net profit over the next H minutes
            # Implemented as a forward-looking rolling window via shift
            rolling_max_col = f"_max_net_{q}_{horizon_name}"
            df = df.with_columns(
                pl.col(net_col)
                .rolling_max(window_size=int(horizon_ns // 60_000_000_000), min_periods=1)
                .shift(-(int(horizon_ns // 60_000_000_000)))
                .alias(rolling_max_col)
            )

            for threshold in _THRESHOLDS_BPS:
                label_col = f"label_arb_q{q}_{horizon_name}_gt{int(threshold)}bps"
                df = df.with_columns(
                    (pl.col(rolling_max_col) > threshold)
                    .cast(pl.Int8)
                    .alias(label_col)
                )

            df = df.drop(rolling_max_col)

    return df
