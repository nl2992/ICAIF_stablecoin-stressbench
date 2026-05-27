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

        # Replace NaN → null before rolling operations.
        # NaN means "insufficient book depth to execute this trade" (not profitable).
        # In Polars, NaN > threshold evaluates to True, which would incorrectly
        # label depth-limited rows as profitable.  Null is ignored by rolling_max.
        _clean = f"_clean_net_{q}"
        df = df.with_columns(pl.col(net_col).fill_nan(None).alias(_clean))

        # Time-based forward-looking rolling max using the reversed-frame trick:
        #   1. Sort descending by ts_col.
        #   2. Build an ascending proxy timestamp: proxy = ts_max + ts_min - ts.
        #      In the reversed frame proxy increases as actual time decreases, so
        #      a backward rolling window on proxy covers FORWARD real time.
        #   3. Use rolling_max_by(by=proxy, window_size=horizon) — the window
        #      [proxy - H, proxy] maps back to actual times [ts, ts + H].
        #   4. Sort back to ascending ts.
        ts_max_val = df[ts_col].max()
        ts_min_val = df[ts_col].min()

        df_rev = df.sort(ts_col, descending=True).with_columns(
            (pl.lit(ts_max_val + ts_min_val) - pl.col(ts_col))
            .cast(pl.Datetime("ns"))
            .alias("_ts_proxy")
        )

        for horizon_name, _ in _HORIZONS_NS.items():
            rolling_max_col = f"_max_net_{q}_{horizon_name}"
            # window_size uses Polars duration strings: "1m", "5m", "15m"
            df_rev = df_rev.with_columns(
                pl.col(_clean)
                .rolling_max_by(
                    by="_ts_proxy",
                    window_size=horizon_name,
                    closed="right",
                    min_periods=1,
                )
                .alias(rolling_max_col)
            )

            for threshold in _THRESHOLDS_BPS:
                label_col = f"label_arb_q{q}_{horizon_name}_gt{int(threshold)}bps"
                df_rev = df_rev.with_columns(
                    pl.when(pl.col(rolling_max_col).is_null())
                    .then(pl.lit(None).cast(pl.Int8))
                    .when(pl.col(rolling_max_col) > threshold)
                    .then(pl.lit(1).cast(pl.Int8))
                    .otherwise(pl.lit(0).cast(pl.Int8))
                    .alias(label_col)
                )
            df_rev = df_rev.drop(rolling_max_col)

        df = df_rev.sort(ts_col).drop("_ts_proxy").drop(_clean)

    return df
