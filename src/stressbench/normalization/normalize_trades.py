"""Normalize raw Bronze trade messages into canonical Silver fact_trade records.

Normalization rules:
    1. Convert all timestamps to UTC nanoseconds.
    2. Keep both exchange timestamp and local receive timestamp.
    3. Preserve native symbol and derive canonical instrument_id.
    4. Never overwrite raw payloads.
    5. Deduplicate by venue + symbol + trade_id + payload_hash.
    6. Flag outlier prices instead of deleting them.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from stressbench.common.ids import instrument_id as make_instrument_id
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# Outlier detection: flag trades where price deviates > N% from rolling median
_OUTLIER_THRESHOLD_PCT = 5.0


def normalize_binance_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw Binance trade messages (``@trade`` stream).

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver trade schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        symbol = row.get("symbol") or payload.get("s", "UNKNOWN")
        records.append(
            {
                "ts_event_ns": int(payload.get("T", 0)) * 1_000_000,  # ms -> ns
                "ts_receive_ns": row["ts_receive_ns"],
                "venue_id": "binance",
                "instrument_id": make_instrument_id("binance", symbol),
                "native_symbol": symbol,
                "trade_id": str(payload.get("t", "")),
                "side": "buy" if payload.get("m") is False else "sell",
                "price": float(payload.get("p", 0)),
                "size": float(payload.get("q", 0)),
                "notional_usd": None,
                "raw_source": "binance:trade",
                "payload_hash": row.get("payload_hash", ""),
                "ingest_batch_id": row.get("ingest_batch_id", ""),
                "is_outlier_price": False,
            }
        )

    if not records:
        return pl.DataFrame()

    result = pl.DataFrame(records)
    result = _flag_outliers(result)
    return result


def normalize_coinbase_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw Coinbase ``matches`` channel messages.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver trade schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = payload.get("type", "")
        if msg_type not in ("match", "last_match"):
            continue

        symbol = payload.get("product_id", row.get("symbol", "UNKNOWN"))
        records.append(
            {
                "ts_event_ns": _parse_coinbase_ts(payload.get("time", "")),
                "ts_receive_ns": row["ts_receive_ns"],
                "venue_id": "coinbase",
                "instrument_id": make_instrument_id("coinbase", symbol),
                "native_symbol": symbol,
                "trade_id": str(payload.get("trade_id", "")),
                "side": payload.get("side", "unknown"),
                "price": float(payload.get("price", 0)),
                "size": float(payload.get("size", 0)),
                "notional_usd": None,
                "raw_source": "coinbase:matches",
                "payload_hash": row.get("payload_hash", ""),
                "ingest_batch_id": row.get("ingest_batch_id", ""),
                "is_outlier_price": False,
            }
        )

    if not records:
        return pl.DataFrame()

    result = pl.DataFrame(records)
    result = _flag_outliers(result)
    return result


def normalize_kraken_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw Kraken ``trade`` channel messages.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver trade schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        if payload.get("channel") != "trade":
            continue

        for trade in payload.get("data", []):
            symbol = trade.get("symbol", row.get("symbol", "UNKNOWN"))
            records.append(
                {
                    "ts_event_ns": int(float(trade.get("timestamp", 0)) * 1e9),
                    "ts_receive_ns": row["ts_receive_ns"],
                    "venue_id": "kraken",
                    "instrument_id": make_instrument_id("kraken", symbol),
                    "native_symbol": symbol,
                    "trade_id": str(trade.get("trade_id", "")),
                    "side": trade.get("side", "unknown"),
                    "price": float(trade.get("price", 0)),
                    "size": float(trade.get("qty", 0)),
                    "notional_usd": None,
                    "raw_source": "kraken:trade",
                    "payload_hash": row.get("payload_hash", ""),
                    "ingest_batch_id": row.get("ingest_batch_id", ""),
                    "is_outlier_price": False,
                }
            )

    if not records:
        return pl.DataFrame()

    result = pl.DataFrame(records)
    result = _flag_outliers(result)
    return result


def _parse_coinbase_ts(ts_str: str) -> int:
    """Parse a Coinbase ISO-8601 timestamp to nanoseconds."""
    if not ts_str:
        return 0
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1e9)
    except ValueError:
        return 0


def _flag_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Flag trades where price deviates more than the outlier threshold from
    the rolling median within the same instrument.

    Args:
        df: Normalized trade DataFrame with ``price`` and ``instrument_id`` columns.

    Returns:
        DataFrame with ``is_outlier_price`` column updated.
    """
    if df.is_empty() or "price" not in df.columns:
        return df

    median_price = df.group_by("instrument_id").agg(
        pl.col("price").median().alias("median_price")
    )
    df = df.join(median_price, on="instrument_id", how="left")
    df = df.with_columns(
        (
            (pl.col("price") - pl.col("median_price")).abs()
            / pl.col("median_price")
            * 100
            > _OUTLIER_THRESHOLD_PCT
        ).alias("is_outlier_price")
    ).drop("median_price")
    return df


def deduplicate_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate trade records by venue, symbol, trade_id, and payload_hash."""
    if df.is_empty():
        return df
    return df.unique(
        subset=["venue_id", "native_symbol", "trade_id", "payload_hash"],
        keep="first",
    )
