"""Tardis historical data loader.

Downloads normalized tick-level datasets from Tardis Machine Server or the
Tardis HTTP API and converts them to the canonical Silver schema.

Supported data types:
    trades, incremental_book_L2, book_snapshot_1s, quotes, book_ticker,
    derivative_ticker, funding, liquidations

Reference:
    https://docs.tardis.dev
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import requests

from stressbench.common.config import bronze_root, get_env
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_TARDIS_API_BASE = "https://api.tardis.dev/v1"
_TARDIS_DATA_TYPES = [
    "trades",
    "incremental_book_L2",
    "book_snapshot_1s",
    "quotes",
    "book_ticker",
    "derivative_ticker",
    "funding",
    "liquidations",
]


def _api_key() -> str | None:
    return get_env("TARDIS_API_KEY")


def pull_tardis_day(
    exchange: str,
    symbol: str,
    data_type: str,
    date: str,
    root: Path | None = None,
    overwrite: bool = False,
) -> str | None:
    """Download one exchange/symbol/data_type/day from Tardis into Bronze.

    Args:
        exchange: Tardis exchange name (e.g. ``"coinbase"``).
        symbol: Tardis symbol string (e.g. ``"USDC-USD"``).
        data_type: One of the supported Tardis data types.
        date: ISO date string ``YYYY-MM-DD``.
        root: Bronze root override.
        overwrite: Re-download even if the file already exists.

    Returns:
        Path string to the downloaded file, or ``None`` on failure.
    """
    root = root or bronze_root()
    out_dir = (
        root
        / "vendor=tardis"
        / f"exchange={exchange}"
        / f"data_type={data_type}"
        / f"symbol={symbol}"
        / f"date={date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{exchange}-{data_type}-{symbol}-{date}.csv.gz"

    if out_file.exists() and not overwrite:
        logger.info("Already exists, skipping: %s", out_file)
        return str(out_file)

    api_key = _api_key()
    if not api_key:
        logger.error(
            "TARDIS_API_KEY not set. Cannot download Tardis data. "
            "Set it in .env or environment."
        )
        return None

    url = (
        f"{_TARDIS_API_BASE}/data-feeds/{exchange}"
        f"?dataTypes={data_type}&from={date}&to={date}&symbols={symbol}"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    logger.info("Downloading Tardis: exchange=%s symbol=%s type=%s date=%s", exchange, symbol, data_type, date)
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("Tardis HTTP error: %s", exc)
        return None
    except requests.RequestException as exc:
        logger.error("Tardis request failed: %s", exc)
        return None

    with open(out_file, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)

    logger.info("Saved: %s", out_file)
    return str(out_file)


def normalize_tardis_csv(path: str) -> pl.DataFrame:
    """Convert a Tardis normalized CSV (or CSV.gz) to the canonical Silver schema.

    Args:
        path: Path to the Tardis CSV or CSV.gz file.

    Returns:
        A :class:`polars.DataFrame` with canonical column names.
    """
    df = pl.read_csv(path, infer_schema_length=10000)

    # Rename common Tardis columns to canonical names
    rename_map = {
        "localTimestamp": "ts_receive_ns",
        "timestamp": "ts_event",
        "exchange": "venue_id",
        "symbol": "native_symbol",
        "id": "trade_id",
        "side": "side",
        "price": "price",
        "amount": "size",
        "asks[0].price": "best_ask",
        "bids[0].price": "best_bid",
        "asks[0].amount": "best_ask_size",
        "bids[0].amount": "best_bid_size",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(existing)

    # Convert localTimestamp (microseconds) to nanoseconds
    if "ts_receive_ns" in df.columns:
        df = df.with_columns(
            (pl.col("ts_receive_ns").cast(pl.Int64) * 1000).alias("ts_receive_ns")
        )

    return df
