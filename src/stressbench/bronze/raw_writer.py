"""Bronze layer: write raw immutable Parquet files with Hive-style partitioning.

Path convention:
    data/bronze/
      venue=binance/
        channel=trade/
          symbol=USDCUSDT/
            date=2026-05-25/
              hour=13/
                part-<uuid>.parquet
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import polars as pl

from stressbench.common.config import bronze_root
from stressbench.common.hashing import payload_hash as _hash
from stressbench.common.ids import new_batch_id
from stressbench.common.logging import get_logger
from stressbench.common.time import date_str, hour_str, now_ns

logger = get_logger(__name__)


def make_raw_record(
    source: str,
    channel: str,
    symbol: str,
    payload: dict[str, Any],
    ts_exchange: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Construct a canonical raw Bronze record from an exchange message.

    Args:
        source: Venue identifier (e.g. ``"binance"``).
        channel: Data channel (``"trade"``, ``"depth"``, ``"book"``, etc.).
        symbol: Native symbol string as received from the exchange.
        payload: Original exchange-native JSON payload as a Python dict.
        ts_exchange: Exchange-reported timestamp string (optional).
        batch_id: Ingest batch UUID; a new one is generated if not provided.

    Returns:
        A dict conforming to :class:`~stressbench.common.schemas.RawMessage`.
    """
    return {
        "source": source,
        "channel": channel,
        "symbol": symbol,
        "ts_exchange": ts_exchange,
        "ts_receive_ns": now_ns(),
        "payload": payload,
        "payload_hash": _hash(payload),
        "schema_version": "raw.v1",
        "ingest_batch_id": batch_id or new_batch_id(),
    }


def write_raw_batch(
    records: list[dict[str, Any]],
    venue: str,
    channel: str,
    symbol: str,
    date: str,
    hour: str,
    root: Path | None = None,
) -> Path | None:
    """Write a batch of raw records to a Parquet file in the Bronze layer.

    Args:
        records: List of raw record dicts (must be non-empty).
        venue: Venue identifier for the Hive partition key.
        channel: Channel identifier for the Hive partition key.
        symbol: Symbol identifier for the Hive partition key.
        date: ISO date string ``YYYY-MM-DD`` for the partition key.
        hour: Zero-padded hour string ``HH`` for the partition key.
        root: Override for the Bronze root directory; uses config default if None.

    Returns:
        The :class:`~pathlib.Path` of the written Parquet file, or ``None`` if
        ``records`` is empty.
    """
    if not records:
        return None

    root = root or bronze_root()
    path = (
        root
        / f"venue={venue}"
        / f"channel={channel}"
        / f"symbol={symbol}"
        / f"date={date}"
        / f"hour={hour}"
    )
    path.mkdir(parents=True, exist_ok=True)

    # Serialise payload dicts to JSON strings for Parquet compatibility
    serialisable = []
    for rec in records:
        row = dict(rec)
        import json
        row["payload"] = json.dumps(row["payload"], sort_keys=True)
        serialisable.append(row)

    file = path / f"part-{uuid.uuid4().hex}.parquet"
    pl.DataFrame(serialisable).write_parquet(file)
    logger.info(
        "Wrote %d records to %s", len(records), file
    )
    return file


def write_raw_record(
    record: dict[str, Any],
    root: Path | None = None,
) -> Path | None:
    """Convenience wrapper to write a single raw record to Bronze.

    Derives the date and hour partition keys from ``ts_receive_ns``.
    """
    ts_ns: int = record["ts_receive_ns"]
    return write_raw_batch(
        records=[record],
        venue=record["source"],
        channel=record["channel"],
        symbol=record["symbol"],
        date=date_str(ts_ns),
        hour=hour_str(ts_ns),
        root=root,
    )
