"""Binance live WebSocket collector.

Subscribes to ``<symbol>@trade``, ``<symbol>@depth``, and
``<symbol>@bookTicker`` streams via the market-data-only endpoint
``wss://data-stream.binance.vision``.

Key behaviours:
- Adds a local nanosecond receive timestamp to every message.
- Writes raw payloads to the Bronze layer via :mod:`stressbench.bronze.raw_writer`.
- Reconnects before the 24-hour connection lifetime expires.
- Handles ping/pong to keep the connection alive.
- Tracks per-symbol message rates and logs connection drops.

Reference:
    https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Sequence

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from stressbench.bronze.raw_writer import make_raw_record, write_raw_batch
from stressbench.common.ids import new_batch_id
from stressbench.common.logging import get_logger
from stressbench.common.time import date_str, hour_str, now_ns

logger = get_logger(__name__)

_BASE_URL = "wss://data-stream.binance.vision/stream"
_MAX_CONNECTION_SECONDS = 23 * 3600  # reconnect before 24-hour limit
_FLUSH_INTERVAL_SECONDS = 30
_CHANNELS = ["trade", "depth", "bookTicker"]


def _build_stream_url(symbols: Sequence[str], time_unit: str = "MICROSECOND") -> str:
    streams = []
    for sym in symbols:
        s = sym.lower()
        for ch in _CHANNELS:
            streams.append(f"{s}@{ch}")
    stream_param = "/".join(streams)
    return f"{_BASE_URL}?streams={stream_param}&timeUnit={time_unit}"


def _channel_from_stream(stream: str) -> str:
    """Extract a normalised channel name from a Binance stream name."""
    if "@trade" in stream:
        return "trade"
    if "@depth" in stream:
        return "depth"
    if "@bookTicker" in stream:
        return "bookTicker"
    return "unknown"


async def run_collector(
    symbols: Sequence[str],
    batch_id: str | None = None,
) -> None:
    """Run the Binance live collector indefinitely, reconnecting as needed.

    Args:
        symbols: List of Binance native symbol strings (e.g. ``["USDCUSDT"]``).
        batch_id: Optional ingest batch UUID; a new one is generated per
            reconnect if not provided.
    """
    url = _build_stream_url(symbols)
    logger.info("Connecting to Binance stream: %s", url)

    while True:
        batch_id = batch_id or new_batch_id()
        buffer: dict[str, list] = {}  # keyed by (channel, symbol)
        start_time = time.monotonic()

        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Binance WebSocket connected (batch_id=%s)", batch_id)
                last_flush = time.monotonic()

                async for raw_msg in ws:
                    ts_ns = now_ns()
                    elapsed = time.monotonic() - start_time

                    # Reconnect before 24-hour limit
                    if elapsed > _MAX_CONNECTION_SECONDS:
                        logger.info("Approaching 24h limit; reconnecting.")
                        break

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message received: %s", raw_msg[:200])
                        continue

                    stream_name = msg.get("stream", "")
                    data = msg.get("data", msg)
                    symbol = data.get("s", "UNKNOWN")
                    channel = _channel_from_stream(stream_name)

                    record = make_raw_record(
                        source="binance",
                        channel=channel,
                        symbol=symbol,
                        payload=data,
                        ts_exchange=str(data.get("T") or data.get("E") or ""),
                        batch_id=batch_id,
                    )
                    record["ts_receive_ns"] = ts_ns

                    key = (channel, symbol)
                    buffer.setdefault(key, []).append(record)

                    # Periodic flush to Bronze
                    now_mono = time.monotonic()
                    if now_mono - last_flush >= _FLUSH_INTERVAL_SECONDS:
                        _flush_buffer(buffer)
                        buffer = {}
                        last_flush = now_mono

        except (ConnectionClosedError, ConnectionClosedOK) as exc:
            logger.warning("Binance connection closed: %s. Reconnecting in 5s.", exc)
        except Exception as exc:
            logger.error("Unexpected error: %s. Reconnecting in 5s.", exc)

        # Flush remaining buffer before reconnect
        if buffer:
            _flush_buffer(buffer)
        buffer = {}
        batch_id = new_batch_id()
        await asyncio.sleep(5)


def _flush_buffer(buffer: dict[tuple, list]) -> None:
    for (channel, symbol), records in buffer.items():
        if not records:
            continue
        ts_ns = records[0]["ts_receive_ns"]
        write_raw_batch(
            records=records,
            venue="binance",
            channel=channel,
            symbol=symbol,
            date=date_str(ts_ns),
            hour=hour_str(ts_ns),
        )
        logger.debug("Flushed %d records for %s/%s", len(records), channel, symbol)
