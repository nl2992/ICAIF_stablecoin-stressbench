"""Kraken WebSocket v2 live collector.

Subscribes to ``book``, ``trade``, and ``instrument`` channels.
Validates book checksums and flags failures without silently repairing.

Reference:
    https://docs.kraken.com/api/docs/websocket-v2/book
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

_WS_URL = "wss://ws.kraken.com/v2"
_FLUSH_INTERVAL_SECONDS = 30
_BOOK_DEPTH = 100


def _subscribe_message(
    channel: str, pairs: Sequence[str], depth: int = _BOOK_DEPTH
) -> str:
    params: dict = {"channel": channel, "symbol": list(pairs)}
    if channel == "book":
        params["depth"] = depth
    return json.dumps({"method": "subscribe", "params": params})


async def run_collector(
    pairs: Sequence[str],
    channels: Sequence[str] | None = None,
    batch_id: str | None = None,
) -> None:
    """Run the Kraken live collector indefinitely, reconnecting on errors.

    Args:
        pairs: List of Kraken pair strings (e.g. ``["USDC/USD"]``).
        channels: Channels to subscribe to; defaults to
            ``["book", "trade", "instrument"]``.
        batch_id: Optional ingest batch UUID.
    """
    if channels is None:
        channels = ["book", "trade", "instrument"]

    logger.info("Connecting to Kraken WebSocket v2: %s", _WS_URL)

    while True:
        batch_id = batch_id or new_batch_id()
        buffer: dict[tuple, list] = {}

        try:
            async with websockets.connect(_WS_URL, ping_interval=20) as ws:
                logger.info("Kraken WebSocket connected (batch_id=%s)", batch_id)

                for channel in channels:
                    await ws.send(_subscribe_message(channel, pairs))

                last_flush = time.monotonic()

                async for raw_msg in ws:
                    ts_ns = now_ns()

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message: %s", raw_msg[:200])
                        continue

                    channel_name = msg.get("channel", "unknown")
                    msg_type = msg.get("type", "")  # "snapshot" | "update"

                    # Extract symbol from data array
                    data = msg.get("data", [])
                    symbol = "UNKNOWN"
                    if data and isinstance(data, list) and isinstance(data[0], dict):
                        symbol = data[0].get("symbol", "UNKNOWN")

                    # Checksum validation for book messages
                    if channel_name == "book" and data:
                        for entry in data:
                            checksum = entry.get("checksum")
                            if checksum is not None:
                                computed = _compute_kraken_checksum(entry)
                                if computed != checksum:
                                    logger.warning(
                                        "Checksum mismatch for %s: expected %s, computed %s",
                                        symbol,
                                        checksum,
                                        computed,
                                    )
                                    msg["_checksum_failed"] = True

                    record = make_raw_record(
                        source="kraken",
                        channel=channel_name,
                        symbol=symbol,
                        payload=msg,
                        ts_exchange=str(msg.get("timestamp", "")),
                        batch_id=batch_id,
                    )
                    record["ts_receive_ns"] = ts_ns

                    key = (channel_name, symbol)
                    buffer.setdefault(key, []).append(record)

                    now_mono = time.monotonic()
                    if now_mono - last_flush >= _FLUSH_INTERVAL_SECONDS:
                        _flush_buffer(buffer)
                        buffer = {}
                        last_flush = now_mono

        except (ConnectionClosedError, ConnectionClosedOK) as exc:
            logger.warning("Kraken connection closed: %s. Reconnecting in 5s.", exc)
        except Exception as exc:
            logger.error("Unexpected error: %s. Reconnecting in 5s.", exc)

        if buffer:
            _flush_buffer(buffer)
        buffer = {}
        batch_id = new_batch_id()
        await asyncio.sleep(5)


def _compute_kraken_checksum(book_entry: dict) -> int | None:
    """Compute a simple Kraken-style book checksum for validation.

    Kraken's checksum covers the top 10 bid and ask price/quantity pairs.
    This is a simplified implementation; see Kraken docs for the exact spec.

    Returns:
        Integer checksum, or ``None`` if the entry lacks bid/ask data.
    """
    bids = book_entry.get("bids", [])
    asks = book_entry.get("asks", [])
    if not bids and not asks:
        return None

    def fmt(val: float) -> str:
        return f"{val:.5f}".replace(".", "").lstrip("0") or "0"

    parts = []
    for level in bids[:10]:
        parts.append(fmt(level.get("price", 0)))
        parts.append(fmt(level.get("qty", 0)))
    for level in asks[:10]:
        parts.append(fmt(level.get("price", 0)))
        parts.append(fmt(level.get("qty", 0)))

    import binascii

    raw = "".join(parts).encode("ascii")
    return binascii.crc32(raw) & 0xFFFFFFFF


def _flush_buffer(buffer: dict[tuple, list]) -> None:
    for (channel, symbol), records in buffer.items():
        if not records:
            continue
        ts_ns = records[0]["ts_receive_ns"]
        write_raw_batch(
            records=records,
            venue="kraken",
            channel=channel,
            symbol=symbol,
            date=date_str(ts_ns),
            hour=hour_str(ts_ns),
        )
        logger.debug("Flushed %d records for %s/%s", len(records), channel, symbol)
