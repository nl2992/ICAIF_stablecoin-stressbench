"""Coinbase Exchange live WebSocket collector.

Subscribes to ``heartbeat``, ``level2``, and ``matches`` channels.
Uses sequence numbers from heartbeat messages to detect missed messages.

Reference:
    https://docs.cdp.coinbase.com/exchange/websocket-feed/channels
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

_WS_URL = "wss://advanced-trade-ws.coinbase.com"
_FLUSH_INTERVAL_SECONDS = 30


def _subscribe_message(product_ids: Sequence[str], channels: Sequence[str]) -> str:
    return json.dumps(
        {
            "type": "subscribe",
            "product_ids": list(product_ids),
            "channel": channels[0] if len(channels) == 1 else "subscriptions",
            "channels": list(channels),
        }
    )


async def run_collector(
    product_ids: Sequence[str],
    channels: Sequence[str] | None = None,
    batch_id: str | None = None,
) -> None:
    """Run the Coinbase live collector indefinitely, reconnecting on errors.

    Args:
        product_ids: List of Coinbase product IDs (e.g. ``["USDC-USD"]``).
        channels: Channels to subscribe to; defaults to
            ``["heartbeat", "level2", "matches"]``.
        batch_id: Optional ingest batch UUID.
    """
    if channels is None:
        channels = ["heartbeat", "level2", "matches"]

    logger.info("Connecting to Coinbase WebSocket: %s", _WS_URL)

    # Per-product sequence tracking for gap detection
    last_sequence: dict[str, int] = {}

    while True:
        batch_id = batch_id or new_batch_id()
        buffer: dict[tuple, list] = {}

        try:
            async with websockets.connect(_WS_URL, ping_interval=20) as ws:
                logger.info("Coinbase WebSocket connected (batch_id=%s)", batch_id)

                # Subscribe to each channel
                for channel in channels:
                    sub_msg = json.dumps(
                        {
                            "type": "subscribe",
                            "product_ids": list(product_ids),
                            "channel": channel,
                        }
                    )
                    await ws.send(sub_msg)

                last_flush = time.monotonic()

                async for raw_msg in ws:
                    ts_ns = now_ns()

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON message: %s", raw_msg[:200])
                        continue

                    msg_type = msg.get("type", "unknown")
                    channel_name = msg.get("channel", msg_type)
                    product_id = msg.get("product_id", msg.get("product_ids", ["UNKNOWN"]))
                    if isinstance(product_id, list):
                        product_id = product_id[0] if product_id else "UNKNOWN"

                    # Sequence gap detection for heartbeat messages
                    if msg_type == "heartbeat":
                        seq = msg.get("sequence", 0)
                        if product_id in last_sequence:
                            gap = seq - last_sequence[product_id] - 1
                            if gap > 0:
                                logger.warning(
                                    "Sequence gap detected for %s: expected %d, got %d (gap=%d)",
                                    product_id,
                                    last_sequence[product_id] + 1,
                                    seq,
                                    gap,
                                )
                                msg["_sequence_gap"] = True
                        last_sequence[product_id] = seq

                    record = make_raw_record(
                        source="coinbase",
                        channel=channel_name,
                        symbol=product_id,
                        payload=msg,
                        ts_exchange=str(msg.get("time", "")),
                        batch_id=batch_id,
                    )
                    record["ts_receive_ns"] = ts_ns

                    key = (channel_name, product_id)
                    buffer.setdefault(key, []).append(record)

                    now_mono = time.monotonic()
                    if now_mono - last_flush >= _FLUSH_INTERVAL_SECONDS:
                        _flush_buffer(buffer)
                        buffer = {}
                        last_flush = now_mono

        except (ConnectionClosedError, ConnectionClosedOK) as exc:
            logger.warning("Coinbase connection closed: %s. Reconnecting in 5s.", exc)
        except Exception as exc:
            logger.error("Unexpected error: %s. Reconnecting in 5s.", exc)

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
            venue="coinbase",
            channel=channel,
            symbol=symbol,
            date=date_str(ts_ns),
            hour=hour_str(ts_ns),
        )
        logger.debug("Flushed %d records for %s/%s", len(records), channel, symbol)
