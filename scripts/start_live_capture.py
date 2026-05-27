#!/usr/bin/env python3
"""Start live WebSocket data capture to the Bronze layer.

Subscribes to trade, depth, and book-ticker streams from one or more venues
and writes raw JSON payloads to the Bronze Hive-partitioned Parquet store.
Collectors run concurrently and reconnect automatically on disconnection.

Usage:
    # All venues, default symbols
    python scripts/start_live_capture.py

    # Binance only
    python scripts/start_live_capture.py --venues binance

    # Specific symbols per venue
    python scripts/start_live_capture.py \\
        --binance-symbols USDCUSDT BTCUSDT \\
        --coinbase-symbols USDC-USD BTC-USD \\
        --kraken-pairs USDC/USD BTC/USD

    # Custom Bronze root
    python scripts/start_live_capture.py --bronze-root /mnt/data/bronze
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# Default symbols for each venue (matches configs/instruments.yaml)
_DEFAULT_BINANCE_SYMBOLS = ["USDCUSDT", "BTCUSDT", "BTCUSDC", "ETHUSDT", "ETHUSDC"]
_DEFAULT_COINBASE_PRODUCT_IDS = ["USDC-USD", "BTC-USD", "ETH-USD", "USDT-USD"]
_DEFAULT_KRAKEN_PAIRS = ["USDC/USD", "BTC/USD", "ETH/USD"]

_ALL_VENUES = ["binance", "coinbase", "kraken"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live WebSocket capture to Bronze layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--venues",
        nargs="*",
        default=_ALL_VENUES,
        choices=_ALL_VENUES,
        help="Venues to collect from (default: all).",
    )
    parser.add_argument(
        "--binance-symbols",
        nargs="*",
        default=_DEFAULT_BINANCE_SYMBOLS,
        metavar="SYM",
        help="Binance native symbol strings (e.g. USDCUSDT BTCUSDT).",
    )
    parser.add_argument(
        "--coinbase-symbols",
        nargs="*",
        default=_DEFAULT_COINBASE_PRODUCT_IDS,
        metavar="ID",
        help="Coinbase product IDs (e.g. USDC-USD BTC-USD).",
    )
    parser.add_argument(
        "--kraken-pairs",
        nargs="*",
        default=_DEFAULT_KRAKEN_PAIRS,
        metavar="PAIR",
        help="Kraken pair strings (e.g. USDC/USD BTC/USD).",
    )
    parser.add_argument(
        "--bronze-root",
        default="data/bronze",
        help="Root directory for Bronze Parquet output (default: data/bronze).",
    )
    return parser.parse_args()


async def _run_all(args: argparse.Namespace) -> None:
    """Launch one asyncio task per selected venue and wait for all."""
    tasks: list[asyncio.Task] = []  # type: ignore[type-arg]
    venues = set(args.venues or _ALL_VENUES)

    if "binance" in venues and args.binance_symbols:
        from stressbench.ingestion.binance_ws import run_collector as binance_run

        logger.info("Starting Binance collector for: %s", args.binance_symbols)
        tasks.append(
            asyncio.create_task(
                binance_run(symbols=args.binance_symbols),
                name="binance",
            )
        )

    if "coinbase" in venues and args.coinbase_symbols:
        from stressbench.ingestion.coinbase_ws import run_collector as coinbase_run

        logger.info("Starting Coinbase collector for: %s", args.coinbase_symbols)
        tasks.append(
            asyncio.create_task(
                coinbase_run(product_ids=args.coinbase_symbols),
                name="coinbase",
            )
        )

    if "kraken" in venues and args.kraken_pairs:
        from stressbench.ingestion.kraken_ws import run_collector as kraken_run

        logger.info("Starting Kraken collector for: %s", args.kraken_pairs)
        tasks.append(
            asyncio.create_task(
                kraken_run(pairs=args.kraken_pairs),
                name="kraken",
            )
        )

    if not tasks:
        logger.error("No collectors started — check --venues and symbol arguments.")
        return

    logger.info(
        "Live capture running (%d collector(s)). Press Ctrl-C to stop.", len(tasks)
    )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Collectors cancelled; shutting down.")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


def main() -> None:
    args = parse_args()

    bronze_root = Path(args.bronze_root)
    bronze_root.mkdir(parents=True, exist_ok=True)
    logger.info("Bronze root: %s", bronze_root.resolve())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(sig: signal.Signals) -> None:  # noqa: F821
        logger.info("Received %s — cancelling collectors.", sig.name)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    try:
        loop.run_until_complete(_run_all(args))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Live capture stopped.")


if __name__ == "__main__":
    main()
