#!/usr/bin/env python3
"""Pull raw market data from all configured venues and write to Bronze layer.

Usage:
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07 --venues binance coinbase
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07 --mode archive
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

from stressbench.common.config import load_config
from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull raw market data to Bronze layer.")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        help="Start datetime (ISO 8601, UTC). Example: 2024-01-01",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        help="End datetime (ISO 8601, UTC). Example: 2024-01-07",
    )
    parser.add_argument(
        "--venues",
        nargs="*",
        default=None,
        help="Venues to pull (default: all configured venues).",
    )
    parser.add_argument(
        "--mode",
        choices=["archive", "live", "tardis"],
        default="archive",
        help="Data source mode (default: archive).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/bronze",
        help="Bronze output directory (default: data/bronze).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pulled without actually pulling.",
    )
    return parser.parse_args()


def pull_binance_archive(start: datetime, end: datetime, output_dir: str, dry_run: bool) -> None:
    from stressbench.ingestion.binance_archive import BinanceArchiveLoader
    loader = BinanceArchiveLoader(output_dir=output_dir)
    if dry_run:
        logger.info("[DRY RUN] Would pull Binance archive: %s → %s", start.date(), end.date())
        return
    loader.pull_range(start, end)


def pull_coinbase(start: datetime, end: datetime, output_dir: str, dry_run: bool) -> None:
    logger.info("Coinbase archive pull not yet implemented; use Tardis mode for historical data.")


def pull_kraken(start: datetime, end: datetime, output_dir: str, dry_run: bool) -> None:
    logger.info("Kraken archive pull not yet implemented; use Tardis mode for historical data.")


def pull_tardis(start: datetime, end: datetime, venues: list[str], output_dir: str, dry_run: bool) -> None:
    from stressbench.ingestion.tardis_loader import TardisLoader
    loader = TardisLoader(output_dir=output_dir)
    if dry_run:
        logger.info("[DRY RUN] Would pull Tardis: venues=%s, %s → %s", venues, start.date(), end.date())
        return
    for venue in venues:
        loader.pull_range(venue=venue, start=start, end=end)


def pull_etherscan(start: datetime, end: datetime, output_dir: str, dry_run: bool) -> None:
    from stressbench.ingestion.etherscan_loader import EtherscanLoader
    from stressbench.common.config import load_token_addresses
    loader = EtherscanLoader(output_dir=output_dir)
    tokens = load_token_addresses()
    if dry_run:
        logger.info("[DRY RUN] Would pull Etherscan: tokens=%s", list(tokens.keys()))
        return
    for symbol, addresses in tokens.items():
        for chain, address in addresses.items():
            if chain == "ethereum":
                loader.pull_transfers(
                    token_address=address,
                    token_symbol=symbol,
                    start=start,
                    end=end,
                )


def main() -> None:
    args = parse_args()
    cfg = load_config()

    all_venues = list(cfg.get("venues", {}).keys())
    venues = args.venues or all_venues

    logger.info(
        "Pulling data: mode=%s, venues=%s, start=%s, end=%s",
        args.mode, venues, args.start.date(), args.end.date(),
    )

    if args.mode == "archive":
        if "binance" in venues:
            pull_binance_archive(args.start, args.end, args.output_dir, args.dry_run)
        if "coinbase" in venues:
            pull_coinbase(args.start, args.end, args.output_dir, args.dry_run)
        if "kraken" in venues:
            pull_kraken(args.start, args.end, args.output_dir, args.dry_run)

    elif args.mode == "tardis":
        pull_tardis(args.start, args.end, venues, args.output_dir, args.dry_run)

    elif args.mode == "live":
        logger.warning("Live mode is for development only; use archive or tardis for benchmark data.")

    # Always pull on-chain data
    pull_etherscan(args.start, args.end, args.output_dir, args.dry_run)

    logger.info("Data pull complete.")


if __name__ == "__main__":
    main()
