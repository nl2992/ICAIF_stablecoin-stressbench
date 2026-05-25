#!/usr/bin/env python3
"""Build Silver and Gold feature tables from Bronze raw data.

Pipeline:
    Bronze (raw JSON/Parquet) →
    Silver (normalised trades, books, on-chain) →
    Gold (microstructure features, basis, fragmentation, settlement, labels)

Usage:
    python scripts/build_features.py --start 2024-01-01 --end 2024-01-07
    python scripts/build_features.py --start 2024-01-01 --end 2024-01-07 --skip-silver
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature tables from Bronze data.")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
    )
    parser.add_argument(
        "--bronze-dir",
        default="data/bronze",
        help="Bronze input directory.",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/silver",
        help="Silver output directory.",
    )
    parser.add_argument(
        "--gold-dir",
        default="data/gold",
        help="Gold output directory.",
    )
    parser.add_argument(
        "--skip-silver",
        action="store_true",
        help="Skip Silver normalisation (use existing Silver data).",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip label generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def build_silver(bronze_dir: str, silver_dir: str, start: datetime, end: datetime, dry_run: bool) -> None:
    """Normalise Bronze raw data into Silver Parquet files."""
    logger.info("Building Silver layer: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping Silver build.")
        return

    from stressbench.normalization.normalize_trades import (
        normalize_binance_trades,
        normalize_coinbase_trades,
        normalize_kraken_trades,
    )
    from stressbench.normalization.normalize_books import (
        normalize_binance_depth,
        normalize_coinbase_level2,
        normalize_kraken_book,
    )
    from stressbench.normalization.normalize_onchain import normalize_etherscan_transfers

    Path(silver_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Silver normalisation complete (stub — wire to actual Bronze files).")


def build_gold_features(silver_dir: str, gold_dir: str, start: datetime, end: datetime, dry_run: bool) -> None:
    """Build Gold microstructure, basis, and fragmentation feature tables."""
    logger.info("Building Gold feature tables: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping Gold feature build.")
        return

    Path(gold_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Gold feature build complete (stub — wire to Silver Parquet files).")


def build_labels(gold_dir: str, start: datetime, end: datetime, dry_run: bool) -> None:
    """Generate forward-looking labels from Gold feature tables."""
    logger.info("Building labels: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping label build.")
        return

    from stressbench.labels.basis_labels import add_basis_labels
    from stressbench.labels.regime_labels import add_regime_labels
    from stressbench.labels.recovery_labels import add_recovery_labels

    logger.info("Label build complete (stub — wire to Gold Parquet files).")


def main() -> None:
    args = parse_args()

    if not args.skip_silver:
        build_silver(args.bronze_dir, args.silver_dir, args.start, args.end, args.dry_run)

    build_gold_features(args.silver_dir, args.gold_dir, args.start, args.end, args.dry_run)

    if not args.skip_labels:
        build_labels(args.gold_dir, args.start, args.end, args.dry_run)

    logger.info("Feature build pipeline complete.")


if __name__ == "__main__":
    main()
