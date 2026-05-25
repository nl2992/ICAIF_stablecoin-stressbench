"""Bronze manifest: track coverage and row counts for each partition.

The manifest is a Parquet file stored at ``data/bronze/_manifest.parquet``.
Each row represents one Parquet part-file and records its venue, channel,
symbol, date, hour, row count, and file path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from stressbench.common.config import bronze_root
from stressbench.common.logging import get_logger
from stressbench.common.time import now_utc

logger = get_logger(__name__)

_MANIFEST_FILE = "_manifest.parquet"


def _manifest_path(root: Path) -> Path:
    return root / _MANIFEST_FILE


def append_manifest_entry(
    file: Path,
    venue: str,
    channel: str,
    symbol: str,
    date: str,
    hour: str,
    row_count: int,
    root: Path | None = None,
) -> None:
    """Append a single entry to the Bronze manifest.

    Args:
        file: Path to the Parquet part-file that was written.
        venue: Venue partition key.
        channel: Channel partition key.
        symbol: Symbol partition key.
        date: Date partition key (``YYYY-MM-DD``).
        hour: Hour partition key (``HH``).
        row_count: Number of rows written to ``file``.
        root: Bronze root directory override.
    """
    root = root or bronze_root()
    manifest_path = _manifest_path(root)

    entry: dict[str, Any] = {
        "file": str(file),
        "venue": venue,
        "channel": channel,
        "symbol": symbol,
        "date": date,
        "hour": hour,
        "row_count": row_count,
        "written_at": now_utc().isoformat(),
    }
    new_row = pl.DataFrame([entry])

    if manifest_path.exists():
        existing = pl.read_parquet(manifest_path)
        updated = pl.concat([existing, new_row])
    else:
        updated = new_row

    updated.write_parquet(manifest_path)
    logger.debug("Manifest updated: %s", entry)


def read_manifest(root: Path | None = None) -> pl.DataFrame:
    """Read the Bronze manifest as a Polars DataFrame.

    Returns an empty DataFrame with the correct schema if the manifest does
    not yet exist.
    """
    root = root or bronze_root()
    manifest_path = _manifest_path(root)
    if not manifest_path.exists():
        return pl.DataFrame(
            schema={
                "file": pl.Utf8,
                "venue": pl.Utf8,
                "channel": pl.Utf8,
                "symbol": pl.Utf8,
                "date": pl.Utf8,
                "hour": pl.Utf8,
                "row_count": pl.Int64,
                "written_at": pl.Utf8,
            }
        )
    return pl.read_parquet(manifest_path)


def coverage_report(root: Path | None = None) -> pl.DataFrame:
    """Return a summary of Bronze coverage grouped by venue, channel, symbol, and date."""
    manifest = read_manifest(root)
    if manifest.is_empty():
        return manifest
    return (
        manifest.group_by(["venue", "channel", "symbol", "date"])
        .agg(
            pl.col("row_count").sum().alias("total_rows"),
            pl.col("file").count().alias("part_files"),
        )
        .sort(["venue", "channel", "symbol", "date"])
    )
