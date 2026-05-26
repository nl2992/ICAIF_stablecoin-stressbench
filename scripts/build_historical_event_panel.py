#!/usr/bin/env python3
"""Build historical event panel from dataset.parquet.

For each event in configs/event_windows_historical.yaml:
    - Extracts rows whose timestamps fall within the event window
    - Tags each row with event_id, data_tier, coverage_score
    - Produces a combined panel parquet

Writes:
    results/experiments_addon/historical_event_panel.parquet
    results/paper_addon/table_14_historical_event_catalog.csv
    results/paper_addon/table_15_event_data_coverage.csv

Usage:
    python scripts/build_historical_event_panel.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.history.event_catalog import EVENT_CATALOG, load_event_windows_yaml
from stressbench.history.data_availability import PREDEFINED_COVERAGE

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build historical event panel")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--config", default="configs/event_windows_historical.yaml")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--paper-output-dir", default="results/paper_addon")
    return p.parse_args()


def _iso_to_ns(iso: str) -> int:
    """Convert ISO 8601 UTC string to nanoseconds since epoch."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)


def _build_panel(df: pl.DataFrame, events: dict) -> pl.DataFrame:
    """Extract rows from df that fall within any event window, tagged with event metadata."""
    frames = []

    for event_id, ev in events.items():
        start_ns = _iso_to_ns(ev["start"])
        end_ns = _iso_to_ns(ev["end"])

        if "ts_1m_ns" not in df.columns:
            logger.warning("ts_1m_ns not found in dataset, cannot filter by timestamp")
            break

        mask = (pl.col("ts_1m_ns") >= start_ns) & (pl.col("ts_1m_ns") <= end_ns)
        sub = df.filter(mask)

        if sub.is_empty():
            logger.info("Event %s: no rows found in dataset (ts range [%d, %d])", event_id, start_ns, end_ns)
            continue

        # Tag rows with event metadata
        tier_val = ev["data_tier"]
        tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
        sub = sub.with_columns([
            pl.lit(event_id).alias("event_id"),
            pl.lit(tier_str).alias("data_tier"),
            pl.lit(float(ev["coverage_score"])).alias("coverage_score"),
            pl.lit(", ".join(ev.get("stablecoins", []))).alias("event_stablecoins"),
        ])
        frames.append(sub)
        logger.info("Event %s: found %d rows (tier=%s)", event_id, len(sub), tier_str)

    if not frames:
        logger.warning("No rows found for any event window")
        # Return empty frame with event columns added
        return df.head(0).with_columns([
            pl.lit("").alias("event_id"),
            pl.lit("").alias("data_tier"),
            pl.lit(0.0).alias("coverage_score"),
            pl.lit("").alias("event_stablecoins"),
        ])

    return pl.concat(frames)


def _build_catalog_table(events: dict) -> list[dict]:
    """Build Table 14: historical event catalog summary."""
    rows = []
    for event_id, ev in events.items():
        tier_val = ev["data_tier"]
        tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
        rows.append({
            "event_id": event_id,
            "display_name": ev.get("display_name", event_id),
            "stablecoins": ", ".join(ev.get("stablecoins", [])),
            "start": ev["start"],
            "end": ev["end"],
            "data_tier": tier_str,
            "coverage_score": ev["coverage_score"],
            "peak_depeg_bps": ev.get("peak_depeg_bps", ""),
            "empirical_use": ev.get("empirical_use", ""),
        })
    return rows


def _build_coverage_table(events: dict) -> list[dict]:
    """Build Table 15: event data source coverage matrix."""
    rows = []
    for event_id in events:
        profile = PREDEFINED_COVERAGE.get(event_id)
        if profile is None:
            continue
        rows.append({
            "event_id": event_id,
            "price_data": int(profile.price_data),
            "trade_data": int(profile.trade_data),
            "l2_data": int(profile.l2_data),
            "dex_pool_data": int(profile.dex_pool_data),
            "onchain_data": int(profile.onchain_data),
            "execution_grade_available": int(profile.execution_grade_available),
            "coverage_score": profile.coverage_score,
        })
    return rows


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    config_path = Path(args.config)
    out_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Load events from YAML
    logger.info("Loading event windows from %s", config_path)
    events = load_event_windows_yaml(config_path)
    logger.info("Loaded %d events", len(events))

    # Load dataset
    parquet_path = data_path / "dataset.parquet" if data_path.is_dir() else data_path
    logger.info("Loading dataset from %s", parquet_path)
    df = pl.read_parquet(str(parquet_path))
    logger.info("Dataset shape: %s rows x %s cols", *df.shape)

    # Build panel
    panel = _build_panel(df, events)
    panel_path = out_dir / "historical_event_panel.parquet"
    panel.write_parquet(str(panel_path))
    logger.info("Wrote panel: %d rows to %s", len(panel), panel_path)
    print(f"Historical event panel: {len(panel)} rows -> {panel_path}")

    # Per-event summary
    if len(panel) > 0 and "event_id" in panel.columns:
        summary = panel.group_by("event_id").agg([
            pl.len().alias("n_rows"),
            pl.col("data_tier").first().alias("data_tier"),
            pl.col("coverage_score").first().alias("coverage_score"),
        ]).sort("event_id")
        logger.info("Panel summary:\n%s", summary)

    # Table 14: Event catalog
    catalog_rows = _build_catalog_table(events)
    t14_path = paper_dir / "table_14_historical_event_catalog.csv"
    with open(t14_path, "w", newline="") as fh:
        if catalog_rows:
            writer = csv.DictWriter(fh, fieldnames=list(catalog_rows[0].keys()))
            writer.writeheader()
            writer.writerows(catalog_rows)
    logger.info("Wrote Table 14 (%d events) to %s", len(catalog_rows), t14_path)
    print(f"Table 14: {len(catalog_rows)} events -> {t14_path}")

    # Table 15: Coverage matrix
    coverage_rows = _build_coverage_table(events)
    t15_path = paper_dir / "table_15_event_data_coverage.csv"
    with open(t15_path, "w", newline="") as fh:
        if coverage_rows:
            writer = csv.DictWriter(fh, fieldnames=list(coverage_rows[0].keys()))
            writer.writeheader()
            writer.writerows(coverage_rows)
    logger.info("Wrote Table 15 (%d events) to %s", len(coverage_rows), t15_path)
    print(f"Table 15: {len(coverage_rows)} events -> {t15_path}")


if __name__ == "__main__":
    main()
