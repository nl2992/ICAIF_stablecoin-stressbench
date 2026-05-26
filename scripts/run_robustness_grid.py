#!/usr/bin/env python3
"""Robustness grid: price-to-execution gap across costs, notionals, horizons.

Writes ONLY to results/experiments_addon/. Baseline files are not modified.

Usage:
    python scripts/run_robustness_grid.py
    python scripts/run_robustness_grid.py \
        --data-dir data/gold \
        --output-dir results/experiments_addon \
        --splits test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stressbench.common.logging import get_logger
from stressbench.experiments.robustness import compute_robustness_grid

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness grid for price-to-execution gap.")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--splits", nargs="+", default=["test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data_dir) / "dataset.parquet"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found at {dataset_path}")

    logger.info("Computing robustness grid for splits=%s …", args.splits)
    rows = compute_robustness_grid(dataset_path, splits=args.splits)
    logger.info("Grid: %d rows", len(rows))

    out_path = output_dir / "robustness_price_execution_gap.csv"
    if rows:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s", out_path)
    else:
        logger.warning("No rows produced — check dataset and column availability.")


if __name__ == "__main__":
    main()
