#!/usr/bin/env python3
"""End-to-end benchmark pipeline orchestration.

Runs: pull_data → build_features → train_models → evaluate_models

Usage:
    python scripts/run_pipeline.py --start 2024-01-01 --end 2024-01-07
    python scripts/run_pipeline.py --start 2024-01-01 --end 2024-01-07 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full benchmark pipeline.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--mode", default="archive", choices=["archive", "tardis"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models/trained")
    parser.add_argument("--output", default="leaderboard.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["pull", "build", "train", "evaluate"],
        help="Pipeline stages to skip.",
    )
    return parser.parse_args()


def run_stage(cmd: list[str], stage: str, dry_run: bool) -> None:
    if dry_run:
        logger.info("[DRY RUN] Would run: %s", " ".join(cmd))
        return
    logger.info("Running stage: %s", stage)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Stage '%s' failed with exit code %d", stage, result.returncode)
        sys.exit(result.returncode)
    logger.info("Stage '%s' complete.", stage)


def main() -> None:
    args = parse_args()
    py = sys.executable

    if "pull" not in args.skip:
        run_stage(
            [
                py, "scripts/pull_data.py",
                "--start", args.start,
                "--end", args.end,
                "--mode", args.mode,
                "--output-dir", f"{args.data_dir}/bronze",
            ] + (["--dry-run"] if args.dry_run else []),
            stage="pull_data",
            dry_run=False,  # Already handled inside pull_data.py
        )

    if "build" not in args.skip:
        run_stage(
            [
                py, "scripts/build_features.py",
                "--start", args.start,
                "--end", args.end,
                "--bronze-dir", f"{args.data_dir}/bronze",
                "--silver-dir", f"{args.data_dir}/silver",
                "--gold-dir", f"{args.data_dir}/gold",
            ] + (["--dry-run"] if args.dry_run else []),
            stage="build_features",
            dry_run=False,
        )

    if "train" not in args.skip:
        run_stage(
            [
                py, "scripts/train_models.py",
                "--data-dir", f"{args.data_dir}/gold",
                "--model-dir", args.model_dir,
            ],
            stage="train_models",
            dry_run=args.dry_run,
        )

    if "evaluate" not in args.skip:
        run_stage(
            [
                py, "scripts/evaluate_models.py",
                "--data-dir", f"{args.data_dir}/gold",
                "--model-dir", args.model_dir,
                "--output", args.output,
            ],
            stage="evaluate_models",
            dry_run=args.dry_run,
        )

    logger.info("Pipeline complete. Leaderboard: %s", args.output)


if __name__ == "__main__":
    main()
