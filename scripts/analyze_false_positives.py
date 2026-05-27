#!/usr/bin/env python3
"""False-positive diagnosis for the benchmark's primary task.

Uses the price_threshold_10bps rule baseline (deterministic from data)
to classify each test row as TP/FP/FN/TN, then computes average feature
profiles for each group to explain why models trade bad windows.

Writes ONLY to results/paper_addon/. Baseline files are not modified.

Usage:
    python scripts/analyze_false_positives.py
    python scripts/analyze_false_positives.py \
        --data-dir data/gold \
        --output-dir results/paper_addon
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_PRIMARY_TASK_LABEL = "label_basis_usdc_1m_gt10bps"
_BASIS_COL = "cross_quote_basis_usdc_bps"
_THRESHOLD_BPS = 10.0

_FEATURE_COLS_FOR_DIAGNOSIS = [
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
    "mid_dispersion_bps_mean",
    "num_active_venues_mean",
    "net_profit_bps_q10000",
    "net_profit_bps_q50000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="False-positive feature profiling.")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/paper_addon")
    p.add_argument("--split", default="test")
    return p.parse_args()


def _mean_or_nan(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def diagnose(dataset_path: Path, split: str, output_dir: Path) -> None:
    df = pl.read_parquet(str(dataset_path))
    sdf = df.filter(pl.col("split") == split)
    n = len(sdf)
    logger.info("Split=%s  n=%d", split, n)

    if _PRIMARY_TASK_LABEL not in sdf.columns:
        logger.error("Label column %s missing — cannot diagnose.", _PRIMARY_TASK_LABEL)
        return
    if _BASIS_COL not in sdf.columns:
        logger.error("Basis column %s missing.", _BASIS_COL)
        return

    basis = sdf[_BASIS_COL].to_numpy().astype(float)
    true_label = sdf[_PRIMARY_TASK_LABEL].to_numpy().astype(float)

    # Price threshold rule: predict positive when |basis| > threshold
    predicted = (np.abs(np.nan_to_num(basis, nan=0.0)) > _THRESHOLD_BPS).astype(np.int8)

    # Available feature columns
    feat_cols = [c for c in _FEATURE_COLS_FOR_DIAGNOSIS if c in sdf.columns]

    groups = {
        "TP": (predicted == 1) & (true_label == 1),
        "FP": (predicted == 1) & (true_label == 0),
        "FN": (predicted == 0) & (true_label == 1),
        "TN": (predicted == 0) & (true_label == 0),
    }

    rows: list[dict] = []
    for group_name, mask in groups.items():
        n_group = int(mask.sum())
        row: dict = {
            "model": "price_threshold_10bps",
            "feature_set": "price_only",
            "task": _PRIMARY_TASK_LABEL,
            "group": group_name,
            "n": n_group,
        }
        for col in feat_cols:
            vals = sdf[col].to_numpy().astype(float)
            row[f"avg_{col}"] = (
                round(_mean_or_nan(vals[mask]), 4) if n_group > 0 else ""
            )
        rows.append(row)
        logger.info("  %s: n=%d", group_name, n_group)

    # Summary: false_positive_cost (mean net_profit in FP group)
    fp_mask = groups["FP"]
    if fp_mask.any() and "net_profit_bps_q10000" in sdf.columns:
        fp_net = sdf["net_profit_bps_q10000"].to_numpy().astype(float)[fp_mask]
        logger.info("  FP mean net_profit_q10000: %.2f bps", _mean_or_nan(fp_net))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "table_5_false_positive_diagnosis.csv"
    if rows:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s", out_path)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data_dir) / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found at {dataset_path}")
    diagnose(dataset_path, args.split, Path(args.output_dir))


if __name__ == "__main__":
    main()
