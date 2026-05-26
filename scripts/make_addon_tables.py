#!/usr/bin/env python3
"""Generate add-on paper tables from addon experiment outputs.

Reads from:
    results/experiments_addon/                  — add-on experiment results
    results/paper/table_4_oracle_gap.csv        — baseline oracle reference (read-only)
    data/gold/dataset.parquet                   — for waterfall computation

Writes to results/paper_addon/ ONLY. Baseline files in results/paper/ are
never modified.

Tables produced:
    table_8_robustness_summary.csv      — price-to-execution gap across parameter grid
    table_9_threshold_ablation.csv      — threshold calibration comparison (from baseline)
    table_10_expected_net_profit.csv    — ExpectedNetProfitRegressor vs baseline best model
    table_5_false_positive_diagnosis.csv— already written by analyze_false_positives.py;
                                          this script copies/derives summary

Usage:
    python scripts/make_addon_tables.py
    python scripts/make_addon_tables.py \
        --data-dir data/gold \
        --baseline-results-dir results/paper \
        --addon-results-dir results/experiments_addon \
        --output-dir results/paper_addon
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import polars as pl
import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate add-on paper tables.")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--baseline-results-dir", default="results/paper")
    p.add_argument("--addon-results-dir", default="results/experiments_addon")
    p.add_argument("--output-dir", default="results/paper_addon")
    return p.parse_args()


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning("No rows for %s", path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Table 8: Robustness summary (pivot of robustness grid)
# ---------------------------------------------------------------------------

def make_table_8_robustness_summary(addon_dir: Path, output_path: Path) -> None:
    """Selected rows from the robustness grid at key parameter combinations."""
    grid_path = addon_dir / "robustness_price_execution_gap.csv"
    if not grid_path.exists():
        logger.warning("robustness grid not found — run run_robustness_grid.py first.")
        return

    with open(grid_path) as fh:
        all_rows = list(csv.DictReader(fh))

    # Summary: base_fee, settlement=0, threshold=10bps, across notionals and horizons
    summary = [
        r for r in all_rows
        if r["fee_regime"] == "base_fee"
        and r["settlement_penalty_bps"] == "0"
        and r["basis_threshold_bps"] == "10"
    ]
    _write_csv(summary, output_path)


# ---------------------------------------------------------------------------
# Table 9: Threshold ablation — compare rules on baseline best model
# ---------------------------------------------------------------------------

def make_table_9_threshold_ablation(baseline_dir: Path, dataset_path: Path, output_path: Path) -> None:
    """Show that negative test result is robust across threshold-selection rules."""
    baseline_path = baseline_dir / "table_3_model_ablation.csv"
    if not baseline_path.exists():
        logger.warning("table_3_model_ablation.csv not found — skipping Table 9.")
        return

    with open(baseline_path) as fh:
        all_rows = list(csv.DictReader(fh))

    # Focus on best non-oracle models for the primary task
    task = "basis_usdc_1m_gt10bps"
    task_rows = [r for r in all_rows if r["task"] == task and r["model"] != "oracle"]

    if not task_rows:
        logger.warning("No rows for task %s in table_3.", task)
        return

    out_rows: list[dict] = []
    for r in task_rows:
        # Each row already has validation_threshold from the total_pnl rule
        out_rows.append({
            "task": r["task"],
            "feature_set": r["feature_set"],
            "model": r["model"],
            "threshold_rule": "validation_total_pnl_min25trades",
            "threshold_value": r.get("validation_threshold", ""),
            "val_n_trades": r.get("validation_n_trades", ""),
            "test_n_trades": r.get("n_trades", ""),
            "test_net_bps": r.get("net_bps_captured", ""),
            "test_final_pnl_usd": r.get("final_pnl_usd", ""),
        })

    _write_csv(out_rows, output_path)


# ---------------------------------------------------------------------------
# Table 10: ExpectedNetProfitRegressor vs baseline
# ---------------------------------------------------------------------------

def make_table_10_expected_net_profit(
    addon_dir: Path,
    baseline_dir: Path,
    output_path: Path,
) -> None:
    """Compare add-on model against best non-oracle from baseline."""
    addon_path = addon_dir / "expected_net_profit_results.csv"
    baseline_path = baseline_dir / "table_4_oracle_gap.csv"

    rows: list[dict] = []

    # Baseline reference
    if baseline_path.exists():
        with open(baseline_path) as fh:
            for r in csv.DictReader(fh):
                if r["task"] in ("executable_arb_q10000_5m", "basis_usdc_1m_gt10bps"):
                    rows.append({
                        "source": "baseline",
                        "model": r["best_model"],
                        "feature_set": "—",
                        "task": r["task"],
                        "calibrated_threshold_bps": "—",
                        "test_n_trades": "—",
                        "test_net_bps_captured": r.get("best_model_net_bps", ""),
                        "oracle_net_bps": r.get("oracle_net_bps", ""),
                        "oracle_capture_pct": r.get("capture_pct", ""),
                    })

    # Add-on results
    if addon_path.exists():
        with open(addon_path) as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "source": "addon",
                    "model": r["model"],
                    "feature_set": r["feature_set"],
                    "task": f"net_profit_regressor/{r['target_col']}",
                    "calibrated_threshold_bps": r.get("calibrated_threshold_bps", ""),
                    "test_n_trades": r.get("test_n_trades", ""),
                    "test_net_bps_captured": r.get("test_net_bps_captured", ""),
                    "oracle_net_bps": "—",
                    "oracle_capture_pct": r.get("test_oracle_capture_pct", ""),
                })
    else:
        logger.warning("expected_net_profit_results.csv not found — run run_addon_experiments.py first.")

    _write_csv(rows, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    addon_dir = Path(args.addon_results_dir)
    baseline_dir = Path(args.baseline_results_dir)
    dataset_path = Path(args.data_dir) / "dataset.parquet"

    logger.info("Generating add-on paper tables → %s", output_dir)

    make_table_8_robustness_summary(
        addon_dir=addon_dir,
        output_path=output_dir / "table_8_robustness_summary.csv",
    )
    make_table_9_threshold_ablation(
        baseline_dir=baseline_dir,
        dataset_path=dataset_path,
        output_path=output_dir / "table_9_threshold_ablation.csv",
    )
    make_table_10_expected_net_profit(
        addon_dir=addon_dir,
        baseline_dir=baseline_dir,
        output_path=output_dir / "table_10_expected_net_profit.csv",
    )

    logger.info("Add-on tables complete.")


if __name__ == "__main__":
    main()
