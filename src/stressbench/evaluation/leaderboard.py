"""Leaderboard builder for the Stablecoin StressBench benchmark.

Aggregates backtest results across models and tasks into a ranked leaderboard.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def build_leaderboard(results: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a leaderboard DataFrame from a list of backtest result dicts.

    Args:
        results: List of dicts as returned by
            :func:`~stressbench.evaluation.backtest.run_backtest`.

    Returns:
        :class:`polars.DataFrame` with one row per model, sorted by
        ``net_bps_captured`` descending.
    """
    rows = []
    for r in results:
        ml = r.get("ml_metrics", {})
        econ = r.get("economic_metrics", {})
        row = {
            "model": r.get("model", "unknown"),
            "task": r.get("task", "unknown"),
            # ML metrics
            "auroc": ml.get("auroc"),
            "auprc": ml.get("auprc"),
            "f1": ml.get("f1"),
            "balanced_accuracy": ml.get("balanced_accuracy"),
            "brier_score": ml.get("brier_score"),
            "mae": ml.get("mae"),
            "rmse": ml.get("rmse"),
            "directional_accuracy": ml.get("directional_accuracy"),
            "spearman_rho": ml.get("spearman_rho"),
            # Economic metrics
            "net_bps_captured": econ.get("net_bps_captured"),
            "hit_rate_above_cost": econ.get("hit_rate_above_cost"),
            "false_positive_cost": econ.get("false_positive_cost"),
            "n_trades": econ.get("n_trades"),
            "final_pnl_usd": econ.get("final_pnl_usd"),
            "max_drawdown_usd": econ.get("max_drawdown_usd"),
            "sharpe_ratio": econ.get("sharpe_ratio"),
        }
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    # Sort by net_bps_captured descending (primary), then AUROC descending
    sort_cols = []
    if "net_bps_captured" in df.columns:
        sort_cols.append(pl.col("net_bps_captured").descending())
    if "auroc" in df.columns:
        sort_cols.append(pl.col("auroc").descending())

    if sort_cols:
        df = df.sort(sort_cols)

    return df


def print_leaderboard(df: pl.DataFrame) -> None:
    """Print the leaderboard to stdout in a readable format."""
    if df.is_empty():
        logger.info("Leaderboard is empty.")
        return
    print("\n=== Stablecoin StressBench Leaderboard ===")
    print(df.to_pandas().to_string(index=False))
    print("=" * 50)
