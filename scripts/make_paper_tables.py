#!/usr/bin/env python3
"""Generate paper-ready LaTeX-friendly CSV tables from committed benchmark outputs.

Reads:
    data/gold/dataset.parquet             — Gold feature + label dataset
    results/smoke/pipeline_row_counts.csv — Pipeline coverage stats
    results/experiments/all_results.csv   — Full experiment grid results (optional)

Writes to results/paper/:
    table_1_data_coverage.csv             — Dataset split coverage
    table_2_price_execution_gap.csv       — Price signal vs executable profit gap
    table_3_model_ablation.csv            — Model × feature-set performance grid
    table_4_oracle_gap.csv                — Oracle upper bound vs best model

Usage:
    python scripts/make_paper_tables.py
    python scripts/make_paper_tables.py --data-dir data/gold --output-dir results/paper
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_NOTIONALS = [10_000, 50_000, 100_000, 500_000]
_THRESHOLDS_BPS = [0, 5, 10, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper tables from benchmark outputs."
    )
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--smoke-dir", default="results/smoke")
    parser.add_argument("--experiments-dir", default="results/experiments")
    parser.add_argument("--output-dir", default="results/paper")
    return parser.parse_args()


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning("No rows for %s — skipping.", path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def make_table_1_data_coverage(
    dataset_path: Path,
    smoke_dir: Path,
    output_path: Path,
) -> None:
    """Table 1: Dataset split coverage (rows, dates, event windows)."""
    import polars as pl

    rows: list[dict] = []

    # Dataset split row counts
    if dataset_path.exists():
        df = pl.read_parquet(str(dataset_path))
        split_counts = df["split"].value_counts().sort("split")
        for row in split_counts.iter_rows(named=True):
            rows.append(
                {
                    "split": row["split"],
                    "rows": row["count"],
                    "source": "dataset.parquet",
                    "notes": "",
                }
            )
        rows.append(
            {
                "split": "total",
                "rows": len(df),
                "source": "dataset.parquet",
                "notes": f"{len(df.columns)} columns ({sum(1 for c in df.columns if c.startswith('label_'))} label cols)",
            }
        )
    else:
        logger.warning("dataset.parquet not found at %s", dataset_path)

    # Pipeline stage row counts from smoke
    smoke_counts = smoke_dir / "pipeline_row_counts.csv"
    if smoke_counts.exists():
        with open(smoke_counts) as fh:
            for line in csv.DictReader(fh):
                rows.append(
                    {
                        "split": f"{line['stage']}.{line['table']}",
                        "rows": line["rows"],
                        "source": "pipeline_row_counts.csv",
                        "notes": line.get("notes", ""),
                    }
                )

    _write_csv(rows, output_path)


def make_table_2_price_execution_gap(
    dataset_path: Path,
    output_path: Path,
) -> None:
    """Table 2: Price-signal prevalence vs executable profit prevalence.

    The benchmark's central empirical claim: the fraction of minutes where a
    price basis signal fires is much larger than the fraction where net profit
    after VWAP cost, fees, and market impact is actually positive.
    """
    import polars as pl

    if not dataset_path.exists():
        logger.warning("dataset.parquet not found — skipping Table 2.")
        return

    df = pl.read_parquet(str(dataset_path))
    rows: list[dict] = []

    # Choose basis columns to report (use all available)
    basis_cols = [
        c
        for c in [
            "cross_quote_basis_bps",
            "cross_quote_basis_usdc_bps",
            "cross_quote_basis_usdt_bps",
        ]
        if c in df.columns
    ]

    for split in ("train", "validation", "test"):
        sdf = df.filter(pl.col("split") == split)
        n = len(sdf)
        if n == 0:
            continue

        for thr in _THRESHOLDS_BPS:
            row: dict = {"split": split, "threshold_bps": thr, "n_minutes": n}

            # Price signal: |basis| > threshold
            for col in basis_cols:
                pct = sdf.filter(pl.col(col).abs() > thr).height / n * 100
                short = col.replace("cross_quote_basis_", "").replace("_bps", "")
                row[f"price_pct_{short}"] = round(pct, 2)

            # Executable signal: net_profit_bps_q{N} > threshold
            for q in _NOTIONALS:
                col = f"net_profit_bps_q{q}"
                if col in sdf.columns:
                    valid = sdf.filter(
                        pl.col(col).is_not_null() & ~pl.col(col).is_nan()
                    )
                    n_valid = len(valid)
                    pct = (
                        valid.filter(pl.col(col) > thr).height / n * 100
                        if n_valid > 0
                        else float("nan")
                    )
                    row[f"exec_pct_q{q}"] = round(pct, 2)

            rows.append(row)

    _write_csv(rows, output_path)


def make_table_3_model_ablation(
    experiments_dir: Path,
    output_path: Path,
    task_filter: list[str] | None = None,
) -> None:
    """Table 3: Model × feature-set ablation from the experiment grid.

    Selects key columns for the paper and sorts by task → net_bps_captured desc.
    """
    all_results = experiments_dir / "all_results.csv"
    if not all_results.exists():
        logger.warning(
            "all_results.csv not found at %s — run scripts/run_experiments.py first.",
            experiments_dir,
        )
        return

    with open(all_results) as fh:
        all_rows = list(csv.DictReader(fh))

    if task_filter:
        all_rows = [r for r in all_rows if r.get("task") in task_filter]

    _PAPER_COLS = [
        "task",
        "feature_set",
        "model",
        "n_train",
        "n_val",
        "n_test",
        "validation_threshold",
        "validation_n_trades",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "brier_score",
        "net_bps_captured",
        "hit_rate_above_cost",
        "n_trades",
        "final_pnl_usd",
        "sharpe_ratio",
    ]

    out_rows = []
    for r in all_rows:
        out_row = {col: r.get(col, "") for col in _PAPER_COLS}
        out_rows.append(out_row)

    # Sort: task → net_bps_captured descending
    def sort_key(r: dict) -> tuple:
        try:
            net = float(r.get("net_bps_captured") or "nan")
        except ValueError:
            net = float("nan")
        return (r.get("task", ""), -net if net == net else float("inf"))

    out_rows.sort(key=sort_key)
    _write_csv(out_rows, output_path)


def make_table_4_oracle_gap(
    experiments_dir: Path,
    output_path: Path,
) -> None:
    """Table 4: Oracle upper bound vs best ML model gap.

    Shows how much of the theoretical maximum is captured by ML models.
    """
    all_results = experiments_dir / "all_results.csv"
    if not all_results.exists():
        logger.warning(
            "all_results.csv not found at %s — run scripts/run_experiments.py first.",
            experiments_dir,
        )
        return

    with open(all_results) as fh:
        all_rows = list(csv.DictReader(fh))

    rows: list[dict] = []
    tasks = sorted({r["task"] for r in all_rows})

    for task in tasks:
        task_rows = [r for r in all_rows if r["task"] == task]
        oracle_rows = [r for r in task_rows if r["model"] == "oracle"]
        non_oracle = [r for r in task_rows if r["model"] != "oracle"]

        def _net(r: dict) -> float:
            try:
                return float(r.get("net_bps_captured") or "nan")
            except ValueError:
                return float("nan")

        oracle_net = max((_net(r) for r in oracle_rows), default=float("nan"))
        # Find best ML model by net_bps, treating NaN as -inf (model made no trades)
        non_oracle_valid = [
            (r, _net(r)) for r in non_oracle if not (_n := _net(r)) != _n or True
        ]
        non_oracle_valid = [(r, v) for r, v in non_oracle_valid if v == v]  # drop NaN
        if non_oracle_valid:
            best_r, best_net = max(non_oracle_valid, key=lambda x: x[1])
            best_model = f"{best_r['model']}@{best_r['feature_set']}"
        else:
            best_net, best_model = float("nan"), "—"
        capture = (
            (best_net / oracle_net * 100)
            if oracle_net > 0 and best_net == best_net
            else float("nan")
        )

        rows.append(
            {
                "task": task,
                "oracle_net_bps": (
                    round(oracle_net, 2) if oracle_net == oracle_net else ""
                ),
                "best_model": best_model,
                "best_model_net_bps": (
                    round(best_net, 2) if best_net == best_net else ""
                ),
                "capture_pct": round(capture, 1) if capture == capture else "",
            }
        )

    _write_csv(rows, output_path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    smoke_dir = Path(args.smoke_dir)
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "dataset.parquet"

    logger.info("Generating paper tables → %s", output_dir)

    make_table_1_data_coverage(
        dataset_path=dataset_path,
        smoke_dir=smoke_dir,
        output_path=output_dir / "table_1_data_coverage.csv",
    )
    make_table_2_price_execution_gap(
        dataset_path=dataset_path,
        output_path=output_dir / "table_2_price_execution_gap.csv",
    )
    make_table_3_model_ablation(
        experiments_dir=experiments_dir,
        output_path=output_dir / "table_3_model_ablation.csv",
    )
    make_table_4_oracle_gap(
        experiments_dir=experiments_dir,
        output_path=output_dir / "table_4_oracle_gap.csv",
    )

    logger.info("Paper tables complete.")


if __name__ == "__main__":
    main()
