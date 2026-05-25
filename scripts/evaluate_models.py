#!/usr/bin/env python3
"""Evaluate trained models on the test split and produce the leaderboard.

Usage:
    python scripts/evaluate_models.py --data-dir data/gold --model-dir models/trained
    python scripts/evaluate_models.py --data-dir data/gold --model-dir models/trained --output leaderboard.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from stressbench.common.logging import get_logger
from stressbench.evaluation.backtest import run_backtest
from stressbench.evaluation.leaderboard import build_leaderboard, print_leaderboard

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark models.")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--model-dir", default="models/trained")
    parser.add_argument("--output", default="leaderboard.csv")
    parser.add_argument(
        "--notional-usd",
        type=float,
        default=50_000.0,
        help="Notional size per trade in USD for economic metrics.",
    )
    return parser.parse_args()


def load_test_data(data_dir: str, label_col: str, feature_cols: list[str] | None):
    """Load test data from Gold Parquet files."""
    import polars as pl

    gold_path = Path(data_dir)
    parquet_files = list(gold_path.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning("No Parquet files found; generating synthetic test data.")
        rng = np.random.default_rng(99)
        n = 2_000
        X = rng.standard_normal((n, 20)).astype(np.float32)
        y = (rng.standard_normal(n) > 0).astype(np.int8)
        y_net = rng.normal(0, 15, n)
        return X, y, y_net

    df = pl.read_parquet(str(gold_path / "*.parquet"))
    test_df = df.filter(pl.col("split") == "test")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if not c.startswith("label_") and c != "split"]

    X = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y = test_df[label_col].to_numpy()

    net_col = "net_profit_bps_q50000"
    y_net = test_df[net_col].to_numpy() if net_col in test_df.columns else np.zeros(len(y))
    return X, y, y_net


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    # Load training metadata
    meta_path = model_dir / "train_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        label_col = meta.get("label", "label_basis_1m_gt10bps")
        task = meta.get("task", "classification")
        feature_cols = meta.get("feature_cols") or None
    else:
        label_col = "label_basis_1m_gt10bps"
        task = "classification"
        feature_cols = None

    logger.info("Loading test data from %s", args.data_dir)
    X_test, y_test, y_net = load_test_data(args.data_dir, label_col, feature_cols)
    logger.info("Test data shape: X=%s, y=%s", X_test.shape, y_test.shape)

    results = []
    pkl_files = list(model_dir.glob(f"*_{label_col}.pkl"))

    if not pkl_files:
        logger.warning("No trained model files found in %s", model_dir)

    for pkl_path in pkl_files:
        model_name = pkl_path.stem.replace(f"_{label_col}", "")
        logger.info("Evaluating model: %s", model_name)
        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            result = run_backtest(
                model=model,
                X_test=X_test,
                y_test=y_test,
                y_net_profit=y_net,
                task=task,
                notional_usd=args.notional_usd,
                model_name=model_name,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", model_name, exc)

    if results:
        leaderboard = build_leaderboard(results)
        print_leaderboard(leaderboard)
        output_path = Path(args.output)
        leaderboard.write_csv(str(output_path))
        logger.info("Leaderboard saved to %s", output_path)
    else:
        logger.warning("No evaluation results produced.")


if __name__ == "__main__":
    main()
