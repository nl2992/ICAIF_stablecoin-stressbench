#!/usr/bin/env python3
"""Train all benchmark models on the training split.

Usage:
    python scripts/train_models.py --data-dir data/gold --model-dir models/trained
    python scripts/train_models.py --data-dir data/gold --model-dir models/trained --models lgbm xgb rf
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_ALL_MODELS = ["last_value", "rolling_mean", "ar1", "logistic", "ridge", "lasso", "lgbm", "xgb", "rf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train benchmark models.")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--model-dir", default="models/trained")
    parser.add_argument(
        "--models",
        nargs="*",
        default=_ALL_MODELS,
        help=f"Models to train. Options: {_ALL_MODELS}",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type.",
    )
    parser.add_argument(
        "--label",
        default="label_basis_1m_gt10bps",
        help="Label column to predict.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help="Feature columns (default: all non-label columns).",
    )
    return parser.parse_args()


def load_train_data(data_dir: str, label_col: str):
    """Load training data from Gold Parquet files."""
    import polars as pl

    gold_path = Path(data_dir)
    parquet_files = list(gold_path.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning("No Parquet files found in %s; generating synthetic data.", data_dir)
        rng = np.random.default_rng(42)
        n = 10_000
        X = rng.standard_normal((n, 20)).astype(np.float32)
        y = (rng.standard_normal(n) > 0).astype(np.int8)
        return X, y, None

    df = pl.read_parquet(str(gold_path / "*.parquet"))
    label_df = df.filter(pl.col("split") == "train")

    feature_cols = [c for c in df.columns if not c.startswith("label_") and c != "split"]
    X = label_df.select(feature_cols).to_numpy().astype(np.float32)
    y = label_df[label_col].to_numpy()
    return X, y, feature_cols


def get_model(name: str, task: str):
    if name == "last_value":
        from stressbench.models.baselines import LastValueBaseline
        return LastValueBaseline()
    elif name == "rolling_mean":
        from stressbench.models.baselines import RollingMeanBaseline
        return RollingMeanBaseline()
    elif name == "ar1":
        from stressbench.models.baselines import AR1Baseline
        return AR1Baseline()
    elif name == "logistic":
        from stressbench.models.baselines import LogisticBaseline
        return LogisticBaseline()
    elif name == "ridge":
        from stressbench.models.baselines import RidgeBaseline
        return RidgeBaseline()
    elif name == "lasso":
        from stressbench.models.baselines import LassoBaseline
        return LassoBaseline()
    elif name == "lgbm":
        from stressbench.models.tree_models import LGBMWrapper
        return LGBMWrapper(task=task)
    elif name == "xgb":
        from stressbench.models.tree_models import XGBWrapper
        return XGBWrapper(task=task)
    elif name == "rf":
        from stressbench.models.tree_models import RandomForestWrapper
        return RandomForestWrapper(task=task)
    else:
        raise ValueError(f"Unknown model: {name}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from %s", args.data_dir)
    X, y, feature_cols = load_train_data(args.data_dir, args.label)
    logger.info("Training data shape: X=%s, y=%s", X.shape, y.shape)

    for model_name in args.models:
        logger.info("Training model: %s", model_name)
        try:
            model = get_model(model_name, args.task)
            model.fit(X, y)
            model_path = model_dir / f"{model_name}_{args.label}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("Saved model to %s", model_path)
        except Exception as exc:
            logger.error("Failed to train %s: %s", model_name, exc)

    # Save feature column metadata
    meta = {
        "label": args.label,
        "task": args.task,
        "feature_cols": feature_cols or [],
        "models_trained": args.models,
    }
    with open(model_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. Models saved to %s", model_dir)


if __name__ == "__main__":
    main()
