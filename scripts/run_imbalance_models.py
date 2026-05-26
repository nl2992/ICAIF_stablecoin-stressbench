#!/usr/bin/env python3
"""Class imbalance model experiment runner.

Runs class-balanced/weighted variants of logistic, RF, XGBoost, LightGBM.

Tasks: basis_usdc_1m_gt10bps, executable_arb_q10000_5m
Feature sets: price_only, price_plus_book

Same calibration protocol as main experiment grid:
    - Threshold calibrated on validation split by maximizing total net P&L
    - Minimum 25 trades required

Writes to results/experiments_addon/imbalance_model_results.csv

Usage:
    python scripts/run_imbalance_models.py
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.experiments.feature_sets import FEATURE_SETS

logger = get_logger(__name__)

_TASKS = {
    "basis_usdc_1m_gt10bps": {
        "label_col": "label_basis_usdc_1m_gt10bps",
        "net_profit_col": "net_profit_bps_q10000",
        "notional_usd": 10_000,
    },
    "executable_arb_q10000_5m": {
        "label_col": "label_arb_q10000_5m_gt0bps",
        "net_profit_col": "net_profit_bps_q10000",
        "notional_usd": 10_000,
    },
}

_FEATURE_SETS = ["price_only", "price_plus_book"]

_ORACLE_NET_BPS = {
    "basis_usdc_1m_gt10bps": 180.0,
    "executable_arb_q10000_5m": 161.0,
}

_FIELDNAMES = [
    "task",
    "feature_set",
    "model",
    "class_balance_method",
    "test_net_bps_captured",
    "test_n_trades",
    "test_hit_rate",
    "test_final_pnl_usd",
    "auroc",
    "oracle_capture_pct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class imbalance model experiments")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments_addon")
    return p.parse_args()


def _build_models() -> dict[str, tuple[Any, str]]:
    """Build (model, class_balance_method) dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    models: dict[str, tuple[Any, str]] = {}

    # Logistic balanced
    models["logistic_balanced"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "class_weight_balanced",
    )

    # RF balanced
    models["rf_balanced"] = (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "class_weight_balanced",
    )

    # XGBoost weighted
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["xgb_weighted"] = (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=10,  # approximate class ratio
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            "scale_pos_weight",
        )
    except ImportError:
        logger.warning("xgboost not available, skipping xgb_weighted")

    # LightGBM weighted
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        models["lgbm_weighted"] = (
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                is_unbalance=True,
                random_state=42,
                verbose=-1,
            ),
            "is_unbalance",
        )
    except ImportError:
        logger.warning("lightgbm not available, skipping lgbm_weighted")

    return models


def _resolve_feature_cols(df: pl.DataFrame, feat_set: str, exclude: set[str]) -> list[str]:
    cols = FEATURE_SETS.get(feat_set)
    if cols is None:
        cols = [c for c in df.columns if c not in exclude and not c.startswith("label_")]
    else:
        cols = [c for c in cols if c in df.columns]
    return cols


def _extract_split(
    df: pl.DataFrame,
    split: str,
    feature_cols: list[str],
    label_col: str,
    net_profit_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdf = df.filter(pl.col("split") == split).filter(pl.col(label_col).is_not_null())
    if sdf.is_empty():
        n = len(feature_cols)
        return np.empty((0, n), np.float32), np.empty(0), np.empty(0)

    X_raw = sdf.select(feature_cols).to_numpy().astype(np.float32)
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        X = np.where(nan_mask, col_medians[None, :], X_raw)
    else:
        X = X_raw

    y = sdf[label_col].to_numpy().astype(np.int8)
    y_net_raw = sdf[net_profit_col].to_numpy().astype(np.float64)
    y_net = np.nan_to_num(y_net_raw, nan=-999.0)
    return X, y, y_net


def _calibrate_threshold(
    y_proba: np.ndarray,
    y_net: np.ndarray,
    n_candidates: int = 17,
    min_trades: int = 25,
) -> float:
    best_t, best_total = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, n_candidates):
        signal = y_proba > t
        n_sig = int(signal.sum())
        if n_sig < min_trades:
            continue
        total = float(np.sum(y_net[signal]))
        if total > best_total:
            best_total = total
            best_t = float(t)
    if best_total == -np.inf:
        return 0.5
    return best_t


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def run_one(
    df: pl.DataFrame,
    task_name: str,
    task_cfg: dict,
    feat_set: str,
    model_name: str,
    model: Any,
    balance_method: str,
) -> dict:
    exclude = {"split", "ts_1m_ns", "basis_primary_asset", "buy_venue", "sell_venue", "depth_source"}
    feature_cols = _resolve_feature_cols(df, feat_set, exclude)

    X_train, y_train, _ = _extract_split(df, "train", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])
    X_val, y_val, y_net_val = _extract_split(df, "validation", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])
    X_test, y_test, y_net_test = _extract_split(df, "test", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])

    # Check class diversity
    if len(np.unique(y_train)) < 2:
        logger.warning("Only one class in training data for task=%s feat=%s model=%s; skipping", task_name, feat_set, model_name)
        return {
            "task": task_name, "feature_set": feat_set, "model": model_name,
            "class_balance_method": balance_method,
            "test_net_bps_captured": float("nan"), "test_n_trades": 0,
            "test_hit_rate": float("nan"), "test_final_pnl_usd": 0.0,
            "auroc": float("nan"), "oracle_capture_pct": float("nan"),
        }

    model.fit(X_train, y_train)

    def _safe_proba(m: Any, X: np.ndarray) -> np.ndarray:
        """Get column-1 proba, handling single-class predict_proba."""
        proba = m.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            return proba.ravel()
        return proba[:, 1]

    # Calibrate threshold on validation
    if len(X_val) > 0:
        y_val_proba = _safe_proba(model, X_val)
        threshold = _calibrate_threshold(y_val_proba, y_net_val)
    else:
        threshold = 0.5

    # Evaluate on test
    y_test_proba = _safe_proba(model, X_test)
    signal = (y_test_proba > threshold).astype(np.int8)

    n_trades = int(signal.sum())
    auroc = _auroc(y_test, y_test_proba)

    if n_trades == 0:
        net_bps = float("nan")
        hit_rate = float("nan")
        final_pnl = 0.0
        oracle_pct = float("nan")
    else:
        traded_net = y_net_test[signal.astype(bool)]
        net_bps = float(np.mean(traded_net))
        hit_rate = float(np.mean(traded_net > 0))
        final_pnl = net_bps * n_trades * task_cfg["notional_usd"] / 10_000
        oracle = _ORACLE_NET_BPS.get(task_name, 161.0)
        oracle_pct = net_bps / oracle if oracle > 0 else float("nan")

    def _r(x: float, d: int = 4) -> float:
        return round(x, d) if not math.isnan(x) else x

    return {
        "task": task_name,
        "feature_set": feat_set,
        "model": model_name,
        "class_balance_method": balance_method,
        "test_net_bps_captured": _r(net_bps),
        "test_n_trades": n_trades,
        "test_hit_rate": _r(hit_rate),
        "test_final_pnl_usd": _r(final_pnl, 2),
        "auroc": _r(auroc),
        "oracle_capture_pct": _r(oracle_pct),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_path / "dataset.parquet" if data_path.is_dir() else data_path
    logger.info("Loading dataset from %s", parquet_path)
    df = pl.read_parquet(str(parquet_path))
    logger.info("Dataset shape: %s", df.shape)

    model_registry = _build_models()
    rows = []

    for task_name, task_cfg in _TASKS.items():
        if task_cfg["label_col"] not in df.columns:
            logger.warning("Label '%s' not found, skipping task %s", task_cfg["label_col"], task_name)
            continue
        for feat_set in _FEATURE_SETS:
            for model_name, (model_obj, balance_method) in model_registry.items():
                logger.info("Running: task=%s feat=%s model=%s", task_name, feat_set, model_name)
                try:
                    # Clone model for each run (sklearn models are stateful after fit)
                    model_copy = copy.deepcopy(model_obj)
                    row = run_one(df, task_name, task_cfg, feat_set, model_name, model_copy, balance_method)
                    rows.append(row)
                    logger.info(
                        "  -> n_trades=%d net_bps=%.4f auroc=%.4f oracle_pct=%.4f",
                        row["test_n_trades"],
                        row["test_net_bps_captured"] if not math.isnan(row["test_net_bps_captured"]) else float("nan"),
                        row["auroc"] if not math.isnan(row["auroc"]) else float("nan"),
                        row["oracle_capture_pct"] if not math.isnan(row["oracle_capture_pct"]) else float("nan"),
                    )
                except Exception as exc:
                    logger.error("FAILED task=%s feat=%s model=%s: %s", task_name, feat_set, model_name, exc, exc_info=True)

    out_path = out_dir / "imbalance_model_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
