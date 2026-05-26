#!/usr/bin/env python3
"""LightGBM hyperparameter sweep experiment.

Sweeps LightGBM (best base model) over:
    n_estimators:       [100, 200, 500]
    learning_rate:      [0.01, 0.05, 0.1]
    num_leaves:         [15, 31, 63]
    min_child_samples:  [20, 50]

Total: 3 * 3 * 3 * 2 = 54 configurations.

Selection criterion:
    Primary:   validation total P&L (sum of net_profit_bps for signalled trades)
    Secondary: AUPRC

Tasks: basis_usdc_1m_gt10bps, executable_arb_q10000_5m
Feature sets: price_only, price_plus_book

Writes to results/experiments_addon/hyperparameter_sweep.csv

Usage:
    python scripts/run_hyperparameter_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import product
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

_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 50],
}

_ORACLE_NET_BPS = {
    "basis_usdc_1m_gt10bps": 180.0,
    "executable_arb_q10000_5m": 161.0,
}

_FIELDNAMES = [
    "task",
    "feature_set",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "val_total_pnl_bps",
    "val_n_trades",
    "val_auprc",
    "test_net_bps_captured",
    "test_n_trades",
    "test_auroc",
    "test_auprc",
    "test_final_pnl_usd",
    "oracle_capture_pct",
    "selected_best",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM hyperparameter sweep")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments_addon")
    return p.parse_args()


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
) -> tuple[float, float, int]:
    best_t, best_total, best_n = 0.5, -np.inf, 0
    for t in np.linspace(0.05, 0.95, n_candidates):
        signal = y_proba > t
        n_sig = int(signal.sum())
        if n_sig < min_trades:
            continue
        total = float(np.sum(y_net[signal]))
        if total > best_total:
            best_total = total
            best_t = float(t)
            best_n = n_sig
    if best_total == -np.inf:
        return 0.5, float("nan"), 0
    return best_t, best_total, best_n


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def run_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_net_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_net_test: np.ndarray,
    params: dict,
    notional_usd: int,
    oracle_net_bps: float,
) -> dict:
    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except ImportError:
        raise ImportError("lightgbm required for hyperparameter sweep")

    model = LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Validation metrics
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_threshold, val_total_pnl, val_n_trades = _calibrate_threshold(y_val_proba, y_net_val)
    val_auprc = _auprc(y_val, y_val_proba)

    # Test metrics
    y_test_proba = model.predict_proba(X_test)[:, 1]
    signal = (y_test_proba > val_threshold).astype(np.int8)
    n_trades = int(signal.sum())

    if n_trades > 0:
        traded_net = y_net_test[signal.astype(bool)]
        net_bps = float(np.mean(traded_net))
        final_pnl = net_bps * n_trades * notional_usd / 10_000
        oracle_pct = net_bps / oracle_net_bps if oracle_net_bps > 0 else float("nan")
    else:
        net_bps = float("nan")
        final_pnl = 0.0
        oracle_pct = float("nan")

    test_auroc = _auroc(y_test, y_test_proba)
    test_auprc = _auprc(y_test, y_test_proba)

    def _r(x: float, d: int = 4) -> float:
        return round(x, d) if not (math.isnan(x) or math.isinf(x)) else x

    return {
        **params,
        "val_total_pnl_bps": _r(val_total_pnl),
        "val_n_trades": val_n_trades,
        "val_auprc": _r(val_auprc),
        "test_net_bps_captured": _r(net_bps),
        "test_n_trades": n_trades,
        "test_auroc": _r(test_auroc),
        "test_auprc": _r(test_auprc),
        "test_final_pnl_usd": _r(final_pnl, 2),
        "oracle_capture_pct": _r(oracle_pct),
        "selected_best": False,
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_path / "dataset.parquet" if data_path.is_dir() else data_path
    logger.info("Loading dataset from %s", parquet_path)
    df = pl.read_parquet(str(parquet_path))

    # Build all parameter combinations
    param_names = list(_PARAM_GRID.keys())
    param_values = list(_PARAM_GRID.values())
    all_params = [
        dict(zip(param_names, combo))
        for combo in product(*param_values)
    ]
    total_configs = len(all_params) * len(_TASKS) * len(_FEATURE_SETS)
    logger.info("Total configurations: %d (%d params x %d tasks x %d feat_sets)",
                total_configs, len(all_params), len(_TASKS), len(_FEATURE_SETS))

    exclude = {"split", "ts_1m_ns", "basis_primary_asset", "buy_venue", "sell_venue", "depth_source"}

    all_rows: dict[tuple[str, str], list[dict]] = {}

    for task_name, task_cfg in _TASKS.items():
        if task_cfg["label_col"] not in df.columns:
            logger.warning("Label '%s' not found, skipping %s", task_cfg["label_col"], task_name)
            continue

        for feat_set in _FEATURE_SETS:
            key = (task_name, feat_set)
            feature_cols = _resolve_feature_cols(df, feat_set, exclude)
            X_train, y_train, _ = _extract_split(df, "train", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])
            X_val, y_val, y_net_val = _extract_split(df, "validation", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])
            X_test, y_test, y_net_test = _extract_split(df, "test", feature_cols, task_cfg["label_col"], task_cfg["net_profit_col"])

            oracle_bps = _ORACLE_NET_BPS.get(task_name, 161.0)
            rows_for_key = []

            for i, params in enumerate(all_params):
                logger.info(
                    "[%d/%d] task=%s feat=%s params=%s",
                    i + 1, len(all_params), task_name, feat_set, params,
                )
                try:
                    row = run_config(
                        X_train, y_train, X_val, y_val, y_net_val,
                        X_test, y_test, y_net_test,
                        params, task_cfg["notional_usd"], oracle_bps,
                    )
                    row["task"] = task_name
                    row["feature_set"] = feat_set
                    rows_for_key.append(row)
                except Exception as exc:
                    logger.error("FAILED %s %s %s: %s", task_name, feat_set, params, exc, exc_info=True)

            # Mark best by primary criterion: val_total_pnl_bps (secondary: val_auprc)
            if rows_for_key:
                best_idx = max(
                    range(len(rows_for_key)),
                    key=lambda i: (
                        rows_for_key[i]["val_total_pnl_bps"] if not math.isnan(rows_for_key[i]["val_total_pnl_bps"]) else -1e9,
                        rows_for_key[i]["val_auprc"] if not math.isnan(rows_for_key[i]["val_auprc"]) else -1e9,
                    ),
                )
                rows_for_key[best_idx]["selected_best"] = True
                logger.info(
                    "Best config for task=%s feat=%s: %s (val_pnl=%.2f, val_auprc=%.4f)",
                    task_name, feat_set,
                    {k: rows_for_key[best_idx][k] for k in param_names},
                    rows_for_key[best_idx]["val_total_pnl_bps"],
                    rows_for_key[best_idx]["val_auprc"],
                )
            all_rows[key] = rows_for_key

    # Flatten and write
    all_flat = [r for rows in all_rows.values() for r in rows]

    out_path = out_dir / "hyperparameter_sweep.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_flat)

    logger.info("Wrote %d rows to %s", len(all_flat), out_path)
    print(f"Wrote {len(all_flat)} rows to {out_path}")


if __name__ == "__main__":
    main()
