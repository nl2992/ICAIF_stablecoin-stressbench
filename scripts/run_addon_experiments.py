#!/usr/bin/env python3
"""Add-on experiment runner: ExpectedNetProfitRegressor.

Directly predicts future_net_profit_bps_q10000 instead of classifying
whether a basis threshold will be exceeded. Compares against original
best non-oracle model from baseline results.

Results write ONLY to results/experiments_addon/. Baseline files are
never touched.

Usage:
    python scripts/run_addon_experiments.py
    python scripts/run_addon_experiments.py \
        --data-dir data/gold \
        --output-dir results/experiments_addon \
        --models expected_net_lgbm expected_net_xgb \
        --feature-sets price_only price_plus_book price_book_frag price_book_settle \
        --target net_profit_q10000_5m
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.models.cost_sensitive import ExpectedNetProfitRegressor
from stressbench.experiments.feature_sets import FEATURE_SETS

logger = get_logger(__name__)

_TARGET_COL = "net_profit_bps_q10000"
_NET_COL = "net_profit_bps_q10000"
_LABEL_COL = "label_arb_q10000_5m_gt0bps"
_TS_COL = "ts_1m_ns"

_MODELS = {
    "expected_net_lgbm": lambda: ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=300),
    "expected_net_xgb":  lambda: ExpectedNetProfitRegressor(base_model="xgb",  n_estimators=300),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add-on: ExpectedNetProfitRegressor experiments.")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--baseline-results", default="results/experiments/all_results.csv")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--models", nargs="+", default=list(_MODELS))
    p.add_argument("--feature-sets", nargs="+", default=["price_only", "price_plus_book", "price_book_frag"])
    p.add_argument("--target", default="net_profit_q10000_5m")
    return p.parse_args()


def _resolve_feature_cols(feat_set: str, df: pl.DataFrame) -> list[str]:
    cols = FEATURE_SETS.get(feat_set)
    if cols is None:
        meta = {"split", _TS_COL}
        label_prefix = {"label_", "split"}
        cols = [c for c in df.columns if not any(c.startswith(p) for p in label_prefix) and c not in meta]
    else:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            logger.warning("Feature set %s has no available columns", feat_set)
    return cols


def _economic_metrics(
    signal: np.ndarray,
    y_net: np.ndarray,
    y_label: np.ndarray,
) -> dict:
    n_trades = int(signal.sum())
    if n_trades == 0:
        return {
            "n_trades": 0,
            "net_bps_captured": "",
            "hit_rate_above_cost": "",
            "false_positive_cost": "",
            "final_pnl_bps": 0.0,
            "oracle_capture_pct": "",
        }

    y_net_clean = np.nan_to_num(y_net, nan=-999.0)
    traded = y_net_clean[signal == 1]
    net_bps = float(np.mean(traded))
    hit_rate = float((traded > 0).mean())

    # False positive cost: trades where label is 0 (not executable)
    fp_mask = (signal == 1) & (y_label == 0)
    fp_cost = float(np.mean(y_net_clean[fp_mask])) if fp_mask.any() else float("nan")

    total_pnl = float(np.sum(traded))

    # Oracle net bps for capture pct
    valid = y_net_clean[y_net_clean > -900]
    oracle_net = float(np.mean(valid[valid > 0])) if (valid > 0).any() else float("nan")
    capture = (net_bps / oracle_net * 100) if oracle_net > 0 and oracle_net == oracle_net else float("nan")

    return {
        "n_trades": n_trades,
        "net_bps_captured": round(net_bps, 3),
        "hit_rate_above_cost": round(hit_rate, 4),
        "false_positive_cost": round(fp_cost, 3) if fp_cost == fp_cost else "",
        "final_pnl_bps": round(total_pnl, 2),
        "oracle_capture_pct": round(capture, 1) if capture == capture else "",
    }


def run_one(
    model_name: str,
    feat_set: str,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target_col: str,
) -> dict:
    feature_cols = _resolve_feature_cols(feat_set, df_train)
    if not feature_cols:
        logger.warning("No feature cols for %s — skipping", feat_set)
        return {}

    X_train = df_train.select(feature_cols).to_numpy().astype(float)
    y_train = df_train[target_col].to_numpy().astype(float)

    X_val = df_val.select(feature_cols).to_numpy().astype(float)
    y_val = df_val[target_col].to_numpy().astype(float)

    X_test = df_test.select(feature_cols).to_numpy().astype(float)
    y_test = df_test[target_col].to_numpy().astype(float)
    y_label_test = df_test[_LABEL_COL].to_numpy().astype(float) if _LABEL_COL in df_test.columns else np.zeros(len(df_test))

    model = _MODELS[model_name]()
    logger.info("  Fitting %s / %s …", model_name, feat_set)
    model.fit(X_train, y_train)

    # Calibrate threshold on validation
    t = model.calibrate_threshold(X_val, y_val, min_trades=25)
    val_signal = model.predict_signal(X_val)
    val_metrics = _economic_metrics(val_signal, y_val, np.zeros(len(y_val)))

    # Test evaluation
    test_signal = model.predict_signal(X_test)
    test_metrics = _economic_metrics(test_signal, y_test, y_label_test)

    return {
        "model": model_name,
        "feature_set": feat_set,
        "target_col": target_col,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_features": len(feature_cols),
        "calibrated_threshold_bps": round(t, 3),
        "val_n_trades": val_metrics["n_trades"],
        "val_net_bps": val_metrics["net_bps_captured"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found at {dataset_path}")

    df = pl.read_parquet(str(dataset_path))
    df_train = df.filter(pl.col("split") == "train")
    df_val   = df.filter(pl.col("split") == "validation")
    df_test  = df.filter(pl.col("split") == "test")

    target_col = _NET_COL  # always net_profit_bps_q10000 for primary add-on task
    if target_col not in df.columns:
        raise RuntimeError(f"Target column {target_col!r} not in dataset")

    rows: list[dict] = []
    for model_name in args.models:
        if model_name not in _MODELS:
            logger.warning("Unknown model %s — skipping", model_name)
            continue
        for feat_set in args.feature_sets:
            result = run_one(model_name, feat_set, df_train, df_val, df_test, target_col)
            if result:
                rows.append(result)
                logger.info(
                    "    %s/%s: test_n_trades=%s  test_net_bps=%s",
                    model_name, feat_set,
                    result.get("test_n_trades", "—"),
                    result.get("test_net_bps_captured", "—"),
                )

    out_path = output_dir / "expected_net_profit_results.csv"
    if rows:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s (%d rows)", out_path, len(rows))
    else:
        logger.warning("No results to write.")


if __name__ == "__main__":
    main()
