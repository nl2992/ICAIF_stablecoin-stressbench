#!/usr/bin/env python3
"""Meta-labeling experiment runner.

Implements López de Prado (2018) meta-labeling on the benchmark dataset.

Primary signal: |cross_quote_basis_usdc_bps| > 10 bps
Meta-label:     label_arb_q10000_5m_gt0bps (profitable execution within 5m)

Tasks:
    basis_usdc_1m_gt10bps    — USDC basis >10 bps classification
    executable_arb_q10000_5m — Executable arbitrage at $10K / 5m

Writes results to results/experiments_addon/meta_labeling_results.csv

Usage:
    python scripts/run_meta_labeling_experiments.py
    python scripts/run_meta_labeling_experiments.py --data-dir data/gold --output-dir results/experiments_addon
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.experiments.feature_sets import FEATURE_SETS
from stressbench.models.meta_labeling import MetaLabelingFilter

logger = get_logger(__name__)

_TASKS = {
    "basis_usdc_1m_gt10bps": {
        "label_col": "label_basis_usdc_1m_gt10bps",
        "net_profit_col": "net_profit_bps_q10000",
        "meta_label_col": "label_arb_q10000_5m_gt0bps",
        "notional_usd": 10_000,
    },
    "executable_arb_q10000_5m": {
        "label_col": "label_arb_q10000_5m_gt0bps",
        "net_profit_col": "net_profit_bps_q10000",
        "meta_label_col": "label_arb_q10000_5m_gt0bps",
        "notional_usd": 10_000,
    },
}

_FEATURE_SETS = ["price_only", "price_plus_book"]
_BASIS_COL = "cross_quote_basis_usdc_bps"
_PRIMARY_THRESHOLD_BPS = 10.0

_FIELDNAMES = [
    "task",
    "feature_set",
    "model",
    "n_primary_fires_train",
    "n_meta_positive_train",
    "test_n_trades",
    "test_net_bps_captured",
    "test_hit_rate",
    "test_final_pnl_usd",
    "test_oracle_capture_pct",
]

_ORACLE_NET_BPS = {
    "basis_usdc_1m_gt10bps": 180.0,
    "executable_arb_q10000_5m": 161.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-labeling experiments")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments_addon")
    return p.parse_args()


def _resolve_feature_cols(
    df: pl.DataFrame, feat_set: str, exclude: set[str]
) -> list[str]:
    cols = FEATURE_SETS.get(feat_set)
    if cols is None:
        cols = [
            c for c in df.columns if c not in exclude and not c.startswith("label_")
        ]
    else:
        cols = [c for c in cols if c in df.columns]
    return cols


def _extract_split(
    df: pl.DataFrame,
    split: str,
    feature_cols: list[str],
    label_col: str,
    meta_label_col: str,
    net_profit_col: str,
    basis_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sdf = df.filter(pl.col("split") == split).filter(pl.col(label_col).is_not_null())
    if sdf.is_empty():
        n = len(feature_cols)
        return (
            np.empty((0, n), dtype=np.float32),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    X_raw = sdf.select(feature_cols).to_numpy().astype(np.float32)
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        X = np.where(nan_mask, col_medians[None, :], X_raw)
    else:
        X = X_raw

    y_label = sdf[label_col].to_numpy().astype(np.int8)
    y_meta = sdf[meta_label_col].to_numpy().astype(np.int8)
    y_net = sdf[net_profit_col].to_numpy().astype(np.float64)
    y_net = np.nan_to_num(y_net, nan=-999.0)

    # Primary signal from basis column
    if basis_col in sdf.columns:
        basis = sdf[basis_col].to_numpy().astype(np.float64)
    else:
        basis = np.zeros(len(sdf), dtype=np.float64)
    y_primary = (np.abs(basis) > _PRIMARY_THRESHOLD_BPS).astype(np.int8)

    return X, y_primary, y_meta, y_net, y_label


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


def _economic_metrics(
    signal: np.ndarray,
    y_net: np.ndarray,
    notional_usd: int,
    oracle_net_bps: float,
) -> dict:
    n_trades = int(signal.sum())
    if n_trades == 0:
        return {
            "test_n_trades": 0,
            "test_net_bps_captured": float("nan"),
            "test_hit_rate": float("nan"),
            "test_final_pnl_usd": 0.0,
            "test_oracle_capture_pct": float("nan"),
        }

    traded_net = y_net[signal.astype(bool)]
    net_bps = float(np.mean(traded_net))
    hit_rate = float(np.mean(traded_net > 0))
    final_pnl = net_bps * n_trades * notional_usd / 10_000

    if oracle_net_bps > 0:
        oracle_capture = net_bps / oracle_net_bps
    else:
        oracle_capture = float("nan")

    return {
        "test_n_trades": n_trades,
        "test_net_bps_captured": round(net_bps, 4),
        "test_hit_rate": round(hit_rate, 4),
        "test_final_pnl_usd": round(final_pnl, 2),
        "test_oracle_capture_pct": (
            round(oracle_capture, 4) if not math.isnan(oracle_capture) else float("nan")
        ),
    }


def run_meta_labeling(
    df: pl.DataFrame,
    task_name: str,
    task_cfg: dict,
    feat_set: str,
) -> dict:
    exclude = {
        "split",
        "ts_1m_ns",
        "basis_primary_asset",
        "buy_venue",
        "sell_venue",
        "depth_source",
    }
    feature_cols = _resolve_feature_cols(df, feat_set, exclude)

    # Find index of basis column in feature_cols
    basis_idx = 0
    if _BASIS_COL in feature_cols:
        basis_idx = feature_cols.index(_BASIS_COL)

    X_train, y_primary_train, y_meta_train, y_net_train, _ = _extract_split(
        df,
        "train",
        feature_cols,
        task_cfg["label_col"],
        task_cfg["meta_label_col"],
        task_cfg["net_profit_col"],
        _BASIS_COL,
    )
    X_val, y_primary_val, y_meta_val, y_net_val, _ = _extract_split(
        df,
        "validation",
        feature_cols,
        task_cfg["label_col"],
        task_cfg["meta_label_col"],
        task_cfg["net_profit_col"],
        _BASIS_COL,
    )
    X_test, y_primary_test, y_meta_test, y_net_test, _ = _extract_split(
        df,
        "test",
        feature_cols,
        task_cfg["label_col"],
        task_cfg["meta_label_col"],
        task_cfg["net_profit_col"],
        _BASIS_COL,
    )

    model = MetaLabelingFilter(
        primary_threshold_bps=_PRIMARY_THRESHOLD_BPS,
        primary_signal_col=basis_idx,
    )
    model.fit(X_train, y_primary_train, y_meta_train)

    n_primary_fires_train = model.n_primary_fires_train
    n_meta_positive_train = model.n_meta_positive_train

    logger.info(
        "Task=%s feat=%s: primary_fires_train=%d meta_pos_train=%d",
        task_name,
        feat_set,
        n_primary_fires_train,
        n_meta_positive_train,
    )

    # Calibrate on validation
    if len(X_val) > 0:
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_threshold = _calibrate_threshold(y_val_proba, y_net_val)
    else:
        val_threshold = 0.5

    # Evaluate on test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_signal = (y_test_proba > val_threshold).astype(np.int8)

    oracle_net_bps = _ORACLE_NET_BPS.get(task_name, 161.0)
    econ = _economic_metrics(
        test_signal, y_net_test, task_cfg["notional_usd"], oracle_net_bps
    )

    return {
        "task": task_name,
        "feature_set": feat_set,
        "model": "MetaLabelingFilter_lgbm",
        "n_primary_fires_train": n_primary_fires_train,
        "n_meta_positive_train": n_meta_positive_train,
        **econ,
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

    rows = []
    for task_name, task_cfg in _TASKS.items():
        if task_cfg["label_col"] not in df.columns:
            logger.warning(
                "Label column %s not found, skipping task %s",
                task_cfg["label_col"],
                task_name,
            )
            continue
        if task_cfg["meta_label_col"] not in df.columns:
            logger.warning(
                "Meta-label column %s not found, skipping task %s",
                task_cfg["meta_label_col"],
                task_name,
            )
            continue
        for feat_set in _FEATURE_SETS:
            logger.info("Running: task=%s feat=%s", task_name, feat_set)
            try:
                row = run_meta_labeling(df, task_name, task_cfg, feat_set)
                rows.append(row)
                logger.info(
                    "  -> n_trades=%s net_bps=%s oracle_pct=%s",
                    row["test_n_trades"],
                    row["test_net_bps_captured"],
                    row["test_oracle_capture_pct"],
                )
            except Exception as exc:
                logger.error(
                    "FAILED task=%s feat=%s: %s",
                    task_name,
                    feat_set,
                    exc,
                    exc_info=True,
                )

    out_path = out_dir / "meta_labeling_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
