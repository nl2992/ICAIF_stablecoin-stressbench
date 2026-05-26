#!/usr/bin/env python3
"""Regime detection experiment runner.

Runs EWMA Z-Score, CUSUM, and BOCPD detectors on the test split using
cross_quote_basis_usdc_bps as the input signal.

Reports:
    - Regime detection accuracy vs label_basis_usdc_1m_gt10bps
    - False alarm rate
    - P(stress | actual stress)
    - Detection delay in minutes

Writes results to results/experiments_addon/regime_detection_results.csv

Usage:
    python scripts/run_regime_detection.py
    python scripts/run_regime_detection.py --data-dir data/gold --output-dir results/experiments_addon
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.models.regime_detection import (
    BOCPDDetector,
    CUSUMDetector,
    EWMAZScoreDetector,
)

logger = get_logger(__name__)

_SIGNAL_COL = "cross_quote_basis_usdc_bps"
_LABEL_COL = "label_basis_usdc_1m_gt10bps"

_FIELDNAMES = [
    "detector",
    "n_test",
    "n_stress_actual",
    "n_stress_predicted",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "p_stress_given_actual_stress",
    "mean_detection_delay_minutes",
    "auroc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime detection experiments")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments_addon")
    return p.parse_args()


def _safe(x: float) -> float:
    """Return nan for inf values."""
    return float("nan") if math.isinf(x) else x


def _detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean detection delay in minutes for true stress onsets.

    Detection delay = number of minutes from stress onset to first detection.
    If a stress episode is never detected, counts as full episode length.

    Args:
        y_true: Binary true stress labels, shape (n,).
        y_pred: Binary predicted stress labels, shape (n,).

    Returns:
        Mean detection delay in minutes (1 step = 1 minute).
    """
    delays = []
    in_stress = False
    onset_i = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_stress:
            # New stress episode starts
            in_stress = True
            onset_i = i
        elif y_true[i] == 0:
            in_stress = False

        if in_stress and y_pred[i] == 1:
            # Detected — delay = steps since onset
            delays.append(float(i - onset_i))
            in_stress = False  # Count only first detection per episode

    # Any open stress episode at the end with no detection
    if in_stress:
        # Not detected before end of series
        delays.append(float(len(y_true) - onset_i))

    return float(np.mean(delays)) if delays else float("nan")


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using trapezoidal rule."""
    from sklearn.metrics import roc_auc_score  # type: ignore
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    detector_name: str,
) -> dict:
    n = len(y_true)
    n_stress = int(y_true.sum())
    n_pred_stress = int(y_pred.sum())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    accuracy = (tp + tn) / n if n > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_acc = (recall + specificity) / 2 if not (math.isnan(recall) or math.isnan(specificity)) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0)
          else float("nan"))
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    p_stress_given_stress = recall  # same as recall

    delay = _detection_delay(y_true, y_pred)
    auroc = _auroc(y_true, y_score)

    return {
        "detector": detector_name,
        "n_test": n,
        "n_stress_actual": n_stress,
        "n_stress_predicted": n_pred_stress,
        "accuracy": round(_safe(accuracy), 4),
        "balanced_accuracy": round(_safe(balanced_acc), 4),
        "precision": round(_safe(precision), 4),
        "recall": round(_safe(recall), 4),
        "f1": round(_safe(f1), 4),
        "false_alarm_rate": round(_safe(false_alarm_rate), 4),
        "p_stress_given_actual_stress": round(_safe(p_stress_given_stress), 4),
        "mean_detection_delay_minutes": round(_safe(delay), 2),
        "auroc": round(_safe(auroc), 4),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_path / "dataset.parquet" if data_path.is_dir() else data_path
    logger.info("Loading dataset from %s", parquet_path)
    df = pl.read_parquet(str(parquet_path))

    if _SIGNAL_COL not in df.columns:
        raise ValueError(f"Signal column '{_SIGNAL_COL}' not found in dataset")
    if _LABEL_COL not in df.columns:
        logger.warning("Label column '%s' not found; using label_basis_1m_gt10bps fallback", _LABEL_COL)
        _label = "label_basis_1m_gt10bps"
    else:
        _label = _LABEL_COL

    # Extract train and test data
    df_train = df.filter(pl.col("split") == "train")
    df_test = df.filter(pl.col("split") == "test")

    X_train_raw = df_train[_SIGNAL_COL].to_numpy().astype(np.float32).reshape(-1, 1)
    y_train = df_train[_label].to_numpy().astype(np.int8)

    X_test_raw = df_test[_SIGNAL_COL].to_numpy().astype(np.float32).reshape(-1, 1)
    y_test = df_test[_label].to_numpy().astype(np.int8)

    # Handle NaN
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0)

    logger.info("Train size: %d  Test size: %d", len(X_train_raw), len(X_test_raw))
    logger.info("Test stress rate: %.3f", y_test.mean())

    detectors = [
        EWMAZScoreDetector(span=20, threshold=2.5, signal_col=0),
        EWMAZScoreDetector(span=60, threshold=3.0, signal_col=0),
        CUSUMDetector(k=0.5, h=5.0, signal_col=0),
        CUSUMDetector(k=0.5, h=3.0, signal_col=0),
        BOCPDDetector(hazard_rate=0.01, threshold=0.5, signal_col=0),
        BOCPDDetector(hazard_rate=0.05, threshold=0.3, signal_col=0),
    ]

    rows = []
    for detector in detectors:
        det_name = f"{detector.name}_{detector.__class__.__name__}"
        # Add parameter suffix for disambiguation
        if isinstance(detector, EWMAZScoreDetector):
            det_name = f"EWMA_span{detector.span}_thr{detector.threshold}"
        elif isinstance(detector, CUSUMDetector):
            det_name = f"CUSUM_k{detector.k}_h{detector.h}"
        elif isinstance(detector, BOCPDDetector):
            det_name = f"BOCPD_hz{detector.hazard_rate}_thr{detector.threshold}"

        logger.info("Running detector: %s", det_name)
        try:
            detector.fit(X_train_raw, y_train)
            y_pred = detector.predict(X_test_raw)
            y_proba = detector.predict_proba(X_test_raw)[:, 1]
            metrics = _compute_metrics(y_test, y_pred, y_proba, det_name)
            rows.append(metrics)
            logger.info(
                "  accuracy=%.3f  recall=%.3f  FAR=%.3f  delay=%.1f min  AUROC=%.3f",
                metrics["accuracy"], metrics["recall"],
                metrics["false_alarm_rate"], metrics["mean_detection_delay_minutes"],
                metrics["auroc"],
            )
        except Exception as exc:
            logger.error("FAILED detector=%s: %s", det_name, exc, exc_info=True)

    out_path = out_dir / "regime_detection_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
