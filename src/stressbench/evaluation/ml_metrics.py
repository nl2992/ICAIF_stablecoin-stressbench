"""ML evaluation metrics for regression and classification tasks."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict with MAE, RMSE, directional accuracy, and Spearman rank correlation.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Directional accuracy: fraction of times sign(pred) == sign(true)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = float(np.mean(direction_true == direction_pred))

    # Spearman rank correlation
    rho, _ = spearmanr(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_acc,
        "spearman_rho": float(rho),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification evaluation metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities for the positive class.
        y_pred_binary: Binary predictions; derived from ``y_pred_proba`` if None.
        threshold: Decision threshold for binary predictions.

    Returns:
        Dict with AUROC, AUPRC, F1, balanced accuracy, Brier score.
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

    try:
        auroc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        auprc = float("nan")

    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred_binary))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
    }


def calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve data for a classifier.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Dict with ``fraction_of_positives`` and ``mean_predicted_value`` arrays.
    """
    from typing import Any
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }
