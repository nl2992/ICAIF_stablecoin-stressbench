"""Naive and econometric baseline models.

Baselines:
    LastValueBaseline: Predict the last observed value (random walk).
    RollingMeanBaseline: Predict the rolling mean over a window.
    AR1Baseline: Autoregressive model of order 1.
    LogisticBaseline: Logistic regression for classification tasks.
    RidgeBaseline: Ridge regression for regression tasks.
    LassoBaseline: Lasso regression for regression tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


class LastValueBaseline:
    """Predict the last observed value of the target (random walk baseline)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastValueBaseline":
        # No fitting required; last value is derived from the last feature column
        self._last_col_idx = -1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._last_col_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # For classification: sigmoid of last value
        vals = self.predict(X)
        proba = 1 / (1 + np.exp(-vals))
        return np.column_stack([1 - proba, proba])


class RollingMeanBaseline:
    """Predict the rolling mean of the target over a fixed window."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RollingMeanBaseline":
        self._mean = float(np.mean(y[-self.window:]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)


class AR1Baseline:
    """Autoregressive model of order 1: y_hat(t) = alpha + beta * y(t-1)."""

    def __init__(self) -> None:
        self._alpha: float = 0.0
        self._beta: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AR1Baseline":
        if len(y) < 2:
            return self
        y_lag = y[:-1]
        y_curr = y[1:]
        cov = np.cov(y_lag, y_curr)
        var = np.var(y_lag)
        self._beta = cov[0, 1] / var if var > 0 else 1.0
        self._alpha = np.mean(y_curr) - self._beta * np.mean(y_lag)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assumes last column of X contains y(t-1)
        y_lag = X[:, -1]
        return self._alpha + self._beta * y_lag


class LogisticBaseline:
    """Logistic regression classifier with standard scaling."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(X))


class RidgeBaseline:
    """Ridge regression with standard scaling."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))


class LassoBaseline:
    """Lasso regression with standard scaling."""

    def __init__(self, alpha: float = 0.01) -> None:
        self._scaler = StandardScaler()
        self._model = Lasso(alpha=alpha, max_iter=5000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoBaseline":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))
