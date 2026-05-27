"""Tree-based models: LightGBM, XGBoost, and Random Forest.

These models are the primary baselines for the benchmark. They accept
the full feature set (book, basis, fragmentation, settlement, issuer events)
and are evaluated on both ML metrics and economic metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


class LGBMWrapper:
    """LightGBM wrapper for both regression and classification tasks."""

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            **kwargs,
        }
        self._model = None
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list | None = None,
        feature_names: list[str] | None = None,
    ) -> "LGBMWrapper":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            )

        if self.task == "regression":
            self._model = lgb.LGBMRegressor(**self.params)
        else:
            self._model = lgb.LGBMClassifier(**self.params)

        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        if feature_names:
            fit_kwargs["feature_name"] = feature_names

        self._model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        return self._model.predict_proba(X)


class XGBWrapper:
    """XGBoost wrapper for both regression and classification tasks."""

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "tree_method": "hist",
            **kwargs,
        }
        self._model = None
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list | None = None,
    ) -> "XGBWrapper":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        if self.task == "regression":
            self._model = xgb.XGBRegressor(**self.params)
        else:
            self._model = xgb.XGBClassifier(**self.params)

        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = 50
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        return self._model.predict_proba(X)


class RandomForestWrapper:
    """Random Forest wrapper for both regression and classification tasks."""

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
        self._model = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestWrapper":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.task == "regression":
            self._model = RandomForestRegressor(**self.params)
        else:
            self._model = RandomForestClassifier(**self.params)

        self._model.fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression tasks.")
        return self._model.predict_proba(X)
