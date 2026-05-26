"""Expected net-profit regressor — add-on model for Stablecoin StressBench.

Directly predicts future executable net_profit_bps rather than classifying
whether a basis threshold will be exceeded. Trades when the predicted net
profit exceeds a validation-calibrated floor.

This model is an add-on to the original benchmark. It does not replace the
classification tasks in results/experiments/. Results write to
results/experiments_addon/.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ExpectedNetProfitRegressor:
    """Predict future executable net_profit_bps directly.

    Parameters
    ----------
    base_model:
        Underlying regression algorithm. One of ``"lgbm"`` or ``"xgb"``.
    threshold_bps:
        Minimum predicted net profit (bps) required to generate a trade signal.
        Calibrated on validation split before test evaluation.
    random_state:
        Random seed for reproducibility.
    **kwargs:
        Passed through to the underlying model constructor.
    """

    def __init__(
        self,
        base_model: str = "lgbm",
        threshold_bps: float = 0.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.base_model = base_model
        self.threshold_bps = threshold_bps
        self.random_state = random_state
        self.kwargs = kwargs
        self._model: Any = None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y_net_profit: np.ndarray) -> "ExpectedNetProfitRegressor":
        """Fit on training data.

        NaN values in y_net_profit (missing depth data) are replaced with
        a large negative sentinel so the model learns to predict low profit
        for windows with no executable depth.
        """
        if self.base_model == "lgbm":
            import lightgbm as lgb
            self._model = lgb.LGBMRegressor(
                n_estimators=self.kwargs.get("n_estimators", 300),
                learning_rate=self.kwargs.get("learning_rate", 0.03),
                num_leaves=self.kwargs.get("num_leaves", 31),
                min_child_samples=self.kwargs.get("min_child_samples", 50),
                subsample=self.kwargs.get("subsample", 0.8),
                colsample_bytree=self.kwargs.get("colsample_bytree", 0.8),
                random_state=self.random_state,
                verbose=-1,
            )
        elif self.base_model == "xgb":
            import xgboost as xgb
            self._model = xgb.XGBRegressor(
                n_estimators=self.kwargs.get("n_estimators", 300),
                learning_rate=self.kwargs.get("learning_rate", 0.03),
                max_depth=self.kwargs.get("max_depth", 4),
                subsample=self.kwargs.get("subsample", 0.8),
                colsample_bytree=self.kwargs.get("colsample_bytree", 0.8),
                random_state=self.random_state,
                tree_method="hist",
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown base_model: {self.base_model!r}. Use 'lgbm' or 'xgb'.")

        y_clean = np.nan_to_num(y_net_profit, nan=-999.0)
        self._model.fit(X, y_clean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted net_profit_bps for each row."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._model.predict(X)

    def predict_signal(self, X: np.ndarray) -> np.ndarray:
        """Return binary trade signal (1 = trade, 0 = abstain)."""
        return (self.predict(X) > self.threshold_bps).astype(np.int8)

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        X_val: np.ndarray,
        y_net_val: np.ndarray,
        n_candidates: int = 19,
        min_trades: int = 25,
    ) -> float:
        """Set threshold_bps to maximize total net P&L on the validation split.

        Parameters
        ----------
        X_val:
            Validation feature matrix.
        y_net_val:
            Realized net_profit_bps on validation split.
        n_candidates:
            Number of threshold candidates to try (linear grid over predicted
            profit range).
        min_trades:
            Minimum number of validation trades required for a threshold to
            qualify.

        Returns
        -------
        float
            The chosen threshold (also stored as ``self.threshold_bps``).
        """
        preds = self.predict(X_val)
        y_clean = np.nan_to_num(y_net_val, nan=-999.0)

        lo, hi = float(np.nanpercentile(preds, 5)), float(np.nanpercentile(preds, 95))
        candidates = np.linspace(lo, hi, n_candidates)

        best_t, best_total = float(np.median(preds)), -np.inf
        for t in candidates:
            mask = preds > t
            n_sig = int(mask.sum())
            if n_sig < min_trades:
                continue
            total = float(np.sum(y_clean[mask]))
            if total > best_total:
                best_total = total
                best_t = float(t)

        self.threshold_bps = best_t
        return best_t

    # ------------------------------------------------------------------
    # Scikit-learn compatibility shim
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probability columns compatible with sklearn pipelines.

        Column 0: P(no trade) = sigmoid(-pred / scale)
        Column 1: P(trade)    = sigmoid(pred / scale)
        """
        pred = self.predict(X)
        scale = max(float(np.std(pred)), 1.0)
        p1 = 1.0 / (1.0 + np.exp(-pred / scale))
        return np.column_stack([1.0 - p1, p1])
