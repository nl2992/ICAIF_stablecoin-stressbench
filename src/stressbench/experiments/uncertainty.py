"""Uncertainty-aware abstention experiments.

Two approaches:

A. Bootstrap ensemble uncertainty
   Train B models on bootstrap samples of the training set.
   Trade only when: mean_pred - k × std_pred > threshold

B. Quantile model
   Fit quantile regressors at the 10th, 50th, and 90th percentiles.
   Trade only when the lower quantile exceeds zero.

Results write to results/experiments_addon/. Baseline files are not modified.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


class BootstrapEnsemble:
    """Ensemble of B bootstrap-trained classifiers for uncertainty estimation.

    Parameters
    ----------
    base_model_factory:
        Callable that returns a fresh sklearn-compatible classifier.
    n_bootstrap:
        Number of bootstrap replicates.
    random_state:
        Master seed (each replicate uses seed + i).
    """

    def __init__(
        self,
        base_model_factory: Any,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ) -> None:
        self.base_model_factory = base_model_factory
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._models: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapEnsemble":
        rng = np.random.default_rng(self.random_state)
        self._models = []
        n = len(X)
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            model = self.base_model_factory()
            model.fit(X[idx], y[idx])
            self._models.append(model)
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_prediction, std_prediction) across bootstrap models."""
        preds = np.array([m.predict_proba(X)[:, 1] for m in self._models])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_signal(self, X: np.ndarray, k: float = 0.0) -> np.ndarray:
        """Trade when mean_pred - k × std_pred > 0.5.

        Parameters
        ----------
        k:
            Uncertainty discount factor. k=0 is the plain ensemble mean;
            higher k requires more confidence to trade.
        """
        mean, std = self.predict_mean_std(X)
        return ((mean - k * std) > 0.5).astype(np.int8)


class QuantileNetProfitModel:
    """Quantile regression for net_profit_bps with conservative abstention.

    Trades only when the lower quantile prediction exceeds zero, i.e., the
    model is confident the window will be profitable.

    Parameters
    ----------
    base_model:
        ``"lgbm"`` (supports native quantile loss) or ``"sklearn"``
        (uses GradientBoostingRegressor with quantile loss).
    quantiles:
        Quantile levels to fit. Default (0.1, 0.5, 0.9).
    random_state:
        Random seed.
    """

    def __init__(
        self,
        base_model: str = "lgbm",
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.quantiles = quantiles
        self.random_state = random_state
        self._models: dict[float, Any] = {}

    def fit(self, X: np.ndarray, y_net_profit: np.ndarray) -> "QuantileNetProfitModel":
        y_clean = np.nan_to_num(y_net_profit, nan=-999.0)
        for q in self.quantiles:
            if self.base_model == "lgbm":
                import lightgbm as lgb
                m = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=self.random_state,
                )
            m.fit(X, y_clean)
            self._models[q] = m
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[float, np.ndarray]:
        """Return predicted value at each quantile level."""
        return {q: m.predict(X) for q, m in self._models.items()}

    def predict_signal(self, X: np.ndarray, trade_quantile: float = 0.1) -> np.ndarray:
        """Trade only when the specified lower-quantile prediction exceeds 0.

        Parameters
        ----------
        trade_quantile:
            Which quantile to use as the conservatism floor. Default 0.1
            means trade only when the 10th-percentile prediction is positive.
        """
        if trade_quantile not in self._models:
            raise ValueError(f"trade_quantile {trade_quantile} not fitted. Available: {list(self._models)}")
        lower = self._models[trade_quantile].predict(X)
        return (lower > 0.0).astype(np.int8)


def abstention_sweep(
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    y_net: np.ndarray,
    k_values: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    base_threshold: float = 0.5,
) -> list[dict]:
    """Compute economic metrics across a sweep of uncertainty discount factors.

    Parameters
    ----------
    mean_preds:
        Bootstrap ensemble mean predictions (probabilities or net_profit_bps).
    std_preds:
        Bootstrap ensemble standard deviations.
    y_net:
        Realized net_profit_bps on the evaluation split.
    k_values:
        Uncertainty discount factors to sweep.
    base_threshold:
        Decision threshold applied to (mean - k × std).

    Returns
    -------
    list[dict]
        One row per k value with trade count and economic metrics.
    """
    y_clean = np.nan_to_num(y_net, nan=-999.0)
    rows = []
    for k in k_values:
        signal = (mean_preds - k * std_preds) > base_threshold
        n_trades = int(signal.sum())
        if n_trades == 0:
            net_bps = float("nan")
            hit_rate = float("nan")
        else:
            traded_net = y_clean[signal]
            net_bps = float(np.mean(traded_net))
            hit_rate = float((traded_net > 0).mean())
        rows.append({
            "k": k,
            "n_trades": n_trades,
            "net_bps_captured": round(net_bps, 3) if net_bps == net_bps else "",
            "hit_rate_above_cost": round(hit_rate, 4) if hit_rate == hit_rate else "",
            "total_pnl_bps": round(float(np.sum(y_clean[signal])), 2) if n_trades > 0 else 0.0,
        })
    return rows
