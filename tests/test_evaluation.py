"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from stressbench.evaluation.economic_metrics import (
    cumulative_pnl,
    economic_summary,
    hit_rate_above_cost,
    max_drawdown,
    net_bps_captured,
)
from stressbench.evaluation.ml_metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["directional_accuracy"] == pytest.approx(1.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)


def test_regression_metrics_random():
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)
    metrics = regression_metrics(y_true, y_pred)
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_classification_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_proba)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_metrics_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    metrics = classification_metrics(y_true, y_proba)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_net_bps_captured_all_positive():
    y_net = np.array([10.0, 20.0, 30.0])
    signal = np.array([1, 1, 1])
    result = net_bps_captured(y_net, signal)
    assert result == pytest.approx(20.0)


def test_net_bps_captured_no_signal():
    y_net = np.array([10.0, 20.0])
    signal = np.array([0, 0])
    result = net_bps_captured(y_net, signal)
    assert np.isnan(result)


def test_hit_rate_above_cost():
    y_net = np.array([5.0, -3.0, 10.0, -1.0])
    signal = np.array([1, 1, 1, 1])
    rate = hit_rate_above_cost(y_net, signal, cost_threshold_bps=0.0)
    assert rate == pytest.approx(0.5)


def test_cumulative_pnl_all_positive():
    y_net = np.array([10.0, 10.0, 10.0])
    signal = np.array([1, 1, 1])
    pnl = cumulative_pnl(y_net, signal, notional_usd=10_000.0)
    assert pnl[-1] == pytest.approx(30.0)  # 3 * 10bps * 10000 / 10000 = 30


def test_max_drawdown_no_drawdown():
    pnl = np.array([0.0, 10.0, 20.0, 30.0])
    assert max_drawdown(pnl) == pytest.approx(0.0)


def test_max_drawdown_with_drawdown():
    pnl = np.array([0.0, 100.0, 50.0, 80.0])
    assert max_drawdown(pnl) == pytest.approx(50.0)


def test_economic_summary_keys():
    rng = np.random.default_rng(42)
    y_net = rng.normal(5, 15, 100)
    signal = (rng.uniform(0, 1, 100) > 0.5).astype(int)
    summary = economic_summary(y_net, signal)
    required_keys = [
        "net_bps_captured", "hit_rate_above_cost", "false_positive_cost",
        "n_trades", "final_pnl_usd", "max_drawdown_usd", "sharpe_ratio",
    ]
    for key in required_keys:
        assert key in summary
