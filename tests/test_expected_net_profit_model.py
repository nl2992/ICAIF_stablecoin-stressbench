"""Tests for ExpectedNetProfitRegressor (add-on model)."""

from __future__ import annotations

import numpy as np
import pytest

from stressbench.models.cost_sensitive import ExpectedNetProfitRegressor


@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 5))
    # Net profit correlated with X[:, 0], noise elsewhere
    y = 5 * X[:, 0] + rng.standard_normal(n) * 2
    # 10% NaN to simulate missing depth data
    nan_idx = rng.choice(n, size=20, replace=False)
    y[nan_idx] = np.nan
    return X, y


def test_fit_predict_shape(synthetic_data):
    X, y = synthetic_data
    model = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_predict_signal_is_binary(synthetic_data):
    X, y = synthetic_data
    model = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10, threshold_bps=0.0)
    model.fit(X, y)
    signal = model.predict_signal(X)
    assert set(np.unique(signal)).issubset({0, 1})


def test_higher_threshold_reduces_or_preserves_trade_count(synthetic_data):
    X, y = synthetic_data
    model_low = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10, threshold_bps=0.0)
    model_high = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10, threshold_bps=10.0)
    model_low.fit(X, y)
    model_high.fit(X, y)
    n_low = int(model_low.predict_signal(X).sum())
    n_high = int(model_high.predict_signal(X).sum())
    assert n_high <= n_low


def test_calibrate_threshold_sets_attribute(synthetic_data):
    X, y = synthetic_data
    X_val, y_val = X[:50], y[:50]
    model = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10)
    model.fit(X[50:], y[50:])
    t = model.calibrate_threshold(X_val, y_val, min_trades=5)
    assert isinstance(t, float)
    assert model.threshold_bps == t


def test_predict_proba_shape(synthetic_data):
    X, y = synthetic_data
    model = ExpectedNetProfitRegressor(base_model="lgbm", n_estimators=10)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_unfitted_model_raises(synthetic_data):
    X, _ = synthetic_data
    model = ExpectedNetProfitRegressor()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X)


def test_xgb_base_model(synthetic_data):
    X, y = synthetic_data
    model = ExpectedNetProfitRegressor(base_model="xgb", n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_invalid_base_model(synthetic_data):
    X, y = synthetic_data
    model = ExpectedNetProfitRegressor(base_model="random_forest_is_not_supported")
    with pytest.raises(ValueError, match="Unknown base_model"):
        model.fit(X, y)
