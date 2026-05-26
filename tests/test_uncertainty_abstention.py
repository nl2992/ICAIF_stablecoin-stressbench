"""Tests for uncertainty-aware abstention models."""

from __future__ import annotations

import numpy as np
import pytest

from stressbench.experiments.uncertainty import (
    BootstrapEnsemble,
    QuantileNetProfitModel,
    abstention_sweep,
)


@pytest.fixture()
def small_data():
    rng = np.random.default_rng(1)
    n = 150
    X = rng.standard_normal((n, 4))
    y = (rng.standard_normal(n) * 5 + 2 * X[:, 0]).astype(np.float32)
    y_cls = (y > 0).astype(np.int8)
    return X, y, y_cls


def _logistic_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=200, random_state=0)


def test_bootstrap_predict_mean_std_shape(small_data):
    X, _, y_cls = small_data
    ens = BootstrapEnsemble(base_model_factory=_logistic_factory, n_bootstrap=5)
    ens.fit(X, y_cls)
    mean, std = ens.predict_mean_std(X)
    assert mean.shape == (len(X),)
    assert std.shape == (len(X),)
    assert (std >= 0).all()


def test_bootstrap_signal_is_binary(small_data):
    X, _, y_cls = small_data
    ens = BootstrapEnsemble(base_model_factory=_logistic_factory, n_bootstrap=5)
    ens.fit(X, y_cls)
    signal = ens.predict_signal(X, k=0.0)
    assert set(np.unique(signal)).issubset({0, 1})


def test_increasing_k_reduces_or_preserves_trade_count(small_data):
    X, _, y_cls = small_data
    ens = BootstrapEnsemble(base_model_factory=_logistic_factory, n_bootstrap=5)
    ens.fit(X, y_cls)
    counts = [int(ens.predict_signal(X, k=k).sum()) for k in [0.0, 0.5, 1.0, 2.0]]
    # Strictly non-increasing as k grows
    for i in range(len(counts) - 1):
        assert counts[i + 1] <= counts[i], f"k sweep counts not monotone: {counts}"


def test_uncertainty_signal_never_exceeds_k0_signal(small_data):
    """With k > 0, trades ⊆ trades with k=0."""
    X, _, y_cls = small_data
    ens = BootstrapEnsemble(base_model_factory=_logistic_factory, n_bootstrap=5)
    ens.fit(X, y_cls)
    sig_0 = ens.predict_signal(X, k=0.0)
    sig_2 = ens.predict_signal(X, k=2.0)
    # Every trade at k=2 must also be a trade at k=0
    assert np.all((sig_2 - sig_0) <= 0)


def test_quantile_model_fits_and_predicts(small_data):
    X, y, _ = small_data
    qm = QuantileNetProfitModel(base_model="lgbm", quantiles=(0.1, 0.5, 0.9))
    qm.fit(X, y)
    qs = qm.predict_quantiles(X)
    assert set(qs.keys()) == {0.1, 0.5, 0.9}
    for arr in qs.values():
        assert arr.shape == (len(X),)


def test_quantile_ordering(small_data):
    """10th percentile ≤ 50th ≤ 90th on average."""
    X, y, _ = small_data
    qm = QuantileNetProfitModel(base_model="lgbm", quantiles=(0.1, 0.5, 0.9))
    qm.fit(X, y)
    qs = qm.predict_quantiles(X)
    assert np.mean(qs[0.1]) <= np.mean(qs[0.5]) + 1e-3
    assert np.mean(qs[0.5]) <= np.mean(qs[0.9]) + 1e-3


def test_quantile_signal_binary(small_data):
    X, y, _ = small_data
    qm = QuantileNetProfitModel(base_model="lgbm", quantiles=(0.1, 0.5, 0.9))
    qm.fit(X, y)
    sig = qm.predict_signal(X, trade_quantile=0.1)
    assert set(np.unique(sig)).issubset({0, 1})


def test_abstention_sweep_monotone_trade_count(small_data):
    """Higher k → fewer or equal trades in the sweep."""
    X, y, _ = small_data
    rng = np.random.default_rng(5)
    mean_preds = rng.uniform(0.3, 0.7, size=len(X))
    std_preds = rng.uniform(0.0, 0.2, size=len(X))
    rows = abstention_sweep(mean_preds, std_preds, y, k_values=(0.0, 0.5, 1.0, 1.5, 2.0))
    counts = [r["n_trades"] for r in rows]
    for i in range(len(counts) - 1):
        assert counts[i + 1] <= counts[i]


def test_abstention_sweep_output_schema(small_data):
    _, y, _ = small_data
    n = len(y)
    mean_preds = np.full(n, 0.6)
    std_preds = np.full(n, 0.1)
    rows = abstention_sweep(mean_preds, std_preds, y)
    required = {"k", "n_trades", "net_bps_captured", "hit_rate_above_cost", "total_pnl_bps"}
    for row in rows:
        assert required.issubset(set(row.keys()))
