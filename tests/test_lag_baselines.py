"""Tests for lag baseline robustness.

Verifies that LastValueBaseline and AR1Baseline:
1. Do NOT silently assume X[:, -1] is the lag target
2. Accept explicit lag_col_idx without error
3. Auto-detect the most correlated column when lag_col_idx is None
4. Produce deterministic predictions given the same lag column
5. Raise RuntimeError on predict() before fit()
6. get_lag_col_idx() returns correct indices for known feature lists
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from stressbench.features.lags import LAG_COLUMN_PREFERENCE, TASK_LAG_COLUMNS, get_lag_col_idx
from stressbench.models.baselines import AR1Baseline, LastValueBaseline


# ── Test 1: explicit lag_col_idx is respected ─────────────────────────────


def test_last_value_explicit_col_idx() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = X[:, 2].copy()  # col 2 IS the lag
    # Put different data in col -1 to prove it's not used
    X[:, -1] = rng.standard_normal(100)

    model = LastValueBaseline(lag_col_idx=2)
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_array_equal(preds, X[:, 2])


def test_ar1_explicit_col_idx() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 5))
    y = 0.8 * X[:, 1] + rng.standard_normal(100) * 0.01
    X[:, -1] = rng.standard_normal(100)  # noise in last col

    model = AR1Baseline(lag_col_idx=1)
    model.fit(X, y)
    preds = model.predict(X)
    # predictions should be alpha + beta * X[:, 1]
    expected = model._alpha + model._beta * X[:, 1]
    np.testing.assert_allclose(preds, expected, rtol=1e-10)


# ── Test 2: auto-detect selects most-correlated column ───────────────────


def test_last_value_auto_detects_best_col() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 6))
    # col 3 has high correlation with y
    y = X[:, 3] * 0.95 + rng.standard_normal(200) * 0.1
    X[:, -1] = rng.standard_normal(200)  # noise

    model = LastValueBaseline()  # lag_col_idx=None
    model.fit(X, y)
    assert model._lag_col_idx == 3, (
        f"Expected auto-detected lag_col_idx=3, got {model._lag_col_idx}"
    )


def test_ar1_auto_detects_best_col() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 6))
    y = X[:, 0] * 0.7 + rng.standard_normal(200) * 0.05
    X[:, -1] = rng.standard_normal(200)  # noise

    model = AR1Baseline()
    model.fit(X, y)
    assert model._lag_col_idx == 0, (
        f"Expected auto-detected lag_col_idx=0, got {model._lag_col_idx}"
    )


# ── Test 3: predict before fit raises RuntimeError ───────────────────────


def test_last_value_predict_before_fit_raises() -> None:
    model = LastValueBaseline()
    X = np.ones((5, 3))
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_ar1_predict_before_fit_raises() -> None:
    model = AR1Baseline()
    X = np.ones((5, 3))
    with pytest.raises(RuntimeError):
        model.predict(X)


# ── Test 4: get_lag_col_idx returns correct index ─────────────────────────


def test_get_lag_col_idx_task_specific() -> None:
    feature_names = [
        "spread_bps",
        "cross_quote_basis_usdc_bps",
        "cross_quote_basis_maxabs_bps",
        "depth_bid",
    ]
    idx = get_lag_col_idx(feature_names, task="basis_usdc_1m_gt10bps")
    assert idx == 1  # cross_quote_basis_usdc_bps is at index 1


def test_get_lag_col_idx_preference_fallback() -> None:
    # Task-specific column not present; should fall back to preference list
    feature_names = [
        "spread_bps",
        "cross_quote_basis_maxabs_bps",
        "depth_bid",
    ]
    idx = get_lag_col_idx(feature_names, task="basis_usdc_1m_gt10bps")
    # cross_quote_basis_usdc_bps not present; next in preference is cross_quote_basis_maxabs_bps
    assert idx == 1  # cross_quote_basis_maxabs_bps


def test_get_lag_col_idx_warns_on_fallback_to_zero() -> None:
    import warnings

    feature_names = ["unknown_col_a", "unknown_col_b"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        idx = get_lag_col_idx(feature_names, task="basis_usdc_1m_gt10bps")
        assert idx == 0
        assert len(w) == 1
        assert "Falling back to column 0" in str(w[0].message)


# ── Test 5: TASK_LAG_COLUMNS covers common tasks ─────────────────────────


def test_task_lag_columns_covers_primary_tasks() -> None:
    primary_tasks = [
        "basis_usdc_1m_gt10bps",
        "basis_usdc_5m_gt25bps",
        "executable_arb_q10000_5m",
    ]
    for task in primary_tasks:
        assert task in TASK_LAG_COLUMNS, f"Missing task in TASK_LAG_COLUMNS: {task}"


# ── Test 6: no X[:, -1] assumption — wrong last column gives wrong result ─


def test_ar1_wrong_last_col_gives_wrong_result() -> None:
    """Confirm that if X[:,-1] is noise (not the lag), using explicit col gives
    different (correct) predictions vs the old X[:,-1] assumption.

    Requires a genuine AR(1) process so that AR1Baseline.fit() produces a
    non-trivial beta; with i.i.d. y both models would get beta≈0 and produce
    the same near-constant forecast regardless of which column is used.
    """
    rng = np.random.default_rng(42)
    n = 300
    # Build a stationary AR(1) process: y[t] = 0.7 * y[t-1] + eps
    eps = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + eps[t]

    # X[:, 0] = actual lagged target y(t-1)  ← correct lag column
    lag_y = np.concatenate([[0.0], y[:-1]])
    noise_col = rng.standard_normal(n)
    X = np.column_stack([lag_y, rng.standard_normal(n), noise_col])

    # Explicit correct col (lag of y)
    model_correct = AR1Baseline(lag_col_idx=0)
    model_correct.fit(X, y)
    preds_correct = model_correct.predict(X)

    # Explicit wrong col (last = pure noise)
    model_wrong = AR1Baseline(lag_col_idx=2)
    model_wrong.fit(X, y)
    preds_wrong = model_wrong.predict(X)

    # Both models share the same alpha/beta (fitted from y's own autocorrelation).
    # beta ≈ 0.7, so predictions = 0.7 * col.  With col 0 = y(t-1) and col 2 =
    # independent noise, the mean absolute difference should be well above 0.1.
    diff = np.mean(np.abs(preds_correct - preds_wrong))
    assert diff > 0.1, (
        "Using wrong lag column should produce noticeably different predictions; "
        f"mean |diff|={diff:.4f}"
    )
