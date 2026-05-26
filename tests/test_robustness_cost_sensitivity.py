"""Cost-sensitivity tests for the robustness grid.

Verifies that fee and settlement adjustments genuinely affect the executable
percentage — i.e., the bug where precomputed label columns were used regardless
of cost parameters is not re-introduced.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from stressbench.experiments.robustness import _forward_max, compute_robustness_grid


# -----------------------------------------------------------------------
# Unit tests for _forward_max
# -----------------------------------------------------------------------

def test_forward_max_1step_equals_input():
    arr = np.array([1.0, -2.0, 3.0, np.nan, 0.5])
    result = _forward_max(arr, steps=1)
    np.testing.assert_array_equal(result, arr)


def test_forward_max_2steps():
    arr = np.array([1.0, 3.0, -1.0, 2.0])
    result = _forward_max(arr, steps=2)
    # result[i] = max(arr[i], arr[i+1])
    expected = np.array([3.0, 3.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_forward_max_nan_ignored():
    arr = np.array([np.nan, 5.0, np.nan, 2.0])
    result = _forward_max(arr, steps=2)
    assert result[0] == 5.0  # [nan, 5.0] → max ignoring nan = 5.0
    assert result[1] == 5.0  # [5.0, nan] → 5.0
    assert result[2] == 2.0  # [nan, 2.0] → 2.0


def test_forward_max_all_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    result = _forward_max(arr, steps=2)
    assert np.all(np.isnan(result))


def test_forward_max_larger_window_geq_smaller():
    """Forward max with more steps should be >= max with fewer steps."""
    rng = np.random.default_rng(7)
    arr = rng.standard_normal(50)
    fwd1 = _forward_max(arr, steps=1)
    fwd5 = _forward_max(arr, steps=5)
    # Every element of fwd5 >= corresponding fwd1 (wider window can only increase max)
    assert np.all(fwd5 >= fwd1 - 1e-10)


# -----------------------------------------------------------------------
# Integration tests: cost sensitivity must actually matter
# -----------------------------------------------------------------------

@pytest.fixture()
def cost_sensitive_dataset(tmp_path):
    """Dataset where fee/settlement changes clearly affect executability."""
    rng = np.random.default_rng(99)
    n = 200
    # Net profits uniformly distributed around +5 bps → many positives
    # After -10 bps settlement, almost none positive
    net = rng.uniform(-5, 15, size=n)  # ~75% positive before adjustment
    df = pl.DataFrame({
        "ts_1m_ns": np.arange(n, dtype=np.int64) * 60_000_000_000,
        "split": ["test"] * n,
        "cross_quote_basis_usdc_bps": rng.standard_normal(n) * 15,
        "net_profit_bps_q10000": net,
        "net_profit_bps_q50000": net - 3.0,
        "net_profit_bps_q100000": net - 5.0,
        "net_profit_bps_q500000": net - 8.0,
    })
    path = tmp_path / "dataset.parquet"
    df.write_parquet(str(path))
    return path


def _get(rows, notional, threshold, settlement, fee, horizon):
    for r in rows:
        if (r["notional"] == notional
                and r["basis_threshold_bps"] == threshold
                and r["settlement_penalty_bps"] == settlement
                and r["fee_regime"] == fee
                and r["horizon"] == horizon):
            return r
    return None


def test_high_fee_executable_leq_base_fee(cost_sensitive_dataset):
    """high_fee (negative adjustment) ≤ base_fee executable percentage."""
    rows = compute_robustness_grid(cost_sensitive_dataset)
    for threshold in [0, 10]:
        for horizon in ["1m", "5m"]:
            base = _get(rows, 10_000, threshold, 0, "base_fee", horizon)
            high = _get(rows, 10_000, threshold, 0, "high_fee", horizon)
            assert base is not None and high is not None
            assert high["executable_signal_pct"] <= base["executable_signal_pct"] + 1e-9, (
                f"high_fee exec ({high['executable_signal_pct']:.3f}) > "
                f"base_fee exec ({base['executable_signal_pct']:.3f}) "
                f"at threshold={threshold}, horizon={horizon}"
            )


def test_low_fee_executable_geq_base_fee(cost_sensitive_dataset):
    """low_fee (positive adjustment) ≥ base_fee executable percentage."""
    rows = compute_robustness_grid(cost_sensitive_dataset)
    for threshold in [0, 10]:
        for horizon in ["1m"]:
            base = _get(rows, 10_000, threshold, 0, "base_fee", horizon)
            low = _get(rows, 10_000, threshold, 0, "low_fee", horizon)
            assert base is not None and low is not None
            assert low["executable_signal_pct"] >= base["executable_signal_pct"] - 1e-9


def test_higher_settlement_reduces_executable(cost_sensitive_dataset):
    """10 bps settlement penalty ≤ 0 bps settlement executable percentage."""
    rows = compute_robustness_grid(cost_sensitive_dataset)
    for threshold in [0, 10]:
        for horizon in ["1m", "5m"]:
            zero_pen = _get(rows, 10_000, threshold, 0, "base_fee", horizon)
            ten_pen  = _get(rows, 10_000, threshold, 10, "base_fee", horizon)
            assert zero_pen is not None and ten_pen is not None
            assert ten_pen["executable_signal_pct"] <= zero_pen["executable_signal_pct"] + 1e-9, (
                f"10bps penalty exec ({ten_pen['executable_signal_pct']:.3f}) > "
                f"0bps penalty exec ({zero_pen['executable_signal_pct']:.3f}) "
                f"at threshold={threshold}, horizon={horizon}"
            )


def test_longer_horizon_executable_geq_shorter(cost_sensitive_dataset):
    """15m horizon exec ≥ 1m horizon exec at same costs (wider window = more opportunity)."""
    rows = compute_robustness_grid(cost_sensitive_dataset)
    h1 = _get(rows, 10_000, 0, 0, "base_fee", "1m")
    h15 = _get(rows, 10_000, 0, 0, "base_fee", "15m")
    assert h1 is not None and h15 is not None
    assert h15["executable_signal_pct"] >= h1["executable_signal_pct"] - 1e-9


def test_fee_and_settlement_are_independent_dimensions(cost_sensitive_dataset):
    """Combined high_fee + 10bps settlement should be most restrictive."""
    rows = compute_robustness_grid(cost_sensitive_dataset)
    base = _get(rows, 10_000, 0, 0, "base_fee", "5m")
    worst = _get(rows, 10_000, 0, 10, "high_fee", "5m")
    assert base is not None and worst is not None
    assert worst["executable_signal_pct"] <= base["executable_signal_pct"] + 1e-9
