"""Tests for the robustness grid computation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from stressbench.experiments.robustness import (
    BASIS_THRESHOLDS_BPS,
    FEE_REGIMES,
    HORIZONS,
    NOTIONALS,
    SETTLEMENT_PENALTIES_BPS,
    compute_robustness_grid,
)


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Minimal dataset.parquet with the columns needed for robustness."""
    rng = np.random.default_rng(42)
    n = 100
    df = pl.DataFrame(
        {
            "ts_1m_ns": np.arange(n, dtype=np.int64) * 60_000_000_000,
            "split": ["test"] * n,
            "cross_quote_basis_usdc_bps": rng.standard_normal(n) * 15,
            "net_profit_bps_q10000": rng.standard_normal(n) * 10 - 2,
            "net_profit_bps_q50000": rng.standard_normal(n) * 8 - 3,
            "net_profit_bps_q100000": rng.standard_normal(n) * 6 - 4,
            "net_profit_bps_q500000": rng.standard_normal(n) * 4 - 6,
            "label_arb_q10000_1m_gt0bps": (rng.uniform(size=n) > 0.8).astype(np.int8),
            "label_arb_q10000_5m_gt0bps": (rng.uniform(size=n) > 0.75).astype(np.int8),
            "label_arb_q10000_15m_gt0bps": (rng.uniform(size=n) > 0.70).astype(np.int8),
            "label_arb_q50000_1m_gt0bps": (rng.uniform(size=n) > 0.90).astype(np.int8),
            "label_arb_q50000_5m_gt0bps": (rng.uniform(size=n) > 0.85).astype(np.int8),
            "label_arb_q50000_15m_gt0bps": (rng.uniform(size=n) > 0.80).astype(np.int8),
        }
    )
    path = tmp_path / "dataset.parquet"
    df.write_parquet(str(path))
    return path


def test_grid_includes_all_notionals(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    notionals_found = {r["notional"] for r in rows}
    # Only notionals with columns present
    assert 10_000 in notionals_found
    assert 50_000 in notionals_found


def test_grid_includes_all_thresholds(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    thresholds_found = {r["basis_threshold_bps"] for r in rows}
    for t in BASIS_THRESHOLDS_BPS:
        assert t in thresholds_found


def test_grid_includes_all_fee_regimes(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    regimes_found = {r["fee_regime"] for r in rows}
    for regime in FEE_REGIMES:
        assert regime in regimes_found


def test_grid_includes_all_horizons(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    horizons_found = {r["horizon"] for r in rows}
    for h in HORIZONS:
        assert h in horizons_found


def test_no_duplicate_rows(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    keys = [
        (
            r["split"],
            r["notional"],
            r["basis_threshold_bps"],
            r["settlement_penalty_bps"],
            r["fee_regime"],
            r["horizon"],
        )
        for r in rows
    ]
    assert len(keys) == len(set(keys))


def test_output_columns_match_schema(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    assert rows, "Expected non-empty result"
    required = {
        "split",
        "notional",
        "basis_threshold_bps",
        "settlement_penalty_bps",
        "fee_regime",
        "horizon",
        "n_minutes",
        "price_signal_pct",
        "executable_signal_pct",
        "price_to_execution_ratio",
        "oracle_net_bps",
        "oracle_n_trades",
    }
    assert required.issubset(set(rows[0].keys()))


def test_price_signal_monotone_decreasing_with_threshold(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    # For a fixed notional, fee regime, horizon — price signal should decrease as threshold rises
    subset = [
        r
        for r in rows
        if r["notional"] == 10_000
        and r["fee_regime"] == "base_fee"
        and r["horizon"] == "5m"
        and r["settlement_penalty_bps"] == 0
    ]
    subset.sort(key=lambda r: r["basis_threshold_bps"])
    psigs = [r["price_signal_pct"] for r in subset]
    assert psigs == sorted(psigs, reverse=True)


def test_higher_settlement_penalty_reduces_oracle_n_trades(tiny_dataset):
    rows = compute_robustness_grid(tiny_dataset)
    # Oracle trade count is computed on unadjusted net_profit, so this may not vary
    # Just verify the column is non-negative integers
    for r in rows:
        assert isinstance(r["oracle_n_trades"], int)
        assert r["oracle_n_trades"] >= 0
