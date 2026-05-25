"""Tests for feature computation modules."""

from __future__ import annotations

import pytest

from stressbench.features.basis import (
    compute_cross_quote_basis_bps,
    compute_fragmentation_features,
    compute_stablecoin_usd_price,
)
from stressbench.features.microstructure import compute_book_snapshot_features


def test_stablecoin_usd_price_direct():
    price = compute_stablecoin_usd_price(0.9998, "USD")
    assert price == pytest.approx(0.9998)


def test_stablecoin_usd_price_via_usdt():
    price = compute_stablecoin_usd_price(0.9998, "USDT", usdt_usd_ref=0.9999)
    assert price == pytest.approx(0.9998 * 0.9999)


def test_stablecoin_usd_price_via_usdc():
    price = compute_stablecoin_usd_price(1.0001, "USDC", usdc_usd_ref=0.9997)
    assert price == pytest.approx(1.0001 * 0.9997)


def test_stablecoin_usd_price_missing_ref():
    price = compute_stablecoin_usd_price(0.9998, "USDT", usdt_usd_ref=None)
    assert price is None


def test_stablecoin_usd_price_none_mid():
    price = compute_stablecoin_usd_price(None, "USD")
    assert price is None


def test_cross_quote_basis_bps_zero():
    basis = compute_cross_quote_basis_bps(50000.0, 50000.0)
    assert basis == pytest.approx(0.0)


def test_cross_quote_basis_bps_positive():
    basis = compute_cross_quote_basis_bps(50000.0, 50005.0)
    assert basis == pytest.approx(1.0)  # 5/50000 * 10000 = 1 bps


def test_cross_quote_basis_bps_none_input():
    assert compute_cross_quote_basis_bps(None, 50000.0) is None
    assert compute_cross_quote_basis_bps(50000.0, None) is None
    assert compute_cross_quote_basis_bps(0.0, 50000.0) is None


def test_fragmentation_features_single_venue():
    result = compute_fragmentation_features({"binance": 1.0001})
    assert result["num_active_venues"] == 1
    assert result["mid_dispersion_bps"] == pytest.approx(0.0)


def test_fragmentation_features_multiple_venues():
    result = compute_fragmentation_features(
        {"binance": 1.0001, "coinbase": 1.0003, "kraken": 0.9998}
    )
    assert result["num_active_venues"] == 3
    assert result["max_minus_min_bps"] is not None
    assert result["max_minus_min_bps"] > 0


def test_fragmentation_features_empty():
    result = compute_fragmentation_features({})
    assert result["num_active_venues"] == 0
    assert result["mid_dispersion_bps"] is None


def test_book_snapshot_features(simple_order_book):
    feat = compute_book_snapshot_features(
        book=simple_order_book,
        ts_ns=1704067200_000_000_000,
        venue_id="binance",
        instrument_id="binance:USDCUSDT",
        trade_count=10,
        trade_volume=500000.0,
    )
    assert feat["mid"] is not None
    assert feat["spread_bps"] is not None
    assert feat["data_quality_score"] == pytest.approx(1.0)
    assert feat["trade_count"] == 10
    assert feat["trade_volume"] == pytest.approx(500000.0)


def test_book_snapshot_features_resync_penalty(simple_order_book):
    feat = compute_book_snapshot_features(
        book=simple_order_book,
        ts_ns=1704067200_000_000_000,
        venue_id="binance",
        instrument_id="binance:USDCUSDT",
        is_resync=True,
    )
    assert feat["data_quality_score"] < 1.0
