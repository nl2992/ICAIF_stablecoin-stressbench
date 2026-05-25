"""Tests for trade normalisation."""

from __future__ import annotations

import json

import polars as pl
import pytest

from stressbench.normalization.normalize_trades import (
    normalize_binance_trades,
    normalize_coinbase_trades,
    normalize_kraken_trades,
)


def test_normalize_binance_trades_basic(sample_trade_df_binance):
    result = normalize_binance_trades(sample_trade_df_binance)
    assert not result.is_empty()
    assert "trade_id" in result.columns
    assert "price" in result.columns
    assert "size" in result.columns
    assert "side" in result.columns
    assert "venue_id" in result.columns
    assert "instrument_id" in result.columns


def test_normalize_binance_trades_price(sample_trade_df_binance):
    result = normalize_binance_trades(sample_trade_df_binance)
    assert result["price"][0] == pytest.approx(0.9998)


def test_normalize_binance_trades_size(sample_trade_df_binance):
    result = normalize_binance_trades(sample_trade_df_binance)
    assert result["size"][0] == pytest.approx(50000.0)


def test_normalize_binance_trades_side(sample_trade_df_binance):
    result = normalize_binance_trades(sample_trade_df_binance)
    # m=False means the buyer was the taker → "buy"
    assert result["side"][0] == "buy"


def test_normalize_binance_trades_venue(sample_trade_df_binance):
    result = normalize_binance_trades(sample_trade_df_binance)
    assert result["venue_id"][0] == "binance"


def test_normalize_binance_trades_empty_df():
    empty = pl.DataFrame(
        {
            "ts_receive_ns": pl.Series([], dtype=pl.Int64),
            "venue_id": pl.Series([], dtype=pl.Utf8),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "payload": pl.Series([], dtype=pl.Utf8),
            "payload_hash": pl.Series([], dtype=pl.Utf8),
        }
    )
    result = normalize_binance_trades(empty)
    assert result.is_empty()


def test_normalize_coinbase_trades_basic():
    payload = {
        "type": "match",
        "trade_id": 999,
        "product_id": "USDC-USD",
        "price": "0.9999",
        "size": "25000.00",
        "side": "buy",
        "time": "2024-01-01T00:00:00.000000Z",
    }
    df = pl.DataFrame(
        {
            "ts_receive_ns": [1704067200_000_000_000],
            "venue_id": ["coinbase"],
            "symbol": ["USDC-USD"],
            "payload": [json.dumps(payload)],
            "payload_hash": ["abc"],
        }
    )
    result = normalize_coinbase_trades(df)
    assert not result.is_empty()
    assert result["price"][0] == pytest.approx(0.9999)


def test_normalize_kraken_trades_basic():
    payload = {
        "channel": "trade",
        "data": [
            {
                "symbol": "USDC/USD",
                "side": "sell",
                "price": 0.9997,
                "qty": 10000.0,
                "trade_id": 777,
                "timestamp": "2024-01-01T00:00:00.000000Z",
            }
        ],
    }
    df = pl.DataFrame(
        {
            "ts_receive_ns": [1704067200_000_000_000],
            "venue_id": ["kraken"],
            "symbol": ["USDC/USD"],
            "payload": [json.dumps(payload)],
            "payload_hash": ["xyz"],
        }
    )
    result = normalize_kraken_trades(df)
    assert not result.is_empty()
    assert result["side"][0] == "sell"
