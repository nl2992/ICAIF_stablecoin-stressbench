"""Shared pytest fixtures for Stablecoin StressBench tests."""

from __future__ import annotations

import json

import polars as pl
import pytest


@pytest.fixture()
def sample_trade_payload_binance() -> dict:
    return {
        "e": "trade",
        "E": 1704067200000,
        "s": "USDCUSDT",
        "t": 123456789,
        "p": "0.9998",
        "q": "50000.00",
        "b": 1000001,
        "a": 1000002,
        "T": 1704067200000,
        "m": False,
        "M": True,
    }


@pytest.fixture()
def sample_depth_payload_binance() -> dict:
    return {
        "e": "depthUpdate",
        "E": 1704067200000,
        "s": "USDCUSDT",
        "T": 1704067200000,
        "b": [["0.9998", "50000"], ["0.9997", "100000"]],
        "a": [["0.9999", "30000"], ["1.0000", "80000"]],
    }


@pytest.fixture()
def sample_trade_df_binance(sample_trade_payload_binance) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_receive_ns": [1704067200_000_000_000],
            "venue_id": ["binance"],
            "symbol": ["USDCUSDT"],
            "payload": [json.dumps(sample_trade_payload_binance)],
            "payload_hash": ["abc123"],
        }
    )


@pytest.fixture()
def sample_depth_df_binance(sample_depth_payload_binance) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_receive_ns": [1704067200_000_000_000],
            "venue_id": ["binance"],
            "symbol": ["USDCUSDT"],
            "payload": [json.dumps(sample_depth_payload_binance)],
            "payload_hash": ["def456"],
        }
    )


@pytest.fixture()
def simple_order_book():
    from stressbench.book.order_book import OrderBook

    book = OrderBook()
    book.apply_snapshot(
        bids=[("1.0001", "50000"), ("1.0000", "100000"), ("0.9999", "200000")],
        asks=[("1.0002", "30000"), ("1.0003", "80000"), ("1.0004", "150000")],
    )
    return book
