"""Tests for VWAP computation and net profit calculation."""

from __future__ import annotations

import pytest

from stressbench.book.order_book import OrderBook
from stressbench.book.vwap import (
    executable_buy_vwap,
    executable_sell_vwap,
    gross_spread_bps,
    net_profit_bps,
    walk_book,
)


@pytest.fixture()
def deep_book():
    book = OrderBook()
    book.apply_snapshot(
        bids=[("1.0001", "100000"), ("1.0000", "200000"), ("0.9999", "300000")],
        asks=[("1.0002", "100000"), ("1.0003", "200000"), ("1.0004", "300000")],
    )
    return book


def test_walk_book_single_level():
    levels = [(1.0002, 100000.0)]
    vwap = walk_book(levels, qty=50000.0)
    assert vwap == pytest.approx(1.0002)


def test_walk_book_multiple_levels():
    levels = [(1.0002, 50000.0), (1.0003, 50000.0)]
    vwap = walk_book(levels, qty=100000.0)
    assert vwap == pytest.approx((1.0002 * 50000 + 1.0003 * 50000) / 100000)


def test_walk_book_insufficient_depth():
    levels = [(1.0002, 10000.0)]
    vwap = walk_book(levels, qty=50000.0)
    assert vwap is None


def test_executable_buy_vwap(deep_book):
    vwap = executable_buy_vwap(deep_book, qty=50000.0)
    assert vwap == pytest.approx(1.0002)


def test_executable_sell_vwap(deep_book):
    vwap = executable_sell_vwap(deep_book, qty=50000.0)
    assert vwap == pytest.approx(1.0001)


def test_gross_spread_bps_positive(deep_book):
    # Buy from deep_book at ask, sell at bid: should be negative (buy > sell)
    spread = gross_spread_bps(deep_book, deep_book, qty=50000.0)
    assert spread is not None
    # Selling at bid (1.0001) vs buying at ask (1.0002): negative spread
    assert spread < 0


def test_net_profit_bps_with_fees(deep_book):
    net = net_profit_bps(
        buy_book=deep_book,
        sell_book=deep_book,
        qty=50000.0,
        taker_fee_buy_bps=10.0,
        taker_fee_sell_bps=10.0,
        withdrawal_fee_usd=1.0,
        gas_fee_usd=5.0,
        settlement_delay_penalty_bps=2.0,
    )
    assert net is not None
    # Should be negative since we're buying and selling on the same book
    assert net < 0


def test_net_profit_bps_none_on_empty_book():
    empty = OrderBook()
    result = net_profit_bps(empty, empty, qty=50000.0)
    assert result is None
