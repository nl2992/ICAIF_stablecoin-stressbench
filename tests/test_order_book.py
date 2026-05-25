"""Tests for the in-memory order book."""

from __future__ import annotations

import pytest

from stressbench.book.order_book import OrderBook


def test_snapshot_initialises_book(simple_order_book):
    assert simple_order_book.best_bid() == pytest.approx(1.0001)
    assert simple_order_book.best_ask() == pytest.approx(1.0002)


def test_mid_price(simple_order_book):
    expected = (1.0001 + 1.0002) / 2
    assert simple_order_book.mid() == pytest.approx(expected)


def test_spread(simple_order_book):
    assert simple_order_book.spread() == pytest.approx(0.0001)


def test_spread_bps(simple_order_book):
    mid = (1.0001 + 1.0002) / 2
    expected = 0.0001 / mid * 10_000
    assert simple_order_book.spread_bps() == pytest.approx(expected, rel=1e-4)


def test_update_removes_level(simple_order_book):
    simple_order_book.apply_update("bid", 1.0001, 0)
    assert simple_order_book.best_bid() == pytest.approx(1.0000)


def test_update_adds_level(simple_order_book):
    simple_order_book.apply_update("ask", 1.00015, 5000)
    assert simple_order_book.best_ask() == pytest.approx(1.00015)


def test_crossed_book_detection():
    book = OrderBook()
    book.apply_snapshot(
        bids=[("1.0003", "10000")],
        asks=[("1.0002", "10000")],
    )
    assert book.is_crossed()


def test_not_crossed_book(simple_order_book):
    assert not simple_order_book.is_crossed()


def test_depth_within_bps(simple_order_book):
    # Bids within 1bp of best_bid=1.0001: threshold = 1.0001 * (1 - 1/10000) = 1.00000
    depth = simple_order_book.depth_within_bps("bid", 1.0)
    assert depth == pytest.approx(50000.0)


def test_imbalance(simple_order_book):
    imb = simple_order_book.imbalance(1.0)
    assert imb is not None
    assert -1.0 <= imb <= 1.0


def test_empty_book_returns_none():
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid() is None
    assert book.spread() is None
    assert book.spread_bps() is None
    assert book.imbalance() is None


def test_sorted_bids_descending(simple_order_book):
    bids = simple_order_book.sorted_bids()
    prices = [p for p, _ in bids]
    assert prices == sorted(prices, reverse=True)


def test_sorted_asks_ascending(simple_order_book):
    asks = simple_order_book.sorted_asks()
    prices = [p for p, _ in asks]
    assert prices == sorted(prices)
