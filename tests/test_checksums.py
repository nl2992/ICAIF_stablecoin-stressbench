"""Tests for order-book checksum validation."""

from __future__ import annotations

from stressbench.book.checksums import kraken_checksum, validate_kraken_checksum
from stressbench.book.order_book import OrderBook


def test_kraken_checksum_deterministic(simple_order_book):
    cs1 = kraken_checksum(simple_order_book)
    cs2 = kraken_checksum(simple_order_book)
    assert cs1 == cs2


def test_kraken_checksum_changes_on_update(simple_order_book):
    cs_before = kraken_checksum(simple_order_book)
    simple_order_book.apply_update("bid", 1.0001, 99999)
    cs_after = kraken_checksum(simple_order_book)
    assert cs_before != cs_after


def test_validate_kraken_checksum_correct(simple_order_book):
    cs = kraken_checksum(simple_order_book)
    assert validate_kraken_checksum(simple_order_book, cs)


def test_validate_kraken_checksum_wrong(simple_order_book):
    assert not validate_kraken_checksum(simple_order_book, 0)


def test_checksum_empty_book():
    book = OrderBook()
    cs = kraken_checksum(book)
    assert isinstance(cs, int)
    assert 0 <= cs <= 0xFFFFFFFF
