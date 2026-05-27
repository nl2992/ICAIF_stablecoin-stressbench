"""Order-book checksum validation utilities.

Implements Kraken-style CRC32 checksum validation for the top book levels.
"""

from __future__ import annotations

import binascii

from stressbench.book.order_book import OrderBook


def kraken_checksum(book: OrderBook, depth: int = 10) -> int:
    """Compute the Kraken-style CRC32 checksum for the top ``depth`` levels.

    Kraken's checksum is computed over the top 10 bid and ask price/quantity
    pairs, formatted as strings with decimal points removed and leading zeros
    stripped.

    Args:
        book: Reconstructed :class:`~stressbench.book.order_book.OrderBook`.
        depth: Number of levels to include (default 10).

    Returns:
        32-bit unsigned integer CRC32 checksum.
    """

    def fmt(val: float) -> str:
        return f"{val:.5f}".replace(".", "").lstrip("0") or "0"

    parts = []
    for price, qty in book.sorted_bids()[:depth]:
        parts.append(fmt(price))
        parts.append(fmt(qty))
    for price, qty in book.sorted_asks()[:depth]:
        parts.append(fmt(price))
        parts.append(fmt(qty))

    raw = "".join(parts).encode("ascii")
    return binascii.crc32(raw) & 0xFFFFFFFF


def validate_kraken_checksum(book: OrderBook, expected: int, depth: int = 10) -> bool:
    """Validate a Kraken book checksum.

    Args:
        book: Reconstructed order book.
        expected: Expected checksum value from the exchange message.
        depth: Number of levels used in the checksum computation.

    Returns:
        ``True`` if the computed checksum matches the expected value.
    """
    computed = kraken_checksum(book, depth)
    return computed == expected
