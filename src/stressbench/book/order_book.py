"""In-memory order-book reconstruction.

Supports snapshot initialisation and incremental update application.
Provides best bid/ask, mid price, spread, and sorted level access.
"""

from __future__ import annotations

from typing import Iterable


class OrderBook:
    """In-memory limit order book for a single instrument.

    Bids are stored as ``{price: size}`` and asks likewise.
    Prices and sizes are stored as Python floats.

    Example::

        book = OrderBook()
        book.apply_snapshot(bids=[("1.0001", "50000"), ("1.0000", "100000")],
                            asks=[("1.0002", "30000"), ("1.0003", "80000")])
        book.best_bid()   # 1.0001
        book.best_ask()   # 1.0002
        book.mid()        # 1.00015
        book.spread_bps() # 1.0
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_snapshot(
        self,
        bids: Iterable[tuple],
        asks: Iterable[tuple],
    ) -> None:
        """Initialise the book from a full snapshot.

        Args:
            bids: Iterable of ``(price, size)`` pairs for the bid side.
            asks: Iterable of ``(price, size)`` pairs for the ask side.
        """
        self.bids = {
            float(p): float(q) for p, q in bids if float(q) > 0
        }
        self.asks = {
            float(p): float(q) for p, q in asks if float(q) > 0
        }

    def apply_update(self, side: str, price: float | str, size: float | str) -> None:
        """Apply an incremental update to one side of the book.

        A size of zero removes the price level.

        Args:
            side: ``"bid"`` or ``"ask"``.
            price: Price level to update.
            size: New quantity at this price level; ``0`` removes the level.
        """
        book = self.bids if side == "bid" else self.asks
        price = float(price)
        size = float(size)
        if size == 0:
            book.pop(price, None)
        else:
            book[price] = size

    def best_bid(self) -> float | None:
        """Return the highest bid price, or ``None`` if the bid side is empty."""
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        """Return the lowest ask price, or ``None`` if the ask side is empty."""
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        """Return the mid-price, or ``None`` if either side is empty."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> float | None:
        """Return the absolute bid-ask spread, or ``None`` if either side is empty."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def spread_bps(self) -> float | None:
        """Return the bid-ask spread in basis points relative to the mid-price."""
        s = self.spread()
        m = self.mid()
        if s is None or m is None or m == 0:
            return None
        return (s / m) * 10_000

    def is_crossed(self) -> bool:
        """Return ``True`` if the book is crossed (best_bid >= best_ask)."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return False
        return bb >= ba

    def sorted_bids(self) -> list[tuple[float, float]]:
        """Return bid levels sorted by price descending (best first)."""
        return sorted(self.bids.items(), key=lambda x: -x[0])

    def sorted_asks(self) -> list[tuple[float, float]]:
        """Return ask levels sorted by price ascending (best first)."""
        return sorted(self.asks.items(), key=lambda x: x[0])

    def depth_within_bps(self, side: str, bps: float) -> float:
        """Return total quantity available within ``bps`` basis points of the best price.

        Args:
            side: ``"bid"`` or ``"ask"``.
            bps: Depth window in basis points.

        Returns:
            Total quantity within the window, or ``0.0`` if the side is empty.
        """
        if side == "bid":
            best = self.best_bid()
            if best is None:
                return 0.0
            threshold = best * (1 - bps / 10_000)
            return sum(q for p, q in self.bids.items() if p >= threshold)
        else:
            best = self.best_ask()
            if best is None:
                return 0.0
            threshold = best * (1 + bps / 10_000)
            return sum(q for p, q in self.asks.items() if p <= threshold)

    def imbalance(self, bps: float = 1.0) -> float | None:
        """Return order-book imbalance within ``bps`` of the best price.

        Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).
        Returns ``None`` if both sides are empty.
        """
        bid_depth = self.depth_within_bps("bid", bps)
        ask_depth = self.depth_within_bps("ask", bps)
        total = bid_depth + ask_depth
        if total == 0:
            return None
        return (bid_depth - ask_depth) / total

    def __repr__(self) -> str:
        bb = self.best_bid()
        ba = self.best_ask()
        return f"OrderBook(best_bid={bb}, best_ask={ba}, bid_levels={len(self.bids)}, ask_levels={len(self.asks)})"
