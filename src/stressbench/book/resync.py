"""Order-book resync utilities.

When a checksum failure or sequence gap is detected, the book must be
re-initialised from the next available snapshot. This module provides
helpers for managing resync state and flagging affected time windows.
"""

from __future__ import annotations

from stressbench.book.order_book import OrderBook
from stressbench.common.logging import get_logger

logger = get_logger(__name__)


class ResyncManager:
    """Tracks resync state for a single order book.

    When a checksum failure or sequence gap is detected, the manager marks
    the book as needing a resync. All records produced during the resync
    window should be flagged with ``is_resync_period=True``.
    """

    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id
        self.book = OrderBook()
        self._needs_resync = True
        self._resync_start_ns: int | None = None

    @property
    def needs_resync(self) -> bool:
        return self._needs_resync

    def apply_snapshot(self, bids, asks, ts_ns: int) -> None:
        """Apply a snapshot and clear the resync flag.

        Args:
            bids: Bid levels as ``(price, size)`` pairs.
            asks: Ask levels as ``(price, size)`` pairs.
            ts_ns: Nanosecond timestamp of the snapshot.
        """
        self.book.apply_snapshot(bids, asks)
        if self._needs_resync:
            logger.info(
                "Book resynced for %s at ts_ns=%d", self.instrument_id, ts_ns
            )
        self._needs_resync = False
        self._resync_start_ns = None

    def apply_update(self, side: str, price, size, ts_ns: int) -> bool:
        """Apply an incremental update.

        Args:
            side: ``"bid"`` or ``"ask"``.
            price: Price level.
            size: New size (0 removes the level).
            ts_ns: Nanosecond timestamp of the update.

        Returns:
            ``True`` if the update was applied; ``False`` if a resync is needed.
        """
        if self._needs_resync:
            logger.debug(
                "Skipping update for %s (resync pending)", self.instrument_id
            )
            return False
        self.book.apply_update(side, price, size)
        return True

    def flag_resync(self, reason: str, ts_ns: int) -> None:
        """Mark the book as requiring a resync.

        Args:
            reason: Human-readable reason for the resync.
            ts_ns: Nanosecond timestamp when the issue was detected.
        """
        logger.warning(
            "Resync required for %s at ts_ns=%d: %s",
            self.instrument_id, ts_ns, reason,
        )
        self._needs_resync = True
        self._resync_start_ns = ts_ns
