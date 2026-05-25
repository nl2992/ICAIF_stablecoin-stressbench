"""ID generation utilities: UUIDs and deterministic batch IDs."""

import uuid


def new_batch_id() -> str:
    """Generate a new random UUID string for an ingest batch."""
    return str(uuid.uuid4())


def instrument_id(venue: str, native_symbol: str) -> str:
    """Construct a canonical instrument ID from venue and native symbol.

    Example:
        instrument_id("binance", "USDCUSDT") -> "binance:USDCUSDT"
    """
    return f"{venue.lower()}:{native_symbol}"
