"""Time utilities: nanosecond timestamps, UTC conversion, floor/ceil helpers."""

import time
from datetime import datetime, timezone


def now_ns() -> int:
    """Return the current wall-clock time in nanoseconds since the Unix epoch."""
    return time.time_ns()


def now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def ns_to_utc(ns: int) -> datetime:
    """Convert a nanosecond Unix timestamp to a UTC datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def utc_to_ns(dt: datetime) -> int:
    """Convert a UTC datetime to nanoseconds since the Unix epoch."""
    return int(dt.timestamp() * 1e9)


def floor_to_second(ns: int) -> int:
    """Floor a nanosecond timestamp to the nearest second boundary."""
    return (ns // 1_000_000_000) * 1_000_000_000


def floor_to_minute(ns: int) -> int:
    """Floor a nanosecond timestamp to the nearest minute boundary."""
    return (ns // 60_000_000_000) * 60_000_000_000


def date_str(ns: int) -> str:
    """Return ISO date string (YYYY-MM-DD) from a nanosecond timestamp."""
    return ns_to_utc(ns).strftime("%Y-%m-%d")


def hour_str(ns: int) -> str:
    """Return zero-padded hour string (HH) from a nanosecond timestamp."""
    return ns_to_utc(ns).strftime("%H")
