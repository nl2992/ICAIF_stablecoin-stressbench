"""Issuer event flag features.

Produces binary flags and time-distance features indicating proximity to
known issuer events (reserve reports, attestations, banking news, etc.).
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_EVENT_WINDOW_HOURS = 6  # ±6 hours around an issuer event


def compute_issuer_event_flags(
    ts_ns: int,
    issuer_events_df: pl.DataFrame,
    stablecoin: str,
) -> dict:
    """Compute issuer event proximity flags for a given timestamp.

    Args:
        ts_ns: Query timestamp in nanoseconds.
        issuer_events_df: Normalized issuer events DataFrame.
        stablecoin: Stablecoin symbol to filter events.

    Returns:
        Dict with issuer event flags and time-distance features.
    """
    window_ns = _EVENT_WINDOW_HOURS * 3_600_000_000_000

    if issuer_events_df.is_empty():
        return {
            "is_issuer_event_window": False,
            "nearest_event_hours": None,
            "nearest_event_type": None,
            "nearest_event_severity": None,
        }

    # Filter to the relevant stablecoin
    events = issuer_events_df.filter(pl.col("stablecoin") == stablecoin)
    if events.is_empty():
        return {
            "is_issuer_event_window": False,
            "nearest_event_hours": None,
            "nearest_event_type": None,
            "nearest_event_severity": None,
        }

    # Parse event times and compute distances
    from datetime import datetime, timezone

    min_distance_ns = float("inf")
    nearest_type = None
    nearest_severity = None

    for row in events.iter_rows(named=True):
        try:
            event_dt = datetime.fromisoformat(
                row["event_time_utc"].replace("Z", "+00:00")
            )
            event_ns = int(event_dt.timestamp() * 1e9)
            distance = abs(ts_ns - event_ns)
            if distance < min_distance_ns:
                min_distance_ns = distance
                nearest_type = row.get("event_type")
                nearest_severity = row.get("event_severity")
        except (ValueError, KeyError):
            continue

    is_in_window = min_distance_ns <= window_ns
    nearest_hours = min_distance_ns / 3_600_000_000_000 if min_distance_ns < float("inf") else None

    return {
        "is_issuer_event_window": is_in_window,
        "nearest_event_hours": nearest_hours,
        "nearest_event_type": nearest_type if is_in_window else None,
        "nearest_event_severity": nearest_severity if is_in_window else None,
    }
