"""Normalize issuer event records into the canonical fact_issuer_event schema."""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_VALID_EVENT_TYPES = {
    "reserve_report",
    "attestation",
    "mint_burn_disclosure",
    "banking_counterparty_news",
    "redemption_update",
    "chain_support_update",
}

_VALID_SEVERITIES = {"low", "medium", "high"}


def normalize_issuer_events(events: list[dict]) -> pl.DataFrame:
    """Validate and normalize a list of issuer event dicts.

    Args:
        events: List of issuer event dicts (as produced by
            :func:`~stressbench.ingestion.issuer_scraper.build_issuer_event`).

    Returns:
        Normalized :class:`polars.DataFrame` conforming to ``fact_issuer_event``.
    """
    if not events:
        return pl.DataFrame()

    validated = []
    for event in events:
        event_type = event.get("event_type", "")
        severity = event.get("event_severity", "medium")

        if event_type not in _VALID_EVENT_TYPES:
            logger.warning("Unknown event_type '%s'; skipping.", event_type)
            continue
        if severity not in _VALID_SEVERITIES:
            logger.warning(
                "Unknown event_severity '%s'; defaulting to 'medium'.", severity
            )
            event = dict(event)
            event["event_severity"] = "medium"

        validated.append(event)

    if not validated:
        return pl.DataFrame()

    return pl.DataFrame(validated)
