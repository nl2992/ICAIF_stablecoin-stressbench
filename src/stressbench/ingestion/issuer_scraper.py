"""Issuer transparency page scraper and event loader.

Collects reserve reports, attestations, and mint/burn disclosures from
stablecoin issuers (Circle, Paxos, MakerDAO/Sky).

Note: Issuer page structures change over time. This module provides a
framework; specific parsers must be validated against live pages before use.

References:
    https://www.circle.com/en/transparency
    https://paxos.com/usdp/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressbench.common.config import bronze_root
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_ISSUER_SOURCES: dict[str, dict[str, str]] = {
    "USDC": {
        "issuer": "Circle",
        "transparency_url": "https://www.circle.com/en/transparency",
        "reserve_api": "https://api.circle.com/v1/stablecoins",
    },
    "USDP": {
        "issuer": "Paxos",
        "transparency_url": "https://paxos.com/usdp/",
    },
    "USDT": {
        "issuer": "Tether",
        "transparency_url": "https://tether.to/en/transparency/",
        "notes": "Machine-readable structure unverified; treat as manual source.",
    },
    "DAI": {
        "issuer": "MakerDAO/Sky",
        "transparency_url": "https://makerburn.com/",
    },
}


def fetch_circle_reserve_api() -> list[dict[str, Any]]:
    """Fetch USDC reserve data from the Circle public API.

    Returns:
        List of stablecoin reserve dicts, or an empty list on failure.
    """
    url = _ISSUER_SOURCES["USDC"]["reserve_api"]
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except requests.RequestException as exc:
        logger.warning("Circle API request failed: %s", exc)
        return []


def build_issuer_event(
    issuer: str,
    stablecoin: str,
    event_type: str,
    description: str,
    source_url: str,
    event_severity: str = "medium",
    effective_date: str | None = None,
) -> dict[str, Any]:
    """Construct a canonical issuer event record.

    Args:
        issuer: Issuer name (e.g. ``"Circle"``).
        stablecoin: Stablecoin symbol (e.g. ``"USDC"``).
        event_type: One of ``reserve_report``, ``attestation``,
            ``mint_burn_disclosure``, ``banking_counterparty_news``,
            ``redemption_update``, ``chain_support_update``.
        description: Human-readable description of the event.
        source_url: URL of the source document or page.
        event_severity: ``"low"``, ``"medium"``, or ``"high"``.
        effective_date: ISO date string of the effective date (optional).

    Returns:
        A dict conforming to the ``fact_issuer_event`` schema.
    """
    return {
        "event_time_utc": datetime.now(tz=timezone.utc).isoformat(),
        "issuer": issuer,
        "stablecoin": stablecoin,
        "event_type": event_type,
        "event_severity": event_severity,
        "source_url": source_url,
        "effective_date": effective_date or "",
        "description": description,
    }


# Manually curated issuer events for the USDC depeg episode (March 2023)
CURATED_ISSUER_EVENTS: list[dict[str, Any]] = [
    build_issuer_event(
        issuer="Circle",
        stablecoin="USDC",
        event_type="banking_counterparty_news",
        description="Silicon Valley Bank (SVB) placed into FDIC receivership. "
        "Circle disclosed $3.3B of USDC reserves held at SVB.",
        source_url="https://www.circle.com/blog/an-update-on-usdc-and-silicon-valley-bank",
        event_severity="high",
        effective_date="2023-03-10",
    ),
    build_issuer_event(
        issuer="Circle",
        stablecoin="USDC",
        event_type="redemption_update",
        description="Circle announced USDC redemptions would resume at 1:1 "
        "following FDIC guarantee of SVB deposits.",
        source_url="https://www.circle.com/blog/usdc-liquidity-operations-update",
        event_severity="high",
        effective_date="2023-03-13",
    ),
]


def save_issuer_events_to_bronze(
    events: list[dict[str, Any]],
    stablecoin: str,
    root: Path | None = None,
) -> Path | None:
    """Save issuer events to Bronze as Parquet.

    Args:
        events: List of issuer event dicts.
        stablecoin: Stablecoin symbol for partitioning.
        root: Bronze root override.

    Returns:
        Path to the written Parquet file, or ``None`` if empty.
    """
    if not events:
        return None

    root = root or bronze_root()
    out_dir = root / "venue=issuer" / "channel=issuer_event" / f"symbol={stablecoin}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"issuer_events-{stablecoin}.parquet"

    df = pl.DataFrame(events)
    df.write_parquet(out_file)
    logger.info("Saved %d issuer events to %s", len(events), out_file)
    return out_file
