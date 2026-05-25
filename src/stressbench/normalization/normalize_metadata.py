"""Normalize venue and instrument metadata into canonical dimension tables."""

from __future__ import annotations

from typing import Any

import polars as pl

from stressbench.common.config import load_instruments, load_venues
from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def build_dim_venue() -> pl.DataFrame:
    """Build the ``dim_venue`` dimension table from the venues config.

    Returns:
        :class:`polars.DataFrame` with venue dimension records.
    """
    venues = load_venues()
    records = []
    for venue_id, cfg in venues.items():
        records.append(
            {
                "venue_id": venue_id,
                "venue_type": cfg.get("type", "unknown"),
                "country_or_entity": cfg.get("country_or_entity", ""),
                "is_cex": int(cfg.get("type") == "cex"),
                "is_dex": int(cfg.get("type") == "dex"),
                "source_url": cfg.get("websocket_url", cfg.get("subgraph_url", "")),
            }
        )
    return pl.DataFrame(records)


def build_dim_instrument() -> pl.DataFrame:
    """Build the ``dim_instrument`` dimension table from the instruments config.

    Returns:
        :class:`polars.DataFrame` with instrument dimension records.
    """
    instruments = load_instruments()
    records = []
    for inst in instruments:
        records.append(
            {
                "instrument_id": inst.get("instrument_id", ""),
                "venue_id": inst.get("venue_id", ""),
                "native_symbol": inst.get("native_symbol", ""),
                "base_asset": inst.get("base_asset", ""),
                "quote_asset": inst.get("quote_asset", ""),
                "instrument_type": inst.get("instrument_type", "spot"),
                "tick_size": inst.get("tick_size", None),
                "lot_size": inst.get("lot_size", None),
            }
        )
    return pl.DataFrame(records)


def build_dim_stablecoin() -> pl.DataFrame:
    """Build the ``dim_stablecoin`` dimension table.

    Returns:
        :class:`polars.DataFrame` with stablecoin dimension records.
    """
    records = [
        {
            "asset": "USDC",
            "issuer": "Circle",
            "backing_type": "fiat_collateralized",
            "primary_chains": "ethereum,polygon,solana,arbitrum",
            "transparency_source": "https://www.circle.com/en/transparency",
        },
        {
            "asset": "USDT",
            "issuer": "Tether",
            "backing_type": "fiat_collateralized",
            "primary_chains": "ethereum,tron,solana",
            "transparency_source": "https://tether.to/en/transparency/",
        },
        {
            "asset": "DAI",
            "issuer": "MakerDAO/Sky",
            "backing_type": "crypto_collateralized",
            "primary_chains": "ethereum",
            "transparency_source": "https://makerburn.com/",
        },
        {
            "asset": "PYUSD",
            "issuer": "PayPal/Paxos",
            "backing_type": "fiat_collateralized",
            "primary_chains": "ethereum,solana",
            "transparency_source": "https://paxos.com/pyusd/",
        },
        {
            "asset": "FDUSD",
            "issuer": "First Digital",
            "backing_type": "fiat_collateralized",
            "primary_chains": "ethereum,bnbchain",
            "transparency_source": "https://firstdigitallabs.com/",
        },
    ]
    return pl.DataFrame(records)
