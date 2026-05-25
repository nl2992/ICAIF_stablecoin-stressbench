"""Pydantic schemas for raw Bronze messages and Silver normalized records."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RawMessage(BaseModel):
    """Canonical schema for every raw Bronze message written to Parquet.

    All raw messages must conform to this schema regardless of venue or channel.
    The ``payload`` field stores the original exchange-native JSON as a dict.
    """

    source: str = Field(..., description="Venue identifier, e.g. 'binance'")
    channel: str = Field(
        ...,
        description="Data channel: trade | depth | level2 | book | issuer | onchain",
    )
    symbol: str = Field(..., description="Native symbol as received from the venue")
    ts_exchange: Optional[str] = Field(
        None, description="Exchange-reported event timestamp (ISO-8601 or epoch ms/us)"
    )
    ts_receive_ns: int = Field(
        ..., description="Local receive timestamp in nanoseconds since Unix epoch"
    )
    payload: dict[str, Any] = Field(
        ..., description="Original exchange-native JSON payload"
    )
    payload_hash: str = Field(
        ..., description="SHA-256 hex digest of the canonical JSON payload"
    )
    schema_version: str = Field(default="raw.v1", description="Schema version tag")
    ingest_batch_id: str = Field(
        ..., description="UUID identifying the ingest batch that produced this record"
    )


class NormalizedTrade(BaseModel):
    """Silver-layer normalized trade record."""

    ts_event_ns: int
    ts_receive_ns: int
    venue_id: str
    instrument_id: str
    trade_id: str
    side: str  # "buy" | "sell" | "unknown"
    price: float
    size: float
    notional_usd: Optional[float] = None
    raw_source: str
    payload_hash: str
    ingest_batch_id: str
    is_outlier_price: bool = False


class NormalizedBookLevel(BaseModel):
    """Silver-layer normalized order-book level record."""

    ts_event_ns: int
    ts_receive_ns: int
    venue_id: str
    instrument_id: str
    side: str  # "bid" | "ask"
    level: int
    price: float
    size: float
    checksum: Optional[str] = None
    raw_source: str
    payload_hash: str
    is_crossed_book: bool = False
    is_negative_size: bool = False
    is_sequence_gap: bool = False
    is_checksum_failed: bool = False
    is_stale_quote: bool = False
    is_resync_period: bool = False
