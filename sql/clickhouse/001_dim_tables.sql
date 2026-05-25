-- ============================================================
-- Stablecoin StressBench: ClickHouse Dimension Tables
-- ============================================================

CREATE DATABASE IF NOT EXISTS stressbench;

USE stressbench;

-- dim_venue: one row per exchange or DEX
CREATE TABLE IF NOT EXISTS dim_venue
(
    venue_id          LowCardinality(String),
    venue_type        LowCardinality(String),   -- 'cex' | 'dex'
    country_or_entity String,
    is_cex            UInt8,
    is_dex            UInt8,
    source_url        String
)
ENGINE = ReplacingMergeTree
ORDER BY venue_id;

-- dim_instrument: one row per tradable instrument per venue
CREATE TABLE IF NOT EXISTS dim_instrument
(
    instrument_id   String,
    venue_id        LowCardinality(String),
    native_symbol   String,
    base_asset      LowCardinality(String),
    quote_asset     LowCardinality(String),
    instrument_type LowCardinality(String),   -- 'spot' | 'perp' | 'dex_pool'
    tick_size       Nullable(Float64),
    lot_size        Nullable(Float64),
    first_seen      DateTime64(6, 'UTC'),
    last_seen       Nullable(DateTime64(6, 'UTC'))
)
ENGINE = ReplacingMergeTree
ORDER BY instrument_id;

-- dim_stablecoin: one row per stablecoin
CREATE TABLE IF NOT EXISTS dim_stablecoin
(
    asset               LowCardinality(String),
    issuer              String,
    backing_type        LowCardinality(String),   -- 'fiat_collateralized' | 'crypto_collateralized' | 'algorithmic'
    primary_chains      Array(String),
    transparency_source String
)
ENGINE = ReplacingMergeTree
ORDER BY asset;
