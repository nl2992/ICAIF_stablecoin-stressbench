-- ============================================================
-- Stablecoin StressBench: ClickHouse Fact Tables
-- ============================================================

USE stressbench;

-- fact_trade: normalised trade events
CREATE TABLE IF NOT EXISTS fact_trade
(
    ts_event        DateTime64(6, 'UTC'),
    ts_receive_ns   UInt64,
    venue_id        LowCardinality(String),
    instrument_id   LowCardinality(String),
    trade_id        String,
    side            LowCardinality(String),   -- 'buy' | 'sell' | 'unknown'
    price           Float64,
    size            Float64,
    notional_usd    Nullable(Float64),
    raw_source      LowCardinality(String),
    payload_hash    String,
    ingest_batch_id String,
    is_outlier_price UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (instrument_id, venue_id, ts_event, trade_id);

-- fact_book_l2_update: incremental order-book updates
CREATE TABLE IF NOT EXISTS fact_book_l2_update
(
    ts_event            DateTime64(6, 'UTC'),
    ts_receive_ns       UInt64,
    venue_id            LowCardinality(String),
    instrument_id       LowCardinality(String),
    side                LowCardinality(String),   -- 'bid' | 'ask'
    price               Float64,
    size                Float64,
    update_type         LowCardinality(String),   -- 'snapshot' | 'update'
    raw_source          LowCardinality(String),
    payload_hash        String,
    is_crossed_book     UInt8 DEFAULT 0,
    is_negative_size    UInt8 DEFAULT 0,
    is_sequence_gap     UInt8 DEFAULT 0,
    is_checksum_failed  UInt8 DEFAULT 0,
    is_stale_quote      UInt8 DEFAULT 0,
    is_resync_period    UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (instrument_id, venue_id, ts_event, side, price);

-- fact_book_snapshot: periodic order-book level snapshots
CREATE TABLE IF NOT EXISTS fact_book_snapshot
(
    ts_event            DateTime64(6, 'UTC'),
    ts_receive_ns       UInt64,
    venue_id            LowCardinality(String),
    instrument_id       LowCardinality(String),
    side                LowCardinality(String),
    level               UInt16,
    price               Float64,
    size                Float64,
    checksum            Nullable(String),
    raw_source          LowCardinality(String),
    payload_hash        String,
    is_crossed_book     UInt8 DEFAULT 0,
    is_negative_size    UInt8 DEFAULT 0,
    is_sequence_gap     UInt8 DEFAULT 0,
    is_checksum_failed  UInt8 DEFAULT 0,
    is_stale_quote      UInt8 DEFAULT 0,
    is_resync_period    UInt8 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (instrument_id, venue_id, ts_event, side, level);

-- fact_quote_bbo: best bid/offer snapshots
CREATE TABLE IF NOT EXISTS fact_quote_bbo
(
    ts_event        DateTime64(6, 'UTC'),
    ts_receive_ns   UInt64,
    venue_id        LowCardinality(String),
    instrument_id   LowCardinality(String),
    best_bid        Nullable(Float64),
    best_bid_size   Nullable(Float64),
    best_ask        Nullable(Float64),
    best_ask_size   Nullable(Float64),
    raw_source      LowCardinality(String),
    payload_hash    String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (instrument_id, venue_id, ts_event);

-- fact_funding: perpetual funding rates
CREATE TABLE IF NOT EXISTS fact_funding
(
    ts_event        DateTime64(6, 'UTC'),
    venue_id        LowCardinality(String),
    instrument_id   LowCardinality(String),
    funding_rate    Float64,
    next_funding_ts Nullable(DateTime64(6, 'UTC')),
    raw_source      LowCardinality(String),
    payload_hash    String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (instrument_id, venue_id, ts_event);

-- fact_fee_schedule: exchange fee schedules (point-in-time)
CREATE TABLE IF NOT EXISTS fact_fee_schedule
(
    effective_date  Date,
    venue_id        LowCardinality(String),
    taker_bps       Float64,
    maker_bps       Float64,
    withdrawal_fee_usd Nullable(Float64),
    notes           String
)
ENGINE = ReplacingMergeTree
ORDER BY (venue_id, effective_date);

-- fact_onchain_transfer: ERC-20 transfer events
CREATE TABLE IF NOT EXISTS fact_onchain_transfer
(
    ts_event        DateTime64(6, 'UTC'),
    chain           LowCardinality(String),
    token_symbol    LowCardinality(String),
    tx_hash         String,
    block_number    UInt64,
    from_address    String,
    to_address      String,
    amount          Float64,
    gas_used        Nullable(UInt64),
    gas_price       Nullable(UInt64),
    contract_address String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts_event)
ORDER BY (chain, token_symbol, ts_event, tx_hash);

-- fact_issuer_reserve_report: periodic reserve disclosures
CREATE TABLE IF NOT EXISTS fact_issuer_reserve_report
(
    report_date     Date,
    issuer          LowCardinality(String),
    stablecoin      LowCardinality(String),
    total_supply    Nullable(Float64),
    reserve_usd     Nullable(Float64),
    reserve_ratio   Nullable(Float64),
    source_url      String,
    retrieved_at    DateTime64(6, 'UTC')
)
ENGINE = ReplacingMergeTree
ORDER BY (issuer, stablecoin, report_date);

-- fact_issuer_event: discrete issuer events
CREATE TABLE IF NOT EXISTS fact_issuer_event
(
    event_time_utc   DateTime64(6, 'UTC'),
    issuer           LowCardinality(String),
    stablecoin       LowCardinality(String),
    event_type       LowCardinality(String),
    event_severity   LowCardinality(String),
    source_url       String,
    effective_date   Nullable(Date),
    description      String
)
ENGINE = MergeTree
ORDER BY (stablecoin, event_time_utc, event_type);

-- fact_venue_status: exchange status and maintenance windows
CREATE TABLE IF NOT EXISTS fact_venue_status
(
    ts_event        DateTime64(6, 'UTC'),
    venue_id        LowCardinality(String),
    status          LowCardinality(String),   -- 'operational' | 'degraded' | 'maintenance' | 'incident'
    affected_pairs  Array(String),
    source_url      String,
    notes           String
)
ENGINE = MergeTree
ORDER BY (venue_id, ts_event);
