-- ============================================================
-- Stablecoin StressBench: ClickHouse Feature (Gold) Tables
-- ============================================================

USE stressbench;

-- feat_book_1s: 1-second order-book microstructure features
CREATE TABLE IF NOT EXISTS feat_book_1s
(
    ts                  DateTime64(6, 'UTC'),
    venue_id            LowCardinality(String),
    instrument_id       LowCardinality(String),
    mid                 Nullable(Float64),
    best_bid            Nullable(Float64),
    best_ask            Nullable(Float64),
    spread_bps          Nullable(Float64),
    depth_bid_1bp       Float64 DEFAULT 0,
    depth_ask_1bp       Float64 DEFAULT 0,
    depth_bid_5bp       Float64 DEFAULT 0,
    depth_ask_5bp       Float64 DEFAULT 0,
    depth_bid_10bp      Float64 DEFAULT 0,
    depth_ask_10bp      Float64 DEFAULT 0,
    imbalance_1bp       Nullable(Float64),
    imbalance_5bp       Nullable(Float64),
    trade_count         UInt32 DEFAULT 0,
    trade_volume        Float64 DEFAULT 0,
    quote_update_count  UInt32 DEFAULT 0,
    data_quality_score  Float32 DEFAULT 1.0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (instrument_id, venue_id, ts);

-- feat_book_1m: 1-minute aggregated microstructure features
CREATE TABLE IF NOT EXISTS feat_book_1m
(
    ts                      DateTime64(6, 'UTC'),
    venue_id                LowCardinality(String),
    instrument_id           LowCardinality(String),
    mid_mean                Nullable(Float64),
    mid_open                Nullable(Float64),
    mid_close               Nullable(Float64),
    mid_high                Nullable(Float64),
    mid_low                 Nullable(Float64),
    spread_bps_mean         Nullable(Float64),
    spread_bps_max          Nullable(Float64),
    depth_bid_10bp_mean     Float64 DEFAULT 0,
    depth_ask_10bp_mean     Float64 DEFAULT 0,
    imbalance_1bp_mean      Nullable(Float64),
    trade_count_1m          UInt32 DEFAULT 0,
    trade_volume_1m         Float64 DEFAULT 0,
    quote_update_count_1m   UInt32 DEFAULT 0,
    data_quality_score_min  Float32 DEFAULT 1.0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (instrument_id, venue_id, ts);

-- feat_stablecoin_fx_1s: USD-normalised stablecoin prices
CREATE TABLE IF NOT EXISTS feat_stablecoin_fx_1s
(
    ts                          DateTime64(6, 'UTC'),
    stablecoin                  LowCardinality(String),
    venue_id                    LowCardinality(String),
    quote_asset                 LowCardinality(String),
    mid_usd                     Nullable(Float64),
    spread_bps                  Nullable(Float64),
    depth_10bp_usd              Nullable(Float64),
    deviation_from_1_usd_bps    Nullable(Float64),
    venue_consensus_deviation_bps Nullable(Float64)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (stablecoin, venue_id, ts);

-- feat_cross_quote_basis_1s: cross-quote implied BTC/ETH prices and basis
CREATE TABLE IF NOT EXISTS feat_cross_quote_basis_1s
(
    ts                      DateTime64(6, 'UTC'),
    base_asset              LowCardinality(String),   -- 'BTC' | 'ETH'
    venue_id                LowCardinality(String),
    quote_asset             LowCardinality(String),   -- 'USDC' | 'USDT'
    btc_usd_direct          Nullable(Float64),
    btc_usd_via_quote       Nullable(Float64),
    cross_quote_basis_bps   Nullable(Float64)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (base_asset, venue_id, quote_asset, ts);

-- feat_fragmentation_1m: cross-venue fragmentation index
CREATE TABLE IF NOT EXISTS feat_fragmentation_1m
(
    ts                              DateTime64(6, 'UTC'),
    stablecoin                      LowCardinality(String),
    num_active_venues               UInt8,
    mid_dispersion_bps              Nullable(Float64),
    depth_weighted_dispersion_bps   Nullable(Float64),
    max_minus_min_bps               Nullable(Float64),
    volume_share_hhi                Nullable(Float64),
    depth_share_hhi                 Nullable(Float64)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (stablecoin, ts);

-- feat_settlement_1m: on-chain settlement proxy features
CREATE TABLE IF NOT EXISTS feat_settlement_1m
(
    ts                      DateTime64(6, 'UTC'),
    chain                   LowCardinality(String),
    stablecoin              LowCardinality(String),
    transfer_count_1m       UInt32 DEFAULT 0,
    transfer_volume_1m      Float64 DEFAULT 0,
    large_transfer_count_1m UInt32 DEFAULT 0,
    mint_count_1h           Nullable(UInt32),
    burn_count_1h           Nullable(UInt32),
    gas_proxy               Nullable(Float64),
    block_lag_proxy         Nullable(Float64),
    dex_swap_volume_1m      Float64 DEFAULT 0,
    dex_net_flow_1m         Float64 DEFAULT 0
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (chain, stablecoin, ts);
