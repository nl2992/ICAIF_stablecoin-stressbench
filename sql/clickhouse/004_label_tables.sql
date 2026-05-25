-- ============================================================
-- Stablecoin StressBench: ClickHouse Label Tables
-- ============================================================

USE stressbench;

-- label_basis_horizon: forward-looking basis forecast labels
CREATE TABLE IF NOT EXISTS label_basis_horizon
(
    ts                          DateTime64(6, 'UTC'),
    stablecoin                  LowCardinality(String),
    venue_id                    LowCardinality(String),
    quote_asset                 LowCardinality(String),
    -- Regression targets
    label_basis_1m              Nullable(Float64),
    label_basis_5m              Nullable(Float64),
    label_basis_15m             Nullable(Float64),
    label_basis_1h              Nullable(Float64),
    -- Classification targets (abs(basis) > threshold)
    label_basis_1m_gt5bps       UInt8 DEFAULT 0,
    label_basis_1m_gt10bps      UInt8 DEFAULT 0,
    label_basis_1m_gt25bps      UInt8 DEFAULT 0,
    label_basis_1m_gt50bps      UInt8 DEFAULT 0,
    label_basis_5m_gt5bps       UInt8 DEFAULT 0,
    label_basis_5m_gt10bps      UInt8 DEFAULT 0,
    label_basis_5m_gt25bps      UInt8 DEFAULT 0,
    label_basis_5m_gt50bps      UInt8 DEFAULT 0,
    label_basis_15m_gt5bps      UInt8 DEFAULT 0,
    label_basis_15m_gt10bps     UInt8 DEFAULT 0,
    label_basis_15m_gt25bps     UInt8 DEFAULT 0,
    label_basis_15m_gt50bps     UInt8 DEFAULT 0,
    label_basis_1h_gt5bps       UInt8 DEFAULT 0,
    label_basis_1h_gt10bps      UInt8 DEFAULT 0,
    label_basis_1h_gt25bps      UInt8 DEFAULT 0,
    label_basis_1h_gt50bps      UInt8 DEFAULT 0,
    -- Split assignment
    split                       LowCardinality(String)   -- 'train' | 'validation' | 'test'
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (stablecoin, venue_id, ts);

-- label_arbitrage_window: executable arbitrage opportunity labels
CREATE TABLE IF NOT EXISTS label_arbitrage_window
(
    ts                              DateTime64(6, 'UTC'),
    stablecoin                      LowCardinality(String),
    source_venue                    LowCardinality(String),
    dest_venue                      LowCardinality(String),
    -- Net profit labels by notional size and horizon
    label_arb_q10000_1m_gt0bps      UInt8 DEFAULT 0,
    label_arb_q10000_5m_gt0bps      UInt8 DEFAULT 0,
    label_arb_q50000_1m_gt0bps      UInt8 DEFAULT 0,
    label_arb_q50000_5m_gt0bps      UInt8 DEFAULT 0,
    label_arb_q100000_1m_gt0bps     UInt8 DEFAULT 0,
    label_arb_q100000_5m_gt0bps     UInt8 DEFAULT 0,
    label_arb_q500000_1m_gt0bps     UInt8 DEFAULT 0,
    label_arb_q500000_5m_gt0bps     UInt8 DEFAULT 0,
    -- At 5 bps threshold
    label_arb_q50000_1m_gt5bps      UInt8 DEFAULT 0,
    label_arb_q50000_5m_gt5bps      UInt8 DEFAULT 0,
    label_arb_q100000_1m_gt5bps     UInt8 DEFAULT 0,
    label_arb_q100000_5m_gt5bps     UInt8 DEFAULT 0,
    -- At 10 bps threshold
    label_arb_q50000_1m_gt10bps     UInt8 DEFAULT 0,
    label_arb_q50000_5m_gt10bps     UInt8 DEFAULT 0,
    -- Actual net profit values (for economic evaluation)
    net_profit_bps_q10000           Nullable(Float64),
    net_profit_bps_q50000           Nullable(Float64),
    net_profit_bps_q100000          Nullable(Float64),
    net_profit_bps_q500000          Nullable(Float64),
    split                           LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (stablecoin, source_venue, dest_venue, ts);

-- label_regime: market regime classification labels
CREATE TABLE IF NOT EXISTS label_regime
(
    ts              DateTime64(6, 'UTC'),
    stablecoin      LowCardinality(String),
    venue_id        LowCardinality(String),
    label_regime    LowCardinality(String),
    split           LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (stablecoin, venue_id, ts);

-- label_recovery: recovery half-life labels
CREATE TABLE IF NOT EXISTS label_recovery
(
    ts                                  DateTime64(6, 'UTC'),
    stablecoin                          LowCardinality(String),
    event_id                            LowCardinality(String),
    label_recovery_halflife_minutes     Nullable(Float64),
    split                               LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (stablecoin, event_id, ts);
