-- ============================================================
-- Stablecoin StressBench: ClickHouse Analytical Views
-- ============================================================

USE stressbench;

-- v_stablecoin_peg_deviation_1m: per-minute peg deviation across venues
CREATE OR REPLACE VIEW v_stablecoin_peg_deviation_1m AS
SELECT
    toStartOfMinute(ts)         AS ts_1m,
    stablecoin,
    venue_id,
    avg(deviation_from_1_usd_bps) AS avg_deviation_bps,
    max(abs(deviation_from_1_usd_bps)) AS max_abs_deviation_bps,
    count()                     AS n_samples
FROM feat_stablecoin_fx_1s
GROUP BY ts_1m, stablecoin, venue_id
ORDER BY ts_1m, stablecoin, venue_id;

-- v_cross_quote_basis_summary: summary of cross-quote basis by stablecoin
CREATE OR REPLACE VIEW v_cross_quote_basis_summary AS
SELECT
    toStartOfMinute(ts)         AS ts_1m,
    base_asset,
    quote_asset,
    venue_id,
    avg(cross_quote_basis_bps)  AS avg_basis_bps,
    stddevPop(cross_quote_basis_bps) AS std_basis_bps,
    max(abs(cross_quote_basis_bps)) AS max_abs_basis_bps,
    count()                     AS n_samples
FROM feat_cross_quote_basis_1s
GROUP BY ts_1m, base_asset, quote_asset, venue_id
ORDER BY ts_1m, base_asset, quote_asset, venue_id;

-- v_fragmentation_daily: daily fragmentation index
CREATE OR REPLACE VIEW v_fragmentation_daily AS
SELECT
    toDate(ts)                  AS date,
    stablecoin,
    avg(mid_dispersion_bps)     AS avg_dispersion_bps,
    max(mid_dispersion_bps)     AS max_dispersion_bps,
    avg(max_minus_min_bps)      AS avg_max_minus_min_bps,
    avg(num_active_venues)      AS avg_active_venues
FROM feat_fragmentation_1m
GROUP BY date, stablecoin
ORDER BY date, stablecoin;

-- v_settlement_congestion_1m: on-chain settlement congestion indicators
CREATE OR REPLACE VIEW v_settlement_congestion_1m AS
SELECT
    toStartOfMinute(ts)         AS ts_1m,
    chain,
    stablecoin,
    sum(transfer_count_1m)      AS total_transfers,
    sum(transfer_volume_1m)     AS total_volume,
    sum(large_transfer_count_1m) AS large_transfers,
    avg(gas_proxy)              AS avg_gas_gwei,
    sum(dex_swap_volume_1m)     AS dex_volume
FROM feat_settlement_1m
GROUP BY ts_1m, chain, stablecoin
ORDER BY ts_1m, chain, stablecoin;

-- v_label_class_balance: label class balance for benchmark splits
CREATE OR REPLACE VIEW v_label_class_balance AS
SELECT
    split,
    stablecoin,
    venue_id,
    countIf(label_basis_1m_gt10bps = 1) AS pos_1m_gt10bps,
    countIf(label_basis_1m_gt10bps = 0) AS neg_1m_gt10bps,
    countIf(label_basis_5m_gt10bps = 1) AS pos_5m_gt10bps,
    countIf(label_basis_5m_gt10bps = 0) AS neg_5m_gt10bps,
    count()                             AS total
FROM label_basis_horizon
GROUP BY split, stablecoin, venue_id
ORDER BY split, stablecoin, venue_id;

-- v_data_quality_summary: data quality monitoring per instrument
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT
    toDate(ts)                  AS date,
    venue_id,
    instrument_id,
    avg(data_quality_score)     AS avg_dq_score,
    min(data_quality_score)     AS min_dq_score,
    countIf(data_quality_score < 0.5) AS low_quality_count,
    count()                     AS total_rows
FROM feat_book_1s
GROUP BY date, venue_id, instrument_id
ORDER BY date, venue_id, instrument_id;
