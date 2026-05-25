# Stablecoin StressBench Implementation Status

This repository is the base code for the ICAIF Stablecoin StressBench project.

## Positioning

Stablecoin StressBench is a benchmark for settlement-risk dislocations in stablecoin markets. It sits at the intersection of financial benchmark construction, blockchain and cryptocurrency, market microstructure, trading and smart order routing, risk management, financial time-series analysis, and graph/network modeling.

## Implemented

- Core Python package under `src/stressbench`.
- Order-book reconstruction, spread, depth, imbalance, and crossed-book checks.
- Executable VWAP and transaction-cost-aware net profit calculations.
- Kraken-style checksum helpers.
- Bronze raw writer and manifest utilities.
- Live WebSocket collectors for Binance, Coinbase, and Kraken.
- Function-level normalization for trade and order-book messages.
- Feature computation modules for microstructure, basis, fragmentation, settlement proxies, issuer events, and graph snapshots.
- Label modules for basis forecasting, arbitrage windows, profitability, regimes, and recovery.
- Baseline, tree, sequence, and placeholder graph model wrappers.
- Evaluation metrics, backtest wrapper, and leaderboard builder.
- ClickHouse DDL files and YAML configs.
- Unit tests for the lower-level utilities and formulas.

## Not Yet Implemented

- End-to-end Bronze-to-Silver-to-Gold data pipeline.
- Historical Coinbase/Kraken archive pulls outside Tardis-style loaders.
- Production-ready feature table materialization.
- Model training/evaluation on real Gold datasets.
- Public benchmark artifact generation and leaderboard publishing.
- Full graph model training loop.
- Live capture entrypoint scripts referenced by older docs and Makefile targets.

## Known Gaps

- Some operational scripts expect class-style loader APIs that are not yet present.
- `scripts/build_features.py` currently contains orchestration stubs for actual file IO.
- The codebase has strong research primitives, but the reproducible benchmark pipeline still needs wiring and validation.
