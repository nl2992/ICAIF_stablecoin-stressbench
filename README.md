# Stablecoin StressBench

**Stablecoin StressBench** is a transaction-cost-aware benchmark for detecting, forecasting, and economically ranking stablecoin dislocations across venues, quote currencies, and settlement rails.

This repository implements a full-featured, reproducible benchmark and research environment designed for evaluating machine learning and econometric models on stablecoin stress-testing and dislocation scenarios.

## Research Objective

Can AI models identify stablecoin dislocations that are still economically meaningful after spreads, depth, fees, transfer frictions, and settlement risk are included?

Stablecoin StressBench proves three things:
1. **Cross-quote and cross-venue dislocations** are measurable and severe during stablecoin stress events.
2. **Naive price-only signals overstate arbitrage** because they ignore executable depth, fees, latency, and transfer constraints.
3. **Joint feature models** (combining order-book state, cross-venue basis, stablecoin FX deviation, venue status, and on-chain settlement features) outperform price/candle-only baselines.

## Repository Structure

```text
stablecoin-stressbench/
  README.md
  pyproject.toml
  .env.example
  Makefile
  docker-compose.yml

  configs/               # YAML configurations for venues, instruments, event windows, fees
  src/stressbench/       # Source code for ingestion, normalization, book reconstruction, features, labels, models, evaluation
  sql/clickhouse/        # ClickHouse DDL schemas for dim, fact, feature, and label tables
  scripts/               # Operational scripts for data capture, pipeline building, training, and evaluation
  notebooks/             # Jupyter notebooks for analysis and visualization
  tests/                 # Pytest suite for core components
```

## Quick Start

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run ClickHouse
```bash
docker-compose up -d
```

### 3. Run Live Capture (Example)
```bash
python scripts/start_live_capture.py
```

## License
MIT License
