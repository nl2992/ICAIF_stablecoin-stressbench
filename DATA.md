# Data Plan

Stablecoin StressBench does not commit raw market or on-chain data to Git.
The repository tracks code, schemas, configs, notebooks, tests, and a small
data directory skeleton. Actual datasets should live locally under `data/` or
in an external artifact store.

## Local Layout

```text
data/
  bronze/   # immutable raw vendor messages and downloaded archive files
  silver/   # normalized canonical trade, book, metadata, and on-chain tables
  gold/     # benchmark features, labels, splits, model-ready matrices
```

## Intended Sources

- Binance public archives from `https://data.binance.vision`
- Live Binance, Coinbase, and Kraken WebSocket captures
- Tardis historical crypto market data, if credentials are available
- Etherscan token transfer data for stablecoin settlement proxies
- Issuer event timelines and venue metadata from the YAML configs

## Git Policy

- Keep raw and generated data out of Git.
- Commit only tiny examples or fixtures when needed for tests.
- Store API keys in `.env`, never in committed files.
- Prefer reproducible pull/build scripts over manually shared files.

## Current Status

The repository currently contains the data layer scaffolding and core
normalization/feature/label code. The full Bronze-to-Silver-to-Gold pipeline is
still being wired, so real benchmark datasets are not yet produced end to end.
