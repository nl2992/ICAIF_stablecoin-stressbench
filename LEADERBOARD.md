# StressBench Leaderboard

Public leaderboard for the Stablecoin StressBench benchmark.
All results use the committed gold dataset and the evaluation protocol in `scripts/evaluate_models.py`.

**Dataset**: `data/gold/dataset.parquet` (56,134 rows × 125 columns)
**Gold dataset DOI / Hugging Face**: *(upload pending — see `data/README.md` for local reproduction)*
**Reproducibility**: `python scripts/make_paper_tables.py` reproduces all paper tables from scratch.

---

## Primary Task: basis_usdc_1m_gt10bps (economic metric: net bps, SVB test split)

| Rank | Model | Feature set | Net bps | Trades | Oracle cap. | Paper ref. |
|------|-------|-------------|---------|--------|------------|------------|
| — | **Oracle (ceiling)** | — | **+161.7** | 316 | **100.0%** | Table 8 |
| — | _No profitable entry_‡ | — | _< 0_ | — | _0/81 paths_ | §results |
| — | GRU supervised (calm) | price+book | −239.0 | 656 | — | Table 12 |
| — | ExpNetProfitRegressor | price\_only | −73.8 | 537 | — | Table 8 |
| — | LightGBM (calm) | all | 0 trades† | — | — | Table 8 |
| — | PriceBasis10bps | price\_only | −269.5 | 2002 | — | Table 8 |
| — | NoTrade | — | 0.0 | 0 | 0.0% | — |

†Calibration threshold search finds no configuration with ≥25 trades and positive expected return.
‡No model selects profitable windows on the CEX order book: a pre-registered 81-path search (training protocol × feature set × model) yields 0 paths surviving a Benjamini–Hochberg FDR(0.10) gate, and even purged in-event cross-validation is −66 to −122 bps (`results/exploration/`). The earlier "+82.5 bps meta-label" rows were the output of a synthetic data generator (`scripts/_synthetic_crossmech.py`); on the real gold panel that transfer is ≈−30 bps, so they have been removed. The genuine positive result is on-chain (see `onchain_ledger.csv`).

## Secondary Task: executive_arb_q10000_5m (oracle = +224.6 bps)

| Rank | Model | Net bps | Gap to oracle | Paper ref. |
|------|-------|---------|--------------|------------|
| — | Oracle | +224.6 | — | Table 7 |
| — | Best ML | −41.1 | 265.7 bps | Table 7 |

## Stress/Recovery Split (Tier-A empirical finding)

All 456 executable arbitrage windows are concentrated in the **stress phase**;
the recovery phase contains **zero** executable windows.

| Window | Minutes | Optical (max-abs) | Executable | Oracle bps | n profitable |
|--------|---------|------------------|------------|-----------|-------------|
| SVB Stress (Mar 10–14 2023) | 7,189 | 53.0% | 6.63% | +259.9 | 456 |
| SVB Recovery (Mar 15–20 2023) | 8,643 | 21.6% | **0.00%** | — | **0** |
| Terra/LUNA (May 2022, val.) | 11,526 | 14.0% | 2.40% | +87.6 | 265 |

---

## How to submit

1. **Fork** this repository.
2. Train your model using `split=train` and `split=validation` only. Do **not** use `split=test` during development.
3. Run inference on the test split (`split=test`) and record predictions.
4. Evaluate using `scripts/evaluate_models.py` with `--split test`.
5. Open a pull request to `LEADERBOARD.md` with one new row, including:
   - Model name and paper/code reference
   - Feature set used (from `src/stressbench/experiments/feature_sets.py`)
   - Net bps, trade count, and oracle capture % on the basis task
   - Link to reproducible code

All submissions must be reproducible from the committed `data/gold/dataset.parquet`.

---

## Evaluation protocol

```bash
# Reproduce all paper tables
python scripts/make_paper_tables.py

# Train and evaluate a model
python scripts/run_experiments.py --task basis_usdc_1m_gt10bps --models lgbm

# Run the full pipeline from raw data
python scripts/run_pipeline.py
```

See `IMPLEMENTATION_STATUS.md` for the full reproducibility manifest.

**Oracle**: the hindsight ceiling that trades every minute where `net(q,t) > 10 bps`.
No deployable model can exceed it.
