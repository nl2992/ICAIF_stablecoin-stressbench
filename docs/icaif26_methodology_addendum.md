# ICAIF 2026 Methodology Addendum

## 1. Problem

Stablecoin dislocations are often measured using quoted price deviations, but quoted price gaps are not necessarily executable. A dislocation is economically meaningful only if it survives order-book depth, VWAP execution, taker fees, market impact, and settlement frictions.

During the March 2023 USDC/SVB de-peg — the primary test event in this benchmark — 35.1% of 1-minute windows showed a primary/max cross-quote basis exceeding 10 bps on price alone (12.65% for the USDC-specific basis). After a full VWAP order-book walk at $10K notional including taker fees and market impact, only 2.88% remained profitable. This **12× price-to-execution gap** defines the core measurement challenge.

## 2. Research Question

**Can AI and econometric models identify stablecoin dislocations that remain profitable after realistic execution costs?**

The benchmark answers this with a structured null result: a hindsight oracle earns 161–225 net bps per trade on the test split, confirming profitable windows exist, but every ML and rule-based model tested produces **negative net bps** out of sample.

## 3. Benchmark Contribution

Stablecoin StressBench introduces an execution-aware benchmark that separates four distinct layers:

| Layer | Measure | Empirical result (test split) |
|---|---|---|
| 1. Price-only dislocation | `|cross_quote_basis_maxabs_bps| > 10 bps` (primary/max basis; USDC-specific: 12.65%) | 35.1% of minutes |
| 2. Gross arbitrage | Raw buy/sell spread | Positive in ~35% of minutes |
| 3. Net executable arbitrage | VWAP walk + fees + impact | 2.88% of minutes at $10K |
| 4. Predictable executable arbitrage | Ex-ante model identifies net-profitable minutes | 0% (all tested models negative) |

This decomposition is the central methodological contribution. Layers 1–3 are measured from data; layer 4 is the open benchmark challenge.

## 4. Execution-Aware Label

For each 1-minute window at notional size $q$, we reconstruct executable prices using real L2 order-book depth (not synthetic kline-inferred depth). The net-profit label is:

```
net_profit_bps(q) =
    gross_spread_bps(q)           [VWAP buy vs sell cross venues]
  − buy_taker_fee_bps             [maker/taker fee schedule]
  − sell_taker_fee_bps
  − fixed_transfer_cost_bps       [settlement / withdrawal cost proxy]
  − settlement_delay_penalty_bps  [opportunity cost of transfer latency]
```

A window is labelled **executable** (`label_arb_q{N}_{horizon}_gt0bps = 1`) iff `future net_profit_bps(q) > 0` within the prediction horizon.

**Depth provenance guarantee**: only `depth_source ∈ {real_l2_snapshot, real_l2_incremental}` contributes to net-profit labels. Synthetic kline depth (`depth_source = synthetic_kline`) is isolated to a proxy file and excluded from paper-grade calculations. Provenance is auditable per row via `depth_sources_used`, `is_paper_grade_depth`, and `depth_source` columns.

## 5. Model Evaluation

We evaluate models using both statistical and economic metrics:

### Statistical metrics
- AUROC: classification skill across thresholds
- AUPRC: precision–recall skill on imbalanced positive class
- Brier score: calibration quality
- Balanced accuracy at chosen threshold

### Economic metrics (primary success criteria)
- **net_bps_captured**: mean net profit bps on trades taken
- **hit_rate_above_cost**: fraction of trades with positive net profit
- **false_positive_cost**: mean net profit bps on trades where model predicted positive but outcome was negative
- **n_trades**: trade count (test split)
- **final_pnl_usd**: cumulative P&L at benchmark notional
- **oracle_capture_pct**: `net_bps_captured / oracle_net_bps` — the headline gap metric

The no-trade baseline (`net_bps_captured = 0`) is the economic anchor; models that lose money are worse than abstaining entirely.

### Threshold calibration

The decision threshold is calibrated on the **validation split** by maximizing total net P&L subject to ≥ 25 trades:

```
threshold* = argmax_{t ∈ [0.05, 0.95]}  Σ_{i: proba_i > t} net_profit_bps_i
             subject to |{i : proba_i > t}| ≥ 25
```

This is economically grounded: a strategy must have sufficient trade count to distinguish signal from sampling noise.

## 6. Core Finding

The benchmark establishes three empirical facts:

1. **Price dislocations are frequent.** 35.1% of SVB test-split minutes exceed 10 bps primary/max cross-quote basis (12.65% for the USDC-specific basis alone).
2. **Executable opportunities are rare.** Only 2.88% survive a $10K VWAP execution filter (12× price-to-execution ratio).
3. **The oracle gap is large.** The hindsight oracle earns +161 bps; the best ML model loses −49 bps; the gap is 210 bps.

The conclusion is not that stablecoin arbitrage is impossible — the oracle proves otherwise — but that **standard classification and regression models do not yet solve the execution-identification problem**.

## 7. Add-On Contributions

The following extensions supplement the core benchmark result:

### 7a. Robustness over costs and notionals
The price-to-execution gap is recomputed across notional sizes ($10K–$500K), fee regimes (±50% on base fees), settlement penalties (0–10 bps), and prediction horizons (1m, 5m, 15m). This tests whether the core finding depends on specific parameter choices.

### 7b. Expected net-profit regressor
Rather than classifying whether a threshold will be exceeded, the `ExpectedNetProfitRegressor` directly predicts `future_net_profit_bps_q10000`. It trades when predicted net profit exceeds a validation-calibrated floor. This targets the economic objective directly.

### 7c. Uncertainty-aware abstention (future work)
The uncertainty module (`src/stressbench/experiments/uncertainty.py`) implements bootstrap ensemble and quantile regression models that abstain when prediction uncertainty is high. Experiments comparing these abstention strategies against the no-trade baseline are reserved for future work due to the computational cost of bootstrap ensembles.

### 7d. Threshold calibration sensitivity
The primary threshold rule maximizes validation total P&L subject to ≥ 25 trades. Sensitivity to this choice is indicated by the robustness grid: fee and settlement parameter changes of realistic magnitudes do not change the qualitative null result (all non-oracle models remain economically negative). A full multi-rule threshold ablation (fixed 0.5/0.7, validation F1, validation mean bps) is planned as a follow-up.

### 7e. False-positive diagnosis
Feature profiles of true positives vs false positives are compared to explain why models trade bad windows (large basis but insufficient depth, high spread, or unfavorable fee conditions).

## 8. Relation to ICAIF 2026

This work contributes to the following ICAIF topic areas:

- **Financial benchmark construction**: systematic train/validation/test split with event-based design and benchmark-freeze protocol
- **Blockchain and cryptocurrency**: stablecoin de-peg mechanics, cross-venue arbitrage, on-chain settlement frictions
- **Market microstructure**: real L2 order-book depth, VWAP execution, spread/depth deterioration during stress
- **Trading and execution**: execution-aware label construction, transaction cost modeling, oracle gap evaluation
- **Validation and calibration of financial AI models**: threshold calibration on economic objectives, out-of-sample robustness, no-lookahead guarantees
- **Uncertainty quantification**: abstention under model uncertainty, confidence-weighted trading signals
