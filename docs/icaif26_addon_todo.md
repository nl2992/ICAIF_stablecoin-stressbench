# ICAIF 2026 Add-On Checklist

Baseline benchmark is frozen at `v0.1.0-benchmark-freeze`. All add-on work
writes to `results/experiments_addon/` and `results/paper_addon/` only.

## Status

| Item | Status | Output |
|---|---|---|
| Methodology addendum | Done | `docs/icaif26_methodology_addendum.md` |
| Paper outline (8-page ACM) | Done | `docs/icaif26_paper_outline.md` |
| Economic benchmark card | Done | `docs/economic_benchmark_card.md` |
| Addon event windows config | Done | `configs/event_windows_addon.yaml` |
| ExpectedNetProfitRegressor model | Done | `src/stressbench/models/cost_sensitive.py` |
| Addon experiment runner | Done | `scripts/run_addon_experiments.py` |
| Addon experiment results | Done | `results/experiments_addon/expected_net_profit_results.csv` |
| Robustness grid (cost-sensitive) | Done (bug fixed) | `results/experiments_addon/robustness_price_execution_gap.csv` |
| Robustness grid script | Done | `scripts/run_robustness_grid.py` |
| False-positive diagnosis | Done | `results/paper_addon/table_5_false_positive_diagnosis.csv` |
| Addon tables (8, 9, 10) | Done | `results/paper_addon/table_{8,9,10}_*.csv` |
| Table 8b cost robustness | Done | `results/paper_addon/table_8b_cost_robustness_summary.csv` |
| Addon figures (8, 9, 11, 12) | Done | `results/paper_addon/figures/figure_{8,9,11,12}_*.png` |
| Extended figures (14–22, Columbia theme) | Done | `results/paper_addon/figures/figure_{14..22}_*.png` |
| Model failure summary table | Done (NaN row fixed) | `results/paper_addon/table_model_failure_summary.csv` |
| Uncertainty abstention module | Done (code + tests only) | `src/stressbench/experiments/uncertainty.py` |
| No-overwrite guard test | Done | `tests/test_addon_outputs_do_not_overwrite.py` |
| Cost-sensitivity tests | Done | `tests/test_robustness_cost_sensitivity.py` |
| Claim-consistency audit fixes | Done | README, all docs — 35.1% now attributed to primary/max basis; 12.65% USDC-specific cited |
| Data card update | Done | `docs/data_card.md` — rows 47,487; cols 125; correct split counts; depth_source vocabulary |
| matplotlib in pyproject.toml | Done | `pyproject.toml` |
| GitHub Actions CI | Done | `.github/workflows/ci.yml` |
| Reproducibility manifest | Done | `docs/reproducibility_manifest.md` |

## Remaining / Optional

| Item | Priority | Notes |
|---|---|---|
| Run uncertainty abstention experiment | Nice | Bootstrap ensemble expensive; move to future work in paper §7 |
| Full threshold-rule ablation (fixed 0.5/0.7, F1, mean bps) | Nice | Currently only total_pnl rule is run; §7d addendum marks as planned |
| Additional stress event (USDC recovery / USDT Curve) | Optional | Needs raw data pull for Mar 15–Apr 1 2023 or Jun 2023 |
| Graph/network fragmentation analysis | Optional | Only if venue centrality improves FP diagnosis |
| Block-bootstrap confidence intervals | Optional | Strengthens null result statistically |
| Seed robustness sweep | Optional | Confirm ML results are not seed-sensitive |
| LaTeX paper draft (`paper/main.tex`) | High | Next major deliverable |

## Key empirical numbers (for paper draft)

**Price-to-execution gap (test split, $10K, >10 bps, 5m horizon)**:
- Base fee: 35.09% price signal → 5.64% executable = **2.2× gap at 5m** (12× at 1m same-minute)
- High fee: 5.46% executable = ratio increases to 2.32×
- +10 bps settlement: 4.88% executable = ratio 2.59×

**Oracle gap (test split)**:
- Oracle: +161.7 bps (basis task), +224.6 bps (executable arb task)
- Best ML classifier: −49 bps
- ExpectedNetProfitRegressor (lgbm, price_only): −61 bps
- Conclusion: direct net-profit prediction still fails; gap is structural, not a threshold artifact

**False-positive diagnosis (price_threshold_10bps rule, test split)**:
- TP: 1,421 windows (true executable dislocations)
- FP: 581 windows (large basis, no executable depth): mean net profit = −38.7 bps
- FN: 583 windows (missed profitable windows)
- TN: 10,672 windows (correctly abstained)

## Do not add

- Large GNN / deep learning sequence models
- Reinforcement learning trading agent
- Live trading bot framing
- Any model whose main claim is profitability on the test split

The paper's contribution is the benchmark and null result. Keep it that way.
