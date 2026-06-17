# StressBench executable-window exploration — results summary

**Total paths logged:** 81  ·  **well-defined paths (real net, n≥30):** 16  ·  **oracle ceiling:** +162.2 bps

**Positive findings surviving Benjamini–Hochberg FDR(0.10) with net>0, CI_lo>0, n≥30:** 0


> No path yields a positive return that survives multiple-testing correction. Selecting profitable executable windows is unsolved on this data.


## Mean net bps by training protocol (well-defined paths)

| protocol | mean net_bps | n_paths | interpretation |
|---|---|---|---|
| inevent_wf | -94.2 | 1 | train 1st-half SVB → test 2nd-half (in-event) |
| inevent_purged_cv | -75.3 | 7 | purged 5-fold CV within SVB (in-distribution upper bound) |
| crossmech | -28.1 | 8 | train Terra → test SVB (cross-mechanism transfer) |

## Best (least-negative) well-defined paths

| protocol | features | model | net_bps | 95% CI | n |
|---|---|---|---|---|---|
| crossmech | price_only | logistic | 7.71 | [-23.4,41.2] | 95 |
| inevent_purged_cv | price_plus_book | logistic | -20.89 | [-29.21,-14.96] | 54 |
| crossmech | book_dynamics | lgbm | -26.57 | [-31.01,-21.93] | 586 |
| crossmech | price_only | lgbm | -29.76 | [-32.58,-26.75] | 1083 |
| crossmech | price_only | random_forest | -30.75 | [-33.72,-27.45] | 1112 |
| crossmech | book_dynamics | random_forest | -31.14 | [-40.22,-22.6] | 45 |
| crossmech | book_dynamics | logistic | -31.8 | [-33.25,-30.34] | 987 |
| crossmech | price_plus_book | lgbm | -33.55 | [-36.48,-30.46] | 422 |

## Conclusion

Across every training protocol, feature set, model, and notional tried, no model selects profitable executable windows out-of-sample. Crucially, even **purged in-event cross-validation** (train and test within the same SVB crisis) is strongly negative, so the executable windows are **unpredictable from this microstructure**, not merely non-transferable. This makes StressBench a genuine open challenge and supports the benchmark's headline contributions (the ~12× optical→executable gap, the AUROC–P&L inversion, and venue-specificity) rather than any profitable model.


## On-chain venue: the positive counterpart (real Curve/AMM data)

On the on-chain venue a *simple deviation rule* (capture every optical fire, |basis|>10bps) is genuinely profitable — execution cost is the bounded pool slippage (+1bp fee), not order-book impact. Net bps per trade, block-bootstrap 95% CI (block=30):

| event | net_bps (no gas) | 95% CI | +30bps-gas robust | n |
|---|---|---|---|---|
| usdt_curve_2023 | +371.2 | [266.1,478.0] | +341.2 [236.1,448.0] | 697 |
| usdc_svb_2023 | +9.4 | [7.6,12.2] | -20.6 [-22.4,-17.8] | 375 |
| terra_luna_2022 | +380.5 | [284.1,510.2] | +350.5 [254.1,480.2] | 549 |
| ftx_2022 | +492.7 | [374.5,607.2] | +462.7 [344.5,577.2] | 353 |
| busd_2023 | +198.8 | [153.9,241.6] | +168.8 [123.9,211.6] | 766 |

**5/5 events profitable with no gas; 4/5 remain robustly profitable after a conservative 30bps gas haircut** (only the small USDC/SVB on-chain dislocation fails). All surviving paths clear the FDR(0.10)+CI+n≥30 gate.

**Venue contrast (the headline):** the *identical* deviation signal that loses money on every CEX protocol above (0 findings across 81 paths) is robustly profitable on-chain. The optical→executable barrier is a property of CEX order-book microstructure, not of stablecoin depegs — and on the executable venue, execution-aware capture works. This is a model-free, mechanism-level result (not an ML selection claim).


*Scope/assumptions:* the CEX gold panel covers 2 calm controls, Terra, SVB, and a 2024 control. On-chain net = |basis| − pool_slippage_10k − 1bp fee (the benchmark's own executable formula, single-leg capturable dislocation); per-row gas was absent in source so a fixed 30bps haircut is used as a conservative robustness check.
