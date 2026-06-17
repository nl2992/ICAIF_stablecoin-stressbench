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


*Scope:* this gold panel covers only 2 calm controls, Terra, SVB, and a 2024 control; no other stress events and no row-level on-chain AMM panel are present, so the on-chain venue (where executable≈100%) is not modelled here.
