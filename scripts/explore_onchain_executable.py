#!/usr/bin/env python3
"""On-chain (Curve/AMM) executable-arbitrage exploration — the venue counterpart to the CEX
exploration in explore_executable_transfer.py.

On-chain, execution cost is the bounded pool slippage (+ a 1bp fee), so a visible dislocation
is almost always capturable. We test whether a simple deviation rule (capture every optical
fire, |basis|>10bps) is genuinely profitable per event, with:
  * net_bps        : benchmark formula |basis| - slippage - fee  (matches run_onchain_amm_gap.py)
  * net_bps_gas30  : stricter, additionally subtracts a fixed 30 bps swap-gas haircut for a
                     $10k notional (per-row gas absent in source; conservative stated assumption)
Significance uses a MOVING-BLOCK bootstrap (block=30 minutes) so serial correlation within an
event does not understate the CI. A path is a FINDING if net>0, block-bootstrap 95% CI_lo>0,
n>=30, and it survives Benjamini-Hochberg FDR(0.10) across the on-chain ledger.

Run:  python scripts/explore_onchain_executable.py
Out:  results/exploration/onchain_ledger.csv
"""
from __future__ import annotations
import csv, datetime as dt
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/gold/onchain_amm_panel.parquet"
LEDGER = ROOT / "results/exploration/onchain_ledger.csv"
EVENTS = ["usdt_curve_2023", "usdc_svb_2023", "terra_luna_2022", "ftx_2022", "busd_2023"]
GAS_HAIRCUT_BPS = 30.0  # fixed conservative swap-gas cost for a $10k notional (stated assumption)


def block_boot(x, block=30, B=2000, seed=0):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        samp = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[b] = samp.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean())


def bh(rows, q=0.10):
    cand = [r for r in rows if r["n"] >= 30 and r["net_bps"] > 0 and r["ci_lo"] > 0]
    if not cand:
        return []
    cand = sorted(cand, key=lambda r: r["boot_p"]); m = len(rows)
    return [r for i, r in enumerate(cand, 1) if r["boot_p"] <= (i / m) * q]


def main():
    df = pl.read_parquet(PANEL)
    batch = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    rows = []
    for metric in ["net_bps", "net_bps_gas30"]:
        # per-event naive deviation rule (capture every optical fire)
        ev_means = []
        for ev in EVENTS:
            d = df.filter((pl.col("event") == ev) & (pl.col("optical_fire") == 1)).sort("t")
            net = d["net_bps"].to_numpy() - (GAS_HAIRCUT_BPS if metric == "net_bps_gas30" else 0.0)
            if len(net) == 0:
                continue
            lo, hi, p = block_boot(net)
            r = dict(venue="onchain", strategy="fire_all_optical", event=ev, net_metric=metric,
                     n=int(len(net)), net_bps=round(float(net.mean()), 2),
                     ci_lo=round(lo, 2), ci_hi=round(hi, 2), boot_p=round(p, 4), batch=batch)
            rows.append(r); ev_means.append(float(net.mean())); print(r)
        # across-events sign summary (unit = event, robust to within-event autocorrelation)
        if ev_means:
            pos = sum(m > 0 for m in ev_means)
            print(f"  [{metric}] events net-positive: {pos}/{len(ev_means)}  median event net = {np.median(ev_means):.1f} bps")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    old = list(csv.DictReader(open(LEDGER))) if LEDGER.exists() else []
    fields = sorted(set().union(*[set(r) for r in rows + old])) if (rows or old) else []
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in old:
            w.writerow(r)
        for r in rows:
            w.writerow(r)
    # findings over this batch (numeric)
    f = bh(rows)
    print(f"\n=== on-chain paths this batch={len(rows)}.  FDR(0.10) survivors (net>0 & CI_lo>0 & n>=30): {len(f)} ===")
    for r in f:
        print("  FINDING:", r["event"], r["net_metric"], f"{r['net_bps']}bps CI[{r['ci_lo']},{r['ci_hi']}] n={r['n']}")


if __name__ == "__main__":
    main()
