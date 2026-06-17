#!/usr/bin/env python3
"""Pull the real on-chain Curve/AMM pool panels into a compact, self-contained table.

Source: per-event on-chain contagion-feature parquets (real Curve pool state from the
on-chain pipeline, Etherscan/The Graph). Default source dir is the sibling network repo;
override with --src. Output is a single compact parquet committed to this repo so the
on-chain exploration is reproducible without the full upstream panels.

Executable-arbitrage net (matches scripts/run_onchain_amm_gap.py exactly):
    abs_basis_bps = |(1 - implied_pool_price) * 1e4|         (capturable pool dislocation)
    net_bps       = abs_basis_bps - |pool_slippage_10k| - FEE_BPS      (FEE_BPS = 1.0)
Per-row gas is not populated in the source panel, so the explorer applies a fixed
conservative gas haircut (30 bps for a $10k swap) as a stated-assumption robustness check.

Run:  python scripts/pull_onchain_panel.py
Out:  data/gold/onchain_amm_panel.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[1]
FEE_BPS = 1.0
MAX_PLAUSIBLE_BPS = 1000.0
GAS_UNITS = 150_000
ETH_USD = 1500.0
NOTIONAL = 10_000.0
EVENTS = {
    "usdt_curve_2023": "USDT/Curve (Jun 2023)", "usdc_svb_2023": "USDC/SVB (Mar 2023)",
    "terra_luna_2022": "Terra/LUNA (May 2022)", "ftx_2022": "FTX (Nov 2022)",
    "busd_2023": "BUSD (Feb 2023)",
}
# modeling features available on-chain (all real pool/flow/gas state, no look-ahead)
FEATS = ["pool_slippage_10k", "reserve_imbalance", "exchange_netflow_1h",
         "mint_burn_net_1h", "gas_base_fee_gwei", "orderbook_imbalance", "spread_bps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT.parent / "stablecoin-contagion-network" / "data" / "gold"))
    args = ap.parse_args()
    src = Path(args.src)
    parts = []
    for ev, label in EVENTS.items():
        f = src / f"dataset_contagion_features_{ev}.parquet"
        if not f.exists():
            print(f"  (skip {ev}: {f} not found)"); continue
        df = pl.read_parquet(f)
        d = df.filter(pl.col("implied_pool_price").is_not_null() & pl.col("pool_slippage_10k").is_not_null())
        ip = d["implied_pool_price"].to_numpy()
        absb = np.abs((1.0 - ip) * 1e4)
        keep = absb <= MAX_PLAUSIBLE_BPS
        slip = np.abs(d["pool_slippage_10k"].to_numpy())
        net = absb - slip - FEE_BPS  # benchmark formula; per-row gas absent in source (handled
        #                              downstream as a fixed conservative haircut in the explorer)
        cols = {
            "event": [ev] * int(keep.sum()),
            "abs_basis_bps": absb[keep],
            "net_bps": net[keep],
            "optical_fire": (absb[keep] > 10).astype(np.int8),
            "net_positive": (net[keep] > 0).astype(np.int8),
            "t": d["event_time_seconds"].to_numpy()[keep] if "event_time_seconds" in d.columns else np.arange(int(keep.sum())),
        }
        for c in FEATS:
            cols[c] = (d[c].fill_null(0).to_numpy()[keep] if c in d.columns else np.zeros(int(keep.sum())))
        parts.append(pl.DataFrame(cols))
        print(f"  {ev:18s} rows={int(keep.sum()):5d} optical={int((absb[keep]>10).sum()):5d}")
    panel = pl.concat(parts, how="vertical")
    out = ROOT / "data/gold/onchain_amm_panel.parquet"
    panel.write_parquet(out)
    print(f"wrote {out.relative_to(ROOT)}  shape={panel.shape}")


if __name__ == "__main__":
    main()
