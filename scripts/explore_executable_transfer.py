#!/usr/bin/env python3
"""Honest out-of-sample exploration of executable-window separability on StressBench.

Question: among price-fire windows (|cross-quote basis| > 10 bps), can any learned model
select the *economically profitable* ones (positive VWAP-book-walk net bps) out-of-sample,
under realistic training protocols? This is the open challenge the benchmark poses.

Design (anti-overfitting / anti-p-hacking guardrails):
  * Real gold panel only (data/gold/dataset.parquet) -- never synthetic data.
  * Honest OOS: the decision threshold is calibrated on the TRAIN segment, never on test.
  * A grid of (training protocol x feature set x model) is tried; EVERY path is logged to
    results/exploration/executable_transfer_ledger.csv (append-only), win or lose.
  * Each path reports a bootstrap one-sided p (H0: mean net <= 0) and a 95% CI.
  * A path counts as a POSITIVE FINDING only if ALL hold: net_bps>0, bootstrap CI lower
    bound>0, n_trades>=30, and it survives Benjamini-Hochberg FDR(0.10) across the whole
    ledger. Trying many paths and keeping the best is dredging; requiring FDR survival is not.

Run:  pip install -e .  &&  python scripts/explore_executable_transfer.py
"""
from __future__ import annotations
import csv, datetime as dt, itertools
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "results/exploration/executable_transfer_ledger.csv"
THR = 10.0; ORACLE = 162.2
BOOK = ["depth_bid_10bp_mean", "depth_ask_10bp_mean", "spread_bps_mean", "imbalance_1bp_mean"]
NETS = {"q10000": "net_profit_bps_q10000", "q50000": "net_profit_bps_q50000"}
WIN = {"terra": ((2022,5,7),(2022,5,14,23,59,59)), "svb": ((2023,3,10),(2023,3,20,23,59,59))}
FIRE_BASIS = {"terra": "cross_quote_basis_maxabs_bps", "svb": "cross_quote_basis_usdc_bps"}


def _seg(df, key):
    a, b = WIN[key]
    return df.filter((pl.col("dt") >= dt.datetime(*a)) & (pl.col("dt") <= dt.datetime(*b))).sort("dt")

def _feat(seg, basiscol, fs):
    b = seg[basiscol].fill_null(0).to_numpy()
    if fs == "price_only":
        return b, b.reshape(-1, 1)
    return b, np.column_stack([b] + [seg[c].fill_null(0).to_numpy() for c in BOOK])

def _model(name):
    if name == "lgbm":
        from stressbench.models.meta_labeling import MetaLabelingFilter
        return "meta", MetaLabelingFilter(primary_threshold_bps=THR, primary_signal_col=0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    return "sk", (LogisticRegression(max_iter=500) if name == "logistic"
                  else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))

def _calib(proba, net, mintr=25):
    bt, bv = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 60):
        s = proba > t
        if s.sum() < mintr:
            continue
        if net[s].sum() > bv:
            bv, bt = net[s].sum(), t
    return bt

def _boot(traded, B=2000, seed=0):
    if len(traded) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); n = len(traded)
    means = np.array([traded[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean())

def run_path(df, protocol, fs, model_name, net_key):
    netcol = NETS[net_key]
    svb = _seg(df, "svb"); bsvb, Xsvb = _feat(svb, FIRE_BASIS["svb"], fs)
    nsvb = svb[netcol].fill_null(-15.0).to_numpy(); fsvb = np.abs(bsvb) > THR
    kind, mdl = _model(model_name)
    def fit_predict(Xtr, ptr, mtr, Xte):
        if kind == "meta":
            mdl.fit(Xtr, ptr, mtr); return mdl.predict_proba(Xte)[:, 1]
        m = mtr[ptr == 1]; xx = Xtr[ptr == 1]
        if len(np.unique(m)) < 2:
            return np.zeros(len(Xte))
        mdl.fit(xx, m); return mdl.predict_proba(Xte)[:, 1]
    if protocol == "crossmech":
        tr = _seg(df, "terra"); btr, Xtr = _feat(tr, FIRE_BASIS["terra"], fs)
        ntr = tr[netcol].fill_null(-15.0).to_numpy(); ftr = np.abs(btr) > THR
        mtr = (ftr & (ntr > 0)).astype(np.int8)
        proba = fit_predict(Xtr, ftr.astype(np.int8), mtr, Xsvb)
        th = _calib(fit_predict(Xtr, ftr.astype(np.int8), mtr, Xtr), ntr); traded = nsvb[proba > th]
    elif protocol == "control2svb":
        ctl = df.filter((pl.col("dt") < dt.datetime(2022, 5, 1)) | (pl.col("dt") > dt.datetime(2023, 4, 1))).sort("dt")
        bc, Xc = _feat(ctl, FIRE_BASIS["svb"], fs); nc = ctl[netcol].fill_null(-15.0).to_numpy()
        fc = np.abs(bc) > THR; mc = (fc & (nc > 0)).astype(np.int8)
        if fc.sum() < 20 or mc.sum() < 2:
            return None
        proba = fit_predict(Xc, fc.astype(np.int8), mc, Xsvb)
        th = _calib(fit_predict(Xc, fc.astype(np.int8), mc, Xc), nc); traded = nsvb[proba > th]
    elif protocol == "inevent_wf":
        cut = svb.height // 2; msvb = (fsvb & (nsvb > 0)).astype(np.int8)
        proba = fit_predict(Xsvb[:cut], fsvb[:cut].astype(np.int8), msvb[:cut], Xsvb[cut:])
        th = _calib(fit_predict(Xsvb[:cut], fsvb[:cut].astype(np.int8), msvb[:cut], Xsvb[:cut]), nsvb[:cut])
        traded = nsvb[cut:][proba > th]
    else:
        return None
    if len(traded) == 0:
        return dict(protocol=protocol, fs=fs, model=model_name, net_key=net_key, n=0,
                    net_bps=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), boot_p=float("nan"))
    lo, hi, p = _boot(traded)
    return dict(protocol=protocol, fs=fs, model=model_name, net_key=net_key, n=int(len(traded)),
                net_bps=round(float(traded.mean()), 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                boot_p=round(p, 4), oracle_cap=round(float(traded.mean()) / ORACLE, 4))

def bh_findings(rows, q=0.10):
    cand = [r for r in rows if r.get("n", 0) >= 30 and r.get("net_bps", -1) > 0
            and r.get("ci_lo", -1) > 0 and not np.isnan(r.get("boot_p", np.nan))]
    if not cand:
        return []
    cand = sorted(cand, key=lambda r: r["boot_p"]); m = len(rows)
    return [r for i, r in enumerate(cand, 1) if r["boot_p"] <= (i / m) * q]

def main():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(ROOT / "data/gold/dataset.parquet").with_columns(
        pl.from_epoch(pl.col("ts_1m_ns"), time_unit="ns").alias("dt"))
    grid = list(itertools.product(["crossmech", "control2svb", "inevent_wf"],
                                  ["price_only", "price_plus_book"],
                                  ["lgbm", "logistic", "random_forest"], ["q10000"]))
    batch = dt.datetime.now().strftime("%Y%m%dT%H%M%S"); rows = []
    for proto, fs, mdl, nk in grid:
        try:
            r = run_path(df, proto, fs, mdl, nk)
        except Exception as e:
            r = dict(protocol=proto, fs=fs, model=mdl, net_key=nk, n=-1, net_bps=float("nan"),
                     ci_lo=float("nan"), ci_hi=float("nan"), boot_p=float("nan"), err=str(e)[:80])
        if r:
            r["batch"] = batch; rows.append(r); print(r)
    allrows = list(csv.DictReader(open(LEDGER))) if LEDGER.exists() else []
    fields = sorted(set().union(*[set(r) for r in rows + allrows])) if (rows or allrows) else []
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in allrows + rows:
            w.writerow(r)
    led = []
    for r in allrows + rows:
        d = dict(r)
        for k in ("n", "net_bps", "ci_lo", "boot_p"):
            try:
                d[k] = float(r.get(k, "nan"))
            except (TypeError, ValueError):
                d[k] = float("nan")
        led.append(d)
    f = bh_findings(led)
    print(f"\n=== ledger={len(led)} paths.  BH-FDR(0.10) survivors (net>0 & CI_lo>0 & n>=30): {len(f)} ===")
    for r in f:
        print("  FINDING:", r)
    if not f:
        print("  No honest positive survives correction -> benchmark conclusion stands "
              "(the executable-window selection problem is unsolved on this venue).")

if __name__ == "__main__":
    main()
