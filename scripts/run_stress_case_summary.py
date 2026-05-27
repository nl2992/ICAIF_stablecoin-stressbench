#!/opt/anaconda3/bin/python
"""
run_stress_case_summary.py
--------------------------
Compute stress-case summary metrics for the historical stress taxonomy.

For Tier-A events (USDC/SVB test + validation): use actual dataset.parquet metrics.
For Tier-B/C events: use synthetic estimates from event_windows_historical.yaml.

Outputs:
  results/paper_addon/table_21_stress_case_metrics.csv  — full metrics table (18 events)
"""

import os
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(REPO_ROOT, "data", "gold", "dataset.parquet")
HIST_YAML = os.path.join(REPO_ROOT, "configs", "event_windows_historical.yaml")
OUT_DIR = os.path.join(REPO_ROOT, "results", "paper_addon")
OUT_PATH = os.path.join(OUT_DIR, "table_21_stress_case_metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset_splits(path: str) -> dict:
    """Load dataset.parquet and return dict of DataFrames keyed by split name."""
    df = pd.read_parquet(path)
    return {split: df[df["split"] == split].copy() for split in df["split"].unique()}


def compute_split_metrics(df: pd.DataFrame) -> dict:
    """Compute actual metrics from a split DataFrame."""
    n = len(df)

    abs_basis = df["cross_quote_basis_maxabs_bps"].abs()
    usdc_basis_abs = df["cross_quote_basis_usdc_bps"].abs()

    # Price-basis rates
    pct_gt_10bps = (abs_basis > 10).mean() * 100
    pct_gt_25bps = (abs_basis > 25).mean() * 100
    pct_gt_50bps = (abs_basis > 50).mean() * 100
    usdc_pct_gt_10bps = (usdc_basis_abs > 10).mean() * 100

    max_abs_basis = abs_basis.max()
    spread_mean = (
        df["spread_bps_mean"].mean()
        if "spread_bps_mean" in df.columns
        else float("nan")
    )

    # Exec label (5-min forward, $10K notional, >0 bps threshold)
    exec_col = "label_arb_q10000_5m_gt0bps"
    if exec_col in df.columns:
        exec_positive_rate = df[exec_col].mean() * 100
        exec_label_available = True
    else:
        exec_positive_rate = float("nan")
        exec_label_available = False

    # Instantaneous net_profit >10 bps (matches paper Table 1 figure)
    if "net_profit_bps_q10000" in df.columns:
        net_gt_10_rate = (df["net_profit_bps_q10000"] > 10).mean() * 100
    else:
        net_gt_10_rate = float("nan")

    return {
        "n_minutes": n,
        "data_source": "dataset.parquet (real L2)",
        "max_abs_basis_bps": round(max_abs_basis, 1),
        "pct_gt_10bps": round(pct_gt_10bps, 2),
        "pct_gt_25bps": round(pct_gt_25bps, 2),
        "pct_gt_50bps": round(pct_gt_50bps, 2),
        "usdc_pct_gt_10bps": round(usdc_pct_gt_10bps, 2),
        "spread_bps_mean": round(spread_mean, 2),
        "exec_label_available": exec_label_available,
        "exec_positive_rate_5m_q10k": round(exec_positive_rate, 2),
        "net_profit_gt10bps_rate": round(net_gt_10_rate, 2),
    }


def build_tier_b_c_row(event_id: str, cfg: dict) -> dict:
    """Build a row for a Tier-B or Tier-C event using YAML estimates."""
    depeg_est = cfg.get("max_depeg_bps_est", None)
    abs_depeg = abs(depeg_est) if depeg_est is not None else None

    # Coarse pct estimates from duration and magnitude
    # We do NOT compute percentages for Tier B/C — they are labelled as "est."
    return {
        "n_minutes": None,
        "data_source": ", ".join(cfg.get("data_sources", [])),
        "max_abs_basis_bps": abs_depeg,
        "pct_gt_10bps": None,  # not computable without minute-level data
        "pct_gt_25bps": None,
        "pct_gt_50bps": None,
        "usdc_pct_gt_10bps": None,
        "spread_bps_mean": None,
        "exec_label_available": False,
        "exec_positive_rate_5m_q10k": None,
        "net_profit_gt10bps_rate": None,
    }


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading dataset.parquet ...")
    splits = load_dataset_splits(DATASET_PATH)
    train_metrics = compute_split_metrics(splits["train"])
    val_metrics = compute_split_metrics(splits["validation"])
    test_metrics = compute_split_metrics(splits["test"])
    print(f"  train: {train_metrics['n_minutes']} rows")
    print(
        f"  validation: {val_metrics['n_minutes']} rows, exec_pos={val_metrics['exec_positive_rate_5m_q10k']:.2f}%"
    )
    print(
        f"  test: {test_metrics['n_minutes']} rows, exec_pos={test_metrics['exec_positive_rate_5m_q10k']:.2f}%"
    )

    # ── Load YAML catalogue ──────────────────────────────────────────────────
    print("Loading historical event catalogue ...")
    with open(HIST_YAML) as f:
        hist = yaml.safe_load(f)
    events_cfg = hist.get("events", {})

    # ── Build rows ───────────────────────────────────────────────────────────
    # Display order: chronological by start date
    event_order = [
        "dai_black_thursday_2020",
        "fei_launch_2021",
        "iron_titan_2021",
        "mim_wonderland_2022",
        "terra_ust_2022",
        "usdd_tron_2022",
        "celsius_3ac_2022",
        "curve_3pool_ust_2022",
        "binance_stablecoin_conversion_2022",
        "husd_depeg_2022",
        "ftx_collapse_2022",
        "usdc_svb_2023",
        "usdc_svb_recovery_2023",
        "busd_regulatory_2023",
        "usdc_dai_secondary_svb_2023",
        "usdt_curve_2023",
        "acala_ausd_2022",
        "usdr_2023",
    ]

    # Friendly names
    event_names = {
        "dai_black_thursday_2020": "DAI Black Thursday 2020",
        "fei_launch_2021": "FEI Launch Stress 2021",
        "iron_titan_2021": "IRON/TITAN Collapse 2021",
        "mim_wonderland_2022": "MIM/Wonderland Shock 2022",
        "terra_ust_2022": "Terra/UST Collapse 2022",
        "usdd_tron_2022": "USDD/TRON Stress 2022",
        "celsius_3ac_2022": "Celsius/3AC Contagion 2022",
        "curve_3pool_ust_2022": "Curve 3Pool Imbalance 2022",
        "binance_stablecoin_conversion_2022": "Binance USDC→BUSD Conv. 2022",
        "husd_depeg_2022": "HUSD Issuer Failure 2022",
        "ftx_collapse_2022": "FTX Collapse 2022",
        "usdc_svb_2023": "USDC/SVB De-peg 2023",
        "usdc_svb_recovery_2023": "USDC/SVB Recovery 2023",
        "busd_regulatory_2023": "BUSD Regulatory 2023",
        "usdc_dai_secondary_svb_2023": "USDC/DAI DeFi Secondary 2023",
        "usdt_curve_2023": "USDT/Curve Pool 2023",
        "acala_ausd_2022": "Acala aUSD Exploit 2022",
        "usdr_2023": "USDR RWA Failure 2023",
    }

    # Role in paper
    event_roles = {
        "terra_ust_2022": "validation_split",
        "usdc_svb_2023": "primary_test_split",
        "usdc_svb_recovery_2023": "test_split_comparator",
        "ftx_collapse_2022": "context_exchange_credit",
        "busd_regulatory_2023": "context_regulatory",
        "usdt_curve_2023": "context_defi_pool",
        "curve_3pool_ust_2022": "context_defi_pool",
        "dai_black_thursday_2020": "context_collateral",
    }

    # Claim permission
    def claim_allowed(tier: str, event_id: str) -> str:
        if tier == "A":
            return "execution-grade: oracle bps, net bps, model eval"
        elif tier == "B":
            return "price-grade estimates only (est.)"
        else:
            return "taxonomy context only; no numerical claims"

    rows = []
    for eid in event_order:
        if eid not in events_cfg:
            print(f"  WARNING: {eid} not in YAML, skipping")
            continue

        cfg = events_cfg[eid]
        tier = cfg.get("data_tier", "C")
        mechanism = cfg.get("mechanism_class", "unknown")
        stablecoins = cfg.get("stablecoins", [])

        # Get metrics
        if eid == "usdc_svb_2023":
            m = test_metrics.copy()
        elif eid == "usdc_svb_recovery_2023":
            # Treat as part of test split (same split in dataset)
            m = test_metrics.copy()
            m["n_minutes"] = "incl. in test"
        elif eid == "terra_ust_2022":
            m = val_metrics.copy()
        else:
            m = build_tier_b_c_row(eid, cfg)

        row = {
            "event_id": eid,
            "event_name": event_names.get(eid, eid),
            "stablecoin": ", ".join(stablecoins[:2]),  # primary 1-2
            "mechanism_class": mechanism,
            "data_tier": tier,
            "start_date": cfg.get("start", "")[:10],
            "role_in_paper": event_roles.get(eid, "context"),
            "n_minutes": m["n_minutes"],
            "data_source": m["data_source"],
            "max_abs_basis_bps": m.get("max_abs_basis_bps"),
            "pct_gt_10bps": m.get("pct_gt_10bps"),
            "pct_gt_25bps": m.get("pct_gt_25bps"),
            "pct_gt_50bps": m.get("pct_gt_50bps"),
            "usdc_pct_gt_10bps": m.get("usdc_pct_gt_10bps"),
            "spread_bps_mean": m.get("spread_bps_mean"),
            "exec_label_available": m.get("exec_label_available", False),
            "exec_positive_rate_5m_q10k": m.get("exec_positive_rate_5m_q10k"),
            "net_profit_gt10bps_rate": m.get("net_profit_gt10bps_rate"),
            "claim_allowed": claim_allowed(tier, eid),
            "empirical_use": cfg.get("empirical_use", ""),
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df_out)} rows → {OUT_PATH}")

    # ── Preview ──────────────────────────────────────────────────────────────
    print("\n── Tier-A rows ──────────────────────────────────────────────────────")
    tier_a = df_out[df_out["data_tier"] == "A"]
    for _, r in tier_a.iterrows():
        print(
            f"  {r['event_id']}: pct_gt_10={r['pct_gt_10bps']}, exec_pos={r['exec_positive_rate_5m_q10k']}"
        )

    print("\n── Validation split row (Terra/UST) ─────────────────────────────────")
    val_row = df_out[df_out["event_id"] == "terra_ust_2022"]
    if not val_row.empty:
        r = val_row.iloc[0]
        print(
            f"  n_minutes={r['n_minutes']}, pct_gt_10={r['pct_gt_10bps']}, exec_pos={r['exec_positive_rate_5m_q10k']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
