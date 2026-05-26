"""Build venue-depth execution provenance table (Table 20).

Documents, per event and per venue, what data was available and whether it
contributed to net-profit labels. This is the authoritative backing for the
claim "execution-grade labels are computed from real L2 routes."

Writes: results/paper_addon/table_20_execution_provenance_by_event_venue.csv

IMPORTANT: Rows with execution_grade=False cannot support net-profit label claims.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "table_20_execution_provenance_by_event_venue.csv"

FIELDNAMES = [
    "event_id",
    "data_tier",
    "venue",
    "instrument",
    "ohlcv_available",
    "trade_tape_available",
    "l2_snapshot_available",
    "l2_incremental_available",
    "depth_source_tag",
    "used_in_net_profit_label",
    "execution_grade",
    "notes",
]

# ---------------------------------------------------------------------------
# Provenance records — manually curated from data_card.md and pipeline code
# ---------------------------------------------------------------------------
# Key:
#   execution_grade=True  ↔ depth_source ∈ {real_l2_snapshot, real_l2_incremental}
#                           AND used_in_net_profit_label=True
#   Coinbase/Kraken historical L2 requires Tardis subscription for SVB period

ROWS = [
    # ── USDC/SVB (Tier A) ──────────────────────────────────────────────────
    {
        "event_id": "usdc_svb_2023",
        "data_tier": "A",
        "venue": "Binance",
        "instrument": "BTCUSDT / BTCUSDC",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "True",
        "l2_incremental_available": "True",
        "depth_source_tag": "real_l2_incremental",
        "used_in_net_profit_label": "True",
        "execution_grade": "True",
        "notes": (
            "Binance USDM futures bookDepth available from Binance Vision archive. "
            "Primary depth source for VWAP net-profit labels."
        ),
    },
    {
        "event_id": "usdc_svb_2023",
        "data_tier": "A",
        "venue": "Coinbase",
        "instrument": "BTC-USD",
        "ohlcv_available": "True",
        "trade_tape_available": "False",
        "l2_snapshot_available": "Partial",
        "l2_incremental_available": "False",
        "depth_source_tag": "real_l2_snapshot (Tardis required for full historical replay)",
        "used_in_net_profit_label": "Partial",
        "execution_grade": "Partial",
        "notes": (
            "Coinbase REST candles (OHLCV) available without subscription. "
            "Full L2 snapshot history for March 2023 requires Tardis archive subscription. "
            "Not included in committed dataset.parquet. "
            "Live capture pipeline supports real_l2_snapshot if subscription is available."
        ),
    },
    {
        "event_id": "usdc_svb_2023",
        "data_tier": "A",
        "venue": "Kraken",
        "instrument": "XXBTZUSD",
        "ohlcv_available": "False",
        "trade_tape_available": "False",
        "l2_snapshot_available": "Partial",
        "l2_incremental_available": "False",
        "depth_source_tag": "real_l2_snapshot (Tardis required for historical replay)",
        "used_in_net_profit_label": "Partial",
        "execution_grade": "Partial",
        "notes": (
            "Kraken data not available from free public archive. "
            "Full L2 snapshot history requires Tardis subscription. "
            "Not in committed dataset.parquet. "
            "Live capture pipeline supports real_l2_snapshot via Kraken WS."
        ),
    },
    # ── USDC Recovery (Tier A partial) ───────────────────────────────────────
    {
        "event_id": "usdc_svb_recovery_2023",
        "data_tier": "A",
        "venue": "Binance",
        "instrument": "BTCUSDT / BTCUSDC",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "True",
        "l2_incremental_available": "True",
        "depth_source_tag": "real_l2_incremental",
        "used_in_net_profit_label": "True",
        "execution_grade": "True",
        "notes": "Same as SVB period; Binance archive covers recovery window.",
    },
    {
        "event_id": "usdc_svb_recovery_2023",
        "data_tier": "A",
        "venue": "Coinbase",
        "instrument": "BTC-USD",
        "ohlcv_available": "True",
        "trade_tape_available": "False",
        "l2_snapshot_available": "Partial",
        "l2_incremental_available": "False",
        "depth_source_tag": "real_l2_snapshot (Tardis required)",
        "used_in_net_profit_label": "Partial",
        "execution_grade": "Partial",
        "notes": "Same as SVB period; Tardis required for full L2.",
    },
    # ── Tier B events (price-grade only) ─────────────────────────────────────
    {
        "event_id": "terra_ust_2022",
        "data_tier": "B",
        "venue": "Binance",
        "instrument": "BTCUSDT / BTCUSDC",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "Partial",
        "l2_incremental_available": "False",
        "depth_source_tag": "synthetic_kline (OHLCV-derived)",
        "used_in_net_profit_label": "False",
        "execution_grade": "False",
        "notes": (
            "Binance Vision kline archives available. Real L2 depth not captured "
            "during May 2022. Synthetic kline depth excluded from paper-grade labels. "
            "Price-basis claims only."
        ),
    },
    {
        "event_id": "ftx_collapse_2022",
        "data_tier": "B",
        "venue": "Binance",
        "instrument": "BTCUSDT / BTCUSDC",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "Partial",
        "l2_incremental_available": "False",
        "depth_source_tag": "synthetic_kline",
        "used_in_net_profit_label": "False",
        "execution_grade": "False",
        "notes": "Binance archive available. Full L2 not captured during Nov 2022. Price-basis claims only.",
    },
    {
        "event_id": "busd_regulatory_2023",
        "data_tier": "B",
        "venue": "Binance",
        "instrument": "BBUSDUSDT",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "True",
        "l2_incremental_available": "False",
        "depth_source_tag": "real_l2_snapshot (Binance archive, but BUSD not benchmark instrument)",
        "used_in_net_profit_label": "False",
        "execution_grade": "False",
        "notes": (
            "BUSD data available from Binance archive but BUSD is not a benchmark "
            "instrument (benchmark focuses on USDC/USDT/BTC routes). Price-basis claims only."
        ),
    },
    {
        "event_id": "usdt_curve_2023",
        "data_tier": "B",
        "venue": "Binance",
        "instrument": "BTCUSDT / BTCUSDC",
        "ohlcv_available": "True",
        "trade_tape_available": "True",
        "l2_snapshot_available": "False",
        "l2_incremental_available": "False",
        "depth_source_tag": "synthetic_kline",
        "used_in_net_profit_label": "False",
        "execution_grade": "False",
        "notes": (
            "June 2023 falls after benchmark test split. Binance klines available. "
            "Full L2 not in benchmark dataset. Price-basis claims only."
        ),
    },
]


def main() -> None:
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ROWS)

    exec_grade = sum(1 for r in ROWS if r["execution_grade"] == "True")
    partial = sum(1 for r in ROWS if r["execution_grade"] == "Partial")
    not_exec = sum(1 for r in ROWS if r["execution_grade"] == "False")
    print(f"Wrote {len(ROWS)} venue-event rows → {OUT_FILE}")
    print(f"  Execution-grade (True): {exec_grade}")
    print(f"  Partial (Tardis needed): {partial}")
    print(f"  Not execution-grade: {not_exec}")
    print()
    print("Key finding: Binance is the confirmed real-L2 venue for VWAP labels.")
    print("Coinbase and Kraken require Tardis subscription for historical L2 replay.")


if __name__ == "__main__":
    main()
