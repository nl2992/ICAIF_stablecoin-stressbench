"""Rebuild historical catalogue tables 14 and 15 from the 18-event YAML.

Writes:
  results/paper_addon/table_14_historical_event_catalog.csv
  results/paper_addon/table_15_event_data_coverage.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv

import yaml

YAML_PATH = ROOT / "configs" / "event_windows_historical.yaml"
OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

TIER_LABEL = {
    "A": "Tier A (execution-grade)",
    "B": "Tier B (price/liquidity-grade)",
    "C": "Tier C (context-grade)",
}

# Readable mechanism class names
MECHANISM_LABELS = {
    "algorithmic_reflexive": "Algorithmic / Reflexive",
    "fiat_reserve_bank_shock": "Fiat-Reserve Bank Shock",
    "regulatory_issuer_winddown": "Regulatory / Issuer Winddown",
    "exchange_credit_liquidity": "Exchange Credit / Liquidity",
    "defi_pool_imbalance": "DeFi Pool Imbalance",
    "collateral_liquidation_oracle": "Collateral / Liquidation",
    "rwa_niche_stablecoin": "RWA / Niche Stablecoin",
}


def load_events() -> dict:
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]


# ── Table 14: event catalog ───────────────────────────────────────────────────

T14_FIELDS = [
    "event_id",
    "display_name",
    "mechanism_class",
    "stablecoins",
    "start",
    "end",
    "data_tier",
    "coverage_score",
    "max_depeg_bps_est",
    "duration_class",
    "verification_status",
    "empirical_use",
]

# Map from event_id to human-readable display name
DISPLAY_NAMES = {
    "fei_launch_2021": "FEI Launch Stress",
    "iron_titan_2021": "IRON/TITAN Collapse",
    "mim_wonderland_2022": "MIM/Wonderland Shock",
    "terra_ust_2022": "Terra/UST Collapse",
    "usdd_tron_2022": "USDD/TRON Stress",
    "celsius_3ac_2022": "Celsius/3AC Contagion",
    "husd_depeg_2022": "HUSD Issuer Failure",
    "ftx_collapse_2022": "FTX Collapse",
    "busd_regulatory_2023": "BUSD Regulatory Winddown",
    "binance_stablecoin_conversion_2022": "Binance USDC→BUSD Conversion",
    "usdc_svb_2023": "USDC/SVB Stress (PRIMARY)",
    "usdc_svb_recovery_2023": "USDC Recovery (Post-SVB)",
    "curve_3pool_ust_2022": "Curve 3Pool / UST Stress",
    "usdt_curve_2023": "USDT/Curve Pool Stress",
    "usdc_dai_secondary_svb_2023": "USDC/DAI DeFi Secondary Stress",
    "dai_black_thursday_2020": "DAI Black Thursday",
    "acala_ausd_2022": "Acala aUSD Exploit",
    "usdr_2023": "USDR RWA Failure",
}


def build_table14(events: dict) -> list[dict]:
    rows = []
    for event_id, ev in events.items():
        coins = ev.get("stablecoins", [])
        stablecoins_str = (
            ", ".join(coins) if isinstance(coins, list) else str(coins)
        )
        rows.append(
            {
                "event_id": event_id,
                "display_name": DISPLAY_NAMES.get(event_id, event_id),
                "mechanism_class": MECHANISM_LABELS.get(
                    ev.get("mechanism_class", ""), ev.get("mechanism_class", "")
                ),
                "stablecoins": stablecoins_str,
                "start": str(ev.get("start", ""))[:10],
                "end": str(ev.get("end", ""))[:10],
                "data_tier": ev.get("data_tier", ""),
                "coverage_score": ev.get("coverage_score", ""),
                "max_depeg_bps_est": ev.get("max_depeg_bps_est", ""),
                "duration_class": ev.get("duration_class", ""),
                "verification_status": ev.get("verification_status", ""),
                "empirical_use": ev.get("empirical_use", ""),
            }
        )
    return rows


# ── Table 15: data coverage matrix ──────────────────────────────────────────

T15_FIELDS = [
    "event_id",
    "data_tier",
    "coverage_score",
    "has_l2_depth",
    "has_trade_tape",
    "has_ohlcv",
    "has_onchain",
    "has_defi_pool",
    "data_sources_count",
    "data_sources_list",
]

L2_SOURCES = {
    "binance_real_l2_snapshot",
    "binance_real_l2_incremental",
    "coinbase_real_l2_snapshot",
    "kraken_real_l2_snapshot",
    "huobi_partial_orderbook",
    "binance_partial_orderbook",
}
TAPE_SOURCES = {
    "binance_trade_tape",
    "coinbase_trade_tape",
    "kraken_trade_tape",
}
OHLCV_SOURCES = {
    "binance_ohlcv",
    "coinbase_ohlcv",
    "kraken_ohlcv",
    "coingecko_ohlcv",
    "huobi_ohlcv",
    "poloniex_ohlcv",
}
ONCHAIN_SOURCES = {
    "usdc_onchain_redemptions",
    "terra_onchain_transactions",
    "usdt_onchain_movements",
    "ethereum_onchain_liquidations",
    "makerdao_vault_data",
    "onchain_celsius_outflows",
    "acala_onchain_data",
    "tangible_protocol_onchain",
    "uniswap_v2_swaps",
    "polygon_dex_swaps",
    "abracadabra_onchain",
    "usdt_onchain_transfers",
}
DEFI_SOURCES = {
    "curve_pool_reserves",
    "curve_3pool_reserves",
    "curve_mim_3pool_reserves",
    "curve_steth_pool_reserves",
    "subgraph_curve_swaps",
    "makerdao_psm_data",
}


def build_table15(events: dict) -> list[dict]:
    rows = []
    for event_id, ev in events.items():
        sources = set(ev.get("data_sources", []))
        rows.append(
            {
                "event_id": event_id,
                "data_tier": ev.get("data_tier", ""),
                "coverage_score": ev.get("coverage_score", ""),
                "has_l2_depth": "1" if sources & L2_SOURCES else "0",
                "has_trade_tape": "1" if sources & TAPE_SOURCES else "0",
                "has_ohlcv": "1" if sources & OHLCV_SOURCES else "0",
                "has_onchain": "1" if sources & ONCHAIN_SOURCES else "0",
                "has_defi_pool": "1" if sources & DEFI_SOURCES else "0",
                "data_sources_count": len(sources),
                "data_sources_list": "; ".join(sorted(sources)),
            }
        )
    return rows


# ── main ─────────────────────────────────────────────────────────────────────


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def main() -> None:
    events = load_events()
    print(f"Loaded {len(events)} events from {YAML_PATH.name}")

    t14 = build_table14(events)
    write_csv(OUT_DIR / "table_14_historical_event_catalog.csv", T14_FIELDS, t14)

    t15 = build_table15(events)
    write_csv(OUT_DIR / "table_15_event_data_coverage.csv", T15_FIELDS, t15)

    # Summary
    tiers = {}
    for ev in events.values():
        t = ev.get("data_tier", "?")
        tiers[t] = tiers.get(t, 0) + 1
    for tier, count in sorted(tiers.items()):
        print(f"  Tier {tier}: {count} events")


if __name__ == "__main__":
    main()
