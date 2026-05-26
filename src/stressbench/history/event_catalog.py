"""Historical stablecoin stress event catalogue.

Provides:
    DataTier         — Enum for A/B/C tier classification.
    EVENT_CATALOG    — Dict mapping event_id to metadata dict.
    get_tier_a_events()  — Returns only Tier A events.
    get_tier_b_events()  — Returns only Tier B events.
    get_tier_c_events()  — Returns only Tier C events.
    load_event_windows_yaml(path) — Loads the YAML config and merges with catalog.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class DataTier(str, Enum):
    """Data availability tier for historical stress events."""

    A = "A"  # Execution-grade: full L2, VWAP labels computable
    B = "B"  # Price-grade: OHLCV/DEX/trades, no full L2
    C = "C"  # Context-grade: partial data, historical context only


EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "iron_titan_2021": {
        "event_id": "iron_titan_2021",
        "display_name": "IRON/TITAN Collapse",
        "stablecoins": ["IRON"],
        "mechanism": (
            "Algorithmic stablecoin bank run; TITAN governance token collapse; "
            "IRON redeemability mechanism failed under reflexive sell pressure"
        ),
        "start": "2021-06-16T00:00:00Z",
        "end": "2021-06-17T23:59:59Z",
        "data_tier": DataTier.C,
        "data_sources": ["coingecko_ohlcv", "polygon_dex_swaps"],
        "empirical_use": "Taxonomy context only; no benchmark tasks",
        "coverage_score": 0.25,
        "peak_depeg_bps": -10000,  # Terminal: IRON to $0
        "notes": (
            "IRON was traded exclusively on QuickSwap (Polygon DEX). "
            "No CEX L2 depth available. Benchmark venues had minimal exposure. "
            "Terminal depeg: IRON to $0 in <24 hours."
        ),
    },
    "terra_ust_2022": {
        "event_id": "terra_ust_2022",
        "display_name": "Terra/UST Collapse",
        "stablecoins": ["UST", "USDT", "DAI"],
        "mechanism": (
            "Algorithmic stablecoin death spiral; LUNA mint/burn mechanism failure; "
            "Anchor Protocol bank run (~$14B TVL withdrawal)"
        ),
        "start": "2022-05-07T00:00:00Z",
        "end": "2022-05-14T23:59:59Z",
        "data_tier": DataTier.B,
        "data_sources": [
            "binance_ohlcv",
            "coinbase_ohlcv",
            "kraken_ohlcv",
            "curve_pool_reserves",
            "terra_onchain_transactions",
        ],
        "empirical_use": "Validation split illustrative; price-basis claims only",
        "coverage_score": 0.50,
        "peak_depeg_bps": -9800,  # UST to ~$0.02
        "notes": (
            "UST terminal depeg (-98%). USDT brief stress (<15 bps). "
            "USDC minimal (<10 bps). Contagion too brief for label construction. "
            "Curve 3pool imbalance was 12h leading indicator."
        ),
    },
    "ftx_collapse_2022": {
        "event_id": "ftx_collapse_2022",
        "display_name": "FTX Collapse",
        "stablecoins": ["USDT", "USDC", "DAI"],
        "mechanism": (
            "Exchange insolvency shock; FTX balance sheet fraud revealed; "
            "withdrawal halt and Chapter 11 filing"
        ),
        "start": "2022-11-06T00:00:00Z",
        "end": "2022-11-12T23:59:59Z",
        "data_tier": DataTier.B,
        "data_sources": [
            "binance_ohlcv",
            "coinbase_ohlcv",
            "kraken_ohlcv",
            "binance_partial_orderbook",
            "usdt_onchain_movements",
        ],
        "empirical_use": "Illustrative exchange-specific credit shock; not in benchmark splits",
        "coverage_score": 0.50,
        "peak_depeg_bps": -20,  # USDT -20 bps on Kraken
        "notes": (
            "USDT peak -20 bps (Kraken); USDC -5 bps. "
            "Stress was FTX-specific; benchmark venues show small contagion. "
            "No full L2 tape in benchmark dataset for this period."
        ),
    },
    "busd_regulatory_2023": {
        "event_id": "busd_regulatory_2023",
        "display_name": "BUSD Regulatory Winddown",
        "stablecoins": ["BUSD", "USDC", "USDT"],
        "mechanism": (
            "Regulatory enforcement; NYDFS halts BUSD minting; "
            "SEC Wells notice; Binance conversion program"
        ),
        "start": "2023-02-13T00:00:00Z",
        "end": "2023-03-08T23:59:59Z",
        "data_tier": DataTier.B,
        "data_sources": [
            "binance_ohlcv",
            "binance_partial_orderbook",
            "paxos_attestation_reports",
        ],
        "empirical_use": "Illustrative regulatory winddown dynamics; not in primary benchmark splits",
        "coverage_score": 0.50,
        "peak_depeg_bps": -30,  # BUSD -30 bps during conversion rush
        "notes": (
            "BUSD peak -30 bps during conversion rush. "
            "USDC/USDT +10 bps appreciation as conversion destination. "
            "Occurred 3 weeks before SVB event."
        ),
    },
    "usdc_svb_2023": {
        "event_id": "usdc_svb_2023",
        "display_name": "USDC/SVB Stress",
        "stablecoins": ["USDC", "DAI", "USDT"],
        "mechanism": (
            "Reserve bank insolvency; SVB seized by FDIC; "
            "Circle held ~$3.3B (~8% of reserves) at SVB"
        ),
        "start": "2023-03-10T00:00:00Z",
        "end": "2023-03-14T23:59:59Z",
        "data_tier": DataTier.A,
        "data_sources": [
            "binance_real_l2_snapshot",
            "coinbase_real_l2_snapshot",
            "kraken_real_l2_snapshot",
            "binance_trade_tape",
            "coinbase_trade_tape",
            "kraken_trade_tape",
            "usdc_onchain_redemptions",
        ],
        "empirical_use": "PRIMARY benchmark test split; all paper claims anchored here",
        "coverage_score": 1.0,
        "peak_depeg_bps": -1300,  # USDC to ~$0.87
        "notes": (
            "USDC peak -1300 bps (~$0.87). DAI -200 bps. USDT +50 bps premium. "
            "Oracle earns 161-225 net bps/trade. All models produce negative net bps. "
            "12x price-to-execution gap."
        ),
    },
    "usdc_svb_recovery_2023": {
        "event_id": "usdc_svb_recovery_2023",
        "display_name": "USDC Recovery (Post-SVB)",
        "stablecoins": ["USDC", "DAI", "USDT"],
        "mechanism": (
            "Post-SVB recovery; US Treasury+Fed+FDIC joint statement guaranteed full SVB deposits"
        ),
        "start": "2023-03-15T00:00:00Z",
        "end": "2023-04-01T23:59:59Z",
        "data_tier": DataTier.A,
        "data_sources": [
            "binance_real_l2_snapshot",
            "coinbase_real_l2_snapshot",
            "kraken_real_l2_snapshot",
            "binance_trade_tape",
            "coinbase_trade_tape",
        ],
        "empirical_use": "Test split recovery window; validates regime generalization",
        "coverage_score": 0.75,
        "peak_depeg_bps": -10,  # Normal cross-venue basis noise only
        "notes": (
            "USDC fully restored to $1.000 by Mar 15. "
            "Very few executable arbitrage windows in recovery period. "
            "Important comparator: model FP rate should drop post-recovery."
        ),
    },
    "usdt_curve_2023": {
        "event_id": "usdt_curve_2023",
        "display_name": "USDT/Curve Pool Stress",
        "stablecoins": ["USDT", "USDC", "DAI"],
        "mechanism": (
            "Curve pool imbalance; Tether reserve concerns re-emerge; "
            "brief USDT discount"
        ),
        "start": "2023-06-12T00:00:00Z",
        "end": "2023-06-15T23:59:59Z",
        "data_tier": DataTier.B,
        "data_sources": [
            "binance_ohlcv",
            "coinbase_ohlcv",
            "curve_pool_reserves",
            "usdt_onchain_transfers",
        ],
        "empirical_use": "Out-of-sample illustrative; motivates on-chain Curve data integration",
        "coverage_score": 0.50,
        "peak_depeg_bps": -8,  # USDT -8 bps
        "notes": (
            "USDT peak -8 bps; USDC +3 bps premium. Very short duration (hours). "
            "Below benchmark test split; insufficient label density. "
            "Curve pool data available but not integrated into benchmark dataset."
        ),
    },
}


def get_tier_a_events() -> dict[str, dict[str, Any]]:
    """Return all Tier A (execution-grade) events from the catalog."""
    return {k: v for k, v in EVENT_CATALOG.items() if v["data_tier"] == DataTier.A}


def get_tier_b_events() -> dict[str, dict[str, Any]]:
    """Return all Tier B (price-grade) events from the catalog."""
    return {k: v for k, v in EVENT_CATALOG.items() if v["data_tier"] == DataTier.B}


def get_tier_c_events() -> dict[str, dict[str, Any]]:
    """Return all Tier C (context-grade) events from the catalog."""
    return {k: v for k, v in EVENT_CATALOG.items() if v["data_tier"] == DataTier.C}


def load_event_windows_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load event windows from YAML config, merging with EVENT_CATALOG metadata.

    The YAML format follows configs/event_windows_historical.yaml.
    Each event is merged with its corresponding EVENT_CATALOG entry if found.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dict mapping event_id to merged metadata dict.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event windows YAML not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    events_raw = raw.get("events", {})
    merged: dict[str, dict[str, Any]] = {}

    for event_id, yaml_data in events_raw.items():
        entry = dict(yaml_data)
        entry["event_id"] = event_id

        # Normalise data_tier to DataTier enum
        tier_str = str(entry.get("data_tier", "C")).upper()
        entry["data_tier"] = DataTier(tier_str)

        # Merge with catalog entry if available (catalog values are overridden by YAML)
        if event_id in EVENT_CATALOG:
            catalog_entry = dict(EVENT_CATALOG[event_id])
            catalog_entry.update(entry)
            entry = catalog_entry

        merged[event_id] = entry

    return merged
