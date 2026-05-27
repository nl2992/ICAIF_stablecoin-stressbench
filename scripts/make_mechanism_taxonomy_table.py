"""Build mechanism taxonomy summary table (Table 18).

One row per mechanism_class. Summarises event counts, tier distribution,
max depeg, and empirical tractability.

Writes: results/paper_addon/table_18_mechanism_taxonomy_summary.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
YAML_PATH = ROOT / "configs" / "event_windows_historical.yaml"
OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "table_18_mechanism_taxonomy_summary.csv"

MECHANISM_LABELS = {
    "algorithmic_reflexive": "Algorithmic / Reflexive",
    "fiat_reserve_bank_shock": "Fiat-Reserve Bank Shock",
    "regulatory_issuer_winddown": "Regulatory / Issuer Winddown",
    "exchange_credit_liquidity": "Exchange Credit / Liquidity",
    "defi_pool_imbalance": "DeFi Pool Imbalance",
    "collateral_liquidation_oracle": "Collateral / Liquidation",
    "rwa_niche_stablecoin": "RWA / Niche Stablecoin",
}

MECHANISM_DESCRIPTION = {
    "algorithmic_reflexive": "Reflexive mint/burn or governance-token backing; failure triggered by bank-run dynamics",
    "fiat_reserve_bank_shock": "Fiat-collateral reserve held at stressed bank; redemption uncertainty drives CEX basis",
    "regulatory_issuer_winddown": "Regulatory order or issuer policy reduces stablecoin supply; orderly managed winddown",
    "exchange_credit_liquidity": "Exchange insolvency or withdrawal freeze; credit contagion through shared collateral",
    "defi_pool_imbalance": "DEX/AMM pool becomes imbalanced; on-chain price diverges from CEX mid-price",
    "collateral_liquidation_oracle": "Collateral price crash triggers forced liquidations; DAI trades above peg (demand shock)",
    "rwa_niche_stablecoin": "Real-world-asset backing or protocol exploit; illiquid collateral unable to meet redemptions",
}

FIELDNAMES = [
    "mechanism_class",
    "mechanism_label",
    "mechanism_description",
    "n_events",
    "tier_A",
    "tier_B",
    "tier_C",
    "max_depeg_bps_range",
    "duration_classes",
    "verified_events",
    "benchmark_role",
]


def main() -> None:
    with open(YAML_PATH, encoding="utf-8") as f:
        events = yaml.safe_load(f)["events"]

    # Group by mechanism_class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for ev_id, ev in events.items():
        mc = ev.get("mechanism_class", "unknown")
        by_class[mc].append({**ev, "_id": ev_id})

    rows = []
    for mc in MECHANISM_LABELS:
        evs = by_class.get(mc, [])
        if not evs:
            continue

        tiers = [e.get("data_tier", "C") for e in evs]
        depeg_vals = [abs(e.get("max_depeg_bps_est", 0)) for e in evs]
        depeg_min = min(depeg_vals) if depeg_vals else 0
        depeg_max = max(depeg_vals) if depeg_vals else 0
        durations = sorted({e.get("duration_class", "?") for e in evs})
        verified = sum(1 for e in evs if e.get("verification_status") == "verified")

        # Determine benchmark_role
        has_tier_a = "A" in tiers
        if has_tier_a:
            benchmark_role = "Primary benchmark source (Tier A)"
        elif mc in ("algorithmic_reflexive",):
            benchmark_role = "Validation split (Terra/UST), context for others"
        elif mc in ("exchange_credit_liquidity",):
            benchmark_role = "Illustrative credit contagion; not in benchmark splits"
        elif mc in ("regulatory_issuer_winddown",):
            benchmark_role = "Illustrative regulatory winddown; not in benchmark splits"
        elif mc in ("defi_pool_imbalance",):
            benchmark_role = "DeFi contagion illustration; motivates on-chain adapter"
        elif mc in ("collateral_liquidation_oracle",):
            benchmark_role = "Above-peg stress illustration; different mechanism class"
        else:
            benchmark_role = "Taxonomy / mechanism diversity only"

        depeg_range = (
            (f"−{depeg_max}" if depeg_min == 0 else f"−{depeg_min} to −{depeg_max}")
            if depeg_max > 0
            else "0 (operational event)"
        )
        # Special case: DAI Black Thursday is +150 bps
        if mc == "collateral_liquidation_oracle":
            depeg_range = "+150 (above peg)"

        rows.append(
            {
                "mechanism_class": mc,
                "mechanism_label": MECHANISM_LABELS[mc],
                "mechanism_description": MECHANISM_DESCRIPTION.get(mc, ""),
                "n_events": len(evs),
                "tier_A": tiers.count("A"),
                "tier_B": tiers.count("B"),
                "tier_C": tiers.count("C"),
                "max_depeg_bps_range": depeg_range,
                "duration_classes": ", ".join(durations),
                "verified_events": verified,
                "benchmark_role": benchmark_role,
            }
        )

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total_events = sum(r["n_events"] for r in rows)
    print(
        f"Wrote {len(rows)} mechanism classes, {total_events} total events → {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
