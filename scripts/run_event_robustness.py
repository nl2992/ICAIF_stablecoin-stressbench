"""Event robustness table: how would price-to-execution gap vary across event types?

This script builds a hypothetical robustness table showing that the 12× price-
to-execution gap documented for USDC/SVB (Tier A) would manifest differently
under each mechanism class, based on the structural characteristics of each
mechanism and its typical depth/liquidity profile.

This is NOT a claim about other events having specific execution gaps —
we do not have Tier A data for them. The table is clearly labelled as
"structural characterisation based on mechanism class" and must be framed
as such in the paper.

Writes: results/paper_addon/table_19_event_robustness.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "paper_addon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "table_19_event_robustness.csv"

FIELDNAMES = [
    "mechanism_class",
    "mechanism_label",
    "rep_event",
    "data_tier",
    "typical_price_basis_magnitude",
    "depth_availability_at_stress",
    "expected_price_exec_ratio",
    "execution_tractability",
    "why_gap_differs",
    "paper_claim_level",
]

# Structural characterisation based on mechanism class
# Sources: domain knowledge about CEX depth during each stress type
ROWS = [
    {
        "mechanism_class": "fiat_reserve_bank_shock",
        "mechanism_label": "Fiat-Reserve Bank Shock",
        "rep_event": "USDC/SVB Mar 2023",
        "data_tier": "A",
        "typical_price_basis_magnitude": "100–1300 bps (reserve uncertainty drives basis)",
        "depth_availability_at_stress": "Moderate — CEX depth shrinks as MMs pull quotes",
        "expected_price_exec_ratio": "~12× (empirically verified at $10K/1m)",
        "execution_tractability": "Low — basis real but executable windows rare (2.88%)",
        "why_gap_differs": "Reserve uncertainty creates quoted-price gaps; MMs pull depth simultaneously",
        "paper_claim_level": "execution_grade",
    },
    {
        "mechanism_class": "algorithmic_reflexive",
        "mechanism_label": "Algorithmic / Reflexive",
        "rep_event": "Terra/UST May 2022",
        "data_tier": "B",
        "typical_price_basis_magnitude": "Very large (100–10000 bps); terminal in UST case",
        "depth_availability_at_stress": "Very low — terminal events exhaust all available depth",
        "expected_price_exec_ratio": "Likely >> 12× (est.); terminal events not executable",
        "execution_tractability": "Very low — depth exhausted before execution possible",
        "why_gap_differs": "Reflexive collapse drains both sides of order book; no arbitrage depth",
        "paper_claim_level": "price_grade_est",
    },
    {
        "mechanism_class": "exchange_credit_liquidity",
        "mechanism_label": "Exchange Credit / Liquidity",
        "rep_event": "FTX Collapse Nov 2022",
        "data_tier": "B",
        "typical_price_basis_magnitude": "Small (5–50 bps at benchmark venues); larger on FTX itself",
        "depth_availability_at_stress": "High on non-FTX venues (Binance/Coinbase unaffected)",
        "expected_price_exec_ratio": "Possibly low (2–5× est.) — depth available at benchmark venues",
        "execution_tractability": "Potentially higher — healthy venues maintain depth",
        "why_gap_differs": "Credit shock isolated to one exchange; cross-venue depth intact",
        "paper_claim_level": "price_grade_est",
    },
    {
        "mechanism_class": "regulatory_issuer_winddown",
        "mechanism_label": "Regulatory / Issuer Winddown",
        "rep_event": "BUSD Regulatory Feb 2023",
        "data_tier": "B",
        "typical_price_basis_magnitude": "Small (10–30 bps); orderly conversion",
        "depth_availability_at_stress": "Moderate — Binance maintained BUSD depth during conversion",
        "expected_price_exec_ratio": "Likely low (2–5× est.) — managed process, not panic",
        "execution_tractability": "Higher — orderly winddown preserves some depth",
        "why_gap_differs": "Known regulatory schedule reduces panic; depth degrades slowly",
        "paper_claim_level": "price_grade_est",
    },
    {
        "mechanism_class": "defi_pool_imbalance",
        "mechanism_label": "DeFi Pool Imbalance",
        "rep_event": "USDT/Curve Jun 2023",
        "data_tier": "B",
        "typical_price_basis_magnitude": "Small-to-moderate (8–500 bps, larger within pool)",
        "depth_availability_at_stress": "CEX depth mostly intact; on-chain pool liquidity depleted",
        "expected_price_exec_ratio": "Moderate (est. 5–10×) — CEX side more liquid",
        "execution_tractability": "Moderate if CEX-only; low if requires on-chain settlement",
        "why_gap_differs": "On-chain settlement adds transfer latency and gas cost; CEX-to-DEX arbitrage harder",
        "paper_claim_level": "price_grade_est",
    },
    {
        "mechanism_class": "collateral_liquidation_oracle",
        "mechanism_label": "Collateral / Liquidation",
        "rep_event": "DAI Black Thursday Mar 2020",
        "data_tier": "B",
        "typical_price_basis_magnitude": "+150 bps (above peg — demand shock, not discount)",
        "depth_availability_at_stress": "Low on DAI; ETH side depth collapsed",
        "expected_price_exec_ratio": "N/A — above-peg stress; different mechanism class",
        "execution_tractability": "Different framing: DAI above peg means sell DAI (not buy)",
        "why_gap_differs": "Collateral liquidation creates demand shock; different arb direction",
        "paper_claim_level": "price_grade_est",
    },
    {
        "mechanism_class": "rwa_niche_stablecoin",
        "mechanism_label": "RWA / Niche Stablecoin",
        "rep_event": "USDR Oct 2023",
        "data_tier": "B",
        "typical_price_basis_magnitude": "Large (500–5000 bps est.); illiquid collateral",
        "depth_availability_at_stress": "Very low — niche assets with thin CEX presence",
        "expected_price_exec_ratio": "Not comparable — no benchmark-venue liquidity",
        "execution_tractability": "Very low — insufficient CEX depth for execution",
        "why_gap_differs": "Niche assets have no execution-grade CEX order books",
        "paper_claim_level": "price_grade_est",
    },
]


def main() -> None:
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ROWS)

    exec_grade = sum(1 for r in ROWS if r["paper_claim_level"] == "execution_grade")
    print(f"Wrote {len(ROWS)} mechanism robustness rows → {OUT_FILE}")
    print(f"  Execution-grade (citable): {exec_grade}")
    print(f"  Price-grade estimates: {len(ROWS) - exec_grade}")


if __name__ == "__main__":
    main()
