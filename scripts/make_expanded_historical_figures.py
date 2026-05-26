"""Generate 4 expanded historical figures (25-28) for the paper.

Figure 25: Mechanism taxonomy bar chart — event count + max depeg by mechanism class
Figure 26: 18-event data coverage matrix (event × data source type)
Figure 27: Expanded event timeline (2020–2023) with tier and mechanism class
Figure 28: Tier B depeg severity panel — magnitude comparison across non-Tier-A events

Uses Columbia academic colour palette:
  Navy:   #003057
  Blue:   #75B2DD
  Gold:   #F2A900
  Grey:   #6B7280
  Red:    #CC2529

Output: results/paper_addon/figures/figure_{25,26,27,28}_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUT_DIR = ROOT / "results" / "paper_addon" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YAML_PATH = ROOT / "configs" / "event_windows_historical.yaml"

# ── Columbia palette ─────────────────────────────────────────────────────────
C_NAVY = "#003057"
C_BLUE = "#75B2DD"
C_GOLD = "#F2A900"
C_GREY = "#6B7280"
C_RED = "#CC2529"
C_GREEN = "#1B7A3E"

TIER_COLORS = {"A": C_NAVY, "B": C_BLUE, "C": C_GREY}

MECHANISM_LABELS = {
    "algorithmic_reflexive": "Algorithmic /\nReflexive",
    "fiat_reserve_bank_shock": "Fiat-Reserve\nBank Shock",
    "regulatory_issuer_winddown": "Regulatory /\nIssuer Winddown",
    "exchange_credit_liquidity": "Exchange Credit /\nLiquidity",
    "defi_pool_imbalance": "DeFi Pool\nImbalance",
    "collateral_liquidation_oracle": "Collateral /\nLiquidation",
    "rwa_niche_stablecoin": "RWA /\nNiche Stablecoin",
}

MECHANISM_ORDER = list(MECHANISM_LABELS.keys())
MECHANISM_COLORS = [C_RED, C_NAVY, C_GOLD, C_BLUE, C_GREEN, C_GREY, "#8B4513"]

# ── Load events ───────────────────────────────────────────────────────────────


def load_events() -> dict:
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]


# ── Figure 25: mechanism taxonomy bar chart ──────────────────────────────────


def fig25_mechanism_taxonomy(events: dict) -> None:
    from collections import defaultdict

    by_class: dict[str, list] = defaultdict(list)
    for ev_id, ev in events.items():
        mc = ev.get("mechanism_class", "unknown")
        by_class[mc].append(ev)

    counts = []
    max_depegs = []
    tiers_per_class = []  # list of tier distributions as (A, B, C)

    for mc in MECHANISM_ORDER:
        evs = by_class.get(mc, [])
        counts.append(len(evs))
        depegs = [abs(e.get("max_depeg_bps_est", 0)) for e in evs]
        max_depegs.append(max(depegs) if depegs else 0)
        tiers_per_class.append(
            (
                sum(1 for e in evs if e.get("data_tier") == "A"),
                sum(1 for e in evs if e.get("data_tier") == "B"),
                sum(1 for e in evs if e.get("data_tier") == "C"),
            )
        )

    labels = [MECHANISM_LABELS[mc] for mc in MECHANISM_ORDER]
    x = np.arange(len(MECHANISM_ORDER))
    bar_w = 0.55

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 3]}
    )
    fig.suptitle(
        "Stablecoin Stress Mechanism Taxonomy: Event Counts and Depeg Severity",
        fontsize=13,
        fontweight="bold",
        color=C_NAVY,
        y=1.02,
    )

    # Left: event count stacked by tier
    bottoms_a = np.zeros(len(x))
    bottoms_b = np.array([t[0] for t in tiers_per_class], dtype=float)
    bottoms_c = np.array([t[0] + t[1] for t in tiers_per_class], dtype=float)

    tier_a_vals = [t[0] for t in tiers_per_class]
    tier_b_vals = [t[1] for t in tiers_per_class]
    tier_c_vals = [t[2] for t in tiers_per_class]

    ax1.bar(x, tier_c_vals, bar_w, bottom=bottoms_c, color=C_GREY, label="Tier C", alpha=0.85)
    ax1.bar(x, tier_b_vals, bar_w, bottom=bottoms_b, color=C_BLUE, label="Tier B", alpha=0.85)
    ax1.bar(x, tier_a_vals, bar_w, bottom=bottoms_a, color=C_NAVY, label="Tier A", alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Number of Events", fontsize=10, color=C_NAVY)
    ax1.set_title("Events by Mechanism Class and Data Tier", fontsize=10, color=C_NAVY)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.legend(fontsize=8, framealpha=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(colors=C_NAVY)

    # Right: max depeg bps (log scale) per mechanism
    bar_colors = [MECHANISM_COLORS[i] for i in range(len(MECHANISM_ORDER))]
    bars = ax2.barh(
        labels, max_depegs, color=bar_colors, edgecolor="white", linewidth=0.5
    )
    # Add value labels
    for bar, val in zip(bars, max_depegs):
        if val > 0:
            ax2.text(
                val + 50, bar.get_y() + bar.get_height() / 2,
                f"{int(val)} bps" if val < 5000 else f"−{int(val/100):.0f}%",
                va="center", fontsize=7.5, color=C_NAVY,
            )

    ax2.set_xlabel("Peak Depeg Magnitude (bps, log scale) — est. for Tier B/C", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 30000)
    ax2.set_title("Max Depeg by Mechanism Class", fontsize=10, color=C_NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(colors=C_NAVY)

    # Footnote
    fig.text(
        0.01, -0.04,
        "Tier B/C max depeg values are estimates from event_windows_historical.yaml; "
        "use 'est.' notation in paper. Tier A = USDC/SVB only.",
        fontsize=7, color=C_GREY, style="italic",
    )

    plt.tight_layout()
    out = OUT_DIR / "figure_25_mechanism_taxonomy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 26: 18-event coverage matrix ──────────────────────────────────────


def fig26_coverage_matrix(events: dict) -> None:
    source_types = ["L2 Depth", "Trade Tape", "OHLCV", "On-chain", "DEX/Pool"]
    L2 = {"binance_real_l2_snapshot", "coinbase_real_l2_snapshot",
          "kraken_real_l2_snapshot", "binance_partial_orderbook",
          "huobi_partial_orderbook"}
    TAPE = {"binance_trade_tape", "coinbase_trade_tape", "kraken_trade_tape"}
    OHLCV = {"binance_ohlcv", "coinbase_ohlcv", "kraken_ohlcv",
             "coingecko_ohlcv", "huobi_ohlcv", "poloniex_ohlcv"}
    ONCHAIN = {"usdc_onchain_redemptions", "terra_onchain_transactions",
               "usdt_onchain_movements", "ethereum_onchain_liquidations",
               "makerdao_vault_data", "onchain_celsius_outflows",
               "acala_onchain_data", "tangible_protocol_onchain",
               "uniswap_v2_swaps", "polygon_dex_swaps", "abracadabra_onchain",
               "usdt_onchain_transfers", "paxos_attestation_reports"}
    DEX = {"curve_pool_reserves", "curve_3pool_reserves", "curve_mim_3pool_reserves",
           "curve_steth_pool_reserves", "subgraph_curve_swaps",
           "makerdao_psm_data", "binance_announcements"}

    type_sets = [L2, TAPE, OHLCV, ONCHAIN, DEX]

    ev_ids = list(events.keys())
    ev_labels = []
    for eid in ev_ids:
        ev = events[eid]
        tier = ev.get("data_tier", "C")
        # Short name
        name = eid.replace("_", " ").title()
        name = (name[:28] + "…") if len(name) > 28 else name
        ev_labels.append(f"[{tier}] {name}")

    matrix = np.zeros((len(ev_ids), len(source_types)), dtype=float)
    for i, eid in enumerate(ev_ids):
        sources = set(events[eid].get("data_sources", []))
        for j, stype in enumerate(type_sets):
            if sources & stype:
                # Partial vs full
                matrix[i, j] = 1.0 if events[eid].get("data_tier") == "A" else 0.5

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(source_types)))
    ax.set_xticklabels(source_types, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(ev_labels)))
    ax.set_yticklabels(ev_labels, fontsize=8)

    # Tier A rows get navy border
    for i, eid in enumerate(ev_ids):
        tier = events[eid].get("data_tier", "C")
        if tier == "A":
            for j in range(len(source_types)):
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (j - 0.5, i - 0.5), 1, 1,
                        boxstyle="square,pad=0",
                        edgecolor=C_NAVY, linewidth=1.5, fill=False,
                    )
                )

    ax.set_title(
        "Historical Event × Data Source Coverage Matrix\n(dark = Tier A, medium = Tier B, white = absent)",
        fontsize=11, fontweight="bold", color=C_NAVY,
    )
    ax.set_xlabel("Data Source Type", fontsize=10)
    ax.set_ylabel("Event (Tier)", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Absent", "Tier B (partial)", "Tier A (full)"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = OUT_DIR / "figure_26_coverage_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 27: expanded event timeline ───────────────────────────────────────


def fig27_event_timeline(events: dict) -> None:
    import datetime

    def parse_date(s: str) -> datetime.date:
        return datetime.date.fromisoformat(str(s)[:10])

    ev_data = []
    for eid, ev in events.items():
        try:
            start = parse_date(ev["start"])
            end = parse_date(ev["end"])
        except (KeyError, ValueError):
            continue
        mc = ev.get("mechanism_class", "unknown")
        tier = ev.get("data_tier", "C")
        depeg = abs(ev.get("max_depeg_bps_est", 0))
        ev_data.append({"id": eid, "start": start, "end": end,
                        "mc": mc, "tier": tier, "depeg": depeg})

    ev_data.sort(key=lambda x: x["start"])

    fig, ax = plt.subplots(figsize=(14, 6))

    ref_date = datetime.date(2020, 1, 1)

    def to_days(d: datetime.date) -> float:
        return (d - ref_date).days

    # Y positions: separate by mechanism class
    mc_y = {mc: i for i, mc in enumerate(MECHANISM_ORDER)}

    for ev in ev_data:
        y = mc_y.get(ev["mc"], 0)
        xs = to_days(ev["start"])
        xe = to_days(ev["end"])
        width = max(xe - xs, 3)  # min 3 days visible
        tier = ev["tier"]
        color = TIER_COLORS[tier]
        # Size by depeg magnitude
        height = 0.5 + min(ev["depeg"] / 3000, 0.45)
        rect = mpatches.FancyBboxPatch(
            (xs, y - height / 2), width, height,
            boxstyle="round,pad=1",
            facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.85,
        )
        ax.add_patch(rect)
        # Label short event ID
        short = ev["id"].replace("_202", " '2").replace("_", " ")[:18]
        ax.text(
            xs + width / 2, y, short,
            ha="center", va="center", fontsize=5.5, color="white", fontweight="bold",
        )

    # X-axis: date ticks
    quarter_starts = []
    current = datetime.date(2020, 1, 1)
    end_date = datetime.date(2024, 1, 1)
    while current < end_date:
        quarter_starts.append(current)
        m = current.month + 3
        y2 = current.year + (m - 1) // 12
        m2 = (m - 1) % 12 + 1
        current = datetime.date(y2, m2, 1)

    tick_days = [to_days(d) for d in quarter_starts]
    tick_labels = [d.strftime("%b '%y") if d.month == 1 else d.strftime("%b") for d in quarter_starts]

    ax.set_xlim(0, to_days(end_date))
    ax.set_ylim(-0.8, len(MECHANISM_ORDER) - 0.2)
    ax.set_xticks(tick_days[::2])
    ax.set_xticklabels(tick_labels[::2], fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(MECHANISM_ORDER)))
    ax.set_yticklabels(
        [MECHANISM_LABELS[mc].replace("\n", " ") for mc in MECHANISM_ORDER],
        fontsize=8,
    )

    # Legend
    handles = [
        mpatches.Patch(facecolor=TIER_COLORS["A"], label="Tier A (execution-grade)"),
        mpatches.Patch(facecolor=TIER_COLORS["B"], label="Tier B (price-grade)"),
        mpatches.Patch(facecolor=TIER_COLORS["C"], label="Tier C (context-grade)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(
        "Stablecoin Stress Event Timeline 2020–2023 by Mechanism Class and Data Tier\n"
        "(bar width = event duration; height proportional to peak depeg magnitude)",
        fontsize=11, fontweight="bold", color=C_NAVY,
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "figure_27_event_timeline_expanded.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 28: Tier B depeg severity panel ───────────────────────────────────


def fig28_tierb_depeg_panel(events: dict) -> None:
    # Collect Tier B events with nonzero depeg
    tier_b_events = [
        (eid, ev) for eid, ev in events.items()
        if ev.get("data_tier") == "B" and abs(ev.get("max_depeg_bps_est", 0)) > 0
    ]
    tier_b_events.sort(key=lambda x: abs(x[1].get("max_depeg_bps_est", 0)), reverse=True)

    labels = []
    depegs = []
    colors_list = []
    mc_list = []

    for eid, ev in tier_b_events:
        d = abs(ev.get("max_depeg_bps_est", 0))
        mc = ev.get("mechanism_class", "unknown")
        short = eid.replace("_202", " '2").replace("_", " ").title()[:28]
        labels.append(short)
        depegs.append(d)
        mc_idx = MECHANISM_ORDER.index(mc) if mc in MECHANISM_ORDER else 0
        colors_list.append(MECHANISM_COLORS[mc_idx])
        mc_list.append(mc)

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(labels))
    bars = ax.barh(y, depegs, color=colors_list, edgecolor="white", linewidth=0.5, alpha=0.88)

    # Add value labels
    for bar, val in zip(bars, depegs):
        ax.text(
            val + 20, bar.get_y() + bar.get_height() / 2,
            f"~{int(val)} bps",
            va="center", fontsize=8, color=C_NAVY,
        )

    # Reference line: USDC/SVB Tier A peak
    ax.axvline(1300, color=C_NAVY, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(1320, len(labels) - 0.5, "USDC/SVB\nTier A peak\n(−1300 bps)",
            fontsize=7.5, color=C_NAVY, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Peak Depeg Magnitude (bps) — estimated, Tier B events only", fontsize=9)
    ax.set_title(
        "Tier B Historical Events: Peak Depeg Severity (Estimated)\n"
        "vs. USDC/SVB Tier A Benchmark (−1300 bps verified)",
        fontsize=11, fontweight="bold", color=C_NAVY,
    )

    # Legend by mechanism
    seen = set()
    handles = []
    for mc, color in zip(mc_list, colors_list):
        if mc not in seen:
            seen.add(mc)
            handles.append(
                mpatches.Patch(facecolor=color, label=MECHANISM_LABELS.get(mc, mc).replace("\n", " "))
            )
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.01, -0.04,
        "All Tier B magnitudes are estimates (est.) from event_windows_historical.yaml. "
        "Exact values require on-chain or exchange data pull. Do not cite as exact bps in paper.",
        fontsize=7, color=C_GREY, style="italic",
    )

    plt.tight_layout()
    out = OUT_DIR / "figure_28_tierb_depeg_panel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    events = load_events()
    print(f"Loaded {len(events)} events. Generating figures 25–28 ...")

    fig25_mechanism_taxonomy(events)
    fig26_coverage_matrix(events)
    fig27_event_timeline(events)
    fig28_tierb_depeg_panel(events)

    print("Done. All figures written to results/paper_addon/figures/")


if __name__ == "__main__":
    main()
