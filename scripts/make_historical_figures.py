#!/usr/bin/env python3
"""Generate historical event figures.

Columbia academic theme (navy #003057, blue #75B2DD, gold #F2A900).

Figures produced (written to results/paper_addon/figures/):
    figure_23_historical_depeg_timeline.png
        Horizontal timeline of all events with tier colors,
        labeled with stablecoin names, event names, and peak depeg magnitudes.

    figure_24_event_coverage_matrix.png
        Heatmap with events as rows, data sources as columns,
        colored by availability.

Usage:
    python scripts/make_historical_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stressbench.common.logging import get_logger
from stressbench.history.data_availability import PREDEFINED_COVERAGE
from stressbench.history.event_catalog import EVENT_CATALOG, load_event_windows_yaml

logger = get_logger(__name__)

# Columbia University palette
_NAVY = "#003057"
_BLUE = "#75B2DD"
_GOLD = "#F2A900"
_GRAY = "#AAAAAA"
_WHITE = "#FFFFFF"
_LIGHT_BLUE = "#B8D9F0"

# Tier colors: A=gold, B=blue, C=gray
_TIER_COLORS = {
    "A": _GOLD,
    "B": _BLUE,
    "C": _GRAY,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make historical event figures")
    p.add_argument("--config", default="configs/event_windows_historical.yaml")
    p.add_argument("--output-dir", default="results/paper_addon/figures")
    return p.parse_args()


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=_WHITE)
    logger.info("Saved figure: %s", path)
    print(f"Saved: {path}")


def figure_23_timeline(events: dict, out_dir: Path) -> None:
    """Figure 23: Horizontal timeline of historical depeg events with tier colors."""
    import matplotlib

    matplotlib.use("Agg")
    from datetime import datetime

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    # Sort events by start date
    def _start_dt(ev: dict) -> datetime:
        return datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))

    sorted_events = sorted(events.items(), key=lambda kv: _start_dt(kv[1]))

    # Axis: time range
    all_starts = [_start_dt(ev) for _, ev in sorted_events]
    all_ends = [
        datetime.fromisoformat(ev["end"].replace("Z", "+00:00"))
        for _, ev in sorted_events
    ]

    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter, date2num

    t_min = min(all_starts)
    t_max = max(all_ends)

    y_positions = list(range(len(sorted_events)))
    bar_height = 0.5

    for i, (event_id, ev) in enumerate(sorted_events):
        tier_val = ev["data_tier"]
        tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
        color = _TIER_COLORS.get(tier_str, _GRAY)

        start_num = date2num(_start_dt(ev))
        end_num = date2num(datetime.fromisoformat(ev["end"].replace("Z", "+00:00")))
        width = end_num - start_num

        ax.barh(
            i,
            width,
            left=start_num,
            height=bar_height,
            color=color,
            edgecolor=_NAVY,
            linewidth=0.8,
            alpha=0.85,
        )

        # Event label (left-aligned inside or outside bar)
        display_name = ev.get("display_name", event_id)
        stablecoins = ", ".join(ev.get("stablecoins", []))
        peak_bps = ev.get("peak_depeg_bps", None)
        if peak_bps is not None:
            label_text = f"{display_name}\n{stablecoins}\n{peak_bps:+d} bps"
        else:
            label_text = f"{display_name}\n{stablecoins}"

        mid_num = start_num + width / 2
        ax.text(
            mid_num,
            i,
            label_text,
            ha="center",
            va="center",
            fontsize=6.5,
            color=_NAVY,
            fontweight="bold",
            wrap=True,
        )

    # Tier legend
    legend_patches = [
        mpatches.Patch(color=_GOLD, label="Tier A — Execution-grade (full L2)"),
        mpatches.Patch(color=_BLUE, label="Tier B — Price-grade (OHLCV/trades)"),
        mpatches.Patch(color=_GRAY, label="Tier C — Context-grade (partial data)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor=_NAVY,
    )

    # Format x-axis as dates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [ev.get("display_name", eid)[:25] for eid, ev in sorted_events],
        fontsize=8,
        color=_NAVY,
    )
    ax.invert_yaxis()

    ax.set_xlabel("Date", fontsize=10, color=_NAVY)
    ax.set_title(
        "Figure 23: Historical Stablecoin Stress Events — Timeline and Data Tier",
        fontsize=11,
        color=_NAVY,
        fontweight="bold",
        pad=12,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(_NAVY)
    ax.tick_params(colors=_NAVY)

    plt.tight_layout()
    _savefig(fig, out_dir / "figure_23_historical_depeg_timeline.png")
    plt.close(fig)


def figure_24_coverage_matrix(events: dict, out_dir: Path) -> None:
    """Figure 24: Event x data source coverage heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Data source columns
    source_labels = [
        "Price\nData",
        "Trade\nTape",
        "L2 Order\nBook",
        "DEX Pool\nData",
        "On-chain\nData",
        "Exec-grade\nAvailable",
    ]
    source_attrs = [
        "price_data",
        "trade_data",
        "l2_data",
        "dex_pool_data",
        "onchain_data",
        "execution_grade_available",
    ]

    # Sort events by start date for consistency with Figure 23
    from datetime import datetime

    def _start_dt(ev: dict) -> datetime:
        return datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))

    sorted_event_ids = sorted(events.keys(), key=lambda eid: _start_dt(events[eid]))
    event_labels = [events[eid].get("display_name", eid) for eid in sorted_event_ids]

    # Build matrix
    n_events = len(sorted_event_ids)
    n_sources = len(source_attrs)
    matrix = np.zeros((n_events, n_sources), dtype=np.float32)

    for i, event_id in enumerate(sorted_event_ids):
        profile = PREDEFINED_COVERAGE.get(event_id)
        if profile is None:
            continue
        for j, attr in enumerate(source_attrs):
            matrix[i, j] = float(getattr(profile, attr, False))

    # Custom colormap: navy for available, light gray for not available
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([_GRAY + "55", _NAVY])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    # Add tier color band on the left as a colored column
    # Append tier column
    tier_vals = []
    tier_map = {"A": 1.0, "B": 0.5, "C": 0.0}
    for event_id in sorted_event_ids:
        tier_val = events[event_id].get("data_tier", "C")
        tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
        tier_vals.append(tier_map.get(tier_str, 0.0))

    # Cell annotations: checkmark / X
    for i in range(n_events):
        for j in range(n_sources):
            val = matrix[i, j]
            text = "✓" if val > 0 else "✗"
            color = _WHITE if val > 0 else _NAVY
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                fontweight="bold",
            )

    # Add coverage score annotation on the right
    ax.set_xlim(-0.5, n_sources - 0.5)
    for i, event_id in enumerate(sorted_event_ids):
        profile = PREDEFINED_COVERAGE.get(event_id)
        score = profile.coverage_score if profile else 0.25
        ax.text(
            n_sources - 0.5 + 0.6,
            i,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color=_NAVY,
            fontweight="bold",
        )

    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_labels, fontsize=8, color=_NAVY)
    ax.set_yticks(range(n_events))
    ax.set_yticklabels(event_labels, fontsize=8, color=_NAVY)

    ax.set_title(
        "Figure 24: Event Data Source Coverage Matrix",
        fontsize=11,
        color=_NAVY,
        fontweight="bold",
        pad=12,
    )

    # Add "Coverage Score" label on right side
    ax.text(
        n_sources - 0.5 + 0.6,
        -0.7,
        "Coverage\nScore",
        ha="left",
        va="center",
        fontsize=7,
        color=_NAVY,
        fontstyle="italic",
    )

    # Add tier color indicators on the far right via text
    tier_colors_text = {"A": _GOLD, "B": _BLUE, "C": _GRAY}
    for i, event_id in enumerate(sorted_event_ids):
        tier_val = events[event_id].get("data_tier", "C")
        tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
        tc = tier_colors_text.get(tier_str, _GRAY)
        ax.text(
            n_sources - 0.5 + 1.5,
            i,
            f"[{tier_str}]",
            ha="left",
            va="center",
            fontsize=8,
            color=tc,
            fontweight="bold",
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(_NAVY)
    ax.tick_params(colors=_NAVY)

    plt.tight_layout()
    _savefig(fig, out_dir / "figure_24_event_coverage_matrix.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading events from %s", config_path)
    events = load_event_windows_yaml(config_path)
    logger.info("Loaded %d events", len(events))

    figure_23_timeline(events, out_dir)
    figure_24_coverage_matrix(events, out_dir)


if __name__ == "__main__":
    main()
