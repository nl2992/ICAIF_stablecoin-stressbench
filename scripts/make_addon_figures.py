#!/usr/bin/env python3
"""Generate add-on paper figures.

Writes to results/paper_addon/figures/ ONLY. Baseline figures in
results/paper/figures/ are never modified.

Figures produced:
    Figure 8  — Price-to-execution gap by notional (robustness)
    Figure 9  — Price-to-execution gap by cost regime (robustness)
    Figure 11 — Signal waterfall: total → price → executable → model → oracle
    Figure 12 — False-positive feature profile (TP vs FP)

Usage:
    python scripts/make_addon_figures.py
    python scripts/make_addon_figures.py \
        --data-dir data/gold \
        --baseline-paper-dir results/paper \
        --addon-results-dir results/experiments_addon \
        --output-dir results/paper_addon/figures
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_COLORS = {
    "price":       "#2166ac",
    "exec_10k":    "#d73027",
    "exec_50k":    "#fc8d59",
    "exec_100k":   "#fdae61",
    "exec_500k":   "#fee090",
    "oracle":      "#4d4d4d",
    "no_trade":    "#bababa",
    "tp":          "#1a9641",
    "fp":          "#d73027",
    "fn":          "#fc8d59",
    "tn":          "#a6d96a",
}


def _savefig(fig, path: Path, fmt: str = "png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 8: Robustness by notional
# ---------------------------------------------------------------------------

def figure_8_robustness_notional(addon_dir: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    grid_path = addon_dir / "robustness_price_execution_gap.csv"
    if not grid_path.exists():
        logger.warning("robustness grid not found — skipping Figure 8.")
        return

    with open(grid_path) as fh:
        all_rows = list(csv.DictReader(fh))

    # Filter: base_fee, no settlement, threshold=10, horizon=5m
    rows = [
        r for r in all_rows
        if r["fee_regime"] == "base_fee"
        and r["settlement_penalty_bps"] == "0"
        and r["basis_threshold_bps"] == "10"
        and r["horizon"] == "5m"
    ]
    if not rows:
        logger.warning("No matching rows for Figure 8.")
        return

    notionals = sorted({int(r["notional"]) for r in rows})
    exec_pcts = []
    for q in notionals:
        subset = [r for r in rows if int(r["notional"]) == q]
        if subset:
            exec_pcts.append(float(subset[0]["executable_signal_pct"]))
        else:
            exec_pcts.append(float("nan"))

    price_pct = float(rows[0]["price_signal_pct"]) if rows else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(notionals))
    bars = ax.bar(x, exec_pcts, color=_COLORS["exec_10k"], alpha=0.85, label="Executable (net > 0 in 5m)")
    ax.axhline(price_pct, ls="--", lw=1.2, color=_COLORS["price"],
               label=f"Price signal |basis|>10bps ({price_pct:.1f}%)")

    for xi, pct in zip(x, exec_pcts):
        if pct == pct:
            ratio = price_pct / pct if pct > 0 else float("inf")
            label = f"{ratio:.0f}×" if ratio < 1000 else "∞"
            ax.text(xi, pct + 0.1, label, ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${q//1000}K" for q in notionals])
    ax.set_xlabel("Notional size")
    ax.set_ylabel("% of test-split minutes")
    ax.set_title("Figure 8 — Price-to-Execution Gap by Notional (>10 bps, 5m horizon)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_8_robustness_notional", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Robustness by cost regime
# ---------------------------------------------------------------------------

def figure_9_robustness_costs(addon_dir: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    grid_path = addon_dir / "robustness_price_execution_gap.csv"
    if not grid_path.exists():
        logger.warning("robustness grid not found — skipping Figure 9.")
        return

    with open(grid_path) as fh:
        all_rows = list(csv.DictReader(fh))

    # notional=10k, threshold=10, horizon=5m, vary settlement and fee regime
    rows = [
        r for r in all_rows
        if r["notional"] == "10000"
        and r["basis_threshold_bps"] == "10"
        and r["horizon"] == "5m"
    ]
    if not rows:
        logger.warning("No matching rows for Figure 9.")
        return

    fee_regimes = ["base_fee", "low_fee", "high_fee", "institutional_fee"]
    penalties = sorted({int(r["settlement_penalty_bps"]) for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4))
    colors_list = [_COLORS["exec_10k"], _COLORS["exec_50k"], _COLORS["exec_100k"], "#2166ac"]

    for i, regime in enumerate(fee_regimes):
        exec_pcts = []
        for pen in penalties:
            subset = [
                r for r in rows
                if r["fee_regime"] == regime and int(r["settlement_penalty_bps"]) == pen
            ]
            if subset:
                exec_pcts.append(float(subset[0]["executable_signal_pct"]))
            else:
                exec_pcts.append(float("nan"))
        ax.plot(penalties, exec_pcts, marker="o", lw=1.5,
                color=colors_list[i % len(colors_list)],
                label=regime.replace("_", " "))

    ax.set_xlabel("Settlement penalty (bps)")
    ax.set_ylabel("Executable % of test minutes")
    ax.set_title("Figure 9 — Price-to-Execution Gap by Cost Regime ($10K, >10 bps, 5m)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_9_robustness_costs", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: Signal waterfall
# ---------------------------------------------------------------------------

def figure_11_signal_waterfall(
    dataset_path: Path,
    baseline_results: Path,
    output_dir: Path,
    fmt: str,
) -> None:
    import polars as pl
    import matplotlib.pyplot as plt
    import numpy as np

    if not dataset_path.exists():
        logger.warning("dataset.parquet not found — skipping Figure 11.")
        return

    df = pl.read_parquet(str(dataset_path))
    test = df.filter(pl.col("split") == "test")
    n_total = len(test)

    basis_col = "cross_quote_basis_usdc_bps"
    net_col = "net_profit_bps_q10000"
    label_col = "label_arb_q10000_5m_gt0bps"

    n_price = int(test.filter(pl.col(basis_col).abs() > 10).height) if basis_col in test.columns else 0
    n_exec = int(test.filter(pl.col(net_col) > 0).height) if net_col in test.columns else 0

    # Best non-oracle model trades from baseline
    n_model_trades = 0
    n_model_hits = 0
    n_oracle = 0
    if baseline_results.exists():
        with open(baseline_results) as fh:
            rows = list(csv.DictReader(fh))
        # Best non-oracle model: highest net_bps_captured that is not NaN
        non_oracle = [
            r for r in rows
            if r["model"] != "oracle"
            and r.get("n_trades", "") not in ("", "0", "nan")
        ]
        if non_oracle:
            def _safe_float(x: str) -> float:
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return float("-inf")
            best = max(non_oracle, key=lambda r: _safe_float(r.get("net_bps_captured", "")))
            n_model_trades = int(best.get("n_trades", 0) or 0)
            hit_rate = _safe_float(best.get("hit_rate_above_cost", "0"))
            n_model_hits = int(n_model_trades * hit_rate) if hit_rate == hit_rate else 0

        oracle_rows = [r for r in rows if r["model"] == "oracle"]
        if oracle_rows:
            n_oracle = int(oracle_rows[0].get("n_trades", 0) or 0)

    labels = [
        f"Total minutes\n({n_total:,})",
        f"Price signal >10 bps\n({n_price:,}  {n_price/n_total*100:.1f}%)",
        f"Executable at $10K\n({n_exec:,}  {n_exec/n_total*100:.1f}%)",
        f"Best model trades\n({n_model_trades:,})",
        f"Best model hits\n({n_model_hits:,})",
        f"Oracle trades\n({n_oracle:,})",
    ]
    values = [n_total, n_price, n_exec, n_model_trades, n_model_hits, n_oracle]

    colors = [
        "#4393c3", _COLORS["price"], _COLORS["exec_10k"],
        "#762a83", "#9970ab", _COLORS["oracle"],
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.85)

    for xi, val in zip(x, values):
        ax.text(xi, val + n_total * 0.01, f"{val:,}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Window count (test split)")
    ax.set_title("Figure 11 — Signal Waterfall: Price → Executable → Model → Oracle")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_11_signal_waterfall", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: False-positive feature profile
# ---------------------------------------------------------------------------

def figure_12_false_positive_profile(paper_addon_dir: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fp_table = paper_addon_dir / "table_5_false_positive_diagnosis.csv"
    if not fp_table.exists():
        logger.warning("table_5 not found — run analyze_false_positives.py first. Skipping Figure 12.")
        return

    with open(fp_table) as fh:
        rows = list(csv.DictReader(fh))

    feature_labels = {
        "avg_cross_quote_basis_usdc_bps": "USDC basis (bps)",
        "avg_spread_bps_mean": "Bid-ask spread (bps)",
        "avg_depth_bid_10bp_mean": "Bid depth 10bp (BTC)",
        "avg_net_profit_bps_q10000": "Net profit $10K (bps)",
        "avg_imbalance_1bp_mean": "Order imbalance",
    }

    groups_order = ["TP", "FP"]
    group_rows = {r["group"]: r for r in rows if r["group"] in groups_order}

    feats = [f for f in feature_labels if all(f in r for r in group_rows.values())]
    if not feats:
        logger.warning("No matching feature columns in table_5 for Figure 12.")
        return

    x = np.arange(len(feats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, group in enumerate(groups_order):
        if group not in group_rows:
            continue
        vals = []
        for f in feats:
            try:
                v = float(group_rows[group].get(f, "nan") or "nan")
            except ValueError:
                v = float("nan")
            vals.append(v)
        color = _COLORS["tp"] if group == "TP" else _COLORS["fp"]
        ax.bar(x + (i - 0.5) * width, vals, width, label=group, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([feature_labels[f] for f in feats], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Average value (test split)")
    ax.set_title("Figure 12 — Feature Profile: True Positives vs False Positives")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_12_false_positive_profile", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate add-on paper figures.")
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--baseline-paper-dir", default="results/paper")
    p.add_argument("--addon-results-dir", default="results/experiments_addon")
    p.add_argument("--output-dir", default="results/paper_addon/figures")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    addon_dir = Path(args.addon_results_dir)
    paper_addon_dir = output_dir.parent  # results/paper_addon
    dataset_path = Path(args.data_dir) / "dataset.parquet"
    fmt = args.format

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.error("matplotlib not installed — pip install matplotlib")
        return

    logger.info("Generating add-on figures → %s", output_dir)
    figure_8_robustness_notional(addon_dir, output_dir, fmt)
    figure_9_robustness_costs(addon_dir, output_dir, fmt)
    figure_11_signal_waterfall(dataset_path, Path(args.addon_results_dir).parent / "experiments" / "all_results.csv", output_dir, fmt)
    figure_12_false_positive_profile(paper_addon_dir, output_dir, fmt)
    logger.info("Add-on figures complete.")


if __name__ == "__main__":
    main()
