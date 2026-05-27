#!/usr/bin/env python3
"""Extended paper figures — Columbia University academic theme.

Generates 9 additional figures for the ICAIF 2026 paper. Writes ONLY to
results/paper_addon/figures/. Baseline figures are never modified.

Figures produced:
    figure_14_event_timeline        — Study design: event windows on calendar
    figure_15_model_comparison      — All models ranked by test net bps
    figure_16_cumulative_pnl        — Cumulative P&L: oracle vs ML vs no-trade
    figure_17_roc_curves            — ROC curves across model families
    figure_18_feature_importance    — LGBM feature importance (primary task)
    figure_19_basis_heatmap         — USDC basis intensity by hour-of-day
    figure_20_cost_decomposition    — Gross basis → net profit cost waterfall
    figure_21_horizon_ratio         — Price-to-execution ratio by horizon
    figure_22_calibration           — Reliability diagram: predicted vs actual

Notes:
    Figures 16–18, 22 re-fit lightweight models (logistic / LGBM) for
    visualisation only. These fits do not overwrite official experiment results.

Usage:
    python scripts/make_paper_figures_extended.py
    python scripts/make_paper_figures_extended.py \
        --data-dir data/gold \
        --results-dir results/experiments \
        --addon-results-dir results/experiments_addon \
        --output-dir results/paper_addon/figures
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Columbia University academic colour palette
# -----------------------------------------------------------------------

_C = {
    "navy": "#003057",  # Columbia Navy (primary)
    "blue": "#75B2DD",  # Columbia Blue (accent)
    "gold": "#F2A900",  # Columbia Gold (highlight / oracle)
    "light": "#C4D8E2",  # Columbia Light Blue (backgrounds)
    "gray": "#878787",  # Mid gray
    "lgray": "#D9D9D9",  # Light gray (grid / minor)
    "red": "#C4302B",  # Alert / negative
    "green": "#2C7340",  # Positive (profitable)
    "orange": "#E86100",  # Warning
    "white": "#FFFFFF",
}

_SPLIT_COLORS = {
    "train": _C["navy"],
    "validation": _C["orange"],
    "test": _C["red"],
}

_MODEL_FAMILY_COLORS = {
    "oracle": _C["gold"],
    "price": _C["blue"],
    "ml": _C["navy"],
    "addon": _C["orange"],
    "no_trade": _C["gray"],
}


def _columbia_style():
    """Apply Columbia academic rcParams to matplotlib."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.framealpha": 0.92,
            "legend.edgecolor": _C["lgray"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": _C["gray"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": _C["gray"],
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "figure.facecolor": _C["white"],
            "axes.facecolor": "#FAFBFC",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.8,
        }
    )


def _savefig(fig, path: Path, fmt: str = "png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    logger.info("Saved %s", out)


# -----------------------------------------------------------------------
# Shared model-fit utilities (visualisation-only; do not overwrite results)
# -----------------------------------------------------------------------


def _prep_splits(dataset_path: Path):
    """Load dataset and return (df, X_tr, y_tr, X_val, y_val, X_te, y_te, net_te)."""
    import polars as pl

    df = pl.read_parquet(str(dataset_path))

    _FEAT_COLS = [
        c
        for c in [
            "cross_quote_basis_usdc_bps",
            "cross_quote_basis_usdt_bps",
            "cross_quote_basis_maxabs_bps",
            "cross_quote_basis_primary_bps",
            "spread_bps_mean",
            "depth_bid_10bp_mean",
            "depth_ask_10bp_mean",
            "imbalance_1bp_mean",
            "num_active_venues_mean",
            "mid_dispersion_bps_mean",
        ]
        if c in df.columns
    ]
    _LABEL = "label_basis_usdc_1m_gt10bps"
    _NET = "net_profit_bps_q10000"
    _TS = "ts_1m_ns"

    def _split(s):
        return df.filter(pl.col("split") == s).sort(_TS)

    tr = _split("train")
    val = _split("validation")
    te = _split("test")

    def _to_X(sdf):
        return sdf.select(_FEAT_COLS).to_numpy().astype(float)

    def _to_y(sdf):
        if _LABEL not in sdf.columns:
            return np.zeros(len(sdf))
        raw = sdf[_LABEL].to_numpy().astype(float)
        return np.nan_to_num(raw, nan=0.0)

    X_tr_raw = _to_X(tr)
    # Compute column medians from training data; fall back to 0.0 for all-NaN columns
    train_medians = np.where(
        np.all(np.isnan(X_tr_raw), axis=0),
        0.0,
        np.nanmedian(X_tr_raw, axis=0),
    )

    def _impute(Xraw):
        mask = np.isnan(Xraw)
        if mask.any():
            col_idx = np.where(mask)[1]
            Xraw[mask] = train_medians[col_idx]
        # Final safety net for any remaining NaN (e.g., all-NaN columns)
        return np.nan_to_num(Xraw, nan=0.0)

    X_tr, y_tr = _impute(X_tr_raw), _to_y(tr)
    X_val, y_val = _impute(_to_X(val)), _to_y(val)
    X_te, y_te = _impute(_to_X(te)), _to_y(te)

    net_te = (
        te[_NET].to_numpy().astype(float) if _NET in te.columns else np.zeros(len(te))
    )
    ts_te = te[_TS].to_numpy().astype(np.int64)

    return df, _FEAT_COLS, X_tr, y_tr, X_val, y_val, X_te, y_te, net_te, ts_te


def _fit_logistic(X_tr, y_tr):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(
        C=1.0, max_iter=500, class_weight="balanced", random_state=42
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_lgbm(X_tr, y_tr):
    import lightgbm as lgb

    pos = max(y_tr.sum(), 1)
    scale = (len(y_tr) - pos) / pos
    m = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    return m


def _calibrate_clf(clf, X_val, y_val, net_val, min_trades=25):
    """Return (threshold, mean_net_bps) via total-P&L calibration."""
    proba = clf.predict_proba(X_val)[:, 1]
    net_clean = np.nan_to_num(net_val, nan=-999.0)
    best_t, best_total = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 19):
        sig = proba > t
        if sig.sum() < min_trades:
            continue
        total = float(np.sum(net_clean[sig]))
        if total > best_total:
            best_total = total
            best_t = float(t)
    return best_t


# -----------------------------------------------------------------------
# Figure 14: Study design / event timeline
# -----------------------------------------------------------------------


def figure_14_event_timeline(output_dir: Path, fmt: str) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd

    _columbia_style()

    events = {
        "train": [
            ("Jan 10–16, 2022", "2022-01-10", "2022-01-17"),
            ("Feb 1–7, 2023", "2023-02-01", "2023-02-08"),
            ("Q1 2024 control", "2024-01-15", "2024-01-22"),
        ],
        "validation": [("Terra/LUNA collapse", "2022-05-07", "2022-05-15")],
        "test": [
            ("USDC/SVB depeg", "2023-03-10", "2023-03-15"),
            ("USDC recovery", "2023-03-15", "2023-03-21"),
        ],
    }

    split_labels = {
        "train": "Train (calm control)",
        "validation": "Validation (Terra/LUNA)",
        "test": "Test (USDC/SVB)",
    }
    split_y = {"train": 2, "validation": 1, "test": 0}
    split_col = {"train": _C["navy"], "validation": _C["orange"], "test": _C["red"]}

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(pd.Timestamp("2021-12-01"), pd.Timestamp("2024-04-01"))
    ax.set_ylim(-0.5, 2.8)

    for split, windows in events.items():
        y = split_y[split]
        col = split_col[split]
        for label, start, end in windows:
            ts0 = pd.Timestamp(start)
            ts1 = pd.Timestamp(end)
            ax.barh(
                y,
                (ts1 - ts0).days,
                left=ts0,
                height=0.55,
                color=col,
                alpha=0.80,
                edgecolor="white",
                linewidth=0.5,
            )
            cx = ts0 + (ts1 - ts0) / 2
            ax.text(
                cx,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )

    # Vertical stress-event markers
    for date, note in [("2022-05-09", "UST\nde-peg"), ("2023-03-10", "SVB\ncollapse")]:
        ts = pd.Timestamp(date)
        ax.axvline(ts, color=_C["red"], lw=1.2, ls="--", alpha=0.6)
        ax.text(
            ts,
            2.65,
            note,
            ha="center",
            va="top",
            fontsize=7.5,
            color=_C["red"],
            style="italic",
        )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(
        [split_labels["test"], split_labels["validation"], split_labels["train"]],
        fontsize=9,
    )
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    ax.set_xlabel("Calendar date")
    ax.set_title("Figure 14 — Study Design: Event Windows and Data Splits")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_14_event_timeline", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 15: All-model comparison bar chart
# -----------------------------------------------------------------------


def figure_15_model_comparison(results_dir: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    _columbia_style()
    all_csv = results_dir / "all_results.csv"
    if not all_csv.exists():
        logger.warning("all_results.csv not found — skipping Figure 15.")
        return

    with open(all_csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["task"] == "basis_usdc_1m_gt10bps"]

    if not rows:
        logger.warning("No basis_usdc_1m_gt10bps rows — skipping Figure 15.")
        return

    def _net(r):
        try:
            return float(r.get("net_bps_captured") or "nan")
        except ValueError:
            return float("nan")

    rows = [r for r in rows if _net(r) == _net(r)]
    rows.sort(key=_net)

    labels = [f"{r['model']}\n({r['feature_set']})" for r in rows]
    vals = [_net(r) for r in rows]

    colors = []
    for r in rows:
        if r["model"] == "oracle":
            colors.append(_C["gold"])
        elif r["model"] == "no_trade":
            colors.append(_C["gray"])
        elif r["model"] in ("logistic", "lasso", "ridge", "rf", "lgbm", "xgb"):
            colors.append(_C["navy"] if _net(r) > -50 else _C["blue"])
        else:
            colors.append(_C["orange"])

    fig, ax = plt.subplots(figsize=(7, max(4, len(rows) * 0.38)))
    bars = ax.barh(
        range(len(rows)),
        vals,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color="black", lw=0.9, zorder=5)

    for i, (v, bar) in enumerate(zip(vals, bars)):
        xoff = 2 if v >= 0 else -2
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + xoff, i, f"{v:+.0f}", va="center", ha=ha, fontsize=7.5, color=_C["navy"]
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Net bps captured (test split)")
    ax.set_title(
        "Figure 15 — Model Performance: Net bps Captured (basis_usdc_1m_gt10bps)"
    )
    handles = [
        mpatches.Patch(color=_C["gold"], label="Oracle (hindsight ceiling)"),
        mpatches.Patch(color=_C["navy"], label="ML model"),
        mpatches.Patch(color=_C["orange"], label="Rule baseline"),
        mpatches.Patch(color=_C["gray"], label="No-trade (floor)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_15_model_comparison", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 16: Cumulative P&L over test split
# -----------------------------------------------------------------------


def figure_16_cumulative_pnl(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    _columbia_style()
    df, feat_cols, X_tr, y_tr, X_val, y_val, X_te, y_te, net_te, ts_te = _prep_splits(
        dataset_path
    )

    # Net profit for validation (needed for threshold calibration)
    import polars as pl

    val_df = df.filter(pl.col("split") == "validation").sort("ts_1m_ns")
    net_val = (
        val_df["net_profit_bps_q10000"].to_numpy().astype(float)
        if "net_profit_bps_q10000" in val_df.columns
        else np.zeros(len(val_df))
    )

    # Fit logistic
    clf = _fit_logistic(X_tr, y_tr)
    thr = _calibrate_clf(clf, X_val, y_val, net_val)
    proba_te = clf.predict_proba(X_te)[:, 1]
    signal_te = (proba_te > thr).astype(np.int8)

    # Compute P&L series
    net_clean = np.nan_to_num(net_te, nan=0.0)
    dt = pd.to_datetime(ts_te, unit="ns", utc=True)

    oracle_pnl = np.cumsum(np.where(net_clean > 0, net_clean, 0.0))
    model_pnl = np.cumsum(signal_te * net_clean)
    notrade = np.zeros(len(net_clean))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(
        dt,
        oracle_pnl,
        color=_C["gold"],
        lw=2.0,
        label=f"Oracle upper bound (+{oracle_pnl[-1]:.0f} bps total)",
        zorder=4,
    )
    ax.plot(
        dt,
        model_pnl,
        color=_C["red"],
        lw=1.5,
        label=f"Logistic@price+book  ({model_pnl[-1]:+.0f} bps total)",
        zorder=3,
    )
    ax.plot(
        dt,
        notrade,
        color=_C["gray"],
        lw=1.2,
        ls="--",
        label="No-trade baseline (0 bps)",
        zorder=2,
    )
    ax.fill_between(dt, model_pnl, 0, where=model_pnl < 0, alpha=0.12, color=_C["red"])
    ax.axhline(0, color=_C["gray"], lw=0.7, zorder=1)

    # Mark SVB peak
    ax.axvspan(
        pd.Timestamp("2023-03-11", tz="UTC"),
        pd.Timestamp("2023-03-13", tz="UTC"),
        alpha=0.07,
        color=_C["red"],
        label="Peak USDC de-peg (Mar 11–13)",
    )

    ax.set_xlabel("UTC date (SVB test window)")
    ax.set_ylabel("Cumulative net P&L (bps)")
    ax.set_title(
        "Figure 16 — Cumulative P&L: Oracle vs Logistic Classifier vs No-Trade"
    )
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_16_cumulative_pnl", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 17: ROC curves
# -----------------------------------------------------------------------


def figure_17_roc_curves(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import auc, roc_curve

    _columbia_style()
    _, feat_cols, X_tr, y_tr, _, _, X_te, y_te, _, _ = _prep_splits(dataset_path)

    models_to_plot = [
        ("Logistic (L2)", _fit_logistic(X_tr, y_tr), _C["navy"]),
        ("LightGBM", _fit_lgbm(X_tr, y_tr), _C["blue"]),
        (
            "Random (chance)",
            DummyClassifier(strategy="uniform", random_state=0).fit(X_tr, y_tr),
            _C["lgray"],
        ),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        [0, 1], [0, 1], ls="--", color=_C["lgray"], lw=1.2, label="Random (AUC = 0.50)"
    )

    for name, clf, col in models_to_plot:
        if name == "Random (chance)":
            continue  # handled above
        proba = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name}  (AUC = {auroc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 17 — ROC Curves: basis_usdc_1m_gt10bps (test split)")
    ax.legend(fontsize=9, loc="lower right")
    ax.text(
        0.97,
        0.03,
        "Note: AUC > 0.5 ≠ economic profit\n(classification skill ≠ execution success)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color=_C["gray"],
    )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_17_roc_curves", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 18: Feature importance
# -----------------------------------------------------------------------


def figure_18_feature_importance(
    dataset_path: Path, output_dir: Path, fmt: str
) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _columbia_style()
    _, feat_cols, X_tr, y_tr, _, _, _, _, _, _ = _prep_splits(dataset_path)

    clf = _fit_lgbm(X_tr, y_tr)

    importances = clf.feature_importances_
    idx = np.argsort(importances)

    _PRICE = {
        "cross_quote_basis_usdc_bps",
        "cross_quote_basis_usdt_bps",
        "cross_quote_basis_maxabs_bps",
        "cross_quote_basis_primary_bps",
        "deviation_from_1_usd_bps",
    }
    _BOOK = {
        "spread_bps_mean",
        "depth_bid_10bp_mean",
        "depth_ask_10bp_mean",
        "imbalance_1bp_mean",
        "data_quality_score_min",
        "trade_count_1m_total",
        "trade_volume_1m_total",
    }

    def _col_color(c):
        if c in _PRICE:
            return _C["red"]
        if c in _BOOK:
            return _C["navy"]
        return _C["orange"]

    labels = [feat_cols[i] for i in idx]
    vals = [importances[i] for i in idx]
    colors = [_col_color(l) for l in labels]

    # Clean up label names
    _RENAME = {
        "cross_quote_basis_usdc_bps": "USDC cross-quote basis",
        "cross_quote_basis_usdt_bps": "USDT cross-quote basis",
        "cross_quote_basis_maxabs_bps": "Max-abs basis",
        "cross_quote_basis_primary_bps": "Primary basis",
        "deviation_from_1_usd_bps": "Stablecoin deviation (bps)",
        "spread_bps_mean": "Bid-ask spread",
        "depth_bid_10bp_mean": "Bid depth 10 bps",
        "depth_ask_10bp_mean": "Ask depth 10 bps",
        "imbalance_1bp_mean": "Order imbalance 1 bp",
        "num_active_venues_mean": "Active venue count",
        "mid_dispersion_bps_mean": "Mid-price dispersion",
        "data_quality_score_min": "Data quality score",
        "trade_count_1m_total": "Trade count (1 min)",
        "trade_volume_1m_total": "Trade volume (1 min)",
        "max_minus_min_bps_mean": "Max − min cross-venue",
    }
    labels_clean = [_RENAME.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(feat_cols) * 0.42)))
    ax.barh(
        range(len(vals)),
        vals,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_yticks(range(len(labels_clean)))
    ax.set_yticklabels(labels_clean, fontsize=8.5)
    ax.set_xlabel("LightGBM feature importance (split count)")
    ax.set_title("Figure 18 — Feature Importance: LightGBM on basis_usdc_1m_gt10bps")
    patches = [
        mpatches.Patch(color=_C["red"], label="Price / basis signal"),
        mpatches.Patch(color=_C["navy"], label="Microstructure (book)"),
        mpatches.Patch(color=_C["orange"], label="Fragmentation"),
    ]
    ax.legend(handles=patches, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_18_feature_importance", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 19: USDC basis intensity by hour-of-day
# -----------------------------------------------------------------------


def figure_19_basis_heatmap(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl

    _columbia_style()
    df = pl.read_parquet(str(dataset_path))
    test = df.filter(pl.col("split") == "test")

    if "cross_quote_basis_usdc_bps" not in test.columns:
        logger.warning("No basis column — skipping Figure 19.")
        return

    ts = pd.to_datetime(test["ts_1m_ns"].to_list(), unit="ns", utc=True)
    basis = np.abs(
        np.nan_to_num(
            test["cross_quote_basis_usdc_bps"].to_numpy().astype(float), nan=0.0
        )
    )
    dates = [t.date() for t in ts]
    hours = [t.hour for t in ts]

    import pandas as pd

    frame = pd.DataFrame({"date": dates, "hour": hours, "basis": basis})
    pivot = frame.pivot_table(
        values="basis", index="date", columns="hour", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(12, 3.2))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "columbia", [_C["light"], _C["blue"], _C["navy"], _C["red"]], N=256
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=50)
    plt.colorbar(im, ax=ax, label="|USDC basis| (bps)", fraction=0.04, pad=0.02)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(d) for d in pivot.index], fontsize=8)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Date")
    ax.set_title(
        "Figure 19 — USDC Cross-Quote Basis Intensity by Hour-of-Day (SVB Test Split)"
    )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_19_basis_heatmap", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 20: Cost decomposition waterfall
# -----------------------------------------------------------------------


def figure_20_cost_decomposition(
    dataset_path: Path, output_dir: Path, fmt: str
) -> None:
    import matplotlib.pyplot as plt
    import polars as pl

    _columbia_style()
    df = pl.read_parquet(str(dataset_path))
    test = df.filter(pl.col("split") == "test")

    b_col = "cross_quote_basis_usdc_bps"
    n_col = "net_profit_bps_q10000"
    if b_col not in test.columns or n_col not in test.columns:
        logger.warning("Missing columns for Figure 20.")
        return

    # Only rows where both basis is positive and net profit is known
    basis = test[b_col].to_numpy().astype(float)
    net = test[n_col].to_numpy().astype(float)
    mask = (~np.isnan(basis)) & (~np.isnan(net)) & (basis > 0)

    gross = float(np.mean(basis[mask]))
    net_m = float(np.mean(net[mask]))
    total_cost = gross - net_m  # average total execution cost
    fee_est = 5.0  # approximate taker fee per trade (bps)
    slippage = max(0, total_cost - fee_est)

    stages = [
        "Gross basis\n(price signal)",
        "− VWAP slippage\n(depth walk)",
        "− Taker fees\n(buy + sell)",
        "Net profit\n(executable)",
    ]
    values = [gross, -slippage, -fee_est, net_m]
    running = [gross, gross - slippage, gross - slippage - fee_est, net_m]

    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = [0, gross - slippage, gross - slippage - fee_est, 0]
    bar_colors = [
        _C["navy"],
        _C["red"],
        _C["orange"],
        (_C["green"] if net_m > 0 else _C["red"]),
    ]
    bar_heights = [gross, slippage, fee_est, abs(net_m)]

    for i, (bot, h, col, lbl) in enumerate(
        zip(bottoms, bar_heights, bar_colors, stages)
    ):
        ax.bar(
            i,
            h,
            bottom=bot if i < 3 else (net_m if net_m < 0 else 0),
            color=col,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            width=0.55,
        )
        sign = "+" if values[i] >= 0 else ""
        ax.text(
            i,
            running[i] + (0.3 if values[i] >= 0 else -0.6),
            f"{sign}{values[i]:.1f} bps",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=_C["navy"],
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Average bps (test split, basis > 0 windows)")
    ax.set_title("Figure 20 — Cost Decomposition: Gross Basis → Net Executable Profit")
    ax.text(
        0.97,
        0.97,
        f"n = {int(mask.sum()):,} windows with positive basis",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
        color=_C["gray"],
    )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_20_cost_decomposition", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 21: Price-to-execution ratio by prediction horizon
# -----------------------------------------------------------------------


def figure_21_horizon_ratio(addon_dir: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _columbia_style()
    grid_path = addon_dir / "robustness_price_execution_gap.csv"
    if not grid_path.exists():
        logger.warning("robustness grid not found — skipping Figure 21.")
        return

    with open(grid_path) as fh:
        all_rows = list(csv.DictReader(fh))

    # base_fee, settlement=0, threshold=10bps, notional=10k
    rows = [
        r
        for r in all_rows
        if r["fee_regime"] == "base_fee"
        and r["settlement_penalty_bps"] == "0"
        and r["basis_threshold_bps"] == "10"
        and r["notional"] == "10000"
    ]

    horizons = ["1m", "5m", "15m"]
    price_pct = float(rows[0]["price_signal_pct"]) if rows else 0.0
    exec_pcts = []
    for h in horizons:
        subset = [r for r in rows if r["horizon"] == h]
        exec_pcts.append(
            float(subset[0]["executable_signal_pct"]) if subset else float("nan")
        )

    x = np.arange(len(horizons))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        x - width / 2,
        [price_pct] * 3,
        width,
        color=_C["blue"],
        alpha=0.85,
        label=f"Price signal |basis|>10 bps ({price_pct:.1f}%)",
    )
    ax.bar(
        x + width / 2,
        exec_pcts,
        width,
        color=_C["red"],
        alpha=0.85,
        label="Executable (net profit > 0)",
    )

    for xi, ep in zip(x, exec_pcts):
        ratio = price_pct / ep if ep > 0 else float("inf")
        ax.text(
            xi + width / 2,
            ep + 0.3,
            f"{ratio:.1f}×",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=_C["navy"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["1-minute\nhorizon", "5-minute\nhorizon", "15-minute\nhorizon"])
    ax.set_ylabel("% of test-split minutes")
    ax.set_title(
        "Figure 21 — Price-to-Execution Ratio by Prediction Horizon ($10K, base fee)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_21_horizon_ratio", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 22: Calibration / reliability diagram
# -----------------------------------------------------------------------


def figure_22_calibration(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    _columbia_style()
    _, _, X_tr, y_tr, _, _, X_te, y_te, _, _ = _prep_splits(dataset_path)

    models = [
        ("Logistic (L2)", _fit_logistic(X_tr, y_tr), _C["navy"]),
        ("LightGBM", _fit_lgbm(X_tr, y_tr), _C["blue"]),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        [0, 1], [0, 1], ls="--", color=_C["lgray"], lw=1.2, label="Perfect calibration"
    )

    for name, clf, col in models:
        proba = clf.predict_proba(X_te)[:, 1]
        try:
            frac_pos, mean_pred = calibration_curve(
                y_te, proba, n_bins=10, strategy="quantile"
            )
            ax.plot(
                mean_pred, frac_pos, marker="o", ms=5, color=col, lw=1.5, label=name
            )
        except Exception as e:
            logger.warning("Calibration failed for %s: %s", name, e)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Figure 22 — Calibration Diagram: basis_usdc_1m_gt10bps (test split)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.03,
        0.95,
        "Points above diagonal → under-confident\nPoints below → over-confident",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color=_C["gray"],
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_22_calibration", fmt)
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate extended paper figures (Columbia theme)."
    )
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--results-dir", default="results/experiments")
    p.add_argument("--addon-results-dir", default="results/experiments_addon")
    p.add_argument("--output-dir", default="results/paper_addon/figures")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.data_dir) / "dataset.parquet"
    results_dir = Path(args.results_dir)
    addon_dir = Path(args.addon_results_dir)
    fmt = args.format

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.error("matplotlib not installed — pip install matplotlib")
        return

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found at {dataset_path}")

    logger.info("Generating extended paper figures (Columbia theme) → %s", output_dir)

    figure_14_event_timeline(output_dir, fmt)
    figure_15_model_comparison(results_dir, output_dir, fmt)
    figure_16_cumulative_pnl(dataset_path, output_dir, fmt)
    figure_17_roc_curves(dataset_path, output_dir, fmt)
    figure_18_feature_importance(dataset_path, output_dir, fmt)
    figure_19_basis_heatmap(dataset_path, output_dir, fmt)
    figure_20_cost_decomposition(dataset_path, output_dir, fmt)
    figure_21_horizon_ratio(addon_dir, output_dir, fmt)
    figure_22_calibration(dataset_path, output_dir, fmt)

    logger.info("Extended figures complete. Total: 9 figures saved to %s", output_dir)


if __name__ == "__main__":
    main()


# Needed for Figure 15 legend patches
import matplotlib.patches as mpatches  # noqa: E402
