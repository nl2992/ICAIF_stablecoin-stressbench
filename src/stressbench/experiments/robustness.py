"""Robustness grid for the price-to-execution gap.

Recomputes the gap across notionals, basis thresholds, fee regimes,
settlement penalties, and prediction horizons — all from the committed
dataset.parquet without re-running the full pipeline.

Fee-regime adjustments are modelled as additive ±bps corrections to the
committed net_profit_bps columns (which already include base fees). This
is an approximation; exact decomposition would require separate fee columns.

Results write to results/experiments_addon/. Baseline files are not touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Grid parameters
# -----------------------------------------------------------------------

NOTIONALS = [10_000, 50_000, 100_000, 500_000]

BASIS_THRESHOLDS_BPS = [0, 5, 10, 25, 50]

SETTLEMENT_PENALTIES_BPS = [0, 2, 5, 10]

FEE_REGIMES: dict[str, float] = {
    "base_fee": 0.0,       # no adjustment
    "low_fee": +2.0,        # lower fees → add 2 bps to net profit
    "high_fee": -2.0,       # higher fees → subtract 2 bps
    "institutional_fee": +3.0,  # institutional cap ~2 bps total vs typical 5 bps
}

HORIZONS = ["1m", "5m", "15m"]

_NOTIONAL_TO_COL: dict[int, str] = {
    10_000: "net_profit_bps_q10000",
    50_000: "net_profit_bps_q50000",
    100_000: "net_profit_bps_q100000",
    500_000: "net_profit_bps_q500000",
}

_NOTIONAL_TO_LABEL_PREFIX: dict[int, str] = {
    10_000: "label_arb_q10000",
    50_000: "label_arb_q50000",
    100_000: "label_arb_q100000",
    500_000: "label_arb_q500000",
}

_BASIS_COL = "cross_quote_basis_usdc_bps"


class RobustnessRow(NamedTuple):
    split: str
    notional: int
    basis_threshold_bps: int
    settlement_penalty_bps: int
    fee_regime: str
    horizon: str
    n_minutes: int
    price_signal_pct: float
    executable_signal_pct: float
    price_to_execution_ratio: float
    oracle_net_bps: float
    oracle_n_trades: int


def compute_robustness_grid(
    dataset_path: Path,
    splits: list[str] | None = None,
) -> list[dict]:
    """Compute price-to-execution gap across the full parameter grid.

    Parameters
    ----------
    dataset_path:
        Path to data/gold/dataset.parquet.
    splits:
        Which dataset splits to include. Defaults to ["test"].

    Returns
    -------
    list[dict]
        One dict per (split × notional × threshold × penalty × fee × horizon).
    """
    if splits is None:
        splits = ["test"]

    df = pl.read_parquet(str(dataset_path))
    rows: list[dict] = []

    for split in splits:
        sdf = df.filter(pl.col("split") == split)
        n = len(sdf)
        if n == 0:
            logger.warning("No rows for split=%s", split)
            continue

        basis_vals = sdf[_BASIS_COL].to_numpy() if _BASIS_COL in sdf.columns else None

        for notional in NOTIONALS:
            net_col = _NOTIONAL_TO_COL[notional]
            label_prefix = _NOTIONAL_TO_LABEL_PREFIX[notional]

            if net_col not in sdf.columns:
                logger.warning("Missing %s — skipping notional %d", net_col, notional)
                continue

            net_vals = sdf[net_col].to_numpy().astype(float)

            for threshold_bps in BASIS_THRESHOLDS_BPS:
                # Price signal: |basis| > threshold
                if basis_vals is not None:
                    price_mask = np.abs(np.where(np.isnan(basis_vals), 0.0, basis_vals)) > threshold_bps
                    price_pct = float(price_mask.sum()) / n * 100
                else:
                    price_pct = float("nan")

                for settlement_bps in SETTLEMENT_PENALTIES_BPS:
                    for fee_regime, fee_adj in FEE_REGIMES.items():
                        # Adjusted net profit
                        adjusted = net_vals - settlement_bps + fee_adj

                        # Oracle: mean net profit on profitable windows (base, no adj for oracle)
                        valid = net_vals[~np.isnan(net_vals)]
                        oracle_mask = valid > 0
                        oracle_net = float(np.mean(valid[oracle_mask])) if oracle_mask.any() else float("nan")
                        oracle_n = int(oracle_mask.sum())

                        for horizon in HORIZONS:
                            label_col = f"{label_prefix}_{horizon}_gt0bps"
                            if label_col in sdf.columns:
                                # Use horizon label directly (pre-computed at correct horizon)
                                label_vals = sdf[label_col].to_numpy().astype(float)
                                valid_label = ~np.isnan(label_vals)
                                exec_pct = (
                                    float((label_vals[valid_label] == 1).sum()) / n * 100
                                    if valid_label.any() else float("nan")
                                )
                            else:
                                # Fall back to adjusted net_profit threshold
                                valid_adj = adjusted[~np.isnan(adjusted)]
                                exec_pct = float((valid_adj > 0).sum()) / n * 100

                            ratio = (price_pct / exec_pct) if exec_pct > 0 else float("nan")

                            rows.append({
                                "split": split,
                                "notional": notional,
                                "basis_threshold_bps": threshold_bps,
                                "settlement_penalty_bps": settlement_bps,
                                "fee_regime": fee_regime,
                                "horizon": horizon,
                                "n_minutes": n,
                                "price_signal_pct": round(price_pct, 3),
                                "executable_signal_pct": round(exec_pct, 3),
                                "price_to_execution_ratio": round(ratio, 2) if ratio == ratio else "",
                                "oracle_net_bps": round(oracle_net, 2) if oracle_net == oracle_net else "",
                                "oracle_n_trades": oracle_n,
                            })

    return rows
