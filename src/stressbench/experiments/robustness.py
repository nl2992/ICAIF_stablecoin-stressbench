"""Robustness grid for the price-to-execution gap.

Recomputes the gap across notionals, basis thresholds, fee regimes,
settlement penalties, and prediction horizons — all from the committed
dataset.parquet without re-running the full pipeline.

Fee-regime adjustments are modelled as additive ±bps corrections to the
committed net_profit_bps columns (which already include base fees). This
is an approximation; exact decomposition would require separate fee columns.

Executable signal uses the **forward rolling max** of adjusted net_profit over
each horizon window, so fee and settlement parameters genuinely affect the
executable percentage at every (fee, settlement, horizon) combination. This
fixes the previous bug where precomputed label columns were used directly,
causing fee/settlement parameters to have no effect.

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
    "base_fee": 0.0,  # no adjustment
    "low_fee": +2.0,  # lower fees → add 2 bps to net profit
    "high_fee": -2.0,  # higher fees → subtract 2 bps
    "institutional_fee": +3.0,  # institutional cap ~2 bps total vs typical 5 bps
}

HORIZONS = ["1m", "5m", "15m"]

_HORIZON_STEPS: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15}

_NOTIONAL_TO_COL: dict[int, str] = {
    10_000: "net_profit_bps_q10000",
    50_000: "net_profit_bps_q50000",
    100_000: "net_profit_bps_q100000",
    500_000: "net_profit_bps_q500000",
}

_BASIS_COL = "cross_quote_basis_usdc_bps"


def _forward_max(arr: np.ndarray, steps: int) -> np.ndarray:
    """Forward rolling maximum: result[i] = max(arr[i], …, arr[i+steps-1]).

    NaN entries in arr are ignored in the max (nanmax semantics). Windows that
    consist entirely of NaN produce NaN in the result.

    This mirrors how the precomputed label_arb_*_gt0bps columns are built:
    a window is labelled executable if any minute within the horizon has
    positive net profit.
    """
    result = arr.copy().astype(float)
    for k in range(1, steps):
        shifted = np.full_like(arr, np.nan, dtype=float)
        shifted[: len(arr) - k] = arr[k:]
        result = np.fmax(result, shifted)  # fmax ignores NaN
    return result


def compute_robustness_grid(
    dataset_path: Path,
    splits: list[str] | None = None,
) -> list[dict]:
    """Compute price-to-execution gap across the full parameter grid.

    For each (split × notional × basis_threshold × settlement_penalty ×
    fee_regime × horizon):

    - price_signal_pct   = fraction of minutes where |basis| > threshold
    - executable_signal_pct = fraction where forward_max(adjusted_net) > 0
                              over `horizon` steps, where:
                              adjusted_net = net_profit - settlement_bps + fee_adj
    - price_to_execution_ratio = price_signal_pct / executable_signal_pct

    Parameters
    ----------
    dataset_path:
        Path to data/gold/dataset.parquet (must contain net_profit_bps_q* cols).
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
        # Sort by timestamp so forward-window indexing is correct
        sdf = df.filter(pl.col("split") == split)
        if "ts_1m_ns" in sdf.columns:
            sdf = sdf.sort("ts_1m_ns")
        n = len(sdf)
        if n == 0:
            logger.warning("No rows for split=%s", split)
            continue

        basis_vals = (
            sdf[_BASIS_COL].to_numpy().astype(float)
            if _BASIS_COL in sdf.columns
            else None
        )

        for notional in NOTIONALS:
            net_col = _NOTIONAL_TO_COL[notional]
            if net_col not in sdf.columns:
                logger.warning("Missing %s — skipping notional %d", net_col, notional)
                continue

            net_vals = sdf[net_col].to_numpy().astype(float)

            # Oracle: mean net profit on unadjusted profitable windows
            # (oracle is a fixed baseline reference; we don't adjust it for fee scenarios)
            valid = net_vals[~np.isnan(net_vals)]
            oracle_net = (
                float(np.mean(valid[valid > 0])) if (valid > 0).any() else float("nan")
            )
            oracle_n = int((valid > 0).sum())

            for threshold_bps in BASIS_THRESHOLDS_BPS:
                # Price signal: |basis| > threshold (independent of costs)
                if basis_vals is not None:
                    basis_clean = np.where(np.isnan(basis_vals), 0.0, basis_vals)
                    price_mask = np.abs(basis_clean) > threshold_bps
                    price_pct = float(price_mask.sum()) / n * 100
                else:
                    price_pct = float("nan")

                for settlement_bps in SETTLEMENT_PENALTIES_BPS:
                    for fee_regime, fee_adj in FEE_REGIMES.items():
                        # Adjusted net profit: lower fees → higher adjusted profit
                        adjusted = net_vals - settlement_bps + fee_adj

                        for horizon in HORIZONS:
                            steps = _HORIZON_STEPS[horizon]
                            # Forward rolling max of adjusted net profit over horizon
                            fwd_max = _forward_max(adjusted, steps)
                            valid_fwd = ~np.isnan(fwd_max)
                            exec_pct = (
                                float((fwd_max[valid_fwd] > 0).sum()) / n * 100
                                if valid_fwd.any()
                                else float("nan")
                            )

                            ratio = (
                                price_pct / exec_pct
                                if exec_pct > 0 and exec_pct == exec_pct
                                else float("nan")
                            )

                            rows.append(
                                {
                                    "split": split,
                                    "notional": notional,
                                    "basis_threshold_bps": threshold_bps,
                                    "settlement_penalty_bps": settlement_bps,
                                    "fee_regime": fee_regime,
                                    "horizon": horizon,
                                    "n_minutes": n,
                                    "price_signal_pct": round(price_pct, 3),
                                    "executable_signal_pct": round(exec_pct, 3),
                                    "price_to_execution_ratio": (
                                        round(ratio, 2) if ratio == ratio else ""
                                    ),
                                    "oracle_net_bps": (
                                        round(oracle_net, 2)
                                        if oracle_net == oracle_net
                                        else ""
                                    ),
                                    "oracle_n_trades": oracle_n,
                                }
                            )

    return rows
