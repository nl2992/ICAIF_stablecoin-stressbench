"""On-chain settlement state feature computation.

Produces observable settlement proxies from on-chain data.
These are explicitly labelled as proxies, not true settlement state,
because CEX internal settlement is not publicly observable.

Output table: ``feat_settlement_1m``
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def compute_settlement_features_1m(
    transfers_df: pl.DataFrame,
    swaps_df: pl.DataFrame | None = None,
    ts_1m_ns: int | None = None,
    chain: str = "ethereum",
    stablecoin: str = "USDC",
    large_transfer_threshold_usd: float = 1_000_000.0,
) -> dict:
    """Compute 1-minute settlement proxy features from on-chain data.

    Args:
        transfers_df: Normalized ERC-20 transfer DataFrame for the minute window.
        swaps_df: Normalized Uniswap swap DataFrame for the minute window (optional).
        ts_1m_ns: Minute-boundary timestamp in nanoseconds.
        chain: Blockchain identifier (e.g. ``"ethereum"``).
        stablecoin: Token symbol (e.g. ``"USDC"``).
        large_transfer_threshold_usd: Threshold for classifying a transfer as large.

    Returns:
        Dict of settlement features conforming to ``feat_settlement_1m``.
    """
    transfer_count = 0
    transfer_volume = 0.0
    large_transfer_count = 0
    gas_proxy = None
    dex_swap_volume = 0.0
    dex_net_flow = 0.0

    if transfers_df is not None and not transfers_df.is_empty():
        transfer_count = len(transfers_df)
        if "amount" in transfers_df.columns:
            transfer_volume = transfers_df["amount"].sum()
            large_transfer_count = int(
                (transfers_df["amount"] >= large_transfer_threshold_usd).sum()
            )
        if "gas_price" in transfers_df.columns and "gas_used" in transfers_df.columns:
            # Gas proxy: median gas price in gwei
            try:
                gas_prices = transfers_df["gas_price"].cast(pl.Float64) / 1e9
                gas_proxy = float(gas_prices.median())
            except Exception:
                pass

    if swaps_df is not None and not swaps_df.is_empty():
        if "amountUSD" in swaps_df.columns:
            dex_swap_volume = float(swaps_df["amountUSD"].cast(pl.Float64).sum())
        if "amount0" in swaps_df.columns:
            dex_net_flow = float(swaps_df["amount0"].cast(pl.Float64).sum())

    return {
        "ts_1m_ns": ts_1m_ns,
        "chain": chain,
        "stablecoin": stablecoin,
        "transfer_count_1m": transfer_count,
        "transfer_volume_1m": transfer_volume,
        "large_transfer_count_1m": large_transfer_count,
        "mint_count_1h": None,   # Requires mint event detection
        "burn_count_1h": None,   # Requires burn event detection
        "gas_proxy": gas_proxy,
        "block_lag_proxy": None,  # Requires block timestamp data
        "dex_swap_volume_1m": dex_swap_volume,
        "dex_net_flow_1m": dex_net_flow,
    }
