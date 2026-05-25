"""Normalize raw on-chain transfer and swap events into canonical Silver records."""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def normalize_etherscan_transfers(df: pl.DataFrame, token_symbol: str) -> pl.DataFrame:
    """Normalize Etherscan ERC-20 transfer events.

    Args:
        df: Raw DataFrame from Etherscan API (as returned by
            :func:`~stressbench.ingestion.etherscan_loader.fetch_token_transfers`).
        token_symbol: Token symbol (e.g. ``"USDC"``).

    Returns:
        Normalized DataFrame for ``fact_onchain_transfer``.
    """
    if df.is_empty():
        return df

    rename_map = {
        "blockNumber": "block_number",
        "timeStamp": "ts_unix_seconds",
        "hash": "tx_hash",
        "from": "from_address",
        "to": "to_address",
        "value": "raw_value",
        "tokenDecimal": "token_decimals",
        "tokenSymbol": "token_symbol_native",
        "gasUsed": "gas_used",
        "gasPrice": "gas_price",
        "contractAddress": "contract_address",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(existing)

    df = df.with_columns(
        pl.lit(token_symbol).alias("token_symbol"),
        pl.lit("ethereum").alias("chain"),
    )

    # Convert raw value to human-readable amount using token decimals
    if "raw_value" in df.columns and "token_decimals" in df.columns:
        df = df.with_columns(
            (
                pl.col("raw_value").cast(pl.Float64)
                / (10 ** pl.col("token_decimals").cast(pl.Float64))
            ).alias("amount")
        )

    return df


def normalize_uniswap_swaps(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Uniswap v3 swap events from The Graph.

    Args:
        df: Raw swap DataFrame from
            :func:`~stressbench.ingestion.graph_loader.fetch_pool_swaps`.

    Returns:
        Normalized DataFrame for ``fact_onchain_transfer`` (DEX swaps).
    """
    if df.is_empty():
        return df

    df = df.with_columns(
        pl.lit("uniswap_v3").alias("venue_id"),
        pl.lit("ethereum").alias("chain"),
        pl.col("timestamp").cast(pl.Int64).alias("ts_unix_seconds"),
    )

    # Compute net flow direction: positive = token0 in, token1 out
    if "amount0" in df.columns and "amount1" in df.columns:
        df = df.with_columns(
            pl.col("amount0").cast(pl.Float64).alias("amount0"),
            pl.col("amount1").cast(pl.Float64).alias("amount1"),
        )

    return df
