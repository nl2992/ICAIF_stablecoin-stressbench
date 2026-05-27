"""The Graph / Uniswap v3 subgraph loader.

Queries Uniswap v3 pool swaps, liquidity events, and pool state via
The Graph's GraphQL API.

Reference:
    https://thegraph.com/docs/en/querying/querying-the-graph/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressbench.common.config import bronze_root, load_venues
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_UNISWAP_V3_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Known Uniswap v3 pool addresses for stablecoin pairs (Ethereum mainnet)
# Source: Uniswap v3 factory / Etherscan; verify before use
_POOL_ADDRESSES: dict[str, str] = {
    "USDC/USDT_500": "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6",
    "USDC/DAI_500": "0x6c6Bc977E13Df9b0de53b251522280BB72383700",
    "USDC/WETH_500": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    "USDT/WETH_500": "0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36",
    "USDT/DAI_500": "0x6f48ECa74B38d2936B02ab603FF4e36A6C0E3A77",
}


def _graphql_query(query: str, variables: dict[str, Any] | None = None) -> dict | None:
    """Execute a GraphQL query against the Uniswap v3 subgraph."""
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        resp = requests.post(_UNISWAP_V3_SUBGRAPH, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            logger.warning("GraphQL errors: %s", data["errors"])
            return None
        return data.get("data")
    except requests.RequestException as exc:
        logger.error("GraphQL request failed: %s", exc)
        return None


def fetch_pool_swaps(
    pool_address: str,
    start_timestamp: int,
    end_timestamp: int,
    first: int = 1000,
    skip: int = 0,
) -> list[dict[str, Any]]:
    """Fetch swap events for a Uniswap v3 pool in a time range.

    Args:
        pool_address: Lowercase Ethereum address of the Uniswap v3 pool.
        start_timestamp: Unix timestamp (seconds) for the start of the window.
        end_timestamp: Unix timestamp (seconds) for the end of the window.
        first: Number of results to fetch per query (max 1000).
        skip: Number of results to skip (for pagination).

    Returns:
        List of swap event dicts.
    """
    query = """
    query PoolSwaps($pool: String!, $start: Int!, $end: Int!, $first: Int!, $skip: Int!) {
      swaps(
        where: {
          pool: $pool
          timestamp_gte: $start
          timestamp_lte: $end
        }
        first: $first
        skip: $skip
        orderBy: timestamp
        orderDirection: asc
      ) {
        id
        timestamp
        pool { id }
        token0 { symbol }
        token1 { symbol }
        amount0
        amount1
        amountUSD
        sqrtPriceX96
        tick
        logIndex
        transaction { id blockNumber gasUsed gasPrice }
      }
    }
    """
    variables = {
        "pool": pool_address.lower(),
        "start": start_timestamp,
        "end": end_timestamp,
        "first": first,
        "skip": skip,
    }
    data = _graphql_query(query, variables)
    if data is None:
        return []
    return data.get("swaps", [])


def fetch_pool_hourly_data(
    pool_address: str,
    start_timestamp: int,
    end_timestamp: int,
) -> list[dict[str, Any]]:
    """Fetch hourly OHLCV and liquidity data for a Uniswap v3 pool.

    Args:
        pool_address: Lowercase Ethereum address of the Uniswap v3 pool.
        start_timestamp: Unix timestamp (seconds).
        end_timestamp: Unix timestamp (seconds).

    Returns:
        List of hourly data dicts.
    """
    query = """
    query PoolHourly($pool: String!, $start: Int!, $end: Int!) {
      poolHourDatas(
        where: {
          pool: $pool
          periodStartUnix_gte: $start
          periodStartUnix_lte: $end
        }
        orderBy: periodStartUnix
        orderDirection: asc
        first: 1000
      ) {
        periodStartUnix
        liquidity
        sqrtPrice
        token0Price
        token1Price
        volumeToken0
        volumeToken1
        volumeUSD
        feesUSD
        txCount
        open
        high
        low
        close
      }
    }
    """
    variables = {
        "pool": pool_address.lower(),
        "start": start_timestamp,
        "end": end_timestamp,
    }
    data = _graphql_query(query, variables)
    if data is None:
        return []
    return data.get("poolHourDatas", [])


def save_swaps_to_bronze(
    swaps: list[dict[str, Any]],
    pool_label: str,
    date: str,
    root: Path | None = None,
) -> Path | None:
    """Save swap events to Bronze as Parquet.

    Args:
        swaps: List of swap event dicts.
        pool_label: Human-readable pool label for partitioning (e.g. ``"USDC_USDT_500"``).
        date: ISO date string for partitioning.
        root: Bronze root override.

    Returns:
        Path to the written Parquet file, or ``None`` if empty.
    """
    if not swaps:
        return None

    root = root or bronze_root()
    out_dir = (
        root
        / "venue=uniswap_v3"
        / "channel=swap"
        / f"symbol={pool_label}"
        / f"date={date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"swaps-{pool_label}-{date}.parquet"

    # Flatten nested dicts for Parquet compatibility
    flat_swaps = []
    for s in swaps:
        row = dict(s)
        if isinstance(row.get("pool"), dict):
            row["pool_id"] = row.pop("pool", {}).get("id", "")
        if isinstance(row.get("token0"), dict):
            row["token0_symbol"] = row.pop("token0", {}).get("symbol", "")
        if isinstance(row.get("token1"), dict):
            row["token1_symbol"] = row.pop("token1", {}).get("symbol", "")
        if isinstance(row.get("transaction"), dict):
            txn = row.pop("transaction", {})
            row["tx_id"] = txn.get("id", "")
            row["block_number"] = txn.get("blockNumber", "")
            row["gas_used"] = txn.get("gasUsed", "")
            row["gas_price"] = txn.get("gasPrice", "")
        flat_swaps.append(row)

    pl.DataFrame(flat_swaps).write_parquet(out_file)
    logger.info("Saved %d swaps to %s", len(swaps), out_file)
    return out_file
