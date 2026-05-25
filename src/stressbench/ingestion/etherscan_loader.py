"""Etherscan V2 on-chain data loader.

Fetches ERC-20 token transfer events, large mint/burn events, and gas proxies
for USDC, USDT, and DAI on Ethereum mainnet.

Reference:
    https://docs.etherscan.io/v/etherscan-v2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressbench.common.config import bronze_root, get_env, load_token_addresses
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
_CHAIN_ID = 1  # Ethereum mainnet
_MAX_RESULTS = 10000
_RATE_LIMIT_SLEEP = 0.25  # seconds between requests (free tier: 5 req/s)


def _api_key() -> str:
    key = get_env("ETHERSCAN_API_KEY", "")
    if not key:
        logger.warning("ETHERSCAN_API_KEY not set; requests may be rate-limited.")
    return key


def _get(params: dict[str, Any]) -> dict[str, Any] | None:
    """Make a GET request to the Etherscan V2 API."""
    params["chainid"] = _CHAIN_ID
    params["apikey"] = _api_key()
    try:
        resp = requests.get(_ETHERSCAN_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            logger.warning("Etherscan API error: %s", data.get("message"))
            return None
        return data
    except requests.RequestException as exc:
        logger.error("Etherscan request failed: %s", exc)
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


def fetch_token_transfers(
    token_symbol: str,
    start_block: int,
    end_block: int,
    page: int = 1,
    offset: int = _MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Fetch ERC-20 transfer events for a stablecoin.

    Args:
        token_symbol: One of ``"USDC"``, ``"USDT"``, ``"DAI"``.
        start_block: Starting Ethereum block number.
        end_block: Ending Ethereum block number.
        page: Page number for pagination.
        offset: Number of results per page (max 10000).

    Returns:
        List of transfer event dicts from Etherscan.
    """
    addresses = load_token_addresses()
    token_info = addresses.get(token_symbol)
    if not token_info:
        logger.error("Unknown token symbol: %s", token_symbol)
        return []

    contract_address = token_info.get("ethereum", {}).get("address")
    if not contract_address:
        logger.error("No Ethereum address for %s", token_symbol)
        return []

    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": contract_address,
        "startblock": start_block,
        "endblock": end_block,
        "page": page,
        "offset": offset,
        "sort": "asc",
    }
    data = _get(params)
    if data is None:
        return []
    return data.get("result", [])


def fetch_block_by_timestamp(timestamp_utc: int) -> int | None:
    """Return the closest Ethereum block number for a Unix timestamp.

    Args:
        timestamp_utc: Unix timestamp in seconds.

    Returns:
        Block number, or ``None`` on failure.
    """
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": timestamp_utc,
        "closest": "before",
    }
    data = _get(params)
    if data is None:
        return None
    try:
        return int(data["result"])
    except (KeyError, ValueError):
        return None


def save_transfers_to_bronze(
    transfers: list[dict[str, Any]],
    token_symbol: str,
    date: str,
    root: Path | None = None,
) -> Path | None:
    """Save a list of transfer events to Bronze as Parquet.

    Args:
        transfers: List of Etherscan transfer event dicts.
        token_symbol: Token symbol for partitioning.
        date: ISO date string for partitioning.
        root: Bronze root override.

    Returns:
        Path to the written Parquet file, or ``None`` if empty.
    """
    if not transfers:
        return None

    root = root or bronze_root()
    out_dir = (
        root
        / "venue=ethereum"
        / "channel=onchain_transfer"
        / f"symbol={token_symbol}"
        / f"date={date}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"transfers-{token_symbol}-{date}.parquet"

    df = pl.DataFrame(transfers)
    df.write_parquet(out_file)
    logger.info("Saved %d transfers to %s", len(transfers), out_file)
    return out_file
