"""Stablecoin basis and cross-quote implied price features.

Computes:
    - USD-normalised stablecoin prices per venue
    - Cross-quote BTC/ETH implied prices
    - Cross-venue fragmentation index

Output tables: ``feat_stablecoin_fx_1s``, ``feat_cross_quote_basis_1s``,
               ``feat_fragmentation_1m``
"""

from __future__ import annotations

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def compute_stablecoin_usd_price(
    mid: float | None,
    quote_asset: str,
    usdt_usd_ref: float | None = None,
    usdc_usd_ref: float | None = None,
) -> float | None:
    """Compute the USD-normalised price of a stablecoin.

    For each stablecoin ``s``, venue ``v``, and time ``t``::

        price_usd(v, s, t) =
          direct USD mid             if pair is s/USD
          s/USDT mid × USDT/USD ref  if pair is s/USDT
          s/USDC mid × USDC/USD ref  if pair is s/USDC

    Args:
        mid: Mid-price of the stablecoin pair.
        quote_asset: Quote asset of the pair (``"USD"``, ``"USDT"``, or ``"USDC"``).
        usdt_usd_ref: Reference USDT/USD mid-price (required for USDT-quoted pairs).
        usdc_usd_ref: Reference USDC/USD mid-price (required for USDC-quoted pairs).

    Returns:
        USD-normalised mid-price, or ``None`` if the required reference is missing.
    """
    if mid is None:
        return None
    if quote_asset == "USD":
        return mid
    if quote_asset == "USDT":
        if usdt_usd_ref is None:
            return None
        return mid * usdt_usd_ref
    if quote_asset == "USDC":
        if usdc_usd_ref is None:
            return None
        return mid * usdc_usd_ref
    return None


def compute_cross_quote_basis_bps(
    btc_usd_direct: float | None,
    btc_usd_via_quote: float | None,
) -> float | None:
    """Compute cross-quote basis in basis points.

    This is the key benchmark feature. It measures whether the quote currency
    is behaving like cash::

        basis_bps = 10000 × (BTC_USD_via_USDC - BTC_USD_direct) / BTC_USD_direct

    Args:
        btc_usd_direct: Direct BTC/USD mid-price.
        btc_usd_via_quote: Implied BTC/USD price via a stablecoin quote.

    Returns:
        Cross-quote basis in basis points, or ``None`` if either price is missing.
    """
    if btc_usd_direct is None or btc_usd_via_quote is None or btc_usd_direct == 0:
        return None
    return 10_000 * (btc_usd_via_quote - btc_usd_direct) / btc_usd_direct


def compute_fragmentation_features(
    venue_mids: dict[str, float],
    venue_depths: dict[str, float] | None = None,
    venue_spreads: dict[str, float] | None = None,
) -> dict:
    """Compute cross-venue fragmentation features for a stablecoin.

    Args:
        venue_mids: Dict mapping venue_id to USD-normalised mid-price.
        venue_depths: Dict mapping venue_id to depth within 10bp (optional).
        venue_spreads: Dict mapping venue_id to spread in bps (optional).

    Returns:
        Dict of fragmentation features conforming to ``feat_fragmentation_1m``.
    """
    valid_mids = {v: m for v, m in venue_mids.items() if m is not None}
    if not valid_mids:
        return {
            "num_active_venues": 0,
            "mid_dispersion_bps": None,
            "max_minus_min_bps": None,
            "depth_weighted_dispersion_bps": None,
            "volume_share_hhi": None,
            "depth_share_hhi": None,
        }

    mids = list(valid_mids.values())
    mean_mid = sum(mids) / len(mids)
    max_mid = max(mids)
    min_mid = min(mids)

    dispersion_bps = (
        (sum((m - mean_mid) ** 2 for m in mids) / len(mids)) ** 0.5 / mean_mid * 10_000
        if mean_mid > 0
        else None
    )
    max_minus_min_bps = (
        (max_mid - min_mid) / mean_mid * 10_000 if mean_mid > 0 else None
    )

    # Depth-weighted dispersion
    depth_weighted_dispersion_bps = None
    depth_share_hhi = None
    if venue_depths:
        valid_depths = {v: venue_depths.get(v, 0.0) for v in valid_mids}
        total_depth = sum(valid_depths.values())
        if total_depth > 0:
            weights = {v: d / total_depth for v, d in valid_depths.items()}
            depth_weighted_mean = sum(weights[v] * valid_mids[v] for v in valid_mids)
            depth_weighted_dispersion_bps = (
                sum(
                    weights[v] * (valid_mids[v] - depth_weighted_mean) ** 2
                    for v in valid_mids
                )
                ** 0.5
                / depth_weighted_mean
                * 10_000
                if depth_weighted_mean > 0
                else None
            )
            depth_share_hhi = sum(w**2 for w in weights.values())

    return {
        "num_active_venues": len(valid_mids),
        "mid_dispersion_bps": dispersion_bps,
        "max_minus_min_bps": max_minus_min_bps,
        "depth_weighted_dispersion_bps": depth_weighted_dispersion_bps,
        "volume_share_hhi": None,  # Requires trade volume data
        "depth_share_hhi": depth_share_hhi,
    }
