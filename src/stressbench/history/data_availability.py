"""Data availability profiles for historical stablecoin stress events.

Provides:
    DataAvailabilityProfile  — Dataclass describing available data sources.
    coverage_score()         — Computes 0.25/0.50/0.75/1.0 coverage score from sources.
    PREDEFINED_COVERAGE      — Pre-defined profiles for all catalog events.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataAvailabilityProfile:
    """Describes the data available for a historical stress event.

    Attributes:
        event_id: Identifier matching EVENT_CATALOG.
        price_data: OHLCV price data available (CEX or DEX).
        trade_data: Tick-level trade tape available.
        l2_data: Full L2 order book depth snapshots available.
        dex_pool_data: DEX liquidity pool reserves available (e.g. Curve).
        onchain_data: On-chain transfer/settlement data available.
        execution_grade_available: True iff net_profit_bps labels are computable
            (requires l2_data=True for all benchmark venues).
        coverage_score: 0.25 (context-only) / 0.50 (price-grade) /
            0.75 (partial execution) / 1.0 (full execution-grade).
    """

    event_id: str
    price_data: bool = False
    trade_data: bool = False
    l2_data: bool = False
    dex_pool_data: bool = False
    onchain_data: bool = False
    execution_grade_available: bool = False
    coverage_score: float = 0.25

    def __post_init__(self) -> None:
        # Validate coverage_score
        valid = {0.25, 0.50, 0.75, 1.0}
        if self.coverage_score not in valid:
            raise ValueError(
                f"coverage_score must be one of {valid}, got {self.coverage_score}"
            )
        # Execution-grade requires L2 data
        if self.execution_grade_available and not self.l2_data:
            raise ValueError(
                "execution_grade_available=True requires l2_data=True"
            )


def coverage_score(
    price_data: bool = False,
    trade_data: bool = False,
    l2_data: bool = False,
    dex_pool_data: bool = False,
    onchain_data: bool = False,
) -> float:
    """Compute a coverage score based on available data sources.

    Scoring rules:
        - No data at all: 0.25 (minimum, context-only)
        - Price data only: 0.25
        - Price + any of (trade, dex, onchain): 0.50
        - Price + trade + any of (dex, onchain): 0.50
        - Price + trade + L2 (partial coverage): 0.75
        - Price + trade + L2 + (dex or onchain): 1.0

    Args:
        price_data: OHLCV price data available.
        trade_data: Tick-level trade tape available.
        l2_data: Full L2 order book snapshots available.
        dex_pool_data: DEX pool reserves available.
        onchain_data: On-chain transfer data available.

    Returns:
        Coverage score: 0.25, 0.50, 0.75, or 1.0.
    """
    if not price_data and not trade_data and not l2_data:
        return 0.25

    if l2_data and trade_data and price_data:
        if dex_pool_data or onchain_data:
            return 1.0
        return 0.75

    if price_data and (trade_data or dex_pool_data or onchain_data):
        return 0.50

    return 0.25


# ---------------------------------------------------------------------------
# Pre-defined coverage profiles for all catalog events
# ---------------------------------------------------------------------------

PREDEFINED_COVERAGE: dict[str, DataAvailabilityProfile] = {
    "iron_titan_2021": DataAvailabilityProfile(
        event_id="iron_titan_2021",
        price_data=True,
        trade_data=False,
        l2_data=False,
        dex_pool_data=True,
        onchain_data=False,
        execution_grade_available=False,
        coverage_score=0.25,
    ),
    "terra_ust_2022": DataAvailabilityProfile(
        event_id="terra_ust_2022",
        price_data=True,
        trade_data=True,
        l2_data=False,
        dex_pool_data=True,
        onchain_data=True,
        execution_grade_available=False,
        coverage_score=0.50,
    ),
    "ftx_collapse_2022": DataAvailabilityProfile(
        event_id="ftx_collapse_2022",
        price_data=True,
        trade_data=True,
        l2_data=False,
        dex_pool_data=False,
        onchain_data=True,
        execution_grade_available=False,
        coverage_score=0.50,
    ),
    "busd_regulatory_2023": DataAvailabilityProfile(
        event_id="busd_regulatory_2023",
        price_data=True,
        trade_data=False,
        l2_data=False,
        dex_pool_data=False,
        onchain_data=True,
        execution_grade_available=False,
        coverage_score=0.50,
    ),
    "usdc_svb_2023": DataAvailabilityProfile(
        event_id="usdc_svb_2023",
        price_data=True,
        trade_data=True,
        l2_data=True,
        dex_pool_data=False,
        onchain_data=True,
        execution_grade_available=True,
        coverage_score=1.0,
    ),
    "usdc_svb_recovery_2023": DataAvailabilityProfile(
        event_id="usdc_svb_recovery_2023",
        price_data=True,
        trade_data=True,
        l2_data=True,
        dex_pool_data=False,
        onchain_data=False,
        execution_grade_available=True,
        coverage_score=0.75,
    ),
    "usdt_curve_2023": DataAvailabilityProfile(
        event_id="usdt_curve_2023",
        price_data=True,
        trade_data=False,
        l2_data=False,
        dex_pool_data=True,
        onchain_data=True,
        execution_grade_available=False,
        coverage_score=0.50,
    ),
}
