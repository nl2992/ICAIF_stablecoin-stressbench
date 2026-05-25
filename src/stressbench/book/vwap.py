"""Executable VWAP computation for order-book walks.

The key insight of this benchmark is that apparent arbitrage windows often
disappear when measured at executable size rather than at the mid-price.
"""

from __future__ import annotations

from stressbench.book.order_book import OrderBook


def walk_book(
    levels: list[tuple[float, float]],
    qty: float,
) -> float | None:
    """Walk an order-book side to compute the VWAP for a given quantity.

    Args:
        levels: Sorted list of ``(price, size)`` tuples. For buys, pass
            :meth:`~stressbench.book.order_book.OrderBook.sorted_asks`;
            for sells, pass
            :meth:`~stressbench.book.order_book.OrderBook.sorted_bids`.
        qty: Total quantity to fill.

    Returns:
        Volume-weighted average execution price, or ``None`` if the book
        does not have sufficient depth to fill the entire quantity.
    """
    remaining = qty
    cost = 0.0
    filled = 0.0

    for price, size in levels:
        take = min(remaining, size)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break

    if filled < qty - 1e-12:
        return None  # Insufficient depth

    return cost / filled


def executable_buy_vwap(book: OrderBook, qty: float) -> float | None:
    """Return the VWAP for buying ``qty`` units from the ask side.

    Args:
        book: Reconstructed :class:`~stressbench.book.order_book.OrderBook`.
        qty: Quantity to buy.

    Returns:
        Executable buy VWAP, or ``None`` if insufficient depth.
    """
    return walk_book(book.sorted_asks(), qty)


def executable_sell_vwap(book: OrderBook, qty: float) -> float | None:
    """Return the VWAP for selling ``qty`` units into the bid side.

    Args:
        book: Reconstructed :class:`~stressbench.book.order_book.OrderBook`.
        qty: Quantity to sell.

    Returns:
        Executable sell VWAP, or ``None`` if insufficient depth.
    """
    return walk_book(book.sorted_bids(), qty)


def gross_spread_bps(
    buy_book: OrderBook,
    sell_book: OrderBook,
    qty: float,
) -> float | None:
    """Compute the gross spread in basis points between two venues at a given size.

    This is the raw price difference before any fees or transfer costs.

    Args:
        buy_book: Order book of the venue where the asset is bought.
        sell_book: Order book of the venue where the asset is sold.
        qty: Notional quantity.

    Returns:
        Gross spread in basis points, or ``None`` if either book lacks depth.
    """
    buy_vwap = executable_buy_vwap(buy_book, qty)
    sell_vwap = executable_sell_vwap(sell_book, qty)
    if buy_vwap is None or sell_vwap is None or buy_vwap == 0:
        return None
    return (sell_vwap - buy_vwap) / buy_vwap * 10_000


def net_profit_bps(
    buy_book: OrderBook,
    sell_book: OrderBook,
    qty: float,
    taker_fee_buy_bps: float = 10.0,
    taker_fee_sell_bps: float = 10.0,
    withdrawal_fee_usd: float = 1.0,
    gas_fee_usd: float = 5.0,
    settlement_delay_penalty_bps: float = 2.0,
    notional_usd: float | None = None,
) -> float | None:
    """Compute net profit in basis points after all transaction costs.

    Net = gross_spread_bps
          - taker_fee_buy_bps
          - taker_fee_sell_bps
          - (withdrawal_fee_usd + gas_fee_usd) / notional_usd * 10_000
          - settlement_delay_penalty_bps

    Args:
        buy_book: Order book of the buy venue.
        sell_book: Order book of the sell venue.
        qty: Quantity in base units.
        taker_fee_buy_bps: Taker fee on the buy side in basis points.
        taker_fee_sell_bps: Taker fee on the sell side in basis points.
        withdrawal_fee_usd: Fixed withdrawal fee in USD.
        gas_fee_usd: Estimated gas/transfer fee in USD.
        settlement_delay_penalty_bps: Penalty for settlement delay risk.
        notional_usd: Notional value in USD; computed from buy_vwap * qty if None.

    Returns:
        Net profit in basis points, or ``None`` if either book lacks depth.
    """
    gross = gross_spread_bps(buy_book, sell_book, qty)
    if gross is None:
        return None

    buy_vwap = executable_buy_vwap(buy_book, qty)
    if buy_vwap is None:
        return None

    if notional_usd is None:
        notional_usd = buy_vwap * qty

    if notional_usd <= 0:
        return None

    fixed_cost_bps = (withdrawal_fee_usd + gas_fee_usd) / notional_usd * 10_000
    net = (
        gross
        - taker_fee_buy_bps
        - taker_fee_sell_bps
        - fixed_cost_bps
        - settlement_delay_penalty_bps
    )
    return net
