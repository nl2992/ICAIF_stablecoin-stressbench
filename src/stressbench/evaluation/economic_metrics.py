"""Economic evaluation metrics.

These metrics matter more than raw ML metrics because a model that predicts
basis well but flags non-executable opportunities is not successful.

Metrics:
    net_bps_captured: Average net basis points captured per trade
    hit_rate_above_cost: Fraction of predicted opportunities that are profitable
    false_positive_cost: Average cost of false-positive signals
    average_depth_consumed: Average order-book depth consumed per trade
    turnover: Total notional traded
    max_drawdown: Maximum drawdown of cumulative P&L
    sharpe_ratio: Annualised Sharpe-like ratio
    profit_per_unit_turnover: Net P&L per unit of notional traded
"""

from __future__ import annotations

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def net_bps_captured(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
) -> float:
    """Average net basis points captured by the model's positive predictions.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points per opportunity.
        y_pred_signal: Binary signal (1 = model predicts opportunity).

    Returns:
        Mean net bps for predicted opportunities, or ``nan`` if no positives.
    """
    mask = y_pred_signal == 1
    if not mask.any():
        return float("nan")
    return float(np.mean(y_true_net_profit[mask]))


def hit_rate_above_cost(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
    cost_threshold_bps: float = 0.0,
) -> float:
    """Fraction of predicted opportunities where net profit exceeds the cost threshold.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points.
        y_pred_signal: Binary signal.
        cost_threshold_bps: Minimum net profit to count as a hit.

    Returns:
        Hit rate in [0, 1].
    """
    mask = y_pred_signal == 1
    if not mask.any():
        return float("nan")
    hits = y_true_net_profit[mask] > cost_threshold_bps
    return float(hits.mean())


def false_positive_cost(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
    cost_threshold_bps: float = 0.0,
) -> float:
    """Average cost (negative net profit) of false-positive signals.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points.
        y_pred_signal: Binary signal.
        cost_threshold_bps: Threshold below which a prediction is a false positive.

    Returns:
        Mean negative net profit for false positives, or ``nan`` if none.
    """
    mask = (y_pred_signal == 1) & (y_true_net_profit <= cost_threshold_bps)
    if not mask.any():
        return float("nan")
    return float(np.mean(y_true_net_profit[mask]))


def cumulative_pnl(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
    notional_usd: float = 50_000.0,
) -> np.ndarray:
    """Compute cumulative P&L in USD for a strategy that follows the model's signals.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points.
        y_pred_signal: Binary signal.
        notional_usd: Notional size per trade in USD.

    Returns:
        Cumulative P&L array in USD.
    """
    trade_pnl = y_pred_signal * y_true_net_profit / 10_000 * notional_usd
    return np.cumsum(trade_pnl)


def max_drawdown(cum_pnl: np.ndarray) -> float:
    """Compute the maximum drawdown of a cumulative P&L series.

    Args:
        cum_pnl: Cumulative P&L array.

    Returns:
        Maximum drawdown in the same units as ``cum_pnl``.
    """
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    return float(drawdown.max())


def sharpe_ratio(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
    periods_per_year: float = 525_600.0,  # 1-minute periods
) -> float:
    """Compute an annualised Sharpe-like ratio for the strategy.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points.
        y_pred_signal: Binary signal.
        periods_per_year: Number of periods per year for annualisation.

    Returns:
        Annualised Sharpe ratio.
    """
    trade_returns = y_pred_signal * y_true_net_profit
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    if std_return == 0:
        return float("nan")
    return float(mean_return / std_return * np.sqrt(periods_per_year))


def economic_summary(
    y_true_net_profit: np.ndarray,
    y_pred_signal: np.ndarray,
    notional_usd: float = 50_000.0,
    cost_threshold_bps: float = 0.0,
) -> dict[str, float]:
    """Compute a full economic evaluation summary.

    Args:
        y_true_net_profit: Ground-truth net profit in basis points.
        y_pred_signal: Binary signal.
        notional_usd: Notional size per trade in USD.
        cost_threshold_bps: Cost threshold for hit-rate computation.

    Returns:
        Dict of economic evaluation metrics.
    """
    cum_pnl = cumulative_pnl(y_true_net_profit, y_pred_signal, notional_usd)
    n_trades = int(y_pred_signal.sum())
    total_turnover = n_trades * notional_usd

    return {
        "net_bps_captured": net_bps_captured(y_true_net_profit, y_pred_signal),
        "hit_rate_above_cost": hit_rate_above_cost(
            y_true_net_profit, y_pred_signal, cost_threshold_bps
        ),
        "false_positive_cost": false_positive_cost(
            y_true_net_profit, y_pred_signal, cost_threshold_bps
        ),
        "n_trades": n_trades,
        "total_turnover_usd": total_turnover,
        "final_pnl_usd": float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0,
        "max_drawdown_usd": max_drawdown(cum_pnl),
        "sharpe_ratio": sharpe_ratio(y_true_net_profit, y_pred_signal),
        "profit_per_unit_turnover_bps": (
            float(cum_pnl[-1]) / total_turnover * 10_000
            if total_turnover > 0 else float("nan")
        ),
    }
