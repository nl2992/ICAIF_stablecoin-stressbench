"""Named lag-column definitions for time-series baselines.

The AR1Baseline and LastValueBaseline require a column of X that contains
the lagged target value y(t-1). This module provides:

1. NAMED_LAG_COLUMNS: the authoritative list of lag-feature column names
   expected to be present in the Gold dataset for each task/feature-set.

2. get_lag_col_idx(feature_names, task): returns the column index of the
   appropriate lag feature for a given task and feature name list.

These constants prevent baselines from silently relying on column ordering
(the previous fragile default of X[:, -1]).

Usage
-----
    from stressbench.features.lags import get_lag_col_idx

    lag_idx = get_lag_col_idx(feature_names, task="basis_usdc_1m_gt10bps")
    model = AR1Baseline(lag_col_idx=lag_idx)
    model.fit(X_train, y_train)
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Named lag columns
# ---------------------------------------------------------------------------

# For each task family, the preferred lag-feature column name in the dataset.
# These are derived from the Gold pipeline's feat_basis_1m and feat_net_profit_1m
# tables and must match the column names in dataset.parquet.

TASK_LAG_COLUMNS: Dict[str, str] = {
    # Basis classification tasks
    "basis_usdc_1m_gt10bps": "cross_quote_basis_usdc_bps",
    "basis_usdc_5m_gt10bps": "cross_quote_basis_usdc_bps",
    "basis_usdc_5m_gt25bps": "cross_quote_basis_usdc_bps",
    "basis_usdc_15m_gt10bps": "cross_quote_basis_usdc_bps",
    "basis_usdc_1h_gt10bps": "cross_quote_basis_usdc_bps",
    # Maxabs basis tasks
    "basis_maxabs_1m_gt10bps": "cross_quote_basis_maxabs_bps",
    "basis_maxabs_5m_gt10bps": "cross_quote_basis_maxabs_bps",
    # Executable arbitrage tasks
    "executable_arb_q10000_1m": "net_profit_bps_q10000",
    "executable_arb_q10000_5m": "net_profit_bps_q10000",
    "executable_arb_q10000_15m": "net_profit_bps_q10000",
    "executable_arb_q50000_5m": "net_profit_bps_q50000",
}

# Ordered preference list: try these columns in order if the task-specific
# column is absent from the feature set.
LAG_COLUMN_PREFERENCE: List[str] = [
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_maxabs_bps",
    "cross_quote_basis_primary_bps",
    "net_profit_bps_q10000",
    "net_profit_bps_q50000",
]


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------


def get_lag_col_idx(
    feature_names: List[str],
    task: Optional[str] = None,
) -> int:
    """Return the column index of the best lag feature for a given task.

    Priority:
    1. Task-specific lag column from TASK_LAG_COLUMNS (if present in feature_names)
    2. First match from LAG_COLUMN_PREFERENCE (if present in feature_names)
    3. Fallback: column 0 (logged as a warning)

    Args:
        feature_names: Ordered list of feature column names as used in X.
        task: Task name string (e.g. "basis_usdc_1m_gt10bps"). Used to look up
            the preferred lag column in TASK_LAG_COLUMNS.

    Returns:
        Column index (int) of the best available lag feature.

    Raises:
        ValueError: If feature_names is empty.
    """
    if not feature_names:
        raise ValueError("feature_names must be non-empty.")

    # Try task-specific column first
    if task is not None:
        preferred = TASK_LAG_COLUMNS.get(task)
        if preferred is not None and preferred in feature_names:
            return feature_names.index(preferred)

    # Try preference list
    for col in LAG_COLUMN_PREFERENCE:
        if col in feature_names:
            return feature_names.index(col)

    # Fallback to column 0 with a warning
    import warnings

    warnings.warn(
        f"No known lag column found in feature_names for task={task!r}. "
        f"Falling back to column 0. Provide explicit lag_col_idx to suppress this. "
        f"Known lag columns: {LAG_COLUMN_PREFERENCE}",
        UserWarning,
        stacklevel=2,
    )
    return 0
