"""Meta-labeling filter for stablecoin arbitrage signal filtering.

Implements López de Prado (2018) meta-labeling: a two-stage approach where a
primary model generates binary signals, and a secondary (meta) model filters
false positives among the primary fires.

In the benchmark context:
    - Primary signal: |cross_quote_basis_usdc_bps| > threshold (default 10 bps)
    - Meta-label:     1{future net_profit_bps > 0} where primary fires

The meta-model trains only on rows where the primary signal fires, making it
a specialized false-positive detector rather than a general classifier.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning, Chapter 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MetaLabelingFilter:
    """Meta-labeling filter: trains secondary model only where primary fires.

    The combined prediction is:
        predict(x) = 1  iff  primary_fires(x) AND meta_model.predict(x) == 1

    Args:
        primary_threshold_bps: Basis threshold for primary signal (default 10 bps).
        base_clf: Secondary classifier. If None, defaults to LightGBM classifier.
        primary_signal_col: Index of the basis column in X (used to compute primary
            signal from X directly when y_primary_signal is not provided to predict).
            Set to 0 by default.
    """

    name: str = "MetaLabelingFilter"

    def __init__(
        self,
        primary_threshold_bps: float = 10.0,
        base_clf: Any | None = None,
        primary_signal_col: int = 0,
    ) -> None:
        self.primary_threshold_bps = primary_threshold_bps
        self.primary_signal_col = primary_signal_col
        self._base_clf = base_clf
        self._meta_clf: Any = None
        self._n_primary_fires_train: int = 0
        self._n_meta_positive_train: int = 0
        self._fitted: bool = False

    def _default_clf(self) -> Any:
        """Build the default LightGBM meta-classifier."""
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )

    def _primary_fires(self, X: np.ndarray) -> np.ndarray:
        """Compute primary signal from X using the basis column."""
        basis_col = X[:, self.primary_signal_col]
        return (np.abs(basis_col) > self.primary_threshold_bps).astype(np.int8)

    def fit(
        self,
        X: np.ndarray,
        y_primary_signal: np.ndarray,
        y_meta_label: np.ndarray,
    ) -> "MetaLabelingFilter":
        """Train the meta-model on rows where the primary signal fires.

        Args:
            X: Feature matrix, shape (n, p).
            y_primary_signal: Primary binary signal, shape (n,).
                1 where |basis| > threshold, 0 otherwise.
            y_meta_label: Meta-label, shape (n,).
                1 where primary fires AND future net_profit_bps > 0.

        Returns:
            self
        """
        primary_mask = y_primary_signal.astype(bool)
        self._n_primary_fires_train = int(primary_mask.sum())

        if self._n_primary_fires_train == 0:
            # No primary fires in training data — meta-model cannot be fitted
            self._fitted = False
            return self

        X_meta = X[primary_mask]
        y_meta = y_meta_label[primary_mask].astype(np.int8)
        self._n_meta_positive_train = int(y_meta.sum())

        if self._base_clf is not None:
            self._meta_clf = self._base_clf
        else:
            self._meta_clf = self._default_clf()

        # Handle edge case: only one class in training subset
        unique_classes = np.unique(y_meta)
        if len(unique_classes) < 2:
            # Degenerate: all same class — predict that class always
            self._degenerate_class = int(unique_classes[0])
            self._fitted = True
            self._degenerate = True
            return self

        self._degenerate = False
        self._meta_clf.fit(X_meta, y_meta)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict trade signal: 1 only where primary fires AND meta says yes.

        Args:
            X: Feature matrix, shape (n, p).

        Returns:
            Binary predictions, shape (n,). 1 = take the trade.
        """
        primary = self._primary_fires(X)

        if not self._fitted:
            # Meta-model could not be fitted — only signal where primary fires
            return primary

        primary_mask = primary.astype(bool)
        result = np.zeros(len(X), dtype=np.int8)

        if primary_mask.sum() == 0:
            return result

        if getattr(self, "_degenerate", False):
            meta_pred = np.full(primary_mask.sum(), self._degenerate_class, dtype=np.int8)
        else:
            meta_pred = self._meta_clf.predict(X[primary_mask]).astype(np.int8)

        result[primary_mask] = meta_pred
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of meta-label given primary fires.

        For rows where primary does NOT fire, probability is forced to 0.
        For rows where primary fires, returns the meta-model's probability.

        Args:
            X: Feature matrix, shape (n, p).

        Returns:
            Probability array, shape (n, 2). Column 1 is P(execute|features).
        """
        primary = self._primary_fires(X)
        primary_mask = primary.astype(bool)
        proba = np.zeros((len(X), 2), dtype=np.float64)
        proba[:, 0] = 1.0  # Default: P(no trade) = 1

        if not self._fitted or primary_mask.sum() == 0:
            return proba

        if getattr(self, "_degenerate", False):
            p = float(self._degenerate_class)
            proba[primary_mask, 0] = 1.0 - p
            proba[primary_mask, 1] = p
        else:
            meta_proba = self._meta_clf.predict_proba(X[primary_mask])
            proba[primary_mask] = meta_proba

        return proba

    @property
    def n_primary_fires_train(self) -> int:
        """Number of primary fires in training data."""
        return self._n_primary_fires_train

    @property
    def n_meta_positive_train(self) -> int:
        """Number of positive meta-labels in training data (where primary fired)."""
        return self._n_meta_positive_train
