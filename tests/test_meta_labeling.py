"""Tests for MetaLabelingFilter.

Tests:
    test_meta_labeling_only_fits_where_primary_fires
    test_meta_labeling_predict_zero_when_primary_silent
    test_meta_labeling_predict_proba_shape
    test_meta_labeling_fit_predict_roundtrip
    test_meta_labeling_min_trades_constraint
    test_meta_labeling_handles_no_primary_fires_in_split
"""

from __future__ import annotations

import numpy as np
import pytest

from stressbench.models.meta_labeling import MetaLabelingFilter


def _make_data(
    n: int, primary_rate: float = 0.4, meta_pos_rate: float = 0.3, seed: int = 42
) -> tuple:
    """Generate synthetic (X, y_primary, y_meta) data.

    X[:, 0] = basis signal (positive = large basis, negative = small basis)
    y_primary = 1 where |basis| > threshold (10 bps)
    y_meta = 1 where primary fires AND trade is profitable
    """
    rng = np.random.RandomState(seed)
    # Basis values: some large (primary fires), some small
    basis = rng.randn(n) * 20  # ~ N(0, 20 bps)
    other_features = rng.randn(n, 4)
    X = np.column_stack([basis, other_features]).astype(np.float32)

    threshold = 10.0
    y_primary = (np.abs(basis) > threshold).astype(np.int8)

    # Meta label: fires randomly with meta_pos_rate among primary fires
    y_meta = np.zeros(n, dtype=np.int8)
    primary_mask = y_primary.astype(bool)
    n_fires = primary_mask.sum()
    if n_fires > 0:
        positive_among_fires = rng.binomial(1, meta_pos_rate, n_fires).astype(np.int8)
        y_meta[primary_mask] = positive_among_fires

    return X, y_primary, y_meta


class TestMetaLabelingOnlyFitsWherePrimaryFires:
    def test_meta_labeling_only_fits_where_primary_fires(self):
        """Meta-model should be trained only on rows where primary fires."""
        X, y_primary, y_meta = _make_data(300, primary_rate=0.4)

        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta)

        # n_primary_fires_train should equal sum of y_primary
        assert model.n_primary_fires_train == int(
            y_primary.sum()
        ), f"Expected {int(y_primary.sum())} fires, got {model.n_primary_fires_train}"
        # n_meta_positive_train <= n_primary_fires_train
        assert model.n_meta_positive_train <= model.n_primary_fires_train


class TestMetaLabelingPredictZeroWhenPrimarySilent:
    def test_meta_labeling_predict_zero_when_primary_silent(self):
        """Predict = 0 for all rows where primary signal does not fire."""
        X, y_primary, y_meta = _make_data(200, seed=0)
        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta)

        # Create test data where basis is small (primary never fires)
        X_silent = X.copy()
        X_silent[:, 0] = 1.0  # 1 bps << 10 bps threshold

        predictions = model.predict(X_silent)
        assert np.all(
            predictions == 0
        ), f"Expected all zeros when primary silent, got {predictions.sum()} non-zero"


class TestMetaLabelingPredictProbaShape:
    def test_meta_labeling_predict_proba_shape(self):
        """predict_proba should return (n, 2) with values in [0, 1]."""
        X, y_primary, y_meta = _make_data(150, seed=7)
        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta)

        proba = model.predict_proba(X)
        assert proba.shape == (
            len(X),
            2,
        ), f"Expected shape ({len(X)}, 2), got {proba.shape}"
        assert np.all(proba >= 0.0), "Probabilities must be >= 0"
        assert np.all(proba <= 1.0), "Probabilities must be <= 1"
        # Rows should sum to 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(X)), atol=1e-5)


class TestMetaLabelingFitPredictRoundtrip:
    def test_meta_labeling_fit_predict_roundtrip(self):
        """fit then predict should be consistent with predict_proba threshold."""
        X, y_primary, y_meta = _make_data(400, seed=42)
        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta)

        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        # All predictions of 1 should have been triggered by primary signal
        primary = (np.abs(X[:, 0]) > 10.0).astype(np.int8)
        # Where preds == 1, primary must also be 1
        assert np.all(
            primary[preds.astype(bool)] == 1
        ), "All predictions of 1 must have primary signal = 1"

        # Where primary == 0, predict must be 0
        assert np.all(
            preds[~primary.astype(bool)] == 0
        ), "All non-primary-fire rows must have prediction = 0"

        # predict_proba[:, 0] where primary = 0 should be 1.0
        np.testing.assert_allclose(
            proba[~primary.astype(bool)],
            0.0,
            atol=1e-5,
            err_msg="P(trade) must be 0 where primary does not fire",
        )


class TestMetaLabelingMinTradesConstraint:
    def test_meta_labeling_min_trades_constraint(self):
        """Model should remain fitted and produce valid predictions even with imbalanced data."""
        # Very few positives
        X, y_primary, _ = _make_data(100, primary_rate=0.3, seed=99)
        y_meta_sparse = np.zeros(len(X), dtype=np.int8)
        # Only 3 positives among primary fires
        primary_idxs = np.where(y_primary)[0]
        if len(primary_idxs) >= 3:
            y_meta_sparse[primary_idxs[:3]] = 1

        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta_sparse)

        # Model should have been fitted (no exception)
        preds = model.predict(X)
        proba = model.predict_proba(X)

        # Predictions must be binary
        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be 0 or 1"
        # Proba shape
        assert proba.shape == (len(X), 2)


class TestMetaLabelingHandlesNoPrimaryFiresInSplit:
    def test_meta_labeling_handles_no_primary_fires_in_split(self):
        """fit should handle the case where no primary fires exist in training data."""
        n = 100
        rng = np.random.RandomState(5)
        # Basis values all small — no primary fires
        X = np.column_stack(
            [
                rng.uniform(-5, 5, n),  # basis << 10 bps
                rng.randn(n, 3),
            ]
        ).astype(np.float32)
        y_primary = np.zeros(n, dtype=np.int8)  # no fires
        y_meta = np.zeros(n, dtype=np.int8)

        model = MetaLabelingFilter(primary_threshold_bps=10.0, primary_signal_col=0)
        model.fit(X, y_primary, y_meta)  # Should not raise

        assert model.n_primary_fires_train == 0
        assert model._fitted is False  # Cannot fit meta-model with no fires

        # Predict should still work (returns all zeros)
        preds = model.predict(X)
        assert np.all(preds == 0), "All predictions must be 0 when no primary fires"

        proba = model.predict_proba(X)
        assert proba.shape == (n, 2)
        np.testing.assert_allclose(proba[:, 1], 0.0, atol=1e-5)
