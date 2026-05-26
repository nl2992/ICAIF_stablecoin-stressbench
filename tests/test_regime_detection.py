"""Tests for regime detection models.

Tests:
    test_ewma_detects_spike
    test_ewma_returns_to_baseline
    test_cusum_detects_shift
    test_bocpd_interface
    test_all_detectors_predict_proba_in_range
    test_regime_detection_on_constant_series
"""

from __future__ import annotations

import numpy as np
import pytest

from stressbench.models.regime_detection import (
    BOCPDDetector,
    CUSUMDetector,
    EWMAZScoreDetector,
)


def _make_stress_signal(n_calm: int = 100, n_stress: int = 50, seed: int = 42) -> tuple:
    """Generate a signal with calm period followed by stress (spike) period.

    Returns:
        signal_1d: 1-D array of shape (n_calm + n_stress,)
        y_true:    Binary stress labels
        X:         2-D array shape (n, 1) for model input
    """
    rng = np.random.RandomState(seed)
    calm = rng.randn(n_calm) * 2.0  # Low variance: N(0, 2 bps)
    stress = rng.randn(n_stress) * 15.0 + 30.0  # High variance + mean shift
    signal = np.concatenate([calm, stress])
    y_true = np.concatenate([
        np.zeros(n_calm, dtype=np.int8),
        np.ones(n_stress, dtype=np.int8),
    ])
    X = signal.reshape(-1, 1).astype(np.float32)
    return signal, y_true, X


class TestEWMADetectsSpike:
    def test_ewma_detects_spike(self):
        """EWMA detector should fire on stress period (elevated z-score)."""
        signal, y_true, X = _make_stress_signal(n_calm=100, n_stress=50)

        detector = EWMAZScoreDetector(span=20, threshold=2.5, signal_col=0)
        # Train on calm signal only
        X_train = signal[:100].reshape(-1, 1).astype(np.float32)
        y_train = np.zeros(100, dtype=np.int8)
        detector.fit(X_train, y_train)

        preds = detector.predict(X)
        stress_preds = preds[100:]  # Only stress period

        # At least some detections in stress period
        assert stress_preds.sum() > 0, (
            "EWMA should detect at least some stress in the elevated signal period"
        )


class TestEWMAReturnToBaseline:
    def test_ewma_returns_to_baseline(self):
        """EWMA detector should produce fewer alarms during calm period than stress period."""
        signal, y_true, X = _make_stress_signal(n_calm=200, n_stress=50)

        detector = EWMAZScoreDetector(span=20, threshold=2.5, signal_col=0)
        X_train = signal[:100].reshape(-1, 1).astype(np.float32)
        y_train = np.zeros(100, dtype=np.int8)
        detector.fit(X_train, y_train)

        preds = detector.predict(X)
        calm_alarms = int(preds[:200].sum())
        stress_alarms = int(preds[200:].sum())

        # Stress period should have proportionally more alarms
        stress_rate = stress_alarms / 50
        calm_rate = calm_alarms / 200

        assert stress_rate >= calm_rate, (
            f"Stress alarm rate ({stress_rate:.2f}) should be >= calm alarm rate ({calm_rate:.2f})"
        )


class TestCUSUMDetectsShift:
    def test_cusum_detects_shift(self):
        """CUSUM detector should detect mean shift in stress period."""
        signal, y_true, X = _make_stress_signal(n_calm=100, n_stress=50)

        detector = CUSUMDetector(k=0.5, h=4.0, signal_col=0)
        X_train = signal[:100].reshape(-1, 1).astype(np.float32)
        y_train = np.zeros(100, dtype=np.int8)
        detector.fit(X_train, y_train)

        preds = detector.predict(X)
        stress_preds = preds[100:]

        # CUSUM should detect the mean shift (stress has +30 bps mean)
        assert stress_preds.sum() > 0, (
            "CUSUM should detect the mean shift in the stress period"
        )

        # Test predict_proba shape consistency
        proba = detector.predict_proba(X)
        assert proba.shape == (len(X), 2)


class TestBOCPDInterface:
    def test_bocpd_interface(self):
        """BOCPD should implement fit/predict/predict_proba/run_length_proba."""
        signal, y_true, X = _make_stress_signal(n_calm=50, n_stress=20)

        detector = BOCPDDetector(hazard_rate=0.01, threshold=0.5, signal_col=0)
        X_train = signal[:50].reshape(-1, 1).astype(np.float32)
        y_train = np.zeros(50, dtype=np.int8)

        # fit
        result = detector.fit(X_train, y_train)
        assert result is detector, "fit should return self"

        # predict
        preds = detector.predict(X)
        assert preds.shape == (len(X),), f"Expected shape ({len(X)},), got {preds.shape}"
        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary"

        # predict_proba
        proba = detector.predict_proba(X)
        assert proba.shape == (len(X), 2), f"Expected shape ({len(X)}, 2), got {proba.shape}"

        # run_length_proba
        rl = detector.run_length_proba(X)
        assert len(rl) == len(X), f"run_length_proba should return {len(X)} arrays"
        # Each run-length distribution should be a valid probability distribution
        for i, rl_i in enumerate(rl):
            total = float(np.sum(rl_i))
            assert abs(total - 1.0) < 0.01, f"Run length proba at step {i} should sum to ~1, got {total}"


class TestAllDetectorsPredictProbaInRange:
    def test_all_detectors_predict_proba_in_range(self):
        """All detectors' predict_proba should return values in [0, 1]."""
        signal, y_true, X = _make_stress_signal(n_calm=80, n_stress=40)
        X_train = signal[:80].reshape(-1, 1).astype(np.float32)
        y_train = np.zeros(80, dtype=np.int8)

        detectors = [
            EWMAZScoreDetector(span=20, threshold=3.0, signal_col=0),
            CUSUMDetector(k=0.5, h=5.0, signal_col=0),
            BOCPDDetector(hazard_rate=0.05, threshold=0.5, signal_col=0),
        ]

        for detector in detectors:
            detector.fit(X_train, y_train)
            proba = detector.predict_proba(X)

            assert proba.shape == (len(X), 2), (
                f"{detector.name}: Expected shape ({len(X)}, 2), got {proba.shape}"
            )
            assert np.all(proba >= 0.0), f"{detector.name}: Probabilities must be >= 0"
            assert np.all(proba <= 1.0), f"{detector.name}: Probabilities must be <= 1"

            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, np.ones(len(X)), atol=1e-4,
                err_msg=f"{detector.name}: Probability rows must sum to 1",
            )


class TestRegimeDetectionOnConstantSeries:
    def test_regime_detection_on_constant_series(self):
        """All detectors should handle a constant signal without crashing."""
        n = 50
        constant_signal = np.ones(n, dtype=np.float32) * 5.0  # Constant 5 bps
        X = constant_signal.reshape(-1, 1)
        y = np.zeros(n, dtype=np.int8)

        detectors = [
            EWMAZScoreDetector(span=20, threshold=3.0, signal_col=0),
            CUSUMDetector(k=0.5, h=5.0, signal_col=0),
            BOCPDDetector(hazard_rate=0.01, threshold=0.5, signal_col=0),
        ]

        for detector in detectors:
            # Fit on constant
            detector.fit(X[:30], y[:30])

            # Predict should not raise
            preds = detector.predict(X)
            assert preds.shape == (n,), f"{detector.name}: Wrong prediction shape"

            # Proba should be valid
            proba = detector.predict_proba(X)
            assert proba.shape == (n, 2), f"{detector.name}: Wrong proba shape"
            assert np.all(np.isfinite(proba)), f"{detector.name}: Non-finite probabilities on constant series"

            # On a constant series, no dramatic alarms expected
            n_alarms = int(preds.sum())
            assert n_alarms <= n // 2, (
                f"{detector.name}: Too many alarms ({n_alarms}/{n}) on constant series"
            )
