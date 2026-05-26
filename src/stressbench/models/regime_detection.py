"""Regime detection models for stablecoin stress identification.

Three detectors:
    EWMAZScoreDetector  — EWMA mean/std z-score threshold detector
    CUSUMDetector       — Cumulative sum change detection
    BOCPDDetector       — Bayesian Online Changepoint Detection (Gaussian conjugate)

Each detector has a consistent interface:
    fit(X, y)           — Fit detector on training data
    predict(X)          — Binary 0/1 stress regime prediction
    predict_proba(X)    — Smoothed probability of stress regime

References:
    Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection.
    Cintra, R., & Holloway, T. (2023). BOCPD on Curve stablecoin pools.
    Montgomery, D. C. (2012). Introduction to Statistical Quality Control. (EWMA, CUSUM)
    Zeileis et al. (2002). strucchange: CUSUM for structural breaks in R.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_signal(X: np.ndarray, signal_col: int) -> np.ndarray:
    """Extract 1-D signal from feature matrix column or use directly if 1-D."""
    if X.ndim == 1:
        return X.astype(np.float64)
    return X[:, signal_col].astype(np.float64)


# ---------------------------------------------------------------------------
# EWMA Z-Score Detector
# ---------------------------------------------------------------------------

class EWMAZScoreDetector:
    """EWMA-based stress regime detector.

    Tracks exponentially weighted moving average (EWMA) of the signal mean
    and standard deviation. Signals stress when |z-score| > threshold.

    Args:
        span: EWMA span parameter (half-life ~ span/2 minutes).
        threshold: Z-score threshold for stress detection.
        signal_col: Column index of the signal in X (default 0).
    """

    name: str = "EWMAZScoreDetector"

    def __init__(
        self,
        span: int = 20,
        threshold: float = 3.0,
        signal_col: int = 0,
    ) -> None:
        self.span = span
        self.threshold = threshold
        self.signal_col = signal_col
        self._alpha: float = 2.0 / (span + 1)
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EWMAZScoreDetector":
        """Estimate baseline mean and std from training data.

        Args:
            X: Feature matrix or 1-D signal array.
            y: Labels (not used directly; only for API compatibility).

        Returns:
            self
        """
        signal = _extract_signal(X, self.signal_col)
        self._baseline_mean = float(np.nanmean(signal))
        self._baseline_std = float(np.nanstd(signal)) + 1e-8
        self._fitted = True
        return self

    def _ewma_zscore(self, signal: np.ndarray) -> np.ndarray:
        """Compute per-step EWMA z-score."""
        n = len(signal)
        alpha = self._alpha
        z_scores = np.zeros(n, dtype=np.float64)

        ewma_mean = self._baseline_mean
        ewma_var = self._baseline_std ** 2

        for i, x in enumerate(signal):
            ewma_mean = alpha * x + (1 - alpha) * ewma_mean
            ewma_var = alpha * (x - ewma_mean) ** 2 + (1 - alpha) * ewma_var
            ewma_std = np.sqrt(max(ewma_var, 1e-10))
            z_scores[i] = (x - ewma_mean) / ewma_std

        return z_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress regime: 1 when |z-score| > threshold.

        Args:
            X: Feature matrix or 1-D signal array.

        Returns:
            Binary predictions, shape (n,). 1 = stress regime detected.
        """
        signal = _extract_signal(X, self.signal_col)
        z_scores = self._ewma_zscore(signal)
        return (np.abs(z_scores) > self.threshold).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return smoothed probability of stress regime.

        Probability is derived from the normalized absolute z-score:
            P(stress) = sigmoid(|z| - threshold)

        Args:
            X: Feature matrix or 1-D signal array.

        Returns:
            Probability array, shape (n, 2).
        """
        signal = _extract_signal(X, self.signal_col)
        z_scores = self._ewma_zscore(signal)
        raw = np.abs(z_scores) - self.threshold
        p_stress = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p_stress, p_stress])


# ---------------------------------------------------------------------------
# CUSUM Detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """CUSUM-based structural shift detector.

    Computes cumulative sums of standardized deviations above/below the
    baseline mean. Signals stress when either cumsum exceeds a threshold.

    Args:
        k: Slack parameter (allowable deviation in units of std, default 0.5).
        h: Decision threshold (in units of std, default 5.0).
        signal_col: Column index of the signal in X (default 0).
    """

    name: str = "CUSUMDetector"

    def __init__(
        self,
        k: float = 0.5,
        h: float = 5.0,
        signal_col: int = 0,
    ) -> None:
        self.k = k
        self.h = h
        self.signal_col = signal_col
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CUSUMDetector":
        """Estimate baseline mean and std from training data."""
        signal = _extract_signal(X, self.signal_col)
        self._baseline_mean = float(np.nanmean(signal))
        self._baseline_std = float(np.nanstd(signal)) + 1e-8
        self._fitted = True
        return self

    def _cusum_scores(self, signal: np.ndarray) -> np.ndarray:
        """Compute absolute cumulative sum score for each time step."""
        mu = self._baseline_mean
        sigma = self._baseline_std
        k = self.k

        n = len(signal)
        c_pos = np.zeros(n + 1)
        c_neg = np.zeros(n + 1)
        scores = np.zeros(n, dtype=np.float64)

        for i, x in enumerate(signal):
            z = (x - mu) / sigma
            c_pos[i + 1] = max(0.0, c_pos[i] + z - k)
            c_neg[i + 1] = max(0.0, c_neg[i] - z - k)
            scores[i] = max(c_pos[i + 1], c_neg[i + 1])

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress regime: 1 when CUSUM score exceeds threshold h."""
        signal = _extract_signal(X, self.signal_col)
        scores = self._cusum_scores(signal)
        return (scores > self.h).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of stress using normalized CUSUM score."""
        signal = _extract_signal(X, self.signal_col)
        scores = self._cusum_scores(signal)
        # Normalize by threshold and apply sigmoid
        raw = (scores - self.h) / max(self.h, 1e-8)
        p_stress = 1.0 / (1.0 + np.exp(-raw * 5.0))  # sharpened sigmoid
        return np.column_stack([1.0 - p_stress, p_stress])


# ---------------------------------------------------------------------------
# BOCPD Detector
# ---------------------------------------------------------------------------

class BOCPDDetector:
    """Bayesian Online Changepoint Detection (BOCPD).

    Uses a Gaussian model with Normal-Gamma conjugate prior. Maintains a
    posterior over run lengths (time since last changepoint) and detects
    changepoints when the posterior probability of a new run-length reset
    exceeds a threshold.

    Args:
        hazard_rate: Prior probability of a changepoint at each step (lambda).
            Smaller values mean fewer expected changepoints.
        threshold: Posterior probability threshold for stress signal (default 0.5).
        signal_col: Column index of the signal in X (default 0).

    References:
        Adams & MacKay (2007). Bayesian online changepoint detection. arXiv:0710.3742.
        Cintra & Holloway (2023). BOCPD on Curve stablecoin pools.
    """

    name: str = "BOCPDDetector"

    def __init__(
        self,
        hazard_rate: float = 0.01,
        threshold: float = 0.5,
        signal_col: int = 0,
        prior_mean: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.signal_col = signal_col
        # Normal-Gamma prior parameters
        self.prior_mean = prior_mean
        self.prior_kappa = prior_kappa
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BOCPDDetector":
        """Estimate baseline statistics; set prior around training mean."""
        signal = _extract_signal(X, self.signal_col)
        self._baseline_mean = float(np.nanmean(signal))
        self._baseline_std = float(np.nanstd(signal)) + 1e-8
        # Update prior_mean from training data
        self.prior_mean = self._baseline_mean
        self._fitted = True
        return self

    def _student_t_predictive(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Student-t predictive distribution for Normal-Gamma model.

        Returns log predictive probability for each run-length hypothesis.
        """
        from scipy.special import gammaln  # type: ignore

        nu = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        # Avoid numerical issues
        scale2 = np.maximum(scale2, 1e-12)
        t_stat = (x - mu) / np.sqrt(scale2)
        log_p = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1) / 2 * np.log(1 + t_stat ** 2 / nu)
        )
        return log_p

    def run_length_proba(self, X: np.ndarray) -> list[np.ndarray]:
        """Compute posterior run-length probabilities for each time step.

        Args:
            X: Feature matrix or 1-D signal array.

        Returns:
            List of length n, where each element is an array of run-length
            probabilities (length = current time step + 1).
        """
        signal = _extract_signal(X, self.signal_col)
        n = len(signal)
        H = self.hazard_rate

        # Initialize run-length distribution: R=0 is certain at start
        log_R = np.array([0.0])  # log P(r_t | x_{1:t})
        # Normal-Gamma sufficient stats for each run-length hypothesis
        mu = np.array([self.prior_mean])
        kappa = np.array([self.prior_kappa])
        alpha = np.array([self.prior_alpha])
        beta = np.array([self.prior_beta])

        run_length_probas: list[np.ndarray] = []

        for x in signal:
            # 1. Predictive probability under each run-length hypothesis
            log_pred = self._student_t_predictive(x, mu, kappa, alpha, beta)

            # 2. Compute growth probabilities (run length grows by 1)
            log_growth = log_R + log_pred + np.log(1.0 - H)

            # 3. Compute changepoint probability (run length resets to 0)
            # Sum over all run lengths
            log_sum = np.logaddexp.reduce(log_R + log_pred)
            log_cp = log_sum + np.log(H)

            # 4. Combine: new R has length+1 entries + reset
            new_log_R = np.concatenate([[log_cp], log_growth])

            # 5. Normalize
            log_norm = np.logaddexp.reduce(new_log_R)
            new_log_R -= log_norm

            # 6. Update sufficient statistics for run-length > 0
            # For run-length 0 (changepoint), use prior
            mu_new = np.concatenate([
                [self.prior_mean],
                (kappa * mu + x) / (kappa + 1),
            ])
            kappa_new = np.concatenate([[self.prior_kappa], kappa + 1])
            alpha_new = np.concatenate([[self.prior_alpha], alpha + 0.5])
            beta_new = np.concatenate([
                [self.prior_beta],
                beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1)),
            ])

            log_R = new_log_R
            mu = mu_new
            kappa = kappa_new
            alpha = alpha_new
            beta = beta_new

            run_length_probas.append(np.exp(new_log_R))

        return run_length_probas

    def _changepoint_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute probability of being in a new regime (run-length ≤ k)."""
        rl_probas = self.run_length_proba(X)
        n = len(rl_probas)
        # P(stress) = P(run length <= short_run_threshold)
        # A short run (recently changed) indicates stress detection
        short_run_threshold = max(5, int(self.prior_kappa))
        p_stress = np.zeros(n, dtype=np.float64)
        for i, rl in enumerate(rl_probas):
            # P(run length in [0, threshold]) = sum of first k elements
            k = min(short_run_threshold + 1, len(rl))
            p_stress[i] = float(rl[:k].sum())
        return p_stress

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress regime: 1 when changepoint probability > threshold."""
        p_stress = self._changepoint_proba(X)
        return (p_stress > self.threshold).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of stress (changepoint detected).

        Args:
            X: Feature matrix or 1-D signal array.

        Returns:
            Probability array, shape (n, 2).
        """
        p_stress = self._changepoint_proba(X)
        p_stress = np.clip(p_stress, 0.0, 1.0)
        return np.column_stack([1.0 - p_stress, p_stress])
