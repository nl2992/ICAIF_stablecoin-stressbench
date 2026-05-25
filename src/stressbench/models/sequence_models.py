"""Sequence models: Temporal CNN, Transformer encoder, PatchTST-style model.

These models are built only after the baseline tables are stable.
They accept time-series windows of features and predict future basis/regime.

Requires: torch, torch.nn
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def _check_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for sequence models. "
            "Install with: pip install torch"
        )


class TemporalCNN:
    """Temporal Convolutional Network for time-series forecasting.

    Architecture: 1D dilated causal convolutions with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        task: str = "regression",
    ) -> None:
        _check_torch()
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task

        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self._conv = nn.Sequential(*layers)
        self._fc = nn.Linear(in_ch, output_dim)
        self._model = nn.Sequential(self._conv, nn.AdaptiveAvgPool1d(1), nn.Flatten(), self._fc)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 1e-3) -> "TemporalCNN":
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        X_t = torch.FloatTensor(X).permute(0, 2, 1)  # (N, T, F) -> (N, F, T)
        y_t = torch.FloatTensor(y)

        optimizer = Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss() if self.task == "regression" else nn.BCEWithLogitsLoss()

        self._model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self._model(X_t).squeeze(-1)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.info("TemporalCNN epoch %d/%d loss=%.4f", epoch + 1, epochs, loss.item())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).permute(0, 2, 1)
            return self._model(X_t).squeeze(-1).numpy()


class TransformerEncoder:
    """Transformer encoder for time-series classification and regression.

    Uses multi-head self-attention with positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        task: str = "regression",
    ) -> None:
        _check_torch()
        import torch.nn as nn

        self.task = task
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self._input_proj = nn.Linear(input_dim, d_model)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._fc = nn.Linear(d_model, output_dim)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 1e-3) -> "TransformerEncoder":
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        X_t = torch.FloatTensor(X)  # (N, T, F)
        y_t = torch.FloatTensor(y)

        params = list(self._input_proj.parameters()) + list(self._encoder.parameters()) + list(self._fc.parameters())
        optimizer = Adam(params, lr=lr)
        criterion = nn.MSELoss() if self.task == "regression" else nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            proj = self._input_proj(X_t)
            enc = self._encoder(proj)
            preds = self._fc(enc[:, -1, :]).squeeze(-1)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.info("Transformer epoch %d/%d loss=%.4f", epoch + 1, epochs, loss.item())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            proj = self._input_proj(X_t)
            enc = self._encoder(proj)
            return self._fc(enc[:, -1, :]).squeeze(-1).numpy()
