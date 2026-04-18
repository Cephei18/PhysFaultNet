"""Temporal models for anomaly detection."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:

    class WindowCNNPredictor(nn.Module):
        """Minimal 1D CNN predictor for window-to-window sequence modeling."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 1, kernel_size=3, padding=1),
            )

        def forward(self, x):
            if x.ndim != 2:
                raise ValueError("input must have shape (batch, window_size)")
            x = x.unsqueeze(1)
            y = self.net(x)
            return y.squeeze(1)

else:

    class WindowCNNPredictor:  # pragma: no cover - requires torch
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for WindowCNNPredictor but is not installed")
