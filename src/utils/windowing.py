"""Sliding window utilities."""
from __future__ import annotations

import numpy as np


def create_windows(envelope: any, window_size: int, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from a 1D signal.

    X[i] = envelope[i : i + window_size]
    Y[i] = envelope[i + horizon : i + window_size + horizon]

    Returns arrays with shape (num_windows, window_size).
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    x = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if x.size <= window_size + horizon - 1:
        return (
            np.empty((0, window_size), dtype=np.float64),
            np.empty((0, window_size), dtype=np.float64),
        )

    num_windows = x.size - window_size - horizon + 1
    X = np.empty((num_windows, window_size), dtype=np.float64)
    Y = np.empty((num_windows, window_size), dtype=np.float64)

    for i in range(num_windows):
        X[i] = x[i : i + window_size]
        Y[i] = x[i + 1 : i + window_size + 1]

    return X, Y
