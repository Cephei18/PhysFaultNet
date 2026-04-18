"""Signal envelope and spectrum preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def preprocess_signal(signal: any, fs: float) -> dict[str, np.ndarray]:
    """
    Envelope and envelope-spectrum preprocessing.

    Steps:
    1) analytic signal via Hilbert transform
    2) envelope = abs(analytic_signal)
    3) FFT of envelope with np.fft.rfft
    4) frequency axis via np.fft.rfftfreq
    5) magnitude spectrum via abs
    """
    fs_val = float(np.asarray(fs).squeeze())
    if not np.isfinite(fs_val) or fs_val <= 0.0:
        raise ValueError("fs must be a positive finite scalar")

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("signal must contain at least one sample")

    # Keep processing stable with non-finite values.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)

    analytic = hilbert(x)
    envelope = np.abs(analytic)

    # Remove envelope DC before FFT for a more informative spectrum.
    env_for_fft = envelope - np.mean(envelope)
    fft_vals = np.abs(np.fft.rfft(env_for_fft))
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / fs_val)

    if fft_vals.shape != freqs.shape:
        raise RuntimeError("shape mismatch between fft_vals and freqs")

    return {
        "envelope": envelope,
        "fft_vals": fft_vals,
        "freqs": freqs,
    }
