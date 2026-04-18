"""Feature extraction from envelope domain data."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis

from ..preprocessing.quality_gate import _coerce_fault_multipliers


def extract_features(
    envelope: any,
    fft_vals: any,
    freqs: any,
    rpm: float,
    fault_mult: any,
) -> np.ndarray:
    """
    Build physics-guided feature vector from envelope-domain inputs.

    Features: [rms, kurtosis, Ew_bpfi, Ew_bpfo, Ew_bsf]

    Parameters
    ----------
    envelope
        Envelope signal.
    fft_vals
        Magnitude spectrum from FFT.
    freqs
        Frequency axis.
    rpm
        Shaft speed in RPM.
    fault_mult
        Fault multipliers dict or object.

    Returns
    -------
    np.ndarray
        1D feature vector of shape (5,).
    """
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    spec = np.asarray(fft_vals, dtype=np.float64).reshape(-1)
    faxis = np.asarray(freqs, dtype=np.float64).reshape(-1)

    if env.size == 0:
        raise ValueError("envelope must contain at least one sample")
    if spec.size == 0 or faxis.size == 0:
        raise ValueError("fft_vals and freqs must contain at least one sample")
    if spec.shape != faxis.shape:
        raise ValueError("fft_vals and freqs must have the same shape")

    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    faxis = np.nan_to_num(faxis, nan=0.0, posinf=0.0, neginf=0.0)

    rms = float(np.log1p(np.sqrt(np.mean(env ** 2))))
    k = float(kurtosis(env, bias=False)) if env.size > 3 else 0.0

    rpm_val = float(np.asarray(rpm).squeeze())
    shaft_hz = rpm_val / 60.0 if np.isfinite(rpm_val) else np.nan
    multipliers = _coerce_fault_multipliers(fault_mult)
    if multipliers is None or (not np.isfinite(shaft_hz)):
        raise ValueError("fault_mult/rpm are invalid for fault-frequency computation")

    f_bpfi = multipliers["BPFI"] * shaft_hz
    f_bpfo = multipliers["BPFO"] * shaft_hz
    f_bsf = multipliers["BSF"] * shaft_hz

    power = np.abs(spec) ** 2
    nyquist = float(np.max(faxis))

    def get_energy(center_hz: float, half_band_hz: float = 0.5) -> float:
        """Return raw band energy around a frequency."""
        if not np.isfinite(center_hz) or center_hz <= 0.0 or center_hz > nyquist:
            return 0.0
        mask = (faxis >= center_hz - half_band_hz) & (faxis <= center_hz + half_band_hz)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    def get_weighted_harmonic_energy(center_hz: float, half_band_hz: float = 0.5) -> float:
        """Weighted harmonic sum: E(f) + 0.5*E(2f) + 0.25*E(3f), then log1p."""
        if not np.isfinite(center_hz) or center_hz <= 0.0:
            return 0.0
        weighted_power = 0.0
        for harmonic_idx, weight in ((1, 1.0), (2, 0.5), (3, 0.25)):
            harmonic_hz = harmonic_idx * center_hz
            if not np.isfinite(harmonic_hz) or harmonic_hz <= 0.0 or harmonic_hz > nyquist:
                continue
            weighted_power += weight * get_energy(harmonic_hz, half_band_hz=half_band_hz)
        return float(np.log1p(weighted_power))

    ew_bpfi = get_weighted_harmonic_energy(f_bpfi)
    ew_bpfo = get_weighted_harmonic_energy(f_bpfo)
    ew_bsf = get_weighted_harmonic_energy(f_bsf)

    return np.array(
        [
            rms,
            k,
            ew_bpfi,
            ew_bpfo,
            ew_bsf,
        ],
        dtype=np.float64,
    )


def _fft_energy_around_exact(signal: np.ndarray, fs: float, target_hz: float, bandwidth_hz: float = 3.0) -> float:
    """Exact band-energy computation used in sample_level_features generation."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / float(fs))
    mask = (freq >= float(target_hz) - float(bandwidth_hz)) & (freq <= float(target_hz) + float(bandwidth_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask]))


def extract_features_csv_exact(
    signal: any,
    fs: float,
    rpm: float,
    fault_mult: any,
) -> np.ndarray:
    """Reproduce exact CSV feature computation: [rms, kurtosis, E_bpfi, E_bpfo, E_bsf]."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("signal must contain at least one sample")

    fs_val = float(np.asarray(fs).squeeze())
    rpm_val = float(np.asarray(rpm).squeeze())
    if (not np.isfinite(fs_val)) or fs_val <= 0.0:
        raise ValueError("fs must be a positive finite scalar")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    rms = float(np.sqrt(np.mean(x ** 2)))
    k = float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else float("nan")

    shaft_hz = rpm_val / 60.0 if np.isfinite(rpm_val) and rpm_val > 0 else np.nan
    multipliers = _coerce_fault_multipliers(fault_mult)
    if multipliers is None or (not np.isfinite(shaft_hz)):
        e_bpfi = 0.0
        e_bpfo = 0.0
        e_bsf = 0.0
    else:
        bpfi_hz = multipliers["BPFI"] * shaft_hz
        bpfo_hz = multipliers["BPFO"] * shaft_hz
        bsf_hz = multipliers["BSF"] * shaft_hz

        e_bpfi = _fft_energy_around_exact(x, fs_val, bpfi_hz) if np.isfinite(bpfi_hz) else 0.0
        e_bpfo = _fft_energy_around_exact(x, fs_val, bpfo_hz) if np.isfinite(bpfo_hz) else 0.0
        e_bsf = _fft_energy_around_exact(x, fs_val, bsf_hz) if np.isfinite(bsf_hz) else 0.0

    return np.array([rms, k, e_bpfi, e_bpfo, e_bsf], dtype=np.float64)
