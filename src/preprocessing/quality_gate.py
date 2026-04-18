"""Quality gate validation for bearing signals."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks, hilbert

from .preprocess import preprocess_signal


@dataclass(frozen=True)
class QualityGateResult:
    signal: np.ndarray
    metadata: Dict[str, Any]


def init_rejection_counters() -> Dict[str, int]:
    return {
        "rpm_zero": 0,
        "nyquist_violation": 0,
    }


def _bump_counter(counters: Optional[Dict[str, int]], reason: str) -> None:
    if counters is None:
        return
    if reason not in counters:
        counters[reason] = 0
    counters[reason] += 1


def _extract_multiplier_value(source: Any, aliases: list[str]) -> Optional[float]:
    for key in aliases:
        if isinstance(source, Mapping) and key in source:
            value = source[key]
            arr = np.asarray(value).squeeze()
            if arr.size == 0:
                continue
            return float(arr)
        if hasattr(source, key):
            value = getattr(source, key)
            arr = np.asarray(value).squeeze()
            if arr.size == 0:
                continue
            return float(arr)
    return None


def _coerce_fault_multipliers(fault_mult: Any) -> Optional[Dict[str, float]]:
    mapping = {
        "BPFI": ["BPFI", "BPFIMultiple"],
        "BPFO": ["BPFO", "BPFOMultiple"],
        "BSF": ["BSF", "BPF", "BPFMultiple"],
        "FTF": ["FTF", "FTFMultiple"],
    }

    out: Dict[str, float] = {}
    for canonical, aliases in mapping.items():
        value = _extract_multiplier_value(fault_mult, aliases)
        if value is None or not np.isfinite(value):
            return None
        out[canonical] = float(value)
    return out


def quality_gate(
    signal: Any,
    rpm: float,
    fs: float,
    fault_mult: Any,
    counters: Optional[Dict[str, int]] = None,
) -> tuple[Optional[QualityGateResult], Optional[str]]:
    """
    Validate one sample before physics-based feature extraction.

    Rules:
    1) Return None if rpm == 0
    2) shaft frequency = rpm / 60
    3) fault frequencies = multiplier * shaft_frequency
    4) all fault frequencies must be < fs/2 (Nyquist)
    5) return cleaned signal + metadata when valid

    Returns
    -------
    tuple[QualityGateResult | None, str | None]
        (result, reason). On success: (QualityGateResult, None).
        On rejection: (None, reason) where reason is one of:
        - "rpm_zero"
        - "nyquist_violation"
    """
    rpm_val = float(np.asarray(rpm).squeeze())
    fs_val = float(np.asarray(fs).squeeze())

    if not np.isfinite(rpm_val) or rpm_val == 0.0:
        _bump_counter(counters, "rpm_zero")
        return None, "rpm_zero"
    if not np.isfinite(fs_val) or fs_val <= 0.0:
        _bump_counter(counters, "nyquist_violation")
        return None, "nyquist_violation"

    multipliers = _coerce_fault_multipliers(fault_mult)
    if multipliers is None:
        _bump_counter(counters, "nyquist_violation")
        return None, "nyquist_violation"

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        _bump_counter(counters, "nyquist_violation")
        return None, "nyquist_violation"

    # Basic cleaning: remove DC component.
    x_clean = x - np.mean(x)

    shaft_hz = rpm_val / 60.0
    nyquist = fs_val / 2.0

    fault_hz = {
        "BPFI": multipliers["BPFI"] * shaft_hz,
        "BPFO": multipliers["BPFO"] * shaft_hz,
        "BSF": multipliers["BSF"] * shaft_hz,
        "FTF": multipliers["FTF"] * shaft_hz,
    }

    if any((not np.isfinite(f)) or (f <= 0.0) or (f >= nyquist) for f in fault_hz.values()):
        _bump_counter(counters, "nyquist_violation")
        return None, "nyquist_violation"

    metadata = {
        "rpm": rpm_val,
        "fs": fs_val,
        "nyquist": nyquist,
        "shaft_hz": shaft_hz,
        "fault_multipliers": multipliers,
        "fault_frequencies_hz": fault_hz,
        "n_samples": int(x_clean.size),
    }
    return QualityGateResult(signal=x_clean, metadata=metadata), None


def compute_fault_peak_alignment(
    signal: Any,
    rpm: float,
    fs: float,
    fault_mult: Any,
    max_freq: Optional[float] = None,
    peak_prominence_ratio: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare expected fault frequencies against detected envelope-FFT peaks."""
    gate_result, reason = quality_gate(signal=signal, rpm=rpm, fs=fs, fault_mult=fault_mult)
    if gate_result is None:
        raise ValueError(f"Sample rejected by quality_gate with reason: {reason}")

    md = gate_result.metadata
    shaft_hz = float(md["shaft_hz"])
    expected = {
        "BPFI": float(md["fault_frequencies_hz"]["BPFI"]),
        "BPFO": float(md["fault_frequencies_hz"]["BPFO"]),
        "BSF": float(md["fault_frequencies_hz"]["BSF"]),
    }

    proc = preprocess_signal(gate_result.signal, md["fs"])
    freqs = proc["freqs"]
    fft_vals = proc["fft_vals"]

    nyquist = float(md["nyquist"])
    max_f = nyquist if max_freq is None else min(float(max_freq), nyquist)
    if max_f <= 0:
        raise ValueError("max_freq must be positive")

    mask = (freqs >= 0.0) & (freqs <= max_f)
    sel_freqs = freqs[mask]
    sel_fft = fft_vals[mask]

    if sel_freqs.size == 0:
        raise RuntimeError("No frequency bins available in selected range")

    prominence = float(np.percentile(sel_fft, 90) * peak_prominence_ratio)
    peaks, _ = find_peaks(sel_fft, prominence=prominence)
    peak_freqs = sel_freqs[peaks]
    peak_mags = sel_fft[peaks]

    if peak_freqs.size == 0:
        peak_freqs = np.array([0.0], dtype=np.float64)
        peak_mags = np.array([float(sel_fft[0])], dtype=np.float64)

    dominant_order = np.argsort(peak_mags)[::-1]
    dominant_peaks = [
        (float(peak_freqs[i]), float(peak_mags[i])) for i in dominant_order[: min(10, len(dominant_order))]
    ]

    alignment_error: Dict[str, Dict[str, float]] = {}
    for name, f_exp in expected.items():
        j = int(np.argmin(np.abs(peak_freqs - f_exp)))
        nearest_f = float(peak_freqs[j])
        err_hz = abs(nearest_f - f_exp)
        err_pct = (err_hz / max(f_exp, 1e-12)) * 100.0
        alignment_error[name] = {
            "expected_hz": float(f_exp),
            "nearest_peak_hz": nearest_f,
            "error_hz": float(err_hz),
            "error_pct": float(err_pct),
        }

    if verbose:
        print("Shaft frequency (Hz):", round(shaft_hz, 6))
        print("Expected fault frequencies (Hz):")
        for name in ["BPFI", "BPFO", "BSF"]:
            print(f"- {name}: {alignment_error[name]['expected_hz']:.6f}")
        print("Max frequency analyzed (Hz):", round(max_f, 6))
        print("Dominant FFT peaks (Hz, magnitude):")
        for f_peak, m_peak in dominant_peaks:
            print(f"- ({f_peak:.6f}, {m_peak:.6f})")
        print("Alignment error:")
        for name in ["BPFI", "BPFO", "BSF"]:
            item = alignment_error[name]
            print(
                f"- {name}: nearest={item['nearest_peak_hz']:.6f}, "
                f"err_hz={item['error_hz']:.6f}, err_pct={item['error_pct']:.6f}%"
            )

    return {
        "shaft_hz": shaft_hz,
        "expected_fault_frequencies_hz": expected,
        "dominant_peaks": dominant_peaks,
        "alignment_error": alignment_error,
        "max_frequency_analyzed_hz": float(max_f),
    }


def process_dataset_with_quality_gate(dataset_root: str | Path) -> Dict[str, Any]:
    """Run quality_gate on an entire dataset and return aggregate stats."""

    def _extract_signals(raw_data: Any) -> list[np.ndarray]:
        arr = np.asarray(raw_data)
        if arr.dtype == object:
            return [np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.reshape(-1)]
        if arr.ndim == 2:
            return [arr[i].astype(np.float64).reshape(-1) for i in range(arr.shape[0])]
        return [np.asarray(arr, dtype=np.float64).reshape(-1)]

    def _align_vector(v: Any, n: int, dtype=float) -> np.ndarray:
        a = np.asarray(v, dtype=dtype).reshape(-1)
        if a.size == n:
            return a
        if a.size == 1:
            return np.full(n, a.item(), dtype=dtype)
        if a.size > n:
            return a[:n]
        out = np.empty(n, dtype=dtype)
        out[: a.size] = a
        out[a.size :] = a[-1]
        return out

    counters = init_rejection_counters()
    total = 0
    passed = 0
    valid_rpms: list[float] = []
    invalid_pass_count = 0

    root = Path(dataset_root)
    for mat_path in sorted(root.glob("*/*.mat")):
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        for subset in ["DS", "FS", "Upper", "Lower"]:
            if subset not in mat:
                continue
            struct = mat[subset]
            signals = _extract_signals(struct.rawData)
            n = len(signals)
            rpms = _align_vector(struct.RPM, n, dtype=np.float64)
            srs = _align_vector(struct.samplingRate, n, dtype=np.float64)

            ff = struct.faultFrequencies
            fault_mult = {
                "BPFI": float(np.asarray(getattr(ff, "BPFIMultiple")).squeeze()),
                "BPFO": float(np.asarray(getattr(ff, "BPFOMultiple")).squeeze()),
                "BSF": float(np.asarray(getattr(ff, "BPFMultiple")).squeeze()),
                "FTF": float(np.asarray(getattr(ff, "FTFMultiple")).squeeze()),
            }

            for i, sig in enumerate(signals):
                total += 1
                result, reason = quality_gate(
                    signal=sig,
                    rpm=float(rpms[i]),
                    fs=float(srs[i]),
                    fault_mult=fault_mult,
                    counters=counters,
                )
                if result is None:
                    if reason not in {"rpm_zero", "nyquist_violation"}:
                        invalid_pass_count += 1
                    continue

                passed += 1
                valid_rpms.append(float(result.metadata["rpm"]))

                # Defensive integrity check: no invalid samples should pass.
                nyquist = float(result.metadata["nyquist"])
                if any(v <= 0.0 or v >= nyquist for v in result.metadata["fault_frequencies_hz"].values()):
                    invalid_pass_count += 1

    removed = total - passed
    removed_pct = (removed / total * 100.0) if total else np.nan

    stats = {
        "total_samples": total,
        "passed_samples": passed,
        "removed_samples": removed,
        "removed_pct": float(removed_pct),
        "min_rpm_after_filtering": float(np.min(valid_rpms)) if valid_rpms else np.nan,
        "max_rpm_after_filtering": float(np.max(valid_rpms)) if valid_rpms else np.nan,
        "rejection_counters": counters,
        "nyquist_violation_pct": float((counters["nyquist_violation"] / total * 100.0) if total else np.nan),
        "invalid_samples_passed": int(invalid_pass_count),
    }
    return stats
