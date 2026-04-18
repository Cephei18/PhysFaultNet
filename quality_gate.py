from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks, hilbert
from scipy.stats import kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


@dataclass(frozen=True)
class QualityGateResult:
    signal: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class OneClassSVMResult:
    scaler: RobustScaler
    model: OneClassSVM
    decision_scores: np.ndarray
    anomaly_scores: np.ndarray


@dataclass(frozen=True)
class OneClassSVMSearchResult:
    best_nu: float
    best_result: OneClassSVMResult
    results_by_nu: Dict[float, Dict[str, Any]]
    selection_metric: str


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
    # Canonical keys used by downstream code.
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


def preprocess_signal(signal: Any, fs: float) -> Dict[str, np.ndarray]:
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


def create_windows(envelope: Any, window_size: int, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
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


def compute_fault_peak_alignment(
    signal: Any,
    rpm: float,
    fs: float,
    fault_mult: Any,
    max_freq: Optional[float] = None,
    peak_prominence_ratio: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    For one sample, compare expected fault frequencies against detected envelope-FFT peaks.

    Returns a dictionary with expected frequencies, detected peaks, and alignment errors.
    """
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


def extract_features(
    envelope: Any,
    fft_vals: Any,
    freqs: Any,
    rpm: float,
    fault_mult: Any,
) -> np.ndarray:
    """
    Build physics-guided feature vector from envelope-domain inputs.

    Returns:
    [rms, kurtosis, Ew_bpfi, Ew_bpfo, Ew_bsf]
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
        """Return raw band energy around a frequency, with empty/out-of-band bands mapped to 0."""
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

    Parameters
    ----------
    signal
        1D numeric signal array-like.
    rpm
        Shaft speed in revolutions per minute.
    fs
        Sampling frequency in Hz.
    fault_mult
        Fault multipliers. Accepts either:
        - dict with keys like BPFI/BPFO/BSF/FTF or *Multiple variants
        - object with attributes like BPFIMultiple/BPFOMultiple/BPFMultiple/FTFMultiple

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


def process_dataset_with_quality_gate(dataset_root: str | Path) -> Dict[str, Any]:
    """
    Run quality_gate on an entire dataset and return aggregate stats.
    """

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


def _validate_feature_inputs(feature_matrix: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(feature_matrix, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)

    if x.ndim != 2:
        raise ValueError("feature_matrix must be a 2D array of shape (n_samples, n_features)")
    if y.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_matrix and labels must have the same number of samples")
    if x.shape[0] == 0:
        raise ValueError("feature_matrix must contain at least one sample")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


def _preprocess_ocsvm_features(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    healthy_label: int = 0,
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Preprocess OCSVM features with:
    1) log1p on RMS column (column 0)
    2) RobustScaler fit on healthy samples only
    3) transform all samples using the healthy-fitted scaler
    """
    x_proc = np.array(feature_matrix, dtype=np.float64, copy=True)

    # Column 0 is RMS; clip negatives to 0 for safe log1p behavior.
    rms = np.maximum(x_proc[:, 0], 0.0)
    x_proc[:, 0] = np.log1p(rms)

    healthy_mask = labels == healthy_label
    if not np.any(healthy_mask):
        raise ValueError("no healthy samples found for training")

    x_train_healthy = x_proc[healthy_mask]

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train_healthy)
    x_all_scaled = scaler.transform(x_proc)
    return x_train_scaled, x_all_scaled, scaler


def _preprocess_final_ocsvm_features(
    feature_matrix: Any,
    labels: Any,
    healthy_label: int = 0,
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Preprocess final 5D feature vectors using RobustScaler fit only on healthy samples.
    """
    x, y = _validate_feature_inputs(feature_matrix, labels)
    healthy_mask = y == healthy_label
    if not np.any(healthy_mask):
        raise ValueError("no healthy samples found for training")

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(np.array(x[healthy_mask], dtype=np.float64, copy=True))
    x_all_scaled = scaler.transform(np.array(x, dtype=np.float64, copy=True))
    return x_train_scaled, x_all_scaled, scaler


def train_ocsvm(X_train: Any, nu: float) -> OneClassSVM:
    """
    Train an RBF One-Class SVM on preprocessed healthy-only data.
    """
    x_train = np.asarray(X_train, dtype=np.float64)
    if x_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if x_train.shape[0] == 0:
        raise ValueError("X_train must contain at least one sample")
    if not np.isfinite(nu) or nu <= 0.0 or nu >= 1.0:
        raise ValueError("nu must be in the open interval (0, 1)")

    model = OneClassSVM(kernel="rbf", nu=float(nu), gamma="scale")
    model.fit(x_train)
    return model


def one_class_svm_pipeline(
    feature_matrix: Any,
    labels: Any,
    healthy_label: int = 0,
    nu: float = 0.05,
) -> OneClassSVMResult:
    """
    Train an One-Class SVM using only healthy samples, then score all samples.

    Steps:
    1) use healthy samples for training
    2) fit RobustScaler on healthy train only and transform all samples
    3) train OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    4) compute decision_function scores for all samples
    5) anomaly_scores = -decision_scores
    """
    x, y = _validate_feature_inputs(feature_matrix, labels)

    x_train_scaled, x_all_scaled, scaler = _preprocess_ocsvm_features(
        feature_matrix=x,
        labels=y,
        healthy_label=healthy_label,
    )

    model = train_ocsvm(x_train_scaled, nu=nu)

    decision_scores = model.decision_function(x_all_scaled).reshape(-1)
    anomaly_scores = -decision_scores

    return OneClassSVMResult(
        scaler=scaler,
        model=model,
        decision_scores=decision_scores,
        anomaly_scores=anomaly_scores,
    )


def sweep_ocsvm_nu(
    feature_matrix: Any,
    labels: Any,
    nu_values: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05),
    healthy_label: int = 0,
    selection_metric: str = "roc_auc",
) -> OneClassSVMSearchResult:
    """
    Train and evaluate One-Class SVM across multiple nu values.

    Returns the best model according to selection_metric, with ROC-AUC used by default.
    """
    x, y = _validate_feature_inputs(feature_matrix, labels)
    if selection_metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("selection_metric must be 'roc_auc' or 'pr_auc'")
    if len(nu_values) == 0:
        raise ValueError("nu_values must contain at least one value")

    x_train_scaled, x_all_scaled, scaler = _preprocess_ocsvm_features(
        feature_matrix=x,
        labels=y,
        healthy_label=healthy_label,
    )

    y_true = (y != healthy_label).astype(int)
    results_by_nu: Dict[float, Dict[str, Any]] = {}
    best_nu: Optional[float] = None
    best_score: Optional[float] = None
    best_result: Optional[OneClassSVMResult] = None

    for nu in nu_values:
        model = train_ocsvm(x_train_scaled, nu=nu)
        decision_scores = model.decision_function(x_all_scaled).reshape(-1)
        anomaly_scores = -decision_scores

        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            roc_auc = float(roc_auc_score(y_true, anomaly_scores))
            pr_auc = float(average_precision_score(y_true, anomaly_scores))
        except Exception as exc:
            raise RuntimeError(f"failed to compute metrics for nu={nu}: {exc}") from exc

        result = OneClassSVMResult(
            scaler=scaler,
            model=model,
            decision_scores=decision_scores,
            anomaly_scores=anomaly_scores,
        )

        results_by_nu[float(nu)] = {
            "result": result,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        metric_value = roc_auc if selection_metric == "roc_auc" else pr_auc
        if best_score is None or metric_value > best_score:
            best_score = metric_value
            best_nu = float(nu)
            best_result = result

    if best_nu is None or best_result is None:
        raise RuntimeError("nu sweep did not produce a valid result")

    return OneClassSVMSearchResult(
        best_nu=best_nu,
        best_result=best_result,
        results_by_nu=results_by_nu,
        selection_metric=selection_metric,
    )


def final_ocsvm_pipeline(
    feature_matrix: Any,
    labels: Any,
    healthy_label: int = 0,
    nu: float = 0.03,
) -> tuple[np.ndarray, RobustScaler, OneClassSVM]:
    """
    Final OCSVM pipeline for the final 5D feature set.

    Steps:
    1) X_train = X[labels == healthy_label]
    2) RobustScaler fit on healthy only, transform train and all samples
    3) Train OneClassSVM(kernel='rbf', nu=0.03, gamma='scale') on healthy-scaled data
    4) Score all samples with negative decision_function

    Returns
    -------
    scores : np.ndarray
        Anomaly scores for all samples.
    scaler : RobustScaler
        Fitted scaler.
    model : OneClassSVM
        Trained One-Class SVM model.
    """
    x_train_scaled, x_all_scaled, scaler = _preprocess_final_ocsvm_features(
        feature_matrix=feature_matrix,
        labels=labels,
        healthy_label=healthy_label,
    )

    model = train_ocsvm(x_train_scaled, nu=nu)
    scores = -model.decision_function(x_all_scaled).reshape(-1)
    return scores, scaler, model


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
