"""Dataset loading and collection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis

from ..preprocessing import preprocess_signal, quality_gate
from ..features import extract_features


@dataclass
class SampleRecord:
    sample_id: str
    folder_id: str
    split: str
    subset: str
    file_path: str
    sample_idx: int
    sampling_rate: float
    rpm: float
    label: int
    signal_length: int
    fault_ftf_mult: float
    fault_bsf_mult: float
    fault_bpfo_mult: float
    fault_bpfi_mult: float
    rms: float
    kurtosis: float
    skewness: float
    envelope_kurtosis: float
    E_bpfi: float
    E_bpfo: float
    E_bsf: float
    nan_count: int
    inf_count: int
    signal_hash: str


def _safe_float(x) -> float:
    """Safely convert to float, handling nested arrays."""
    arr = np.asarray(x).squeeze()
    return float(arr) if arr.size > 0 else np.nan


def _mat_fault_multipliers(fault_frequencies) -> Dict[str, float]:
    """Extract fault multipliers from matfile structure."""
    return {
        "FTF": _safe_float(fault_frequencies.FTFMultiple),
        "BSF": _safe_float(fault_frequencies.BPFMultiple),
        "BPFO": _safe_float(fault_frequencies.BPFOMultiple),
        "BPFI": _safe_float(fault_frequencies.BPFIMultiple),
    }


def _signal_hash(signal: np.ndarray) -> str:
    """Compute hash of signal for integrity checks."""
    return hashlib.sha256(signal.tobytes()).hexdigest()[:16]


def _fft_energy_around(freqs: np.ndarray, mag: np.ndarray, center_hz: float, half_band: float = 0.5) -> float:
    """Compute energy in band around center frequency."""
    if not np.isfinite(center_hz) or center_hz <= 0:
        return 0.0
    mask = (freqs >= center_hz - half_band) & (freqs <= center_hz + half_band)
    if not np.any(mask):
        return 0.0
    power = np.abs(mag[mask]) ** 2
    return float(np.log1p(np.sum(power)))


def _envelope(signal: np.ndarray) -> np.ndarray:
    """Compute envelope via Hilbert transform."""
    analytic = hilbert(signal)
    return np.abs(analytic)


def _envelope_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute envelope spectrum."""
    env = _envelope(signal)
    env = env - np.mean(env)
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, d=1.0 / fs)
    return freqs, mag


def _extract_signals(raw_data) -> List[np.ndarray]:
    """Extract individual signals from matfile data."""
    arr = np.asarray(raw_data)
    if arr.dtype == object:
        return [np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.reshape(-1)]
    if arr.ndim == 2:
        return [arr[i].astype(np.float64).reshape(-1) for i in range(arr.shape[0])]
    return [np.asarray(arr, dtype=np.float64).reshape(-1)]


def _align_vector(v, n: int, dtype=float) -> np.ndarray:
    """Align vector to length n by broadcasting or truncation."""
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


def collect_dataset_records(dataset_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and extract features from all samples in the SCA bearing dataset."""
    records: List[SampleRecord] = []
    file_summaries: List[Dict] = []

    root = Path(dataset_root)
    for mat_path in sorted(root.glob("*/*.mat")):
        folder_id = mat_path.parent.name
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        for subset in ["DS", "FS", "Upper", "Lower"]:
            if subset not in mat:
                continue

            struct = mat[subset]
            signals = _extract_signals(struct.rawData)
            n = len(signals)
            rpms = _align_vector(struct.RPM, n, dtype=np.float64)
            srs = _align_vector(struct.samplingRate, n, dtype=np.float64)
            multipliers = _mat_fault_multipliers(struct.faultFrequencies)

            subset_idx = 0
            for i, signal in enumerate(signals):
                signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                signal = signal - np.mean(signal)

                rpm = float(rpms[i])
                fs = float(srs[i])

                fault_mult = multipliers
                qg_result, qg_reason = quality_gate(
                    signal=signal,
                    rpm=rpm,
                    fs=fs,
                    fault_mult=fault_mult,
                )

                if qg_result is None:
                    label = -1
                    features = np.full(5, np.nan)
                else:
                    label = 0
                    proc = preprocess_signal(qg_result.signal, fs)
                    features = extract_features(
                        envelope=proc["envelope"],
                        fft_vals=proc["fft_vals"],
                        freqs=proc["freqs"],
                        rpm=rpm,
                        fault_mult=fault_mult,
                    )

                freqs, mag = _envelope_spectrum(signal, fs)
                shaft_hz = rpm / 60.0
                e_bpfi = _fft_energy_around(freqs, mag, multipliers["BPFI"] * shaft_hz) if np.isfinite(shaft_hz) else 0.0
                e_bpfo = _fft_energy_around(freqs, mag, multipliers["BPFO"] * shaft_hz) if np.isfinite(shaft_hz) else 0.0
                e_bsf = _fft_energy_around(freqs, mag, multipliers["BSF"] * shaft_hz) if np.isfinite(shaft_hz) else 0.0

                sample_id = f"{folder_id}_{subset}_{subset_idx}"
                record = SampleRecord(
                    sample_id=sample_id,
                    folder_id=folder_id,
                    split="train" if int(folder_id) <= 8 else "test",
                    subset=subset,
                    file_path=str(mat_path),
                    sample_idx=int(i),
                    sampling_rate=fs,
                    rpm=rpm,
                    label=int(label),
                    signal_length=int(signal.size),
                    fault_ftf_mult=float(multipliers["FTF"]),
                    fault_bsf_mult=float(multipliers["BSF"]),
                    fault_bpfo_mult=float(multipliers["BPFO"]),
                    fault_bpfi_mult=float(multipliers["BPFI"]),
                    rms=float(features[0]) if len(features) > 0 else np.nan,
                    kurtosis=float(features[1]) if len(features) > 1 else np.nan,
                    skewness=float(np.sqrt(np.mean((signal - np.mean(signal)) ** 3))),
                    envelope_kurtosis=float(kurtosis(_envelope(signal), bias=False)) if signal.size > 3 else np.nan,
                    E_bpfi=e_bpfi,
                    E_bpfo=e_bpfo,
                    E_bsf=e_bsf,
                    nan_count=int(np.isnan(signal).sum()),
                    inf_count=int(np.isinf(signal).sum()),
                    signal_hash=_signal_hash(signal),
                )
                records.append(record)
                subset_idx += 1

    records_df = pd.DataFrame([vars(r) for r in records])

    file_summaries = []
    for folder_id in records_df["folder_id"].unique():
        subset_df = records_df[records_df["folder_id"] == folder_id]
        file_summaries.append({
            "folder_id": folder_id,
            "n_samples": int(len(subset_df)),
            "n_healthy": int((subset_df["label"] == 0).sum()),
            "n_faulty": int((subset_df["label"] == -1).sum()),
            "rpm_min": float(subset_df["rpm"].min()),
            "rpm_max": float(subset_df["rpm"].max()),
            "fs_min": float(subset_df["sampling_rate"].min()),
            "fs_max": float(subset_df["sampling_rate"].max()),
        })

    files_df = pd.DataFrame(file_summaries)

    return records_df, files_df
