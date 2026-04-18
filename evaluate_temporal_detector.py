"""Evaluate temporal anomaly detector on healthy vs faulty samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from src.models.temporal_model import WindowCNNPredictor
from src.preprocessing.preprocess import preprocess_signal
from src.utils.windowing import create_windows


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


def _training_exclusion_ids(dataset_root: Path) -> set[tuple[str, str, int]]:
    """
    Match train_demo.py selection:
    - file: 1/train.mat
    - subsets: DS, FS
    - first 5 healthy samples in each subset
    """
    excluded: set[tuple[str, str, int]] = set()
    mat_path = dataset_root / "1" / "train.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    for subset_name in ["DS", "FS"]:
        if subset_name not in mat:
            continue
        struct = mat[subset_name]
        signals = _extract_signals(struct.rawData)
        labels = _align_vector(struct.label, len(signals), dtype=np.int32)

        count = 0
        for idx, label in enumerate(labels):
            if count >= 5:
                break
            if int(label) != 0:
                continue
            excluded.add((str(mat_path), subset_name, int(idx)))
            count += 1

    return excluded


def _sample_candidates(
    dataset_root: Path,
    excluded_ids: set[tuple[str, str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    healthy: list[dict[str, Any]] = []
    faulty: list[dict[str, Any]] = []

    for mat_path in sorted(dataset_root.glob("*/*.mat")):
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        for subset in ["DS", "FS", "Upper", "Lower"]:
            if subset not in mat:
                continue

            struct = mat[subset]
            signals = _extract_signals(struct.rawData)
            rpms = _align_vector(struct.RPM, len(signals), dtype=np.float64)
            srs = _align_vector(struct.samplingRate, len(signals), dtype=np.float64)
            labels = _align_vector(struct.label, len(signals), dtype=np.int32)

            for i, signal in enumerate(signals):
                fs = float(srs[i])
                if (not np.isfinite(fs)) or fs <= 0:
                    continue

                entry = {
                    "file_path": str(mat_path),
                    "subset": subset,
                    "sample_idx": int(i),
                    "signal": np.asarray(signal, dtype=np.float64).reshape(-1),
                    "rpm": float(rpms[i]),
                    "fs": fs,
                    "label": int(labels[i]),
                }

                sid = (str(mat_path), subset, int(i))
                if int(labels[i]) == 0:
                    if sid in excluded_ids:
                        continue
                    healthy.append(entry)
                else:
                    faulty.append(entry)

    return healthy, faulty


def compute_prediction_error(
    model: WindowCNNPredictor,
    signal: np.ndarray,
    fs: float,
    window_size: int,
    horizon: int = 5,
    return_mean: bool = False,
) -> float | tuple[float, float]:
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size <= window_size + horizon:
        raise ValueError("signal too short for chosen window_size")

    proc = preprocess_signal(signal, fs)
    envelope = proc["envelope"]

    # Per-signal normalization (requested): mean 0, std 1
    mu = float(np.mean(envelope))
    sigma = float(np.std(envelope))
    if (not np.isfinite(sigma)) or sigma < 1e-12:
        raise ValueError("invalid envelope std for normalization")
    envelope_norm = (envelope - mu) / sigma

    X, Y = create_windows(envelope_norm, window_size=window_size, horizon=horizon)
    if X.shape[0] == 0:
        raise ValueError("no windows created")

    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(Y).float()

    with torch.no_grad():
        pred = model(x_tensor)
        # Per-window MSE, then aggregate by max for anomaly scoring.
        window_mse = torch.mean((pred - y_tensor) ** 2, dim=1)

    window_mse_np = window_mse.detach().cpu().numpy()
    if (window_mse_np.size == 0) or (not np.all(np.isfinite(window_mse_np))):
        raise ValueError("non-finite prediction error")

    max_error = float(np.max(window_mse_np))
    mean_error = float(np.mean(window_mse_np))
    if return_mean:
        return max_error, mean_error
    return max_error


def main() -> None:
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    model_path = root / "trained_models" / "envelope_dynamics_demo.pt"
    plot_path = root / "trained_models" / "temporal_error_hist_overlay.png"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = WindowCNNPredictor()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    excluded_ids = _training_exclusion_ids(dataset_root)
    healthy_pool, faulty_pool = _sample_candidates(dataset_root, excluded_ids)

    if len(healthy_pool) < 100:
        raise RuntimeError(f"Not enough healthy candidates: {len(healthy_pool)}")
    if len(faulty_pool) < 100:
        raise RuntimeError(f"Not enough faulty candidates: {len(faulty_pool)}")

    rng = np.random.default_rng(42)
    healthy_idx = rng.choice(len(healthy_pool), size=100, replace=False)
    faulty_idx = rng.choice(len(faulty_pool), size=100, replace=False)

    healthy_sel = [healthy_pool[i] for i in healthy_idx]
    faulty_sel = [faulty_pool[i] for i in faulty_idx]

    healthy_errors: list[float] = []
    faulty_errors: list[float] = []
    healthy_mean_errors: list[float] = []
    faulty_mean_errors: list[float] = []

    for row in healthy_sel:
        try:
            max_err, mean_err = compute_prediction_error(
                model,
                row["signal"],
                row["fs"],
                window_size=512,
                horizon=5,
                return_mean=True,
            )
            healthy_errors.append(max_err)
            healthy_mean_errors.append(mean_err)
        except Exception:
            continue

    for row in faulty_sel:
        try:
            max_err, mean_err = compute_prediction_error(
                model,
                row["signal"],
                row["fs"],
                window_size=512,
                horizon=5,
                return_mean=True,
            )
            faulty_errors.append(max_err)
            faulty_mean_errors.append(mean_err)
        except Exception:
            continue

    if len(healthy_errors) == 0 or len(faulty_errors) == 0:
        raise RuntimeError("No valid errors computed for at least one class")

    mean_healthy = float(np.mean(healthy_errors))
    mean_faulty = float(np.mean(faulty_errors))

    labels = [0] * len(healthy_errors) + [1] * len(faulty_errors)
    scores = healthy_errors + faulty_errors
    roc_auc = float(roc_auc_score(labels, scores))

    plt.figure(figsize=(8, 4.5))
    plt.hist(healthy_errors, bins=30, alpha=0.55, label="Healthy", density=True)
    plt.hist(faulty_errors, bins=30, alpha=0.55, label="Faulty", density=True)
    plt.xlabel("Prediction Error (Max Window MSE)")
    plt.ylabel("Density")
    plt.title("Temporal Model Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()

    print(f"healthy_count: {len(healthy_errors)}")
    print(f"faulty_count: {len(faulty_errors)}")
    print(f"mean_window_mse_healthy = {float(np.mean(healthy_mean_errors)):.8f}")
    print(f"mean_window_mse_faulty = {float(np.mean(faulty_mean_errors)):.8f}")
    print(f"mean_healthy = {mean_healthy:.8f}")
    print(f"mean_faulty = {mean_faulty:.8f}")
    print(f"ROC-AUC = {roc_auc:.6f}")
    print(f"Faulty higher than healthy: {mean_faulty > mean_healthy}")
    print(f"Histogram saved: {plot_path}")


if __name__ == "__main__":
    main()
