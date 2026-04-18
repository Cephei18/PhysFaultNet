"""Evaluate SVM, temporal, and fused anomaly scores on the same sample set."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

from src.features.extract_features import extract_features
from src.models import (
    ScoreFusion,
    WindowCNNPredictor,
    evaluate_scores,
    final_ocsvm_pipeline,
    fuse_scores,
    normalize_scores,
    plot_score_histogram_overlay,
)
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

                sid = (str(mat_path), subset, int(i))
                if int(labels[i]) == 0 and sid in excluded_ids:
                    continue

                entry = {
                    "file_path": str(mat_path),
                    "subset": subset,
                    "sample_idx": int(i),
                    "signal": np.asarray(signal, dtype=np.float64).reshape(-1),
                    "rpm": float(rpms[i]),
                    "fs": fs,
                    "label": int(labels[i]),
                    "fault_mult": struct.faultFrequencies,
                }

                if int(labels[i]) == 0:
                    healthy.append(entry)
                else:
                    faulty.append(entry)

    return healthy, faulty


def _temporal_score(model: WindowCNNPredictor, signal: np.ndarray, fs: float, window_size: int = 512, horizon: int = 5) -> float:
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if sig.size <= window_size + horizon:
        raise ValueError("signal too short")

    env = preprocess_signal(sig, fs)["envelope"]
    mu = float(np.mean(env))
    sigma = float(np.std(env))
    if (not np.isfinite(sigma)) or sigma < 1e-12:
        raise ValueError("invalid envelope std")
    env_n = (env - mu) / sigma

    x, y = create_windows(env_n, window_size=window_size, horizon=horizon)
    if x.shape[0] == 0:
        raise ValueError("no windows")

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()

    with torch.no_grad():
        pred = model(xt)
        window_mse = torch.mean((pred - yt) ** 2, dim=1)

    wmse = window_mse.detach().cpu().numpy()
    if (wmse.size == 0) or (not np.all(np.isfinite(wmse))):
        raise ValueError("invalid temporal score")
    return float(np.max(wmse))


def _svm_feature_vector(signal: np.ndarray, fs: float, rpm: float, fault_mult: Any) -> np.ndarray:
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    proc = preprocess_signal(sig, fs)
    feat = extract_features(
        envelope=proc["envelope"],
        fft_vals=proc["fft_vals"],
        freqs=proc["freqs"],
        rpm=rpm,
        fault_mult=fault_mult,
    )
    return np.asarray(feat, dtype=np.float64).reshape(-1)


def main() -> None:
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    model_path = root / "trained_models" / "envelope_dynamics_demo.pt"
    plot_path = root / "trained_models" / "svm_temporal_fused_hist_overlay.png"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing temporal model: {model_path}")

    temporal_model = WindowCNNPredictor()
    temporal_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    temporal_model.eval()

    excluded_ids = _training_exclusion_ids(dataset_root)
    healthy_pool, faulty_pool = _sample_candidates(dataset_root, excluded_ids)

    if len(healthy_pool) < 100 or len(faulty_pool) < 100:
        raise RuntimeError("Not enough candidates for 100 healthy + 100 faulty")

    rng = np.random.default_rng(42)
    h_idx = rng.choice(len(healthy_pool), size=100, replace=False)
    f_idx = rng.choice(len(faulty_pool), size=100, replace=False)
    selected = [healthy_pool[i] for i in h_idx] + [faulty_pool[i] for i in f_idx]

    temporal_scores: list[float] = []
    features: list[np.ndarray] = []
    labels: list[int] = []

    for row in selected:
        try:
            t_score = _temporal_score(temporal_model, row["signal"], row["fs"], window_size=512, horizon=5)
            f_vec = _svm_feature_vector(row["signal"], row["fs"], row["rpm"], row["fault_mult"])
        except Exception:
            continue

        if f_vec.shape[0] != 5 or (not np.all(np.isfinite(f_vec))):
            continue

        temporal_scores.append(t_score)
        features.append(f_vec)
        labels.append(0 if int(row["label"]) == 0 else 1)

    if len(labels) < 50:
        raise RuntimeError("Too few valid samples after filtering")

    x = np.vstack(features)
    y = np.asarray(labels, dtype=np.int32)

    svm_scores, _, _ = final_ocsvm_pipeline(x, y, healthy_label=0, nu=0.03)
    temporal_scores_arr = np.asarray(temporal_scores, dtype=np.float64)

    healthy_mask = y == 0
    svm_n, tmp_n = normalize_scores(svm_scores, temporal_scores_arr, healthy_mask=healthy_mask)

    fused = fuse_scores(svm_n, tmp_n, alpha=0.7)

    print("\nEvaluating SVM normalized scores")
    roc_svm, pr_svm = evaluate_scores(svm_n, y)

    print("\nEvaluating Temporal normalized scores")
    roc_tmp, pr_tmp = evaluate_scores(tmp_n, y)

    print("\nEvaluating Fused scores")
    roc_fused, pr_fused = evaluate_scores(fused, y)

    print("\nAlpha sweep")
    alphas = [0.6, 0.7, 0.8, 0.85, 0.9]
    best = (None, -1.0, -1.0)

    for a in alphas:
        fusion = ScoreFusion(alpha=a)
        fusion.fit(svm_scores, temporal_scores_arr)
        fused_a, _, _ = fusion.transform(svm_scores, temporal_scores_arr)
        roc_a, pr_a = evaluate_scores(fused_a, y)
        print(f"alpha={a} -> ROC={roc_a:.4f}, PR={pr_a:.4f}")
        if roc_a > best[1]:
            best = (a, float(roc_a), float(pr_a))

    print("Best:", best)

    print("\nSuccess condition checks")
    print("ROC(fused) > ROC(svm):", roc_fused > roc_svm)
    print("PR(fused)  > PR(svm):", pr_fused > pr_svm)

    plot_score_histogram_overlay(
        svm_scores=svm_n,
        temporal_scores=tmp_n,
        fused_scores=fused,
        labels=y,
        output_path=str(plot_path),
    )
    print("Histogram saved:", plot_path)


if __name__ == "__main__":
    main()
