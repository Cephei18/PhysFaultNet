"""Final single-sample prediction pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from ..features import extract_features_csv_exact
from ..preprocessing import preprocess_signal
from ..utils.windowing import create_windows
from .ocsvm import ScoreFusion, fuse_scores
from .temporal_model import WindowCNNPredictor


def _compute_prediction_error_from_envelope(
    temporal_model: WindowCNNPredictor,
    envelope: np.ndarray,
    window_size: int,
    horizon: int,
) -> float:
    """Compute max per-window MSE prediction error for one envelope signal."""
    env = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if env.size <= window_size + horizon:
        raise ValueError("signal too short for temporal window_size/horizon")

    mu = float(np.mean(env))
    sigma = float(np.std(env))
    if (not np.isfinite(sigma)) or sigma < 1e-12:
        raise ValueError("invalid envelope std for temporal normalization")
    env_n = (env - mu) / sigma

    X, Y = create_windows(env_n, window_size=window_size, horizon=horizon)
    if X.shape[0] == 0:
        raise ValueError("no temporal windows created")

    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(Y).float()

    with torch.no_grad():
        pred = temporal_model(x_tensor)
        window_mse = torch.mean((pred - y_tensor) ** 2, dim=1)

    wmse = window_mse.detach().cpu().numpy()
    if (wmse.size == 0) or (not np.all(np.isfinite(wmse))):
        raise ValueError("non-finite temporal prediction error")
    return float(np.max(wmse))


def _compute_svm_score(
    feature_vector: np.ndarray,
    svm_scaler: RobustScaler,
    svm_model: OneClassSVM,
) -> float:
    """Compute OCSVM anomaly score (higher = more anomalous) for one feature vector."""
    fv = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    fv_scaled = svm_scaler.transform(fv)
    score = -svm_model.decision_function(fv_scaled).reshape(-1)[0]
    return float(score)


def predict_sample(
    sample: dict[str, Any],
    temporal_model: WindowCNNPredictor,
    svm_model: OneClassSVM,
    svm_scaler: RobustScaler,
    classifier: BaseEstimator,
    threshold: float,
    score_fusion: ScoreFusion | None = None,
    alpha: float = 0.7,
    window_size: int = 512,
    horizon: int = 5,
) -> int:
    """Predict one sample with anomaly-detection gate + multiclass fault classifier.

    Required sample keys:
    - signal: 1D array-like
    - fs: sampling rate
    - rpm: shaft RPM
    - fault_mult: fault multiplier object/dict for feature extraction

    Returns
    -------
    int
        0 for healthy, else multiclass fault label predicted by classifier.
    """
    required = ["signal", "fs", "rpm", "fault_mult"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise ValueError(f"sample is missing required keys: {missing}")

    signal = np.asarray(sample["signal"], dtype=np.float64).reshape(-1)
    fs = float(sample["fs"])
    rpm = float(sample["rpm"])
    fault_mult = sample["fault_mult"]

    # Step 1: preprocessing
    proc = preprocess_signal(signal, fs)
    envelope = proc["envelope"]

    # Step 2: anomaly detection
    feature_vector = extract_features_csv_exact(
        signal=signal,
        fs=fs,
        rpm=rpm,
        fault_mult=fault_mult,
    )
    svm_score = _compute_svm_score(feature_vector, svm_scaler=svm_scaler, svm_model=svm_model)
    temporal_score = _compute_prediction_error_from_envelope(
        temporal_model=temporal_model,
        envelope=envelope,
        window_size=window_size,
        horizon=horizon,
    )

    if score_fusion is not None:
        fused_arr, _, _ = score_fusion.transform(np.array([svm_score]), np.array([temporal_score]))
        fused_score = float(fused_arr[0])
    else:
        fused_arr = fuse_scores(np.array([svm_score]), np.array([temporal_score]), alpha=alpha)
        fused_score = float(fused_arr[0])

    # Step 3: threshold
    if fused_score < float(threshold):
        return 0  # healthy
    else:
        print("Classifier triggered")
        # Step 4: classification
        raw_output = classifier.predict([feature_vector])
        print("Classifier raw output:", raw_output)
        fault_class = int(raw_output[0])
        return fault_class
