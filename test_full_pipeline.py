"""Run full end-to-end pipeline on the test split and report multiclass results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_curve

from evaluate_temporal_detector import compute_prediction_error
from src.models import ScoreFusion, WindowCNNPredictor, generate_multiclass_labels, predict_sample
from src.models.ocsvm import final_ocsvm_pipeline
from src.features import extract_features
from src.preprocessing import preprocess_signal


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


def _build_sample_entry(row: pd.Series, signal: np.ndarray) -> dict[str, Any]:
    return {
        "signal": np.asarray(signal, dtype=np.float64).reshape(-1),
        "fs": float(row["sampling_rate"]),
        "rpm": float(row["rpm"]),
        "fault_mult": {
            "BPFI": float(row["fault_bpfi_mult"]),
            "BPFO": float(row["fault_bpfo_mult"]),
            "BSF": float(row["fault_bsf_mult"]),
            "FTF": float(row["fault_ftf_mult"]),
        },
    }


def main() -> None:
    root = Path("/home/teaching/Hackathon_dl")
    csv_path = root / "analysis_outputs" / "sample_level_features.csv"
    model_path = root / "trained_models" / "envelope_dynamics_demo.pt"
    plot_path = root / "trained_models" / "full_pipeline_confusion_matrix.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature table: {csv_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing temporal model: {model_path}")

    df = pd.read_csv(csv_path)

    # Build multiclass ground-truth labels from energy dominance + healthy class.
    X_tab = df[["rms", "kurtosis", "E_bpfi", "E_bpfo", "E_bsf"]].to_numpy(dtype=np.float64)
    y_multiclass = generate_multiclass_labels(X_tab, df["label"].to_numpy())

    train_mask = df["split"].astype(str).str.lower() == "train"
    test_mask = ~train_mask

    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise RuntimeError("Train/test split not found in sample_level_features.csv")

    X_train = X_tab[train_mask.to_numpy()]
    y_train = y_multiclass[train_mask.to_numpy()]

    # 1) Train SVM anomaly branch on train split.
    _, svm_scaler, svm_model = final_ocsvm_pipeline(X_train, y_train, healthy_label=0, nu=0.03)

    # 2) Train fault classifier on faulty train samples only.
    fault_mask = y_train != 0
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train[fault_mask], y_train[fault_mask])
    print("Classifier quick check on fault-train[:10]:", clf.predict(X_train[fault_mask][:10]))

    # 3) Load temporal model.
    temporal_model = WindowCNNPredictor()
    temporal_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    temporal_model.eval()

    # 4) Fit score fusion + threshold on train split scores.
    train_rows = df[train_mask].reset_index(drop=True)
    fusion_train_subset = min(300, len(train_rows))
    rng = np.random.default_rng(7)
    train_idx = rng.choice(len(train_rows), size=fusion_train_subset, replace=False)
    train_rows = train_rows.iloc[train_idx].reset_index(drop=True)
    x_train_subset = X_tab[train_mask.to_numpy()][train_idx]
    y_train_subset = y_train[train_idx]
    print(f"Fusion fit subset (train): {fusion_train_subset} samples")
    train_svm_scores: list[float] = []
    train_tmp_scores: list[float] = []
    train_labels: list[int] = []

    mat_cache: dict[str, Any] = {}
    subset_cache: dict[tuple[str, str], list[np.ndarray]] = {}

    for i, row in train_rows.iterrows():
        file_path = str(row["file_path"])
        subset = str(row["subset"])
        sample_idx = int(row["sample_idx"])

        if file_path not in mat_cache:
            mat_cache[file_path] = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        mat = mat_cache[file_path]

        key = (file_path, subset)
        if key not in subset_cache:
            subset_cache[key] = _extract_signals(mat[subset].rawData)
        signals = subset_cache[key]

        if sample_idx >= len(signals):
            continue

        signal = np.asarray(signals[sample_idx], dtype=np.float64).reshape(-1)

        try:
            tmp_score = compute_prediction_error(
                model=temporal_model,
                signal=signal,
                fs=float(row["sampling_rate"]),
                window_size=512,
                horizon=5,
                return_mean=False,
            )
            x_one = x_train_subset[i : i + 1]
            svm_score = float(-svm_model.decision_function(svm_scaler.transform(x_one)).reshape(-1)[0])
        except Exception:
            continue

        train_svm_scores.append(svm_score)
        train_tmp_scores.append(float(tmp_score))
        train_labels.append(int(y_train_subset[i]))

        if (len(train_svm_scores) % 50) == 0:
            print(f"Fusion train progress: {len(train_svm_scores)}/{fusion_train_subset}")

    if len(train_svm_scores) == 0:
        raise RuntimeError("No valid train scores for fusion fitting")

    fusion = ScoreFusion(alpha=0.7)
    fusion.fit(np.asarray(train_svm_scores), np.asarray(train_tmp_scores))
    fused_train, _, _ = fusion.transform(np.asarray(train_svm_scores), np.asarray(train_tmp_scores))

    train_labels_arr = np.asarray(train_labels, dtype=np.int32)
    healthy_fused = fused_train[train_labels_arr == 0]
    if healthy_fused.size == 0:
        raise RuntimeError("No healthy train samples available for threshold estimation")
    threshold_auto = float(np.percentile(healthy_fused, 95.0))
    threshold = 5.3
    print("Threshold auto (95th healthy train):", threshold_auto)
    print("Threshold override (quick fix):", threshold)

    # 5) Predict a random subset of test samples with full pipeline.
    test_rows = df[test_mask].reset_index(drop=True)
    subset_size = min(100, len(test_rows))
    rng = np.random.default_rng(42)
    chosen_idx = rng.choice(len(test_rows), size=subset_size, replace=False)
    test_rows = test_rows.iloc[chosen_idx].reset_index(drop=True)
    y_test_subset = y_multiclass[test_mask.to_numpy()][chosen_idx]

    print(f"Evaluating random test subset: {subset_size} samples")

    y_true: list[int] = []
    y_pred_full: list[int] = []
    fused_scores: list[float] = []
    debug_prints = 0

    for i, row in test_rows.iterrows():
        file_path = str(row["file_path"])
        subset = str(row["subset"])
        sample_idx = int(row["sample_idx"])

        if file_path not in mat_cache:
            mat_cache[file_path] = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        mat = mat_cache[file_path]

        key = (file_path, subset)
        if key not in subset_cache:
            subset_cache[key] = _extract_signals(mat[subset].rawData)
        signals = subset_cache[key]

        if sample_idx >= len(signals):
            continue

        signal = np.asarray(signals[sample_idx], dtype=np.float64).reshape(-1)
        sample = _build_sample_entry(row, signal)

        try:
            proc = preprocess_signal(signal, float(row["sampling_rate"]))
            feat_vec = extract_features(
                envelope=proc["envelope"],
                fft_vals=proc["fft_vals"],
                freqs=proc["freqs"],
                rpm=float(row["rpm"]),
                fault_mult=sample["fault_mult"],
            )
            x_one = np.asarray(feat_vec, dtype=np.float64).reshape(1, -1)
            svm_score = float(-svm_model.decision_function(svm_scaler.transform(x_one)).reshape(-1)[0])

            tmp_score = compute_prediction_error(
                model=temporal_model,
                signal=signal,
                fs=float(row["sampling_rate"]),
                window_size=512,
                horizon=5,
                return_mean=False,
            )
            fused_val = float(fusion.transform(np.array([svm_score]), np.array([float(tmp_score)]))[0][0])

            if debug_prints < 5:
                print("features shape:", np.array(feat_vec).shape)
                print("features:", feat_vec)
                print("prediction:", clf.predict([feat_vec]))
                debug_prints += 1

            pred = predict_sample(
                sample=sample,
                temporal_model=temporal_model,
                svm_model=svm_model,
                svm_scaler=svm_scaler,
                classifier=clf,
                threshold=threshold,
                score_fusion=fusion,
                window_size=512,
                horizon=5,
            )
        except Exception:
            continue

        fused_scores.append(fused_val)
        y_pred_full.append(int(pred))
        y_true.append(int(y_test_subset[i]))

        if (len(y_pred_full) % 20) == 0:
            print(f"Progress: {len(y_pred_full)}/{subset_size} predictions")

    if len(y_true) == 0:
        raise RuntimeError("No valid predictions were produced on test split")

    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred_full, dtype=np.int32)
    fused_scores_arr = np.asarray(fused_scores, dtype=np.float64)

    print("classification_report(y_true, y_pred_full)")
    print(classification_report(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3], zero_division=0))

    print("Fused score stats:")
    print("min:", fused_scores_arr.min())
    print("max:", fused_scores_arr.max())
    print("mean:", fused_scores_arr.mean())
    print("Healthy mean:", fused_scores_arr[y_true_arr == 0].mean())
    print("Faulty mean :", fused_scores_arr[y_true_arr != 0].mean())

    labels_binary = (y_true_arr != 0).astype(int)
    prec, rec, thr = precision_recall_curve(labels_binary, fused_scores_arr)
    target_precision = 0.9
    meets = np.where(prec[:-1] >= target_precision)[0]
    if meets.size > 0 and thr.size > 0:
        idx = int(meets[0])
        threshold_pr = float(thr[idx])
    else:
        threshold_pr = float(threshold)
    print("New threshold:", threshold_pr)

    # Confusion matrix plot.
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1, 2, 3],
        display_labels=["healthy", "inner", "ball", "outer"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title("Full Pipeline Confusion Matrix (Test Split)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()

    # Requested checks.
    true_classes = sorted(set(y_true_arr.tolist()))
    pred_classes = sorted(set(y_pred_arr.tolist()))
    print("y_true unique classes:", true_classes)
    print("y_pred_full unique classes:", pred_classes)
    print("All 4 classes present in y_true:", set([0, 1, 2, 3]).issubset(set(true_classes)))

    healthy_mask_test = y_true_arr == 0
    fault_mask_test = y_true_arr != 0
    healthy_acc = float(np.mean(y_pred_arr[healthy_mask_test] == 0)) if np.any(healthy_mask_test) else float("nan")
    fault_acc = float(np.mean(y_pred_arr[fault_mask_test] == y_true_arr[fault_mask_test])) if np.any(fault_mask_test) else float("nan")
    print("Healthy correctly identified (recall for class 0):", healthy_acc)
    print("Faults correctly classified (fault-only accuracy):", fault_acc)
    print("Confusion matrix plot saved:", plot_path)


if __name__ == "__main__":
    main()
