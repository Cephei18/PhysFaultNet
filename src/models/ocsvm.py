"""One-Class SVM anomaly detection model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


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


class ScoreFusion:
    def __init__(self, alpha: float = 0.7):
        self.alpha = float(alpha)
        self.svm_scaler = RobustScaler()
        self.tmp_scaler = RobustScaler()

    def fit(self, svm_scores: Any, tmp_scores: Any) -> None:
        svm = np.asarray(svm_scores, dtype=np.float64).reshape(-1)
        tmp = np.asarray(tmp_scores, dtype=np.float64).reshape(-1)
        if svm.size == 0 or tmp.size == 0:
            raise ValueError("svm_scores and tmp_scores must be non-empty")
        if svm.shape[0] != tmp.shape[0]:
            raise ValueError("svm_scores and tmp_scores must have the same length")

        self.svm_scaler.fit(svm.reshape(-1, 1))
        self.tmp_scaler.fit(tmp.reshape(-1, 1))

    def transform(self, svm_scores: Any, tmp_scores: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        svm = np.asarray(svm_scores, dtype=np.float64).reshape(-1)
        tmp = np.asarray(tmp_scores, dtype=np.float64).reshape(-1)
        if svm.size == 0 or tmp.size == 0:
            raise ValueError("svm_scores and tmp_scores must be non-empty")
        if svm.shape[0] != tmp.shape[0]:
            raise ValueError("svm_scores and tmp_scores must have the same length")

        svm_n = self.svm_scaler.transform(svm.reshape(-1, 1)).ravel()
        tmp_n = self.tmp_scaler.transform(tmp.reshape(-1, 1)).ravel()
        fused = self.alpha * svm_n + (1.0 - self.alpha) * tmp_n
        return fused, svm_n, tmp_n


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
    """Preprocess OCSVM features with log1p and RobustScaler."""
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
    """Preprocess final 5D feature vectors using RobustScaler fit on healthy samples."""
    x, y = _validate_feature_inputs(feature_matrix, labels)
    healthy_mask = y == healthy_label
    if not np.any(healthy_mask):
        raise ValueError("no healthy samples found for training")

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(np.array(x[healthy_mask], dtype=np.float64, copy=True))
    x_all_scaled = scaler.transform(np.array(x, dtype=np.float64, copy=True))
    return x_train_scaled, x_all_scaled, scaler


def train_ocsvm(X_train: Any, nu: float) -> OneClassSVM:
    """Train an RBF One-Class SVM on preprocessed healthy-only data."""
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
    """Train and score One-Class SVM using healthy samples for training."""
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
    """Train and evaluate One-Class SVM across multiple nu values."""
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
    """Final OCSVM pipeline for the final 5D feature set.

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


def normalize_svm_temporal_scores(
    svm_scores: Any,
    temporal_scores: Any,
    healthy_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SVM and temporal scores independently using RobustScaler.

    If healthy_mask is provided, each scaler is fit on healthy-only scores.
    Otherwise, each scaler is fit on all scores.
    """
    svm = np.asarray(svm_scores, dtype=np.float64).reshape(-1)
    tmp = np.asarray(temporal_scores, dtype=np.float64).reshape(-1)

    if svm.size == 0 or tmp.size == 0:
        raise ValueError("svm_scores and temporal_scores must be non-empty")
    if svm.shape[0] != tmp.shape[0]:
        raise ValueError("svm_scores and temporal_scores must have the same length")
    if not np.all(np.isfinite(svm)) or not np.all(np.isfinite(tmp)):
        raise ValueError("scores must be finite")

    fit_svm = svm
    fit_tmp = tmp
    if healthy_mask is not None:
        mask = np.asarray(healthy_mask).reshape(-1)
        if mask.shape[0] != svm.shape[0]:
            raise ValueError("healthy_mask must have the same length as scores")
        mask = mask.astype(bool)
        if not np.any(mask):
            raise ValueError("healthy_mask selects no samples")
        fit_svm = svm[mask]
        fit_tmp = tmp[mask]

    scaler_svm = RobustScaler()
    scaler_tmp = RobustScaler()

    scaler_svm.fit(fit_svm.reshape(-1, 1))
    svm_n = scaler_svm.transform(svm.reshape(-1, 1)).ravel()

    scaler_tmp.fit(fit_tmp.reshape(-1, 1))
    tmp_n = scaler_tmp.transform(tmp.reshape(-1, 1)).ravel()

    return svm_n, tmp_n


def normalize_scores(
    svm_scores: Any,
    temporal_scores: Any,
    healthy_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Alias for robust score normalization of SVM and temporal scores."""
    return normalize_svm_temporal_scores(
        svm_scores=svm_scores,
        temporal_scores=temporal_scores,
        healthy_mask=healthy_mask,
    )


def fuse_scores(svm_n: Any, tmp_n: Any, alpha: float = 0.7) -> np.ndarray:
    """Fuse normalized SVM and temporal scores with a weighted sum."""
    svm = np.asarray(svm_n, dtype=np.float64).reshape(-1)
    tmp = np.asarray(tmp_n, dtype=np.float64).reshape(-1)

    if svm.size == 0 or tmp.size == 0:
        raise ValueError("svm_n and tmp_n must be non-empty")
    if svm.shape[0] != tmp.shape[0]:
        raise ValueError("svm_n and tmp_n must have the same length")
    if not np.all(np.isfinite(svm)) or not np.all(np.isfinite(tmp)):
        raise ValueError("svm_n and tmp_n must be finite")

    alpha_val = float(alpha)
    if not np.isfinite(alpha_val) or alpha_val < 0.0 or alpha_val > 1.0:
        raise ValueError("alpha must be in [0, 1]")

    fused = alpha_val * svm + (1.0 - alpha_val) * tmp
    return fused


def evaluate_scores(scores: Any, labels: Any) -> tuple[float, float]:
    """Evaluate anomaly scores with ROC-AUC and PR-AUC."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)

    if s.size == 0 or y.size == 0:
        raise ValueError("scores and labels must be non-empty")
    if s.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must have the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")

    y01 = y.astype(np.int32)
    if not np.all(np.isin(y01, [0, 1])):
        raise ValueError("labels must be binary with values in {0, 1}")

    roc = float(roc_auc_score(y01, s))
    pr = float(average_precision_score(y01, s))

    healthy = s[y01 == 0]
    faulty = s[y01 == 1]
    if healthy.size == 0 or faulty.size == 0:
        raise ValueError("labels must contain both healthy (0) and faulty (1) samples")

    mean_healthy = float(np.mean(healthy))
    mean_faulty = float(np.mean(faulty))

    print("ROC-AUC:", roc)
    print("PR-AUC :", pr)
    print("mean healthy:", mean_healthy)
    print("mean faulty :", mean_faulty)
    print("faulty > healthy:", mean_faulty > mean_healthy)

    return roc, pr


def plot_score_histogram_overlay(
    svm_scores: Any,
    temporal_scores: Any,
    fused_scores: Any,
    labels: Any | None = None,
    output_path: str | None = None,
    bins: int = 35,
) -> None:
    """Plot score histograms to compare SVM, temporal, and fused separations."""
    import matplotlib.pyplot as plt

    svm = np.asarray(svm_scores, dtype=np.float64).reshape(-1)
    tmp = np.asarray(temporal_scores, dtype=np.float64).reshape(-1)
    fused = np.asarray(fused_scores, dtype=np.float64).reshape(-1)

    if svm.size == 0 or tmp.size == 0 or fused.size == 0:
        raise ValueError("score arrays must be non-empty")
    if not (svm.shape[0] == tmp.shape[0] == fused.shape[0]):
        raise ValueError("svm_scores, temporal_scores, and fused_scores must have the same length")
    if (not np.all(np.isfinite(svm))) or (not np.all(np.isfinite(tmp))) or (not np.all(np.isfinite(fused))):
        raise ValueError("all score arrays must be finite")

    if labels is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.6))
        ax.hist(svm, bins=bins, alpha=0.45, density=True, label="SVM")
        ax.hist(tmp, bins=bins, alpha=0.45, density=True, label="Temporal")
        ax.hist(fused, bins=bins, alpha=0.45, density=True, label="Fused")
        ax.set_title("Score Histogram Overlay")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
    else:
        y = np.asarray(labels).reshape(-1).astype(np.int32)
        if y.shape[0] != svm.shape[0]:
            raise ValueError("labels must have the same length as scores")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("labels must be binary with values in {0, 1}")

        h_mask = y == 0
        f_mask = y == 1
        if (not np.any(h_mask)) or (not np.any(f_mask)):
            raise ValueError("labels must contain both healthy (0) and faulty (1)")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
        items = [
            ("SVM", svm),
            ("Temporal", tmp),
            ("Fused", fused),
        ]
        for ax, (name, arr) in zip(axes, items):
            ax.hist(arr[h_mask], bins=bins, alpha=0.55, density=True, label="Healthy")
            ax.hist(arr[f_mask], bins=bins, alpha=0.55, density=True, label="Faulty")
            ax.set_title(name)
            ax.set_xlabel("Score")
        axes[0].set_ylabel("Density")
        axes[-1].legend()
        fig.suptitle("Score Separation Overlay")
        fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=140)
    plt.close(fig)


def learn_fusion(
    svm_n: Any,
    tmp_n: Any,
    labels: Any,
    random_state: int = 42,
) -> tuple[LogisticRegression, np.ndarray]:
    """Learn fusion weights with logistic regression and return fused probabilities."""
    svm = np.asarray(svm_n, dtype=np.float64).reshape(-1)
    tmp = np.asarray(tmp_n, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(np.int32)

    if svm.size == 0 or tmp.size == 0 or y.size == 0:
        raise ValueError("svm_n, tmp_n, and labels must be non-empty")
    if not (svm.shape[0] == tmp.shape[0] == y.shape[0]):
        raise ValueError("svm_n, tmp_n, and labels must have the same length")
    if not np.all(np.isfinite(svm)) or not np.all(np.isfinite(tmp)):
        raise ValueError("svm_n and tmp_n must be finite")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("labels must be binary with values in {0, 1}")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain both classes for logistic fusion")

    x_fusion = np.c_[svm, tmp]
    fusion_model = LogisticRegression(random_state=random_state)
    fusion_model.fit(x_fusion, y)
    fused = fusion_model.predict_proba(x_fusion)[:, 1]
    return fusion_model, fused


def predict_fused_scores(
    fusion_model: LogisticRegression,
    svm_n: Any,
    tmp_n: Any,
) -> np.ndarray:
    """Predict fused anomaly probabilities from normalized SVM and temporal scores."""
    svm = np.asarray(svm_n, dtype=np.float64).reshape(-1)
    tmp = np.asarray(tmp_n, dtype=np.float64).reshape(-1)
    if svm.size == 0 or tmp.size == 0:
        raise ValueError("svm_n and tmp_n must be non-empty")
    if svm.shape[0] != tmp.shape[0]:
        raise ValueError("svm_n and tmp_n must have the same length")
    if not np.all(np.isfinite(svm)) or not np.all(np.isfinite(tmp)):
        raise ValueError("svm_n and tmp_n must be finite")

    x_fusion = np.c_[svm, tmp]
    return fusion_model.predict_proba(x_fusion)[:, 1]


def generate_multiclass_labels(features: Any, binary_labels: Any) -> np.ndarray:
    """Generate multiclass labels from binary labels and fault-energy features.

    Assumed feature columns:
    - E_bpfi at index 2
    - E_bpfo at index 3
    - E_bsf at index 4

    Output mapping:
    - 0: healthy
    - 1: inner race fault (BPFI dominant)
    - 2: ball fault (BSF dominant)
    - 3: outer race fault (BPFO dominant)
    """
    x = np.asarray(features, dtype=np.float64)
    y_bin = np.asarray(binary_labels).reshape(-1)

    if x.ndim != 2:
        raise ValueError("features must be a 2D array of shape (N, d)")
    if x.shape[0] != y_bin.shape[0]:
        raise ValueError("features and binary_labels must have the same number of samples")
    if x.shape[1] <= 4:
        raise ValueError("features must include E_bpfi/E_bpfo/E_bsf at indices 2/3/4")

    y_multiclass: list[int] = []
    for row, label in zip(x, y_bin):
        if int(label) == 0:
            y_multiclass.append(0)
            continue

        e_bpfi = float(row[2])
        e_bpfo = float(row[3])
        e_bsf = float(row[4])

        energies = np.array([e_bpfi, e_bsf, e_bpfo], dtype=np.float64)
        if not np.all(np.isfinite(energies)):
            energies = np.nan_to_num(energies, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

        idx = int(np.argmax(energies))
        if idx == 0:
            y_multiclass.append(1)
        elif idx == 1:
            y_multiclass.append(2)
        else:
            y_multiclass.append(3)

    return np.asarray(y_multiclass, dtype=np.int32)
