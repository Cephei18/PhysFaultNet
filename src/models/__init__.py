"""Models module."""
from .ocsvm import (
    ScoreFusion,
    OneClassSVMResult,
    OneClassSVMSearchResult,
    one_class_svm_pipeline,
    sweep_ocsvm_nu,
    final_ocsvm_pipeline,
    normalize_svm_temporal_scores,
    normalize_scores,
    fuse_scores,
    evaluate_scores,
    plot_score_histogram_overlay,
    learn_fusion,
    predict_fused_scores,
    generate_multiclass_labels,
)
from .multiclass import (
    train_multiclass_fault_classifier,
    train_fault_only_classifier,
    evaluate_multiclass_classifier,
)
from .final_predictor import predict_sample
from .temporal_model import WindowCNNPredictor

__all__ = [
    "ScoreFusion",
    "OneClassSVMResult",
    "OneClassSVMSearchResult",
    "one_class_svm_pipeline",
    "sweep_ocsvm_nu",
    "final_ocsvm_pipeline",
    "normalize_svm_temporal_scores",
    "normalize_scores",
    "fuse_scores",
    "evaluate_scores",
    "plot_score_histogram_overlay",
    "learn_fusion",
    "predict_fused_scores",
    "generate_multiclass_labels",
    "train_multiclass_fault_classifier",
    "train_fault_only_classifier",
    "evaluate_multiclass_classifier",
    "predict_sample",
    "WindowCNNPredictor",
]
