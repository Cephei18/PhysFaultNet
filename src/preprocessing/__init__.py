"""Signal preprocessing."""
from .preprocess import preprocess_signal
from .quality_gate import (
    QualityGateResult,
    quality_gate,
    compute_fault_peak_alignment,
    process_dataset_with_quality_gate,
)

__all__ = [
    "preprocess_signal",
    "quality_gate",
    "QualityGateResult",
    "compute_fault_peak_alignment",
    "process_dataset_with_quality_gate",
]
