"""Test the refactored pipeline."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from src.preprocessing.quality_gate import quality_gate
from src.preprocessing.preprocess import preprocess_signal
from src.features.extract_features import extract_features


def test_pipeline():
    """Load a sample and run it through the full pipeline."""
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    
    # Load first test sample from folder 1
    sample_path = dataset_root / "1" / "test.mat"
    mat_data = loadmat(sample_path, squeeze_me=True, struct_as_record=False)
    
    # Find first available subset (DS, FS, Upper, or Lower)
    subset_name = None
    for subset in ["DS", "FS", "Upper", "Lower"]:
        if subset in mat_data:
            subset_name = subset
            break
    
    if subset_name is None:
        print("✗ No valid subset found in mat file")
        return
    
    struct = mat_data[subset_name]
    
    # Extract first signal
    signals = struct.rawData
    if not hasattr(signals, '__len__'):
        signals = [signals]
    
    signal = np.asarray(signals[0], dtype=np.float64).reshape(-1)
    sampling_rate = float(np.asarray(struct.samplingRate).reshape(-1)[0])
    rpm = float(np.asarray(struct.RPM).reshape(-1)[0])
    
    # Standard fault multipliers for this dataset
    fault_mult = struct.faultFrequencies
    
    print(f"✓ Loaded sample: shape={signal.shape}, subset={subset_name}")
    print(f"  Sampling rate: {sampling_rate} Hz, RPM: {rpm}")
    
    # Pass through quality_gate
    result, rejection_reason = quality_gate(
        signal=signal,
        rpm=rpm,
        fs=sampling_rate,
        fault_mult=fault_mult
    )
    
    if result is None:
        print(f"✗ Quality gate rejection: {rejection_reason}")
        return
    
    print(f"✓ Quality gate passed")
    
    # Preprocess
    processed = preprocess_signal(result.signal, fs=sampling_rate)
    print(f"✓ Preprocessed: envelope shape={processed['envelope'].shape}")
    
    # Extract features
    features = extract_features(
        envelope=processed["envelope"],
        fft_vals=processed["fft_vals"],
        freqs=processed["freqs"],
        rpm=rpm,
        fault_mult=fault_mult
    )
    print(f"✓ Extracted features: shape={features.shape}")
    print(f"  RMS (log): {features[0]:.4f}")
    print(f"  Kurtosis: {features[1]:.4f}")
    print(f"  E_BPFI (log): {features[2]:.4f}")
    print(f"  E_BPFO (log): {features[3]:.4f}")
    print(f"  E_BSF (log): {features[4]:.4f}")
    
    # Verify outputs are valid
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert features.size == 5, "Features should have 5 elements"
    assert np.all(np.isfinite(features)), "All feature values should be finite"
    
    print("\n✅ Pipeline working after refactor")


if __name__ == "__main__":
    test_pipeline()
