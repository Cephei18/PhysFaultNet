"""End-to-end training pipeline with real SCA bearing data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch

from src.preprocessing.quality_gate import quality_gate
from src.preprocessing.preprocess import preprocess_signal
from src.utils.windowing import create_windows
from train_envelope_dynamics import train_envelope_dynamics_predictor


def collect_healthy_envelopes(dataset_root: Path, num_files: int = 3) -> np.ndarray:
    """
    Extract envelope windows from healthy (label=0) samples.

    Parameters
    ----------
    dataset_root
        Path to SCA bearing dataset.
    num_files
        Max number of files to process (for demo).

    Returns
    -------
    np.ndarray
        All envelope windows, shape (total_windows, window_size).
    """
    all_windows = []
    files_processed = 0

    # Iterate through dataset folders
    for folder_path in sorted(dataset_root.glob("*")):
        if not folder_path.is_dir():
            continue

        for mat_file in folder_path.glob("*.mat"):
            if files_processed >= num_files:
                break

            print(f"\nProcessing: {mat_file.relative_to(dataset_root)}")

            try:
                mat = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
            except Exception as e:
                print(f"  ✗ Load error: {e}")
                continue

            # Try all subsets
            for subset_name in ["DS", "FS", "Upper", "Lower"]:
                if subset_name not in mat:
                    continue

                struct = mat[subset_name]

                # Extract signals
                try:
                    raw_signals = struct.rawData
                    if not hasattr(raw_signals, '__len__'):
                        raw_signals = [raw_signals]
                    signals = [np.asarray(s, dtype=np.float64).reshape(-1) for s in raw_signals]
                except Exception as e:
                    print(f"  ✗ Signal extraction error: {e}")
                    continue

                # Get metadata
                try:
                    sampling_rate = float(np.asarray(struct.samplingRate).reshape(-1)[0])
                    rpm = float(np.asarray(struct.RPM).reshape(-1)[0])
                    labels = np.asarray(struct.label).reshape(-1)
                    fault_mult = struct.faultFrequencies
                except Exception as e:
                    print(f"  ✗ Metadata error: {e}")
                    continue

                # Process healthy samples only (label == 0)
                for idx, (signal, label) in enumerate(zip(signals, labels)):
                    if label != 0:  # Only healthy
                        continue

                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                    signal = signal - np.mean(signal)

                    # Quality gate
                    result, rejection = quality_gate(
                        signal=signal,
                        rpm=rpm,
                        fs=sampling_rate,
                        fault_mult=fault_mult,
                    )
                    if result is None:
                        continue

                    # Preprocess
                    try:
                        proc = preprocess_signal(result.signal, fs=sampling_rate)
                        envelope = proc["envelope"]
                    except Exception as e:
                        print(f"    ✗ Preprocess error: {e}")
                        continue

                    # Extract windows
                    try:
                        X, Y = create_windows(envelope, window_size=512)
                        if X.shape[0] > 0:
                            all_windows.append(X)
                            print(f"    ✓ {subset_name}[{idx}]: {X.shape[0]} windows")
                    except Exception as e:
                        print(f"    ✗ Windowing error: {e}")
                        continue

            files_processed += 1
            if files_processed >= num_files:
                break

    if len(all_windows) == 0:
        print("\n✗ No windows extracted!")
        return np.array([])

    windows = np.vstack(all_windows)
    print(f"\nTotal healthy windows: {windows.shape}")
    return windows


def main():
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    output_dir = root / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENVELOPE DYNAMICS PREDICTOR - END-TO-END TRAINING")
    print("=" * 70)

    # Collect healthy envelopes
    print("\n[Step 1] Extracting healthy envelope windows from dataset...")
    X_train = collect_healthy_envelopes(dataset_root, num_files=5)

    if X_train.size == 0:
        print("✗ Failed to extract training data!")
        return

    print(f"\n[Step 2] Training envelope dynamics predictor...")
    print(f"  Input shape: {X_train.shape}")

    # Train model
    model = train_envelope_dynamics_predictor(
        X_train=X_train,
        Y_train=None,
        epochs=15,
        batch_size=64,
        learning_rate=1e-3,
        device="cpu",
    )

    # Save model
    model_path = output_dir / "envelope_dynamics_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[Step 3] Model saved to {model_path}")

    # Test on a single window
    print(f"\n[Step 4] Testing inference...")
    test_window = torch.randn(1, X_train.shape[1]).float()
    with torch.no_grad():
        pred = model(test_window)
    print(f"  Input shape: {test_window.shape}")
    print(f"  Output shape: {pred.shape}")
    print(f"  Prediction valid: {torch.isfinite(pred).all().item()}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
