"""Quick demo of envelope dynamics training on minimal data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

from src.preprocessing.quality_gate import quality_gate
from src.preprocessing.preprocess import preprocess_signal
from src.utils.windowing import create_windows
from train_envelope_dynamics import train_envelope_dynamics_predictor


def main():
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    output_dir = root / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENVELOPE DYNAMICS PREDICTOR - DEMO")
    print("=" * 70)

    # Load just first file for quick demo
    print("\n[Step 1] Extracting healthy envelope windows...")
    all_windows = []
    
    mat_file = dataset_root / "1" / "train.mat"
    mat = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    
    for subset_name in ["DS", "FS"]:
        if subset_name not in mat:
            continue
        
        struct = mat[subset_name]
        raw_signals = struct.rawData
        if not hasattr(raw_signals, '__len__'):
            raw_signals = [raw_signals]
        signals = [np.asarray(s, dtype=np.float64).reshape(-1) for s in raw_signals]
        
        sampling_rate = float(np.asarray(struct.samplingRate).reshape(-1)[0])
        rpm = float(np.asarray(struct.RPM).reshape(-1)[0])
        labels = np.asarray(struct.label).reshape(-1)
        fault_mult = struct.faultFrequencies
        
        # Take only first 5 healthy samples per subset
        count = 0
        for idx, (signal, label) in enumerate(zip(signals, labels)):
            if count >= 5:
                break
            if label != 0:
                continue
            
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            signal = signal - np.mean(signal)
            
            result, _ = quality_gate(signal, rpm, sampling_rate, fault_mult)
            if result is None:
                continue
            
            proc = preprocess_signal(result.signal, fs=sampling_rate)
            envelope = proc["envelope"]
            
            X, Y = create_windows(envelope, window_size=512)
            if X.shape[0] > 0:
                all_windows.append(X)
                print(f"  ✓ {subset_name}[{idx}]: {X.shape[0]} windows")
                count += 1
    
    if len(all_windows) == 0:
        print("✗ No windows extracted!")
        return
    
    X_train = np.vstack(all_windows)
    print(f"\nTotal windows: {X_train.shape}")
    
    # Train model
    print(f"\n[Step 2] Training envelope dynamics predictor...")
    model = train_envelope_dynamics_predictor(
        X_train=X_train,
        Y_train=None,
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        device="cpu",
        return_loss_history=True,
    )
    model, loss_history = model

    first_loss = float(loss_history[0])
    final_loss = float(loss_history[-1])
    finite_losses = bool(np.all(np.isfinite(loss_history)))
    deltas = np.diff(np.asarray(loss_history, dtype=np.float64))
    steady_ratio = float(np.mean(deltas <= 0.0)) if deltas.size > 0 else 1.0
    steady_decrease = steady_ratio >= 0.8 and final_loss <= first_loss

    print(f"\n[Validation]")
    print(f"  Loss @ epoch 1: {first_loss:.6f}")
    print(f"  Loss @ final epoch: {final_loss:.6f}")
    print(f"  Finite losses (no NaN/Inf): {finite_losses}")
    print(f"  Steady decrease check (>=80% non-increasing): {steady_decrease}")

    plot_path = output_dir / "envelope_dynamics_loss_curve.png"
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.8)
    plt.title("Envelope Dynamics Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()
    print(f"  Loss curve saved: {plot_path}")
    
    # Save model
    model_path = output_dir / "envelope_dynamics_demo.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[Step 3] Model saved to {model_path}")
    
    # Test inference
    print(f"\n[Step 4] Testing inference...")
    test_window = torch.randn(1, X_train.shape[1]).float()
    with torch.no_grad():
        pred = model(test_window)
    print(f"  Input shape: {test_window.shape}")
    print(f"  Output shape: {pred.shape}")
    print(f"  Prediction valid: {torch.isfinite(pred).all().item()}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
