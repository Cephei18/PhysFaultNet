"""Training script for temporal CNN model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

from src.data import collect_dataset_records
from src.preprocessing import preprocess_signal
from src.features import extract_features
from src.utils import create_windows
from src.models import WindowCNNPredictor


def train_temporal_model(dataset_root: Path, output_dir: Path, epochs: int = 10, batch_size: int = 32) -> None:
    """Train a temporal CNN model on envelope windows."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    records_df, _ = collect_dataset_records(dataset_root)

    print("Extracting windows...")
    windows_list = []

    for idx, row in records_df.iterrows():
        signal_path = row["file_path"]
        sample_idx = row["sample_idx"]

        try:
            from scipy.io import loadmat

            mat = loadmat(signal_path, squeeze_me=True, struct_as_record=False)
            subset = row["subset"]
            raw_signals = mat[subset].rawData
            signals = []
            if isinstance(raw_signals, np.ndarray):
                if raw_signals.ndim == 2:
                    signals = [raw_signals[i] for i in range(raw_signals.shape[0])]
                else:
                    signals = [raw_signals]
            signal = np.asarray(signals[sample_idx]).reshape(-1)
        except Exception as e:
            print(f"  Skipping {idx}: {e}")
            continue

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = signal - np.mean(signal)

        try:
            proc = preprocess_signal(signal, row["sampling_rate"])
            envelope = proc["envelope"]
            X, Y = create_windows(envelope, window_size=512)
            if X.shape[0] > 0:
                windows_list.append(X)
        except Exception as e:
            print(f"  Skipping {idx}: {e}")
            continue

    if len(windows_list) == 0:
        print("No valid windows extracted!")
        return

    all_windows = np.vstack(windows_list)
    print(f"Total windows: {all_windows.shape[0]}")

    # Normalize
    all_windows = (all_windows - np.mean(all_windows)) / (np.std(all_windows) + 1e-8)

    X_tensor = torch.from_numpy(all_windows).float()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WindowCNNPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            pred = model(batch)
            # Target is the next window in sequence (shifted by 1)
            target = batch.clone()
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    # Save model
    model_path = output_dir / "temporal_cnn_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    root = Path("/home/teaching/Hackathon_dl")
    dataset_root = root / "SCA bearing dataset"
    output_dir = root / "analysis_outputs"

    train_temporal_model(dataset_root, output_dir, epochs=5, batch_size=32)
