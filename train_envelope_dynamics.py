"""Training for envelope dynamics predictor with proper batching and normalization."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.temporal_model import WindowCNNPredictor


def train_envelope_dynamics_predictor(
    X_train: np.ndarray,
    Y_train: np.ndarray | None = None,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    return_loss_history: bool = False,
) -> WindowCNNPredictor | tuple[WindowCNNPredictor, list[float]]:
    """
    Train 1D CNN predictor on healthy window envelopes.

    Parameters
    ----------
    X_train
        Input windows, shape (N, W) where N=num_windows, W=window_size.
    Y_train
        Target windows (optional). If None, use X_train as target (next-step prediction).
    epochs
        Number of training epochs (default: 15).
    batch_size
        Batch size for DataLoader (default: 64).
    learning_rate
        Adam learning rate (default: 1e-3).
    device
        Device to train on: "cpu" or "cuda" (default: "cpu").
    return_loss_history
        If True, also return per-epoch average training loss.

    Returns
    -------
    WindowCNNPredictor | tuple[WindowCNNPredictor, list[float]]
        Trained model (in eval mode). If return_loss_history=True,
        returns (model, loss_history).
    """
    # Ensure inputs are float64
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")

    N, W = X_train.shape
    print(f"Input: shape={X_train.shape}, dtype={X_train.dtype}")

    # Normalization: mean 0, std 1 per window
    X_mean = np.mean(X_train, axis=1, keepdims=True)
    X_std = np.std(X_train, axis=1, keepdims=True) + 1e-8
    X_normalized = (X_train - X_mean) / X_std

    print(f"Normalized: mean={X_normalized.mean():.6f}, std={X_normalized.std():.6f}")

    # If no target provided, use next-step prediction (shift by 1)
    if Y_train is None:
        Y_train = X_normalized[1:].copy()
        X_normalized = X_normalized[:-1].copy()
        print(f"Using next-step targets: X shape={X_normalized.shape}, Y shape={Y_train.shape}")
    else:
        Y_train = np.asarray(Y_train, dtype=np.float64)
        if Y_train.shape != X_normalized.shape:
            raise ValueError(f"Y_train shape {Y_train.shape} != X_train shape {X_normalized.shape}")
        print(f"Using provided targets: Y shape={Y_train.shape}")

    # Convert to torch tensors (float32 for GPU compatibility)
    X_tensor = torch.from_numpy(X_normalized).float()  # (N, W)
    Y_tensor = torch.from_numpy(Y_train).float()  # (N, W)

    # Reshape for model: (batch, 1, window_size)
    # Model will unsqueeze internally, so we keep as (batch, window_size)
    print(f"Tensor shapes: X={X_tensor.shape}, Y={Y_tensor.shape}")

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize model and optimizer
    model = WindowCNNPredictor()
    if torch.cuda.is_available() and device.lower() == "cuda":
        model = model.cuda()
        device = "cuda"
    else:
        model = model.cpu()
        device = "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Loss: MSELoss")
    print(f"Batch size: {batch_size}, Epochs: {epochs}\n")

    # Training loop
    model.train()
    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            pred = model(batch_x)  # output shape: (batch, window_size)

            # Compute loss
            loss = criterion(pred, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        loss_history.append(float(avg_loss))
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.6f}")

    model.eval()
    print(f"\n✅ Training complete. Model in eval mode.")
    if return_loss_history:
        return model, loss_history
    return model


if __name__ == "__main__":
    # Example: Generate synthetic healthy windows for testing
    print("Testing envelope dynamics predictor training...\n")

    # Create synthetic data: 1000 windows of 256 samples each
    X_train_synthetic = np.random.randn(1000, 256).astype(np.float64)

    # Train model
    trained_model = train_envelope_dynamics_predictor(
        X_train=X_train_synthetic,
        Y_train=None,
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
        device="cpu",
    )

    # Test inference
    test_window = torch.randn(1, 256).float()
    with torch.no_grad():
        pred = trained_model(test_window)
    print(f"\nTest inference: input shape={test_window.shape}, output shape={pred.shape}")
    print("✅ Test passed!")
