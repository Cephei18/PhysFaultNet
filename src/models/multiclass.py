"""Multiclass fault classification utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_multiclass_fault_classifier(
    X_features: Any,
    y_multiclass: Any,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Train a multiclass fault classifier and return test predictions.

    Returns
    -------
    model, X_test, y_test, y_pred
    """
    X = np.asarray(X_features, dtype=np.float64)
    y = np.asarray(y_multiclass).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X_features must be a 2D array of shape (N, d)")
    if y.ndim != 1:
        raise ValueError("y_multiclass must be a 1D array of shape (N,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_features and y_multiclass must have the same number of samples")
    if X.shape[0] < 2:
        raise ValueError("At least 2 samples are required")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


def train_fault_only_classifier(
    X_features: Any,
    y_multiclass: Any,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Train only on faulty classes (1, 2, 3), excluding healthy class 0.

    Returns
    -------
    model, X_test, y_test, y_pred
    """
    X = np.asarray(X_features, dtype=np.float64)
    y = np.asarray(y_multiclass).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X_features must be a 2D array of shape (N, d)")
    if y.ndim != 1:
        raise ValueError("y_multiclass must be a 1D array of shape (N,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_features and y_multiclass must have the same number of samples")

    mask = y != 0
    X_fault = X[mask]
    y_fault = y[mask]

    if X_fault.shape[0] < 2:
        raise ValueError("Not enough faulty samples after filtering")
    if np.unique(y_fault).size < 2:
        raise ValueError("Fault-only training requires at least two fault classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X_fault,
        y_fault,
        test_size=0.2,
        stratify=y_fault,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


def evaluate_multiclass_classifier(y_test: Any, y_pred: Any) -> tuple[float, str, np.ndarray]:
    """Evaluate multiclass predictions with accuracy, report, and confusion matrix."""
    y_true = np.asarray(y_test).reshape(-1)
    y_hat = np.asarray(y_pred).reshape(-1)

    if y_true.size == 0 or y_hat.size == 0:
        raise ValueError("y_test and y_pred must be non-empty")
    if y_true.shape[0] != y_hat.shape[0]:
        raise ValueError("y_test and y_pred must have the same length")

    acc = float(accuracy_score(y_true, y_hat))
    report = classification_report(y_true, y_hat)
    cm = confusion_matrix(y_true, y_hat)

    print("Accuracy:", acc)
    print(report)
    print(cm)

    return acc, report, cm
