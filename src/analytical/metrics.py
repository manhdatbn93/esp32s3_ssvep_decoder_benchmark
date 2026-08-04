"""Small metric helpers shared by classical runners."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int]) -> np.ndarray:
    index = {int(label): i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), strict=True):
        cm[index[int(t)], index[int(p)]] += 1
    return cm
