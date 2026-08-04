"""Dataset-neutral Multivariate Synchronization Index."""

from __future__ import annotations

import numpy as np

from src.classical.filters import ensure_trials_time_channels


def _sync_index(X_ct: np.ndarray, Y_rt: np.ndarray) -> float:
    X = X_ct - X_ct.mean(axis=1, keepdims=True)
    Y = Y_rt - Y_rt.mean(axis=1, keepdims=True)
    X = X / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
    Y = Y / np.maximum(Y.std(axis=1, keepdims=True), 1e-12)
    T = X.shape[1]
    c11 = (X @ X.T) / T
    c22 = (Y @ Y.T) / T
    c12 = (X @ Y.T) / T
    C = np.block([[c11, c12], [c12.T, c22]])
    e1, q1 = np.linalg.eigh(c11)
    e2, q2 = np.linalg.eigh(c22)
    inv1 = q1 @ np.diag(np.maximum(e1, 1e-12) ** -0.5) @ q1.T
    inv2 = q2 @ np.diag(np.maximum(e2, 1e-12) ** -0.5) @ q2.T
    U = np.block([[inv1, np.zeros((X.shape[0], Y.shape[0]))], [np.zeros((Y.shape[0], X.shape[0])), inv2]])
    R = U @ C @ U.T
    eigvals = np.maximum(np.real(np.linalg.eigvals(R)), 1e-20)
    p = eigvals / eigvals.sum()
    return float(1.0 + np.sum(p * np.log(p)) / np.log(len(p)))


def msi_scores(X: np.ndarray, references: np.ndarray) -> np.ndarray:
    X = ensure_trials_time_channels(X)
    refs = np.asarray(references, dtype=np.float64)
    if refs.ndim != 3 or refs.shape[1] != X.shape[1]:
        raise ValueError("references must be [classes, time, ref_features]")
    out = np.zeros((X.shape[0], refs.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        x_ct = X[i].T
        for cls in range(refs.shape[0]):
            out[i, cls] = _sync_index(x_ct, refs[cls].T)
    return out.astype(np.float32)


def msi_predict(X: np.ndarray, references: np.ndarray, *, labels: list[int] | None = None) -> np.ndarray:
    scores = msi_scores(X, references)
    labels_arr = np.asarray(labels if labels is not None else list(range(scores.shape[1])), dtype=np.int64)
    return labels_arr[np.argmax(scores, axis=1)]
