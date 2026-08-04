"""Dataset-neutral canonical correlation analysis for SSVEP."""

from __future__ import annotations

import numpy as np

from src.classical.filters import bandpass_filter, ensure_trials_time_channels


def first_canonical_correlation(X: np.ndarray, Y: np.ndarray, *, reg: float = 1e-4) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be [time, features] with matching time")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    n = max(X.shape[0] - 1, 1)
    cxx = (X.T @ X) / n + reg * np.eye(X.shape[1])
    cyy = (Y.T @ Y) / n + reg * np.eye(Y.shape[1])
    cxy = (X.T @ Y) / n
    mat = np.linalg.pinv(cxx) @ cxy @ np.linalg.pinv(cyy) @ cxy.T
    eigvals = np.real(np.linalg.eigvals(mat))
    rho_sq = float(np.clip(np.nanmax(eigvals), 0.0, 1.0))
    return float(np.sqrt(rho_sq))


def cca_scores(
    X: np.ndarray,
    references: np.ndarray,
    *,
    fs_hz: float | None = None,
    bandpass_hz: tuple[float, float] | None = None,
    filter_order: int = 4,
    reg: float = 1e-4,
) -> np.ndarray:
    """Return scores with shape [trials, classes]."""
    X = ensure_trials_time_channels(X)
    if bandpass_hz is not None:
        if fs_hz is None:
            raise ValueError("fs_hz is required when bandpass_hz is set")
        X = bandpass_filter(X, fs_hz=fs_hz, low_hz=bandpass_hz[0], high_hz=bandpass_hz[1], order=filter_order)
    refs = np.asarray(references, dtype=np.float64)
    if refs.ndim != 3 or refs.shape[1] != X.shape[1]:
        raise ValueError("references must be [classes, time, ref_features]")
    out = np.zeros((X.shape[0], refs.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        for cls in range(refs.shape[0]):
            out[i, cls] = first_canonical_correlation(X[i], refs[cls], reg=reg)
    return out.astype(np.float32)


def cca_predict(
    X: np.ndarray,
    references: np.ndarray,
    *,
    labels: list[int] | None = None,
    **kwargs,
) -> np.ndarray:
    scores = cca_scores(X, references, **kwargs)
    labels_arr = np.asarray(labels if labels is not None else list(range(scores.shape[1])), dtype=np.int64)
    return labels_arr[np.argmax(scores, axis=1)]
