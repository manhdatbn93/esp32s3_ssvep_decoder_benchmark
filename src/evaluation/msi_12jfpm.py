"""
Multivariate Synchronization Index (MSI) for 12JFPM SSVEP.

Training-free method. Computes a synchronization index between
multichannel EEG and sine/cosine reference signals for each candidate
frequency. No filter bank — single-pass on the trial data.

Reference: Zhang et al. (2014), "Multivariate synchronization index
for frequency recognition of SSVEP-based brain-computer interface."

Data contract: trial shape (T, C), T=512, C=8, fs=256 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.datasets.parse_12jfpm import TARGET_FREQUENCIES_HZ

__all__ = [
    "MSIConfig",
    "msi_scores_for_trial",
    "predict_msi_class",
    "msi_predict_batch",
]


@dataclass
class MSIConfig:
    sampling_rate_hz: float = 256.0
    n_harmonics: int = 3


def _reference_matrix(n_samples: int, fs: float, freq_hz: float, n_harmonics: int) -> np.ndarray:
    """Sin/cos reference signals: shape (2*H, T)."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    rows = []
    for h in range(1, n_harmonics + 1):
        rows.append(np.sin(2.0 * np.pi * h * freq_hz * t))
        rows.append(np.cos(2.0 * np.pi * h * freq_hz * t))
    return np.stack(rows, axis=0)


def _synchronization_index(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute MSI between EEG channels X (C, T) and reference Y (2H, T).

    Returns a synchronization index S in [0, 1].
    """
    n_c, T = X.shape
    n_h = Y.shape[0]

    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    x_std = X.std(axis=1, keepdims=True)
    y_std = Y.std(axis=1, keepdims=True)
    x_std[x_std < 1e-12] = 1.0
    y_std[y_std < 1e-12] = 1.0
    X = X / x_std
    Y = Y / y_std

    c11 = (X @ X.T) / T
    c22 = (Y @ Y.T) / T
    c12 = (X @ Y.T) / T
    c21 = c12.T

    C = np.block([[c11, c12], [c21, c22]])

    eig_c11 = np.linalg.eigvalsh(c11)
    eig_c22 = np.linalg.eigvalsh(c22)
    eig_c11 = np.maximum(eig_c11, 1e-12)
    eig_c22 = np.maximum(eig_c22, 1e-12)

    D1 = np.diag(eig_c11 ** -0.5)
    D2 = np.diag(eig_c22 ** -0.5)
    _, Q1 = np.linalg.eigh(c11)
    _, Q2 = np.linalg.eigh(c22)
    C11_inv_sqrt = Q1 @ D1 @ Q1.T
    C22_inv_sqrt = Q2 @ D2 @ Q2.T

    U = np.block([
        [C11_inv_sqrt, np.zeros((n_c, n_h))],
        [np.zeros((n_h, n_c)), C22_inv_sqrt],
    ])

    R = U @ C @ U.T
    eigvals = np.real(np.linalg.eigvals(R))
    eigvals = np.maximum(eigvals, 1e-20)
    E = eigvals / eigvals.sum()
    S = 1.0 + np.sum(E * np.log(E)) / np.log(n_c + n_h)
    return float(S)


def msi_scores_for_trial(
    trial_tc: np.ndarray,
    cfg: MSIConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    """MSI synchronization index per class. Higher is better."""
    cfg = cfg or MSIConfig()
    freqs = target_frequencies_hz or TARGET_FREQUENCIES_HZ
    t, c = trial_tc.shape
    X = trial_tc.T  # (C, T)
    scores = np.zeros(len(freqs), dtype=np.float64)
    for k, fk in enumerate(freqs):
        Y = _reference_matrix(t, cfg.sampling_rate_hz, fk, cfg.n_harmonics)
        scores[k] = _synchronization_index(X, Y)
    return scores.astype(np.float32)


def predict_msi_class(
    trial_tc: np.ndarray,
    cfg: MSIConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> int:
    s = msi_scores_for_trial(trial_tc, cfg, target_frequencies_hz=target_frequencies_hz)
    return int(np.argmax(s))


def msi_predict_batch(
    x: np.ndarray,
    cfg: MSIConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    cfg = cfg or MSIConfig()
    n = x.shape[0]
    pred = np.empty(n, dtype=np.int64)
    for i in range(n):
        pred[i] = predict_msi_class(x[i], cfg, target_frequencies_hz=target_frequencies_hz)
    return pred
