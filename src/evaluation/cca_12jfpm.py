"""
Single-band canonical correlation analysis (CCA) for 12JFPM SSVEP.

One wide Butterworth bandpass and one first canonical correlation per
candidate class — no filter-bank fusion. This is the standard
"CCA-based SSVEP" baseline using sin/cos references at fundamental
plus harmonics.

Data contract: trial shape (T, C), T=512, C=8, fs=256 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.fbcca_12jfpm import (
    _bandpass_filter,
    _first_canonical_correlation,
    _reference_matrix,
)
from src.datasets.parse_12jfpm import TARGET_FREQUENCIES_HZ

__all__ = [
    "CCAConfig",
    "cca_scores_for_trial",
    "predict_cca_class",
    "cca_predict_batch",
    "accuracy",
]


@dataclass
class CCAConfig:
    """Single-band CCA frozen defaults for 12JFPM benchmark."""

    sampling_rate_hz: float = 256.0
    n_harmonics: int = 3
    single_band_hz: tuple[float, float] = (8.0, 55.0)
    butterworth_order: int = 4
    cca_reg: float = 1e-4


def cca_scores_for_trial(
    trial_tc: np.ndarray,
    cfg: CCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    """First canonical correlation per class, single bandpass."""
    cfg = cfg or CCAConfig()
    freqs = target_frequencies_hz or TARGET_FREQUENCIES_HZ
    n_cls = len(freqs)
    lo, hi = cfg.single_band_hz
    t, _c = trial_tc.shape
    scores = np.zeros(n_cls, dtype=np.float64)

    x_band = _bandpass_filter(
        trial_tc,
        fs=cfg.sampling_rate_hz,
        low_hz=lo,
        high_hz=hi,
        order=cfg.butterworth_order,
    )
    for k in range(n_cls):
        fk = float(freqs[k])
        y_ref = _reference_matrix(t, cfg.sampling_rate_hz, fk, cfg.n_harmonics).T
        rho = _first_canonical_correlation(x_band, y_ref, cfg.cca_reg)
        scores[k] = rho

    return scores.astype(np.float32)


def predict_cca_class(
    trial_tc: np.ndarray,
    cfg: CCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> int:
    s = cca_scores_for_trial(trial_tc, cfg, target_frequencies_hz=target_frequencies_hz)
    return int(np.argmax(s))


def cca_predict_batch(
    x: np.ndarray,
    cfg: CCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    cfg = cfg or CCAConfig()
    n = x.shape[0]
    pred = np.empty(n, dtype=np.int64)
    for i in range(n):
        pred[i] = predict_cca_class(
            x[i], cfg, target_frequencies_hz=target_frequencies_hz
        )
    return pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))
