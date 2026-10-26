"""Dataset-neutral filter-bank CCA."""

from __future__ import annotations

import numpy as np

from src.analytical.cca import cca_scores
from src.analytical.filters import cheby1_filter_bank, ensure_trials_time_channels


def pan_weights(n_bands: int) -> np.ndarray:
    return np.asarray([(i + 1) ** (-1.25) + 0.25 for i in range(n_bands)], dtype=np.float64)


def fbcca_scores(
    X: np.ndarray,
    references: np.ndarray,
    *,
    fs_hz: float,
    passband_low_hz: list[float] | tuple[float, ...],
    stopband_low_hz: list[float] | tuple[float, ...],
    highcut_pass_hz: float,
    highcut_stop_hz: float,
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    reg: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Return per-subband scores and diagnostic fusions."""
    X = ensure_trials_time_channels(X)
    FB = cheby1_filter_bank(
        X,
        fs_hz=fs_hz,
        passband_low_hz=passband_low_hz,
        stopband_low_hz=stopband_low_hz,
        highcut_pass_hz=highcut_pass_hz,
        highcut_stop_hz=highcut_stop_hz,
    )
    per = np.stack([cca_scores(FB[bi], references, reg=reg) for bi in range(FB.shape[0])], axis=0)
    w = pan_weights(FB.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != FB.shape[0]:
        raise ValueError("weights length must match number of bands")
    weighted = np.tensordot(w, per * per, axes=(0, 0))
    equal = np.mean(per * per, axis=0)
    return {
        "per_subband_scores": per.astype(np.float32),
        "weighted": weighted.astype(np.float32),
        "first_subband": (per[0] * per[0]).astype(np.float32),
        "equal_weight": equal.astype(np.float32),
        "first_3_subbands": np.mean((per[: min(3, per.shape[0])] ** 2), axis=0).astype(np.float32),
    }


def fbcca_predict(X: np.ndarray, references: np.ndarray, *, labels: list[int] | None = None, fusion: str = "weighted", **kwargs) -> np.ndarray:
    scores = fbcca_scores(X, references, **kwargs)[fusion]
    labels_arr = np.asarray(labels if labels is not None else list(range(scores.shape[1])), dtype=np.int64)
    return labels_arr[np.argmax(scores, axis=1)]
