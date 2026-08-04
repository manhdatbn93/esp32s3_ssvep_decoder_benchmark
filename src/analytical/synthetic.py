"""Deterministic synthetic SSVEP data for algorithm self-tests."""

from __future__ import annotations

import numpy as np


def make_synthetic_ssvep(
    *,
    target_frequencies: list[float] | tuple[float, ...],
    fs_hz: float,
    window_samples: int,
    channels: int,
    trials_per_class: int,
    noise_std: float = 0.25,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(window_samples, dtype=np.float64) / float(fs_hz)
    n_classes = len(target_frequencies)
    mix = rng.normal(size=(n_classes, channels, 4))
    X = np.zeros((n_classes * trials_per_class, window_samples, channels), dtype=np.float64)
    y = np.zeros(n_classes * trials_per_class, dtype=np.int64)
    row = 0
    for cls, freq in enumerate(target_frequencies):
        base = np.stack([
            np.sin(2 * np.pi * float(freq) * t),
            np.cos(2 * np.pi * float(freq) * t),
            0.5 * np.sin(4 * np.pi * float(freq) * t),
            0.5 * np.cos(4 * np.pi * float(freq) * t),
        ], axis=0)
        for _ in range(trials_per_class):
            jitter = 1.0 + 0.05 * rng.normal(size=(channels, 1))
            signal = (jitter * mix[cls]) @ base
            X[row] = signal.T + noise_std * rng.normal(size=(window_samples, channels))
            y[row] = cls
            row += 1
    return X.astype(np.float32), y
