"""Reference signal generation for classical SSVEP methods."""

from __future__ import annotations

import numpy as np


def generate_references(
    target_frequencies: list[float] | tuple[float, ...],
    *,
    fs_hz: float,
    window_samples: int,
    harmonics: int,
) -> np.ndarray:
    """Return references with shape [n_classes, time, 2 * harmonics]."""
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if harmonics <= 0:
        raise ValueError("harmonics must be positive")
    t = np.arange(window_samples, dtype=np.float64) / float(fs_hz)
    refs = np.zeros((len(target_frequencies), window_samples, 2 * harmonics), dtype=np.float64)
    for class_idx, freq in enumerate(target_frequencies):
        if float(freq) <= 0:
            raise ValueError("target frequencies must be positive")
        for h in range(1, harmonics + 1):
            refs[class_idx, :, 2 * (h - 1)] = np.sin(2.0 * np.pi * h * float(freq) * t)
            refs[class_idx, :, 2 * (h - 1) + 1] = np.cos(2.0 * np.pi * h * float(freq) * t)
    return refs
