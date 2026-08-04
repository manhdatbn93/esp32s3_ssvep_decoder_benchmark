"""Dataset-neutral filtering helpers for [trials, time, channels] arrays."""

from __future__ import annotations

import numpy as np
from scipy import signal


def validate_band_edges(low_hz: float, high_hz: float, *, fs_hz: float) -> None:
    nyq = 0.5 * float(fs_hz)
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError("filter edges must be positive")
    if low_hz >= high_hz:
        raise ValueError(f"invalid band: low_hz={low_hz} >= high_hz={high_hz}")
    if high_hz >= nyq:
        raise ValueError(f"high_hz={high_hz} must be below Nyquist={nyq}")


def ensure_trials_time_channels(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected X shape [trials, time, channels], got {X.shape}")
    if X.shape[1] < X.shape[2]:
        raise ValueError(f"time axis appears too short for [trials, time, channels]: {X.shape}")
    return X


def bandpass_filter(
    X: np.ndarray,
    *,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass along the time axis."""
    validate_band_edges(low_hz, high_hz, fs_hz=fs_hz)
    X = ensure_trials_time_channels(np.asarray(X, dtype=np.float64))
    wn = [low_hz / (0.5 * fs_hz), high_hz / (0.5 * fs_hz)]
    b, a = signal.butter(order, wn, btype="band")
    out = signal.filtfilt(b, a, X, axis=1)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("bandpass output contains non-finite values")
    return out.astype(np.float32)


def cheby1_filter_bank(
    X: np.ndarray,
    *,
    fs_hz: float,
    passband_low_hz: list[float] | tuple[float, ...],
    stopband_low_hz: list[float] | tuple[float, ...],
    highcut_pass_hz: float,
    highcut_stop_hz: float,
    gpass_db: float = 3.0,
    gstop_db: float = 40.0,
    ripple_db: float = 0.5,
) -> np.ndarray:
    """Return filter-bank output with shape [bands, trials, time, channels]."""
    X = ensure_trials_time_channels(np.asarray(X, dtype=np.float64))
    if len(passband_low_hz) != len(stopband_low_hz):
        raise ValueError("passband_low_hz and stopband_low_hz must have same length")
    validate_band_edges(min(passband_low_hz), highcut_pass_hz, fs_hz=fs_hz)
    validate_band_edges(min(stopband_low_hz), highcut_stop_hz, fs_hz=fs_hz)
    nyq = 0.5 * fs_hz
    bands = []
    for pass_lo, stop_lo in zip(passband_low_hz, stopband_low_hz, strict=True):
        wp = [float(pass_lo) / nyq, float(highcut_pass_hz) / nyq]
        ws = [float(stop_lo) / nyq, float(highcut_stop_hz) / nyq]
        n, wn = signal.cheb1ord(wp, ws, gpass_db, gstop_db)
        b, a = signal.cheby1(n, ripple_db, wn, btype="bandpass")
        bands.append(signal.filtfilt(b, a, X, axis=1))
    out = np.stack(bands, axis=0).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("filter-bank output contains non-finite values")
    return out
