"""
Filter-bank canonical correlation analysis (FBCCA) for 12JFPM SSVEP.

Algorithm (Pan-style, default for PC / firmware / replay):
  - 10 Chebyshev Type I bandpasses (cheb1ord / cheby1 / filtfilt)
    with non-uniform pass/stop edges.
  - Band weights  w_i = i^(-1.25) + 0.25  for i = 1..10.
  - Fusion: score[k] += w_i * rho_{k,i}^2  where rho is the first
    canonical correlation from a ridge-regularized eigenvalue formulation.

Shared helpers (_bandpass_filter, _reference_matrix,
_first_canonical_correlation) are also used by cca_12jfpm.py.

Data contract: trial shape (T, C) or batch (N, T, C), T=512, C=8,
fs=256 Hz; class order TARGET_FREQUENCIES_HZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.datasets.parse_12jfpm import TARGET_FREQUENCIES_HZ

__all__ = [
    "FBCCAConfig",
    "band_weights",
    "fbcca_scores_for_trial",
    "predict_fbcca_class",
    "fbcca_predict_batch",
    "accuracy",
    "_bandpass_filter",
    "_reference_matrix",
    "_first_canonical_correlation",
]


# ---------------------------------------------------------------------------
# Shared numerical helpers (also used by CCA)
# ---------------------------------------------------------------------------

def _bandpass_filter(
    x: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along time axis."""
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 1e-6)
    hi = min(high_hz / nyq, 1.0 - 1e-6)
    if lo >= hi:
        raise ValueError(f"Invalid band [{low_hz}, {high_hz}] Hz for fs={fs}")
    b, a = signal.butter(order, [lo, hi], btype="band")
    if x.ndim == 1:
        return signal.filtfilt(b, a, x).astype(np.float32)
    out = np.zeros_like(x, dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = signal.filtfilt(b, a, x[:, c])
    return out.astype(np.float32)


def _reference_matrix(
    n_samples: int,
    fs: float,
    fundamental_hz: float,
    n_harmonics: int,
) -> np.ndarray:
    """Sine/cosine reference signals: rows [sin, cos] for harmonics 1..H."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    rows: list[np.ndarray] = []
    two_pi = 2.0 * np.pi
    for h in range(1, n_harmonics + 1):
        fh = fundamental_hz * h
        rows.append(np.sin(two_pi * fh * t))
        rows.append(np.cos(two_pi * fh * t))
    return np.stack(rows, axis=0).astype(np.float64)


def _first_canonical_correlation(
    x_channels_time: np.ndarray,
    y_refs_time: np.ndarray,
    reg: float,
) -> float:
    """First canonical correlation coefficient magnitude in [0, 1]."""
    x_channels_time = np.asarray(x_channels_time, dtype=np.float64)
    y_refs_time = np.asarray(y_refs_time, dtype=np.float64)
    t, c = x_channels_time.shape
    _, r = y_refs_time.shape
    if t < max(c, r) + 2:
        return 0.0

    xc = x_channels_time - np.mean(x_channels_time, axis=0, keepdims=True)
    yc = y_refs_time - np.mean(y_refs_time, axis=0, keepdims=True)
    n = t - 1 if t > 1 else 1
    cxx = (xc.T @ xc) / n
    cyy = (yc.T @ yc) / n
    cxy = (xc.T @ yc) / n
    c1, c2 = cxx.shape[0], cyy.shape[0]
    cxx = cxx + reg * np.eye(c1)
    cyy = cyy + reg * np.eye(c2)

    try:
        inv_cxx = np.linalg.inv(cxx)
        inv_cyy = np.linalg.inv(cyy)
    except np.linalg.LinAlgError:
        inv_cxx = np.linalg.pinv(cxx)
        inv_cyy = np.linalg.pinv(cyy)
    mat = inv_cxx @ cxy @ inv_cyy @ cxy.T
    eigvals = np.linalg.eigvals(mat)
    rho_sq = float(np.nanmax(np.real(eigvals)))
    rho_sq = float(np.clip(rho_sq, 0.0, 1.0))
    return float(np.sqrt(rho_sq))


# ---------------------------------------------------------------------------
# Pan-style FBCCA
# ---------------------------------------------------------------------------

def band_weights(n_bands: int) -> list[float]:
    """w_i = i^(-1.25) + 0.25 for i = 1..n_bands."""
    return [float((i + 1) ** (-1.25) + 0.25) for i in range(n_bands)]


@dataclass
class FBCCAConfig:
    """Frozen Pan-style defaults for 12JFPM benchmark."""

    sampling_rate_hz: float = 256.0
    n_harmonics: int = 3
    cca_reg: float = 1e-4
    passband_hz: tuple[float, ...] = (6.0, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0, 78.0)
    stopband_hz: tuple[float, ...] = (4.0, 10.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0)
    highcut_pass_hz: float = 80.0
    highcut_stop_hz: float = 90.0
    gpass_db: float = 3.0
    gstop_db: float = 40.0
    cheby_ripple_db: float = 0.5


def _filter_one_band_pan(
    x_tc: np.ndarray,
    *,
    fs: float,
    pass_lo_hz: float,
    stop_lo_hz: float,
    high_pass_hz: float,
    high_stop_hz: float,
    gpass: float,
    gstop: float,
    rp: float,
    padlen: int,
) -> np.ndarray:
    """One Chebyshev Type I bandpass with uniform padlen."""
    x_tc = np.asarray(x_tc, dtype=np.float64)
    nyq = 0.5 * fs
    wp = [pass_lo_hz / nyq, high_pass_hz / nyq]
    ws = [stop_lo_hz / nyq, high_stop_hz / nyq]
    n, wn = signal.cheb1ord(wp, ws, gpass, gstop)
    b, a = signal.cheby1(n, rp, wn, btype="bandpass")
    out = np.zeros_like(x_tc, dtype=np.float64)
    for c in range(x_tc.shape[1]):
        out[:, c] = signal.filtfilt(b, a, x_tc[:, c], padlen=padlen)
    return out.astype(np.float32)


def _uniform_filtfilt_padlen(cfg: FBCCAConfig) -> int:
    """Max scipy filtfilt padlen across all bands."""
    fs = cfg.sampling_rate_hz
    nyq = 0.5 * fs
    pb = cfg.passband_hz
    sb = cfg.stopband_hz
    m = 0
    for i in range(len(pb)):
        wp = [pb[i] / nyq, cfg.highcut_pass_hz / nyq]
        ws = [sb[i] / nyq, cfg.highcut_stop_hz / nyq]
        n, wn = signal.cheb1ord(wp, ws, cfg.gpass_db, cfg.gstop_db)
        b, a = signal.cheby1(n, cfg.cheby_ripple_db, wn, btype="bandpass")
        padlen = 3 * (max(len(b), len(a)) - 1)
        m = max(m, padlen)
    return int(m)


def _filter_bank(x_tc: np.ndarray, cfg: FBCCAConfig) -> list[np.ndarray]:
    """Return list of 10 band-pass filtered (T, C) arrays."""
    pb = cfg.passband_hz
    sb = cfg.stopband_hz
    assert len(pb) == len(sb)
    padlen = _uniform_filtfilt_padlen(cfg)
    bands: list[np.ndarray] = []
    for i in range(len(pb)):
        bands.append(
            _filter_one_band_pan(
                x_tc,
                fs=cfg.sampling_rate_hz,
                pass_lo_hz=pb[i],
                stop_lo_hz=sb[i],
                high_pass_hz=cfg.highcut_pass_hz,
                high_stop_hz=cfg.highcut_stop_hz,
                gpass=cfg.gpass_db,
                gstop=cfg.gstop_db,
                rp=cfg.cheby_ripple_db,
                padlen=padlen,
            )
        )
    return bands


def fbcca_scores_for_trial(
    trial_tc: np.ndarray,
    cfg: FBCCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    """Fused scores shape (12,). Fusion: sum_i w_i * rho_{k,i}^2."""
    cfg = cfg or FBCCAConfig()
    freqs = target_frequencies_hz or TARGET_FREQUENCIES_HZ
    n_cls = len(freqs)
    t, _c = trial_tc.shape
    weights = band_weights(len(cfg.passband_hz))

    bands = _filter_bank(trial_tc, cfg)
    scores = np.zeros(n_cls, dtype=np.float64)

    for x_band, wb in zip(bands, weights, strict=True):
        for k in range(n_cls):
            fk = float(freqs[k])
            y_ref = _reference_matrix(t, cfg.sampling_rate_hz, fk, cfg.n_harmonics).T
            rho = _first_canonical_correlation(x_band, y_ref, cfg.cca_reg)
            scores[k] += wb * (rho * rho)

    return scores.astype(np.float32)


def predict_fbcca_class(
    trial_tc: np.ndarray,
    cfg: FBCCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> int:
    s = fbcca_scores_for_trial(trial_tc, cfg, target_frequencies_hz=target_frequencies_hz)
    return int(np.argmax(s))


def fbcca_predict_batch(
    x: np.ndarray,
    cfg: FBCCAConfig | None = None,
    *,
    target_frequencies_hz: list[float] | None = None,
) -> np.ndarray:
    cfg = cfg or FBCCAConfig()
    n = x.shape[0]
    pred = np.empty(n, dtype=np.int64)
    for i in range(n):
        pred[i] = predict_fbcca_class(x[i], cfg, target_frequencies_hz=target_frequencies_hz)
    return pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))
