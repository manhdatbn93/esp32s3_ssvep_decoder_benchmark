"""
Task-Related Component Analysis (TRCA) for 12JFPM SSVEP.

TRCA is a training-based method: spatial filters are learned from
training trials to maximize inter-trial covariance for each class,
then applied to test trials for classification via correlation with
class templates.

This implements ensemble TRCA (eTRCA) by default, where spatial
filters from all classes are concatenated for richer projection.

The published evaluation uses the deployed second-order-section filter bank
(``deployed_sos``). The historical transfer-function candidate
(``screening_ba``) is a separate realization of the same band specification.

Data contract: trial shape (T, C), T=512, C=8, fs=256 Hz.
Requires train/test split per LOSO fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sig
from scipy.sparse.linalg import eigs

from src.datasets.parse_12jfpm import TARGET_FREQUENCIES_HZ

__all__ = [
    "TRCAConfig",
    "TRCAModel",
    "trca_predict_fold",
    "TRCA_REALIZATIONS",
]

# deployed_sos is the filter-bank realization exported to the ESP32-S3 and
# used by the published host reference. screening_ba is the earlier
# transfer-function candidate and is not that reference.
TRCA_REALIZATIONS = ("deployed_sos", "screening_ba")

N_CLASSES = len(TARGET_FREQUENCIES_HZ)


@dataclass
class TRCAConfig:
    sampling_rate_hz: float = 256.0
    n_bands: int = 10
    ensemble: bool = True
    passband_hz: tuple[float, ...] = (6.0, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0, 78.0)
    stopband_hz: tuple[float, ...] = (4.0, 10.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0)
    highcut_pass_hz: float = 80.0
    highcut_stop_hz: float = 90.0
    gpass_db: float = 3.0
    gstop_db: float = 40.0
    cheby_ripple_db: float = 0.5


def _apply_filter_bank(
    X: np.ndarray, cfg: TRCAConfig,
) -> np.ndarray:
    """
    Apply n_bands Chebyshev filter bank to X of shape (N, T, C).
    Returns (n_bands, N, C, T) — note channel-first for TRCA internal use.
    """
    fs = cfg.sampling_rate_hz
    nyq = 0.5 * fs
    N, T, C = X.shape
    X_ct = X.transpose(0, 2, 1)  # (N, C, T)
    result = np.zeros((cfg.n_bands, N, C, T), dtype=np.float64)

    for i in range(cfg.n_bands):
        wp = [cfg.passband_hz[i] / nyq, cfg.highcut_pass_hz / nyq]
        ws = [cfg.stopband_hz[i] / nyq, cfg.highcut_stop_hz / nyq]
        n, wn = sig.cheb1ord(wp, ws, cfg.gpass_db, cfg.gstop_db)
        b, a = sig.cheby1(n, cfg.cheby_ripple_db, wn, btype="bandpass")
        padlen = 3 * (max(len(b), len(a)) - 1)
        result[i] = sig.filtfilt(b, a, X_ct, axis=-1, padlen=padlen)

    return result


def _trca_eigvec(eeg_trials: np.ndarray) -> np.ndarray:
    """
    Compute first TRCA eigenvector from trials (C, T, n_trials).
    Returns w of shape (C,).
    """
    n_ch, n_t, n_tr = eeg_trials.shape
    S = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_tr - 1):
        x1 = eeg_trials[:, :, i] - eeg_trials[:, :, i].mean(axis=1, keepdims=True)
        for j in range(i + 1, n_tr):
            x2 = eeg_trials[:, :, j] - eeg_trials[:, :, j].mean(axis=1, keepdims=True)
            S += x1 @ x2.T + x2 @ x1.T

    UX = eeg_trials.reshape(n_ch, n_t * n_tr)
    UX = UX - UX.mean(axis=1, keepdims=True)
    Q = UX @ UX.T

    Q += 1e-6 * np.eye(n_ch)
    try:
        _, V = eigs(S, k=1, M=Q, which="LM")
        w = np.real(V[:, 0])
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(Q, S))
        w = eigvecs[:, -1]
    return w


class TRCAModel:
    """Fitted TRCA model for one filter-bank configuration."""

    def __init__(self, cfg: TRCAConfig | None = None):
        self.cfg = cfg or TRCAConfig()
        self.templates = None  # (n_classes, n_bands, C, T)
        self.W = None  # (n_bands, n_classes, C)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit TRCA on training data.
        X_train: (N, T, C) float32
        y_train: (N,) int64 class labels 0..11
        """
        cfg = self.cfg
        FB = _apply_filter_bank(X_train, cfg)  # (n_bands, N, C, T)

        n_classes = N_CLASSES
        self.templates = np.zeros((n_classes, cfg.n_bands, FB.shape[2], FB.shape[3]))
        self.W = np.zeros((cfg.n_bands, n_classes, FB.shape[2]))

        for cls in range(n_classes):
            mask = y_train == cls
            for bi in range(cfg.n_bands):
                trials_ct = FB[bi, mask]  # (n_trials, C, T)
                self.templates[cls, bi] = trials_ct.mean(axis=0)
                trials_ctn = trials_ct.transpose(1, 2, 0)  # (C, T, n_trials)
                w = _trca_eigvec(trials_ctn)
                self.W[bi, cls] = w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for X_test (N, T, C)."""
        cfg = self.cfg
        FB = _apply_filter_bank(X_test, cfg)  # (n_bands, N, C, T)
        n_test = X_test.shape[0]
        n_classes = N_CLASSES

        fb_coefs = np.array([
            (i + 1) ** (-1.25) + 0.25 for i in range(cfg.n_bands)
        ])

        preds = np.empty(n_test, dtype=np.int64)
        for trial in range(n_test):
            rho = np.zeros(n_classes)
            for bi in range(cfg.n_bands):
                test_ct = FB[bi, trial]  # (C, T)
                for cls in range(n_classes):
                    template = self.templates[cls, bi]
                    if cfg.ensemble:
                        w = self.W[bi].T  # (C, n_classes)
                    else:
                        w = self.W[bi, cls:cls+1].T  # (C, 1)
                    proj_test = (test_ct.T @ w).flatten()
                    proj_template = (template.T @ w).flatten()
                    r = np.corrcoef(proj_test, proj_template)[0, 1]
                    rho[cls] += fb_coefs[bi] * r
            preds[trial] = int(np.argmax(rho))
        return preds


def _cheby_bank_params(cfg: TRCAConfig) -> dict:
    return {
        "passband_hz": list(cfg.passband_hz),
        "stopband_hz": list(cfg.stopband_hz),
        "highcut_pass_hz": cfg.highcut_pass_hz,
        "highcut_stop_hz": cfg.highcut_stop_hz,
        "gpass_db": cfg.gpass_db,
        "gstop_db": cfg.gstop_db,
        "cheby_ripple_db": cfg.cheby_ripple_db,
    }


def _predict_deployed_sos(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: TRCAConfig,
) -> np.ndarray:
    """Score with the SOS fit/predict used to export the deployed TRCA constants."""
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from export.common import compute_cheby_filter_bank
    from export.export_trca_constants import _trca_fit_sos, _trca_predict_sos

    banks = compute_cheby_filter_bank(_cheby_bank_params(cfg))
    templates, spatial_filters = _trca_fit_sos(X_train, y_train, cfg, banks)
    predictions, _scores = _trca_predict_sos(X_test, templates, spatial_filters, cfg, banks)
    return predictions


def trca_predict_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: TRCAConfig | None = None,
    *,
    realization: str = "deployed_sos",
) -> np.ndarray:
    """Fit on the training trials and return test labels.

    ``deployed_sos`` reuses the second-order-section filter bank in
    ``export_trca_constants`` (the published host and firmware contract).
    ``screening_ba`` keeps the historical transfer-function ``filtfilt``
    candidate. Other classical backends are unaffected.
    """
    if realization not in TRCA_REALIZATIONS:
        raise ValueError(
            f"Unknown TRCA realization {realization!r}; expected one of {TRCA_REALIZATIONS}"
        )
    cfg = cfg or TRCAConfig()
    if realization == "screening_ba":
        model = TRCAModel(cfg)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    return _predict_deployed_sos(X_train, y_train, X_test, cfg)
