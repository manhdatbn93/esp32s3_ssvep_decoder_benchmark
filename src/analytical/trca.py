"""Dataset-neutral single-band and ensemble TRCA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from src.classical.filters import bandpass_filter, cheby1_filter_bank, ensure_trials_time_channels


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom <= 1e-12 else float((a @ b) / denom)


def _trca_filter(trials_nct: np.ndarray, *, regularization: float) -> tuple[np.ndarray, dict[str, float | str]]:
    if trials_nct.ndim != 3:
        raise ValueError("TRCA class trials must be [trials, channels, time]")
    n_trials, n_ch, _n_time = trials_nct.shape
    if n_trials < 2:
        raise ValueError("TRCA requires at least two training trials per class")
    S = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_trials - 1):
        xi = trials_nct[i] - trials_nct[i].mean(axis=1, keepdims=True)
        for j in range(i + 1, n_trials):
            xj = trials_nct[j] - trials_nct[j].mean(axis=1, keepdims=True)
            S += xi @ xj.T + xj @ xi.T
    UX = trials_nct.transpose(1, 0, 2).reshape(n_ch, -1)
    UX = UX - UX.mean(axis=1, keepdims=True)
    Q = UX @ UX.T + regularization * np.eye(n_ch)
    try:
        eigvals, eigvecs = linalg.eigh(S, Q, check_finite=False)
        solver = "scipy.linalg.eigh"
    except Exception:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Q) @ S)
        solver = "numpy.linalg.eig_pinv"
    idx = int(np.argmax(np.real(eigvals)))
    w = np.real(eigvecs[:, idx])
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    return w.astype(np.float64), {
        "eigensolver": solver,
        "condition_q": float(np.linalg.cond(Q)),
        "max_eigenvalue": float(np.real(eigvals[idx])),
    }


@dataclass
class TRCAModel:
    """Single-band TRCA model using [trials, time, channels] public input."""

    labels: list[int] | None = None
    fs_hz: float | None = None
    bandpass_hz: tuple[float, float] | None = None
    filter_order: int = 4
    regularization: float = 1e-6
    ensemble: bool = False
    class_order_: list[int] = field(default_factory=list)
    templates_: np.ndarray | None = None
    spatial_filters_: np.ndarray | None = None
    metadata_: dict | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TRCAModel":
        X_train = ensure_trials_time_channels(X_train)
        y_train = np.asarray(y_train).ravel()
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train trial counts differ")
        X_fit = X_train
        if self.bandpass_hz is not None:
            if self.fs_hz is None:
                raise ValueError("fs_hz is required when bandpass_hz is set")
            X_fit = bandpass_filter(X_fit, fs_hz=self.fs_hz, low_hz=self.bandpass_hz[0], high_hz=self.bandpass_hz[1], order=self.filter_order)
        labels = list(self.labels if self.labels is not None else sorted(int(v) for v in np.unique(y_train)))
        X_nct = X_fit.transpose(0, 2, 1)
        templates = []
        filters = []
        per_class_meta = {}
        for label in labels:
            trials = X_nct[y_train == label]
            if len(trials) == 0:
                raise ValueError(f"missing training trials for class {label}")
            templates.append(trials.mean(axis=0))
            w, meta = _trca_filter(trials, regularization=self.regularization)
            filters.append(w)
            per_class_meta[str(label)] = meta
        self.class_order_ = labels
        self.templates_ = np.stack(templates, axis=0)
        self.spatial_filters_ = np.stack(filters, axis=0)
        self.metadata_ = {
            "input_shape_before": list(X_train.shape),
            "input_shape_internal": list(X_nct.shape),
            "template_shape": list(self.templates_.shape),
            "spatial_filter_shape": list(self.spatial_filters_.shape),
            "class_order": labels,
            "label_base": int(min(labels)) if labels else None,
            "regularization": self.regularization,
            "score_formula": "corr(w[class].T@test, w[class].T@template[class]) or ensemble concatenated class filters",
            "finite_checks": bool(np.all(np.isfinite(self.templates_)) and np.all(np.isfinite(self.spatial_filters_))),
            "per_class": per_class_meta,
        }
        return self

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        if self.templates_ is None or self.spatial_filters_ is None:
            raise RuntimeError("TRCAModel is not fitted")
        X_test = ensure_trials_time_channels(X_test)
        X_eval = X_test
        if self.bandpass_hz is not None:
            X_eval = bandpass_filter(X_eval, fs_hz=float(self.fs_hz), low_hz=self.bandpass_hz[0], high_hz=self.bandpass_hz[1], order=self.filter_order)
        X_nct = X_eval.transpose(0, 2, 1)
        scores = np.zeros((X_nct.shape[0], len(self.class_order_)), dtype=np.float64)
        W_all = self.spatial_filters_.T
        for i, trial in enumerate(X_nct):
            for cls_idx in range(len(self.class_order_)):
                if self.ensemble:
                    z_test = trial.T @ W_all
                    z_tpl = self.templates_[cls_idx].T @ W_all
                else:
                    w = self.spatial_filters_[cls_idx]
                    z_test = w @ trial
                    z_tpl = w @ self.templates_[cls_idx]
                scores[i, cls_idx] = _corr(z_test, z_tpl)
        return scores.astype(np.float32)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X_test)
        return np.asarray(self.class_order_, dtype=np.int64)[np.argmax(scores, axis=1)]


def trca_predict_fold(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, **kwargs) -> tuple[np.ndarray, dict]:
    model = TRCAModel(**kwargs).fit(X_train, y_train)
    return model.predict(X_test), dict(model.metadata_ or {})


def direct_template_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, labels: list[int] | None = None) -> np.ndarray:
    X_train = ensure_trials_time_channels(X_train)
    X_test = ensure_trials_time_channels(X_test)
    y_train = np.asarray(y_train).ravel()
    class_order = list(labels if labels is not None else sorted(int(v) for v in np.unique(y_train)))
    templates = np.stack([X_train[y_train == label].mean(axis=0) for label in class_order], axis=0)
    scores = np.zeros((X_test.shape[0], len(class_order)), dtype=np.float64)
    for i, trial in enumerate(X_test):
        for j, template in enumerate(templates):
            scores[i, j] = _corr(trial, template)
    return np.asarray(class_order, dtype=np.int64)[np.argmax(scores, axis=1)]


def filterbank_trca_predict_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    fs_hz: float,
    passband_low_hz: list[float] | tuple[float, ...],
    stopband_low_hz: list[float] | tuple[float, ...],
    highcut_pass_hz: float,
    highcut_stop_hz: float,
    weights: list[float] | tuple[float, ...] | np.ndarray,
    labels: list[int] | None = None,
    ensemble: bool = False,
    regularization: float = 1e-6,
    fusion_bands: int | None = None,
) -> tuple[np.ndarray, dict]:
    FB_train = cheby1_filter_bank(
        X_train,
        fs_hz=fs_hz,
        passband_low_hz=passband_low_hz,
        stopband_low_hz=stopband_low_hz,
        highcut_pass_hz=highcut_pass_hz,
        highcut_stop_hz=highcut_stop_hz,
    )
    FB_test = cheby1_filter_bank(
        X_test,
        fs_hz=fs_hz,
        passband_low_hz=passband_low_hz,
        stopband_low_hz=stopband_low_hz,
        highcut_pass_hz=highcut_pass_hz,
        highcut_stop_hz=highcut_stop_hz,
    )
    n_bands = FB_train.shape[0] if fusion_bands is None else min(fusion_bands, FB_train.shape[0])
    w = np.asarray(weights, dtype=np.float64)[:n_bands]
    models = []
    scores = None
    metas = []
    for bi in range(n_bands):
        model = TRCAModel(labels=labels, regularization=regularization, ensemble=ensemble).fit(FB_train[bi], y_train)
        band_scores = model.predict_scores(FB_test[bi])
        scores = w[bi] * band_scores if scores is None else scores + w[bi] * band_scores
        models.append(model)
        metas.append(model.metadata_)
    assert scores is not None
    class_order = models[0].class_order_
    return np.asarray(class_order, dtype=np.int64)[np.argmax(scores, axis=1)], {
        "n_bands_used": n_bands,
        "weights": w.tolist(),
        "ensemble": ensemble,
        "band_metadata": metas,
    }
