"""
12JFPM preprocessing pipeline.

Produces a flat trial set from raw .mat files:
  X : float32 [N, T, C]   — time-major per trial (T=window, C=channels)
  y : int64   [N]          — class label 0..11
  subject : object [N]     — subject id string per trial

Iteration order: subjects s1..s10, within each: class 0..11, trial 0..14.
This is a stable, deterministic ordering that matches the reference
output from existing_work/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .parse_12jfpm import (
    N_CLASSES,
    N_SUBJECTS,
    N_TRIALS_PER_CLASS,
    ONSET_SAMPLE_0BASED,
    discover_subject_paths,
    read_subject,
    _repo_root,
)


SUBJECTS_ALL = [f"s{i}" for i in range(1, N_SUBJECTS + 1)]
TRIALS_PER_SUBJECT = N_CLASSES * N_TRIALS_PER_CLASS  # 180


@dataclass
class PreprocessConfig:
    raw_data_root: Path | None = None
    onset_sample_0: int = ONSET_SAMPLE_0BASED
    window_length: int = 512
    normalize: str = "channel_zscore"
    subjects: list[str] = field(default_factory=lambda: list(SUBJECTS_ALL))
    output_dir: Path | None = None
    seed: int = 42


def _normalize_trial(x: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a single trial ``x`` of shape ``[T, C]``."""
    if mode == "none":
        return x.astype(np.float32)
    if mode == "channel_zscore":
        m = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True) + 1e-8
        return ((x - m) / s).astype(np.float32)
    if mode == "global_minmax":
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    raise ValueError(f"Unknown normalize mode: {mode!r}")


def extract_window(
    eeg: np.ndarray,
    *,
    onset_sample_0: int,
    window_length: int,
) -> np.ndarray:
    """
    Crop raw EEG ``[12, 8, 1114, 15]`` to ``[12, 8, window_length, 15]``.
    """
    t0 = onset_sample_0
    t1 = t0 + window_length
    if t1 > eeg.shape[2]:
        raise ValueError(
            f"Window [{t0}:{t1}] exceeds time dim {eeg.shape[2]}; "
            f"reduce window_length."
        )
    return eeg[:, :, t0:t1, :]


def trials_to_arrays(
    eeg_cropped: np.ndarray,
    *,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten ``[12, 8, T, 15]`` to ``X [180, T, 8]``, ``y [180]``.

    Iteration: class 0..11, trial 0..14 (stable deterministic order).
    """
    n_cls, n_ch, n_t, n_tr = eeg_cropped.shape
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for c in range(n_cls):
        for tr in range(n_tr):
            seg = eeg_cropped[c, :, :, tr].T  # [T, C]
            seg = _normalize_trial(seg, normalize)
            xs.append(seg)
            ys.append(c)
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def run_preprocessing(cfg: PreprocessConfig) -> dict[str, Any]:
    """
    Run the full preprocessing pipeline and save ``processed.npz`` + ``meta.json``.

    Returns a metadata dict.
    """
    output_dir = Path(cfg.output_dir or _repo_root() / "data" / "processed" / "12JFPM")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    subj_ids: list[str] = []

    for sid in cfg.subjects:
        rec = read_subject(sid, data_root=cfg.raw_data_root)
        eeg = np.asarray(rec["eeg"])
        cropped = extract_window(
            eeg,
            onset_sample_0=cfg.onset_sample_0,
            window_length=cfg.window_length,
        )
        X, y = trials_to_arrays(cropped, normalize=cfg.normalize)
        all_X.append(X)
        all_y.append(y)
        subj_ids.extend([sid] * len(y))

    Xs = np.concatenate(all_X, axis=0)
    ys = np.concatenate(all_y, axis=0)
    subjects_arr = np.asarray(subj_ids, dtype=object)

    npz_path = output_dir / "processed.npz"
    np.savez_compressed(npz_path, X=Xs, y=ys, subject=subjects_arr)

    meta = {
        "n_samples": int(Xs.shape[0]),
        "shape_X": list(Xs.shape),
        "dtype_X": str(Xs.dtype),
        "dtype_y": str(ys.dtype),
        "onset_sample_0": cfg.onset_sample_0,
        "window_length": cfg.window_length,
        "normalize": cfg.normalize,
        "subjects": cfg.subjects,
        "output_file": str(npz_path),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_processed_npz(
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(X, y, subject)`` from a saved ``processed.npz``."""
    path = path or (_repo_root() / "data" / "processed" / "12JFPM" / "processed.npz")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["subject"]


def config_from_yaml(yaml_path: Path) -> PreprocessConfig:
    """Build a ``PreprocessConfig`` from a YAML file."""
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    prep = doc.get("preprocess", {})
    data = doc.get("data", {})

    raw_root = data.get("raw_data_root")
    return PreprocessConfig(
        raw_data_root=Path(raw_root) if raw_root else None,
        onset_sample_0=int(prep.get("onset_sample_0", ONSET_SAMPLE_0BASED)),
        window_length=int(prep.get("window_length", 512)),
        normalize=str(prep.get("normalize", "channel_zscore")),
        subjects=[str(s) for s in data.get("subjects", SUBJECTS_ALL)],
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        seed=int(doc.get("seed", 42)),
    )
