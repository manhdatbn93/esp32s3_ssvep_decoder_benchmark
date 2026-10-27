"""Read the processed 12JFPM trial order without changing it.

``preprocess_12jfpm`` stores subjects s1..s10, and within each subject
class 0..11 and trial 0..14. Each row is already the onset-38 window of
512 samples after one channel-wise z-score. This module does not z-score
again and does not read raw MAT files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUBJECTS = tuple(f"s{i}" for i in range(1, 11))
N_CLASSES = 12
TRIALS_PER_CLASS = 15
TRIALS_PER_SUBJECT = N_CLASSES * TRIALS_PER_CLASS


def resolve_processed_npz(dataset_root: Path) -> Path:
    root = Path(dataset_root)
    candidates = [root, root / "processed.npz", root / "12JFPM" / "processed.npz"]
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "processed.npz":
            return candidate
        nested = candidate / "processed.npz" if candidate.is_dir() else None
        if nested is not None and nested.is_file():
            return nested
    raise FileNotFoundError(
        "12JFPM processed.npz was not found under the dataset root. "
        "This ablation does not substitute synthetic trials."
    )


def load_processed_trials(dataset_root: Path) -> dict:
    """Return X, y, subjects, and checked per-subject trial identities."""
    npz_path = resolve_processed_npz(dataset_root)
    with np.load(npz_path, allow_pickle=True) as data:
        x = np.asarray(data["X"])
        y = np.asarray(data["y"])
        if "subject" in data:
            subjects = np.asarray(data["subject"])
        elif "subjects" in data:
            subjects = np.asarray(data["subjects"])
        else:
            raise KeyError("processed.npz has neither subject nor subjects")
    if x.shape != (len(SUBJECTS) * TRIALS_PER_SUBJECT, 512, 8):
        raise ValueError(f"Unexpected X shape {x.shape}")
    if y.shape != (x.shape[0],) or subjects.shape != (x.shape[0],):
        raise ValueError("Label or subject vector length does not match X")
    if x.dtype != np.float32:
        raise ValueError(f"Expected float32 trials, got {x.dtype}")

    identities = []
    for subject_index, subject in enumerate(SUBJECTS):
        base = subject_index * TRIALS_PER_SUBJECT
        for class_id in range(N_CLASSES):
            for trial_idx in range(TRIALS_PER_CLASS):
                row = base + class_id * TRIALS_PER_CLASS + trial_idx
                if str(subjects[row]) != subject or int(y[row]) != class_id:
                    raise ValueError(
                        f"Trial order mismatch at row {row}: "
                        f"subject={subjects[row]!s} label={y[row]}"
                    )
                trial = x[row]
                if not np.isfinite(trial).all():
                    raise ValueError(f"Non-finite EEG at {subject} c={class_id} t={trial_idx}")
                identities.append((subject, subject_index + 1, class_id, trial_idx, row))
    meta = _scientific_meta(npz_path.parent / "meta.json")
    return {
        "X": x,
        "y": y.astype(np.int64, copy=False),
        "subjects": np.asarray([str(s) for s in subjects]),
        "identities": identities,
        "meta": meta,
        "npz_name": npz_path.name,
    }


def _scientific_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    kept = {}
    for key in ("n_samples", "shape_X", "dtype_X", "onset_sample_0", "window_length", "normalize", "subjects"):
        if key in raw:
            kept[key] = raw[key]
    return kept


def calibration_global_indices(
    y: np.ndarray,
    subjects: np.ndarray,
    held_out_subject: str,
    n_per_class: int = 16,
) -> np.ndarray:
    """Global rows selected by ``quantize_int8._load_calibration_raw``.

    The held-out subject is excluded. Within each class, the first
    ``n_per_class`` rows of the remaining processed order are kept.
    """
    mask = np.asarray([str(s) != held_out_subject for s in subjects])
    train_rows = np.flatnonzero(mask)
    y_train = y[mask]
    chosen: list[int] = []
    for class_id in sorted(int(c) for c in np.unique(y_train)):
        local = np.flatnonzero(y_train == class_id)[:n_per_class]
        chosen.extend(int(v) for v in local)
    if any(str(subjects[train_rows[i]]) == held_out_subject for i in chosen):
        raise ValueError("Calibration selection included the held-out subject")
    return train_rows[np.asarray(chosen, dtype=np.int64)]
