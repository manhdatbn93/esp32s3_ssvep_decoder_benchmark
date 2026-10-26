"""
12JFPM (Nakanishi et al., 2015) parser.

Dataset: 12JFPM (Nakanishi et al.). Obtain the publicly released epochs from the source cited in the article.

Layout (from repository README, verified on s1.mat):
  eeg  : (12 targets, 8 channels, 1114 samples, 15 trials)
  train: (12, 18, 1114, 15) — 18-channel variant (unused here)

Sample rate : 256 Hz
Onset       : sample index 38 (0-based)
Classes     : 12 frequency targets
Subjects    : s1 .. s10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DATASET_NAME = "12JFPM"
SAMPLE_RATE_HZ = 256.0
N_SUBJECTS = 10
N_CLASSES = 12
N_CHANNELS = 8
N_TRIALS_PER_CLASS = 15
N_SAMPLES_RAW = 1114
ONSET_SAMPLE_0BASED = 38

TARGET_FREQUENCIES_HZ = [
    9.25, 11.25, 13.25,
    9.75, 11.75, 13.75,
    10.25, 12.25, 14.25,
    10.75, 12.75, 14.75,
]


def _repo_root() -> Path:
    """Return the project repo root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def _find_data_root(hint: Path | None = None) -> Path:
    """Locate the directory containing s1.mat .. s10.mat."""
    if hint is not None:
        hint = Path(hint)
        if hint.is_dir() and any(hint.glob("s*.mat")):
            return hint

    candidates = [
        _repo_root() / "data" / "raw" / "12JFPM" / "extracted" / "12JFPM_SSVEP-master" / "data",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("s*.mat")):
            return c

    root = _repo_root() / "data" / "raw" / "12JFPM"
    if root.is_dir():
        for d in sorted(root.rglob("s1.mat")):
            return d.parent

    raise FileNotFoundError(
        "Cannot locate 12JFPM .mat files (s1.mat .. s10.mat).\n"
        "  Pass --data-root <folder> to replay tools, or place data under:\n"
        f"    {_repo_root() / 'data' / 'raw' / '12JFPM' / 'extracted' / '12JFPM_SSVEP-master' / 'data'}\n"
        "  Place released subject MAT files so that s1.mat is in one folder. See docs/dataset.md."
    )


def discover_subject_paths(data_root: Path | None = None) -> dict[str, Path]:
    """Return ``{'s1': Path(...), ..., 's10': Path(...)}``."""
    root = _find_data_root(data_root)
    out: dict[str, Path] = {}
    for p in sorted(root.glob("s*.mat")):
        stem = p.stem.lower()
        if stem.startswith("s") and stem[1:].isdigit():
            out[stem] = p.resolve()
    return out


def read_subject(
    subject_id: str,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    Load one subject's .mat and return a standardized record.

    Parameters
    ----------
    subject_id : e.g. ``'s1'`` or ``'S1'``.
    data_root  : directory containing ``s1.mat`` .. ``s10.mat``.
    """
    sid = subject_id.lower()
    if not sid.startswith("s"):
        sid = "s" + sid.lstrip("s")

    paths = discover_subject_paths(data_root)
    if sid not in paths:
        raise FileNotFoundError(
            f"No .mat for subject {subject_id!r} under "
            f"{data_root or _find_data_root(None)}"
        )

    import scipy.io as sio

    mat = sio.loadmat(str(paths[sid]))
    eeg = np.asarray(mat["eeg"])  # [12, 8, 1114, 15]

    return {
        "dataset_name": DATASET_NAME,
        "subject_id": sid,
        "eeg": eeg,
        "sample_rate": SAMPLE_RATE_HZ,
        "n_classes": eeg.shape[0],
        "n_channels": eeg.shape[1],
        "n_samples": eeg.shape[2],
        "n_trials": eeg.shape[3],
        "source_path": str(paths[sid]),
    }
