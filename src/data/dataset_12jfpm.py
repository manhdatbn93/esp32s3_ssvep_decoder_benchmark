"""
PyTorch Dataset for preprocessed 12JFPM trials.

Sample shape: [T, C] float32 (time x channels), e.g. [512, 8].
Label: int64 in 0..11.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def split_by_subjects(
    subjects: np.ndarray,
    train_subjects: list[str],
    val_subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for train and val rows."""
    subj = subjects.astype(str)
    train_m = np.isin(subj, train_subjects)
    val_m = np.isin(subj, val_subjects)
    if (train_m & val_m).sum() != 0:
        raise ValueError("train and val subject sets overlap")
    return train_m, val_m


def outer_train_mask(subjects: np.ndarray, outer_test_subject: str) -> np.ndarray:
    """Boolean mask for all trials from outer-training subjects (nine subjects)."""
    subj = subjects.astype(str)
    return np.isin(subj, [s for s in np.unique(subj) if str(s) != outer_test_subject])


def stratified_train_inner_val_indices(
    y: np.ndarray,
    pool_indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified trial-level split of pool_indices into train and inner-validation.

    pool_indices must already exclude the outer held-out test subject.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_parts: list[np.ndarray] = []
    inner_parts: list[np.ndarray] = []

    for c in np.unique(y[pool_indices]):
        cls_idx = pool_indices[y[pool_indices] == c]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n < 2:
            train_parts.append(cls_idx)
            continue
        n_inner = int(round(n * val_fraction))
        n_inner = max(1, min(n_inner, n - 1))
        inner_parts.append(cls_idx[:n_inner])
        train_parts.append(cls_idx[n_inner:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    inner_idx = np.concatenate(inner_parts) if inner_parts else np.array([], dtype=np.int64)
    return train_idx, inner_idx


def assert_outer_test_excluded(
    subjects: np.ndarray,
    indices: np.ndarray,
    outer_test_subject: str,
    *,
    label: str,
) -> None:
    """Raise if any index row belongs to outer_test_subject."""
    subj = subjects.astype(str)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return
    present = set(subj[idx].tolist())
    if outer_test_subject in present:
        raise ValueError(
            f"{label}: outer test subject {outer_test_subject!r} found in indices "
            f"(present subjects: {sorted(present)})"
        )


class TwelveJFPMDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray | None = None,
    ):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.indices = np.arange(len(X)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.indices[idx])
        x = torch.from_numpy(self.X[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y
