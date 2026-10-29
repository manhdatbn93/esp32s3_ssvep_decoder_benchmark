"""Regression: epoch-metric recording must not change strict-fold training."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from src.training.engine import StrictFoldConfig, train_one_fold_strict


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.reshape(x.shape[0], -1))


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    subjects = []
    for subject in range(1, 11):
        for label in (0, 1):
            for _ in range(4):
                rows.append(rng.normal(size=4).astype(np.float32))
                labels.append(label)
                subjects.append(f"s{subject}")
    return np.stack(rows), np.asarray(labels, np.int64), np.asarray(subjects)


def _train(record: bool, checkpoint_dir):
    torch.manual_seed(123)
    model = _Tiny()
    cfg = StrictFoldConfig(
        fold_idx=1,
        outer_test_subject="s1",
        seed=7,
        inner_val_fraction=0.25,
        epochs=3,
        batch_size=8,
        eval_batch_size=8,
        lr=0.05,
        weight_decay=0.0,
        early_stopping_patience=0,
        num_workers=0,
    )
    X, y, subjects = _synthetic()
    return train_one_fold_strict(
        model,
        X,
        y,
        subjects,
        cfg,
        checkpoint_dir,
        [f"s{i}" for i in range(1, 11)],
        device=torch.device("cpu"),
        verbose=False,
        record_epoch_metrics=record,
    )


def _checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_epoch_metrics_flag_does_not_change_training(tmp_path):
    disabled = _train(False, tmp_path / "off")
    enabled = _train(True, tmp_path / "on")

    assert disabled.epoch_metrics is None
    assert enabled.epoch_metrics is not None
    assert [row["epoch"] for row in enabled.epoch_metrics] == [1, 2, 3]
    assert enabled.final_epoch == 3
    for row in enabled.epoch_metrics:
        assert set(row) == {"epoch", "inner_val_acc", "inner_val_loss"}

    assert disabled.best_inner_val_acc == enabled.best_inner_val_acc
    assert disabled.best_inner_val_acc_epoch == enabled.best_inner_val_acc_epoch
    assert disabled.best_inner_val_loss == enabled.best_inner_val_loss
    assert disabled.best_inner_val_loss_epoch == enabled.best_inner_val_loss_epoch
    assert disabled.final_epoch == enabled.final_epoch
    assert disabled.inner_val_loss_at_best_acc == enabled.inner_val_loss_at_best_acc
    assert disabled.n_train_samples == enabled.n_train_samples
    assert disabled.n_inner_val_samples == enabled.n_inner_val_samples

    chosen = next(row for row in enabled.epoch_metrics if row["epoch"] == enabled.best_inner_val_acc_epoch)
    assert chosen["inner_val_acc"] == enabled.best_inner_val_acc

    off_ckpt = _checkpoint(disabled.checkpoint_path)
    on_ckpt = _checkpoint(enabled.checkpoint_path)
    assert off_ckpt["epoch"] == on_ckpt["epoch"] == disabled.best_inner_val_acc_epoch
    assert off_ckpt["outer_test_subject"] == "s1"
    assert off_ckpt["protocol"] == "strict_loso_inner_val"
    for key, value in off_ckpt["model_state_dict"].items():
        assert torch.equal(value, on_ckpt["model_state_dict"][key])

    off_manifest = json.loads((tmp_path / "off" / "fold01_split_manifest.json").read_text(encoding="utf-8"))
    on_manifest = json.loads((tmp_path / "on" / "fold01_split_manifest.json").read_text(encoding="utf-8"))
    assert off_manifest == on_manifest
    assert off_manifest["outer_test_subject"] == "s1"
    assert "s1" not in off_manifest["outer_train_subjects"]
