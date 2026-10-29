"""
Single-fold training and evaluation engine for DL baselines.

Legacy (subject-as-validation): held-out subject used for val, early stop, checkpoint.
Strict outer LOSO: inner validation from nine training subjects only; held-out for final test.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.dataset_12jfpm import (
    TwelveJFPMDataset,
    assert_outer_test_excluded,
    outer_train_mask,
    split_by_subjects,
    stratified_train_inner_val_indices,
)


@dataclass
class FoldConfig:
    """Legacy subject-as-validation LOSO (Version A)."""

    fold_idx: int
    val_subject: str
    seed: int = 42
    epochs: int = 200
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    num_workers: int = 0


@dataclass
class StrictFoldConfig:
    """Strict outer LOSO with stratified inner validation from training subjects."""

    fold_idx: int
    outer_test_subject: str
    seed: int = 42
    inner_val_fraction: float = 0.1
    epochs: int = 200
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    num_workers: int = 0
    # Checkpoint: highest inner-validation accuracy (early stop: inner-val loss).
    checkpoint_metric: str = "inner_val_accuracy"


@dataclass
class FoldResult:
    fold_idx: int
    val_subject: str
    best_val_acc: float
    best_val_acc_epoch: int
    best_val_loss: float
    best_val_loss_epoch: int
    final_epoch: int
    checkpoint_path: str
    elapsed_seconds: float


@dataclass
class StrictFoldResult:
    fold_idx: int
    outer_test_subject: str
    best_inner_val_acc: float
    best_inner_val_acc_epoch: int
    best_inner_val_loss: float
    best_inner_val_loss_epoch: int
    inner_val_loss_at_best_acc: float
    final_epoch: int
    checkpoint_path: str
    split_manifest_path: str
    n_train_samples: int
    n_inner_val_samples: int
    elapsed_seconds: float
    epoch_metrics: list[dict[str, Any]] | None = None


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item()


def _run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            bs = xb.size(0)
            loss_sum += loss_fn(logits, yb).item() * bs
            acc_sum += _accuracy_from_logits(logits, yb) * bs
            n += bs
    return loss_sum / max(n, 1), acc_sum / max(n, 1)


def train_one_fold(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    fold_cfg: FoldConfig,
    checkpoint_dir: Path,
    all_subjects: list[str],
    *,
    device: torch.device | None = None,
    verbose: bool = True,
) -> FoldResult:
    """Legacy LOSO: held-out subject is the validation set (Version A)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_subjects = [s for s in all_subjects if s != fold_cfg.val_subject]
    train_m, val_m = split_by_subjects(subjects, train_subjects, [fold_cfg.val_subject])
    train_ds = TwelveJFPMDataset(X, y, np.where(train_m)[0])
    val_ds = TwelveJFPMDataset(X, y, np.where(val_m)[0])

    train_loader = DataLoader(
        train_ds, batch_size=fold_cfg.batch_size, shuffle=True,
        num_workers=fold_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=fold_cfg.eval_batch_size, shuffle=False,
        num_workers=fold_cfg.num_workers,
    )

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=fold_cfg.lr, weight_decay=fold_cfg.weight_decay,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = checkpoint_dir / f"fold{fold_cfg.fold_idx:02d}_best.pt"

    best_val_acc = -1.0
    best_val_acc_epoch = 0
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    no_improve = 0

    t0 = time.perf_counter()

    for epoch in range(1, fold_cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        val_loss, val_acc = _run_epoch_eval(model, val_loader, loss_fn, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "protocol": "subject_as_val",
                "inner_val_acc": val_acc,
                "inner_val_loss": val_loss,
                "fold_idx": fold_cfg.fold_idx,
                "val_subject": fold_cfg.val_subject,
            }, ckpt_best)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if fold_cfg.early_stopping_patience > 0 and no_improve >= fold_cfg.early_stopping_patience:
            if verbose:
                print(f"    Early stop at epoch {epoch} (patience={fold_cfg.early_stopping_patience})")
            break

    elapsed = time.perf_counter() - t0
    return FoldResult(
        fold_idx=fold_cfg.fold_idx,
        val_subject=fold_cfg.val_subject,
        best_val_acc=round(best_val_acc, 6),
        best_val_acc_epoch=best_val_acc_epoch,
        best_val_loss=round(best_val_loss, 6),
        best_val_loss_epoch=best_val_loss_epoch,
        final_epoch=epoch,
        checkpoint_path=str(ckpt_best),
        elapsed_seconds=round(elapsed, 1),
    )


def train_one_fold_strict(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    fold_cfg: StrictFoldConfig,
    checkpoint_dir: Path,
    all_subjects: list[str],
    *,
    device: torch.device | None = None,
    verbose: bool = True,
    record_epoch_metrics: bool = False,
) -> StrictFoldResult:
    """
    Strict outer LOSO: train on a stratified subset of nine subjects' trials;
    early stopping and checkpoint selection use inner validation only.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer = fold_cfg.outer_test_subject
    outer_train_subjects = [s for s in all_subjects if s != outer]

    pool_mask = outer_train_mask(subjects, outer)
    pool_indices = np.where(pool_mask)[0]
    split_seed = fold_cfg.seed + fold_cfg.fold_idx
    train_idx, inner_val_idx = stratified_train_inner_val_indices(
        y, pool_indices, fold_cfg.inner_val_fraction, split_seed,
    )

    assert_outer_test_excluded(subjects, train_idx, outer, label="gradient train")
    assert_outer_test_excluded(subjects, inner_val_idx, outer, label="inner validation")

    train_ds = TwelveJFPMDataset(X, y, train_idx)
    inner_val_ds = TwelveJFPMDataset(X, y, inner_val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=fold_cfg.batch_size, shuffle=True,
        num_workers=fold_cfg.num_workers,
    )
    inner_val_loader = DataLoader(
        inner_val_ds, batch_size=fold_cfg.eval_batch_size, shuffle=False,
        num_workers=fold_cfg.num_workers,
    )

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=fold_cfg.lr, weight_decay=fold_cfg.weight_decay,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = checkpoint_dir / f"fold{fold_cfg.fold_idx:02d}_best.pt"
    manifest_path = checkpoint_dir / f"fold{fold_cfg.fold_idx:02d}_split_manifest.json"

    split_manifest: dict[str, Any] = {
        "protocol": "strict_loso_inner_val",
        "fold_idx": fold_cfg.fold_idx,
        "outer_test_subject": outer,
        "outer_train_subjects": outer_train_subjects,
        "inner_val_fraction": fold_cfg.inner_val_fraction,
        "split_seed": split_seed,
        "n_train_samples": int(len(train_idx)),
        "n_inner_val_samples": int(len(inner_val_idx)),
        "checkpoint_metric": fold_cfg.checkpoint_metric,
        "early_stopping_metric": "inner_val_loss",
        "outer_test_in_train_indices": False,
        "outer_test_in_inner_val_indices": False,
    }
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    best_inner_acc = -1.0
    best_inner_acc_epoch = 0
    best_inner_loss = float("inf")
    best_inner_loss_epoch = 0
    inner_val_loss_at_best_acc = float("inf")
    no_improve = 0
    epoch_metrics: list[dict[str, Any]] | None = [] if record_epoch_metrics else None

    t0 = time.perf_counter()

    for epoch in range(1, fold_cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        inner_loss, inner_acc = _run_epoch_eval(model, inner_val_loader, loss_fn, device)

        if inner_acc > best_inner_acc:
            best_inner_acc = inner_acc
            best_inner_acc_epoch = epoch
            inner_val_loss_at_best_acc = inner_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "protocol": "strict_loso_inner_val",
                "inner_val_acc": inner_acc,
                "inner_val_loss": inner_loss,
                "fold_idx": fold_cfg.fold_idx,
                "outer_test_subject": outer,
                "checkpoint_metric": fold_cfg.checkpoint_metric,
            }, ckpt_best)

        if inner_loss < best_inner_loss:
            best_inner_loss = inner_loss
            best_inner_loss_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch_metrics is not None:
            epoch_metrics.append({
                "epoch": epoch,
                "inner_val_acc": round(inner_acc, 6),
                "inner_val_loss": round(inner_loss, 6),
            })

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    epoch {epoch}: inner_val_acc={inner_acc*100:.2f}% inner_val_loss={inner_loss:.4f}")

        if fold_cfg.early_stopping_patience > 0 and no_improve >= fold_cfg.early_stopping_patience:
            if verbose:
                print(f"    Early stop at epoch {epoch} (inner-val loss patience={fold_cfg.early_stopping_patience})")
            break

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"    Best inner_val_acc: {best_inner_acc*100:.2f}% (epoch {best_inner_acc_epoch})")

    return StrictFoldResult(
        fold_idx=fold_cfg.fold_idx,
        outer_test_subject=outer,
        best_inner_val_acc=round(best_inner_acc, 6),
        best_inner_val_acc_epoch=best_inner_acc_epoch,
        best_inner_val_loss=round(best_inner_loss, 6),
        best_inner_val_loss_epoch=best_inner_loss_epoch,
        inner_val_loss_at_best_acc=round(inner_val_loss_at_best_acc, 6),
        final_epoch=epoch,
        checkpoint_path=str(ckpt_best),
        split_manifest_path=str(manifest_path),
        n_train_samples=int(len(train_idx)),
        n_inner_val_samples=int(len(inner_val_idx)),
        elapsed_seconds=round(elapsed, 1),
        epoch_metrics=epoch_metrics,
    )


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    val_subject: str,
    all_subjects: list[str],
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Legacy: evaluate on the fold's held-out (val) subject."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    train_subjects = [s for s in all_subjects if s != val_subject]
    _, val_m = split_by_subjects(subjects, train_subjects, [val_subject])
    val_ds = TwelveJFPMDataset(X, y, np.where(val_m)[0])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(yb.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    correct = int((preds == labels).sum())
    n = len(labels)

    return {
        "val_subject": val_subject,
        "n_trials": n,
        "correct": correct,
        "accuracy": round(correct / n, 6) if n > 0 else 0.0,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_acc": ckpt.get("inner_val_acc", ckpt.get("val_acc")),
    }


def strict_fold_train_inner_indices(
    y: np.ndarray,
    subjects: np.ndarray,
    outer_test_subject: str,
    all_subjects: list[str],
    inner_val_fraction: float,
    seed: int,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recompute strict train / inner-val indices (split_seed = seed + fold_idx)."""
    pool_mask = outer_train_mask(subjects, outer_test_subject)
    pool_indices = np.where(pool_mask)[0]
    split_seed = seed + fold_idx
    train_idx, inner_val_idx = stratified_train_inner_val_indices(
        y, pool_indices, inner_val_fraction, split_seed,
    )
    return train_idx, inner_val_idx, split_seed


def evaluate_at_indices(
    model: nn.Module,
    checkpoint_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    num_classes: int = 12,
) -> dict[str, Any]:
    """Load checkpoint and evaluate accuracy + per-class pred/label counts."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    ds = TwelveJFPMDataset(X, y, np.asarray(indices, dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(yb.numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    correct = int((preds == labels).sum()) if len(labels) else 0
    n = len(labels)
    pred_counts = np.bincount(preds, minlength=num_classes).astype(int).tolist()
    label_counts = np.bincount(labels, minlength=num_classes).astype(int).tolist()
    pred_mode = int(np.argmax(pred_counts)) if n else -1

    return {
        "n_trials": n,
        "correct": correct,
        "accuracy": round(correct / n, 6) if n > 0 else 0.0,
        "pred_counts": pred_counts,
        "label_counts": label_counts,
        "pred_mode_class": pred_mode,
        "unique_pred_classes": sorted(int(c) for c in np.unique(preds)),
        "checkpoint_epoch": ckpt.get("epoch"),
    }


def evaluate_outer_test(
    model: nn.Module,
    checkpoint_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    outer_test_subject: str,
    all_subjects: list[str],
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Strict LOSO: final evaluation on outer held-out test subject only."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    train_subjects = [s for s in all_subjects if s != outer_test_subject]
    _, test_m = split_by_subjects(subjects, train_subjects, [outer_test_subject])
    test_ds = TwelveJFPMDataset(X, y, np.where(test_m)[0])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(yb.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    correct = int((preds == labels).sum())
    n = len(labels)

    return {
        "outer_test_subject": outer_test_subject,
        "n_trials": n,
        "correct": correct,
        "outer_test_accuracy": round(correct / n, 6) if n > 0 else 0.0,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_inner_val_acc": ckpt.get("inner_val_acc"),
    }
