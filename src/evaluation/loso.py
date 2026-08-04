"""
Leave-One-Subject-Out (LOSO) evaluation harness for classical benchmarks.

Provides a generic LOSO loop that:
  1. Loads the frozen processed.npz
  2. For each fold, holds out one subject as the test set
  3. Runs a supplied predict_batch function on the test set
  4. Collects per-fold and aggregate metrics
  5. Saves results to JSON
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class LOSOResult:
    method: str
    n_folds: int
    subjects: list[str]
    per_fold: list[dict[str, Any]]
    mean_accuracy: float
    std_accuracy: float
    total_trials: int
    total_correct: int
    elapsed_seconds: float


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    method_name: str,
    predict_batch_fn: Callable[[np.ndarray], np.ndarray],
    subject_order: list[str] | None = None,
) -> LOSOResult:
    """
    Run LOSO cross-validation.

    Parameters
    ----------
    X : [N, T, C] float32 trial data.
    y : [N] int64 labels.
    subjects : [N] object subject id per trial.
    method_name : identifier for the method.
    predict_batch_fn : callable(X_test) -> y_pred, int64 array.
    subject_order : explicit fold order; defaults to sorted unique subjects.
    """
    unique_subs = subject_order or sorted(
        set(str(s) for s in subjects), key=lambda s: int(s[1:])
    )
    n_folds = len(unique_subs)
    per_fold: list[dict[str, Any]] = []
    total_correct = 0
    total_trials = 0

    t0 = time.perf_counter()

    for fold_idx, val_sub in enumerate(unique_subs):
        mask = np.array([str(s) == val_sub for s in subjects])
        X_test = X[mask]
        y_test = y[mask]

        y_pred = predict_batch_fn(X_test)
        correct = int(np.sum(y_test == y_pred))
        n_test = len(y_test)
        acc = correct / n_test if n_test > 0 else 0.0

        per_fold.append({
            "fold": fold_idx + 1,
            "val_subject": val_sub,
            "n_trials": n_test,
            "correct": correct,
            "accuracy": round(acc, 6),
        })
        total_correct += correct
        total_trials += n_test

    elapsed = time.perf_counter() - t0
    accs = [f["accuracy"] for f in per_fold]

    return LOSOResult(
        method=method_name,
        n_folds=n_folds,
        subjects=unique_subs,
        per_fold=per_fold,
        mean_accuracy=round(float(np.mean(accs)), 6),
        std_accuracy=round(float(np.std(accs)), 6),
        total_trials=total_trials,
        total_correct=total_correct,
        elapsed_seconds=round(elapsed, 2),
    )


def run_loso_trainable(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    method_name: str,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    subject_order: list[str] | None = None,
) -> LOSOResult:
    """
    LOSO for training-based methods.

    fit_predict_fn(X_train, y_train, X_test) -> y_pred
    """
    unique_subs = subject_order or sorted(
        set(str(s) for s in subjects), key=lambda s: int(s[1:])
    )
    n_folds = len(unique_subs)
    per_fold: list[dict[str, Any]] = []
    total_correct = 0
    total_trials = 0

    t0 = time.perf_counter()

    for fold_idx, val_sub in enumerate(unique_subs):
        mask = np.array([str(s) == val_sub for s in subjects])
        X_test = X[mask]
        y_test = y[mask]
        X_train = X[~mask]
        y_train = y[~mask]

        y_pred = fit_predict_fn(X_train, y_train, X_test)
        correct = int(np.sum(y_test == y_pred))
        n_test = len(y_test)
        acc = correct / n_test if n_test > 0 else 0.0

        per_fold.append({
            "fold": fold_idx + 1,
            "val_subject": val_sub,
            "n_trials": n_test,
            "correct": correct,
            "accuracy": round(acc, 6),
        })
        total_correct += correct
        total_trials += n_test

    elapsed = time.perf_counter() - t0
    accs = [f["accuracy"] for f in per_fold]

    return LOSOResult(
        method=method_name,
        n_folds=n_folds,
        subjects=unique_subs,
        per_fold=per_fold,
        mean_accuracy=round(float(np.mean(accs)), 6),
        std_accuracy=round(float(np.std(accs)), 6),
        total_trials=total_trials,
        total_correct=total_correct,
        elapsed_seconds=round(elapsed, 2),
    )


def save_results(result: LOSOResult, output_dir: Path) -> Path:
    """Save LOSO results to a JSON file, return path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.method}_loso_12jfpm.json"
    data = {
        "method": result.method,
        "n_folds": result.n_folds,
        "subjects": result.subjects,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "total_trials": result.total_trials,
        "total_correct": result.total_correct,
        "elapsed_seconds": result.elapsed_seconds,
        "per_fold": result.per_fold,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
