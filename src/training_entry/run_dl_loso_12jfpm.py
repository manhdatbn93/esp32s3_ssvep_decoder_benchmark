#!/usr/bin/env python3
"""
Run full 10-fold LOSO training + evaluation for DL baselines on 12JFPM.

Protocols:
  subject_as_val (Version A): held-out subject = validation + checkpoint selection.
  strict_loso_inner_val: inner validation from nine training subjects only;
                         held-out subject = final test evaluation only.

Usage:
  python -u -m training.run_dl_loso_12jfpm --model eegnet
  python -u -m training.run_dl_loso_12jfpm --model eegnet --protocol strict_loso_inner_val
  python -u -m training.run_dl_loso_12jfpm --model eegnet --protocol strict_loso_inner_val --subjects s1

Diagnostic rerun (does not touch official strict benchmark paths):
  python -u -m training.run_dl_loso_12jfpm --model convca --protocol strict_loso_inner_val \\
      --subjects s4 s6 --diagnostic-tag convca_fold04_fold06_diag
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from export.protocol_cli import (
    DEFAULT_PROTOCOL,
    add_protocol_arg,
    apply_protocol_side_effects,
    warn_if_subject_as_val,
)
from export.subject_paths import (
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    STRICT_EXPORT_SUBDIR,
    checkpoint_dir,
    checkpoint_dir_name,
    fold_for_subject,
    results_dir_strict,
    strict_diagnostic_checkpoint_dir,
    strict_diagnostic_results_path,
    strict_diagnostic_results_root,
)
from src.datasets.preprocess_12jfpm import load_processed_npz
from src.datasets.dataset_12jfpm import split_by_subjects
from src.neural import (
    CCNN,
    ConvCA,
    EEGNet,
    FBtCNN,
    MODEL_REGISTRY,
    SSVEPformer,
    SSVEPNet,
)
from src.training.engine import (
    FoldConfig,
    StrictFoldConfig,
    evaluate_at_indices,
    evaluate_checkpoint,
    evaluate_outer_test,
    strict_fold_train_inner_indices,
    train_one_fold,
    train_one_fold_strict,
)
from src.training.seed import set_seed

SUBJECTS = [f"s{i}" for i in range(1, 11)]

ALL_MODELS = ["eegnet", "ccnn", "ssvepformer", "ssvepnet", "fbtcnn", "convca"]

_MODEL_DISPLAY: dict[str, str] = {
    "eegnet": "EEGNet",
    "ccnn": "CCNN",
    "ssvepformer": "SSVEPformer",
    "ssvepnet": "SSVEPNet",
    "fbtcnn": "FBtCNN",
    "convca": "ConvCA",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_SEP = "=" * 60

_CONFIG_MAP: dict[str, str] = {
    "eegnet": "configs/training/eegnet_12jfpm_strict_loso.yaml",
    "ccnn": "configs/training/ccnn_12jfpm_strict_loso.yaml",
    "ssvepformer": "configs/training/ssvepformer_12jfpm_strict_loso.yaml",
    "ssvepnet": "configs/training/ssvepnet_12jfpm_strict_loso.yaml",
    "fbtcnn": "configs/training/fbtcnn_12jfpm_strict_loso.yaml",
    "convca": "configs/training/convca_12jfpm_strict_loso.yaml",
}


def _display_model(model_name: str) -> str:
    return _MODEL_DISPLAY.get(model_name, model_name.upper())


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _outer_train_subjects(outer_test: str) -> str:
    return ",".join(s for s in SUBJECTS if s != outer_test)


def _log_strict_model_header(
    model_name: str,
    *,
    model_index: int,
    model_total: int,
    protocol: str,
    ckpt_dir: Path,
    results_path: Path,
) -> None:
    label = _display_model(model_name)
    print(_LOG_SEP, flush=True)
    print(f"MODEL {model_index}/{model_total}: {label}", flush=True)
    print(f"Protocol: {protocol}", flush=True)
    print(f"Checkpoint dir: {_rel_path(ckpt_dir)}/", flush=True)
    print(f"Result file: {_rel_path(results_path)}", flush=True)
    print(_LOG_SEP, flush=True)


def _log_strict_fold_header(
    model_name: str,
    *,
    fold_idx: int,
    n_folds: int,
    outer_test: str,
    inner_val_fraction: float,
    split_seed: int,
    patience: int,
) -> None:
    label = _display_model(model_name)
    print(
        f"\n--- {label} | Fold {fold_idx:02d}/{n_folds} | outer_test={outer_test} ---",
        flush=True,
    )
    print(f"Outer train subjects: {_outer_train_subjects(outer_test)}", flush=True)
    print(f"Inner validation fraction: {inner_val_fraction}", flush=True)
    print(f"Split seed: {split_seed}", flush=True)
    print(f"Early stopping: inner_val_loss, patience={patience}", flush=True)
    print("Checkpoint selection: best inner_val_acc", flush=True)


def _log_strict_fold_split_sizes(n_train: int, n_inner_val: int) -> None:
    print(f"Train samples: {n_train}", flush=True)
    print(f"Inner-val samples: {n_inner_val}", flush=True)


def _log_strict_fold_footer(
    result,
    *,
    outer_test_accuracy: float,
    n_correct: int,
    n_trials: int,
) -> None:
    print("Fold complete:", flush=True)
    print(f"  best epoch (inner_val_acc): {result.best_inner_val_acc_epoch}", flush=True)
    print(f"  best inner_val_acc: {result.best_inner_val_acc * 100:.2f}%", flush=True)
    print(f"  best inner_val_loss: {result.best_inner_val_loss:.6f}", flush=True)
    print(
        f"  outer_test_accuracy: {outer_test_accuracy * 100:.2f}% "
        f"({n_correct}/{n_trials})",
        flush=True,
    )
    print(f"  checkpoint: {result.checkpoint_path}", flush=True)
    print(f"  split manifest: {result.split_manifest_path}", flush=True)


def _resolve_strict_run_paths(
    model_name: str,
    *,
    diagnostic_tag: str | None,
    checkpoint_root: Path | None,
    output_root: Path | None,
    results_dir_arg: Path | None,
) -> tuple[Path, Path, bool]:
    """
    Return (checkpoint_dir, results_json_path, is_diagnostic).

    Official benchmark paths are used only when no diagnostic tag or explicit
    diagnostic roots are given.
    """
    is_diag = bool(diagnostic_tag or checkpoint_root or output_root)
    if diagnostic_tag:
        ckpt = strict_diagnostic_checkpoint_dir(model_name, diagnostic_tag, repo=_REPO_ROOT)
        results_path = strict_diagnostic_results_path(model_name, diagnostic_tag, repo=_REPO_ROOT)
    else:
        ckpt = checkpoint_dir(model_name, protocol=PROTOCOL_STRICT_LOSO_INNER_VAL, repo=_REPO_ROOT)
        if results_dir_arg is not None:
            results_path = (
                results_dir_arg
                if results_dir_arg.suffix == ".json"
                else results_dir_arg / f"{model_name}_12jfpm.json"
            )
        else:
            results_path = results_dir_strict(model_name, repo=_REPO_ROOT)

    if checkpoint_root is not None:
        ckpt = checkpoint_root.resolve()
    if output_root is not None:
        results_path = (output_root.resolve() / f"{model_name}_12jfpm.json")

    if is_diag:
        official_ckpt = checkpoint_dir(
            model_name, protocol=PROTOCOL_STRICT_LOSO_INNER_VAL, repo=_REPO_ROOT,
        ).resolve()
        official_results = results_dir_strict(model_name, repo=_REPO_ROOT).resolve()
        if ckpt.resolve() == official_ckpt:
            raise ValueError(
                "Diagnostic run would overwrite official checkpoint dir "
                f"{official_ckpt}. Use --diagnostic-tag or a distinct --checkpoint-root."
            )
        if results_path.resolve() == official_results:
            raise ValueError(
                "Diagnostic run would overwrite official results "
                f"{official_results}. Use --diagnostic-tag or a distinct --output-root."
            )
    return ckpt, results_path, is_diag


def _log_prediction_distribution(title: str, dist: dict) -> None:
    num_classes = len(dist.get("label_counts", [])) or 12
    print(f"  {title} accuracy: {dist['accuracy'] * 100:.2f}% "
          f"({dist['correct']}/{dist['n_trials']})", flush=True)
    print(f"  {title} predicted classes: {dist.get('unique_pred_classes', [])} "
          f"(mode=class {dist.get('pred_mode_class', -1)})", flush=True)
    print(f"  {title} pred counts by class:", flush=True)
    pred_counts = dist.get("pred_counts", [0] * num_classes)
    label_counts = dist.get("label_counts", [0] * num_classes)
    for c in range(num_classes):
        print(f"    class {c:2d}: pred={pred_counts[c]:4d}  label={label_counts[c]:4d}", flush=True)


def _log_strict_model_footer(summary: dict, results_path: Path) -> None:
    mean_pct = summary["mean_accuracy"] * 100
    std_pct = summary["std_accuracy"] * 100
    print(_LOG_SEP, flush=True)
    print(f"{_display_model(summary['model'])} — all folds complete", flush=True)
    print(f"  mean outer_test_accuracy: {mean_pct:.2f}%", flush=True)
    print(f"  std outer_test_accuracy: {std_pct:.2f}%", flush=True)
    print(f"  result JSON: {_rel_path(results_path)}", flush=True)
    print(_LOG_SEP, flush=True)


def _resolve_config(model_name: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    rel = _CONFIG_MAP.get(model_name)
    if rel is None:
        raise FileNotFoundError(f"No default config mapping for model '{model_name}'")
    path = _REPO_ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def _build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    mcfg = cfg["model"]
    common = dict(
        seq_len=int(mcfg["seq_len"]),
        embed_dim=int(mcfg["embed_dim"]),
        num_classes=int(mcfg["num_classes"]),
    )
    if model_name == "eegnet":
        return EEGNet(
            **common,
            F1=int(mcfg.get("F1", 8)),
            D=int(mcfg.get("D", 2)),
            F2=int(mcfg.get("F2", 16)),
            dropout=float(mcfg.get("dropout", 0.25)),
            kernel_length=int(mcfg.get("kernel_length", 64)),
            pool1_stride=int(mcfg.get("pool1_stride", 4)),
            pool2_stride=int(mcfg.get("pool2_stride", 8)),
        )
    if model_name == "ccnn":
        return CCNN(
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            dropout=float(mcfg.get("dropout", 0.25)),
        )
    if model_name == "ssvepformer":
        return SSVEPformer(
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            depth=int(mcfg.get("depth", 1)),
            attention_kernel_length=int(mcfg.get("attention_kernel_length", 31)),
            dropout=float(mcfg.get("dropout", 0.5)),
        )
    if model_name == "ssvepnet":
        return SSVEPNet(
            **common,
            dropout=float(mcfg.get("dropout", 0.5)),
        )
    if model_name == "fbtcnn":
        return FBtCNN(
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            dropout=float(mcfg.get("dropout", 0.4)),
        )
    if model_name == "convca":
        return ConvCA(
            **common,
            dropout_sig=float(mcfg.get("dropout_sig", 0.75)),
            dropout_temp=float(mcfg.get("dropout_temp", 0.15)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _compute_class_templates(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    outer_test_subject: str,
    all_subjects: list[str],
    num_classes: int,
) -> torch.Tensor:
    """Per-class mean templates from outer-training subjects only."""
    train_subs = [s for s in all_subjects if s != outer_test_subject]
    train_mask, _ = split_by_subjects(subjects, train_subs, [outer_test_subject])
    X_train = X[train_mask]
    y_train = y[train_mask]
    templates = []
    for c in range(num_classes):
        templates.append(X_train[y_train == c].mean(axis=0))
    return torch.from_numpy(np.stack(templates)).float()


def _run_one_model_legacy(
    model_name: str,
    cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    device: torch.device,
    ckpt_dir: Path,
    results_dir: Path,
    fold_subjects: list[str],
) -> dict:
    seed = int(cfg.get("seed", 42))
    tcfg = cfg.get("train", {})
    needs_templates = model_name == "convca"
    num_classes = int(cfg["model"]["num_classes"])

    fold_results: list[dict] = []
    total_t0 = time.perf_counter()

    for fold_idx, val_sub in enumerate(fold_subjects, start=1):
        print(f"\n--- Fold {fold_idx}/10: val={val_sub} (subject-as-val) ---", flush=True)
        set_seed(seed)
        model = _build_model(model_name, cfg)
        if needs_templates:
            templates = _compute_class_templates(X, y, subjects, val_sub, SUBJECTS, num_classes)
            model.set_templates(templates)

        epochs = int(tcfg.get("epochs", 200))
        fold_cfg = FoldConfig(
            fold_idx=fold_idx,
            val_subject=val_sub,
            seed=seed,
            epochs=epochs,
            batch_size=int(tcfg.get("batch_size", 32)),
            eval_batch_size=int(cfg.get("eval", {}).get("batch_size", 64)),
            lr=float(tcfg.get("lr", 0.001)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            early_stopping_patience=int(tcfg.get("early_stopping_patience", 20)),
            num_workers=int(tcfg.get("num_workers", 0)),
        )

        result = train_one_fold(
            model, X, y, subjects, fold_cfg, ckpt_dir, SUBJECTS,
            device=device, verbose=True,
        )
        print(f"    Best val_acc: {result.best_val_acc*100:.2f}% (epoch {result.best_val_acc_epoch})", flush=True)

        set_seed(seed)
        eval_model = _build_model(model_name, cfg)
        if needs_templates:
            eval_model.set_templates(templates)
        eval_report = evaluate_checkpoint(
            eval_model, Path(result.checkpoint_path),
            X, y, subjects, val_sub, SUBJECTS, device=device,
        )
        print(f"    Eval accuracy: {eval_report['accuracy']*100:.2f}%", flush=True)

        fold_results.append({
            "fold": fold_idx,
            "val_subject": val_sub,
            "protocol": PROTOCOL_SUBJECT_AS_VAL,
            "n_trials": eval_report["n_trials"],
            "correct": eval_report["correct"],
            "accuracy": eval_report["accuracy"],
            "best_epoch": result.best_val_acc_epoch,
            "final_epoch": result.final_epoch,
            "elapsed_seconds": result.elapsed_seconds,
            "checkpoint": result.checkpoint_path,
        })

    return _summarize_folds(model_name, fold_results, total_t0, results_dir, protocol=PROTOCOL_SUBJECT_AS_VAL)


def _run_one_model_strict(
    model_name: str,
    cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    device: torch.device,
    ckpt_dir: Path,
    results_path: Path,
    fold_subjects: list[str],
    *,
    epochs_override: int | None = None,
    model_index: int = 1,
    model_total: int = 1,
    diagnostic_tag: str | None = None,
    train_seed_override: int | None = None,
    log_pred_distribution: bool = False,
    no_early_stopping: bool = False,
    record_epoch_metrics: bool = False,
) -> dict:
    seed = int(train_seed_override if train_seed_override is not None else cfg.get("seed", 42))
    tcfg = cfg.get("train", {})
    loso_cfg = cfg.get("loso", {})
    inner_frac = float(loso_cfg.get("inner_val_fraction", 0.10))
    needs_templates = model_name == "convca"
    num_classes = int(cfg["model"]["num_classes"])

    fold_results: list[dict] = []
    total_t0 = time.perf_counter()
    n_folds = len(fold_subjects)
    patience = 0 if no_early_stopping else int(tcfg.get("early_stopping_patience", 20))
    if no_early_stopping:
        print("  Early stopping: DISABLED (full epoch budget)", flush=True)

    _log_strict_model_header(
        model_name,
        model_index=model_index,
        model_total=model_total,
        protocol=PROTOCOL_STRICT_LOSO_INNER_VAL,
        ckpt_dir=ckpt_dir,
        results_path=results_path,
    )

    for outer_sub in fold_subjects:
        fold_idx = fold_for_subject(outer_sub)
        split_seed = seed + fold_idx
        set_seed(seed)
        model = _build_model(model_name, cfg)
        if needs_templates:
            templates = _compute_class_templates(X, y, subjects, outer_sub, SUBJECTS, num_classes)
            model.set_templates(templates)

        n_epochs = int(epochs_override if epochs_override is not None else tcfg.get("epochs", 200))
        fold_cfg = StrictFoldConfig(
            fold_idx=fold_idx,
            outer_test_subject=outer_sub,
            seed=seed,
            inner_val_fraction=inner_frac,
            epochs=n_epochs,
            batch_size=int(tcfg.get("batch_size", 32)),
            eval_batch_size=int(cfg.get("eval", {}).get("batch_size", 64)),
            lr=float(tcfg.get("lr", 0.001)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            early_stopping_patience=patience,
            num_workers=int(tcfg.get("num_workers", 0)),
        )

        _log_strict_fold_header(
            model_name,
            fold_idx=fold_idx,
            n_folds=n_folds,
            outer_test=outer_sub,
            inner_val_fraction=inner_frac,
            split_seed=split_seed,
            patience=patience,
        )

        result = train_one_fold_strict(
            model, X, y, subjects, fold_cfg, ckpt_dir, SUBJECTS,
            device=device, verbose=True,
            record_epoch_metrics=record_epoch_metrics,
        )
        _log_strict_fold_split_sizes(result.n_train_samples, result.n_inner_val_samples)

        set_seed(seed)
        eval_model = _build_model(model_name, cfg)
        if needs_templates:
            eval_model.set_templates(templates)
        eval_report = evaluate_outer_test(
            eval_model, Path(result.checkpoint_path),
            X, y, subjects, outer_sub, SUBJECTS, device=device,
        )
        _log_strict_fold_footer(
            result,
            outer_test_accuracy=eval_report["outer_test_accuracy"],
            n_correct=eval_report["correct"],
            n_trials=eval_report["n_trials"],
        )

        inner_dist: dict | None = None
        outer_dist: dict | None = None
        if log_pred_distribution:
            _, inner_val_idx, _ = strict_fold_train_inner_indices(
                y, subjects, outer_sub, SUBJECTS, inner_frac, seed, fold_idx,
            )
            batch_size = int(cfg.get("eval", {}).get("batch_size", 64))
            inner_dist = evaluate_at_indices(
                eval_model, Path(result.checkpoint_path),
                X, y, inner_val_idx, device=device, batch_size=batch_size,
                num_classes=num_classes,
            )
            train_subjects = [s for s in SUBJECTS if s != outer_sub]
            _, test_m = split_by_subjects(subjects, train_subjects, [outer_sub])
            outer_dist = evaluate_at_indices(
                eval_model, Path(result.checkpoint_path),
                X, y, np.where(test_m)[0], device=device, batch_size=batch_size,
                num_classes=num_classes,
            )
            _log_prediction_distribution("inner-val", inner_dist)
            _log_prediction_distribution("outer-test", outer_dist)

        fold_entry: dict = {
            "fold": fold_idx,
            "outer_test_subject": outer_sub,
            "protocol": PROTOCOL_STRICT_LOSO_INNER_VAL,
            "inner_val_fraction": inner_frac,
            "split_seed": seed + fold_idx,
            "n_train_samples": result.n_train_samples,
            "n_inner_val_samples": result.n_inner_val_samples,
            "best_inner_val_acc": result.best_inner_val_acc,
            "best_inner_val_acc_epoch": result.best_inner_val_acc_epoch,
            "best_inner_val_loss": result.best_inner_val_loss,
            "inner_val_loss_at_best_acc": result.inner_val_loss_at_best_acc,
            "checkpoint_metric": "inner_val_accuracy",
            "early_stopping_patience": patience,
            "early_stopping_metric": "inner_val_loss",
            "n_trials": eval_report["n_trials"],
            "correct": eval_report["correct"],
            "outer_test_accuracy": eval_report["outer_test_accuracy"],
            "final_epoch": result.final_epoch,
            "elapsed_seconds": result.elapsed_seconds,
            "checkpoint": result.checkpoint_path,
            "split_manifest": result.split_manifest_path,
            "train_seed": seed,
        }
        if diagnostic_tag:
            fold_entry["diagnostic_tag"] = diagnostic_tag
        if inner_dist is not None:
            fold_entry["inner_val_pred_distribution"] = inner_dist
        if outer_dist is not None:
            fold_entry["outer_test_pred_distribution"] = outer_dist
        if result.epoch_metrics is not None:
            fold_entry["epoch_metrics"] = result.epoch_metrics
            post24 = [m for m in result.epoch_metrics if m["epoch"] > 24]
            if post24:
                best_post24 = max(post24, key=lambda m: m["inner_val_acc"])
                fold_entry["best_inner_val_acc_after_epoch_24"] = best_post24["inner_val_acc"]
                fold_entry["best_epoch_after_24"] = best_post24["epoch"]
        fold_results.append(fold_entry)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summarize_folds(
        model_name, fold_results, total_t0, results_path.parent,
        protocol=PROTOCOL_STRICT_LOSO_INNER_VAL,
        metric_key="outer_test_accuracy",
        quiet_print=True,
    )
    if diagnostic_tag or no_early_stopping:
        summary["diagnostic_tag"] = diagnostic_tag
        summary["train_seed"] = seed
        summary["no_early_stopping"] = no_early_stopping
        summary["epochs"] = int(epochs_override if epochs_override is not None else tcfg.get("epochs", 200))
        summary["official_checkpoint_dir"] = _rel_path(
            checkpoint_dir(model_name, protocol=PROTOCOL_STRICT_LOSO_INNER_VAL, repo=_REPO_ROOT),
        )
        summary["official_results_json"] = _rel_path(
            results_dir_strict(model_name, repo=_REPO_ROOT),
        )
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_strict_model_footer(summary, results_path)
    return summary


def _summarize_folds(
    model_name: str,
    fold_results: list[dict],
    total_t0: float,
    results_dir: Path,
    *,
    protocol: str,
    metric_key: str = "accuracy",
    quiet_print: bool = False,
) -> dict:
    total_elapsed = time.perf_counter() - total_t0
    accs = [f[metric_key] for f in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    total_correct = sum(f["correct"] for f in fold_results)
    total_trials = sum(f["n_trials"] for f in fold_results)

    summary = {
        "model": model_name,
        "protocol": protocol,
        "mean_accuracy": round(mean_acc, 6),
        "std_accuracy": round(std_acc, 6),
        "total_correct": total_correct,
        "total_trials": total_trials,
        "elapsed_seconds": round(total_elapsed, 1),
        "per_fold": fold_results,
    }
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        summary["checkpoint_selection"] = "best inner-validation accuracy"
        summary["early_stopping"] = "inner-validation loss, patience from config"

    if protocol == PROTOCOL_SUBJECT_AS_VAL:
        out_path = results_dir / f"{model_name}_loso_12jfpm.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Results: {out_path}", flush=True)

    if not quiet_print:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} — {protocol}")
        print(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        print(f"  Total: {total_correct}/{total_trials}", flush=True)
    return summary


def _run_one_model(
    model_name: str,
    cfg: dict,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    device: torch.device,
    protocol: str,
    ckpt_dir: Path | None = None,
    results_dir: Path | None = None,
    fold_subjects: list[str] | None = None,
    epochs_override: int | None = None,
    model_index: int = 1,
    model_total: int = 1,
    *,
    diagnostic_tag: str | None = None,
    checkpoint_root: Path | None = None,
    output_root: Path | None = None,
    diagnostic_seed: int | None = None,
    no_early_stopping: bool = False,
) -> dict:
    fold_subjects = fold_subjects or SUBJECTS
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        resolved_ckpt, results_path, is_diag = _resolve_strict_run_paths(
            model_name,
            diagnostic_tag=diagnostic_tag,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            results_dir_arg=results_dir,
        )
        if ckpt_dir is not None:
            ckpt_dir = ckpt_dir.resolve()
            official_ckpt = checkpoint_dir(
                model_name, protocol=PROTOCOL_STRICT_LOSO_INNER_VAL, repo=_REPO_ROOT,
            ).resolve()
            if is_diag and ckpt_dir == official_ckpt:
                raise ValueError(
                    f"--checkpoint-dir {ckpt_dir} is the official benchmark path."
                )
        else:
            ckpt_dir = resolved_ckpt
        if is_diag:
            print(
                "\n*** DIAGNOSTIC RUN — official benchmark paths are not modified ***",
                flush=True,
            )
            print(f"    Official checkpoints: checkpoints/{checkpoint_dir_name(model_name)}/", flush=True)
            print(f"    Official results    : results/{STRICT_EXPORT_SUBDIR}/{model_name}_12jfpm.json", flush=True)
        return _run_one_model_strict(
            model_name, cfg, X, y, subjects, device, ckpt_dir, results_path, fold_subjects,
            epochs_override=epochs_override,
            model_index=model_index,
            model_total=model_total,
            diagnostic_tag=diagnostic_tag,
            train_seed_override=diagnostic_seed,
            log_pred_distribution=is_diag,
            no_early_stopping=no_early_stopping,
            record_epoch_metrics=is_diag or no_early_stopping,
        )

    ckpt_dir = ckpt_dir or (_REPO_ROOT / "checkpoints" / f"{model_name}_loso_12jfpm")
    results_dir = results_dir or (_REPO_ROOT / "results" / f"{model_name}_loso_12jfpm")
    return _run_one_model_legacy(
        model_name, cfg, X, y, subjects, device, ckpt_dir, results_dir, fold_subjects,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="DL LOSO benchmark on 12JFPM")
    ap.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--npz", type=Path, default=None)
    add_protocol_arg(ap)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Restrict folds to these outer test subjects (e.g. s1).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override train.epochs (smoke tests only).")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Override checkpoint directory (official strict path by default).")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Override results JSON or parent directory.")
    ap.add_argument(
        "--diagnostic-tag",
        default=None,
        metavar="TAG",
        help=(
            "Non-destructive strict rerun: write to "
            "checkpoints/<model>_strict_loso_inner_val_12jfpm_<TAG>/ and "
            "results/strict_loso_inner_val_<TAG>/ (never overwrites official paths)."
        ),
    )
    ap.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Explicit diagnostic checkpoint directory (alternative to --diagnostic-tag suffix).",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Explicit diagnostic results directory (writes <model>_12jfpm.json inside).",
    )
    ap.add_argument(
        "--diagnostic-seed",
        type=int,
        default=None,
        help=(
            "Override training/init seed for this run only (split_seed remains seed+fold). "
            "Official config seed unchanged on disk."
        ),
    )
    ap.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable inner-val loss early stopping (train full --epochs budget).",
    )
    args = ap.parse_args()

    if args.diagnostic_tag and (args.checkpoint_root or args.output_root):
        print(
            "Note: --checkpoint-root / --output-root override paths from --diagnostic-tag.",
            flush=True,
        )

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    fold_subjects = [s.strip().lower() for s in args.subjects] if args.subjects else SUBJECTS

    apply_protocol_side_effects(args.protocol)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Protocol: {args.protocol}", flush=True)
    if args.diagnostic_tag:
        print(f"Diagnostic tag: {args.diagnostic_tag}", flush=True)
    if args.diagnostic_seed is not None:
        print(f"Diagnostic train seed: {args.diagnostic_seed}", flush=True)
    if args.no_early_stopping:
        print("Early stopping: disabled for this run", flush=True)
    print(f"Models: {', '.join(models_to_run)}", flush=True)
    if len(fold_subjects) < len(SUBJECTS):
        print(f"Folds (subjects): {', '.join(fold_subjects)}", flush=True)

    X, y, subjects = load_processed_npz(args.npz)
    print(f"  X: {X.shape}, y: {y.shape}", flush=True)

    all_summaries: list[dict] = []

    n_models = len(models_to_run)
    for model_index, model_name in enumerate(models_to_run, start=1):
        config_path = _resolve_config(model_name, args.config if n_models == 1 else None)
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        summary = _run_one_model(
            model_name, cfg, X, y, subjects, device,
            protocol=args.protocol,
            ckpt_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            fold_subjects=fold_subjects,
            epochs_override=args.epochs,
            model_index=model_index,
            model_total=n_models,
            diagnostic_tag=args.diagnostic_tag,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            diagnostic_seed=args.diagnostic_seed,
            no_early_stopping=args.no_early_stopping,
        )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        if args.protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
            if args.diagnostic_tag:
                grand = strict_diagnostic_results_root(
                    args.diagnostic_tag, repo=_REPO_ROOT,
                ) / "dl_summary.json"
            else:
                grand = _REPO_ROOT / "results" / STRICT_EXPORT_SUBDIR / "dl_summary.json"
        else:
            grand = _REPO_ROOT / "results" / "dl_loso_12jfpm_summary.json"
        grand.parent.mkdir(parents=True, exist_ok=True)
        grand.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        print(f"\n  Grand summary: {grand}", flush=True)


if __name__ == "__main__":
    main()
