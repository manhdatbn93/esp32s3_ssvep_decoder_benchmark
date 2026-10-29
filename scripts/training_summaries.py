#!/usr/bin/env python3
"""Aggregate per-fold training summaries and audit existing checkpoints.

This module does not train models. It reads JSON summaries or checkpoint
metadata that already exist. Recovered reports are not original training records.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

PROTOCOL = "strict_loso_inner_val"
EXPECTED_FOLDS = tuple(range(1, 11))
EXPECTED_TRIALS_PER_FOLD = 180
EXPECTED_TOTAL_TRIALS = 1800

UNAVAILABLE_HISTORICAL_METRICS = (
    "final_epoch",
    "epoch_history",
    "elapsed_seconds",
    "outer_test_predictions",
    "original_per_fold_training_summary",
)


def per_fold_results_path(output_root: Path, model: str, fold: int) -> Path:
    return Path(output_root) / model / "training" / f"fold{fold:02d}.json"


def historical_results_path(output_root: Path, model: str) -> Path:
    return Path(output_root) / model / f"{model}_12jfpm.json"


def aggregate_results_path(output_root: Path, model: str) -> Path:
    return Path(output_root) / model / "training" / "aggregate_10fold.json"


def recovered_results_path(output_root: Path, model: str) -> Path:
    return Path(output_root) / model / "training" / "recovered_10fold.json"


def checkpoint_path(output_root: Path, model: str, fold: int) -> Path:
    return Path(output_root) / model / "checkpoints" / f"fold{fold:02d}_best.pt"


def split_manifest_path(output_root: Path, model: str, fold: int) -> Path:
    return Path(output_root) / model / "checkpoints" / f"fold{fold:02d}_split_manifest.json"


def pytorch_summary_path(host_root: Path, model: str, fold: int) -> Path:
    return Path(host_root) / model / f"fold{fold:02d}" / "pytorch" / "summary.json"


def population_std(values: list[float]) -> float:
    count = len(values)
    if count == 0:
        raise ValueError("cannot compute a standard deviation without values")
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return math.sqrt(variance)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _fold_entry(path: Path, expected_model: str | None) -> dict:
    document = _read_json(path)
    folds = document.get("per_fold")
    if not isinstance(folds, list) or len(folds) != 1 or not isinstance(folds[0], dict):
        raise ValueError(f"{path.name} must contain exactly one per_fold entry")
    entry = dict(folds[0])
    model = str(document.get("model") or entry.get("model") or "")
    if expected_model is not None and model != expected_model:
        raise ValueError(f"{path.name} model is {model!r}, expected {expected_model!r}")
    entry["model"] = model
    protocol = str(entry.get("protocol") or document.get("protocol") or "")
    if protocol != PROTOCOL:
        raise ValueError(f"{path.name} protocol is {protocol!r}, expected {PROTOCOL!r}")
    entry["protocol"] = protocol
    try:
        fold = int(entry["fold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} has no integer fold id") from exc
    expected_fold = int(path.stem.replace("fold", ""))
    if fold != expected_fold:
        raise ValueError(f"{path.name} contains fold {fold}, expected {expected_fold}")
    subject = str(entry.get("outer_test_subject") or "")
    if subject != f"s{fold}":
        raise ValueError(f"{path.name} held-out subject is {subject!r}, expected s{fold}")
    entry["fold"] = fold
    entry["source_file"] = path.name
    return entry


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def score_problem(n_trials, correct, accuracy) -> str | None:
    """Return why a fold score is unusable, or None when it is consistent."""
    if not _is_int(n_trials) or n_trials <= 0:
        return "n_trials must be a positive integer"
    if not _is_int(correct) or correct < 0 or correct > n_trials:
        return "correct must be an integer from 0 through n_trials"
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        return "outer_test_accuracy is missing"
    numeric = float(accuracy)
    if not math.isfinite(numeric):
        return "outer_test_accuracy is not finite"
    if numeric < 0.0 or numeric > 1.0:
        return "outer_test_accuracy is outside [0, 1]"
    if abs(numeric - (correct / n_trials)) > 5e-7:
        return "outer_test_accuracy does not match correct/n_trials"
    return None


def aggregate_fold_files(paths: list[Path], *, model: str) -> dict:
    """Combine per-fold training JSON files. Does not invent missing folds."""
    entries: list[dict] = []
    seen: set[int] = set()
    missing_accuracy: list[int] = []
    for path in paths:
        entry = _fold_entry(path, model)
        fold = int(entry["fold"])
        if fold in seen:
            raise ValueError(f"duplicate fold {fold}")
        if fold not in EXPECTED_FOLDS:
            raise ValueError(f"fold {fold} is outside 1..10")
        seen.add(fold)
        if entry.get("outer_test_accuracy") is None:
            missing_accuracy.append(fold)
        entries.append(entry)
    entries.sort(key=lambda item: int(item["fold"]))
    present = [int(item["fold"]) for item in entries]
    missing = [fold for fold in EXPECTED_FOLDS if fold not in seen]
    inconsistent: list[dict] = []
    scored = []
    for item in entries:
        if item.get("outer_test_accuracy") is None and "n_trials" not in item and "correct" not in item:
            continue
        problem = score_problem(item.get("n_trials"), item.get("correct"), item.get("outer_test_accuracy"))
        if problem is None:
            scored.append(item)
        elif item.get("outer_test_accuracy") is None and problem.startswith("outer_test_accuracy is missing"):
            continue
        else:
            inconsistent.append({"fold": int(item["fold"]), "reason": problem})
    accuracies = [float(item["outer_test_accuracy"]) for item in scored]
    total_correct = sum(int(item["correct"]) for item in scored)
    total_trials = sum(int(item["n_trials"]) for item in scored)
    trials_ok = (
        not missing
        and not missing_accuracy
        and not inconsistent
        and len(scored) == 10
        and all(int(item["n_trials"]) == EXPECTED_TRIALS_PER_FOLD for item in scored)
        and total_trials == EXPECTED_TOTAL_TRIALS
    )
    status = "COMPLETE" if trials_ok else "PARTIAL"
    summary = {
        "model": model,
        "protocol": PROTOCOL,
        "status": status,
        "n_folds_included": len(entries),
        "n_folds_expected": 10,
        "n_folds_scored": len(scored),
        "missing_folds": missing,
        "folds_missing_outer_test_accuracy": missing_accuracy,
        "inconsistent_folds": inconsistent,
        "present_folds": present,
        "mean_accuracy": round(sum(accuracies) / len(accuracies), 6) if accuracies else None,
        "std_accuracy": round(population_std(accuracies), 6) if accuracies else None,
        "std_accuracy_kind": "population",
        "total_correct": total_correct if scored else None,
        "total_trials": total_trials if scored else None,
        "expected_trials_per_fold": EXPECTED_TRIALS_PER_FOLD,
        "expected_total_trials": EXPECTED_TOTAL_TRIALS,
        "per_fold": entries,
        "provenance": {
            "kind": "aggregated_training_summaries",
            "source_files": [path.name for path in paths],
            "derived_from_per_fold_training_records": True,
        },
    }
    return summary


def refuse_historical_overwrite(path: Path, model: str) -> None:
    if path.name == f"{model}_12jfpm.json":
        raise ValueError(f"refusing to replace historical summary {path}")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def try_write_complete_aggregate(output_root: Path, model: str) -> Path | None:
    """Write aggregate_10fold.json only when all ten fold files form a COMPLETE set."""
    paths = [per_fold_results_path(output_root, model, fold) for fold in EXPECTED_FOLDS]
    if not all(path.is_file() for path in paths):
        return None
    summary = aggregate_fold_files(paths, model=model)
    if summary["status"] != "COMPLETE":
        return None
    destination = aggregate_results_path(output_root, model)
    refuse_historical_overwrite(destination, model)
    write_json_atomic(destination, summary)
    return destination


def _scalar(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return None
    return None


def read_checkpoint_metadata(path: Path) -> dict:
    """Load scalar checkpoint metadata and drop weight tensors immediately."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    payload.pop("model_state_dict", None)
    payload.pop("optimizer_state_dict", None)
    kept = {}
    for key in (
        "epoch",
        "protocol",
        "inner_val_acc",
        "inner_val_loss",
        "fold_idx",
        "outer_test_subject",
        "checkpoint_metric",
        "val_subject",
    ):
        if key in payload:
            scalar = _scalar(payload[key])
            if scalar is not None:
                kept[key] = scalar
    del payload
    return kept


def audit_model(output_root: Path, model: str) -> dict:
    """Read-only inventory of checkpoints and split manifests."""
    folds = []
    for fold in EXPECTED_FOLDS:
        checkpoint = checkpoint_path(output_root, model, fold)
        manifest = split_manifest_path(output_root, model, fold)
        record = {
            "fold": fold,
            "expected_subject": f"s{fold}",
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_present": checkpoint.is_file(),
            "split_manifest": manifest.as_posix(),
            "split_manifest_present": manifest.is_file(),
            "recovered_metadata": {},
            "issues": [],
            "unavailable_historical_metrics": list(UNAVAILABLE_HISTORICAL_METRICS),
        }
        if checkpoint.is_file():
            metadata = read_checkpoint_metadata(checkpoint)
            record["recovered_metadata"] = metadata
            if metadata.get("fold_idx") not in (None, fold):
                record["issues"].append(
                    f"checkpoint fold_idx {metadata.get('fold_idx')} does not match {fold}"
                )
            if metadata.get("outer_test_subject") not in (None, f"s{fold}"):
                record["issues"].append(
                    f"checkpoint subject {metadata.get('outer_test_subject')!r} does not match s{fold}"
                )
            if metadata.get("protocol") not in (None, PROTOCOL):
                record["issues"].append(
                    f"checkpoint protocol {metadata.get('protocol')!r} is not {PROTOCOL}"
                )
        else:
            record["issues"].append("checkpoint missing")
        if manifest.is_file():
            manifest_doc = _read_json(manifest)
            record["split_manifest_metadata"] = {
                key: manifest_doc.get(key)
                for key in ("protocol", "fold_idx", "outer_test_subject", "split_seed")
            }
            if manifest_doc.get("fold_idx") not in (None, fold):
                record["issues"].append("split manifest fold_idx does not match the filename")
            if manifest_doc.get("outer_test_subject") not in (None, f"s{fold}"):
                record["issues"].append("split manifest subject does not match the fold")
        else:
            record["issues"].append("split manifest missing")
        folds.append(record)
    return {
        "kind": "checkpoint_audit",
        "read_only": True,
        "model": model,
        "outer_test_accuracy": "unavailable_from_checkpoints",
        "unavailable_historical_metrics": list(UNAVAILABLE_HISTORICAL_METRICS),
        "folds": folds,
    }


def _checkpoint_matches(reported: str, expected: Path) -> bool:
    reported_path = Path(str(reported))
    if reported_path.name != expected.name:
        return False
    if reported_path.is_absolute():
        try:
            return reported_path.resolve() == expected.resolve()
        except OSError:
            return False
    reported_parts = tuple(reported.replace("\\", "/").split("/"))
    expected_parts = expected.as_posix().split("/")
    return len(reported_parts) >= 3 and reported_parts[-3:] == tuple(expected_parts[-3:])


def evaluation_rejection(document: dict, *, model: str, fold: int, checkpoint: Path) -> str | None:
    """Return why a host summary cannot be reused, using the PyTorch summary schema."""
    if document.get("runtime") != "pytorch":
        return "runtime is not pytorch"
    if document.get("model") != model:
        return "model does not match"
    subject = document.get("held_out_subject", document.get("outer_test_subject"))
    if subject != f"s{fold}":
        return "held-out subject does not match the fold"
    reported_checkpoint = document.get("checkpoint")
    if not isinstance(reported_checkpoint, str) or not _checkpoint_matches(reported_checkpoint, checkpoint):
        return "checkpoint path does not match the fold checkpoint"
    accuracy = document.get("outer_test_accuracy", document.get("accuracy"))
    return score_problem(document.get("n_trials"), document.get("correct"), accuracy)


def _evaluation_metrics(path: Path, *, model: str, fold: int, checkpoint: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    document = _read_json(path)
    reason = evaluation_rejection(document, model=model, fold=fold, checkpoint=checkpoint)
    if reason is not None:
        return None, reason
    accuracy = document.get("outer_test_accuracy", document.get("accuracy"))
    return {
        "outer_test_accuracy": float(accuracy),
        "n_trials": document["n_trials"],
        "correct": document["correct"],
        "source": path.resolve().as_posix(),
    }, None


def recover_model(
    output_root: Path,
    model: str,
    *,
    host_root: Path | None = None,
    write: bool = False,
) -> dict:
    """Build a recovered report. Writes only when ``write`` is true."""
    audit = audit_model(output_root, model)
    reused = []
    reused_sources = []
    rejected = []
    per_fold = []
    for record in audit["folds"]:
        fold = int(record["fold"])
        checkpoint = checkpoint_path(output_root, model, fold)
        evaluation = None
        if host_root is not None:
            source = pytorch_summary_path(host_root, model, fold)
            evaluation, reason = _evaluation_metrics(source, model=model, fold=fold, checkpoint=checkpoint)
            if reason is not None:
                rejected.append({"fold": fold, "source": source.resolve().as_posix(), "reason": reason})
        if evaluation is not None:
            reused.append(fold)
            reused_sources.append(evaluation["source"])
        entry = {
            "fold": fold,
            "outer_test_subject": record["recovered_metadata"].get("outer_test_subject"),
            "checkpoint_epoch": record["recovered_metadata"].get("epoch"),
            "checkpoint_inner_val_acc": record["recovered_metadata"].get("inner_val_acc"),
            "checkpoint_inner_val_loss": record["recovered_metadata"].get("inner_val_loss"),
            "checkpoint_metric": record["recovered_metadata"].get("checkpoint_metric"),
            "issues": list(record["issues"]),
        }
        if evaluation is not None:
            entry["outer_test_accuracy"] = evaluation["outer_test_accuracy"]
            entry["n_trials"] = evaluation["n_trials"]
            entry["correct"] = evaluation["correct"]
            entry["evaluation_source"] = evaluation["source"]
        per_fold.append(entry)
    scored = [
        item for item in per_fold
        if isinstance(item.get("outer_test_accuracy"), float)
        and isinstance(item.get("n_trials"), int)
        and isinstance(item.get("correct"), int)
        and not item["issues"]
    ]
    accuracies = [float(item["outer_test_accuracy"]) for item in scored]
    total_trials = sum(int(item["n_trials"]) for item in scored)
    complete = (
        len(scored) == 10
        and total_trials == EXPECTED_TOTAL_TRIALS
        and all(int(item["n_trials"]) == EXPECTED_TRIALS_PER_FOLD for item in scored)
    )
    report = {
        "kind": "recovered_training_summary",
        "not_original_training_record": True,
        "model": model,
        "protocol": PROTOCOL,
        "status": "COMPLETE" if complete else "PARTIAL",
        "n_folds_included": len(scored),
        "n_folds_expected": 10,
        "mean_accuracy": round(sum(accuracies) / len(accuracies), 6) if accuracies else None,
        "std_accuracy": round(population_std(accuracies), 6) if accuracies else None,
        "std_accuracy_kind": "population",
        "total_correct": sum(int(item["correct"]) for item in scored) if scored else None,
        "total_trials": total_trials if scored else None,
        "per_fold": per_fold,
        "provenance": {
            "kind": "recovered_from_checkpoints_and_evaluation",
            "recovered_checkpoint_metadata": True,
            "reused_pytorch_evaluation_folds": reused,
            "reused_pytorch_evaluation_sources": reused_sources,
            "rejected_pytorch_evaluations": rejected,
            "recomputed": ["mean_accuracy", "std_accuracy", "total_correct", "total_trials"] if scored else [],
            "missing_historical_metrics": list(UNAVAILABLE_HISTORICAL_METRICS),
            "not_original_training_record": True,
        },
    }
    if write:
        destination = recovered_results_path(output_root, model)
        refuse_historical_overwrite(destination, model)
        write_json_atomic(destination, report)
        report["written_to"] = destination.as_posix()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate or audit training summaries")
    parser.add_argument("command", choices=("aggregate", "audit", "recover"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--host-root", type=Path, default=Path("outputs/host"))
    parser.add_argument("--write", action="store_true", help="Write recovered_10fold.json")
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        destination = try_write_complete_aggregate(args.runs_root, args.model)
        if destination is None:
            raise SystemExit(
                f"No COMPLETE ten-fold summary written for {args.model}. "
                "Per-fold files are missing or not a complete 12JFPM set."
            )
        print(destination.as_posix())
        return 0
    if args.command == "audit":
        print(json.dumps(audit_model(args.runs_root, args.model), indent=2))
        return 0
    report = recover_model(
        args.runs_root,
        args.model,
        host_root=args.host_root,
        write=args.write,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
