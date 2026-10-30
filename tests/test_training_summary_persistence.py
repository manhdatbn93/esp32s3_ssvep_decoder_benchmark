"""Per-fold training summaries, aggregation, and read-only checkpoint audit."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_neural = _load("train_neural", ROOT / "scripts" / "train_neural.py")
summaries = _load("training_summaries", ROOT / "scripts" / "training_summaries.py")


def _fold_document(model: str, fold: int, accuracy: float, *, trials: int = 180, subject: str | None = None) -> dict:
    correct = int(round(accuracy * trials))
    return {
        "model": model,
        "protocol": "strict_loso_inner_val",
        "per_fold": [
            {
                "fold": fold,
                "model": model,
                "outer_test_subject": subject or f"s{fold}",
                "protocol": "strict_loso_inner_val",
                "outer_test_accuracy": accuracy,
                "correct": correct,
                "n_trials": trials,
                "final_epoch": 20 + fold,
                "elapsed_seconds": 1.5,
            }
        ],
    }


def _write_fold(directory: Path, model: str, fold: int, accuracy: float, **kwargs) -> Path:
    path = summaries.per_fold_results_path(directory, model, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_fold_document(model, fold, accuracy, **kwargs)), encoding="utf-8")
    return path


def test_single_fold_results_do_not_share_one_file():
    root = Path("runs")
    paths = [train_neural.training_results_arg(root, "eegnet", fold) for fold in range(1, 11)]
    assert len(set(paths)) == 10
    assert paths[0] == root / "eegnet" / "training" / "fold01.json"
    assert paths[9] == root / "eegnet" / "training" / "fold10.json"
    historical = root / "eegnet" / "eegnet_12jfpm.json"
    assert historical not in paths
    multi = train_neural.training_results_arg(root, "eegnet", None)
    assert multi == root / "eegnet"
    command = train_neural.build_train_command(
        model="eegnet",
        fold=1,
        output_root=root,
        dataset_root=Path("data/processed/12JFPM"),
        config=ROOT / "configs/training/eegnet_12jfpm_strict_loso.yaml",
    )
    results = command[command.index("--results-dir") + 1]
    assert results.replace("\\", "/").endswith("eegnet/training/fold01.json")
    checkpoint = command[command.index("--checkpoint-dir") + 1]
    assert checkpoint.replace("\\", "/").endswith("eegnet/checkpoints")
    all_folds = train_neural.build_train_command(
        model="eegnet",
        fold=None,
        output_root=root,
        dataset_root=Path("data/missing"),
        config=ROOT / "configs/training/eegnet_12jfpm_strict_loso.yaml",
    )
    assert all_folds[all_folds.index("--results-dir") + 1].replace("\\", "/").endswith("eegnet")
    assert "--subjects" not in all_folds


def test_aggregate_complete_and_partial(tmp_path: Path):
    for fold in range(1, 11):
        _write_fold(tmp_path, "eegnet", fold, 0.5)
    paths = [summaries.per_fold_results_path(tmp_path, "eegnet", fold) for fold in range(1, 11)]
    summary = summaries.aggregate_fold_files(paths, model="eegnet")
    assert summary["status"] == "COMPLETE"
    assert summary["n_folds_included"] == 10
    assert summary["mean_accuracy"] == 0.5
    assert summary["std_accuracy"] == 0.0
    assert summary["total_trials"] == 1800
    assert summary["total_correct"] == 900
    assert summary["present_folds"] == list(range(1, 11))
    assert summary["per_fold"][0]["final_epoch"] == 21
    written = summaries.try_write_complete_aggregate(tmp_path, "eegnet")
    assert written == summaries.aggregate_results_path(tmp_path, "eegnet")
    assert written.is_file()
    historical = summaries.historical_results_path(tmp_path, "eegnet")
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_text('{"keep": true}', encoding="utf-8")
    summaries.try_write_complete_aggregate(tmp_path, "eegnet")
    assert historical.read_text(encoding="utf-8") == '{"keep": true}'

    partial_paths = paths[:2]
    for path, accuracy in zip(partial_paths, (0.5, 1.0)):
        path.write_text(json.dumps(_fold_document("eegnet", int(path.stem[-2:]), accuracy)), encoding="utf-8")
    partial = summaries.aggregate_fold_files(partial_paths, model="eegnet")
    assert partial["status"] == "PARTIAL"
    assert partial["mean_accuracy"] == 0.75
    assert partial["std_accuracy"] == 0.25
    assert partial["missing_folds"] == list(range(3, 11))
    assert summaries.try_write_complete_aggregate(tmp_path / "partial", "eegnet") is None


def test_aggregate_rejects_bad_identity(tmp_path: Path):
    first = _write_fold(tmp_path / "a", "eegnet", 1, 0.5)
    second = _write_fold(tmp_path / "b", "eegnet", 1, 0.5)
    with pytest.raises(ValueError, match="duplicate fold"):
        summaries.aggregate_fold_files([first, second], model="eegnet")
    mismatched = tmp_path / "eegnet" / "training" / "fold02.json"
    mismatched.parent.mkdir(parents=True, exist_ok=True)
    mismatched.write_text(
        json.dumps(_fold_document("eegnet", 2, 0.5, subject="s9")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out subject"):
        summaries.aggregate_fold_files([mismatched], model="eegnet")
    other = _write_fold(tmp_path / "c", "ccnn", 1, 0.5)
    with pytest.raises(ValueError, match="model"):
        summaries.aggregate_fold_files([other], model="eegnet")
    missing = _fold_document("eegnet", 3, 0.5)
    del missing["per_fold"][0]["outer_test_accuracy"]
    missing_path = summaries.per_fold_results_path(tmp_path, "eegnet", 3)
    missing_path.write_text(json.dumps(missing), encoding="utf-8")
    result = summaries.aggregate_fold_files([missing_path], model="eegnet")
    assert result["status"] == "PARTIAL"
    assert result["folds_missing_outer_test_accuracy"] == [3]
    assert result["mean_accuracy"] is None


def test_audit_is_read_only_and_recovery_uses_evaluation(tmp_path: Path):
    import torch

    checkpoint = summaries.checkpoint_path(tmp_path, "eegnet", 1)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 12,
            "model_state_dict": {"weight": torch.zeros(2)},
            "protocol": "strict_loso_inner_val",
            "inner_val_acc": 0.7,
            "inner_val_loss": 0.4,
            "fold_idx": 1,
            "outer_test_subject": "s1",
            "checkpoint_metric": "inner_val_accuracy",
        },
        checkpoint,
    )
    before = checkpoint.read_bytes()
    manifest = summaries.split_manifest_path(tmp_path, "eegnet", 1)
    manifest.write_text(
        json.dumps({"protocol": "strict_loso_inner_val", "fold_idx": 1, "outer_test_subject": "s1"}),
        encoding="utf-8",
    )
    audit = summaries.audit_model(tmp_path, "eegnet")
    assert checkpoint.read_bytes() == before
    fold = audit["folds"][0]
    assert fold["recovered_metadata"]["epoch"] == 12
    assert "model_state_dict" not in fold["recovered_metadata"]
    assert "final_epoch" in fold["unavailable_historical_metrics"]
    assert audit["outer_test_accuracy"] == "unavailable_from_checkpoints"
    assert audit["folds"][1]["issues"]

    host = tmp_path / "host"
    evaluation = summaries.pytorch_summary_path(host, "eegnet", 1)
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_text(
        json.dumps(
            {
                "runtime": "pytorch",
                "model": "eegnet",
                "held_out_subject": "s1",
                "checkpoint": str(checkpoint),
                "outer_test_accuracy": 0.25,
                "n_trials": 180,
                "correct": 45,
            }
        ),
        encoding="utf-8",
    )
    preview = summaries.recover_model(tmp_path, "eegnet", host_root=host, write=False)
    assert preview["not_original_training_record"] is True
    assert preview["status"] == "PARTIAL"
    assert preview["per_fold"][0]["outer_test_accuracy"] == 0.25
    assert preview["provenance"]["reused_pytorch_evaluation_folds"] == [1]
    assert preview["provenance"]["reused_pytorch_evaluation_sources"] == [evaluation.resolve().as_posix()]
    assert "elapsed_seconds" in preview["provenance"]["missing_historical_metrics"]
    assert not summaries.recovered_results_path(tmp_path, "eegnet").exists()
    written = summaries.recover_model(tmp_path, "eegnet", host_root=host, write=True)
    assert Path(written["written_to"]).name == "recovered_10fold.json"
    assert summaries.historical_results_path(tmp_path, "eegnet").exists() is False


def test_inconsistent_scores_are_not_complete(tmp_path: Path):
    def write(fold: int, **changes) -> None:
        document = _fold_document("eegnet", fold, 0.5)
        document["per_fold"][0].update(changes)
        path = summaries.per_fold_results_path(tmp_path, "eegnet", fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")

    for fold in range(1, 11):
        write(fold, correct=200, outer_test_accuracy=0.5)
    paths = [summaries.per_fold_results_path(tmp_path, "eegnet", fold) for fold in range(1, 11)]
    impossible = summaries.aggregate_fold_files(paths, model="eegnet")
    assert impossible["status"] != "COMPLETE"
    assert impossible["inconsistent_folds"]

    write(1, correct=90, n_trials=180, outer_test_accuracy=0.4)
    mismatch = summaries.aggregate_fold_files(
        [summaries.per_fold_results_path(tmp_path, "eegnet", 1)],
        model="eegnet",
    )
    assert mismatch["status"] == "PARTIAL"
    assert mismatch["inconsistent_folds"][0]["reason"].startswith("outer_test_accuracy does not match")

    for label, changes in (
        ("nan", {"correct": 90, "n_trials": 180, "outer_test_accuracy": float("nan")}),
        ("inf", {"correct": 90, "n_trials": 180, "outer_test_accuracy": float("inf")}),
        ("zero trials", {"correct": 0, "n_trials": 0, "outer_test_accuracy": 0.0}),
        ("negative trials", {"correct": 0, "n_trials": -1, "outer_test_accuracy": 0.0}),
    ):
        write(1, **changes)
        result = summaries.aggregate_fold_files(
            [summaries.per_fold_results_path(tmp_path, "eegnet", 1)],
            model="eegnet",
        )
        assert result["status"] != "COMPLETE", label
        assert result["n_folds_scored"] == 0, label


def test_mismatched_evaluation_is_not_reused(tmp_path: Path):
    import torch

    checkpoints = []
    for fold in range(1, 11):
        checkpoint = summaries.checkpoint_path(tmp_path, "eegnet", fold)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": {"weight": torch.zeros(1)},
                "protocol": "strict_loso_inner_val",
                "inner_val_acc": 0.5,
                "inner_val_loss": 0.5,
                "fold_idx": fold,
                "outer_test_subject": f"s{fold}",
                "checkpoint_metric": "inner_val_accuracy",
            },
            checkpoint,
        )
        summaries.split_manifest_path(tmp_path, "eegnet", fold).write_text(
            json.dumps(
                {
                    "protocol": "strict_loso_inner_val",
                    "fold_idx": fold,
                    "outer_test_subject": f"s{fold}",
                }
            ),
            encoding="utf-8",
        )
        checkpoints.append(checkpoint)
    host = tmp_path / "host"
    bad = summaries.pytorch_summary_path(host, "eegnet", 1)
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        json.dumps(
            {
                "runtime": "tflite",
                "model": "WRONG_MODEL",
                "held_out_subject": "s99",
                "checkpoint": str(checkpoints[0]),
                "outer_test_accuracy": 0.5,
                "n_trials": 180,
                "correct": 90,
            }
        ),
        encoding="utf-8",
    )
    for fold, checkpoint in enumerate(checkpoints[1:], start=2):
        path = summaries.pytorch_summary_path(host, "eegnet", fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "runtime": "pytorch",
                    "model": "eegnet",
                    "held_out_subject": f"s{fold}",
                    "checkpoint": str(checkpoint),
                    "outer_test_accuracy": 0.5,
                    "n_trials": 180,
                    "correct": 90,
                }
            ),
            encoding="utf-8",
        )
    report = summaries.recover_model(tmp_path, "eegnet", host_root=host, write=False)
    assert report["status"] != "COMPLETE"
    assert 1 not in report["provenance"]["reused_pytorch_evaluation_folds"]
    assert report["provenance"]["rejected_pytorch_evaluations"][0]["source"] == bad.resolve().as_posix()
    assert report["n_folds_included"] == 9
