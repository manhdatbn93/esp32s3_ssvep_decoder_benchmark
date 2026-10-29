"""Selection, paths, and failure handling for scripts/run_pipeline.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load()


def _flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_train_all_covers_six_models_and_ten_folds():
    steps = pipeline.build_steps("train", list(pipeline.NEURAL_MODELS), list(pipeline.FOLDS), None)
    assert len(steps) == 60
    pairs = {(_flag(step.argv, "--model"), int(_flag(step.argv, "--fold"))) for step in steps}
    assert pairs == {(model, fold) for model in pipeline.NEURAL_MODELS for fold in pipeline.FOLDS}
    for step in steps:
        assert step.argv[1].endswith("scripts/train_neural.py")
        assert _flag(step.argv, "--output-root").endswith("/runs")
        assert _flag(step.argv, "--dataset-root").endswith("data/processed/12JFPM")
        assert "int8" not in step.argv
        assert "export_tflite.py" not in step.argv[1]


def test_export_all_uses_supported_precisions():
    steps = pipeline.build_steps("export", list(pipeline.NEURAL_MODELS), list(pipeline.FOLDS), None)
    counts: dict[tuple[str, str], int] = {}
    for step in steps:
        model = _flag(step.argv, "--model")
        quant = _flag(step.argv, "--quantization")
        counts[(model, quant)] = counts.get((model, quant), 0) + 1
        assert step.argv[1].endswith("scripts/export_tflite.py")
        assert _flag(step.argv, "--checkpoint").endswith(
            f"runs/{model}/checkpoints/fold{_flag(step.argv, '--fold').zfill(2)}_best.pt"
        )
        assert _flag(step.argv, "--output").endswith(
            f"artifacts/generated/{model}/fold{_flag(step.argv, '--fold').zfill(2)}/{quant}"
        )
    assert len(steps) == 90
    for model, precisions in pipeline.EXPORT_PRECISIONS.items():
        for precision in precisions:
            assert counts[(model, precision)] == 10
        if "int8" not in precisions:
            assert (model, "int8") not in counts
    host_only = [step for step in steps if "ssvepformer" in step.label or "ssvepnet" in step.label]
    assert host_only
    assert all("not firmware-deployable" in step.label for step in host_only)
    int8 = [step for step in steps if _flag(step.argv, "--quantization") == "int8"]
    assert int8
    assert all("PTQ" in step.label for step in int8)


def test_eegnet_fold1_commands_call_existing_scripts():
    train = pipeline.train_step("eegnet", 1)
    assert train.argv[1].endswith("scripts/train_neural.py")
    assert _flag(train.argv, "--model") == "eegnet"
    assert _flag(train.argv, "--fold") == "1"

    f32 = pipeline.export_step("eegnet", 1, "f32")
    int8 = pipeline.export_step("eegnet", 1, "int8")
    assert f32.argv[1].endswith("scripts/export_tflite.py")
    assert _flag(f32.argv, "--quantization") == "f32"
    assert _flag(f32.argv, "--output").endswith("artifacts/generated/eegnet/fold01/f32")
    assert int8.argv[1].endswith("scripts/export_tflite.py")
    assert _flag(int8.argv, "--quantization") == "int8"
    assert _flag(int8.argv, "--checkpoint").endswith("runs/eegnet/checkpoints/fold01_best.pt")
    assert _flag(int8.argv, "--manifest").endswith("manifests/loso/fold01.json")

    evaluation = pipeline.build_steps("evaluate", ["eegnet"], [1], None)
    assert [ _flag(step.argv, "--runtime") for step in evaluation ] == [
        "pytorch",
        "tflite",
        "stage1",
        "tflite",
        "stage1",
    ]
    assert all(step.argv[1].endswith("scripts/evaluate_host.py") for step in evaluation)
    assert _flag(evaluation[1].argv, "--precision") == "f32"
    assert _flag(evaluation[4].argv, "--precision") == "int8"
    assert evaluation[0].output_dir.as_posix().endswith("outputs/host/eegnet/fold01/pytorch")
    assert evaluation[1].output_dir.as_posix().endswith("outputs/host/eegnet/fold01/tflite_f32")
    assert evaluation[3].output_dir.as_posix().endswith("outputs/host/eegnet/fold01/tflite_int8")
    assert evaluation[4].output_dir.as_posix().endswith("outputs/host/eegnet/fold01/stage1_int8")
    assert _flag(evaluation[4].argv, "--tflite").endswith(
        "artifacts/generated/eegnet/fold01/int8/eegnet_int8.tflite"
    )

    staged = pipeline.stage_step("eegnet", 1, "int8")
    assert staged.argv[1].endswith("scripts/generate_firmware_artifacts.py")
    assert _flag(staged.argv, "--backend") == "eegnet-int8"
    assert _flag(staged.argv, "--artifact-set") == "generated"
    assert _flag(staged.argv, "--output").endswith(
        "artifacts/generated/firmware_staging/eegnet/fold01/int8"
    )
    assert "firmware/generated" not in _flag(staged.argv, "--output")


def test_staging_directories_hold_one_backend():
    f32 = pipeline.staging_dir("eegnet", 1, "f32")
    int8 = pipeline.staging_dir("eegnet", 1, "int8")
    pipeline.refuse_result_output(f32)
    pipeline.refuse_result_output(int8)
    assert f32 != int8
    assert f32 not in int8.parents
    assert int8 not in f32.parents
    assert f32 != pipeline.export_dir("eegnet", 1, "f32")
    steps = pipeline.build_steps("stage", list(pipeline.FIRMWARE_MODELS), list(pipeline.FOLDS), None)
    outputs = [step.output_dir for step in steps]
    assert len(outputs) == len(set(outputs))
    assert len(steps) == 70
    for step in steps:
        assert "firmware/generated" not in _flag(step.argv, "--output")
        assert step.argv[1].endswith("scripts/generate_firmware_artifacts.py")


def test_shared_parents_are_refused():
    for path in (
        pipeline.ROOT / "runs",
        pipeline.ROOT / "artifacts" / "generated",
        pipeline.ROOT / "artifacts" / "reference",
        pipeline.ROOT / "firmware" / "generated",
        pipeline.ROOT / "outputs",
        pipeline.ROOT / "artifacts" / "generated" / "firmware_staging",
    ):
        with pytest.raises(SystemExit, match="shared parent"):
            pipeline.refuse_shared_output(path)
    with pytest.raises(SystemExit, match="training runs"):
        pipeline.refuse_result_output(pipeline.RUNS_ROOT / "eegnet" / "checkpoints")
    parent = pipeline.staging_dir("eegnet", 1, "f32").parent
    child = pipeline.staging_dir("eegnet", 1, "int8")
    with pytest.raises(SystemExit, match="multiple firmware backends"):
        pipeline._refuse_nested_staging(
            [
                pipeline.Step("parent", (), output_dir=parent),
                pipeline.Step("child", (), output_dir=child),
            ]
        )


def test_fbtcnn_header_is_copied_from_published_file(tmp_path: Path):
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "rmtree" not in source
    generator = pipeline.FBTCNN_GENERATOR.read_text(encoding="utf-8")
    assert "argparse" not in generator
    assert "12jfpm_fbtcnn_fold01_val_s1" in generator
    published = [pipeline.fbtcnn_header_source(fold) for fold in pipeline.FOLDS]
    assert all(path.is_file() for path in published)
    assert len({path.read_bytes() for path in published}) == 1

    staged = pipeline.stage_step("fbtcnn", 1, "int8")
    assert "gen_fbtcnn_filter_header.py" not in " ".join(staged.argv)
    assert staged.header_when == "before"
    assert staged.header_src == pipeline.fbtcnn_header_source(1)
    assert staged.header_dst == pipeline.export_dir("fbtcnn", 1, "int8") / "fbtcnn_filter_generated.h"
    exported = pipeline.export_step("fbtcnn", 1, "f32")
    assert exported.header_when == "after"
    assert exported.header_dst == pipeline.export_dir("fbtcnn", 1, "f32") / "fbtcnn_filter_generated.h"

    destination = tmp_path / "export" / "fbtcnn_filter_generated.h"
    step = pipeline.Step(
        "copy",
        (),
        header_src=published[0],
        header_dst=destination,
        header_when="before",
    )
    assert pipeline._copy_header(step) is None
    assert destination.read_bytes() == published[0].read_bytes()


def test_cli_selection_and_dry_run(capsys: pytest.CaptureFixture[str]):
    assert pipeline.main(["train", "--model", "eegnet", "--fold", "1", "--dry-run"]) == 0
    train_out = capsys.readouterr().out
    assert "scripts/train_neural.py" in train_out
    assert "--model eegnet" in train_out
    assert "--fold 1" in train_out
    assert "Dry run: 1 command" in train_out

    assert pipeline.main(["export", "--model", "eegnet", "--fold", "1", "--dry-run"]) == 0
    export_out = capsys.readouterr().out
    assert "scripts/export_tflite.py" in export_out
    assert "--quantization f32" in export_out
    assert "--quantization int8" in export_out
    assert "PTQ" in export_out
    assert "Dry run: 2 command" in export_out

    assert pipeline.main(["evaluate", "--model", "eegnet", "--fold", "1", "--dry-run"]) == 0
    evaluate_out = capsys.readouterr().out
    assert "scripts/evaluate_host.py" in evaluate_out
    assert "--runtime pytorch" in evaluate_out
    assert "--runtime tflite" in evaluate_out
    assert "--runtime stage1" in evaluate_out
    assert "Dry run: 5 command" in evaluate_out

    assert pipeline.main(
        ["stage", "--model", "eegnet", "--fold", "1", "--precision", "int8", "--dry-run"]
    ) == 0
    stage_out = capsys.readouterr().out
    assert "scripts/generate_firmware_artifacts.py" in stage_out
    assert "--backend eegnet-int8" in stage_out
    assert "firmware_staging/eegnet/fold01/int8" in stage_out
    assert "Dry run: 1 command" in stage_out

    assert pipeline.main(["stage", "--all", "--dry-run"]) == 0
    all_out = capsys.readouterr().out
    assert "ssvepformer: host-side float32 export only" in all_out
    assert "ssvepnet: host-side float32 export only" in all_out
    assert "--backend ssvepformer-f32" not in all_out
    assert "--backend ssvepnet-f32" not in all_out
    assert "--backend convca-int8" not in all_out
    assert "--backend convca-f32" in all_out
    assert "firmware/generated" not in all_out
    assert "Dry run: 70 command" in all_out


def test_cli_rejects_invalid_selection():
    with pytest.raises(SystemExit, match="cannot be combined"):
        pipeline.main(["train", "--all", "--model", "eegnet", "--dry-run"])
    with pytest.raises(SystemExit, match="pass --all or --model"):
        pipeline.main(["export", "--dry-run"])
    with pytest.raises(SystemExit, match="--precision is required"):
        pipeline.main(["stage", "--model", "eegnet", "--fold", "1", "--dry-run"])
    with pytest.raises(SystemExit, match="not deployable"):
        pipeline.main(["stage", "--model", "ssvepformer", "--fold", "1", "--precision", "f32"])
    with pytest.raises(SystemExit, match="not deployable"):
        pipeline.main(["stage", "--model", "ssvepnet", "--fold", "1", "--precision", "f32"])
    with pytest.raises(SystemExit, match="Unsupported precision"):
        pipeline.main(["stage", "--model", "convca", "--fold", "1", "--precision", "int8"])
    with pytest.raises(SystemExit, match="1..10"):
        pipeline.main(["train", "--model", "eegnet", "--fold", "11", "--dry-run"])


def test_failure_stops_later_commands(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, ...]] = []

    def fake_call(argv, cwd):
        calls.append(tuple(argv))
        return 1

    monkeypatch.setattr(pipeline.subprocess, "call", fake_call)
    steps = [
        pipeline.Step("first", ("first-command",)),
        pipeline.Step("second", ("second-command",)),
    ]
    assert pipeline.execute(steps, dry_run=False) == 1
    assert calls == [("first-command",)]


def test_dry_run_does_not_execute(monkeypatch: pytest.MonkeyPatch):
    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess was called")

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("header was copied")

    monkeypatch.setattr(pipeline.subprocess, "call", fail_call)
    monkeypatch.setattr(pipeline.shutil, "copy2", fail_copy)
    steps = pipeline.build_steps("export", ["fbtcnn"], [1], "int8")
    assert pipeline.execute(steps, dry_run=True) == 0


def test_missing_input_stops_before_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[int] = []
    monkeypatch.setattr(pipeline.subprocess, "call", lambda *_a, **_k: calls.append(1) or 0)
    missing = tmp_path / "missing.pt"
    step = pipeline.Step("export", ("python", "scripts/export_tflite.py"), requires=(missing,))
    later = pipeline.Step("later", ("python", "scripts/evaluate_host.py"))
    assert pipeline.execute([step, later], dry_run=False) == 1
    assert calls == []


def test_missing_fbtcnn_header_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[int] = []
    monkeypatch.setattr(pipeline.subprocess, "call", lambda *_a, **_k: calls.append(1) or 0)
    step = pipeline.Step(
        "stage",
        ("python", "scripts/generate_firmware_artifacts.py"),
        header_src=tmp_path / "missing.h",
        header_dst=tmp_path / "export" / "fbtcnn_filter_generated.h",
        header_when="before",
    )
    later = pipeline.Step("later", ("next",))
    assert pipeline.execute([step, later], dry_run=False) == 1
    assert calls == []
    assert not (tmp_path / "export").exists()
