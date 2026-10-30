"""Selection, paths, and failure handling for scripts/run_pipeline.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
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
    models, folds, precision, notes = pipeline.resolve_selection(
        "export", all_models=True, model=None, fold=None, precision=None,
    )
    steps = pipeline.build_steps("export", models, folds, precision)
    counts: dict[tuple[str, str], int] = {}
    for step in steps:
        model = _flag(step.argv, "--model")
        quant = _flag(step.argv, "--quantization")
        counts[(model, quant)] = counts.get((model, quant), 0) + 1
        assert model not in pipeline.PYTORCH_ONLY_MODELS
        assert step.argv[1].endswith("scripts/export_tflite.py")
        assert _flag(step.argv, "--checkpoint").endswith(
            f"runs/{model}/checkpoints/fold{_flag(step.argv, '--fold').zfill(2)}_best.pt"
        )
        assert _flag(step.argv, "--output").endswith(
            f"artifacts/generated/{model}/fold{_flag(step.argv, '--fold').zfill(2)}/{quant}"
        )
    assert len(steps) == 70
    for model, precisions in pipeline.EXPORT_PRECISIONS.items():
        for item in precisions:
            assert counts[(model, item)] == 10
        if "int8" not in precisions:
            assert (model, "int8") not in counts
    assert ("ssvepformer", "f32") not in counts
    assert ("ssvepnet", "f32") not in counts
    assert pipeline.pytorch_only_note("ssvepformer") in notes
    assert pipeline.pytorch_only_note("ssvepnet") in notes
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
    assert "ssvepformer: PyTorch-only" in all_out
    assert "ssvepnet: PyTorch-only" in all_out
    assert "--backend ssvepformer-f32" not in all_out
    assert "--backend ssvepnet-f32" not in all_out
    assert "--backend convca-int8" not in all_out
    assert "--backend convca-f32" in all_out
    assert "firmware/generated" not in all_out
    assert "--artifact-set reference" in all_out
    assert "--backend cca" in all_out
    assert "--backend trca" in all_out
    assert "artifacts/reference" not in all_out.split("--output ")[-1]
    assert "Dry run: 110 command" in all_out


def test_cli_rejects_invalid_selection():
    with pytest.raises(SystemExit, match="cannot be combined"):
        pipeline.main(["train", "--all", "--model", "eegnet", "--dry-run"])
    with pytest.raises(SystemExit, match="pass --all or --model"):
        pipeline.main(["export", "--dry-run"])
    with pytest.raises(SystemExit, match="--precision is required"):
        pipeline.main(["stage", "--model", "eegnet", "--fold", "1", "--dry-run"])
    with pytest.raises(SystemExit, match="PyTorch-only"):
        pipeline.main(["stage", "--model", "ssvepformer", "--fold", "1", "--precision", "f32"])
    with pytest.raises(SystemExit, match="PyTorch-only"):
        pipeline.main(["stage", "--model", "ssvepnet", "--fold", "1", "--precision", "f32"])
    with pytest.raises(SystemExit, match="PyTorch-only"):
        pipeline.main(["export", "--model", "ssvepformer", "--fold", "1", "--dry-run"])
    with pytest.raises(SystemExit, match="experimental converter"):
        pipeline.main(["export", "--model", "ssvepnet", "--dry-run"])
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


def _script_name(step: pipeline.Step) -> str:
    return Path(step.argv[1]).name


def test_run_eegnet_fold1_follows_phase_order():
    steps, notes = pipeline.build_workflow(["eegnet"], [1])
    assert notes == []
    names = [_script_name(step) for step in steps]
    assert names == [
        "train_neural.py",
        "export_tflite.py",
        "export_tflite.py",
        "evaluate_host.py",
        "evaluate_host.py",
        "evaluate_host.py",
        "evaluate_host.py",
        "evaluate_host.py",
        "generate_firmware_artifacts.py",
        "generate_firmware_artifacts.py",
    ]
    assert [_flag(step.argv, "--quantization") for step in steps[1:3]] == ["f32", "int8"]
    assert [_flag(step.argv, "--runtime") for step in steps[3:8]] == [
        "pytorch",
        "tflite",
        "stage1",
        "tflite",
        "stage1",
    ]
    assert [_flag(step.argv, "--backend") for step in steps[8:]] == ["eegnet-f32", "eegnet-int8"]
    assert steps[0].argv[1].endswith("scripts/train_neural.py")
    assert _flag(steps[8].argv, "--output").endswith(
        "artifacts/generated/firmware_staging/eegnet/fold01/f32"
    )


def test_run_limits_staging_to_supported_backends():
    convca, convca_notes = pipeline.build_workflow(["convca"], [1])
    assert convca_notes == []
    assert [_flag(step.argv, "--quantization") for step in convca if "--quantization" in step.argv] == ["f32"]
    assert [_flag(step.argv, "--backend") for step in convca if "--backend" in step.argv] == ["convca-f32"]

    for model in pipeline.PYTORCH_ONLY_MODELS:
        host_only, notes = pipeline.build_workflow([model], [1])
        assert notes == [pipeline.pytorch_only_note(model)]
        assert [_script_name(step) for step in host_only] == [
            "train_neural.py",
            "evaluate_host.py",
        ]
        assert _flag(host_only[1].argv, "--runtime") == "pytorch"
        assert "--tflite" not in host_only[1].argv

    models, folds, _, _ = pipeline.resolve_selection(
        "run", all_models=True, model=None, fold=None, precision=None,
    )
    steps, all_notes = pipeline.build_workflow(models, folds, include_classical=True)
    assert len(steps) == 441
    names = [_script_name(step) for step in steps]
    assert names[:60] == ["train_neural.py"] * 60
    assert names[60:130] == ["export_tflite.py"] * 70
    assert names[130:330] == ["evaluate_host.py"] * 200
    assert names[330] == "evaluate_host.py"
    assert names[331:401] == ["generate_firmware_artifacts.py"] * 70
    assert names[401:] == ["generate_firmware_artifacts.py"] * 40
    assert all_notes == [
        pipeline.pytorch_only_note("ssvepformer"),
        pipeline.pytorch_only_note("ssvepnet"),
    ]
    exported = {_flag(step.argv, "--model") for step in steps[60:130]}
    assert exported == set(pipeline.EXPORT_PRECISIONS)
    neural_eval = steps[130:330]
    assert sum(_flag(step.argv, "--runtime") == "pytorch" for step in neural_eval) == 60
    assert sum(_flag(step.argv, "--runtime") == "tflite" for step in neural_eval) == 70
    assert sum(_flag(step.argv, "--runtime") == "stage1" for step in neural_eval) == 70
    classical = steps[330]
    assert _flag(classical.argv, "--runtime") == "classical"
    assert _flag(classical.argv, "--method") == "all"
    assert _flag(classical.argv, "--trca-realization") == "deployed_sos"
    assert _flag(classical.argv, "--output-dir").endswith("outputs/host/classical")
    assert classical.requires == (pipeline.PROCESSED_NPZ,)
    neural_stage = steps[331:401]
    assert {_flag(step.argv, "--artifact-set") for step in neural_stage} == {"generated"}
    staged = " ".join(_flag(step.argv, "--backend") for step in neural_stage)
    assert "ssvepformer-f32" not in staged
    assert "ssvepnet-f32" not in staged
    assert "convca-int8" not in staged
    assert "firmware/generated" not in staged
    reference_stage = steps[401:]
    assert [(_flag(step.argv, "--backend"), int(_flag(step.argv, "--fold"))) for step in reference_stage] == [
        (method, fold) for method in pipeline.CLASSICAL_METHODS for fold in pipeline.FOLDS
    ]
    assert {_flag(step.argv, "--artifact-set") for step in reference_stage} == {"reference"}


def test_evaluate_and_stage_all_include_classical():
    models, folds, precision, _notes = pipeline.resolve_selection(
        "evaluate", all_models=True, model=None, fold=None, precision=None,
    )
    evaluation = pipeline.build_steps("evaluate", models, folds, precision)
    evaluation.append(pipeline.classical_step())
    assert len(evaluation) == 201
    assert _flag(evaluation[-1].argv, "--runtime") == "classical"
    assert _flag(evaluation[-1].argv, "--method") == "all"
    assert _flag(evaluation[-1].argv, "--trca-realization") == "deployed_sos"
    assert sum(_flag(step.argv, "--runtime") == "classical" for step in evaluation) == 1

    staged_models, staged_folds, staged_precision, _stage_notes = pipeline.resolve_selection(
        "stage", all_models=True, model=None, fold=None, precision=None,
    )
    staged = pipeline.with_classical_reference_staging(
        pipeline.build_steps("stage", staged_models, staged_folds, staged_precision),
        staged_folds,
    )
    assert len(staged) == 110
    outputs = [step.output_dir.resolve() for step in staged]
    assert len(outputs) == len(set(outputs))
    reference_root = (pipeline.ROOT / "artifacts" / "reference").resolve()
    generated_root = (pipeline.ROOT / "firmware" / "generated").resolve()
    for left_index, left in enumerate(outputs):
        assert left != reference_root
        assert reference_root not in left.parents
        assert left != generated_root
        for right in outputs[left_index + 1 :]:
            assert left not in right.parents
            assert right not in left.parents
    for step in staged[-40:]:
        backend = _flag(step.argv, "--backend")
        fold = int(_flag(step.argv, "--fold"))
        assert step.output_dir == pipeline.reference_staging_dir(backend, fold)
        assert "artifacts/reference" not in _flag(step.argv, "--output")


def test_run_dry_run_and_failure(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess was called")

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("header was copied")

    monkeypatch.setattr(pipeline.subprocess, "call", fail_call)
    monkeypatch.setattr(pipeline.shutil, "copy2", fail_copy)
    assert pipeline.main(["run", "--model", "eegnet", "--fold", "1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Dry run: 10 command" in out
    train_at = out.index("scripts/train_neural.py")
    export_at = out.index("scripts/export_tflite.py", train_at)
    evaluate_at = out.index("scripts/evaluate_host.py", export_at)
    stage_at = out.index("scripts/generate_firmware_artifacts.py", evaluate_at)
    assert train_at < export_at < evaluate_at < stage_at

    assert pipeline.main(["run", "--model", "ssvepformer", "--fold", "1", "--dry-run"]) == 0
    host_out = capsys.readouterr().out
    assert "ssvepformer: PyTorch-only" in host_out
    assert "export_tflite.py" not in host_out
    assert "generate_firmware_artifacts.py" not in host_out
    assert "--runtime pytorch" in host_out
    assert "Dry run: 2 command" in host_out

    calls: list[str] = []

    def fail_first(argv, cwd):
        calls.append(Path(argv[1]).name)
        return 1

    monkeypatch.setattr(pipeline.subprocess, "call", fail_first)
    steps, _notes = pipeline.build_workflow(["eegnet"], [1])
    runnable = [replace(step, requires=()) for step in steps]
    assert pipeline.execute(runnable, dry_run=False) == 1
    assert calls == ["train_neural.py"]


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


def test_dry_run_does_not_create_a_run_log(monkeypatch: pytest.MonkeyPatch):
    import pipeline_log

    def boom(*_args, **_kwargs):
        raise AssertionError("run log created")

    monkeypatch.setattr(pipeline_log.RunLog, "create", boom)
    assert pipeline.main(["train", "--model", "eegnet", "--fold", "1", "--dry-run"]) == 0
    assert pipeline.execute(
        [pipeline.Step("preview", ("python", "scripts/train_neural.py"))],
        dry_run=True,
    ) == 0


def test_pipeline_log_success_failure_and_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pipeline_log

    first = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "run", ["run", "--all"], 1)
    second = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "run", ["run", "--all"], 1)
    assert first.directory != second.directory
    assert first.summary["status"] == "RUNNING"
    assert json.loads(first.summary_path.read_text(encoding="utf-8"))["planned_steps"] == 1

    ok = pipeline.Step(
        "print ok",
        (sys.executable, "-c", "print('ok')"),
        phase="evaluate",
        model="synthetic",
    )
    assert pipeline.execute([ok], dry_run=False, logger=first) == 0
    summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED"
    assert summary["completed_steps"] == 1
    assert summary["failed_steps"] == 0
    assert summary["remaining_steps"] == 0
    events = [json.loads(line) for line in first.events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["RUNNING", "COMPLETED"]
    assert "ok" in (first.directory / events[-1]["log"]).read_text(encoding="utf-8")

    failed = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 3)
    bad = pipeline.Step(
        "exit 3",
        (sys.executable, "-c", "import sys; sys.exit(3)"),
        phase="export",
        model="eegnet",
        fold=1,
        precision="f32",
    )
    later = pipeline.Step("later", (sys.executable, "-c", "print('later')"), phase="stage")
    assert pipeline.execute([ok, bad, later], dry_run=False, logger=failed) == 3
    failed_summary = json.loads(failed.summary_path.read_text(encoding="utf-8"))
    assert failed_summary["status"] == "FAILED"
    assert failed_summary["completed_steps"] == 1
    assert failed_summary["failed_steps"] == 1
    assert failed_summary["remaining_steps"] == 1
    assert failed_summary["first_failed_step_index"] == 2
    assert failed_summary["failure_exit_code"] == 3
    assert "later" not in failed.events_path.read_text(encoding="utf-8")

    snow = pipeline.Step(
        "snowman",
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write('\u2603'.encode())"),
        phase="export",
    )
    encoded = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 1)
    assert pipeline.execute([snow], dry_run=False, logger=encoded) == 0
    logged = list((encoded.directory / "logs").glob("*.log"))[0].read_bytes()
    assert "\u2603".encode() in logged

    binary = pipeline.Step(
        "binary",
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"),
        phase="export",
    )
    binary_log = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 1)
    assert pipeline.execute([binary], dry_run=False, logger=binary_log) == 0

    missing = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 2)
    blocked = pipeline.Step("need file", ("python", "scripts/export_tflite.py"), requires=(tmp_path / "missing.pt",))
    assert pipeline.execute([blocked, later], dry_run=False, logger=missing) == 1
    assert json.loads(missing.summary_path.read_text(encoding="utf-8"))["status"] == "FAILED"
    assert (missing.directory / "logs").exists()

    header = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "stage", ["stage"], 1)
    header_step = pipeline.Step(
        "header",
        ("python", "scripts/generate_firmware_artifacts.py"),
        header_src=tmp_path / "missing.h",
        header_dst=tmp_path / "out" / "fbtcnn_filter_generated.h",
        header_when="before",
        phase="stage",
        model="fbtcnn",
    )
    assert pipeline.execute([header_step], dry_run=False, logger=header) == 1
    assert not (tmp_path / "out").exists()

    interrupted = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "train", ["train"], 1)

    def raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_log, "run_logged_process", raise_interrupt)
    running = pipeline.Step("sleep", (sys.executable, "-c", "print('no')"), phase="train", model="synthetic", fold=1)
    assert pipeline.execute([running], dry_run=False, logger=interrupted) == 130
    interrupted_summary = json.loads(interrupted.summary_path.read_text(encoding="utf-8"))
    assert interrupted_summary["status"] == "INTERRUPTED"
    statuses = [json.loads(line)["status"] for line in interrupted.events_path.read_text(encoding="utf-8").splitlines()]
    assert statuses == ["RUNNING", "INTERRUPTED"]
    assert "COMPLETED" not in statuses

    orphan = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "train", ["train"], 1)
    orphan.begin_step(1, running)
    last = json.loads(orphan.events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "RUNNING"
    assert json.loads(orphan.summary_path.read_text(encoding="utf-8"))["status"] == "RUNNING"


def test_flushed_output_is_logged_before_child_exits(tmp_path: Path):
    import threading
    import time

    import pipeline_log

    script = tmp_path / "slow_print.py"
    script.write_text(
        "import sys, time\n"
        "sys.stdout.buffer.write(b'READY\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(1.5)\n"
        "sys.stdout.buffer.write(b'DONE\\n')\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    log = tmp_path / "live.log"
    result: dict[str, int] = {}

    def runner() -> None:
        result["code"] = pipeline_log.run_logged_process(
            [sys.executable, str(script)],
            tmp_path,
            log,
        )

    thread = threading.Thread(target=runner)
    thread.start()
    seen = False
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if log.is_file() and b"READY" in log.read_bytes():
            seen = True
            break
        time.sleep(0.05)
    still_running = thread.is_alive()
    thread.join(timeout=5)
    assert seen
    assert still_running
    assert result["code"] == 0
    assert b"DONE" in log.read_bytes()


def test_subprocess_launch_failure_marks_run_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pipeline_log

    def boom(*_args, **_kwargs):
        raise OSError("simulated launch failure")

    monkeypatch.setattr(pipeline_log.subprocess, "Popen", boom)
    logger = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 1)
    step = pipeline.Step(
        "launch",
        (sys.executable, "-c", "print('should not run')"),
        phase="export",
        model="eegnet",
        fold=1,
        precision="f32",
    )
    assert pipeline.execute([step], dry_run=False, logger=logger) == 1
    summary = json.loads(logger.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["completed_steps"] == 0
    assert summary["first_failed_step_index"] == 1
    log_text = (logger.directory / summary["failed_log"]).read_text(encoding="utf-8")
    assert "simulated launch failure" in log_text
    statuses = [json.loads(line)["status"] for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert "COMPLETED" not in statuses
    assert statuses[-1] == "FAILED"


def test_unexpected_runtime_error_is_recorded_then_reraised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pipeline_log

    def boom(_step, _log_path):
        raise RuntimeError("orchestrator broke")

    monkeypatch.setattr(pipeline, "_run_step", boom)
    logger = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 1)
    step = pipeline.Step("explode", ("python", "export"), phase="export", model="eegnet", fold=1)
    with pytest.raises(RuntimeError, match="orchestrator broke"):
        pipeline.execute([step], dry_run=False, logger=logger)
    summary = json.loads(logger.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["completed_steps"] == 0
    log_text = (logger.directory / summary["failed_log"]).read_text(encoding="utf-8")
    assert "RuntimeError" in log_text
    assert "Traceback" in log_text
    statuses = [json.loads(line)["status"] for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert statuses == ["RUNNING", "FAILED"]

    def hide_failure(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline_log.RunLog, "finish_step", hide_failure)
    hidden = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "export", ["export"], 1)
    with pytest.raises(RuntimeError, match="orchestrator broke"):
        pipeline.execute([step], dry_run=False, logger=hidden)


def test_aggregation_failure_is_recorded_then_reraised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pipeline_log

    def boom(_model):
        raise RuntimeError("aggregate broke")

    monkeypatch.setattr(pipeline, "_maybe_aggregate", boom)
    logger = pipeline_log.RunLog.create(tmp_path, pipeline.ROOT, "train", ["train"], 2)
    step = pipeline.Step(
        "train synthetic fold 01",
        (sys.executable, "-c", "print('trained')"),
        phase="train",
        model="synthetic",
        fold=1,
    )
    later = pipeline.Step("later", (sys.executable, "-c", "print('later')"), phase="export")
    with pytest.raises(RuntimeError, match="aggregate broke"):
        pipeline.execute([step, later], dry_run=False, logger=logger)
    summary = json.loads(logger.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["completed_steps"] == 0
    assert summary["first_failed_step_label"] == step.label
    log_text = (logger.directory / summary["failed_log"]).read_text(encoding="utf-8")
    assert "RuntimeError: aggregate broke" in log_text
    statuses = [json.loads(line)["status"] for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert "COMPLETED" not in statuses
    assert statuses[-1] == "FAILED"
    assert "later" not in logger.events_path.read_text(encoding="utf-8")
