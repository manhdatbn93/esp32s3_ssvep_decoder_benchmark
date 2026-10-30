#!/usr/bin/env python3
"""Thin orchestrator for training, TFLite export, host evaluation, and firmware staging.

This script only builds and runs the existing public commands. It does not
train, convert, quantize, evaluate, or package firmware itself.

  python scripts/run_pipeline.py run --all
  python scripts/run_pipeline.py train --all
  python scripts/run_pipeline.py export --all
  python scripts/run_pipeline.py evaluate --all
  python scripts/run_pipeline.py train --model eegnet --fold 1
  python scripts/run_pipeline.py export --model eegnet --fold 1
  python scripts/run_pipeline.py evaluate --model eegnet --fold 1
  python scripts/run_pipeline.py stage --model eegnet --fold 1 --precision int8

``--dry-run`` prints the planned commands and does not run them. The first
failing command stops the pipeline.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

NEURAL_MODELS = (
    "ccnn",
    "convca",
    "eegnet",
    "fbtcnn",
    "ssvepformer",
    "ssvepnet",
)
# Standard-workflow TFLite exports. INT8 is post-training quantization of the
# float checkpoint, not a separate training run. SSVEPformer and SSVEPNet stay
# out of this map: scripts/export_tflite.py still has an experimental float32
# converter, and this pipeline does not call it.
EXPORT_PRECISIONS = {
    "ccnn": ("f32", "int8"),
    "convca": ("f32",),
    "eegnet": ("f32", "int8"),
    "fbtcnn": ("f32", "int8"),
}
# Firmware backends accepted by scripts/generate_firmware_artifacts.py.
FIRMWARE_MODELS = ("ccnn", "convca", "eegnet", "fbtcnn")
PYTORCH_ONLY_MODELS = ("ssvepformer", "ssvepnet")
CLASSICAL_METHODS = ("cca", "fbcca", "msi", "trca")
FOLDS = tuple(range(1, 11))

DATASET_ROOT = ROOT / "data" / "processed" / "12JFPM"
PROCESSED_NPZ = DATASET_ROOT / "processed.npz"
RUNS_ROOT = ROOT / "runs"

# The existing generator has no output argument. It always writes here.
# Coefficients are subject-independent Butterworth banks. The published copy
# for each fold already lives under artifacts/reference/preprocessing/fbtcnn/.
FBTCNN_GENERATOR = ROOT / "src" / "export" / "gen_fbtcnn_filter_header.py"
FBTCNN_GENERATOR_OUTPUT = (
    ROOT / "src" / "export" / "12jfpm_fbtcnn_fold01_val_s1" / "fbtcnn_filter_generated.h"
)
FBTCNN_HEADER_NAME = "fbtcnn_filter_generated.h"

_SHARED_OUTPUTS = (
    ROOT,
    ROOT / "runs",
    ROOT / "artifacts",
    ROOT / "artifacts" / "generated",
    ROOT / "artifacts" / "reference",
    ROOT / "artifacts" / "generated" / "firmware_staging",
    ROOT / "outputs",
    ROOT / "outputs" / "host",
    ROOT / "firmware",
    ROOT / "firmware" / "generated",
    ROOT / "data",
    ROOT / "data" / "processed",
    ROOT / "data" / "raw",
)


@dataclass(frozen=True)
class Step:
    """One public command, plus an optional FBtCNN header copy around it."""

    label: str
    argv: tuple[str, ...]
    requires: tuple[Path, ...] = ()
    header_src: Path | None = None
    header_dst: Path | None = None
    header_when: str | None = None
    output_dir: Path | None = None
    phase: str | None = None
    model: str | None = None
    fold: int | None = None
    precision: str | None = None


def checkpoint_path(model: str, fold: int) -> Path:
    return RUNS_ROOT / model / "checkpoints" / f"fold{fold:02d}_best.pt"


def export_dir(model: str, fold: int, precision: str) -> Path:
    return ROOT / "artifacts" / "generated" / model / f"fold{fold:02d}" / precision


def tflite_path(model: str, fold: int, precision: str) -> Path:
    name = f"{model}_float32.tflite" if precision == "f32" else f"{model}_int8.tflite"
    return export_dir(model, fold, precision) / name


def host_dir(model: str, fold: int, runtime: str) -> Path:
    return ROOT / "outputs" / "host" / model / f"fold{fold:02d}" / runtime


def staging_dir(model: str, fold: int, precision: str) -> Path:
    return (
        ROOT
        / "artifacts"
        / "generated"
        / "firmware_staging"
        / model
        / f"fold{fold:02d}"
        / precision
    )


def reference_staging_dir(method: str, fold: int) -> Path:
    """Isolated copy destination for one retained classical backend and fold."""
    return (
        ROOT
        / "artifacts"
        / "generated"
        / "firmware_staging"
        / method
        / f"fold{fold:02d}"
        / "reference"
    )


def manifest_path(fold: int) -> Path:
    return ROOT / "manifests" / "loso" / f"fold{fold:02d}.json"


def fbtcnn_header_source(fold: int) -> Path:
    return (
        ROOT
        / "artifacts"
        / "reference"
        / "preprocessing"
        / "fbtcnn"
        / f"fold{fold:02d}"
        / FBTCNN_HEADER_NAME
    )


def refuse_shared_output(path: Path) -> None:
    """Reject a directory the firmware stager would wipe if it is a shared parent.

    ``generate_firmware_artifacts.py`` deletes its ``--output`` directory.
    Checkpoints, exports, and other results must not sit in that directory.
    """
    resolved = path.resolve()
    for shared in _SHARED_OUTPUTS:
        if resolved == shared.resolve():
            raise SystemExit(
                f"Refusing to use a shared parent as an output directory: {resolved}"
            )


def refuse_result_output(path: Path) -> None:
    """Reject export, evaluation, and staging paths that would wipe other results."""
    refuse_shared_output(path)
    resolved = path.resolve()
    runs = RUNS_ROOT.resolve()
    if resolved == runs or runs in resolved.parents:
        raise SystemExit(f"Refusing to place this output under training runs: {resolved}")


def capability_error(model: str) -> SystemExit:
    return SystemExit(
        f"{model} is PyTorch-only in the standard workflow. "
        "TFLite export, TFLite evaluation, Stage 1, and firmware staging are not included. "
        "scripts/export_tflite.py still contains an experimental converter; "
        "this pipeline does not call it and does not treat that conversion as validated."
    )


def pytorch_only_note(model: str) -> str:
    return (
        f"{model}: PyTorch-only in the standard workflow; "
        "TFLite export, TFLite evaluation, Stage 1, and firmware staging are not included"
    )


def _precisions(model: str, precision: str | None) -> tuple[str, ...]:
    if model not in EXPORT_PRECISIONS:
        raise capability_error(model)
    allowed = EXPORT_PRECISIONS[model]
    if precision is None:
        return allowed
    if precision not in allowed:
        raise SystemExit(
            f"Unsupported precision {precision} for {model}. "
            f"Supported: {', '.join(allowed)}"
        )
    return (precision,)


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).as_posix()


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(argv: tuple[str, ...]) -> str:
    parts = []
    for token in argv:
        if any(ch in token for ch in " \t"):
            parts.append(f'"{token}"')
        else:
            parts.append(token)
    return " ".join(parts)


def train_step(model: str, fold: int) -> Step:
    output = RUNS_ROOT
    checkpoint = checkpoint_path(model, fold)
    if checkpoint.parent != output / model / "checkpoints":
        raise SystemExit(f"Unexpected checkpoint directory: {checkpoint}")
    return Step(
        label=f"train {model} fold {fold:02d}",
        argv=(
            sys.executable,
            _script("train_neural.py"),
            "--model",
            model,
            "--fold",
            str(fold),
            "--dataset-root",
            DATASET_ROOT.as_posix(),
            "--output-root",
            output.as_posix(),
        ),
        requires=(PROCESSED_NPZ,),
        output_dir=checkpoint_path(model, fold).parent,
        phase="train",
        model=model,
        fold=fold,
    )


def export_step(model: str, fold: int, precision: str) -> Step:
    if model not in EXPORT_PRECISIONS:
        raise capability_error(model)
    out = export_dir(model, fold, precision)
    refuse_result_output(out)
    kind = "PTQ of the existing checkpoint" if precision == "int8" else "float32 export"
    label = f"export {model} fold {fold:02d} {precision} ({kind})"
    header_src = header_dst = header_when = None
    if model == "fbtcnn":
        header_src = fbtcnn_header_source(fold)
        header_dst = out / FBTCNN_HEADER_NAME
        header_when = "after"
    return Step(
        label=label,
        argv=(
            sys.executable,
            _script("export_tflite.py"),
            "--model",
            model,
            "--fold",
            str(fold),
            "--checkpoint",
            checkpoint_path(model, fold).as_posix(),
            "--dataset-root",
            DATASET_ROOT.as_posix(),
            "--manifest",
            manifest_path(fold).as_posix(),
            "--quantization",
            precision,
            "--output",
            out.as_posix(),
        ),
        requires=(checkpoint_path(model, fold), manifest_path(fold), PROCESSED_NPZ),
        header_src=header_src,
        header_dst=header_dst,
        header_when=header_when,
        output_dir=out,
        phase="export",
        model=model,
        fold=fold,
        precision=precision,
    )


def _eval_step(model: str, fold: int, runtime: str, precision: str | None) -> Step:
    if runtime == "pytorch":
        out = host_dir(model, fold, "pytorch")
        label = f"evaluate {model} fold {fold:02d} pytorch"
    elif runtime == "tflite":
        out = host_dir(model, fold, f"tflite_{precision}")
        label = f"evaluate {model} fold {fold:02d} tflite {precision}"
    elif runtime == "stage1":
        out = host_dir(model, fold, f"stage1_{precision}")
        label = f"evaluate {model} fold {fold:02d} stage1 {precision}"
    else:
        raise SystemExit(f"Unknown evaluation runtime: {runtime}")
    refuse_result_output(out)
    argv = [
        sys.executable,
        _script("evaluate_host.py"),
        "--runtime",
        runtime,
        "--model",
        model,
        "--fold",
        str(fold),
        "--artifact-set",
        "generated",
        "--checkpoint",
        checkpoint_path(model, fold).as_posix(),
        "--npz",
        PROCESSED_NPZ.as_posix(),
        "--manifest",
        manifest_path(fold).as_posix(),
        "--output-dir",
        out.as_posix(),
    ]
    requires = [checkpoint_path(model, fold), PROCESSED_NPZ, manifest_path(fold)]
    if precision is not None:
        tflite = tflite_path(model, fold, precision)
        argv.extend(["--precision", precision, "--tflite", tflite.as_posix()])
        requires.append(tflite)
    return Step(
        label=label,
        argv=tuple(argv),
        requires=tuple(requires),
        output_dir=out,
        phase="evaluate",
        model=model,
        fold=fold,
        precision=precision,
    )


def stage_step(model: str, fold: int, precision: str) -> Step:
    if model not in FIRMWARE_MODELS:
        raise capability_error(model)
    if precision not in EXPORT_PRECISIONS[model]:
        raise SystemExit(
            f"Unsupported precision {precision} for {model}. "
            f"Supported: {', '.join(EXPORT_PRECISIONS[model])}"
        )
    out = staging_dir(model, fold, precision)
    exported = export_dir(model, fold, precision)
    if out.resolve() == exported.resolve() or out.resolve() in exported.resolve().parents:
        raise SystemExit(f"Refusing to stage inside the export directory: {out}")
    expected = (
        "firmware_staging",
        model,
        f"fold{fold:02d}",
        precision,
    )
    relative = out.resolve().relative_to((ROOT / "artifacts" / "generated").resolve())
    if relative.parts != expected:
        raise SystemExit(f"Staging directory is not a single-backend leaf: {out}")
    refuse_result_output(out)
    tflite = tflite_path(model, fold, precision)
    header_src = header_dst = header_when = None
    if model == "fbtcnn":
        header_src = fbtcnn_header_source(fold)
        header_dst = exported / FBTCNN_HEADER_NAME
        header_when = "before"
    return Step(
        label=f"stage {model} fold {fold:02d} {precision}",
        argv=(
            sys.executable,
            _script("generate_firmware_artifacts.py"),
            "--backend",
            f"{model}-{precision}",
            "--fold",
            str(fold),
            "--artifact-set",
            "generated",
            "--tflite",
            tflite.as_posix(),
            "--output",
            out.as_posix(),
        ),
        requires=(tflite, manifest_path(fold)),
        header_src=header_src,
        header_dst=header_dst,
        header_when=header_when,
        output_dir=out,
        phase="stage",
        model=model,
        fold=fold,
        precision=precision,
    )


def resolve_selection(
    action: str,
    *,
    all_models: bool,
    model: str | None,
    fold: int | None,
    precision: str | None,
) -> tuple[list[str], list[int], str | None, list[str]]:
    if all_models and (model or fold is not None):
        raise SystemExit("--all cannot be combined with --model or --fold")
    if not all_models and not model:
        raise SystemExit("pass --all or --model")
    if fold is not None and fold not in FOLDS:
        raise SystemExit("--fold must be in 1..10")
    if action == "stage" and not all_models and precision is None:
        raise SystemExit("--precision is required unless --all is set")
    if action == "export" and model in PYTORCH_ONLY_MODELS:
        raise capability_error(model)
    if action == "stage" and model in PYTORCH_ONLY_MODELS:
        raise capability_error(model)
    if (
        action == "stage"
        and model
        and precision
        and precision not in EXPORT_PRECISIONS[model]
    ):
        raise SystemExit(
            f"Unsupported precision {precision} for {model}. "
            f"Supported: {', '.join(EXPORT_PRECISIONS[model])}"
        )

    models = list(NEURAL_MODELS) if all_models else [model]
    notes: list[str] = []
    if action in {"export", "stage"} and all_models:
        kept: list[str] = []
        for name in models:
            if name in PYTORCH_ONLY_MODELS:
                notes.append(pytorch_only_note(name))
                continue
            if (
                action == "stage"
                and precision is not None
                and precision not in EXPORT_PRECISIONS[name]
            ):
                notes.append(
                    f"{name} {precision}: not a supported firmware precision"
                )
                continue
            kept.append(name)
        models = kept
    elif action == "evaluate" and (
        all_models or model in PYTORCH_ONLY_MODELS
    ):
        for name in models:
            if name in PYTORCH_ONLY_MODELS:
                notes.append(pytorch_only_note(name))
    folds = list(FOLDS) if fold is None else [fold]
    return models, folds, precision, notes


def build_steps(
    action: str,
    models: list[str],
    folds: list[int],
    precision: str | None,
) -> list[Step]:
    steps: list[Step] = []
    for model in models:
        for fold in folds:
            if action == "train":
                steps.append(train_step(model, fold))
            elif action == "export":
                for quant in _precisions(model, precision):
                    steps.append(export_step(model, fold, quant))
            elif action == "evaluate":
                steps.append(_eval_step(model, fold, "pytorch", None))
                if model in EXPORT_PRECISIONS:
                    for quant in _precisions(model, precision):
                        steps.append(_eval_step(model, fold, "tflite", quant))
                        steps.append(_eval_step(model, fold, "stage1", quant))
            elif action == "stage":
                for quant in _precisions(model, precision):
                    steps.append(stage_step(model, fold, quant))
            else:
                raise SystemExit(f"Unknown action: {action}")
    _refuse_nested_staging(steps)
    return steps


def classical_output_dir() -> Path:
    return ROOT / "outputs" / "host" / "classical"


def classical_step() -> Step:
    """One LOSO pass for CCA, FBCCA, MSI, and published TRCA deployed_sos."""
    out = classical_output_dir()
    refuse_result_output(out)
    return Step(
        label="evaluate classical LOSO (cca, fbcca, msi, trca deployed_sos)",
        argv=(
            sys.executable,
            _script("evaluate_host.py"),
            "--runtime",
            "classical",
            "--method",
            "all",
            "--trca-realization",
            "deployed_sos",
            "--npz",
            PROCESSED_NPZ.as_posix(),
            "--output-dir",
            out.as_posix(),
        ),
        requires=(PROCESSED_NPZ,),
        output_dir=out,
        phase="evaluate",
    )


def classical_reference_stage_step(method: str, fold: int) -> Step:
    """Copy one retained classical bundle into its own staging directory."""
    if method not in CLASSICAL_METHODS:
        raise SystemExit(f"Unsupported classical backend: {method}")
    if fold not in FOLDS:
        raise SystemExit("--fold must be in 1..10")
    out = reference_staging_dir(method, fold)
    refuse_result_output(out)
    reference_root = (ROOT / "artifacts" / "reference").resolve()
    resolved = out.resolve()
    if resolved == reference_root or reference_root in resolved.parents:
        raise SystemExit(f"Refusing to stage into reference artifacts: {resolved}")
    relative = resolved.relative_to((ROOT / "artifacts" / "generated").resolve())
    expected = ("firmware_staging", method, f"fold{fold:02d}", "reference")
    if relative.parts != expected:
        raise SystemExit(f"Classical staging directory is not an isolated leaf: {out}")
    return Step(
        label=f"stage {method} fold {fold:02d} reference",
        argv=(
            sys.executable,
            _script("generate_firmware_artifacts.py"),
            "--backend",
            method,
            "--fold",
            str(fold),
            "--artifact-set",
            "reference",
            "--output",
            out.as_posix(),
        ),
        output_dir=out,
        phase="stage",
        model=method,
        fold=fold,
        precision="reference",
    )


def classical_reference_stage_steps(folds: list[int]) -> list[Step]:
    return [
        classical_reference_stage_step(method, fold)
        for method in CLASSICAL_METHODS
        for fold in folds
    ]


def with_classical_reference_staging(steps: list[Step], folds: list[int]) -> list[Step]:
    combined = [*steps, *classical_reference_stage_steps(folds)]
    _refuse_nested_staging(combined)
    return combined


def build_workflow(
    models: list[str],
    folds: list[int],
    *,
    include_classical: bool = False,
) -> tuple[list[Step], list[str]]:
    """Train, export supported TFLite, evaluate, then stage.

    Classical LOSO evaluation is part of the evaluation phase. Reference
    classical staging follows the neural staging commands.
    """
    notes = [pytorch_only_note(name) for name in models if name in PYTORCH_ONLY_MODELS]
    export_models = [name for name in models if name in EXPORT_PRECISIONS]
    steps: list[Step] = []
    steps.extend(build_steps("train", models, folds, None))
    if export_models:
        steps.extend(build_steps("export", export_models, folds, None))
    steps.extend(build_steps("evaluate", models, folds, None))
    if include_classical:
        steps.append(classical_step())
    if export_models:
        steps.extend(build_steps("stage", export_models, folds, None))
    if include_classical:
        steps = with_classical_reference_staging(steps, folds)
    return steps, notes


def _refuse_nested_staging(steps: list[Step]) -> None:
    staged = [step.output_dir.resolve() for step in steps if step.output_dir and "firmware_staging" in step.output_dir.parts]
    for index, left in enumerate(staged):
        for right in staged[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise SystemExit(
                    "Refusing to stage multiple firmware backends in one directory: "
                    f"{left} and {right}"
                )


def _copy_header(step: Step) -> int | None:
    if step.header_src is None or step.header_dst is None:
        return None
    if not step.header_src.is_file():
        print(f"FAILED: FBtCNN filter header not found: {step.header_src}", file=sys.stderr)
        print(
            "src/export/gen_fbtcnn_filter_header.py writes only to "
            f"{FBTCNN_GENERATOR_OUTPUT.relative_to(ROOT).as_posix()} and cannot "
            "target the export directory. The published header is missing, so "
            "this run stopped without changing that generator.",
            file=sys.stderr,
        )
        print("Stopped. Later models and folds were not started.", file=sys.stderr)
        return 1
    step.header_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(step.header_src, step.header_dst)
    print(f"  copied {_display(step.header_src)} -> {_display(step.header_dst)}")
    return None


def _load_training_summaries():
    import importlib.util

    path = Path(__file__).resolve().parent / "training_summaries.py"
    spec = importlib.util.spec_from_file_location("training_summaries", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maybe_aggregate(model: str) -> None:
    try:
        written = _load_training_summaries().try_write_complete_aggregate(RUNS_ROOT, model)
    except ValueError as exc:
        print(f"  training aggregate not written: {exc}", file=sys.stderr)
        return
    if written is not None:
        print(f"  training aggregate: {written}")


def _print_step(index: int, total: int, step: Step) -> None:
    print(f"[{index}/{total}] {step.label}")
    if step.argv:
        print(f"  {_fmt(step.argv)}")
    if step.header_src is not None and step.header_dst is not None:
        print(
            "  header: "
            f"{_display(step.header_src)} -> "
            f"{_display(step.header_dst)} ({step.header_when} the command)"
        )


def _run_step(step: Step, log_path: Path | None) -> tuple[int, str | None]:
    """Run one step. Returns an exit code and an optional failure note."""
    missing = [path for path in step.requires if not path.is_file()]
    if missing:
        note = "\n".join(f"FAILED: required file not found: {path}" for path in missing)
        print(note, file=sys.stderr)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(note + "\n", encoding="utf-8")
        return 1, note
    if step.header_when == "before":
        copied = _copy_header(step)
        if copied:
            note = "FBtCNN filter header copy failed"
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(note + "\n", encoding="utf-8")
            return copied, note
    code = 0
    if step.argv:
        if log_path is None:
            code = subprocess.call(step.argv, cwd=str(ROOT))
        else:
            from pipeline_log import run_logged_process

            code = run_logged_process(list(step.argv), ROOT, log_path)
        if code != 0:
            print(f"FAILED ({code}): {_fmt(step.argv)}", file=sys.stderr)
            return code, None
    if step.header_when == "after":
        produced = tflite_from_step(step)
        if produced is not None and not produced.is_file():
            note = f"FAILED: export did not write {produced}"
            print(note, file=sys.stderr)
            if log_path is not None:
                with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(note + "\n")
            return 1, note
        copied = _copy_header(step)
        if copied:
            note = "FBtCNN filter header copy failed"
            if log_path is not None:
                with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(note + "\n")
            return copied, note
    return 0, None


def _print_failure(index: int, total: int, step: Step, code: int, logger) -> None:
    print(f"FAILED at step {index}/{total}", file=sys.stderr)
    if step.argv:
        print(f"Command: {_fmt(step.argv)}", file=sys.stderr)
    print(f"Exit code: {code}", file=sys.stderr)
    if logger is not None:
        relative = logger.summary.get("failed_log")
        if relative:
            print(f"Failed log:\n{(logger.directory / relative).resolve()}", file=sys.stderr)
        print(f"Summary: {logger.summary_path.resolve()}", file=sys.stderr)
        print(f"Completed: {logger.summary['completed_steps']}", file=sys.stderr)
        print(f"Remaining: {logger.summary['remaining_steps']}", file=sys.stderr)
    print("Pipeline status: FAILED", file=sys.stderr)
    print("Stopped. Later models and folds were not started.", file=sys.stderr)


def execute(steps: list[Step], *, dry_run: bool, logger=None) -> int:
    total = len(steps)
    if dry_run:
        for index, step in enumerate(steps, start=1):
            _print_step(index, total, step)
        return 0
    try:
        for index, step in enumerate(steps, start=1):
            _print_step(index, total, step)
            log_path = logger.begin_step(index, step) if logger is not None else None
            try:
                code, note = _run_step(step, log_path)
                if code != 0:
                    if logger is not None:
                        logger.finish_step(index, step, "FAILED", code, note=note)
                    _print_failure(index, total, step, code, logger)
                    return code
                if step.phase == "train" and step.model:
                    _maybe_aggregate(step.model)
            except KeyboardInterrupt:
                if logger is not None:
                    logger.finish_step(index, step, "INTERRUPTED", 130, note="Interrupted")
                print("Pipeline status: INTERRUPTED", file=sys.stderr)
                return 130
            except OSError as exc:
                note = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                try:
                    if logger is not None:
                        logger.finish_step(index, step, "FAILED", 1, note=note)
                    _print_failure(index, total, step, 1, logger)
                except Exception:
                    pass
                return 1
            except Exception:
                note = traceback.format_exc()
                try:
                    if logger is not None:
                        logger.finish_step(index, step, "FAILED", 1, note=note)
                    _print_failure(index, total, step, 1, logger)
                except Exception:
                    pass
                raise
            if logger is not None:
                logger.finish_step(index, step, "COMPLETED", 0)
        if logger is not None:
            logger.complete()
        print("Pipeline status: COMPLETED")
        print(f"Successful commands: {total}/{total}")
        print("Failed commands: 0")
        if logger is not None:
            print(f"Summary: {logger.summary_path}")
        return 0
    except KeyboardInterrupt:
        if logger is not None and logger.summary["status"] == "RUNNING":
            from pipeline_log import utc_now

            logger.summary["status"] = "INTERRUPTED"
            logger.summary["ended_at"] = utc_now()
            logger.write_summary()
        print("Pipeline status: INTERRUPTED", file=sys.stderr)
        return 130


def tflite_from_step(step: Step) -> Path | None:
    if "--output" not in step.argv or "--quantization" not in step.argv:
        return None
    output = Path(step.argv[step.argv.index("--output") + 1])
    quant = step.argv[step.argv.index("--quantization") + 1]
    model = step.argv[step.argv.index("--model") + 1]
    name = f"{model}_float32.tflite" if quant == "f32" else f"{model}_int8.tflite"
    return output / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training, TFLite export, host evaluation, firmware staging, or all four."
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("run", "Train, export, evaluate, then stage supported firmware backends"),
        ("train", "Train neural models with scripts/train_neural.py"),
        ("export", "Export TFLite with scripts/export_tflite.py"),
        ("evaluate", "Evaluate PyTorch, TFLite, and Stage 1 with scripts/evaluate_host.py"),
        ("stage", "Stage one firmware backend with scripts/generate_firmware_artifacts.py"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--all", action="store_true", help="All six neural models and ten LOSO folds")
        command.add_argument("--model", choices=NEURAL_MODELS)
        command.add_argument("--fold", type=int, help="LOSO fold 1..10 (held-out subject sN)")
        command.add_argument("--dry-run", action="store_true", help="Print planned commands and exit")
        if name == "stage":
            command.add_argument("--precision", choices=["f32", "int8"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models, folds, precision, notes = resolve_selection(
        args.action,
        all_models=args.all,
        model=args.model,
        fold=args.fold,
        precision=getattr(args, "precision", None),
    )
    if args.action == "run":
        steps, run_notes = build_workflow(
            models, folds, include_classical=args.all
        )
        notes.extend(run_notes)
    elif args.action == "evaluate":
        steps = build_steps(args.action, models, folds, precision)
        if args.all:
            steps.append(classical_step())
    elif args.action == "stage":
        steps = build_steps(args.action, models, folds, precision)
        if args.all and precision is None:
            steps = with_classical_reference_staging(steps, folds)
    else:
        steps = build_steps(args.action, models, folds, precision)
    for note in notes:
        print(f"NOTE: {note}")
    if any(
        step.argv and "--quantization" in step.argv and "int8" in step.argv
        for step in steps
    ):
        print(
            "INT8 uses the existing PTQ calibration path in scripts/export_tflite.py. "
            "It does not train a separate model."
        )
    stages_firmware = any(
        step.argv and step.argv[1].endswith("generate_firmware_artifacts.py") for step in steps
    )
    if args.action == "stage" or stages_firmware:
        print("Each staging directory receives one firmware backend.")
    if any(step.header_src is not None for step in steps):
        print(
            "FBtCNN filter header: copy the published subject-independent header "
            "beside the TFLite file. "
            f"{FBTCNN_GENERATOR.relative_to(ROOT).as_posix()} writes only to "
            f"{FBTCNN_GENERATOR_OUTPUT.relative_to(ROOT).as_posix()} and is not retargeted."
        )
    if not steps:
        raise SystemExit("No commands to run.")
    if args.dry_run:
        print(f"Dry run: {len(steps)} command(s). Nothing will be executed.")
        return execute(steps, dry_run=True)
    from pipeline_log import RunLog

    logger = RunLog.create(
        ROOT / "outputs" / "pipeline_runs",
        ROOT,
        args.action,
        list(argv) if argv is not None else sys.argv[1:],
        len(steps),
    )
    return execute(steps, dry_run=False, logger=logger)


if __name__ == "__main__":
    raise SystemExit(main())
