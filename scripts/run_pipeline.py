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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEURAL_MODELS = (
    "ccnn",
    "convca",
    "eegnet",
    "fbtcnn",
    "ssvepformer",
    "ssvepnet",
)
# Matches scripts/export_tflite.py SUPPORTED. INT8 is post-training quantization
# of the float checkpoint, not a separate training run.
EXPORT_PRECISIONS = {
    "ccnn": ("f32", "int8"),
    "convca": ("f32",),
    "eegnet": ("f32", "int8"),
    "fbtcnn": ("f32", "int8"),
    "ssvepformer": ("f32",),
    "ssvepnet": ("f32",),
}
# Firmware backends accepted by scripts/generate_firmware_artifacts.py.
# SSVEPformer and SSVEPNet stay host-side float32 exports.
FIRMWARE_MODELS = ("ccnn", "convca", "eegnet", "fbtcnn")
HOST_ONLY_MODELS = ("ssvepformer", "ssvepnet")
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


def _precisions(model: str, precision: str | None) -> tuple[str, ...]:
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
    )


def export_step(model: str, fold: int, precision: str) -> Step:
    out = export_dir(model, fold, precision)
    refuse_result_output(out)
    host_only = model in HOST_ONLY_MODELS
    kind = "host-side only, not firmware-deployable" if host_only else "PTQ of the existing checkpoint"
    if precision == "f32" and not host_only:
        kind = "float32 export"
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
    )


def stage_step(model: str, fold: int, precision: str) -> Step:
    if model in HOST_ONLY_MODELS:
        raise SystemExit(
            f"{model} is a host-side float32 export only and is not deployable on firmware"
        )
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
    if action == "stage" and model in HOST_ONLY_MODELS:
        raise SystemExit(
            f"{model} is a host-side float32 export only and is not deployable on firmware"
        )
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
    if action == "stage" and all_models:
        kept: list[str] = []
        for name in models:
            if name in HOST_ONLY_MODELS:
                notes.append(
                    f"{name}: host-side float32 export only, not deployable on firmware"
                )
                continue
            if precision is not None and precision not in EXPORT_PRECISIONS[name]:
                notes.append(
                    f"{name} {precision}: not a supported firmware precision"
                )
                continue
            kept.append(name)
        models = kept
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


def build_workflow(
    models: list[str],
    folds: list[int],
) -> tuple[list[Step], list[str]]:
    """Train, export, evaluate, then stage, using the existing step builders."""
    steps: list[Step] = []
    for action in ("train", "export", "evaluate"):
        steps.extend(build_steps(action, models, folds, None))
    notes: list[str] = []
    stage_models: list[str] = []
    for name in models:
        if name in HOST_ONLY_MODELS:
            notes.append(
                f"{name}: host-side float32 export only, not deployable on firmware"
            )
            continue
        stage_models.append(name)
    if stage_models:
        steps.extend(build_steps("stage", stage_models, folds, None))
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


def execute(steps: list[Step], *, dry_run: bool) -> int:
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.label}")
        if step.argv:
            print(f"  {_fmt(step.argv)}")
        if step.header_src is not None and step.header_dst is not None:
            print(
                "  header: "
                f"{_display(step.header_src)} -> "
                f"{_display(step.header_dst)} ({step.header_when} the command)"
            )
        if dry_run:
            continue
        missing = [path for path in step.requires if not path.is_file()]
        if missing:
            for path in missing:
                print(f"FAILED: required file not found: {path}", file=sys.stderr)
            print("Stopped. Later models and folds were not started.", file=sys.stderr)
            return 1
        if step.header_when == "before":
            copied = _copy_header(step)
            if copied:
                return copied
        if step.argv:
            code = subprocess.call(step.argv, cwd=str(ROOT))
            if code != 0:
                print(f"FAILED ({code}): {_fmt(step.argv)}", file=sys.stderr)
                print("Stopped. Later models and folds were not started.", file=sys.stderr)
                return code
        if step.header_when == "after":
            produced = tflite_from_step(step)
            if produced is not None and not produced.is_file():
                print(f"FAILED: export did not write {produced}", file=sys.stderr)
                print("Stopped. Later models and folds were not started.", file=sys.stderr)
                return 1
            copied = _copy_header(step)
            if copied:
                return copied
    return 0


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
        steps, run_notes = build_workflow(models, folds)
        notes.extend(run_notes)
    else:
        steps = build_steps(args.action, models, folds, precision)
    for note in notes:
        print(f"NOTE: {note}")
    if args.action in {"export", "run"}:
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
    return execute(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
