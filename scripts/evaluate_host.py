#!/usr/bin/env python3
"""Host evaluation for classical, PyTorch, and TFLite backends."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSICAL = ("cca", "fbcca", "msi", "trca")
NEURAL = ("ccnn", "convca", "eegnet", "fbtcnn", "ssvepformer", "ssvepnet")
REALIZED_NEURAL = ("ccnn", "convca", "eegnet", "fbtcnn")


def _classical(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.training_entry.run_classical_loso_12jfpm",
        "--method",
        args.method,
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.npz:
        cmd.extend(["--npz", str(args.npz)])
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.call(cmd, cwd=str(ROOT))


def _resolve_model(args: argparse.Namespace) -> str | None:
    if args.model:
        return args.model
    if args.backend:
        return args.backend.replace("-", "_").split("_")[0]
    return None


def _pytorch_dry_run(args: argparse.Namespace) -> int:
    from src.evaluation.neural_host import build_model, load_yaml, resolve_config
    from src.neural import MODEL_REGISTRY

    model = _resolve_model(args)
    models = [model] if model else list(NEURAL)
    for name in models:
        cfg_path = resolve_config(name, args.config if model else None)
        cfg = load_yaml(cfg_path)
        built = build_model(name, cfg)
        print(f"OK pytorch {name}: {cfg_path.relative_to(ROOT).as_posix()}")
        print(f"  constructor: {built.__class__.__module__}.{built.__class__.__name__}")
        print(f"  registry: {MODEL_REGISTRY[name].__name__}")
        if args.artifact_set == "reference":
            if name not in REALIZED_NEURAL:
                raise SystemExit(f"No retained reference checkpoint for {name}")
            if args.fold is None:
                raise SystemExit("--fold is required with --artifact-set reference")
            from src.artifacts.reference import resolve_checkpoint

            art = resolve_checkpoint(
                name, args.fold, verify_checksum=not args.skip_checksum
            )
            print(f"  reference: {art['path'].relative_to(ROOT).as_posix()}")
            print(f"  held_out: {art['held_out_subject']}")
    return 0


def _tflite_dry_run(args: argparse.Namespace) -> int:
    from src.evaluation.neural_host import TFLITE_MODELS, preprocess_trial_for_tflite
    import numpy as np

    model = _resolve_model(args)
    if args.artifact_set == "reference":
        models = [model] if model else list(REALIZED_NEURAL)
    else:
        models = [model] if model else sorted(TFLITE_MODELS)
    for name in models:
        if name not in TFLITE_MODELS:
            raise SystemExit(f"Unsupported TFLite model: {name}")
        dummy = np.zeros((512, 8), dtype=np.float32)
        shaped = preprocess_trial_for_tflite(name, dummy)
        print(f"OK tflite {name}: preprocess -> {tuple(shaped.shape)}")
        if args.artifact_set == "reference":
            if name not in REALIZED_NEURAL:
                raise SystemExit(f"No retained reference TFLite for {name}")
            if args.fold is None:
                raise SystemExit("--fold is required with --artifact-set reference")
            precision = args.precision or "f32"
            from src.artifacts.reference import resolve_tflite

            art = resolve_tflite(
                name, precision, args.fold, verify_checksum=not args.skip_checksum
            )
            print(f"  reference: {art['path'].relative_to(ROOT).as_posix()}")
            print(f"  held_out: {art['held_out_subject']}")
        elif args.tflite:
            print(f"  artifact: {args.tflite}")
            if not args.tflite.is_file():
                print(f"  warning: artifact not found yet ({args.tflite})")
    return 0


def _pytorch_run(args: argparse.Namespace) -> int:
    from src.evaluation.neural_host import evaluate_pytorch

    model = _resolve_model(args)
    if not model:
        raise SystemExit("--model or --backend is required for --runtime pytorch")
    checkpoint = args.checkpoint
    if args.artifact_set == "reference":
        if args.fold is None:
            raise SystemExit("--fold is required with --artifact-set reference")
        from src.artifacts.reference import resolve_checkpoint

        art = resolve_checkpoint(
            model, args.fold, verify_checksum=not args.skip_checksum
        )
        checkpoint = art["path"]
    if not checkpoint:
        raise SystemExit("--checkpoint is required unless --artifact-set reference")
    if not args.npz:
        raise SystemExit("--npz is required for --runtime pytorch")
    out = args.output_dir or Path("outputs/host/pytorch") / model
    summary = evaluate_pytorch(
        model,
        checkpoint=checkpoint,
        npz=args.npz,
        fold=args.fold,
        manifest=args.manifest,
        config=args.config,
        output_dir=out,
    )
    acc = summary.get("accuracy", summary.get("outer_test_accuracy"))
    print(
        f"pytorch {model}: accuracy={acc} "
        f"n_trials={summary.get('n_trials')} out={out}"
    )
    return 0


def _tflite_run(args: argparse.Namespace) -> int:
    from src.evaluation.neural_host import evaluate_tflite

    model = _resolve_model(args)
    if not model:
        raise SystemExit("--model or --backend is required for --runtime tflite")
    tflite = args.tflite
    if args.artifact_set == "reference":
        if args.fold is None:
            raise SystemExit("--fold is required with --artifact-set reference")
        precision = args.precision or "f32"
        from src.artifacts.reference import resolve_tflite

        art = resolve_tflite(
            model, precision, args.fold, verify_checksum=not args.skip_checksum
        )
        tflite = art["path"]
    if not tflite:
        raise SystemExit("--tflite is required unless --artifact-set reference")
    if not args.npz:
        raise SystemExit("--npz is required for --runtime tflite")
    out = args.output_dir or Path("outputs/host/tflite") / model
    summary = evaluate_tflite(
        model,
        tflite=tflite,
        npz=args.npz,
        fold=args.fold,
        manifest=args.manifest,
        output_dir=out,
    )
    print(
        f"tflite {model}: accuracy={summary['accuracy']} "
        f"n_trials={summary['n_trials']} out={out}"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Host evaluation for classical, PyTorch, and TFLite backends"
    )
    ap.add_argument(
        "--runtime",
        choices=["classical", "pytorch", "tflite"],
        default="classical",
        help="Execution backend family",
    )
    ap.add_argument(
        "--method",
        choices=[*CLASSICAL, "all"],
        default="all",
        help="Classical method when --runtime classical",
    )
    ap.add_argument(
        "--model",
        choices=list(NEURAL),
        default=None,
        help="Neural model when --runtime pytorch or tflite",
    )
    ap.add_argument(
        "--backend",
        default=None,
        help="Alias for model or classical/neural backend name",
    )
    ap.add_argument(
        "--precision",
        choices=["f32", "int8"],
        default=None,
        help="TFLite precision when using reference artifacts",
    )
    ap.add_argument(
        "--artifact-set",
        choices=["reference", "generated"],
        default="generated",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--tflite", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--local-outputs-only",
        action="store_true",
        help="Print local outputs/host/ path and exit",
    )
    args = ap.parse_args()
    if args.output and not args.output_dir:
        args.output_dir = args.output
    if args.dataset_root and not args.npz:
        cand = args.dataset_root
        if cand.is_file() and cand.suffix == ".npz":
            args.npz = cand
        else:
            for p in (
                cand / "processed.npz",
                cand / "processed" / "12JFPM" / "processed.npz",
            ):
                if p.is_file():
                    args.npz = p
                    break

    if args.local_outputs_only:
        print("Local host metrics: outputs/host/")
        return

    if args.runtime == "classical":
        raise SystemExit(_classical(args))
    if args.runtime == "pytorch":
        raise SystemExit(_pytorch_dry_run(args) if args.dry_run else _pytorch_run(args))
    if args.runtime == "tflite":
        raise SystemExit(_tflite_dry_run(args) if args.dry_run else _tflite_run(args))
    raise SystemExit(f"Unknown runtime: {args.runtime}")


if __name__ == "__main__":
    main()
