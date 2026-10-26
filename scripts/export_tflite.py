#!/usr/bin/env python3
"""Public export entry. Default output under artifacts/generated/.

Routes to the canonical per-model conversion path (``src/export/convert_all_tflite.py``
for float32, ``src/export/quantize_int8.py`` for int8 PTQ) and writes only inside
``--output``. Requires the pinned conversion environment in
``environment/tflite_conversion_py310.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONUTF8", "1")
# torch.onnx progress messages contain non-cp1252 characters on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.neural import MODEL_REGISTRY

PROTECTED_TREES = (
    ROOT / "artifacts" / "archived_readonly",
    ROOT / "artifacts" / "reference",
)
SUPPORTED = {
    name: quant
    for name, quant in {
        "ccnn": ("f32", "int8"),
        "convca": ("f32",),
        "eegnet": ("f32", "int8"),
        "fbtcnn": ("f32", "int8"),
        "ssvepformer": ("f32",),
        "ssvepnet": ("f32",),
    }.items()
    if name in MODEL_REGISTRY
}
# Float32 export of these models needs no dataset: weights come from the checkpoint.
# ConvCA needs fold templates (not stored in the retained checkpoints) and int8 needs
# calibration trials, both drawn from the processed dataset.
F32_NEEDS_DATASET = {"convca"}


def refuse_protected_overwrite(path: Path) -> None:
    path = path.resolve()
    for protected in PROTECTED_TREES:
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            raise SystemExit(f"Refusing to write inside protected artifact tree: {path}")


def resolve_processed_npz(dataset_root: Path | None) -> Path | None:
    """Same resolution rule as scripts/evaluate_host.py."""
    if dataset_root is None:
        return None
    if dataset_root.is_file() and dataset_root.suffix == ".npz":
        return dataset_root
    for cand in (
        dataset_root / "processed.npz",
        dataset_root / "processed" / "12JFPM" / "processed.npz",
    ):
        if cand.is_file():
            return cand
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _io_shapes(tflite_path: Path) -> dict:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return {
        "tflite_input_shape": [int(x) for x in inp["shape"]],
        "tflite_input_dtype": inp["dtype"].__name__,
        "tflite_output_shape": [int(x) for x in out["shape"]],
        "tflite_output_dtype": out["dtype"].__name__,
    }


def export_model(
    model: str,
    fold: int,
    checkpoint: Path,
    output: Path,
    quantization: str = "f32",
    npz_path: Path | None = None,
) -> dict:
    """Export one fold of one model. Returns the export report (also written to disk)."""
    import numpy as np

    import export.convert_all_tflite as cat
    from export.common import get_fold_test_vectors, load_processed_npz
    from export.subject_paths import PROTOCOL_STRICT_LOSO_INNER_VAL

    refuse_protected_overwrite(output)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    held_out = f"s{fold}"

    npz = load_processed_npz(npz_path) if npz_path is not None else None
    if npz is None and (quantization == "int8" or model in F32_NEEDS_DATASET):
        raise SystemExit(
            f"{model} {quantization} export needs the processed dataset "
            "(pass --dataset-root pointing at processed.npz or its folder)."
        )

    cat.ACTIVE_FOLD = fold
    cat.ACTIVE_VAL_SUBJECT = held_out
    cat.ACTIVE_PROTOCOL = PROTOCOL_STRICT_LOSO_INNER_VAL
    cat._ckpt_for = lambda _name: checkpoint
    if npz is not None:
        cat.load_processed_npz = lambda *_a, **_k: npz
    vectors = get_fold_test_vectors(npz, val_subject=held_out) if npz is not None else []

    work = output / "_work"
    work.mkdir(parents=True, exist_ok=True)
    info = cat.HANDLERS[model](vectors, work)
    if info["status"] != "onnx_ok":
        raise SystemExit(f"ONNX export failed for {model}: {info['status']}")

    tf_dir = work / f"{model}_tf"
    conv = cat._run_onnx2tf(
        info["onnx_path"], tf_dir, timeout=900, extra_args=info.get("onnx2tf_extra_args")
    )
    f32_src = cat._find_tflite_files(tf_dir)["float32"]
    if not conv["success"] or f32_src is None:
        raise SystemExit(f"onnx2tf conversion failed for {model}: {conv.get('error', '')[:300]}")

    report = {
        "model": model,
        "fold": fold,
        "held_out_subject": held_out,
        "protocol": PROTOCOL_STRICT_LOSO_INNER_VAL,
        "quantization": quantization,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "preprocessing_note": info["preprocessing_note"],
    }

    if quantization == "f32":
        dst = output / f"{model}_float32.tflite"
        shutil.copy(f32_src, dst)
    else:
        import export.quantize_int8 as q

        q.ACTIVE_FOLD = fold
        q.ACTIVE_VAL_SUBJECT = held_out
        q.ACTIVE_PROTOCOL = PROTOCOL_STRICT_LOSO_INNER_VAL
        calib_raw, calib_subjects, per_class = q._load_calibration_raw(npz)
        calib = q.CALIB_FN[model](calib_raw)
        dst = output / f"{model}_int8.tflite"
        result = q._quantize_model(tf_dir, calib, dst)
        if result.get("method") != "full_int8_float_io":
            if dst.exists():
                dst.unlink()
            raise SystemExit(
                f"Full-integer PTQ failed for {model}; refusing dynamic-range fallback: {result}"
            )
        report.update(
            calibration_n=int(len(calib)),
            calibration_subjects=calib_subjects,
            calibration_per_class=per_class,
            quantization_method=result["method"],
        )

    report["tflite"] = dst.name
    report["tflite_sha256"] = _sha256(dst)
    report["tflite_bytes"] = dst.stat().st_size
    report.update(_io_shapes(dst))
    if vectors:
        pre = info.get("preprocess")
        x = pre(vectors[0]) if pre is not None else vectors[0][np.newaxis]
        report["first_vector_tflite_argmax"] = int(
            np.argmax(cat._run_tflite_inference(dst, x.astype(np.float32)))
        )
    (output / f"{model}_{quantization}_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    shutil.rmtree(work, ignore_errors=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SUPPORTED))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="processed.npz, or a folder containing it. Required for ConvCA and int8.",
    )
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--quantization", choices=["f32", "int8"], default="f32")
    ap.add_argument("--output", type=Path, default=Path("artifacts/generated"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without exporting weights",
    )
    args = ap.parse_args()
    refuse_protected_overwrite(args.output)

    if args.quantization not in SUPPORTED[args.model]:
        raise SystemExit(
            f"Unsupported quantization {args.quantization} for model {args.model}. "
            f"Allowed: {', '.join(SUPPORTED[args.model])}"
        )
    if args.fold < 1 or args.fold > 10:
        raise SystemExit("--fold must be in 1..10")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.manifest is not None:
        if not args.manifest.is_file():
            raise SystemExit(f"Manifest not found: {args.manifest}")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if int(manifest.get("fold_index", -1)) != args.fold:
            raise SystemExit(
                f"Manifest fold_index {manifest.get('fold_index')} does not match --fold {args.fold}"
            )
    if args.dataset_root is not None and not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")
    npz_path = resolve_processed_npz(args.dataset_root)
    if args.dataset_root is not None and npz_path is None:
        raise SystemExit(
            f"No processed.npz under {args.dataset_root}; run scripts/prepare_12jfpm.py first."
        )

    out = Path(args.output)
    if args.dry_run:
        ctor = MODEL_REGISTRY[args.model]
        print("dry-run ok")
        print("constructor:", ctor.__module__ + "." + ctor.__name__)
        print("model:", args.model)
        print("fold:", args.fold)
        print("checkpoint:", args.checkpoint)
        print("quantization:", args.quantization)
        print("output:", out)
        return

    report = export_model(
        args.model, args.fold, args.checkpoint.resolve(), out, args.quantization, npz_path
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
