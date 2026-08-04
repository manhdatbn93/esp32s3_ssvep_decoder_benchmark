#!/usr/bin/env python3
"""
Export EEGNet to TFLite (float32 + int8) and flat binary weights.

Usage:
  python -m export.export_eegnet_tflite [--out export/12jfpm_eegnet_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from export.common import (
    CANONICAL_FOLD,
    CANONICAL_VAL_SUBJECT,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.neural.eegnet import EEGNet

_DEFAULT_OUT = "export/12jfpm_eegnet_fold01_val_s1"

TENSOR_ORDER = [
    "block1.0.weight",
    "block1.1.weight",
    "block1.1.bias",
    "block1.1.running_mean",
    "block1.1.running_var",
    "block2.0.weight",
    "block2.1.weight",
    "block2.1.bias",
    "block2.1.running_mean",
    "block2.1.running_var",
    "block3.0.weight",
    "block3.1.weight",
    "block3.2.weight",
    "block3.2.bias",
    "block3.2.running_mean",
    "block3.2.running_var",
    "fc.weight",
    "fc.bias",
]


def _load_checkpoint(ckpt_path: Path) -> EEGNet:
    model = EEGNet(
        seq_len=512, embed_dim=8, num_classes=12,
        F1=8, D=2, F2=16, dropout=0.25, kernel_length=64,
        pool1_stride=4, pool2_stride=8,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _export_flat_weights(model: EEGNet, out_dir: Path, ckpt_note: str) -> dict:
    sd = model.state_dict()
    layout = []
    offset = 0
    blob_parts = []
    arrays = {}
    for name in TENSOR_ORDER:
        arr = sd[name].detach().cpu().float().numpy()
        arrays[name] = arr
        flat = arr.ravel(order="C")
        nbytes = flat.nbytes
        layout.append({
            "name": name, "dtype": "float32",
            "shape": list(arr.shape), "offset_bytes": offset, "numel": int(arr.size),
        })
        blob_parts.append(flat.tobytes())
        offset += nbytes

    (out_dir / "eegnet_weights_flat.bin").write_bytes(b"".join(blob_parts))
    (out_dir / "eegnet_flat_layout.json").write_text(
        json.dumps({"tensors": layout, "total_bytes": offset}, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "eegnet_weights.npz", **arrays)

    lines = [
        "/* Generated — do not edit. */",
        "#ifndef EEGNET_LAYOUT_GENERATED_H",
        "#define EEGNET_LAYOUT_GENERATED_H",
        "", f"#define EEGNET_FLAT_TOTAL_BYTES ({offset}u)", "",
    ]
    for t in layout:
        safe = t["name"].replace(".", "_")
        lines.append(f"#define EEGNET_OFF_{safe} ({t['offset_bytes']}u)")
    lines += ["", "#endif", ""]
    (out_dir / "eegnet_layout_generated.h").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "model": "EEGNet", "seq_len": 512, "embed_dim": 8, "num_classes": 12,
        "F1": 8, "D": 2, "F2": 16,
        "input_tensor_layout": "[B, T, C] row-major float32, B=1, T=512, C=8",
        "checkpoint_note": ckpt_note, "flat_tensor_order": TENSOR_ORDER,
    }
    (out_dir / "eegnet_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"total_bytes": offset}


def _try_tflite_export(model: EEGNet, out_dir: Path, npz: dict) -> dict:
    """Attempt ONNX -> TFLite conversion. Returns status dict."""
    status = {"float32": "not_attempted", "int8": "not_attempted", "errors": []}

    try:
        import onnx  # noqa: F401
    except ImportError:
        status["errors"].append("onnx not installed")
        return status
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pass

    onnx_path = out_dir / "eegnet.onnx"
    dummy = torch.randn(1, 512, 8)
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes=None, opset_version=13,
        )
        status["onnx"] = "success"
    except Exception as e:
        status["onnx"] = f"failed: {e}"
        status["errors"].append(str(e))
        return status

    try:
        import onnx2tf  # noqa: F401
        has_onnx2tf = True
    except ImportError:
        has_onnx2tf = False

    try:
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False

    if has_onnx2tf:
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "onnx2tf", "-i", str(onnx_path),
                 "-o", str(out_dir / "eegnet_tf"), "-oiqt", "-ioqd", "uint8"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                tflite_f32 = out_dir / "eegnet_tf" / "eegnet_float32.tflite"
                tflite_i8 = out_dir / "eegnet_tf" / "eegnet_full_integer_quant.tflite"
                if tflite_f32.exists():
                    import shutil
                    shutil.copy(tflite_f32, out_dir / "eegnet_float32.tflite")
                    status["float32"] = "success"
                if tflite_i8.exists():
                    import shutil
                    shutil.copy(tflite_i8, out_dir / "eegnet_int8.tflite")
                    status["int8"] = "success"
            else:
                status["errors"].append(f"onnx2tf failed: {result.stderr[:500]}")
        except Exception as e:
            status["errors"].append(f"onnx2tf exception: {e}")
    elif has_tf:
        try:
            import tf2onnx  # noqa: F401
            status["errors"].append("tf2onnx available but onnx2tf preferred; skipping auto-convert")
        except ImportError:
            status["errors"].append("Neither onnx2tf nor tf2onnx installed; manual TFLite conversion needed")
    else:
        status["errors"].append("No TFLite conversion tools installed (need onnx2tf or tensorflow)")

    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EEGNet for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--processed-npz", type=Path, default=None)
    ap.add_argument("--skip-tflite", action="store_true")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"EEGNet export -> {out_dir}")

    ckpt_path = args.checkpoint or (
        repo_root() / "checkpoints" / "eegnet_loso_12jfpm" / "fold01_best.pt")
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found: {ckpt_path}")
        print("Creating random-init model for structure test only.")
        model = EEGNet(seq_len=512, embed_dim=8, num_classes=12,
                       F1=8, D=2, F2=16, dropout=0.25, kernel_length=64,
                       pool1_stride=4, pool2_stride=8)
        torch.manual_seed(0)
        model.eval()
        ckpt_note = "random_init — not a benchmark checkpoint"
    else:
        model = _load_checkpoint(ckpt_path)
        ckpt_note = str(ckpt_path)

    info = _export_flat_weights(model, out_dir, ckpt_note)
    print(f"  -> flat weights ({info['total_bytes']} bytes)")
    print("  -> eegnet_meta.json, eegnet_layout_generated.h")

    npz = load_processed_npz(args.processed_npz)
    vectors = get_fold_test_vectors(npz)
    save_test_vectors(vectors, out_dir)

    refs = {}
    for i, v in enumerate(vectors):
        with torch.no_grad():
            logits = model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten()
        refs[f"idx{i}"] = {
            "logits": [float(x) for x in logits],
            "argmax": int(np.argmax(logits)),
        }
    write_host_reference(out_dir, refs)
    print(f"  -> {len(vectors)} test vectors + host_reference.json")

    tflite_status = {"float32": "skipped", "int8": "skipped"}
    if not args.skip_tflite:
        tflite_status = _try_tflite_export(model, out_dir, npz)
        print(f"  TFLite float32: {tflite_status.get('float32', 'unknown')}")
        print(f"  TFLite int8:    {tflite_status.get('int8', 'unknown')}")
        if tflite_status.get("errors"):
            for e in tflite_status["errors"]:
                print(f"    WARNING: {e}")

    (out_dir / "tflite_status.json").write_text(
        json.dumps(tflite_status, indent=2), encoding="utf-8")

    write_metadata(out_dir, "eegnet", {
        "flat_weights_bytes": info["total_bytes"],
        "tflite_float32": tflite_status.get("float32", "unknown"),
        "tflite_int8": tflite_status.get("int8", "unknown"),
        "checkpoint": ckpt_note,
    })
    print("EEGNet export complete.")


if __name__ == "__main__":
    main()
