#!/usr/bin/env python3
"""
Export CCNN to flat binary weights + TFLite (float32/int8) for ESP32-S3.

CCNN uses FFT preprocessing, which is not a TFLite op. The export
splits the model: FFT is done as preprocessing (on-device via esp-dsp),
and only the post-FFT CNN is exported to TFLite.

Usage:
  python -m export.export_ccnn_tflite [--out export/12jfpm_ccnn_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export.common import (
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.neural.ccnn import CCNN, _complex_spectrum_features

_DEFAULT_OUT = "export/12jfpm_ccnn_fold01_val_s1"

TENSOR_ORDER = [
    "layer1.0.weight", "layer1.0.bias",
    "layer1.1.weight", "layer1.1.bias", "layer1.1.running_mean", "layer1.1.running_var",
    "layer2.0.weight", "layer2.0.bias",
    "layer2.1.weight", "layer2.1.bias", "layer2.1.running_mean", "layer2.1.running_var",
    "classifier.1.weight", "classifier.1.bias",
]


class CCNNPostFFT(nn.Module):
    """Post-FFT submodel for TFLite export. Input: [B, 1, 8, 220] spectrum."""

    def __init__(self, full_model: CCNN):
        super().__init__()
        self.layer1 = full_model.layer1
        self.layer2 = full_model.layer2
        self.classifier = full_model.classifier

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.layer1(spec)
        x = self.layer2(x)
        return self.classifier(x)


def _load_checkpoint(ckpt_path: Path) -> CCNN:
    model = CCNN(seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.25)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _export_flat_weights(model: CCNN, out_dir: Path, ckpt_note: str) -> dict:
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

    (out_dir / "ccnn_weights_flat.bin").write_bytes(b"".join(blob_parts))
    (out_dir / "ccnn_flat_layout.json").write_text(
        json.dumps({"tensors": layout, "total_bytes": offset}, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "ccnn_weights.npz", **arrays)

    lines = [
        "/* Generated — do not edit. */",
        "#ifndef CCNN_LAYOUT_GENERATED_H",
        "#define CCNN_LAYOUT_GENERATED_H",
        "", f"#define CCNN_FLAT_TOTAL_BYTES ({offset}u)", "",
    ]
    for t in layout:
        safe = t["name"].replace(".", "_")
        lines.append(f"#define CCNN_OFF_{safe} ({t['offset_bytes']}u)")
    lines += ["", "#endif", ""]
    (out_dir / "ccnn_layout_generated.h").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "model": "CCNN", "seq_len": 512, "embed_dim": 8, "num_classes": 12,
        "sampling_rate_hz": 256.0, "nfc": 220,
        "input_tensor_layout": "[B, T, C] row-major float32 -> FFT -> [B, 1, 8, 220]",
        "checkpoint_note": ckpt_note, "flat_tensor_order": TENSOR_ORDER,
        "fft_note": "FFT preprocessing outside TFLite; post-FFT CNN input is [1, 1, 8, 220]",
    }
    (out_dir / "ccnn_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"total_bytes": offset}


def _try_tflite_export(model: CCNN, out_dir: Path) -> dict:
    status = {"float32": "not_attempted", "int8": "not_attempted", "errors": []}

    post_fft = CCNNPostFFT(model)
    post_fft.eval()

    onnx_path = out_dir / "ccnn_postfft.onnx"
    dummy = torch.randn(1, 1, 8, 220)
    try:
        torch.onnx.export(
            post_fft, dummy, str(onnx_path),
            input_names=["spectrum"], output_names=["logits"],
            dynamic_axes=None, opset_version=13,
        )
        status["onnx"] = "success"
    except Exception as e:
        status["onnx"] = f"failed: {e}"
        status["errors"].append(str(e))
        return status

    try:
        import onnx2tf  # noqa: F401
        import subprocess
        result = subprocess.run(
            ["python", "-m", "onnx2tf", "-i", str(onnx_path),
             "-o", str(out_dir / "ccnn_tf"), "-oiqt", "-ioqd", "uint8"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            import shutil
            for src_name, dst_name, key in [
                ("ccnn_postfft_float32.tflite", "ccnn_postfft_float32.tflite", "float32"),
                ("ccnn_postfft_full_integer_quant.tflite", "ccnn_postfft_int8.tflite", "int8"),
            ]:
                src = out_dir / "ccnn_tf" / src_name
                if src.exists():
                    shutil.copy(src, out_dir / dst_name)
                    status[key] = "success"
        else:
            status["errors"].append(f"onnx2tf failed: {result.stderr[:500]}")
    except ImportError:
        status["errors"].append("onnx2tf not installed; manual TFLite conversion needed")
    except Exception as e:
        status["errors"].append(f"TFLite conversion error: {e}")

    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Export CCNN for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--processed-npz", type=Path, default=None)
    ap.add_argument("--skip-tflite", action="store_true")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CCNN export -> {out_dir}")

    ckpt_path = args.checkpoint or (
        repo_root() / "checkpoints" / "ccnn_loso_12jfpm" / "fold01_best.pt")
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found: {ckpt_path}")
        model = CCNN(seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.25)
        torch.manual_seed(0)
        model.eval()
        ckpt_note = "random_init"
    else:
        model = _load_checkpoint(ckpt_path)
        ckpt_note = str(ckpt_path)

    info = _export_flat_weights(model, out_dir, ckpt_note)
    print(f"  -> flat weights ({info['total_bytes']} bytes)")

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
        tflite_status = _try_tflite_export(model, out_dir)
        print(f"  TFLite float32: {tflite_status.get('float32', 'unknown')}")
        print(f"  TFLite int8:    {tflite_status.get('int8', 'unknown')}")

    (out_dir / "tflite_status.json").write_text(
        json.dumps(tflite_status, indent=2), encoding="utf-8")

    write_metadata(out_dir, "ccnn", {
        "flat_weights_bytes": info["total_bytes"],
        "tflite_float32": tflite_status.get("float32", "unknown"),
        "tflite_int8": tflite_status.get("int8", "unknown"),
        "fft_preprocessing": "external (esp-dsp on device, numpy on host)",
        "tflite_input_shape": "[1, 1, 8, 220]",
        "checkpoint": ckpt_note,
    })
    print("CCNN export complete.")


if __name__ == "__main__":
    main()
