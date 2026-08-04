#!/usr/bin/env python3
"""
Export SSVEPformer to flat binary weights + ONNX (post-FFT submodel).

SSVEPformer uses FFT preprocessing like CCNN. The post-FFT submodel
takes [B, C=8, 560] spectrum features as input.

Usage:
  python -m export.export_ssvepformer_tflite [--out export/12jfpm_ssvepformer_fold01_val_s1]
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
from src.neural.ssvepformer import SSVEPformer, _ssvepformer_spectrum

_DEFAULT_OUT = "export/12jfpm_ssvepformer_fold01_val_s1"


class SSVEPformerPostFFT(nn.Module):
    """Post-FFT submodel. Input: [B, C=8, 560] spectrum features."""

    def __init__(self, full: SSVEPformer):
        super().__init__()
        self.to_patch_embedding = full.to_patch_embedding
        self.transformer = full.transformer
        self.mlp_head = full.mlp_head

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(spec)
        for attn_block, ff_block in self.transformer:
            x = attn_block(x) + x
            x = ff_block(x) + x
        return self.mlp_head(x)


def _load_checkpoint(ckpt_path: Path) -> SSVEPformer:
    model = SSVEPformer(seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _export_flat_weights(model: SSVEPformer, out_dir: Path, ckpt_note: str) -> dict:
    sd = model.state_dict()
    layout = []
    offset = 0
    blob_parts = []
    arrays = {}
    for name, param in sorted(sd.items()):
        if "num_batches_tracked" in name:
            continue
        arr = param.detach().cpu().float().numpy()
        arrays[name] = arr
        flat = arr.ravel(order="C")
        nbytes = flat.nbytes
        layout.append({
            "name": name, "dtype": "float32",
            "shape": list(arr.shape), "offset_bytes": offset, "numel": int(arr.size),
        })
        blob_parts.append(flat.tobytes())
        offset += nbytes

    (out_dir / "ssvepformer_weights_flat.bin").write_bytes(b"".join(blob_parts))
    (out_dir / "ssvepformer_flat_layout.json").write_text(
        json.dumps({"tensors": layout, "total_bytes": offset}, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "ssvepformer_weights.npz", **arrays)

    meta = {
        "model": "SSVEPformer", "seq_len": 512, "embed_dim": 8, "num_classes": 12,
        "sampling_rate_hz": 256.0,
        "fft_note": "FFT preprocessing outside TFLite; post-FFT input is [1, 8, 560]",
        "checkpoint_note": ckpt_note,
    }
    (out_dir / "ssvepformer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"total_bytes": offset}


def _try_onnx_postfft(model: SSVEPformer, out_dir: Path) -> dict:
    status = {"onnx_postfft": "not_attempted", "errors": []}
    post = SSVEPformerPostFFT(model)
    post.eval()
    dummy = torch.randn(1, 8, 560)
    p = out_dir / "ssvepformer_postfft.onnx"
    try:
        torch.onnx.export(post, dummy, str(p),
                          input_names=["spectrum"], output_names=["logits"], opset_version=13)
        status["onnx_postfft"] = f"success ({p.stat().st_size} bytes)"
    except Exception as e:
        status["onnx_postfft"] = f"failed: {e}"
        status["errors"].append(str(e))
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SSVEPformer for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"SSVEPformer export -> {out_dir}")

    ckpt_path = args.checkpoint or (
        repo_root() / "checkpoints" / "ssvepformer_loso_12jfpm" / "fold01_best.pt")
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found: {ckpt_path}")
        model = SSVEPformer(seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0)
        torch.manual_seed(0)
        model.eval()
        ckpt_note = "random_init"
    else:
        model = _load_checkpoint(ckpt_path)
        ckpt_note = str(ckpt_path)

    info = _export_flat_weights(model, out_dir, ckpt_note)
    print(f"  -> flat weights ({info['total_bytes']} bytes)")

    onnx_status = _try_onnx_postfft(model, out_dir)
    print(f"  ONNX post-FFT: {onnx_status['onnx_postfft']}")

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

    (out_dir / "tflite_status.json").write_text(json.dumps({
        "float32": "pending_postfft_conversion",
        "int8": "pending_postfft_conversion",
        "onnx_postfft": onnx_status["onnx_postfft"],
        "notes": "Post-FFT submodel exportable to ONNX; TFLite conversion pending onnx2tf",
    }, indent=2), encoding="utf-8")

    write_metadata(out_dir, "ssvepformer", {
        "flat_weights_bytes": info["total_bytes"],
        "fft_preprocessing": "external (same approach as CCNN)",
        "checkpoint": ckpt_note,
    })
    print("SSVEPformer export complete.")


if __name__ == "__main__":
    main()
