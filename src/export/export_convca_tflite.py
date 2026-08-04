#!/usr/bin/env python3
"""
Export ConvCA with fold-specific baked templates for ESP32-S3.

ConvCA is a two-input model (trial + templates). For deployment,
class templates are computed from training data and baked into the
model graph as constants. This produces a standard single-input
model suitable for TFLite export.

Usage:
  python -m export.export_convca_tflite [--out export/12jfpm_convca_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export.common import (
    CANONICAL_VAL_SUBJECT,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.neural.convca import ConvCA

_DEFAULT_OUT = "export/12jfpm_convca_fold01_val_s1"


class ConvCABaked(nn.Module):
    """ConvCA with templates baked as buffer. Single-input for ONNX/TFLite."""

    def __init__(self, base: ConvCA, templates: torch.Tensor):
        super().__init__()
        self.sig_conv1 = base.sig_conv1
        self.sig_conv2 = base.sig_conv2
        self.sig_conv3 = base.sig_conv3
        self.sig_dropout = base.sig_dropout
        self.temp_conv1 = base.temp_conv1
        self.temp_conv2 = base.temp_conv2
        self.temp_dropout = base.temp_dropout
        self.dense = base.dense
        self.register_buffer("templates", templates)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        B = x_btc.shape[0]
        sig = x_btc.unsqueeze(1)
        temp = self.templates.permute(2, 1, 0).unsqueeze(0).expand(B, -1, -1, -1)

        sig_enc = self.sig_conv3(self.sig_conv2(self.sig_conv1(sig)))
        sig_enc = self.sig_dropout(sig_enc)
        temp_enc = self.temp_conv2(self.temp_conv1(temp))
        temp_enc = self.temp_dropout(temp_enc)

        x = sig_enc.squeeze(1)
        t = temp_enc.squeeze(1)
        corr_xt = torch.einsum("btk,btl->blk", x, t)
        corr_xx = torch.einsum("btk,btk->bk", x, x)
        corr_tt = torch.einsum("btl,btl->bl", t, t)
        corr = corr_xt.squeeze(-1) / (torch.sqrt(corr_tt) * torch.sqrt(corr_xx) + 1e-8)
        return self.dense(corr)


def _compute_templates(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                       val_subject: str, n_classes: int) -> torch.Tensor:
    """Compute per-class mean templates from training data."""
    mask_val = np.array([str(s) == val_subject for s in subjects])
    X_train = X[~mask_val]
    y_train = y[~mask_val]
    templates = []
    for cls in range(n_classes):
        cls_data = X_train[y_train == cls]
        templates.append(cls_data.mean(axis=0))
    return torch.from_numpy(np.stack(templates)).float()


def _load_checkpoint(ckpt_path: Path) -> ConvCA:
    model = ConvCA(seq_len=512, embed_dim=8, num_classes=12)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _try_onnx_export(baked: ConvCABaked, out_dir: Path) -> dict:
    status = {"onnx": "not_attempted", "errors": []}
    baked.eval()
    dummy = torch.randn(1, 512, 8)
    p = out_dir / "convca_baked.onnx"
    try:
        torch.onnx.export(baked, dummy, str(p),
                          input_names=["input"], output_names=["logits"], opset_version=13)
        status["onnx"] = f"success ({p.stat().st_size} bytes)"
    except Exception as e:
        status["onnx"] = f"failed: {e}"
        status["errors"].append(str(e))
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ConvCA for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ConvCA export -> {out_dir}")

    ckpt_path = args.checkpoint or (
        repo_root() / "checkpoints" / "convca_loso_12jfpm" / "fold01_best.pt")

    npz = load_processed_npz(args.processed_npz)
    X, y = npz["X"], npz["y"]
    subjects = npz.get("subjects", npz.get("subject"))

    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found: {ckpt_path}")
        model = ConvCA(seq_len=512, embed_dim=8, num_classes=12)
        torch.manual_seed(0)
        model.eval()
        ckpt_note = "random_init"
    else:
        model = _load_checkpoint(ckpt_path)
        ckpt_note = str(ckpt_path)

    templates = _compute_templates(X, y, subjects, CANONICAL_VAL_SUBJECT, 12)
    model.set_templates(templates)
    print(f"  Templates shape: {templates.shape}")

    baked = ConvCABaked(model, templates)
    baked.eval()

    sd = model.state_dict()
    arrays = {}
    for name, param in sorted(sd.items()):
        if "num_batches_tracked" in name:
            continue
        arrays[name] = param.detach().cpu().float().numpy()
    np.savez_compressed(out_dir / "convca_weights.npz", **arrays)
    np.save(out_dir / "convca_templates.npy", templates.numpy())

    onnx_status = _try_onnx_export(baked, out_dir)
    print(f"  ONNX baked: {onnx_status['onnx']}")

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
        "float32": "pending_baked_conversion",
        "int8": "pending_baked_conversion",
        "onnx_baked": onnx_status["onnx"],
        "notes": "Templates baked as constants; einsum may need TFLite op mapping",
    }, indent=2), encoding="utf-8")

    write_metadata(out_dir, "convca", {
        "fold_specific_templates": True,
        "templates_shape": list(templates.shape),
        "checkpoint": ckpt_note,
    })
    print("ConvCA export complete.")


if __name__ == "__main__":
    main()
