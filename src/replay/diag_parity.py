#!/usr/bin/env python3
"""
Diagnostic: compare host-side f32 vs i8 TFLite inference for EEGNet.

Runs BOTH float32 and int8 models on the host (no device needed)
using the same z-scored trial data. Reports:
  - input tensor shape/dtype expected by each model
  - first 5 values of the preprocessed input
  - full logit vectors
  - argmax
  - f32-vs-i8 difference (should be ~0.2)
  - effect of XNNPACK vs reference interpreter

Run from repo root:
  python tools/replay/diag_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datasets.parse_12jfpm import read_subject


def _preprocess_zscore(tc: np.ndarray) -> np.ndarray:
    x = tc.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return ((x - mean) / std).astype(np.float32)


def run_tflite(path: str, feed: np.ndarray, *, use_xnnpack: bool = True) -> np.ndarray:
    from src.replay.host_interpreter import (
        HOST_REFERENCE_OP_RESOLVER,
        NO_DEFAULT_DELEGATES_OP_RESOLVER,
        make_interpreter,
    )

    resolver = HOST_REFERENCE_OP_RESOLVER if use_xnnpack else NO_DEFAULT_DELEGATES_OP_RESOLVER
    interp = make_interpreter(path, resolver)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print(f"  Input  shape={list(inp['shape'])}  dtype={inp['dtype']}")
    print(f"  Output shape={list(out['shape'])}  dtype={out['dtype']}")
    if inp["quantization"] != (0.0, 0):
        print(f"  Input  quantization: scale={inp['quantization'][0]}, zp={inp['quantization'][1]}")
    if out["quantization"] != (0.0, 0):
        print(f"  Output quantization: scale={out['quantization'][0]}, zp={out['quantization'][1]}")

    expected = tuple(inp["shape"])
    if feed.shape != expected:
        print(f"  *** Shape mismatch: feed={feed.shape} vs expected={expected}, reshaping")
        feed = feed.reshape(expected)

    if inp["dtype"] in (np.int8, np.uint8):
        s, z = inp["quantization"]
        print(f"  *** Manually quantizing input (scale={s}, zp={z})")
        feed = np.clip(feed / s + z,
                       np.iinfo(inp["dtype"]).min,
                       np.iinfo(inp["dtype"]).max).astype(inp["dtype"])
    else:
        feed = feed.astype(inp["dtype"])

    interp.set_tensor(inp["index"], feed)
    interp.invoke()
    raw = interp.get_tensor(out["index"])

    if out["dtype"] in (np.int8, np.uint8):
        s, z = out["quantization"]
        print(f"  *** Manually dequantizing output (scale={s}, zp={z})")
        raw = (raw.astype(np.float32) - z) * s

    return raw.flatten().astype(np.float32)


def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    f32_path = str(REPO / "artifacts" / "reference" / "tflite" / "eegnet_f32" / "fold01" / "eegnet_float32.tflite")
    i8_path = str(REPO / "artifacts" / "reference" / "tflite" / "eegnet_int8" / "fold01" / "eegnet_int8.tflite")

    print("=" * 60)
    print("EEGNet parity diagnostic (host-only, no device)")
    print("=" * 60)
    print(f"f32 model: {f32_path}")
    print(f"i8  model: {i8_path}")

    mat_dir = os.environ.get("SSVEP_RAW_DATA_ROOT")
    rec = read_subject("s1", data_root=Path(mat_dir) if mat_dir else None)
    eeg = np.asarray(rec["eeg"])

    for trial_label, c, t in [("c=0,t=0", 0, 0), ("c=5,t=3", 5, 3)]:
        raw = np.asarray(eeg[c, :, :, t], dtype=np.float64).T.astype(np.float32)
        crop = raw[38:550, :].astype(np.float32)
        crop_z = _preprocess_zscore(crop)

        feed = crop_z.astype(np.float32).T[np.newaxis, ...]  # [1, 8, 512]

        print(f"\n{'-'*60}")
        print(f"Trial {trial_label}:  crop_z shape={crop_z.shape}  feed shape={feed.shape}")
        print(f"  feed[0,0,:5] = {feed[0,0,:5]}")
        print(f"  feed[0,1,:5] = {feed[0,1,:5]}")
        print(f"  feed min={feed.min():.4f}  max={feed.max():.4f}  mean={feed.mean():.4f}")

        print(f"\n--- float32 model (XNNPACK) ---")
        out_f32 = run_tflite(f32_path, feed.copy(), use_xnnpack=True)
        print(f"  logits = {out_f32}")
        print(f"  argmax = {np.argmax(out_f32)}")

        print(f"\n--- int8 model (XNNPACK) ---")
        out_i8_xnn = run_tflite(i8_path, feed.copy(), use_xnnpack=True)
        print(f"  logits = {out_i8_xnn}")
        print(f"  argmax = {np.argmax(out_i8_xnn)}")

        print(f"\n--- int8 model (NO XNNPACK) ---")
        out_i8_ref = run_tflite(i8_path, feed.copy(), use_xnnpack=False)
        print(f"  logits = {out_i8_ref}")
        print(f"  argmax = {np.argmax(out_i8_ref)}")

        diff_f32_i8x = np.abs(out_f32 - out_i8_xnn)
        diff_f32_i8r = np.abs(out_f32 - out_i8_ref)
        diff_i8x_i8r = np.abs(out_i8_xnn - out_i8_ref)

        print(f"\n--- Parity summary for {trial_label} ---")
        print(f"  f32 vs i8(XNNPACK):  max_diff={diff_f32_i8x.max():.8f}  "
              f"argmax agree={np.argmax(out_f32)==np.argmax(out_i8_xnn)}")
        print(f"  f32 vs i8(ref):      max_diff={diff_f32_i8r.max():.8f}  "
              f"argmax agree={np.argmax(out_f32)==np.argmax(out_i8_ref)}")
        print(f"  i8(XNNPACK) vs i8(ref): max_diff={diff_i8x_i8r.max():.8f}  "
              f"argmax agree={np.argmax(out_i8_xnn)==np.argmax(out_i8_ref)}")

    print(f"\n{'='*60}")
    print("If i8(XNNPACK) vs i8(ref) shows large diffs, XNNPACK is the issue.")
    print("If all host diffs are ~0.2, the host is fine and the device is different.")
    print("Compare host logits above with device logits from the terminal.")


if __name__ == "__main__":
    main()
