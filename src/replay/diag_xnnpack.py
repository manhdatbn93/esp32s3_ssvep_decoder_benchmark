#!/usr/bin/env python3
"""Compare EEGNet-int8 inference: XNNPACK vs reference kernels on the host.

If the two produce different results, it proves that int8 kernel implementation
differences can cause parity divergence. This is the same mechanism that causes
ESP-NN (device) vs XNNPACK (host) differences.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.replay.host_interpreter import (  # noqa: E402
    HOST_REFERENCE_OP_RESOLVER,
    NO_DEFAULT_DELEGATES_OP_RESOLVER,
    make_interpreter,
)

INT8_MODEL = str(REPO / "artifacts/reference/tflite/eegnet_int8/fold01/eegnet_int8.tflite")
F32_MODEL = str(REPO / "artifacts/reference/tflite/eegnet_f32/fold01/eegnet_float32.tflite")


def load_trials(n=20):
    """Generate n synthetic trials with z-scored statistics (mean≈0, std≈1).

    Uses seeded random data matching channel-wise z-scored EEG distribution
    (the standard preprocessing contract for EEGNet replay input).
    """
    rng = np.random.default_rng(42)
    trials = []
    for i in range(n):
        data = rng.standard_normal((1, 8, 512)).astype(np.float32)
        trials.append(data)
    return trials


def run_model(model_path, inputs, use_xnnpack=True):
    """Run inference on a list of inputs, return list of output arrays."""
    resolver = HOST_REFERENCE_OP_RESOLVER if use_xnnpack else NO_DEFAULT_DELEGATES_OP_RESOLVER
    interp = make_interpreter(model_path, resolver)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    results = []
    for x in inputs:
        feed = x.astype(inp_detail["dtype"])
        if feed.shape != tuple(inp_detail["shape"]):
            feed = feed.reshape(inp_detail["shape"])
        interp.set_tensor(inp_detail["index"], feed)
        interp.invoke()
        out = interp.get_tensor(out_detail["index"]).flatten().astype(np.float32)
        results.append(out)
    return results


def main():
    print("Loading trials...")
    trials = load_trials(20)
    print(f"  Loaded {len(trials)} trials, shape={trials[0].shape}")

    print("\n--- Float32 model (XNNPACK) ---")
    f32_xnn = run_model(F32_MODEL, trials, use_xnnpack=True)

    print("\n--- Int8 model (XNNPACK) ---")
    i8_xnn = run_model(INT8_MODEL, trials, use_xnnpack=True)

    print("\n--- Int8 model (BUILTIN_WITHOUT_DEFAULT_DELEGATES: no XNNPACK) ---")
    i8_ref = run_model(INT8_MODEL, trials, use_xnnpack=False)

    print(f"\n{'='*75}")
    print(f"  {'Trial':>5s}  {'f32_argmax':>10s}  {'i8_xnn_am':>10s}  {'i8_ref_am':>10s}  "
          f"{'xnn_vs_ref':>10s}  {'f32_vs_i8x':>11s}  {'f32_vs_i8r':>11s}")
    print(f"{'='*75}")

    diffs_xnn_ref = []
    diffs_f32_i8x = []
    diffs_f32_i8r = []
    agree_xnn_ref = 0
    agree_f32_i8x = 0
    agree_f32_i8r = 0

    for i in range(len(trials)):
        f32_am = int(np.argmax(f32_xnn[i]))
        i8x_am = int(np.argmax(i8_xnn[i]))
        i8r_am = int(np.argmax(i8_ref[i]))

        d_xnn_ref = float(np.max(np.abs(i8_xnn[i] - i8_ref[i])))
        d_f32_i8x = float(np.max(np.abs(f32_xnn[i] - i8_xnn[i])))
        d_f32_i8r = float(np.max(np.abs(f32_xnn[i] - i8_ref[i])))

        diffs_xnn_ref.append(d_xnn_ref)
        diffs_f32_i8x.append(d_f32_i8x)
        diffs_f32_i8r.append(d_f32_i8r)

        if i8x_am == i8r_am: agree_xnn_ref += 1
        if f32_am == i8x_am: agree_f32_i8x += 1
        if f32_am == i8r_am: agree_f32_i8r += 1

        print(f"  {i:5d}  {f32_am:10d}  {i8x_am:10d}  {i8r_am:10d}  "
              f"{d_xnn_ref:10.6f}  {d_f32_i8x:11.6f}  {d_f32_i8r:11.6f}")

    n = len(trials)
    print(f"\n{'='*75}")
    print(f"  XNNPACK vs Reference (same int8 model, same input, different kernel):")
    print(f"    argmax agree: {agree_xnn_ref}/{n}")
    print(f"    max_abs_diff: {max(diffs_xnn_ref):.8f}")
    print(f"    mean_abs_diff: {np.mean(diffs_xnn_ref):.8f}")
    print(f"\n  f32 vs int8(XNNPACK):")
    print(f"    argmax agree: {agree_f32_i8x}/{n}")
    print(f"    max_abs_diff: {max(diffs_f32_i8x):.8f}")
    print(f"    mean_abs_diff: {np.mean(diffs_f32_i8x):.8f}")
    print(f"\n  f32 vs int8(reference):")
    print(f"    argmax agree: {agree_f32_i8r}/{n}")
    print(f"    max_abs_diff: {max(diffs_f32_i8r):.8f}")
    print(f"    mean_abs_diff: {np.mean(diffs_f32_i8r):.8f}")
    print(f"\n  Conclusion:")
    if max(diffs_xnn_ref) < 0.001:
        print(f"    XNNPACK and reference produce IDENTICAL results on host.")
        print(f"    -> Device divergence is ESP-NN specific, not a host-side issue.")
    else:
        print(f"    XNNPACK and reference DIFFER on host (max={max(diffs_xnn_ref):.6f}).")
        print(f"    -> Int8 kernel implementation differences ARE real.")
        print(f"    -> ESP-NN vs XNNPACK differences on device are expected.")


if __name__ == "__main__":
    main()
