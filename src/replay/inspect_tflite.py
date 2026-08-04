#!/usr/bin/env python3
"""Inspect TFLite model tensor types, quantization params, and operator list.

Usage:
    python tools/replay/inspect_tflite.py export/12jfpm_eegnet_fold01_val_s1_int8/eegnet_int8.tflite
    python tools/replay/inspect_tflite.py --all     # inspect all deployed models
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REPO = Path(__file__).resolve().parents[2]

# All deployed model artifacts
DEPLOYED_MODELS = {
    "eegnet_f32":  "export/12jfpm_eegnet_fold01_val_s1/eegnet_float32.tflite",
    "eegnet_int8": "export/12jfpm_eegnet_fold01_val_s1_int8/eegnet_int8.tflite",
    "ccnn_f32":    "export/12jfpm_ccnn_fold01_val_s1/ccnn_float32.tflite",
    "ccnn_int8":   "export/12jfpm_ccnn_fold01_val_s1_int8/ccnn_int8.tflite",
    "convca_f32":  "export/12jfpm_convca_fold01_val_s1/convca_float32.tflite",
    "fbtcnn_f32":  "export/12jfpm_fbtcnn_fold01_val_s1/fbtcnn_float32.tflite",
    "fbtcnn_int8": "export/12jfpm_fbtcnn_fold01_val_s1_int8/fbtcnn_int8.tflite",
}


def inspect_model(path: str, label: str = "") -> dict:
    """Inspect a single TFLite model and return a summary dict."""
    import tensorflow as tf
    import numpy as np

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # Classify model type
    in_dtype = inp["dtype"]
    out_dtype = out["dtype"]
    in_quant = inp["quantization"]  # (scale, zp)

    has_internal_int8 = False
    ops = set()
    for t in interp.get_tensor_details():
        if t["dtype"] == np.int8 and t["index"] != inp["index"] and t["index"] != out["index"]:
            has_internal_int8 = True
        name = t["name"]
        if "tfl.quantize" in name.lower():
            ops.add("QUANTIZE")
        if "tfl.dequantize" in name.lower():
            ops.add("DEQUANTIZE")

    if in_dtype == np.float32 and out_dtype == np.float32 and has_internal_int8:
        model_class = "float-IO-quantized-internal (full_int8_float_io)"
    elif in_dtype == np.int8 and out_dtype == np.int8:
        model_class = "full-int8-IO"
    elif in_dtype == np.float32 and out_dtype == np.float32 and not has_internal_int8:
        model_class = "float32 (no quantization)"
    else:
        model_class = f"other (in={in_dtype}, out={out_dtype})"

    result = {
        "label": label,
        "path": path,
        "input_dtype": str(in_dtype),
        "input_shape": list(inp["shape"]),
        "input_quantization": in_quant,
        "output_dtype": str(out_dtype),
        "output_shape": list(out["shape"]),
        "output_quantization": out["quantization"],
        "model_class": model_class,
        "has_internal_int8": has_internal_int8,
    }
    return result


def print_model_info(path: str, label: str = ""):
    """Print detailed info for a single model."""
    import tensorflow as tf
    import numpy as np

    print(f"{'=' * 70}")
    print(f"  Model: {label or path}")
    print(f"  Path:  {path}")
    print(f"{'=' * 70}")

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_q = inp.get("quantization_parameters", {})
    out_q = out.get("quantization_parameters", {})
    in_scales = in_q.get("scales", np.array([]))
    in_zps = in_q.get("zero_points", np.array([]))
    out_scales = out_q.get("scales", np.array([]))
    out_zps = out_q.get("zero_points", np.array([]))

    print(f"\n  INPUT  tensor[{inp['index']}]:")
    print(f"    dtype : {inp['dtype']}")
    print(f"    shape : {inp['shape']}")
    print(f"    quant : scale={in_scales}, zp={in_zps}")
    print(f"    legacy: {inp['quantization']}")

    print(f"\n  OUTPUT tensor[{out['index']}]:")
    print(f"    dtype : {out['dtype']}")
    print(f"    shape : {out['shape']}")
    print(f"    quant : scale={out_scales}, zp={out_zps}")
    print(f"    legacy: {out['quantization']}")

    # Count internal tensor types
    n_f32 = n_i8 = n_other = 0
    for t in interp.get_tensor_details():
        if t["dtype"] == np.float32:
            n_f32 += 1
        elif t["dtype"] == np.int8:
            n_i8 += 1
        else:
            n_other += 1

    has_internal_int8 = n_i8 > 0
    if inp["dtype"] == np.float32 and out["dtype"] == np.float32 and has_internal_int8:
        model_class = "float-IO, internally-quantized (full_int8_float_io)"
    elif inp["dtype"] == np.int8 and out["dtype"] == np.int8:
        model_class = "full-int8-IO"
    elif inp["dtype"] == np.float32 and out["dtype"] == np.float32 and not has_internal_int8:
        model_class = "float32 (no quantization)"
    else:
        model_class = f"other (in={inp['dtype']}, out={out['dtype']})"

    print(f"\n  CLASSIFICATION: {model_class}")
    print(f"    internal tensors: {n_f32} float32, {n_i8} int8, {n_other} other")

    print(f"\n  RUNTIME CONTRACT:")
    print(f"    Feed float32 input  -> YES" if inp["dtype"] == np.float32
          else f"    Feed int8 input     -> YES (manual quantize needed)")
    print(f"    Read float32 output -> YES" if out["dtype"] == np.float32
          else f"    Read int8 output    -> YES (manual dequantize needed)")
    if has_internal_int8 and inp["dtype"] == np.float32:
        print(f"    Internal int8 computation handled by TFLite runtime (Quantize/Dequantize ops in graph)")
        print(f"    Caller does NOT manually quantize/dequantize — runtime does it internally")

    # Zero-input test
    print(f"\n  ZERO-INPUT TEST:")
    test_input = np.zeros(tuple(inp["shape"]), dtype=inp["dtype"])
    interp.set_tensor(inp["index"], test_input)
    interp.invoke()
    raw_out = interp.get_tensor(out["index"])
    out_flat = raw_out.flatten()[:12]
    print(f"    output[0:12] = {out_flat}")

    print()


def main():
    args = sys.argv[1:]

    if not args or args == ["--all"]:
        print("Inspecting all deployed models...\n")
        for label, rel_path in DEPLOYED_MODELS.items():
            full_path = str(REPO / rel_path)
            if os.path.isfile(full_path):
                print_model_info(full_path, label)
            else:
                print(f"  {label}: NOT FOUND at {full_path}\n")
    else:
        for path in args:
            if os.path.isfile(path):
                print_model_info(path)
            else:
                print(f"  NOT FOUND: {path}")


if __name__ == "__main__":
    main()
