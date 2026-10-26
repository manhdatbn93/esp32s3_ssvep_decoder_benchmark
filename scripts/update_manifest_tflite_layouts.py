#!/usr/bin/env python3
"""Record interpreter-reported TFLite I/O layouts in artifacts/reference/manifest.json.

The original ``input_shape``/``output_shape`` fields held the logical PyTorch/ONNX
graph shape, which onnx2tf converts to channels-last before writing the TFLite file.
This script renames them to ``logical_input_shape``/``logical_output_shape`` and adds
``tflite_input_shape``/``tflite_output_shape`` (and dtypes) read from each retained
TFLite artifact. Only manifest metadata changes; artifact bytes are not touched.

    python scripts/update_manifest_tflite_layouts.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "reference" / "manifest.json"
LAYOUT_KEYS = ("tflite_input_shape", "tflite_input_dtype", "tflite_output_shape", "tflite_output_dtype")


def interpreter_layout(path: Path) -> dict:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return {
        "tflite_input_shape": [int(x) for x in inp["shape"]],
        "tflite_input_dtype": inp["dtype"].__name__,
        "tflite_output_shape": [int(x) for x in out["shape"]],
        "tflite_output_dtype": out["dtype"].__name__,
    }


def updated_entry(entry: dict) -> dict:
    if entry.get("artifact_type") != "tflite":
        return entry
    layout = interpreter_layout(ROOT / entry["relative_path"])
    new: dict = {}
    for key, value in entry.items():
        if key == "input_shape":
            new["logical_input_shape"] = value
        elif key == "output_shape":
            new["logical_output_shape"] = value
        elif key in LAYOUT_KEYS:
            continue
        else:
            new[key] = value
        if key == "output_dtype":
            new.update(layout)
    if "tflite_input_shape" not in new:
        new.update(layout)
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Fail if the manifest is not up to date")
    args = ap.parse_args()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    new_entries = [updated_entry(e) for e in entries]
    text = json.dumps(new_entries, indent=2) + "\n"
    if args.check:
        current = MANIFEST.read_text(encoding="utf-8")
        if current != text:
            print("manifest TFLite layouts are out of date")
            return 1
        print("manifest TFLite layouts up to date")
        return 0
    MANIFEST.write_bytes(text.encode("utf-8"))
    n = sum(1 for e in new_entries if e.get("artifact_type") == "tflite")
    print(f"updated {n} TFLite entries in {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
