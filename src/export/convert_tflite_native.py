#!/usr/bin/env python3
"""
Convert EEGNet and CCNN ONNX models to TFLite float32 + int8.

Uses onnx-tf to convert ONNX -> TF SavedModel, then tf.lite.TFLiteConverter.

Usage:
  python -m export.convert_tflite_native [--model eegnet|ccnn|all]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _ccnn_fft_preprocess(raw_tc: np.ndarray) -> np.ndarray:
    """[512,8] -> [1,1,8,220] post-FFT features (matching CCNN preprocessing)."""
    resolution, fs = 0.2930, 256.0
    freq_lo, freq_hi = 3.0, 35.0
    T = raw_tc.shape[0]
    nfft = round(fs / resolution)
    idx_lo = round(freq_lo / resolution)
    idx_hi = round(freq_hi / resolution) + 1
    x_ct = raw_tc.T.astype(np.float64)
    spectrum = np.fft.fft(x_ct, n=nfft, axis=-1) / (T / 2.0)
    features = np.concatenate(
        [spectrum[:, idx_lo:idx_hi].real, spectrum[:, idx_lo:idx_hi].imag], axis=-1
    ).astype(np.float32)
    return features.reshape(1, 1, 8, -1)


def convert_one(model_name: str) -> dict:
    """Convert a single model ONNX -> TFLite float32 + int8."""
    import tensorflow as tf

    export_dir = REPO / "export" / f"12jfpm_{model_name}_fold01_val_s1"
    onnx_files = list(export_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"  {model_name}: no ONNX file found")
        return {"model": model_name, "float32": "no_onnx", "int8": "no_onnx"}

    onnx_path = onnx_files[0]
    saved_model_dir = export_dir / f"{model_name}_tf_saved"
    print(f"  {model_name}: {onnx_path.name} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # Step 1: ONNX -> TF SavedModel via onnx-tf
    try:
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        tf_rep.export_graph(str(saved_model_dir))
        print(f"    -> TF SavedModel exported to {saved_model_dir.name}")
    except Exception as e:
        msg = str(e)[:200]
        print(f"    ONNX->TF conversion failed: {msg}")
        result = {"model": model_name, "float32": f"failed: {msg}", "int8": "not_attempted"}
        (export_dir / "tflite_status.json").write_text(json.dumps(result, indent=2))
        return result

    # Step 2: Float32 TFLite
    f32_status = "not_attempted"
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_f32 = converter.convert()
        f32_path = export_dir / f"{model_name}_float32.tflite"
        f32_path.write_bytes(tflite_f32)
        f32_status = "success"
        print(f"    -> {f32_path.name} ({len(tflite_f32) / 1024:.1f} KB)")
    except Exception as e:
        f32_status = f"failed: {str(e)[:150]}"
        print(f"    Float32 failed: {f32_status}")

    # Step 3: Int8 TFLite with representative dataset
    int8_status = "not_attempted"
    needs_fft = model_name == "ccnn"
    try:
        calib_data = []
        for i in range(5):
            p = export_dir / f"smoke_input_idx{i}_float32.bin"
            if p.exists():
                raw = np.fromfile(str(p), dtype=np.float32).reshape(512, 8)
                if needs_fft:
                    calib_data.append(_ccnn_fft_preprocess(raw))
                else:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(str(onnx_path))
                    inp_shape = sess.get_inputs()[0].shape
                    calib_data.append(raw.reshape(inp_shape))

        def representative_dataset():
            for d in calib_data:
                yield [d]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_int8 = converter.convert()
        int8_path = export_dir / f"{model_name}_int8.tflite"
        int8_path.write_bytes(tflite_int8)
        int8_status = "success"
        print(f"    -> {int8_path.name} ({len(tflite_int8) / 1024:.1f} KB)")
    except Exception as e:
        int8_status = f"failed: {str(e)[:150]}"
        print(f"    Int8 failed: {int8_status}")

    # Step 4: Verify TFLite float32 parity
    if f32_status == "success":
        try:
            f32_path = export_dir / f"{model_name}_float32.tflite"
            interpreter = tf.lite.Interpreter(model_path=str(f32_path))
            interpreter.allocate_tensors()
            inp_det = interpreter.get_input_details()[0]
            out_det = interpreter.get_output_details()[0]

            ref = json.loads((export_dir / "host_reference.json").read_text())
            print(f"    TFLite float32 parity check:")
            for i in range(min(3, len(calib_data))):
                key = f"idx{i}"
                if key not in ref:
                    continue
                vec = calib_data[i].astype(np.float32)
                interpreter.set_tensor(inp_det["index"], vec)
                interpreter.invoke()
                tfl_out = interpreter.get_tensor(out_det["index"]).flatten()
                exp_logits = np.array(ref[key]["logits"], dtype=np.float32)
                max_diff = float(np.max(np.abs(tfl_out - exp_logits)))
                argmax_match = int(np.argmax(tfl_out)) == ref[key]["argmax"]
                status = "PASS" if (argmax_match and max_diff < 0.05) else "FAIL"
                print(f"      {key}: max_diff={max_diff:.2e} argmax={'match' if argmax_match else 'MISMATCH'} [{status}]")
        except Exception as e:
            print(f"    Parity check error: {e}")

    result = {
        "model": model_name,
        "onnx_path": onnx_path.name,
        "float32": f32_status,
        "int8": int8_status,
    }
    (export_dir / "tflite_status.json").write_text(json.dumps(result, indent=2))

    # Clean up SavedModel to save space
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["eegnet", "ccnn", "all"])
    args = ap.parse_args()

    print("=" * 60)
    print("TFLite Conversion (onnx-tf -> TF SavedModel -> TFLite)")
    print("=" * 60)

    models = ["eegnet", "ccnn"] if args.model == "all" else [args.model]
    results = {}
    for m in models:
        results[m] = convert_one(m)

    print("\n" + "=" * 60)
    print("Summary:")
    for m, r in results.items():
        print(f"  {m}: float32={r['float32']} | int8={r['int8']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
