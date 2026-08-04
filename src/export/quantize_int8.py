#!/usr/bin/env python3
"""
Full int8 quantization + host-side parity verification for ESP32-S3 deploy.

Uses representative calibration data from the 12JFPM training set to
calibrate activation ranges. Produces per-model int8 TFLite artifacts with
parity reports against the float32 TFLite baseline.

Run from repo root:
  $env:TF_ENABLE_ONEDNN_OPTS="0"; $env:PYTHONUTF8="1"
  .venv_tflite310\\Scripts\\python.exe -u -m export.quantize_int8
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from export.common import get_fold_test_vectors, load_processed_npz
from export.protocol_cli import (
    DEFAULT_PROTOCOL,
    add_protocol_arg,
    add_test_subject_args,
    apply_protocol_side_effects,
    resolve_test_subject,
)
from export.subject_paths import (
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    assert_not_legacy_artifact,
    export_dir,
    export_dir_name,
    fold_for_subject,
)

ACTIVE_FOLD: int = 1
ACTIVE_VAL_SUBJECT: str = "s1"
ACTIVE_PROTOCOL: str = DEFAULT_PROTOCOL

N_CALIB_PER_CLASS = 16
MODELS = ["eegnet", "ccnn", "convca", "fbtcnn"]

# ── Calibration data ────────────────────────────────────────────────

def _load_calibration_raw(
    npz: dict, n_per_class: int = N_CALIB_PER_CLASS,
) -> tuple[np.ndarray, list[str], dict[int, int]]:
    """Balanced calibration from outer-training subjects only (excludes held-out).

    Takes the first ``n_per_class`` trials per class from the masked training pool
    (trial order in processed.npz). The saved ``calibration_subjects`` list is the
    set of subject ids that contribute at least one selected trial — not necessarily
    all nine outer-training subjects. The outer test subject is always excluded.

    Returns ([N, 512, 8] float32, calibration subject ids, per-class counts).
    """
    subjects = npz.get("subjects", npz.get("subject"))
    X, y = npz["X"], npz["y"]
    mask = np.array([str(s) != ACTIVE_VAL_SUBJECT for s in subjects])
    X_train, y_train = X[mask], y[mask]
    subj_train = subjects[mask]

    indices: list[int] = []
    per_class: dict[int, int] = {}
    for c in sorted(np.unique(y_train)):
        cls_idx = np.where(y_train == c)[0]
        take = cls_idx[:n_per_class].tolist()
        per_class[int(c)] = len(take)
        indices.extend(take)

    calib_subjects = sorted(set(str(subj_train[i]) for i in indices))
    return X_train[indices].astype(np.float32), calib_subjects, per_class


def _calib_eegnet(raw: np.ndarray) -> np.ndarray:
    """[N,512,8] → [N,8,512] (onnx2tf transposed to NHWC-like 3D)."""
    return np.transpose(raw, (0, 2, 1))


def _calib_ccnn(raw: np.ndarray) -> np.ndarray:
    """[N,512,8] → FFT → [N,8,220,1] (NHWC)."""
    from src.neural.ccnn import _complex_spectrum_features
    out = []
    for i in range(len(raw)):
        t = torch.from_numpy(raw[i]).unsqueeze(0)
        spec = _complex_spectrum_features(t, 256.0).numpy()
        out.append(np.transpose(spec, (0, 2, 3, 1)))
    return np.concatenate(out, axis=0)


def _calib_convca(raw: np.ndarray) -> np.ndarray:
    """[N,512,8] → [N,512,8] unchanged (-kat kept original layout)."""
    return raw


def _calib_fbtcnn(raw: np.ndarray) -> np.ndarray:
    """[N,512,8] → 4-band Butterworth → [N,8,512,4] (NHWC)."""
    from src.neural.fbtcnn import FBtCNN
    model = FBtCNN(seq_len=512, embed_dim=8, num_classes=12,
                   sampling_rate_hz=256.0, dropout=0.4)
    out = []
    for i in range(len(raw)):
        x = torch.from_numpy(raw[i]).unsqueeze(0)
        x = x.transpose(1, 2).unsqueeze(1)
        bands = model._filter_bank(x)
        stacked = torch.stack(bands, dim=1)
        if stacked.dim() == 5:
            stacked = stacked.squeeze(2)
        out.append(np.transpose(stacked.numpy(), (0, 2, 3, 1)))
    return np.concatenate(out, axis=0)


CALIB_FN = {
    "eegnet": _calib_eegnet,
    "ccnn": _calib_ccnn,
    "convca": _calib_convca,
    "fbtcnn": _calib_fbtcnn,
}

# ── Quantization ────────────────────────────────────────────────────

def _quantize_model(saved_model_dir: Path, calib_data: np.ndarray,
                    out_path: Path) -> dict:
    """Full int8 quantization with float32 IO (best for esp-tflite-micro).

    Falls back to dynamic-range if full int8 fails.
    """
    import tensorflow as tf

    result = {"method": None, "success": False}

    # Attempt 1: full int8 with representative dataset
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_gen(calib_data)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        result.update(method="full_int8_float_io", success=True,
                      size_kb=round(len(tflite_model) / 1024, 1))
        return result
    except Exception as e:
        result["full_int8_error"] = str(e)[:300]
        print(f"    full int8 failed: {str(e)[:200]}")

    # Attempt 2: dynamic-range quantization (no representative data needed)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        result.update(method="dynamic_range", success=True,
                      size_kb=round(len(tflite_model) / 1024, 1))
        return result
    except Exception as e:
        result["dynamic_range_error"] = str(e)[:300]
        print(f"    dynamic-range also failed: {str(e)[:200]}")

    return result


def _make_rep_gen(calib_data: np.ndarray):
    """Create a representative dataset generator for TFLiteConverter."""
    def gen():
        for i in range(len(calib_data)):
            yield [calib_data[i:i+1].astype(np.float32)]
    return gen


# ── Parity checks ──────────────────────────────────────────────────

def _run_tflite(path: Path, data: np.ndarray) -> np.ndarray:
    """Run TFLite inference, handling quantized IO if needed."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    feed = data.astype(np.float32)
    if inp["dtype"] in (np.int8, np.uint8):
        s, z = inp["quantization"]
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
        return (raw.astype(np.float32) - z) * s
    return raw.astype(np.float32)


def _parity(ref: np.ndarray, test: np.ndarray, label: str) -> dict:
    r, t = ref.flatten().astype(np.float32), test.flatten().astype(np.float32)
    diff = np.abs(r - t)
    return {
        "comparison": label,
        "argmax_match": bool(np.argmax(r) == np.argmax(t)),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "ref_argmax": int(np.argmax(r)),
        "test_argmax": int(np.argmax(t)),
    }


def _inspect_ops(path: Path) -> list[str]:
    import tensorflow as tf
    try:
        interp = tf.lite.Interpreter(model_path=str(path))
        return sorted(set(d["op_name"] for d in interp._get_ops_details()))
    except Exception:
        return ["(inspection failed)"]


# ── Per-model processing ───────────────────────────────────────────

_PREPROCESSING_NOTES = {
    "eegnet": "none — raw [B,T,C] transposed to [B,C,T] by onnx2tf",
    "ccnn": "FFT external: [512,8] → [1,8,220,1] (NHWC)",
    "convca": "none — raw [B,T,C], templates baked (-kat kept layout)",
    "fbtcnn": "4-band Butterworth filtfilt external → [1,8,512,4] (NHWC)",
}


def _model_cfg(name: str) -> dict:
    """Resolve per-fold paths at call time so --val-subject / --fold / --protocol work."""
    f32_dir = export_dir(
        name, ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, protocol=ACTIVE_PROTOCOL, repo=REPO,
    )
    saved_model = f32_dir / f"{name}_tf"
    f32_tflite = f32_dir / f"{name}_float32.tflite"
    return {
        "saved_model": str(saved_model),
        "float32": str(f32_tflite),
        "preprocessing": _PREPROCESSING_NOTES.get(name, ""),
    }


MODEL_CFG = type("_LazyCfg", (), {
    "__contains__": lambda self, k: k in _PREPROCESSING_NOTES,
    "__getitem__": lambda self, k: _model_cfg(k),
})()


def process_model(name: str, calib_raw: np.ndarray,
                  test_vectors: list[np.ndarray], npz: dict,
                  calib_subjects: list[str],
                  calib_per_class: dict[int, int]) -> dict:
    print(f"\n{'='*60}")
    tag_label = "test" if ACTIVE_PROTOCOL == PROTOCOL_STRICT_LOSO_INNER_VAL else "val"
    print(f"  {name.upper()} — int8 quantization "
          f"(fold{ACTIVE_FOLD:02d}/{tag_label}_{ACTIVE_VAL_SUBJECT})")
    print(f"{'='*60}")

    cfg = MODEL_CFG[name]
    saved_model_dir = Path(cfg["saved_model"])
    f32_path = Path(cfg["float32"])
    out_dir = export_dir(
        name, ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, int8=True,
        protocol=ACTIVE_PROTOCOL, repo=REPO,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"model": name, "out_dir": str(out_dir), "preprocessing": cfg["preprocessing"]}

    # Prepare calibration data
    calib_fn = CALIB_FN[name]
    print(f"  Preparing {len(calib_raw)} calibration samples...")
    t0 = time.time()
    calib = calib_fn(calib_raw)
    print(f"  Calibration shape: {calib.shape} ({time.time()-t0:.1f}s)")
    report["calib_shape"] = list(calib.shape)
    report["calib_n_samples"] = len(calib)
    report["calib_strategy"] = (
        f"Balanced {N_CALIB_PER_CLASS} trials/class from outer-training pool "
        f"(excludes outer test {ACTIVE_VAL_SUBJECT}); "
        f"subjects listed are those contributing selected trials, not all nine train subjects"
    )
    report["calibration_subjects"] = calib_subjects
    report["calibration_per_class_counts"] = calib_per_class
    report["outer_test_subject_excluded"] = ACTIVE_VAL_SUBJECT
    report["protocol"] = ACTIVE_PROTOCOL
    assert_not_legacy_artifact(f32_path, protocol=ACTIVE_PROTOCOL, context="f32 tflite")
    assert_not_legacy_artifact(out_dir, protocol=ACTIVE_PROTOCOL, context="int8 export dir")

    if not saved_model_dir.is_dir():
        msg = f"SavedModel directory missing (run convert_all_tflite first): {saved_model_dir}"
        print(f"  {msg}")
        report["status"] = "missing_saved_model"
        report["error"] = msg
        _write_report(out_dir, name, report)
        return report

    if not f32_path.is_file():
        msg = f"Float32 TFLite missing (run convert_all_tflite first): {f32_path}"
        print(f"  {msg}")
        report["status"] = "missing_float32_tflite"
        report["error"] = msg
        _write_report(out_dir, name, report)
        return report

    report["float32_size_kb"] = round(f32_path.stat().st_size / 1024, 1)

    # Quantize
    int8_path = out_dir / f"{name}_int8.tflite"
    print(f"  Quantizing via TFLiteConverter...")
    quant_result = _quantize_model(saved_model_dir, calib, int8_path)
    report["quantization"] = quant_result

    if not quant_result.get("success"):
        print(f"  QUANTIZATION FAILED")
        report["status"] = "quantization_failed"
        _write_report(out_dir, name, report)
        return report

    size_kb = quant_result.get("size_kb")
    print(f"  {quant_result.get('method', 'unknown')}: {size_kb} KB")
    report["int8_size_kb"] = size_kb

    # Inspect operators
    ops = _inspect_ops(int8_path)
    report["int8_ops"] = ops
    print(f"  Ops: {ops}")

    # IO details
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(int8_path))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    report["int8_input_shape"] = list(inp_d["shape"])
    report["int8_input_dtype"] = str(inp_d["dtype"])
    report["int8_output_shape"] = list(out_d["shape"])
    report["int8_output_dtype"] = str(out_d["dtype"])
    if inp_d["quantization"] != (0.0, 0):
        report["int8_input_quant"] = {
            "scale": float(inp_d["quantization"][0]),
            "zero_point": int(inp_d["quantization"][1]),
        }

    # Parity checks: float32 TFLite vs int8 TFLite
    print(f"  Parity checks (float32 vs int8)...")
    parity = []
    for i, v in enumerate(test_vectors[:5]):
        test_input = calib_fn(v[np.newaxis])
        try:
            f32_out = _run_tflite(f32_path, test_input)
            i8_out = _run_tflite(int8_path, test_input)
            p = _parity(f32_out.flatten(), i8_out.flatten(),
                        f"vec{i}_f32_vs_int8")
            parity.append(p)
            tag = "PASS" if p["argmax_match"] else "FAIL"
            print(f"    {p['comparison']}: {tag} "
                  f"(max_diff={p['max_abs_diff']:.6f})")
        except Exception as e:
            parity.append({"comparison": f"vec{i}_f32_vs_int8",
                           "error": str(e)[:200]})
            print(f"    vec{i}: ERROR — {str(e)[:100]}")

    report["parity"] = parity
    n_match = sum(1 for p in parity if p.get("argmax_match", False))
    n_total = sum(1 for p in parity if "error" not in p)
    report["argmax_agreement"] = f"{n_match}/{n_total}"

    max_diffs = [p["max_abs_diff"] for p in parity if "max_abs_diff" in p]
    if max_diffs:
        report["worst_max_diff"] = max(max_diffs)
        report["mean_max_diff"] = float(np.mean(max_diffs))

    report["status"] = "complete"
    report["elapsed_s"] = round(time.time() - t0, 1)

    _write_report(out_dir, name, report)
    return report


def _write_report(out_dir: Path, name: str, report: dict):
    (out_dir / "int8_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    meta = {
        "method": name,
        "dataset": "12JFPM",
        "fold": ACTIVE_FOLD,
        "val_subject": ACTIVE_VAL_SUBJECT,
        "quantization": report.get("quantization", {}).get("method", "unknown"),
        "int8_size_kb": report.get("int8_size_kb"),
        "float32_size_kb": report.get("float32_size_kb"),
        "preprocessing": report.get("preprocessing"),
        "calibration": report.get("calib_strategy"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Int8 quantization + host-side parity (default: strict_loso_inner_val).",
    )
    add_test_subject_args(ap, default="s1")
    add_protocol_arg(ap)
    ap.add_argument("--fold", type=int, default=None,
                    help="Fold number (default: inferred from test subject).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Models to quantize; default: eegnet ccnn convca fbtcnn.")
    args = ap.parse_args()

    global ACTIVE_FOLD, ACTIVE_VAL_SUBJECT, ACTIVE_PROTOCOL
    ACTIVE_PROTOCOL = args.protocol
    apply_protocol_side_effects(ACTIVE_PROTOCOL)
    ACTIVE_VAL_SUBJECT = resolve_test_subject(args)
    ACTIVE_FOLD = (
        args.fold if args.fold is not None else fold_for_subject(ACTIVE_VAL_SUBJECT)
    )

    print("=" * 60)
    print("Int8 Quantization + Host-Side Parity Pipeline")
    print(f"Python: {sys.version}")
    print(f"Repo:   {REPO}")
    print(f"Fold:   {ACTIVE_FOLD:02d}  val_subject={ACTIVE_VAL_SUBJECT}")
    print("=" * 60)

    npz = load_processed_npz()
    test_vectors = get_fold_test_vectors(npz, val_subject=ACTIVE_VAL_SUBJECT)
    print(f"Loaded {len(test_vectors)} test vectors for {ACTIVE_VAL_SUBJECT}")

    calib_raw, calib_subjects, calib_per_class = _load_calibration_raw(npz)
    print(f"Calibration pool: {calib_raw.shape} "
          f"({N_CALIB_PER_CLASS} per class, subjects={calib_subjects})")

    models = args.models or [m for m in sys.argv[1:] if m in MODEL_CFG] or MODELS
    models = [m for m in models if m in MODEL_CFG]

    results = {}
    for name in models:
        try:
            results[name] = process_model(
                name, calib_raw, test_vectors, npz, calib_subjects, calib_per_class,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"model": name, "status": f"fatal: {e}"}

    # Summary
    print(f"\n{'='*60}")
    print("INT8 QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    tag = export_dir_name("eegnet", ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, protocol=ACTIVE_PROTOCOL)
    tag = tag.replace("12jfpm_eegnet_", "")
    summary_path = REPO / "export" / f"int8_quantization_summary_{tag}.json"
    if ACTIVE_PROTOCOL == PROTOCOL_STRICT_LOSO_INNER_VAL:
        summary_path = (
            REPO / "export" / PROTOCOL_STRICT_LOSO_INNER_VAL
            / f"int8_quantization_summary_{tag}.json"
        )
    if ACTIVE_VAL_SUBJECT == "s1" and ACTIVE_FOLD == 1 and ACTIVE_PROTOCOL == PROTOCOL_SUBJECT_AS_VAL:
        legacy = REPO / "export" / "int8_quantization_summary.json"
        if legacy.exists():
            summary_path = legacy
    summary_path.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")

    def _kb(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, (int, float)):
            return f"{v:.1f}"
        return str(v)

    for name, r in results.items():
        f32 = _kb(r.get("float32_size_kb"))
        i8 = _kb(r.get("int8_size_kb"))
        q = r.get("quantization") or {}
        method = q.get("method") or "—"
        agree = r.get("argmax_agreement", "—")
        worst = r.get("worst_max_diff", "—")
        if isinstance(worst, float):
            worst = f"{worst:.6f}"
        status = r.get("status", "unknown")
        print(f"  {name:10s} | f32 {f32:>7s} KB → int8 {i8:>7s} KB "
              f"| {method:25s} | argmax={agree!s:5s} | max_diff={worst!s:>10s} "
              f"| {status}")

    print(f"\nFull report: {summary_path}")


if __name__ == "__main__":
    main()
