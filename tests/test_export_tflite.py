"""Export wrapper regression tests.

The conversion tests run only in the pinned conversion environment
(environment/tflite_conversion_py310.txt) and compare a fresh fold-1 export
with the retained reference artifact byte-for-byte.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "reference" / "manifest.json"
PROCESSED_NPZ = Path(
    os.environ.get("SSVEP_PROCESSED_NPZ", ROOT / "data" / "processed" / "12JFPM" / "processed.npz")
)

needs_converter = pytest.mark.skipif(
    importlib.util.find_spec("onnx2tf") is None or importlib.util.find_spec("tensorflow") is None,
    reason="pinned TFLite conversion environment not installed",
)
needs_dataset = pytest.mark.skipif(
    not PROCESSED_NPZ.is_file(), reason="processed 12JFPM NPZ not available (set SSVEP_PROCESSED_NPZ)"
)


def _reference(artifact_id: str) -> dict:
    return next(e for e in json.loads(MANIFEST.read_text(encoding="utf-8")) if e["artifact_id"] == artifact_id)


def _run_export(tmp_path: Path, model: str, quant: str, extra: list[str] | None = None) -> Path:
    out = tmp_path / f"{model}_{quant}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_tflite.py"),
        "--model", model,
        "--fold", "1",
        "--checkpoint", str(ROOT / "artifacts" / "reference" / "checkpoints" / model / "fold01" / "fold01_best.pt"),
        "--manifest", str(ROOT / "manifests" / "loso" / "fold01.json"),
        "--quantization", quant,
        "--output", str(out),
        *(extra or []),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    return out


def test_export_refuses_reference_tree():
    cmd = [
        sys.executable, str(ROOT / "scripts" / "export_tflite.py"),
        "--model", "eegnet", "--fold", "1",
        "--checkpoint", str(ROOT / "artifacts" / "reference" / "checkpoints" / "eegnet" / "fold01" / "fold01_best.pt"),
        "--output", str(ROOT / "artifacts" / "reference" / "tflite"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "protected artifact tree" in (proc.stdout + proc.stderr)


def test_export_rejects_mismatched_manifest():
    cmd = [
        sys.executable, str(ROOT / "scripts" / "export_tflite.py"),
        "--model", "eegnet", "--fold", "2", "--dry-run",
        "--checkpoint", str(ROOT / "artifacts" / "reference" / "checkpoints" / "eegnet" / "fold02" / "fold02_best.pt"),
        "--manifest", str(ROOT / "manifests" / "loso" / "fold01.json"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "does not match --fold" in (proc.stdout + proc.stderr)


@needs_converter
def test_eegnet_f32_fold1_export_is_byte_identical(tmp_path):
    out = _run_export(tmp_path, "eegnet", "f32")
    report = json.loads((out / "eegnet_f32_export_report.json").read_text(encoding="utf-8"))
    assert report["tflite_sha256"] == _reference("eegnet_f32_tflite_fold01")["sha256"]
    assert report["tflite_input_shape"] == [1, 8, 512]
    assert not (out / "_work").exists()


@needs_converter
@needs_dataset
def test_eegnet_int8_fold1_export_is_byte_identical(tmp_path):
    out = _run_export(tmp_path, "eegnet", "int8", ["--dataset-root", str(PROCESSED_NPZ)])
    report = json.loads((out / "eegnet_int8_export_report.json").read_text(encoding="utf-8"))
    assert report["tflite_sha256"] == _reference("eegnet_int8_tflite_fold01")["sha256"]
    assert report["quantization_method"] == "full_int8_float_io"
    assert report["calibration_n"] == 192
