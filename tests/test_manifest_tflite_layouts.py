from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "reference" / "manifest.json"


def _tflite_entries() -> list[dict]:
    return [e for e in json.loads(MANIFEST.read_text(encoding="utf-8")) if e["artifact_type"] == "tflite"]


def test_manifest_has_no_ambiguous_shape_fields():
    for entry in _tflite_entries():
        assert "input_shape" not in entry and "output_shape" not in entry, entry["artifact_id"]
        assert "logical_input_shape" in entry and "tflite_input_shape" in entry, entry["artifact_id"]


@pytest.mark.parametrize("entry", _tflite_entries(), ids=lambda e: e["artifact_id"])
def test_manifest_tflite_layout_matches_interpreter(entry):
    tf = pytest.importorskip("tensorflow")
    interp = tf.lite.Interpreter(model_path=str(ROOT / entry["relative_path"]))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert [int(x) for x in inp["shape"]] == entry["tflite_input_shape"]
    assert [int(x) for x in out["shape"]] == entry["tflite_output_shape"]
    assert inp["dtype"].__name__ == entry["tflite_input_dtype"]
    assert out["dtype"].__name__ == entry["tflite_output_dtype"]
