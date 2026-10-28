"""Consistency checks for the static int8 operator audit."""

from collections import Counter

from analysis.operator_precision_audit.audit_tflite_ops import _classify, audit


def test_classification_follows_opcode_and_activation_dtype():
    assert _classify("QUANTIZE", ["FLOAT32"], ["INT8"]) == "precision_conversion"
    assert _classify("DEQUANTIZE", ["INT8"], ["FLOAT32"]) == "precision_conversion"
    assert _classify("ELU", ["FLOAT32"], ["FLOAT32"]) == "floating_point_numerical"
    assert _classify("CONV_2D", ["INT8"], ["INT8"]) == "quantized_numerical"
    assert _classify("RESHAPE", ["FLOAT32"], ["FLOAT32"]) == "shape_layout_movement"


def test_retained_int8_graphs_have_one_structure_per_backend():
    summaries, rows, errors = audit()
    assert errors == []
    assert len(summaries) == 30
    assert len(rows) == 550
    assert all(row["hash_matches_manifest"] for row in summaries)
    signatures = {}
    for summary in summaries:
        signatures.setdefault(summary["backend"], set()).add(summary["structure_signature"])
    assert {backend: len(found) for backend, found in signatures.items()} == {
        "CCNN-int8": 1,
        "EEGNet-int8": 1,
        "FBtCNN-int8": 1,
    }
    float_ops = Counter(
        (row["backend"], row["opcode"])
        for row in rows
        if row["classification"] == "floating_point_numerical"
    )
    assert float_ops == Counter({
        ("EEGNet-int8", "ELU"): 20,
        ("FBtCNN-int8", "ELU"): 40,
    })
    elu = Counter(row["backend"] for row in rows if row["opcode"] == "ELU")
    assert elu == Counter({"EEGNet-int8": 20, "FBtCNN-int8": 40})
    assert all(row["optimization_modes"] == "PTQ_FULL_INTEGER" for row in summaries)
    assert all(row["other_or_uncertain"] == 0 for row in summaries)
