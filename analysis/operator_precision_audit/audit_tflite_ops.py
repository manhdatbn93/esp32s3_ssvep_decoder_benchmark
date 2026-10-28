"""Read-only FlatBuffer audit of the 30 retained int8 TFLite graphs.

The classification uses operator opcodes and tensor types in the model file.
It does not invoke an interpreter, load EEG, or rewrite any artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_schema
from tensorflow.lite.python import schema_py_generated as schema


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "artifacts" / "reference"
TARGETS = (
    ("CCNN-int8", "ccnn", "artifacts/reference/tflite/ccnn_int8/fold{fold:02d}/ccnn_int8.tflite"),
    ("EEGNet-int8", "eegnet", "artifacts/reference/tflite/eegnet_int8/fold{fold:02d}/eegnet_int8.tflite"),
    ("FBtCNN-int8", "fbtcnn", "artifacts/reference/tflite/fbtcnn_int8/fold{fold:02d}/fbtcnn_int8.tflite"),
)

CONVERSION_OPS = {"QUANTIZE", "DEQUANTIZE", "CAST"}
MOVEMENT_OPS = {
    "RESHAPE", "TRANSPOSE", "SQUEEZE", "EXPAND_DIMS", "PAD", "STRIDED_SLICE",
    "SLICE", "GATHER", "CONCATENATION", "PACK", "UNPACK",
}
NUMERICAL_OPS = {
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D",
    "MAX_POOL_2D", "MEAN", "SUM", "MUL", "ADD", "SUB", "DIV", "SQRT", "ELU",
    "RELU", "RELU6", "LOGISTIC", "TANH", "SOFTMAX", "BATCH_MATMUL",
}
INTEGER_ACTIVATIONS = {"INT8", "UINT8"}
OPTION_CLASSES = {
    schema.BuiltinOptions.Conv2DOptions: schema.Conv2DOptions,
    schema.BuiltinOptions.DepthwiseConv2DOptions: schema.DepthwiseConv2DOptions,
    schema.BuiltinOptions.Pool2DOptions: schema.Pool2DOptions,
    schema.BuiltinOptions.FullyConnectedOptions: schema.FullyConnectedOptions,
}

def _names(cls) -> dict[int, str]:
    return {getattr(cls, name): name for name in dir(cls) if name.isupper() or name[:1].isupper()}


TENSOR_NAMES = _names(schema.TensorType)
OPERATOR_NAMES = _names(schema.BuiltinOperator)
ACTIVATION_NAMES = _names(schema.ActivationFunctionType)
MODEL_TYPE_NAMES = _names(conversion_schema.ModelType)
OPTIMIZATION_NAMES = _names(conversion_schema.ModelOptimizationMode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enum(table: dict[int, str], value: int) -> str:
    return table.get(value, f"UNKNOWN_{value}")


def _buffer_bytes(model, index: int) -> bytes:
    if index < 0 or index >= model.BuffersLength():
        return b""
    buffer = model.Buffers(index)
    if buffer is None or buffer.DataLength() == 0:
        return b""
    return bytes(buffer.Data(i) for i in range(buffer.DataLength()))


def _tensor_name(tensor) -> str:
    raw = tensor.Name()
    return raw.decode("utf-8", errors="replace") if raw else ""


def _quant_scales(tensor) -> int:
    quantization = tensor.Quantization()
    if quantization is None:
        return 0
    return int(quantization.ScaleLength())


def _fused_activation(operator) -> str:
    option_type = operator.BuiltinOptionsType()
    option_cls = OPTION_CLASSES.get(option_type)
    table = operator.BuiltinOptions()
    if option_cls is None or table is None:
        return ""
    options = option_cls()
    options.Init(table.Bytes, table.Pos)
    return _enum(ACTIVATION_NAMES, options.FusedActivationFunction())


def _conversion_metadata(model) -> dict[str, str]:
    found = {"optimization_modes": "", "model_type": "", "tensorflow_version": "", "select_tf_ops": ""}
    for index in range(model.MetadataLength()):
        entry = model.Metadata(index)
        name = entry.Name().decode("utf-8", errors="replace") if entry.Name() else ""
        if name != "CONVERSION_METADATA":
            continue
        payload = _buffer_bytes(model, entry.Buffer())
        metadata = conversion_schema.ConversionMetadata.GetRootAs(payload, 0)
        environment = metadata.Environment()
        options = metadata.Options()
        if environment is not None:
            version = environment.TensorflowVersion()
            found["tensorflow_version"] = version.decode("utf-8", errors="replace") if version else ""
            found["model_type"] = _enum(MODEL_TYPE_NAMES, environment.ModelType())
        if options is not None:
            modes = [
                _enum(OPTIMIZATION_NAMES, options.ModelOptimizationModes(i))
                for i in range(options.ModelOptimizationModesLength())
            ]
            found["optimization_modes"] = "|".join(modes)
            found["select_tf_ops"] = str(bool(options.EnableSelectTfOps())).lower()
    return found


def _runtime_version(model) -> str:
    for index in range(model.MetadataLength()):
        entry = model.Metadata(index)
        name = entry.Name().decode("utf-8", errors="replace") if entry.Name() else ""
        if name == "min_runtime_version":
            return _buffer_bytes(model, entry.Buffer()).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return ""


def _classify(opcode: str, activation_inputs: list[str], activation_outputs: list[str]) -> str:
    if opcode in CONVERSION_OPS:
        return "precision_conversion"
    if opcode in MOVEMENT_OPS:
        return "shape_layout_movement"
    if opcode not in NUMERICAL_OPS:
        return "other_or_uncertain"
    inputs = set(activation_inputs)
    outputs = set(activation_outputs)
    if inputs and outputs and inputs <= INTEGER_ACTIVATIONS and outputs <= INTEGER_ACTIVATIONS:
        return "quantized_numerical"
    if inputs == {"FLOAT32"} and outputs == {"FLOAT32"}:
        return "floating_point_numerical"
    return "other_or_uncertain"


def _manifest_hashes() -> dict[str, str]:
    entries = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    return {
        entry["relative_path"].replace("\\", "/"): entry["sha256"]
        for entry in entries
        if entry.get("artifact_type") == "tflite" and entry.get("precision") == "int8"
        and entry.get("model") in {"ccnn", "eegnet", "fbtcnn"}
    }


def _opcode_name(model, operator) -> tuple[str, int]:
    code = model.OperatorCodes(operator.OpcodeIndex())
    builtin = code.BuiltinCode() or code.DeprecatedBuiltinCode()
    name = _enum(OPERATOR_NAMES, builtin)
    custom = code.CustomCode()
    if custom:
        name = "CUSTOM:" + custom.decode("utf-8", errors="replace")
    return name, int(code.Version())


def inspect_model(path: Path, backend: str, fold: int, relative: str, expected_hash: str) -> tuple[dict, list[dict]]:
    data = path.read_bytes()
    model = schema.Model.GetRootAsModel(data, 0)
    actual_hash = hashlib.sha256(data).hexdigest()
    metadata = _conversion_metadata(model)
    rows = []
    produced = set()
    for subgraph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_index)
        graph_inputs = {subgraph.Inputs(i) for i in range(subgraph.InputsLength())}
        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            for slot in range(operator.OutputsLength()):
                produced.add(operator.Outputs(slot))
        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            opcode, version = _opcode_name(model, operator)
            input_parts = []
            output_parts = []
            constant_types = []
            activation_inputs = []
            activation_outputs = []
            input_scales = []
            output_scales = []
            for slot in range(operator.InputsLength()):
                index = operator.Inputs(slot)
                if index < 0:
                    input_parts.append("-1:optional")
                    continue
                tensor = subgraph.Tensors(index)
                dtype = _enum(TENSOR_NAMES, tensor.Type())
                constant = bool(_buffer_bytes(model, tensor.Buffer())) or (index not in graph_inputs and index not in produced)
                role = "constant" if constant else "activation"
                input_parts.append(f"{index}:{dtype}:{role}:q{_quant_scales(tensor)}")
                input_scales.append(str(_quant_scales(tensor)))
                if constant:
                    constant_types.append(dtype)
                else:
                    activation_inputs.append(dtype)
            for slot in range(operator.OutputsLength()):
                index = operator.Outputs(slot)
                tensor = subgraph.Tensors(index)
                dtype = _enum(TENSOR_NAMES, tensor.Type())
                output_parts.append(f"{index}:{dtype}:q{_quant_scales(tensor)}")
                output_scales.append(str(_quant_scales(tensor)))
                activation_outputs.append(dtype)
            rows.append({
                "backend": backend,
                "fold": fold,
                "relative_path": relative,
                "sha256": actual_hash,
                "subgraph": subgraph_index,
                "operator_index": operator_index,
                "opcode": opcode,
                "version": version,
                "classification": _classify(opcode, activation_inputs, activation_outputs),
                "inputs": ";".join(input_parts),
                "outputs": ";".join(output_parts),
                "constant_dtypes": "|".join(constant_types),
                "activation_input_dtypes": "|".join(activation_inputs),
                "activation_output_dtypes": "|".join(activation_outputs),
                "fused_activation": _fused_activation(operator),
                "input_quant_scale_counts": "|".join(input_scales),
                "output_quant_scale_counts": "|".join(output_scales),
            })
    first = model.Subgraphs(0)
    input_tensor = first.Tensors(first.Inputs(0))
    output_tensor = first.Tensors(first.Outputs(0))
    counts = Counter(row["classification"] for row in rows)
    opcodes = Counter(row["opcode"] for row in rows)
    summary = {
        "backend": backend,
        "fold": fold,
        "relative_path": relative,
        "bytes": len(data),
        "sha256": actual_hash,
        "manifest_sha256": expected_hash,
        "hash_matches_manifest": actual_hash == expected_hash,
        "file_identifier": data[4:8].decode("ascii", errors="replace"),
        "schema_version": int(model.Version()),
        "subgraphs": int(model.SubgraphsLength()),
        "operator_count": len(rows),
        "input_dtype": _enum(TENSOR_NAMES, input_tensor.Type()),
        "output_dtype": _enum(TENSOR_NAMES, output_tensor.Type()),
        "quantized_numerical": counts["quantized_numerical"],
        "floating_point_numerical": counts["floating_point_numerical"],
        "precision_conversion": counts["precision_conversion"],
        "shape_layout_movement": counts["shape_layout_movement"],
        "other_or_uncertain": counts["other_or_uncertain"],
        "quantize_count": opcodes["QUANTIZE"],
        "dequantize_count": opcodes["DEQUANTIZE"],
        "elu_count": opcodes["ELU"],
        "float_computational_opcodes": "|".join(
            f"{row['opcode']}#{row['operator_index']}"
            for row in rows if row["classification"] == "floating_point_numerical"
        ),
        "fused_activations": "|".join(
            f"{row['opcode']}#{row['operator_index']}:{row['fused_activation']}"
            for row in rows if row["fused_activation"] not in {"", "NONE"}
        ),
        "optimization_modes": metadata["optimization_modes"],
        "conversion_model_type": metadata["model_type"],
        "conversion_tensorflow_version": metadata["tensorflow_version"],
        "select_tf_ops": metadata["select_tf_ops"],
        "min_runtime_version": _runtime_version(model),
        "structure_signature": " ".join(
            f"{row['opcode']}:{row['version']}:{row['classification']}:{row['activation_input_dtypes']}->{row['activation_output_dtypes']}:{row['fused_activation']}"
            for row in rows
        ),
    }
    return summary, rows


def expected_paths() -> list[tuple[str, int, str]]:
    found = []
    for backend, _model, pattern in TARGETS:
        for fold in range(1, 11):
            found.append((backend, fold, pattern.format(fold=fold)))
    return found


def audit(repo: Path = ROOT) -> tuple[list[dict], list[dict], list[str]]:
    manifest = _manifest_hashes()
    summaries = []
    rows = []
    errors = []
    expected = expected_paths()
    expected_set = {relative for _backend, _fold, relative in expected}
    if set(manifest) != expected_set:
        errors.append("reference manifest int8 inventory does not match the 30 expected graphs")
    present = {
        path.relative_to(repo).as_posix()
        for path in (repo / "artifacts" / "reference" / "tflite").glob("*_int8/fold*/ *.tflite".replace(" ", ""))
    }
    if present != expected_set:
        errors.append(
            "on-disk int8 graphs differ from the expected 30: "
            + ", ".join(sorted(present.symmetric_difference(expected_set)))
        )
    for backend, fold, relative in expected:
        path = repo / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        summary, operator_rows = inspect_model(path, backend, fold, relative, manifest.get(relative, ""))
        summaries.append(summary)
        rows.extend(operator_rows)
        if not summary["hash_matches_manifest"]:
            errors.append(f"SHA-256 mismatch: {relative}")
        if summary["subgraphs"] != 1:
            errors.append(f"unexpected subgraph count: {relative}")
        if summary["other_or_uncertain"]:
            errors.append(f"unclassified operators: {relative}")
    return summaries, rows, errors


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "operator_precision_audit"))
    args = parser.parse_args(argv)
    summaries, rows, errors = audit(Path(args.repo))
    out = Path(args.out)
    summary_fields = [
        "backend", "fold", "relative_path", "bytes", "sha256", "manifest_sha256",
        "hash_matches_manifest", "file_identifier", "schema_version", "subgraphs",
        "operator_count", "input_dtype", "output_dtype", "quantized_numerical",
        "floating_point_numerical", "precision_conversion", "shape_layout_movement",
        "other_or_uncertain", "quantize_count", "dequantize_count", "elu_count",
        "float_computational_opcodes", "fused_activations", "optimization_modes",
        "conversion_model_type", "conversion_tensorflow_version", "select_tf_ops",
        "min_runtime_version",
    ]
    inventory_fields = [
        "backend", "fold", "relative_path", "sha256", "subgraph", "operator_index",
        "opcode", "version", "classification", "inputs", "outputs", "constant_dtypes",
        "activation_input_dtypes", "activation_output_dtypes", "fused_activation",
        "input_quant_scale_counts", "output_quant_scale_counts",
    ]
    _write_csv(out / "operator_precision_summary.csv", summaries, summary_fields)
    _write_csv(out / "operator_inventory_30_models.csv", rows, inventory_fields)
    by_backend = {}
    for summary in summaries:
        by_backend.setdefault(summary["backend"], set()).add(summary["structure_signature"])
    for backend, signatures in by_backend.items():
        if len(signatures) != 1:
            errors.append(f"fold operator structure differs for {backend}")
    if errors:
        print(f"FAIL: {len(errors)} operator-precision audit check(s) failed")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"PASS: {len(summaries)} int8 graphs, {len(rows)} operators")
    for backend, signatures in by_backend.items():
        sample = next(row for row in summaries if row["backend"] == backend and row["fold"] == 1)
        print(
            f"  {backend}: ops {sample['operator_count']}, "
            f"quantized {sample['quantized_numerical']}, "
            f"float {sample['floating_point_numerical']} ({sample['float_computational_opcodes'] or 'none'}), "
            f"Q/DQ {sample['quantize_count']}/{sample['dequantize_count']}, "
            f"movement {sample['shape_layout_movement']}, "
            f"ELU {sample['elu_count']}, folds identical"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
