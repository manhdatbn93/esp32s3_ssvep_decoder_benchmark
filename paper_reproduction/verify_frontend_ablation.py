"""Verify the published frontend-ablation reference outputs.

This check reads the released tables and their SHA-256 inventory. It does not
load EEG, open a TFLite graph, or repeat inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


DEFAULT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "frontend_ablation"
SOURCE_COMMIT = "3d5854621b1d70785153437a7d59505143e636f4"

SCIENTIFIC_FILES = (
    "EXPERIMENT_MANIFEST.json",
    "RESULT_SUMMARY.csv",
    "SUBJECT_LEVEL_SUMMARY.csv",
    "table_a_feature_fidelity.csv",
    "table_b_output_fidelity.csv",
    "table_c_top1_stability.csv",
    "table_d_subject_level.csv",
    "table_e_archive_parity.csv",
    "flip_trials.csv",
    "per_trial_diagnostics.csv",
    "historical_stage_check.json",
    "ptq_frontend_feature_summary.csv",
    "ptq_overlap.json",
)
ALLOWED_EXTRA = {"README.md", "RELEASE_MANIFEST.sha256.json"}

BACKENDS = ("ccnn_f32", "ccnn_int8", "fbtcnn_f32", "fbtcnn_int8")
EVIDENCE_NAMES = {
    "ccnn_f32": "CCNN-f32",
    "ccnn_int8": "CCNN-int8",
    "fbtcnn_f32": "FBtCNN-f32",
    "fbtcnn_int8": "FBtCNN-int8",
}
EXPECTED_CHANGES = {"ccnn_f32": 0, "ccnn_int8": 1, "fbtcnn_f32": 0, "fbtcnn_int8": 2}
EXPECTED_FLIP_KINDS = {
    ("ccnn_int8", "beneficial"): 1,
    ("fbtcnn_int8", "beneficial"): 1,
    ("fbtcnn_int8", "both_wrong"): 1,
}
TRIALS_PER_BACKEND = 1800
TOTAL_TRIALS = 7200

# Manuscript rounding is three significant figures. The last retained digit is
# matched after rounding; bitwise equality of the stored float is not required.
PUBLISHED_SIG_DIGITS = 3
PUBLISHED = {
    "ccnn_feature_mean_relative_l2": 1.23e-7,
    "fbtcnn_feature_mean_relative_l2": 2.63e-4,
    "fbtcnn_feature_p95_relative_l2": 5.22e-4,
    "fbtcnn_int8_mean_output_relative_l2": 1.23e-2,
    "fbtcnn_int8_max_abs_output": 0.359,
}
INNER_OVERLAPS = (15, 16, 22, 19, 20, 26, 19, 23, 27, 22)
HISTORICAL_COUNTS = {
    "CCNN-int8": (1772, 1733, 1731),
    "EEGNet-int8": (1622, 1766, 1627),
    "FBtCNN-int8": (1786, 1794, 1788),
}


def round_sig(value: float, digits: int = PUBLISHED_SIG_DIGITS) -> float:
    if value == 0:
        return 0.0
    power = digits - 1 - math.floor(math.log10(abs(value)))
    return round(value, power)


def published_equal(value: float, reference: float) -> bool:
    return round_sig(value) == round_sig(reference)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def verify(results_dir: Path) -> list[str]:
    errors: list[str] = []
    root = results_dir.resolve()
    if not root.is_dir():
        return [f"results directory does not exist: {root}"]

    present = {path.name for path in root.iterdir() if path.is_file()}
    missing = [name for name in (*SCIENTIFIC_FILES, "RELEASE_MANIFEST.sha256.json") if name not in present]
    if missing:
        _fail(errors, "missing required files: " + ", ".join(missing))
    unexpected = sorted(present - set(SCIENTIFIC_FILES) - ALLOWED_EXTRA)
    if unexpected:
        _fail(errors, "unexpected files: " + ", ".join(unexpected))
    if missing:
        return errors

    release_path = root / "RELEASE_MANIFEST.sha256.json"
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"release manifest is unreadable: {exc}"]
    if release.get("source_commit") != SOURCE_COMMIT:
        _fail(errors, "release manifest source commit does not match the recorded experiment")
    listed = release.get("files")
    if not isinstance(listed, list) or not listed:
        return errors + ["release manifest has no file inventory"]
    names = [item.get("filename") for item in listed]
    if "RELEASE_MANIFEST.sha256.json" in names:
        _fail(errors, "release manifest includes its own checksum")
    if sorted(names) != sorted(SCIENTIFIC_FILES):
        _fail(errors, "release manifest inventory does not match the scientific outputs")
    for item in listed:
        path = root / str(item.get("filename"))
        digest = _sha256(path)
        size = path.stat().st_size
        if digest != item.get("sha256") or size != item.get("bytes"):
            _fail(errors, f"checksum or size mismatch: {path.name}")

    try:
        manifest = json.loads((root / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [f"experiment manifest is unreadable: {exc}"]
    if manifest.get("source_commit") != SOURCE_COMMIT:
        _fail(errors, "experiment manifest source commit changed")
    if manifest.get("tensorflow") != "2.20.0":
        _fail(errors, "experiment manifest TensorFlow version is not the verified 2.20.0 run")
    toolchain = manifest.get("toolchain") or {}
    if toolchain.get("python") != "3.13.3" or toolchain.get("numpy") != "2.2.6":
        _fail(errors, "experiment manifest Python or NumPy version changed")
    if manifest.get("backends") != list(BACKENDS):
        _fail(errors, "experiment manifest backend list changed")
    command = manifest.get("command") or []
    if any(":" in str(part) and "\\" in str(part) for part in command):
        _fail(errors, "experiment manifest still contains a local absolute path")
    interpreters = manifest.get("interpreter") or {}
    expected_graphs = [f"{backend}/fold{fold:02d}" for backend in BACKENDS for fold in range(1, 11)]
    if sorted(interpreters) != sorted(expected_graphs):
        _fail(errors, "retained TFLite interpreter inventory is incomplete")
    if any(record.get("op_resolver") != "AUTO" for record in interpreters.values()):
        _fail(errors, "an interpreter record does not use OpResolverType.AUTO")
    output_hashes = manifest.get("output_sha256") or {}
    for name, digest in output_hashes.items():
        path = root / name
        if not path.is_file() or _sha256(path) != digest:
            _fail(errors, f"experiment output hash does not match the copied file: {name}")

    try:
        rows = _read_csv(root / "per_trial_diagnostics.csv")
        summary = _read_csv(root / "RESULT_SUMMARY.csv")
        table_a = _read_csv(root / "table_a_feature_fidelity.csv")
        table_b = _read_csv(root / "table_b_output_fidelity.csv")
        table_c = _read_csv(root / "table_c_top1_stability.csv")
        table_e = _read_csv(root / "table_e_archive_parity.csv")
        flips = _read_csv(root / "flip_trials.csv")
        ptq_rows = _read_csv(root / "ptq_frontend_feature_summary.csv")
        stage = json.loads((root / "historical_stage_check.json").read_text(encoding="utf-8"))
        overlap = json.loads((root / "ptq_overlap.json").read_text(encoding="utf-8"))
    except (OSError, csv.Error, json.JSONDecodeError, KeyError) as exc:
        return errors + [f"a result table is malformed: {exc}"]

    if len(rows) != TOTAL_TRIALS:
        _fail(errors, f"trial count is {len(rows)}, expected {TOTAL_TRIALS}")
    identities = [(row["backend"], row["subject"], row["class_id"], row["trial_idx"]) for row in rows]
    if len(identities) != len(set(identities)):
        _fail(errors, "duplicated trial identities")
    counts = Counter(row["backend"] for row in rows)
    if dict(counts) != {backend: TRIALS_PER_BACKEND for backend in BACKENDS}:
        _fail(errors, f"per-backend trial counts are {dict(counts)}")
    changes = Counter(row["backend"] for row in rows if row["prediction_a"] != row["prediction_b"])
    for backend, expected in EXPECTED_CHANGES.items():
        if changes[backend] != expected:
            _fail(errors, f"{backend} prediction changes are {changes[backend]}, expected {expected}")
    if sum(int(row["path_b_matches_archive_argmax"]) for row in rows) != TOTAL_TRIALS:
        _fail(errors, "deployment frontend did not reproduce every archived host top-1")
    if any(row["archive_host_argmax"] != row["archive_argmax_from_stored_logits"] for row in rows):
        _fail(errors, "stored host argmax disagrees with the stored logit argmax")

    summary_by_backend = {row["backend"]: row for row in summary}
    if list(summary_by_backend) != list(BACKENDS) or [row["backend"] for row in table_c] != list(BACKENDS):
        _fail(errors, "summary backend order or identity changed")
    for backend, expected in EXPECTED_CHANGES.items():
        row = summary_by_backend.get(backend)
        if row is None or int(row["n"]) != TRIALS_PER_BACKEND or int(row["flip_count"]) != expected:
            _fail(errors, f"RESULT_SUMMARY.csv disagrees for {backend}")
            continue
        if int(row["harmful"]) != 0:
            _fail(errors, f"{backend} reports a harmful flip")
    kind_counts = Counter((row["backend"], row["flip_kind"]) for row in flips)
    if kind_counts != Counter(EXPECTED_FLIP_KINDS):
        _fail(errors, f"flip kinds are {dict(kind_counts)}")

    feature = {row["model"]: row for row in table_a}
    for model in ("ccnn_f32", "ccnn_int8"):
        if not published_equal(float(feature[model]["mean_relative_l2"]), PUBLISHED["ccnn_feature_mean_relative_l2"]):
            _fail(errors, f"{model} feature mean is outside the published three-significant-figure value")
    for model in ("fbtcnn_f32", "fbtcnn_int8"):
        row = feature[model]
        if not published_equal(float(row["mean_relative_l2"]), PUBLISHED["fbtcnn_feature_mean_relative_l2"]):
            _fail(errors, f"{model} feature mean is outside the published three-significant-figure value")
        if not published_equal(float(row["p95_relative_l2"]), PUBLISHED["fbtcnn_feature_p95_relative_l2"]):
            _fail(errors, f"{model} feature P95 is outside the published three-significant-figure value")
    output = {row["backend"]: row for row in table_b}
    int8 = output["fbtcnn_int8"]
    if not published_equal(float(int8["mean_relative_output_l2"]), PUBLISHED["fbtcnn_int8_mean_output_relative_l2"]):
        _fail(errors, "FBtCNN-int8 mean output deviation is outside the published value")
    if not published_equal(float(int8["max_abs_logit_error"]), PUBLISHED["fbtcnn_int8_max_abs_output"]):
        _fail(errors, "FBtCNN-int8 maximum absolute output difference is outside the published value")

    archive_matches = 0
    for row in table_e:
        if row["backend"] not in BACKENDS or int(row["n"]) != TRIALS_PER_BACKEND:
            _fail(errors, "archive-parity table has an unexpected backend")
        archive_matches += int(row["archived_host_top1_matches"])
    if archive_matches != TOTAL_TRIALS:
        _fail(errors, f"archive parity is {archive_matches}/{TOTAL_TRIALS}")

    join = stage.get("join") or {}
    if not join.get("passed"):
        _fail(errors, "historical stage check did not pass")
    if join.get("label_or_trial_number_mismatches") != 0:
        _fail(errors, "historical stage check reports a label or trial-number mismatch")
    if join.get("replay_rows_without_stage1") != 0 or join.get("stage1_rows_without_replay") != 0:
        _fail(errors, "historical stage check reports a trial-identity mismatch")
    if join.get("export_host_vs_replay_host_prediction_mismatches") != {}:
        _fail(errors, "historical stage check reports an exported-host prediction mismatch")
    rates = stage.get("rates") or {}
    for name, expected in HISTORICAL_COUNTS.items():
        rate = rates.get(name) or {}
        got = (rate.get("stage1_count"), rate.get("stage2_count"), rate.get("chain_count"))
        if got != expected:
            _fail(errors, f"historical {name} counts changed from {expected} to {got}")

    if not isinstance(overlap, list) or [row.get("fold") for row in overlap] != list(range(1, 11)):
        _fail(errors, "PTQ overlap does not contain the ten LOSO folds")
    else:
        for row, expected_inner in zip(overlap, INNER_OVERLAPS):
            if row.get("n_calibration") != 192 or row.get("calibration_test_overlap") != 0:
                _fail(errors, f"PTQ fold {row.get('fold')} calibration selection changed")
            if row.get("calibration_inner_overlap") != expected_inner:
                _fail(errors, f"PTQ fold {row.get('fold')} inner-validation overlap changed")
    if len(ptq_rows) != 20:
        _fail(errors, "PTQ feature summary does not have one row per family and fold")
    for row in ptq_rows:
        if row["family"] not in {"ccnn", "fbtcnn"} or int(row["n"]) != 192 or int(row["calibration_test_overlap"]) != 0:
            _fail(errors, "PTQ feature summary disagrees with the recorded calibration selection")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(DEFAULT_DIR),
        help="frontend ablation output directory (default: the published reference outputs)",
    )
    args = parser.parse_args(argv)
    errors = verify(Path(args.results_dir))
    if errors:
        print(f"FAIL: {len(errors)} frontend ablation check(s) failed")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: frontend ablation reference outputs")
    print(f"  trials {TOTAL_TRIALS}; prediction changes 0, 1, 0, 2; archive parity {TOTAL_TRIALS}/{TOTAL_TRIALS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
