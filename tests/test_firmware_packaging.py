"""Firmware packaging and staging contract tests (no private paths)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
GEN = FIRMWARE / "generated"
REF = ROOT / "artifacts" / "reference"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_staging_module():
    path = ROOT / "scripts" / "generate_firmware_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_firmware_artifacts", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def test_no_active_smoke_fw_dependency():
    text = _read(FIRMWARE / "replay_fw" / "CMakeLists.txt")
    assert "smoke_fw" not in text
    main = _read(FIRMWARE / "replay_fw" / "main" / "CMakeLists.txt")
    assert "smoke_fw" not in main
    assert (FIRMWARE / "components" / "ssvep_dispatch" / "CMakeLists.txt").is_file()
    assert not (FIRMWARE / "smoke_fw").exists()


def test_no_generated_smoke_headers_dependency():
    hits = []
    for path in FIRMWARE.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".cmake", ".txt", ".c", ".h", ".cpp", ".md", ".py"}:
            continue
        data = _read(path)
        if "generated_smoke_headers" in data:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_no_strict_loso_inner_val_dependency():
    hits = []
    for path in (FIRMWARE / "cmake").rglob("*"):
        if path.is_file() and "strict_loso_inner_val" in _read(path):
            hits.append(str(path.relative_to(ROOT)))
    top = _read(FIRMWARE / "replay_fw" / "CMakeLists.txt")
    assert "strict_loso_inner_val" not in top
    assert hits == []


def test_dispatch_component_registers():
    cmake = _read(FIRMWARE / "components" / "ssvep_dispatch" / "CMakeLists.txt")
    assert "idf_component_register" in cmake
    assert "ssvep_dispatch.c" in cmake
    assert (FIRMWARE / "components" / "ssvep_dispatch" / "include" / "ssvep_dispatch.h").is_file()


def test_extra_component_dirs_points_at_firmware_components():
    text = _read(FIRMWARE / "replay_fw" / "CMakeLists.txt")
    assert "SHARED_COMPONENTS_DIR" in text
    assert 'EXTRA_COMPONENT_DIRS' in text
    assert "smoke_fw/components" not in text


def test_replay_tcp_uses_common_identity_not_neural_only_unconditional():
    text = _read(FIRMWARE / "replay_fw" / "main" / "replay_tcp.c")
    assert "backend_identity.h" in text
    assert "ssvep_backend_identity_name" in text
    assert "model_identity_hex16" in text
    # EEGNet identity helpers may exist but must be gated
    if "eegnet_model_identity.h" in text:
        assert "SMOKE_ENABLE_EEGNET" in text


def test_model_identity_lives_in_common():
    assert (FIRMWARE / "components" / "common" / "model_identity.c").is_file()
    assert (FIRMWARE / "components" / "common" / "model_identity.h").is_file()
    assert not (FIRMWARE / "components" / "dl_backends" / "model_identity.c").exists()


def test_missing_staging_cmake_message_present():
    text = _read(FIRMWARE / "cmake" / "export_artifact_root.cmake")
    assert "selected_backend.cmake" in text
    assert "Stage artifacts first" in text


@pytest.mark.parametrize(
    "backend,kind",
    [
        ("cca", "analytical"),
        ("eegnet-f32", "neural"),
        ("eegnet-int8", "neural"),
    ],
)
def test_staging_self_contained(backend, kind, tmp_path):
    mod = _load_staging_module()
    out = tmp_path / "generated"
    meta = mod.stage_reference(backend, 1, out, verify_checksum=True)
    assert meta["kind"] == kind
    assert (out / "selected_backend.cmake").is_file()
    assert (out / "backend_identity.c").is_file()
    assert (out / "vectors").is_dir()
    cmake = _read(out / "selected_backend.cmake")
    assert "SSVEP_IDF_BACKEND" in cmake
    assert "C:/" not in cmake
    assert "C:\\" not in cmake
    assert re.search(r"/Users/|/home/", cmake) is None

    if backend.endswith("int8"):
        f32_dirs = [
            p
            for p in out.iterdir()
            if p.is_dir()
            and p.name.startswith("12jfpm_")
            and p.name.endswith("fold01_test_s1")
            and "_int8" not in p.name
        ]
        assert f32_dirs == []

    if kind == "analytical":
        assert not any(p.suffix == ".tflite" for p in out.rglob("*"))
    else:
        assert any(p.suffix == ".tflite" for p in out.rglob("*"))


def test_staging_does_not_modify_reference_checksums(tmp_path):
    before = (REF / "checksums.sha256").read_text(encoding="utf-8")
    mod = _load_staging_module()
    mod.stage_reference("cca", 1, tmp_path / "generated", verify_checksum=True)
    after = (REF / "checksums.sha256").read_text(encoding="utf-8")
    assert before == after


EEGNET_INT8 = REF / "tflite" / "eegnet_int8" / "fold01" / "eegnet_int8.tflite"
EEGNET_F32 = REF / "tflite" / "eegnet_f32" / "fold01" / "eegnet_float32.tflite"
CCNN_INT8 = REF / "tflite" / "ccnn_int8" / "fold01" / "ccnn_int8.tflite"
FBTCNN_INT8 = REF / "tflite" / "fbtcnn_int8" / "fold01" / "fbtcnn_int8.tflite"
FBTCNN_HEADER = REF / "preprocessing" / "fbtcnn" / "fold01" / "fbtcnn_filter_generated.h"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staging_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_firmware_artifacts.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def _reference_checksums() -> str:
    return (REF / "checksums.sha256").read_text(encoding="utf-8")


def test_generated_eegnet_int8_staging_matches_source_bytes(tmp_path):
    checksums = _reference_checksums()
    source_digest = _sha256(EEGNET_INT8)
    incoming = tmp_path / "incoming" / "custom_model.tflite"
    incoming.parent.mkdir()
    shutil.copyfile(EEGNET_INT8, incoming)
    out = tmp_path / "generated"
    mod = _load_staging_module()
    meta = mod.stage_generated("eegnet-int8", 1, incoming, out)

    staged = out / "12jfpm_eegnet_fold01_test_s1_int8" / "eegnet_int8.tflite"
    staged_bytes = staged.read_bytes()
    assert staged_bytes == incoming.read_bytes() == EEGNET_INT8.read_bytes()
    digest = hashlib.sha256(staged_bytes).hexdigest()
    crc = zlib.crc32(staged_bytes) & 0xFFFFFFFF
    assert digest == source_digest
    assert meta["tflite_sha256"] == digest
    assert meta["tflite_crc32"] == f"0x{crc:08x}"
    assert meta["artifact_set"] == "generated"
    assert meta["backend"] == "eegnet-int8"
    assert meta["idf_backend"] == "eegnet_int8"
    assert meta["fold"] == 1
    assert meta["held_out_subject"] == "s1"
    assert meta["tag"] == "fold01_test_s1"
    assert meta["kind"] == "neural"
    assert meta["files"] == ["eegnet_int8.tflite"]
    assert meta["stage_dir"].replace("\\", "/").endswith("12jfpm_eegnet_fold01_test_s1_int8")
    assert meta["tflite"].replace("\\", "/").endswith(
        "12jfpm_eegnet_fold01_test_s1_int8/eegnet_int8.tflite"
    )

    selected = json.loads((out / "selected_backend.json").read_text(encoding="utf-8"))
    staging = json.loads((out / "staging_meta.json").read_text(encoding="utf-8"))
    assert selected == staging == meta

    cmake = _read(out / "selected_backend.cmake")
    assert 'set(SSVEP_SELECTED_BACKEND "eegnet-int8")' in cmake
    assert 'set(SSVEP_IDF_BACKEND "eegnet_int8")' in cmake
    assert 'set(SSVEP_SELECTED_FOLD "1")' in cmake
    assert 'set(SSVEP_SELECTED_SUBJECT "s1")' in cmake
    assert 'set(SSVEP_SELECTED_TAG "fold01_test_s1")' in cmake
    assert 'set(SSVEP_BACKEND_KIND "neural")' in cmake
    assert 'set(SSVEP_STAGE_SUBDIR "12jfpm_eegnet_fold01_test_s1_int8")' in cmake
    assert "C:/" not in cmake and "C:\\" not in cmake

    identity = _read(out / "backend_identity.c")
    assert (out / "backend_identity.h").is_file()
    assert digest in identity
    assert f"return {crc}u;" in identity
    assert 'static const char kName[] = "eegnet-int8";' in identity
    assert 'static const char kSubject[] = "s1";' in identity
    assert ", ".join(str(b) for b in staged_bytes[:16]) in identity
    assert "SMOKE_EEGNET_N_VECTORS 0" in _read(out / "vectors" / "smoke_vectors_eegnet.h")
    assert not (out / "eegnet-int8").exists()
    assert _reference_checksums() == checksums
    assert _sha256(EEGNET_INT8) == source_digest


def test_generated_fbtcnn_requires_existing_filter_header(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    model = incoming / "fbtcnn_int8.tflite"
    shutil.copyfile(FBTCNN_INT8, model)
    out = tmp_path / "generated"
    mod = _load_staging_module()
    with pytest.raises(SystemExit, match="fbtcnn_filter_generated.h"):
        mod.stage_generated("fbtcnn-int8", 1, model, out)
    assert not out.exists()

    shutil.copyfile(FBTCNN_HEADER, incoming / "fbtcnn_filter_generated.h")
    meta = mod.stage_generated("fbtcnn-int8", 1, model, out)
    stage = out / "12jfpm_fbtcnn_fold01_test_s1_int8"
    staged = stage / "fbtcnn_int8.tflite"
    assert staged.read_bytes() == model.read_bytes()
    assert (stage / "fbtcnn_filter_generated.h").read_bytes() == FBTCNN_HEADER.read_bytes()
    assert meta["files"] == ["fbtcnn_int8.tflite", "fbtcnn_filter_generated.h"]
    assert meta["tflite_sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()
    assert meta["tflite_sha256"] != hashlib.sha256(
        staged.read_bytes() + (stage / "fbtcnn_filter_generated.h").read_bytes()
    ).hexdigest()


def test_generated_rejects_invalid_inputs(tmp_path):
    checksums = _reference_checksums()
    int8_digest = _sha256(EEGNET_INT8)
    f32_digest = _sha256(EEGNET_F32)
    mod = _load_staging_module()
    out = tmp_path / "generated"
    missing = _staging_cmd(
        ["--backend", "eegnet-int8", "--fold", "1", "--artifact-set", "generated", "--output", str(out)]
    )
    assert missing.returncode != 0
    assert "--tflite is required" in missing.stderr + missing.stdout
    assert not out.exists()

    absent = tmp_path / "missing.tflite"
    invalid_path = _staging_cmd(
        [
            "--backend", "eegnet-int8", "--fold", "1", "--artifact-set", "generated",
            "--tflite", str(absent), "--output", str(out),
        ]
    )
    assert invalid_path.returncode != 0
    assert "not found" in (invalid_path.stderr + invalid_path.stdout)
    assert not out.exists()

    for fold in ("0", "11"):
        invalid_fold = _staging_cmd(
            [
                "--backend", "eegnet-int8", "--fold", fold, "--artifact-set", "generated",
                "--tflite", str(EEGNET_INT8), "--output", str(out),
            ]
        )
        assert invalid_fold.returncode != 0
        assert "1..10" in invalid_fold.stderr + invalid_fold.stdout
        assert not out.exists()

    garbage = tmp_path / "garbage.tflite"
    garbage.write_bytes(b"not a tflite model")
    with pytest.raises(SystemExit, match="Incompatible TFLite model"):
        mod.stage_generated("eegnet-int8", 1, garbage, out)
    assert not out.exists()

    with pytest.raises(SystemExit, match="precision"):
        mod.stage_generated("eegnet-int8", 1, EEGNET_F32, out)
    assert not out.exists()

    with pytest.raises(SystemExit, match="input shape"):
        mod.stage_generated("eegnet-int8", 1, CCNN_INT8, out)
    assert not out.exists()

    with pytest.raises(SystemExit, match="Unsupported"):
        mod.stage_generated("ssvepformer", 1, EEGNET_INT8, out)
    with pytest.raises(SystemExit, match="analytical"):
        mod.stage_generated("cca", 1, EEGNET_INT8, out)
    assert not out.exists()
    assert _reference_checksums() == checksums
    assert _sha256(EEGNET_INT8) == int8_digest
    assert _sha256(EEGNET_F32) == f32_digest


def test_generated_dry_run_and_reference_dry_run_create_no_files(tmp_path):
    checksums = _reference_checksums()
    source = _sha256(EEGNET_INT8)
    generated_before = list(GEN.rglob("*")) if GEN.exists() else None
    out = tmp_path / "generated"
    generated = _staging_cmd(
        [
            "--backend", "eegnet-int8", "--fold", "1", "--artifact-set", "generated",
            "--tflite", str(EEGNET_INT8), "--output", str(out), "--dry-run",
        ]
    )
    assert generated.returncode == 0, generated.stderr + generated.stdout
    assert "12jfpm_eegnet_fold01_test_s1_int8" in generated.stdout
    assert "eegnet_int8.tflite" in generated.stdout
    assert "held_out=s1" in generated.stdout
    assert not out.exists()

    reference = _staging_cmd(
        [
            "--backend", "eegnet-int8", "--fold", "1", "--artifact-set", "reference",
            "--output", str(tmp_path / "reference"), "--dry-run",
        ]
    )
    assert reference.returncode == 0, reference.stderr + reference.stdout
    assert "OK reference eegnet-int8" in reference.stdout
    assert "held_out=s1" in reference.stdout
    assert not (tmp_path / "reference").exists()
    assert _reference_checksums() == checksums
    assert _sha256(EEGNET_INT8) == source
    assert (list(GEN.rglob("*")) if GEN.exists() else None) == generated_before
