"""Firmware packaging and staging contract tests (no private paths)."""
from __future__ import annotations

import importlib.util
import re
import sys
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
