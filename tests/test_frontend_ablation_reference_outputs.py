"""Checks for the published frontend-ablation reference outputs.

These tests read the released tables. They do not load EEG or run TensorFlow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from paper_reproduction.verify_frontend_ablation import (
    SCIENTIFIC_FILES,
    SOURCE_COMMIT,
    verify,
)


ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "paper_reproduction" / "reference_outputs" / "frontend_ablation"


def test_published_outputs_pass():
    errors = verify(CANON)
    assert errors == []


def test_release_inventory_matches_scientific_files():
    import json
    release = json.loads((CANON / "RELEASE_MANIFEST.sha256.json").read_text(encoding="utf-8"))
    names = [item["filename"] for item in release["files"]]
    assert release["source_commit"] == SOURCE_COMMIT
    assert "RELEASE_MANIFEST.sha256.json" not in names
    assert sorted(names) == sorted(SCIENTIFIC_FILES)
    assert all((CANON / name).is_file() for name in SCIENTIFIC_FILES)


def test_missing_file_fails(tmp_path):
    copied = tmp_path / "results"
    shutil.copytree(CANON, copied)
    (copied / "flip_trials.csv").unlink()
    errors = verify(copied)
    assert errors
    assert any("missing" in error for error in errors)


def test_corrupted_file_fails(tmp_path):
    copied = tmp_path / "results"
    shutil.copytree(CANON, copied)
    target = copied / "RESULT_SUMMARY.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    errors = verify(copied)
    assert errors
    assert any("checksum" in error or "hash" in error for error in errors)


def test_malformed_json_fails(tmp_path):
    copied = tmp_path / "results"
    shutil.copytree(CANON, copied)
    (copied / "ptq_overlap.json").write_text("{", encoding="utf-8")
    errors = verify(copied)
    assert errors
    assert any("checksum" in error or "malformed" in error or "unreadable" in error for error in errors)
