"""Git LFS pointer files must fail before checksum comparison."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts.reference import _verify_entry


def test_lfs_pointer_raises_actionable_error(tmp_path: Path):
    pointer = tmp_path / "eegnet_float32.tflite"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        b"size 1234\n"
    )
    entry = {
        "artifact_id": "eegnet_f32_tflite_fold01",
        "fold": 1,
        "held_out_subject": "s1",
        "relative_path": pointer.name,
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="git lfs pull") as exc:
        _verify_entry(entry, tmp_path, verify_checksum=True)
    message = str(exc.value)
    assert "eegnet_f32_tflite_fold01" in message
    assert "Git LFS pointer" in message
