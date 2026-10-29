"""Offline tests for scripts/setup_dataset.py. No network and no real 12JFPM files."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_dataset.py"
PINNED_COMMIT = "2d4e281a430b4676f75c9270db839e963f172da5"


def _load():
    spec = importlib.util.spec_from_file_location("setup_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def setup_mod():
    return _load()


@pytest.fixture(scope="module")
def valid_mat_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("mat-template") / "template.mat"
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((12, 8, 1114, 15)).astype(np.float32)
    sio.savemat(path, {"eeg": eeg})
    return path.read_bytes()


def _write_subjects(directory: Path, payload: bytes, subjects: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        (directory / f"{subject}.mat").write_bytes(payload)


def _valid_processed(path: Path, *, swap_subjects: bool = False, non_finite: bool = False, drop_x: bool = False) -> None:
    module = _load()
    subjects = module.expected_subject_ids()
    labels = module.expected_labels()
    if swap_subjects:
        subjects = subjects[180:] + subjects[:180]
    x = np.zeros((1800, 512, 8), dtype=np.float32)
    x[0, 0, 0] = 1.0
    if non_finite:
        x[1, 0, 0] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y": np.asarray(labels, dtype=np.int64),
        "subject": np.asarray(subjects, dtype=object),
    }
    if not drop_x:
        payload["X"] = x
    np.savez_compressed(path, **payload)


def _reject_download(*_args: object) -> None:
    raise AssertionError("download")


def _protected_bytes() -> dict[Path, bytes]:
    paths = [
        ROOT / "artifacts" / "reference" / "checksums.sha256",
        ROOT / "src" / "data" / "preprocess_12jfpm.py",
        ROOT / "src" / "data" / "parse_12jfpm.py",
        ROOT / "configs" / "protocol" / "preprocess_12jfpm.yaml",
        ROOT / "README.md",
    ]
    return {path: path.read_bytes() for path in paths}


def test_subject_urls_are_commit_pinned_https(setup_mod):
    url = setup_mod.subject_url("s3")
    assert url == (
        "https://raw.githubusercontent.com/mnakanishi/12JFPM_SSVEP/"
        f"{PINNED_COMMIT}/data/s3.mat"
    )
    text = SCRIPT.read_text(encoding="utf-8")
    assert "CERT_NONE" not in text
    assert "_create_unverified_context" not in text
    assert "verify=False" not in text


def test_cli_help_does_not_download():
    before = _protected_bytes()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "12JFPM" in completed.stdout
    assert _protected_bytes() == before


def test_all_ten_subjects_already_valid(tmp_path, setup_mod, valid_mat_bytes):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    _valid_processed(tmp_path / "data" / "processed" / "12JFPM" / "processed.npz")

    def _forbid(url: str, dest: Path) -> None:
        raise AssertionError(f"unexpected download {url}")

    code = setup_mod.setup_dataset(tmp_path, downloader=_forbid)
    assert code == 0


def test_all_raw_files_missing_downloads_then_preprocesses(tmp_path, setup_mod, valid_mat_bytes, capsys):
    seen: list[str] = []

    def _download(url: str, dest: Path) -> None:
        seen.append(url)
        dest.write_bytes(valid_mat_bytes)

    code = setup_mod.setup_dataset(tmp_path, downloader=_download)
    output = capsys.readouterr().out
    assert code == 0, output
    assert seen == [setup_mod.subject_url(subject) for subject in setup_mod.SUBJECTS]
    assert "DOWNLOAD / PASS" in output
    assert "Preprocessing ... PASS" in output
    assert (tmp_path / "data" / "processed" / "12JFPM" / "processed.npz").is_file()
    assert not (tmp_path / "data" / "processed" / "12JFPM" / ".preprocess_staging").exists()


def test_some_subjects_missing(tmp_path, setup_mod, valid_mat_bytes):
    nested = tmp_path / "data" / "raw" / "12JFPM" / "extracted" / "12JFPM_SSVEP-master" / "data"
    present = ["s1", "s2", "s4"]
    _write_subjects(nested, valid_mat_bytes, present)
    decoy = tmp_path / "data" / "raw" / "12JFPM" / "s1.mat"
    decoy.write_bytes(b"do-not-use")
    _valid_processed(tmp_path / "data" / "processed" / "12JFPM" / "processed.npz")
    processed_before = (tmp_path / "data" / "processed" / "12JFPM" / "processed.npz").read_bytes()

    def _download(url: str, dest: Path) -> None:
        dest.write_bytes(valid_mat_bytes)

    code = setup_mod.setup_dataset(tmp_path, downloader=_download)
    assert code == 0
    for subject in setup_mod.SUBJECTS:
        assert (nested / f"{subject}.mat").is_file()
    assert decoy.read_bytes() == b"do-not-use"
    assert (tmp_path / "data" / "processed" / "12JFPM" / "processed.npz").read_bytes() == processed_before


def test_invalid_existing_mat_is_not_overwritten(tmp_path, setup_mod, valid_mat_bytes):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, [subject for subject in setup_mod.SUBJECTS if subject != "s3"])
    invalid = raw / "s3.mat"
    invalid.write_bytes(b"not-a-mat")
    before = invalid.read_bytes()

    def _download(url: str, dest: Path) -> None:
        raise AssertionError("invalid file must not be replaced")

    code = setup_mod.setup_dataset(tmp_path, downloader=_download)
    assert code != 0
    assert invalid.read_bytes() == before
    assert not (raw / "s3.mat.partial").exists()


def test_interrupted_download_removes_partial(tmp_path, setup_mod, valid_mat_bytes):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, ["s1"])
    kept = (raw / "s1.mat").read_bytes()

    def _download(url: str, dest: Path) -> None:
        dest.write_bytes(b"partial-body")
        raise TimeoutError("connection dropped")

    code = setup_mod.setup_dataset(tmp_path, downloader=_download)
    assert code != 0
    assert (raw / "s1.mat").read_bytes() == kept
    assert not (raw / "s2.mat").exists()
    assert not (raw / "s2.mat.partial").exists()


def test_invalid_download_is_not_installed(tmp_path, setup_mod):
    def _download(url: str, dest: Path) -> None:
        dest.write_bytes(b"this is not a matlab file")

    code = setup_mod.setup_dataset(tmp_path, downloader=_download)
    raw = tmp_path / "data" / "raw" / "12JFPM"
    assert code != 0
    assert not (raw / "s1.mat").exists()
    assert not (raw / "s1.mat.partial").exists()
    assert not (tmp_path / "data" / "processed" / "12JFPM" / "processed.npz").exists()


def test_existing_valid_processed_is_reused(tmp_path, setup_mod, valid_mat_bytes, capsys):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    npz = tmp_path / "data" / "processed" / "12JFPM" / "processed.npz"
    _valid_processed(npz)
    before = npz.read_bytes()

    def _preprocess(*_args, **_kwargs):
        raise AssertionError("preprocessing must not run")

    setup_mod_runner = setup_mod
    original = setup_mod_runner._run_canonical_preprocess
    setup_mod_runner._run_canonical_preprocess = _preprocess
    try:
        code = setup_mod_runner.setup_dataset(tmp_path, downloader=lambda *_args: None)
    finally:
        setup_mod_runner._run_canonical_preprocess = original
    output = capsys.readouterr().out
    assert code == 0
    assert "SKIP (verified)" in output
    assert npz.read_bytes() == before


def test_missing_processed_runs_canonical_preprocessing(tmp_path, setup_mod, valid_mat_bytes, capsys):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    code = setup_mod.setup_dataset(tmp_path, downloader=_reject_download)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "Preprocessing ... PASS" in output
    npz = tmp_path / "data" / "processed" / "12JFPM" / "processed.npz"
    summary = setup_mod.validate_processed(npz)
    assert summary == {"n_trials": 1800, "n_subjects": 10, "n_classes": 12}
    meta = json.loads((npz.parent / "meta.json").read_text(encoding="utf-8"))
    assert Path(meta["output_file"]).resolve() == npz.resolve()
    assert meta["n_samples"] == 1800
    assert meta["shape_X"] == [1800, 512, 8]
    assert meta["dtype_X"] == "float32"
    assert meta["onset_sample_0"] == 38
    assert meta["window_length"] == 512
    assert meta["normalize"] == "channel_zscore"
    assert meta["subjects"] == [f"s{i}" for i in range(1, 11)]


def test_invalid_processed_is_not_overwritten(tmp_path, setup_mod, valid_mat_bytes):
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    npz = tmp_path / "data" / "processed" / "12JFPM" / "processed.npz"
    _valid_processed(npz, non_finite=True)
    before = npz.read_bytes()
    code = setup_mod.setup_dataset(tmp_path, downloader=lambda *_args: None)
    assert code != 0
    assert npz.read_bytes() == before

    _valid_processed(npz, drop_x=True)
    before = npz.read_bytes()
    code = setup_mod.setup_dataset(tmp_path, downloader=lambda *_args: None)
    assert code != 0
    assert npz.read_bytes() == before


def test_incorrect_ordering_fails_validation(tmp_path, setup_mod, valid_mat_bytes):
    npz = tmp_path / "processed.npz"
    _valid_processed(npz, swap_subjects=True)
    with pytest.raises(ValueError, match="subject order"):
        setup_mod.validate_processed(npz)

    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    installed = tmp_path / "data" / "processed" / "12JFPM" / "processed.npz"
    _valid_processed(installed, swap_subjects=True)
    before = installed.read_bytes()
    code = setup_mod.setup_dataset(tmp_path, downloader=lambda *_args: None)
    assert code != 0
    assert installed.read_bytes() == before


def test_setup_does_not_modify_protected_project_files(tmp_path, setup_mod, valid_mat_bytes):
    before = _protected_bytes()
    raw = tmp_path / "data" / "raw" / "12JFPM"
    _write_subjects(raw, valid_mat_bytes, setup_mod.SUBJECTS)
    _valid_processed(tmp_path / "data" / "processed" / "12JFPM" / "processed.npz")
    code = setup_mod.setup_dataset(tmp_path, downloader=_reject_download)
    assert code == 0
    assert _protected_bytes() == before
