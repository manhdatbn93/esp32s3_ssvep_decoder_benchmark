#!/usr/bin/env python3
"""Prepare the local 12JFPM dataset: raw subject files, preprocessing, and validation.

    python scripts/setup_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.parse_12jfpm import (  # noqa: E402
    N_CHANNELS,
    N_CLASSES,
    N_SAMPLES_RAW,
    N_SUBJECTS,
    N_TRIALS_PER_CLASS,
    SAMPLE_RATE_HZ,
)

UPSTREAM_COMMIT = "2d4e281a430b4676f75c9270db839e963f172da5"
UPSTREAM_RAW_URL = (
    "https://raw.githubusercontent.com/mnakanishi/12JFPM_SSVEP/"
    f"{UPSTREAM_COMMIT}/data/{{subject}}.mat"
)
SUBJECTS = [f"s{i}" for i in range(1, N_SUBJECTS + 1)]
RAW_EEG_SHAPE = (N_CLASSES, N_CHANNELS, N_SAMPLES_RAW, N_TRIALS_PER_CLASS)
PROCESSED_SAMPLES = 512
TRIALS_PER_SUBJECT = N_CLASSES * N_TRIALS_PER_CLASS
N_PROCESSED_TRIALS = N_SUBJECTS * TRIALS_PER_SUBJECT
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_12jfpm.py"
PREPARE_CONFIG = REPO_ROOT / "configs" / "protocol" / "preprocess_12jfpm.yaml"

Downloader = Callable[[str, Path], None]


def subject_url(subject: str) -> str:
    return UPSTREAM_RAW_URL.format(subject=subject)


def default_raw_dir(repo: Path) -> Path:
    return repo / "data" / "raw" / "12JFPM"


def processed_npz_path(repo: Path) -> Path:
    return repo / "data" / "processed" / "12JFPM" / "processed.npz"


def locate_raw_directory(repo: Path) -> Path:
    """Match src.data.parse_12jfpm discovery, including the legacy nested layout."""
    nested = default_raw_dir(repo) / "extracted" / "12JFPM_SSVEP-master" / "data"
    if nested.is_dir() and any(nested.glob("s*.mat")):
        return nested
    root = default_raw_dir(repo)
    if root.is_dir():
        found = sorted(root.rglob("s1.mat"))
        if found:
            return found[0].parent
    return root


def _rel(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_raw_mat(path: Path) -> None:
    """Check one subject file. The sampling rate is the dataset contract, not a MAT field."""
    if float(SAMPLE_RATE_HZ) != 256.0:
        raise ValueError(f"dataset contract sampling rate is {SAMPLE_RATE_HZ}, expected 256 Hz")
    import scipy.io as sio

    try:
        mat = sio.loadmat(str(path), variable_names=["eeg"])
    except Exception as exc:
        raise ValueError(f"unreadable MAT file ({exc})") from exc
    if "eeg" not in mat:
        raise ValueError("missing eeg array")
    eeg = np.asarray(mat["eeg"])
    if tuple(eeg.shape) != RAW_EEG_SHAPE:
        raise ValueError(f"eeg shape {tuple(int(x) for x in eeg.shape)}, expected {RAW_EEG_SHAPE}")
    if not np.issubdtype(eeg.dtype, np.number):
        raise ValueError(f"eeg dtype {eeg.dtype} is not numeric")


def expected_subject_ids() -> list[str]:
    return [subject for subject in SUBJECTS for _ in range(TRIALS_PER_SUBJECT)]


def expected_labels() -> list[int]:
    labels: list[int] = []
    for _subject in SUBJECTS:
        for class_id in range(N_CLASSES):
            labels.extend([class_id] * N_TRIALS_PER_CLASS)
    return labels


def validate_processed(path: Path) -> dict[str, int]:
    """Validate the canonical processed NPZ. Raises ValueError when the contract fails."""
    if not path.is_file():
        raise ValueError(f"processed dataset not found: {path}")
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"unreadable processed dataset ({exc})") from exc
    try:
        missing = [key for key in ("X", "y", "subject") if key not in npz.files]
        if missing:
            raise ValueError(f"missing arrays: {', '.join(missing)}")
        x = np.asarray(npz["X"])
        y = np.asarray(npz["y"])
        subject = np.asarray(npz["subject"])
        if x.shape != (N_PROCESSED_TRIALS, PROCESSED_SAMPLES, N_CHANNELS) or x.dtype != np.float32:
            raise ValueError(f"X must be float32 {(N_PROCESSED_TRIALS, PROCESSED_SAMPLES, N_CHANNELS)}, got {x.dtype} {x.shape}")
        if y.shape != (N_PROCESSED_TRIALS,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"y must be integer shape {(N_PROCESSED_TRIALS,)}, got {y.dtype} {y.shape}")
        if subject.shape != (N_PROCESSED_TRIALS,):
            raise ValueError(f"subject must have shape {(N_PROCESSED_TRIALS,)}, got {subject.shape}")
        if not np.isfinite(x).all():
            raise ValueError("X contains non-finite values")
        subjects = [str(item) for item in subject.tolist()]
        if subjects != expected_subject_ids():
            raise ValueError("subject order is not s1..s10 with 180 trials each")
        labels = [int(item) for item in y.tolist()]
        if labels != expected_labels():
            raise ValueError("class order is not 15 trials of classes 0..11 within each subject")
    finally:
        npz.close()
    return {"n_trials": N_PROCESSED_TRIALS, "n_subjects": N_SUBJECTS, "n_classes": N_CLASSES}


def download_url(url: str, dest: Path) -> None:
    """Stream one HTTPS response. The default SSL context verifies certificates."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS dataset URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "esp32s3_ssvep_decoder_benchmark"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response:
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _install_download(url: str, dest: Path, downloader: Downloader) -> None:
    temporary = dest.with_name(dest.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        downloader(url, temporary)
        validate_raw_mat(temporary)
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_canonical_preprocess(raw_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--config",
        str(PREPARE_CONFIG),
        "--raw-data-root",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip())
    return int(completed.returncode)


def _preprocess_into(repo: Path, raw_dir: Path, final_dir: Path) -> int:
    """Run canonical preprocessing into a staging directory, then publish the NPZ."""
    final_npz = final_dir / "processed.npz"
    if final_npz.exists():
        print(f"      refusing to overwrite {_rel(repo, final_npz)}")
        return 1
    staging = final_dir / ".preprocess_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        code = _run_canonical_preprocess(raw_dir, staging)
        if code != 0:
            print("[2/3] Preprocessing ... FAIL")
            print("      canonical preprocessing failed. processed.npz was not written.")
            return code
        produced = staging / "processed.npz"
        try:
            summary = validate_processed(produced)
        except ValueError as exc:
            print("[2/3] Preprocessing ... FAIL")
            print(f"      generated dataset failed validation: {exc}")
            print("      processed.npz was not written.")
            return 1
        final_dir.mkdir(parents=True, exist_ok=True)
        os.replace(produced, final_npz)
        meta = staging / "meta.json"
        if meta.is_file():
            document = json.loads(meta.read_text(encoding="utf-8"))
            document["output_file"] = str(final_npz)
            (final_dir / "meta.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print("[2/3] Preprocessing ... PASS")
    print(f"      {summary['n_trials']} trials, {PROCESSED_SAMPLES} samples, {N_CHANNELS} channels")
    return 0


def setup_dataset(repo: Path | None = None, *, downloader: Downloader | None = None) -> int:
    repo = (repo or REPO_ROOT).resolve()
    fetch = downloader or download_url
    print("12JFPM Dataset Setup")
    print("----------------------------------")

    raw_dir = locate_raw_directory(repo)
    existing_failures: list[str] = []
    states: list[tuple[str, str]] = []
    for subject in SUBJECTS:
        path = raw_dir / f"{subject}.mat"
        if not path.exists():
            states.append((subject, "missing"))
            continue
        try:
            validate_raw_mat(path)
        except ValueError as exc:
            existing_failures.append(f"      {_rel(repo, path)} is present but invalid ({exc}). The file was not modified.")
            states.append((subject, "invalid"))
        else:
            states.append((subject, "valid"))

    if existing_failures:
        print("[1/3] Raw dataset")
        for subject, state in states:
            if state == "invalid":
                print(f"      {subject}.mat  ... FAIL")
        print("\n".join(existing_failures))
        return 1

    missing = [subject for subject, state in states if state == "missing"]
    if not missing:
        print(f"[1/3] Raw dataset ... PASS ({len(SUBJECTS)} existing)")
    else:
        print("[1/3] Raw dataset")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for subject, state in states:
            if state == "valid":
                print(f"      {subject}.mat  ... PASS")
                continue
            print(f"      {subject}.mat  ... ", end="", flush=True)
            dest = raw_dir / f"{subject}.mat"
            try:
                _install_download(subject_url(subject), dest, fetch)
            except Exception as exc:
                print("FAIL")
                print(f"      {subject}.mat download failed: {exc}")
                print("      previously valid subject files were left unchanged.")
                return 1
            print("DOWNLOAD / PASS")

    final_npz = processed_npz_path(repo)
    if final_npz.exists():
        try:
            validate_processed(final_npz)
        except ValueError as exc:
            print("[2/3] Preprocessing ... FAIL")
            print("[3/3] Validation ... FAIL")
            print(f"      {_rel(repo, final_npz)} exists but failed validation: {exc}")
            print("      the existing file was not modified.")
            return 1
        print("[2/3] Preprocessing ... SKIP (verified)")
    else:
        code = _preprocess_into(repo, raw_dir, final_npz.parent)
        if code != 0:
            return code

    try:
        summary = validate_processed(final_npz)
    except ValueError as exc:
        print("[3/3] Validation ... FAIL")
        print(f"      {exc}")
        return 1
    print("[3/3] Validation ... PASS")
    print(f"      {summary['n_subjects']} subjects, {summary['n_classes']} classes")
    print()
    print("DATASET READY")
    print(_rel(repo, final_npz))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the 12JFPM dataset from the pinned upstream release.")
    parser.parse_args(argv)
    return setup_dataset()


if __name__ == "__main__":
    raise SystemExit(main())
