"""Resolve retained MCU reference artifacts under artifacts/reference/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "reference"
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

REQUIRED_FIELDS = {
    "artifact_id",
    "artifact_type",
    "backend",
    "model",
    "precision",
    "fold",
    "held_out_subject",
    "relative_path",
    "file_size_bytes",
    "sha256",
    "firmware_role",
    "required_for_reference_replay",
    "class_count",
}


def repo_root() -> Path:
    return ROOT


def default_artifact_root() -> Path:
    return DEFAULT_ARTIFACT_ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference_manifest(artifact_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Reference manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Reference manifest must be a JSON list")
    ids: set[str] = set()
    for entry in data:
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"Manifest entry missing fields {sorted(missing)}: {entry.get('artifact_id')}")
        aid = entry["artifact_id"]
        if aid in ids:
            raise ValueError(f"Duplicate artifact_id: {aid}")
        ids.add(aid)
    return data


def _abs_path(entry: dict[str, Any], artifact_root: Path) -> Path:
    rel = entry["relative_path"].replace("\\", "/")
    if rel.startswith("artifacts/reference/"):
        return ROOT / rel
    return artifact_root / rel


def _held_out_for_fold(fold: int) -> str:
    if fold < 1 or fold > 10:
        raise ValueError(f"fold must be in 1..10, got {fold}")
    man = ROOT / "manifests" / "loso" / f"fold{fold:02d}.json"
    if not man.is_file():
        raise FileNotFoundError(f"LOSO manifest missing: {man}")
    doc = json.loads(man.read_text(encoding="utf-8"))
    return str(doc["held_out_subject"])


def _match(
    entries: list[dict[str, Any]],
    *,
    artifact_type: str | None = None,
    model: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        if artifact_type is not None and e["artifact_type"] != artifact_type:
            continue
        if model is not None and e["model"] != model:
            continue
        if backend is not None and e["backend"] != backend:
            continue
        if precision is not None and e["precision"] != precision:
            continue
        if fold is not None and int(e["fold"]) != int(fold):
            continue
        out.append(e)
    return out


def _one(matches: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not matches:
        raise FileNotFoundError(f"No reference artifact for {label}")
    if len(matches) > 1:
        ids = [m["artifact_id"] for m in matches]
        raise ValueError(f"Ambiguous reference artifacts for {label}: {ids}")
    return matches[0]


def _verify_entry(
    entry: dict[str, Any],
    artifact_root: Path,
    *,
    verify_checksum: bool = True,
) -> Path:
    path = _abs_path(entry, artifact_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference file for {entry['artifact_id']}: {path}")
    expected_subject = _held_out_for_fold(int(entry["fold"]))
    if str(entry["held_out_subject"]) != expected_subject:
        raise ValueError(
            f"Held-out subject mismatch for {entry['artifact_id']}: "
            f"manifest={entry['held_out_subject']} loso={expected_subject}"
        )
    with path.open("rb") as handle:
        prefix = handle.read(len(_LFS_POINTER_PREFIX))
    if prefix == _LFS_POINTER_PREFIX:
        raise ValueError(
            f"{entry['artifact_id']} is a Git LFS pointer, not the retained artifact: {path}. "
            "Clone with Git and run git lfs pull."
        )
    if verify_checksum:
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {entry['artifact_id']}")
    return path


def resolve_checkpoint(
    model: str,
    fold: int,
    *,
    artifact_root: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    entries = load_reference_manifest(root)
    entry = _one(
        _match(entries, artifact_type="checkpoint", model=model, fold=fold),
        f"checkpoint model={model} fold={fold}",
    )
    path = _verify_entry(entry, root, verify_checksum=verify_checksum)
    return {**entry, "path": path}


def resolve_tflite(
    backend: str,
    precision: str,
    fold: int,
    *,
    artifact_root: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    entries = load_reference_manifest(root)
    backend_key = backend.replace("_", "-")
    if "-" not in backend_key:
        backend_key = f"{backend_key}-{precision}"
    # Accept backend already including precision.
    if backend_key.endswith(("-f32", "-int8")):
        precision = backend_key.split("-")[-1]
        model = backend_key.rsplit("-", 1)[0]
    else:
        model = backend_key
        backend_key = f"{model}-{precision}"
    entry = _one(
        _match(entries, artifact_type="tflite", backend=backend_key, fold=fold),
        f"tflite backend={backend_key} fold={fold}",
    )
    path = _verify_entry(entry, root, verify_checksum=verify_checksum)
    return {**entry, "path": path, "model": model, "precision": precision}


def resolve_analytical_bundle(
    backend: str,
    fold: int,
    *,
    artifact_root: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    entries = load_reference_manifest(root)
    files = _match(entries, artifact_type="analytical", backend=backend, fold=fold)
    if not files:
        raise FileNotFoundError(f"No analytical bundle for backend={backend} fold={fold}")
    paths = []
    for entry in files:
        paths.append(_verify_entry(entry, root, verify_checksum=verify_checksum))
    return {
        "backend": backend,
        "fold": fold,
        "held_out_subject": files[0]["held_out_subject"],
        "entries": files,
        "paths": paths,
    }


def resolve_preprocessing(
    model: str,
    fold: int,
    *,
    artifact_root: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    entries = load_reference_manifest(root)
    files = _match(entries, artifact_type="preprocessing", model=model, fold=fold)
    if not files:
        raise FileNotFoundError(f"No preprocessing artifact for model={model} fold={fold}")
    entry = _one(files, f"preprocessing model={model} fold={fold}")
    path = _verify_entry(entry, root, verify_checksum=verify_checksum)
    return {**entry, "path": path}


def verify_reference_artifacts(
    artifact_root: Path | None = None,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root) if artifact_root else DEFAULT_ARTIFACT_ROOT
    entries = load_reference_manifest(root)
    errors: list[str] = []

    ck = [e for e in entries if e["artifact_type"] == "checkpoint"]
    tf = [e for e in entries if e["artifact_type"] == "tflite"]
    an = [e for e in entries if e["artifact_type"] == "analytical"]
    pr = [e for e in entries if e["artifact_type"] == "preprocessing"]

    if len(ck) != 40:
        errors.append(f"expected 40 checkpoints, found {len(ck)}")
    if len(tf) != 70:
        errors.append(f"expected 70 TFLite files, found {len(tf)}")

    for model in ("ccnn", "convca", "eegnet", "fbtcnn"):
        folds = sorted(int(e["fold"]) for e in ck if e["model"] == model)
        if folds != list(range(1, 11)):
            errors.append(f"checkpoint folds incomplete for {model}: {folds}")

    for backend in (
        "ccnn-f32",
        "ccnn-int8",
        "convca-f32",
        "eegnet-f32",
        "eegnet-int8",
        "fbtcnn-f32",
        "fbtcnn-int8",
    ):
        folds = sorted(int(e["fold"]) for e in tf if e["backend"] == backend)
        if folds != list(range(1, 11)):
            errors.append(f"tflite folds incomplete for {backend}: {folds}")

    tuples = [(e["backend"], e["fold"], e["precision"]) for e in tf]
    if len(tuples) != len(set(tuples)):
        errors.append("duplicated backend-fold-precision TFLite tuples")

    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        errors.append("missing checksums.sha256")
    else:
        listed = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            listed[rel.replace("\\", "/")] = digest
        for e in entries:
            rel = e["relative_path"].replace("\\", "/")
            if rel not in listed:
                errors.append(f"missing checksum line for {rel}")
            elif listed[rel] != e["sha256"]:
                errors.append(f"checksum file mismatch for {rel}")

    for e in entries:
        try:
            _verify_entry(e, root, verify_checksum=verify_checksum)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    # analytical completeness
    expected_an = {
        "cca": {"cca_config_generated.h", "cca_ref_lut_generated.h"},
        "fbcca": {"fbcca_config_generated.h", "fbcca_ref_lut_generated.h"},
        "msi": {"msi_config_generated.h", "msi_ref_lut_generated.h"},
        "trca": {
            "trca_config_generated.h",
            "trca_templates_generated.h",
            "trca_spatial_filters_generated.h",
            "replay_reference.npz",
        },
    }
    for backend, names in expected_an.items():
        for fold in range(1, 11):
            have = {
                Path(e["relative_path"]).name
                for e in an
                if e["backend"] == backend and int(e["fold"]) == fold
            }
            if have != names:
                errors.append(f"analytical incomplete {backend} fold{fold}: {sorted(have)}")

    for fold in range(1, 11):
        have = [e for e in pr if int(e["fold"]) == fold and e["model"] == "fbtcnn"]
        if len(have) != 1:
            errors.append(f"fbtcnn filter header count fold{fold}={len(have)}")

    return {
        "ok": not errors,
        "n_entries": len(entries),
        "n_checkpoints": len(ck),
        "n_tflite": len(tf),
        "n_analytical": len(an),
        "n_preprocessing": len(pr),
        "errors": errors,
    }
