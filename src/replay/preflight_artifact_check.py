#!/usr/bin/env python3
"""
Pre-flight artifact check for multi-subject on-device replay (no hardware, no build).

Validates export layout, naming, fold↔subject mapping, firmware-relevant files,
and HostReference-style paths for the deployed backend subset.

Usage (repo root):
  python -m tools.replay.preflight_artifact_check
  python -m tools.replay.preflight_artifact_check --out-json results/replay_multisubject/preflight_artifact_check.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from export.subject_paths import (  # noqa: E402
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    STRICT_EXPORT_SUBDIR,
    all_subjects,
    export_dir_name,
    fold_for_subject,
    tag_for,
)

_CLASSICAL_HEADERS: dict[str, list[str]] = {
    "cca": ["cca_config_generated.h", "cca_ref_lut_generated.h"],
    "fbcca": ["fbcca_config_generated.h", "fbcca_ref_lut_generated.h"],
    "msi": ["msi_config_generated.h", "msi_ref_lut_generated.h"],
    "trca": [
        "trca_config_generated.h",
        "trca_templates_generated.h",
        "trca_spatial_filters_generated.h",
    ],
}

_DL_TFLITE: dict[str, tuple[str, bool, str]] = {
    "eegnet_f32": ("eegnet", False, "eegnet_float32.tflite"),
    "eegnet_int8": ("eegnet", True, "eegnet_int8.tflite"),
    "ccnn_f32": ("ccnn", False, "ccnn_float32.tflite"),
    "ccnn_int8": ("ccnn", True, "ccnn_int8.tflite"),
    "convca_f32": ("convca", False, "convca_float32.tflite"),
    "fbtcnn_f32": ("fbtcnn", False, "fbtcnn_float32.tflite"),
    "fbtcnn_int8": ("fbtcnn", True, "fbtcnn_int8.tflite"),
}

_DEPLOYED_ORDER: list[str] = [
    "cca",
    "fbcca",
    "msi",
    "trca",
    "eegnet_f32",
    "eegnet_int8",
    "ccnn_f32",
    "ccnn_int8",
    "convca_f32",
    "fbtcnn_f32",
    "fbtcnn_int8",
]

_TAG_RE = re.compile(r"^fold(\d{2})_(?:val|test)_(s\d+)$")


@dataclass
class CellResult:
    subject: str
    fold: int
    expected_tag: str
    backend: str
    status: str
    detail: str
    expected_dir: str
    missing: list[str]
    warnings: list[str]
    host_ref_path: str | None = None
    re_export_command: str = ""


def _validate_tag_vs_subject(tag: str, subject: str, fold: int) -> str | None:
    m = _TAG_RE.match(tag)
    if not m:
        return f"tag {tag!r} does not match fold{{NN}}_(val|test)_{{sN}}"
    fn, subj = int(m.group(1)), m.group(2)
    if subj != subject:
        return f"tag subject {subj!r} != {subject!r}"
    if fn != fold:
        return f"tag fold {fn:02d} != LOSO fold {fold} for {subject}"
    return None


def _check_metadata_json(p: Path, subject: str, fold: int) -> list[str]:
    w: list[str] = []
    if not p.is_file():
        return w
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        w.append(f"metadata.json unreadable: {e}")
        return w
    vs = meta.get("val_subject") or meta.get("valSubject")
    vf = meta.get("fold") or meta.get("CANONICAL_FOLD")
    if vs is not None and str(vs).lower() != subject:
        w.append(f"metadata val_subject={vs!r} != folder subject {subject!r}")
    if vf is not None:
        try:
            fi = int(vf)
            if fi != fold:
                w.append(f"metadata fold={fi} != expected {fold}")
        except (TypeError, ValueError):
            w.append(f"metadata fold={vf!r} not an int")
    return w


def _check_trca_npz(p: Path) -> tuple[list[str], list[str]]:
    miss: list[str] = []
    warn: list[str] = []
    if not p.is_file():
        return ["replay_reference.npz missing"], []
    try:
        import numpy as np

        d = np.load(str(p), allow_pickle=True)
        for k in ("scores", "class_ids", "trial_idxs"):
            if k not in d:
                miss.append(f"npz missing key {k!r}")
        if "scores" in d and d["scores"].shape != (180, 12):
            warn.append(f"npz scores shape {d['scores'].shape} (expected (180, 12))")
        for k in ("class_ids", "trial_idxs"):
            if k in d and len(d[k]) != 180:
                warn.append(f"npz {k} len {len(d[k])} (expected 180)")
    except Exception as e:
        miss.append(f"npz load failed: {e}")
    return miss, warn


def _re_export_command(backend: str, subject: str, *, protocol: str) -> str:
    fold = fold_for_subject(subject)
    proto_flag = ""
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        proto_flag = f" --protocol {PROTOCOL_STRICT_LOSO_INNER_VAL}"
    if backend in ("cca", "fbcca", "msi", "trca"):
        return (
            f"python -m tools.export.export_all_subjects "
            f"--subjects {subject} --backends {backend}{proto_flag}"
        )
    if backend in _DL_TFLITE:
        model, is_int8, _ = _DL_TFLITE[backend]
        if is_int8:
            return (
                f"python -m export.quantize_int8 --val-subject {subject} "
                f"--fold {fold} --models {model}{proto_flag}"
            )
        return (
            f"python -m export.convert_all_tflite --val-subject {subject} "
            f"--fold {fold} --models {model}{proto_flag}"
        )
    return ""


def check_cell(
    export_root: Path, subject: str, fold: int, backend: str, *, protocol: str,
) -> CellResult:
    tag = tag_for(subject, fold, protocol=protocol)
    missing: list[str] = []
    warnings: list[str] = []
    host_ref_path: str | None = None
    recmd = _re_export_command(backend, subject, protocol=protocol)

    tag_err = _validate_tag_vs_subject(tag, subject, fold)
    if tag_err:
        return CellResult(
            subject=subject,
            fold=fold,
            expected_tag=tag,
            backend=backend,
            status="Naming mismatch",
            detail=tag_err,
            expected_dir="",
            missing=[],
            warnings=[],
            host_ref_path=None,
            re_export_command=recmd,
        )

    if backend in _CLASSICAL_HEADERS:
        model = backend
        dir_name = export_dir_name(model, subject, fold, int8=False, protocol=protocol)
        dpath = export_root / dir_name
        if not dpath.is_dir():
            return CellResult(
                subject=subject,
                fold=fold,
                expected_tag=tag,
                backend=backend,
                status="Missing artifact",
                detail=f"export directory missing: {dpath.relative_to(_REPO)}",
                expected_dir=str(dpath.relative_to(_REPO)),
                missing=[dir_name],
                warnings=[],
                host_ref_path=None,
                re_export_command=recmd,
            )
        if dpath.name != dir_name:
            return CellResult(
                subject=subject,
                fold=fold,
                expected_tag=tag,
                backend=backend,
                status="Naming mismatch",
                detail=f"dir name {dpath.name!r} != expected {dir_name!r}",
                expected_dir=dir_name,
                missing=[],
                warnings=[],
                host_ref_path=None,
                re_export_command=recmd,
            )
        for fn in _CLASSICAL_HEADERS[backend]:
            if not (dpath / fn).is_file():
                missing.append(fn)
        if backend == "trca":
            npz = dpath / "replay_reference.npz"
            host_ref_path = str(npz.relative_to(_REPO))
            nm, nw = _check_trca_npz(npz)
            missing.extend(nm)
            warnings.extend(nw)
        else:
            host_ref_path = f"live: src.benchmarks.{backend}_12jfpm"

        warnings.extend(_check_metadata_json(dpath / "metadata.json", subject, fold))

        if missing:
            st, det = "Missing artifact", "; ".join(missing[:6])
        elif warnings:
            st, det = "Needs manual inspection", "; ".join(warnings[:4])
        else:
            st, det = "Ready", "complete"

        return CellResult(
            subject=subject,
            fold=fold,
            expected_tag=tag,
            backend=backend,
            status=st,
            detail=det,
            expected_dir=str(dpath.relative_to(_REPO)),
            missing=missing,
            warnings=warnings,
            host_ref_path=host_ref_path,
            re_export_command=recmd,
        )

    if backend in _DL_TFLITE:
        model, is_int8, tflite_name = _DL_TFLITE[backend]
        dir_name = export_dir_name(model, subject, fold, int8=is_int8, protocol=protocol)
        dpath = export_root / dir_name
        tflite_path = dpath / tflite_name
        host_ref_path = str(tflite_path.relative_to(_REPO))

        if not dpath.is_dir():
            return CellResult(
                subject=subject,
                fold=fold,
                expected_tag=tag,
                backend=backend,
                status="Missing artifact",
                detail=f"missing dir {dpath.relative_to(_REPO)}",
                expected_dir=str(dpath.relative_to(_REPO)),
                missing=[dir_name],
                warnings=[],
                host_ref_path=host_ref_path,
                re_export_command=recmd,
            )

        if not tflite_path.is_file():
            missing.append(tflite_name)

        if model == "fbtcnn":
            if not (dpath / "fbtcnn_filter_generated.h").is_file():
                missing.append("fbtcnn_filter_generated.h")

        warnings.extend(_check_metadata_json(dpath / "metadata.json", subject, fold))

        if tflite_path.is_file() and tflite_path.stat().st_size < 4096:
            warnings.append(
                f"{tflite_name} small ({tflite_path.stat().st_size} B)"
            )

        if missing:
            return CellResult(
                subject=subject,
                fold=fold,
                expected_tag=tag,
                backend=backend,
                status="Missing artifact",
                detail="; ".join(missing),
                expected_dir=str(dpath.relative_to(_REPO)),
                missing=missing,
                warnings=warnings,
                host_ref_path=host_ref_path,
                re_export_command=recmd,
            )
        if warnings:
            return CellResult(
                subject=subject,
                fold=fold,
                expected_tag=tag,
                backend=backend,
                status="Needs manual inspection",
                detail="; ".join(warnings),
                expected_dir=str(dpath.relative_to(_REPO)),
                missing=[],
                warnings=warnings,
                host_ref_path=host_ref_path,
                re_export_command=recmd,
            )
        return CellResult(
            subject=subject,
            fold=fold,
            expected_tag=tag,
            backend=backend,
            status="Ready",
            detail="complete",
            expected_dir=str(dpath.relative_to(_REPO)),
            missing=[],
            warnings=[],
            host_ref_path=host_ref_path,
            re_export_command=recmd,
        )

    return CellResult(
        subject=subject,
        fold=fold,
        expected_tag=tag,
        backend=backend,
        status="Needs manual inspection",
        detail=f"unknown backend {backend!r}",
        expected_dir="",
        missing=[],
        warnings=[],
        host_ref_path=None,
        re_export_command="",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=_REPO
        / "results"
        / "replay_multisubject"
        / "preflight_artifact_check.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=list(all_subjects()),
        help="Subjects to check (default s1..s10)",
    )
    from export.protocol_cli import DEFAULT_PROTOCOL, add_protocol_arg, apply_protocol_side_effects  # noqa: E402
    add_protocol_arg(ap)
    args = ap.parse_args()
    apply_protocol_side_effects(args.protocol)

    export_root = _REPO / "export"
    if args.protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        export_root = export_root / STRICT_EXPORT_SUBDIR
    subjects = [s.strip().lower() for s in args.subjects]

    cells: list[CellResult] = []
    for subj in subjects:
        fd = fold_for_subject(subj)
        for be in _DEPLOYED_ORDER:
            cells.append(check_cell(export_root, subj, fd, be, protocol=args.protocol))

    by_status: dict[str, int] = {}
    for c in cells:
        by_status[c.status] = by_status.get(c.status, 0) + 1

    not_ready = [c for c in cells if c.status != "Ready"]
    verdict_ready = len(not_ready) == 0

    report: dict[str, Any] = {
        "schema": "preflight_artifact_check_v1",
        "repo": str(_REPO),
        "export_root": str(export_root.relative_to(_REPO)),
        "subjects": subjects,
        "backends": _DEPLOYED_ORDER,
        "host_reference_notes": {
            "cca_fbcca_msi": (
                "HostReference uses live Python benchmarks; firmware uses "
                "export/*/ *_config_generated.h + *_ref_lut_generated.h."
            ),
            "trca": (
                "HostReference loads export/12jfpm_trca_{tag}/replay_reference.npz."
            ),
            "dl": (
                "HostReference uses export_dir_name() + .tflite per host_reference.py."
            ),
            "firmware_tag": (
                "TARGET_FOLD_SUBJECT_TAG must equal tag_for(subject) for LOSO pairing."
            ),
        },
        "summary": {
            "total_cells": len(cells),
            "by_status": by_status,
            "all_ready": verdict_ready,
        },
        "cells": [asdict(c) for c in cells],
        "not_ready": [asdict(c) for c in not_ready],
    }

    out_json = (args.out_json if args.out_json.is_absolute()
                else _REPO / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "fold",
                "expected_tag",
                "backend",
                "status",
                "detail",
                "expected_dir",
                "missing",
                "warnings",
                "host_ref_path",
                "re_export_command",
            ],
        )
        w.writeheader()
        for c in cells:
            row = asdict(c)
            row["missing"] = ";".join(c.missing)
            row["warnings"] = ";".join(c.warnings)
            w.writerow(row)

    try:
        print(f"Wrote {out_json.relative_to(_REPO)}")
        print(f"Wrote {out_csv.relative_to(_REPO)}")
    except ValueError:
        print(f"Wrote {out_json}")
        print(f"Wrote {out_csv}")
    print(f"Summary: {by_status}")
    print(f"All ready: {verdict_ready}")
    if not_ready:
        print(f"Not ready: {len(not_ready)} cells — see report")
    return 0 if verdict_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
