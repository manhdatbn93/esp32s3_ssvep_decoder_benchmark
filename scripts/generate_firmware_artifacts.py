#!/usr/bin/env python3
"""Stage fold-matched firmware inputs under firmware/generated/."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEN = ROOT / "firmware" / "generated"
REF = ROOT / "artifacts" / "reference"

CLASSICAL = {"cca", "fbcca", "msi", "trca"}
NEURAL_ALIASES = {
    "ccnn": ("ccnn", "f32"),
    "ccnn-f32": ("ccnn", "f32"),
    "ccnn_f32": ("ccnn", "f32"),
    "ccnn-int8": ("ccnn", "int8"),
    "ccnn_int8": ("ccnn", "int8"),
    "convca": ("convca", "f32"),
    "convca-f32": ("convca", "f32"),
    "convca_f32": ("convca", "f32"),
    "eegnet": ("eegnet", "f32"),
    "eegnet-f32": ("eegnet", "f32"),
    "eegnet_f32": ("eegnet", "f32"),
    "eegnet-int8": ("eegnet", "int8"),
    "eegnet_int8": ("eegnet", "int8"),
    "fbtcnn": ("fbtcnn", "f32"),
    "fbtcnn-f32": ("fbtcnn", "f32"),
    "fbtcnn_f32": ("fbtcnn", "f32"),
    "fbtcnn-int8": ("fbtcnn", "int8"),
    "fbtcnn_int8": ("fbtcnn", "int8"),
}

# Map staging backend key -> vector header prefix used by *_smoke.c
VECTOR_PREFIX = {
    "cca": "CCA",
    "fbcca": "FBCCA",
    "msi": "MSI",
    "trca": "TRCA",
    "ccnn": "CCNN",
    "convca": "CONVCA",
    "eegnet": "EEGNET",
    "fbtcnn": "FBTCNN",
}

IDF_BACKEND = {
    "cca": "cca",
    "fbcca": "fbcca",
    "msi": "msi",
    "trca": "trca",
    "ccnn-f32": "ccnn_f32",
    "ccnn-int8": "ccnn_int8",
    "convca-f32": "convca_f32",
    "eegnet-f32": "eegnet_f32",
    "eegnet-int8": "eegnet_int8",
    "fbtcnn-f32": "fbtcnn_f32",
    "fbtcnn-int8": "fbtcnn_int8",
}


def _refuse_reference_write(path: Path) -> None:
    ref = REF.resolve()
    path = path.resolve()
    if path == ref or ref in path.parents:
        raise SystemExit(f"Refusing to write inside reference artifacts: {path}")


def _tag(fold: int, subject: str) -> str:
    return f"fold{fold:02d}_test_{subject}"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _crc32_file(path: Path) -> int:
    return zlib.crc32(path.read_bytes()) & 0xFFFFFFFF


def _clear_generated(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _write_vector_stub(path: Path, prefix: str) -> None:
    """Minimal boot-vector header so *_smoke.c compiles without private smoke vectors."""
    p = prefix.upper()
    lines = [
        f"#ifndef SMOKE_VECTORS_{p}_H",
        f"#define SMOKE_VECTORS_{p}_H",
        "#include <stddef.h>",
        f"#define SMOKE_{p}_N_VECTORS 0",
        f"#define SMOKE_{p}_HAS_EXPECTED_SCORES 0",
        f"static const float SMOKE_{p}_ZERO[512 * 8] = {{0}};",
    ]
    for i in range(5):
        lines.append(f"#define SMOKE_{p}_VEC{i} SMOKE_{p}_ZERO")
    lines.append(f"static const float SMOKE_{p}_EXPECTED_SCORES[1][12] = {{{{0}}}};")
    if p == "TRCA":
        lines.append(f"static const int SMOKE_{p}_EXPECTED_ARGMAX_DATA[5] = {{0}};")
        lines.append(f"#define SMOKE_{p}_EXPECTED_ARGMAX SMOKE_{p}_EXPECTED_ARGMAX_DATA")
    lines.append(f"#endif /* SMOKE_VECTORS_{p}_H */")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_backend_identity(
    output: Path,
    *,
    backend: str,
    fold: int,
    subject: str,
    kind: str,
    digest_hex: str,
    crc32: int,
    size: int,
    first16: bytes,
    last16: bytes,
) -> None:
    first_list = ", ".join(str(b) for b in first16)
    last_list = ", ".join(str(b) for b in last16)
    hdr = f"""#ifndef SSVEP_BACKEND_IDENTITY_H
#define SSVEP_BACKEND_IDENTITY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {{
#endif
const char *ssvep_backend_identity_name(void);
const char *ssvep_backend_identity_kind(void);
const char *ssvep_backend_identity_subject(void);
int ssvep_backend_identity_fold(void);
uint32_t ssvep_backend_identity_crc32(void);
uint32_t ssvep_backend_identity_size(void);
const char *ssvep_backend_identity_sha256(void);
const uint8_t *ssvep_backend_identity_first16(void);
const uint8_t *ssvep_backend_identity_last16(void);
#ifdef __cplusplus
}}
#endif
#endif
"""
    src = f"""#include "backend_identity.h"
static const char kName[] = "{backend}";
static const char kKind[] = "{kind}";
static const char kSubject[] = "{subject}";
static const char kSha[] = "{digest_hex}";
static const uint8_t kFirst16[16] = {{ {first_list} }};
static const uint8_t kLast16[16] = {{ {last_list} }};
const char *ssvep_backend_identity_name(void) {{ return kName; }}
const char *ssvep_backend_identity_kind(void) {{ return kKind; }}
const char *ssvep_backend_identity_subject(void) {{ return kSubject; }}
int ssvep_backend_identity_fold(void) {{ return {int(fold)}; }}
uint32_t ssvep_backend_identity_crc32(void) {{ return {crc32}u; }}
uint32_t ssvep_backend_identity_size(void) {{ return {int(size)}u; }}
const char *ssvep_backend_identity_sha256(void) {{ return kSha; }}
const uint8_t *ssvep_backend_identity_first16(void) {{ return kFirst16; }}
const uint8_t *ssvep_backend_identity_last16(void) {{ return kLast16; }}
"""
    (output / "backend_identity.h").write_text(hdr, encoding="utf-8")
    (output / "backend_identity.c").write_text(src, encoding="utf-8")


def _write_selected_cmake(
    output: Path,
    *,
    backend: str,
    idf_backend: str,
    fold: int,
    subject: str,
    tag: str,
    kind: str,
    stage_subdir: str,
) -> None:
    text = f"""# Generated by scripts/generate_firmware_artifacts.py — do not edit.
set(SSVEP_SELECTED_BACKEND "{backend}")
set(SSVEP_IDF_BACKEND "{idf_backend}")
set(SSVEP_SELECTED_FOLD "{fold}")
set(SSVEP_SELECTED_SUBJECT "{subject}")
set(SSVEP_SELECTED_TAG "{tag}")
set(SSVEP_BACKEND_KIND "{kind}")
set(SSVEP_STAGE_SUBDIR "{stage_subdir}")
set(SSVEP_BACKEND_IDENTITY_C "${{CMAKE_CURRENT_LIST_DIR}}/backend_identity.c")
set(SSVEP_BACKEND_IDENTITY_H "${{CMAKE_CURRENT_LIST_DIR}}/backend_identity.h")
"""
    (output / "selected_backend.cmake").write_text(text, encoding="utf-8")


def _bundle_digest(paths: list[Path]) -> tuple[str, int, int, bytes, bytes]:
    h = hashlib.sha256()
    raw = bytearray()
    for p in sorted(paths, key=lambda x: x.name):
        data = p.read_bytes()
        h.update(data)
        raw.extend(data)
    digest = h.hexdigest()
    crc = zlib.crc32(bytes(raw)) & 0xFFFFFFFF
    first16 = bytes(raw[:16]).ljust(16, b"\0")
    last16 = bytes(raw[-16:]).rjust(16, b"\0") if raw else bytes(16)
    return digest, crc, len(raw), first16, last16


def stage_reference(backend: str, fold: int, output: Path, *, verify_checksum: bool) -> dict:
    from src.artifacts.reference import (
        resolve_analytical_bundle,
        resolve_preprocessing,
        resolve_tflite,
    )

    _refuse_reference_write(output)
    _clear_generated(output)

    key = backend.strip().lower().replace("_", "-")
    vectors_dir = output / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    if key in CLASSICAL:
        bundle = resolve_analytical_bundle(
            key, fold, artifact_root=REF, verify_checksum=verify_checksum
        )
        subject = bundle["held_out_subject"]
        tag = _tag(fold, subject)
        stage_dir = output / f"12jfpm_{key}_{tag}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        copied: list[Path] = []
        for src in bundle["paths"]:
            if src.suffix == ".npz":
                continue
            dst = stage_dir / src.name
            shutil.copy2(src, dst)
            staged.append(dst.name)
            copied.append(dst)
        prefix = VECTOR_PREFIX[key]
        _write_vector_stub(vectors_dir / f"smoke_vectors_{key}.h", prefix)
        digest, crc, size, first16, last16 = _bundle_digest(copied)
        display = key
        idf_backend = IDF_BACKEND[key]
        kind = "analytical"
        _write_backend_identity(
            output,
            backend=display,
            fold=fold,
            subject=subject,
            kind=kind,
            digest_hex=digest,
            crc32=crc,
            size=size,
            first16=first16,
            last16=last16,
        )
        _write_selected_cmake(
            output,
            backend=display,
            idf_backend=idf_backend,
            fold=fold,
            subject=subject,
            tag=tag,
            kind=kind,
            stage_subdir=stage_dir.name,
        )
        meta = {
            "artifact_set": "reference",
            "backend": display,
            "idf_backend": idf_backend,
            "fold": fold,
            "held_out_subject": subject,
            "tag": tag,
            "kind": kind,
            "stage_dir": _rel(stage_dir),
            "files": staged,
            "bundle_sha256": digest,
            "bundle_crc32": f"0x{crc:08x}",
            "export_artifact_root": _rel(output),
        }
        (output / "selected_backend.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        (output / "staging_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return meta

    if key not in NEURAL_ALIASES:
        raise SystemExit(f"Unsupported backend for reference staging: {backend}")
    model, precision = NEURAL_ALIASES[key]
    art = resolve_tflite(
        model, precision, fold, artifact_root=REF, verify_checksum=verify_checksum
    )
    subject = art["held_out_subject"]
    tag = _tag(fold, subject)
    if precision == "int8":
        stage_dir = output / f"12jfpm_{model}_{tag}_int8"
    else:
        stage_dir = output / f"12jfpm_{model}_{tag}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    dst = stage_dir / art["path"].name
    shutil.copy2(art["path"], dst)
    staged = [dst.name]
    if model == "fbtcnn":
        filt = resolve_preprocessing(
            "fbtcnn", fold, artifact_root=REF, verify_checksum=verify_checksum
        )
        fdst = stage_dir / filt["path"].name
        shutil.copy2(filt["path"], fdst)
        staged.append(fdst.name)
    prefix = VECTOR_PREFIX[model]
    _write_vector_stub(vectors_dir / f"smoke_vectors_{model}.h", prefix)
    data = dst.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    first16 = data[:16].ljust(16, b"\0")
    last16 = data[-16:].rjust(16, b"\0") if data else bytes(16)
    display = f"{model}-{precision}"
    idf_backend = IDF_BACKEND[display]
    kind = "neural"
    _write_backend_identity(
        output,
        backend=display,
        fold=fold,
        subject=subject,
        kind=kind,
        digest_hex=digest,
        crc32=crc,
        size=len(data),
        first16=first16,
        last16=last16,
    )
    _write_selected_cmake(
        output,
        backend=display,
        idf_backend=idf_backend,
        fold=fold,
        subject=subject,
        tag=tag,
        kind=kind,
        stage_subdir=stage_dir.name,
    )
    meta = {
        "artifact_set": "reference",
        "backend": display,
        "idf_backend": idf_backend,
        "fold": fold,
        "held_out_subject": subject,
        "tag": tag,
        "kind": kind,
        "stage_dir": _rel(stage_dir),
        "files": staged,
        "tflite": _rel(dst),
        "tflite_sha256": digest,
        "tflite_crc32": f"0x{crc:08x}",
        "export_artifact_root": _rel(output),
    }
    (output / "selected_backend.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (output / "staging_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--artifact-set",
        choices=["reference", "generated"],
        default="generated",
    )
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--tflite", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=GEN)
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _refuse_reference_write(args.output)

    if args.artifact_set == "reference":
        if args.dry_run:
            from src.artifacts.reference import (
                resolve_analytical_bundle,
                resolve_tflite,
            )

            key = args.backend.strip().lower().replace("_", "-")
            if key in CLASSICAL:
                bundle = resolve_analytical_bundle(
                    key, args.fold, verify_checksum=not args.skip_checksum
                )
                print(
                    f"OK reference {key} fold{args.fold:02d}: "
                    f"{len(bundle['paths'])} files held_out={bundle['held_out_subject']}"
                )
            else:
                model, precision = NEURAL_ALIASES[key]
                art = resolve_tflite(
                    model, precision, args.fold, verify_checksum=not args.skip_checksum
                )
                print(
                    f"OK reference {model}-{precision} fold{args.fold:02d}: "
                    f"{art['path'].name} held_out={art['held_out_subject']}"
                )
            return
        meta = stage_reference(
            args.backend,
            args.fold,
            args.output,
            verify_checksum=not args.skip_checksum,
        )
        print(f"Staged reference artifacts to {meta['stage_dir']}")
        print(f"Wrote {meta['export_artifact_root']}/selected_backend.cmake")
        print(f"Set EXPORT_ARTIFACT_ROOT to {args.output} for idf.py build")
        return

    out = args.output / args.backend / f"fold{args.fold:02d}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")
    print("Generated mode expects user-supplied conversion outputs.")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.tflite:
        print(f"TFLite: {args.tflite}")


if __name__ == "__main__":
    main()
