"""Table 5 resource profile from the archived April 2026 profiling evidence.

Footprint values come from the per-backend ``idf.py size`` screenshots (transcribed
in ``idf_size_transcription.csv``). Heap drop and minimum free stack are parsed
from the UART ``RES_PROFILE phase=session_summary`` line of each backend session.
Used and allocated TFLM arena are parsed from the smoke-test ``TFLM_ARENA`` lines.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .constants import ALL_BACKENDS, CLASSICAL_BACKENDS, EVIDENCE, NEURAL_BACKENDS
from .evidence import read_csv

UART_FILES = {
    "CCA": "cca", "CCNN-f32": "ccnn_f32", "CCNN-int8": "ccnn_int8", "ConvCA-f32": "convca",
    "EEGNet-f32": "eegnet_f32", "EEGNet-int8": "eegnet_int8", "FBCCA": "fbcca",
    "FBtCNN-f32": "fbtcnn_f32", "FBtCNN-int8": "fbtcnn_int8", "MSI": "msi", "TRCA": "trca",
}
PROFILE_FILES = {**UART_FILES, "ConvCA-f32": "convca_f32"}

SESSION_RE = re.compile(r"(?:RES_PROFILE )?phase=session_summary backend=(\S+) (.*)")
ARENA_RE = re.compile(r"TFLM_ARENA backend=(\S+) (.*)")
FIELD_RE = re.compile(r"(\w+)=(\S+)")


def kib_1dp(n_bytes: int) -> Decimal:
    return (Decimal(n_bytes) / Decimal(1024)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def read_log(path: Path) -> str:
    """UART captures were saved either as UTF-8 or as UTF-16 with a byte-order mark."""
    data = path.read_bytes()
    return data.decode("utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else data.decode("utf-8-sig")


def _fields(text: str) -> dict[str, str]:
    return dict(FIELD_RE.findall(text))


def parse_session_summary(path: Path, backend: str) -> dict[str, int]:
    lines = [m for m in (SESSION_RE.search(x) for x in read_log(path).splitlines()) if m]
    if len(lines) != 1 or lines[0].group(1) != backend:
        raise ValueError(f"{path.name}: expected exactly one session_summary for {backend}, found {len(lines)}")
    f = _fields(lines[0].group(2))
    return {"sess_est_peak_drop_int": int(f["sess_est_peak_drop_int"]), "stack_hwm_words": int(f["stack_hwm_words"])}


def parse_arena_log(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in read_log(path).splitlines():
        m = ARENA_RE.search(line)
        if not m:
            continue
        backend, f = m.group(1), _fields(m.group(2))
        if backend in out:
            raise ValueError(f"{path.name}: more than one TFLM_ARENA record for {backend}")
        out[backend] = {"cap": int(f["cap"]), "used": int(f["used"]), "region": f["region"], "path": f["path"]}
    return out


def table5(evidence: Path = EVIDENCE) -> dict[str, dict[str, Any]]:
    root = evidence / "table5"
    shots = {r["backend"]: r for r in read_csv(root / "idf_size_transcription.csv")}
    arena = parse_arena_log(root / "smoke_arena" / "tensor_arenatxt.txt")
    if set(shots) != set(ALL_BACKENDS):
        raise ValueError("footprint transcription does not cover the 11 backends")
    if set(arena) != set(NEURAL_BACKENDS):
        raise ValueError(f"arena log covers {sorted(arena)}, expected the 7 neural backends")
    out = {}
    for backend in ALL_BACKENDS:
        s = shots[backend]
        uart = parse_session_summary(root / "uart_replay_sessions" / f"uart_{UART_FILES[backend]}.log", backend)
        profile = parse_session_summary(
            root / "uart_replay_sessions" / f"uart_{PROFILE_FILES[backend]}_profile.txt", backend)
        if uart != profile:
            raise ValueError(f"{backend}: UART log and profile session summaries differ")
        row = {
            "total_image_size_bytes": int(s["total_image_size_bytes"]),
            "flash_code_bytes": int(s["flash_code_bytes"]),
            "flash_data_bytes": int(s["flash_data_bytes"]),
            "diram_used_bytes": int(s["diram_used_bytes"]),
            "heap_drop_bytes": uart["sess_est_peak_drop_int"],
            "min_free_stack_bytes": uart["stack_hwm_words"],
            "arena_used_bytes": arena[backend]["used"] if backend in arena else None,
            "arena_capacity_bytes": arena[backend]["cap"] if backend in arena else None,
            "arena_region": arena[backend]["region"] if backend in arena else None,
        }
        row.update({
            "app_kib": kib_1dp(row["total_image_size_bytes"]),
            "code_kib": kib_1dp(row["flash_code_bytes"]),
            "data_kib": kib_1dp(row["flash_data_bytes"]),
            "diram_kib": kib_1dp(row["diram_used_bytes"]),
            "heap_drop_kib": kib_1dp(row["heap_drop_bytes"]),
            "arena_used_kib": kib_1dp(row["arena_used_bytes"]) if row["arena_used_bytes"] is not None else None,
            "arena_capacity_kib": kib_1dp(row["arena_capacity_bytes"]) if row["arena_capacity_bytes"] is not None else None,
        })
        out[backend] = row
    for backend in CLASSICAL_BACKENDS:
        if out[backend]["arena_used_bytes"] is not None:
            raise ValueError(f"{backend}: analytical backend must not have a TFLM arena")
    return out


def policy_resource_profile(t5: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float | None]]:
    """(application image KiB, used TFLM arena KiB) as used by the deployment policies."""
    return {
        b: (float(r["app_kib"]), None if r["arena_used_kib"] is None else float(r["arena_used_kib"]))
        for b, r in t5.items()
    }
