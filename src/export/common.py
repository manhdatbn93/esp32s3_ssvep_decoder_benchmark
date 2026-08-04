"""
Shared utilities for export scripts.

Handles test vector generation, metadata writing, and reference
signal LUT computation used by all classical backends.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from src.datasets.parse_12jfpm import (
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE_HZ,
    TARGET_FREQUENCIES_HZ,
)

CANONICAL_FOLD = 1
CANONICAL_VAL_SUBJECT = "s1"
T = 512
C = N_CHANNELS
FS = SAMPLE_RATE_HZ
N_HARMONICS = 3
FREQS = TARGET_FREQUENCIES_HZ


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_processed_npz(path: Path | None = None) -> dict:
    path = path or repo_root() / "data" / "processed" / "12JFPM" / "processed.npz"
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def get_fold_test_vectors(
    npz: dict, val_subject: str | None = None, n_vectors: int = 5
) -> list[np.ndarray]:
    # Resolve default at call time so orchestrators can override
    # CANONICAL_VAL_SUBJECT globally for multi-subject exports.
    if val_subject is None:
        val_subject = CANONICAL_VAL_SUBJECT
    subjects = npz.get("subjects", npz.get("subject"))
    X = npz["X"]
    mask = np.array([str(s) == val_subject for s in subjects])
    X_val = X[mask]
    indices = list(range(min(n_vectors, len(X_val))))
    return [X_val[i].astype(np.float32) for i in indices]


def save_test_vectors(vectors: list[np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, v in enumerate(vectors):
        p = out_dir / f"smoke_input_idx{i}_float32.bin"
        p.write_bytes(v.astype(np.float32).tobytes())
        paths.append(p)
    return paths


def write_metadata(out_dir: Path, method: str, extra: dict | None = None) -> Path:
    meta = {
        "method": method,
        "dataset": "12JFPM",
        "fold": CANONICAL_FOLD,
        "val_subject": CANONICAL_VAL_SUBJECT,
        "sampling_rate_hz": FS,
        "onset_sample": 38,
        "window_length": T,
        "n_channels": C,
        "n_classes": N_CLASSES,
        "normalization": "channel_zscore",
        "target_frequencies_hz": FREQS,
        "n_harmonics": N_HARMONICS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    p = out_dir / "metadata.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def write_host_reference(out_dir: Path, refs: dict) -> Path:
    p = out_dir / "host_reference.json"
    p.write_text(json.dumps(refs, indent=2), encoding="utf-8")
    return p


def build_reference_lut(
    n_samples: int = T,
    fs: float = FS,
    freqs: list[float] | None = None,
    n_harmonics: int = N_HARMONICS,
) -> np.ndarray:
    """
    Precompute sin/cos reference signals for all classes and harmonics.

    Returns shape [n_classes, 2*n_harmonics, n_samples] float64.
    Row layout per class: [sin(h1), cos(h1), sin(h2), cos(h2), ...].
    """
    freqs = freqs or FREQS
    n_cls = len(freqs)
    t_vec = np.arange(n_samples, dtype=np.float64) / fs
    lut = np.zeros((n_cls, 2 * n_harmonics, n_samples), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for k, fk in enumerate(freqs):
        for h in range(n_harmonics):
            fh = fk * (h + 1)
            lut[k, 2 * h, :] = np.sin(two_pi * fh * t_vec)
            lut[k, 2 * h + 1, :] = np.cos(two_pi * fh * t_vec)
    return lut


def emit_ref_lut_header(
    lut: np.ndarray,
    header_path: Path,
    guard_name: str,
    array_name: str,
) -> None:
    """Write a C header containing the reference signal LUT as static const float."""
    n_cls, n_rows, n_samples = lut.shape
    lines = [
        f"/* Auto-generated reference signal LUT — do not edit. */",
        f"#ifndef {guard_name}",
        f"#define {guard_name}",
        "",
        f"#define REF_LUT_N_CLASSES {n_cls}",
        f"#define REF_LUT_N_ROWS {n_rows}  /* 2 * n_harmonics */",
        f"#define REF_LUT_N_SAMPLES {n_samples}",
        "",
        f"static const float {array_name}[{n_cls}][{n_rows}][{n_samples}] = {{",
    ]
    for k in range(n_cls):
        lines.append(f"  {{ /* class {k}: {FREQS[k]} Hz */")
        for r in range(n_rows):
            vals = ", ".join(f"{v:.8f}f" for v in lut[k, r, :].astype(np.float32))
            comma = "," if r < n_rows - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if k < n_cls - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append(f"#endif /* {guard_name} */")
    lines.append("")
    header_path.write_text("\n".join(lines), encoding="utf-8")


def compute_sos_filter(low_hz: float, high_hz: float, order: int = 4) -> tuple:
    """Compute Butterworth SOS and sosfilt_zi for bandpass."""
    nyq = 0.5 * FS
    lo = max(low_hz / nyq, 1e-6)
    hi = min(high_hz / nyq, 1.0 - 1e-6)
    sos = scipy_signal.butter(order, [lo, hi], btype="band", output="sos")
    zi = scipy_signal.sosfilt_zi(sos)
    return sos, zi


def compute_cheby_filter_bank(cfg_params: dict) -> list[dict]:
    """Compute Chebyshev Type I filter bank SOS + zi for FBCCA/TRCA bands."""
    nyq = 0.5 * FS
    passband_hz = cfg_params["passband_hz"]
    stopband_hz = cfg_params["stopband_hz"]
    highcut_pass = cfg_params["highcut_pass_hz"]
    highcut_stop = cfg_params["highcut_stop_hz"]
    gpass = cfg_params["gpass_db"]
    gstop = cfg_params["gstop_db"]
    rp = cfg_params["cheby_ripple_db"]

    banks = []
    for i in range(len(passband_hz)):
        wp = [passband_hz[i] / nyq, highcut_pass / nyq]
        ws = [stopband_hz[i] / nyq, highcut_stop / nyq]
        n, wn = scipy_signal.cheb1ord(wp, ws, gpass, gstop)
        sos = scipy_signal.cheby1(n, rp, wn, btype="bandpass", output="sos")
        zi = scipy_signal.sosfilt_zi(sos)
        padlen = 3 * max(sos.shape[0] * 2, 1)
        banks.append({"sos": sos, "zi": zi, "padlen": padlen, "order": n})
    return banks
