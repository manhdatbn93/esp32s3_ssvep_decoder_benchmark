#!/usr/bin/env python3
"""
Export CCA constants + precomputed reference LUT for ESP32-S3 deployment.

Usage:
  python -m export.export_cca_constants [--out export/12jfpm_cca_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export.common import (
    CANONICAL_VAL_SUBJECT,
    C,
    FREQS,
    FS,
    N_HARMONICS,
    T,
    build_reference_lut,
    compute_sos_filter,
    emit_ref_lut_header,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.benchmarks.cca_12jfpm import CCAConfig, cca_scores_for_trial

_DEFAULT_OUT = "export/12jfpm_cca_fold01_val_s1"


def _emit_cca_config_header(out_dir: Path, sos: np.ndarray, zi: np.ndarray, cfg: CCAConfig) -> Path:
    n_sos = sos.shape[0]
    edge = 3 * n_sos
    lines = [
        "/* Auto-generated CCA config for 12JFPM — do not edit. */",
        "#ifndef CCA_CONFIG_GENERATED_H",
        "#define CCA_CONFIG_GENERATED_H",
        "",
        f"#define CCA_FS_HZ {cfg.sampling_rate_hz}",
        f"#define CCA_T {T}",
        f"#define CCA_C {C}",
        f"#define CCA_N_HARM {cfg.n_harmonics}",
        f"#define CCA_R {2 * cfg.n_harmonics}",
        f"#define CCA_NUM_CLASSES {len(FREQS)}",
        f"#define CCA_N_SOS {n_sos}",
        f"#define CCA_EDGE {edge}",
        f"#define CCA_REG {cfg.cca_reg}",
        "",
        f"static const double CCA_SOS[{n_sos}][6] = {{",
    ]
    for s in range(n_sos):
        vals = ", ".join(f"{v}" for v in sos[s])
        comma = "," if s < n_sos - 1 else ""
        lines.append(f"  {{ {vals} }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append(f"static const double CCA_SOSFILT_ZI[{n_sos}][2] = {{")
    for s in range(n_sos):
        vals = ", ".join(f"{v}" for v in zi[s])
        comma = "," if s < n_sos - 1 else ""
        lines.append(f"  {{ {vals} }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append(f"static const double CCA_TARGET_HZ[{len(FREQS)}] = {{")
    lines.append(f"  {', '.join(str(f) for f in FREQS)}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* CCA_CONFIG_GENERATED_H */")
    lines.append("")
    p = out_dir / "cca_config_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export CCA constants for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CCA export -> {out_dir}")

    cfg = CCAConfig()
    lo, hi = cfg.single_band_hz
    sos, zi = compute_sos_filter(lo, hi, order=cfg.butterworth_order)

    _emit_cca_config_header(out_dir, sos, zi, cfg)
    print("  -> cca_config_generated.h")

    lut = build_reference_lut()
    emit_ref_lut_header(
        lut, out_dir / "cca_ref_lut_generated.h",
        "CCA_REF_LUT_GENERATED_H", "CCA_REF_LUT",
    )
    print("  -> cca_ref_lut_generated.h")

    npz = load_processed_npz(args.processed_npz)
    vectors = get_fold_test_vectors(npz)
    save_test_vectors(vectors, out_dir)
    print(f"  -> {len(vectors)} test vectors")

    refs = {}
    for i, v in enumerate(vectors):
        scores = cca_scores_for_trial(v, cfg)
        refs[f"idx{i}"] = {
            "scores": [float(s) for s in scores],
            "argmax": int(np.argmax(scores)),
        }
    write_host_reference(out_dir, refs)
    print("  -> host_reference.json")

    write_metadata(out_dir, "cca", {
        "bandpass_hz": list(cfg.single_band_hz),
        "butterworth_order": cfg.butterworth_order,
        "cca_reg": cfg.cca_reg,
        "n_harmonics": cfg.n_harmonics,
        "n_sos_sections": int(sos.shape[0]),
    })
    print("  -> metadata.json")
    print("CCA export complete.")


if __name__ == "__main__":
    main()
