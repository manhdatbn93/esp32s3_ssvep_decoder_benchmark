#!/usr/bin/env python3
"""
Export FBCCA constants + precomputed reference LUT for ESP32-S3 deployment.

Usage:
  python -m export.export_fbcca_constants [--out export/12jfpm_fbcca_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export.common import (
    FREQS,
    FS,
    N_HARMONICS,
    T,
    C,
    build_reference_lut,
    compute_cheby_filter_bank,
    emit_ref_lut_header,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.benchmarks.fbcca_12jfpm import FBCCAConfig, band_weights, fbcca_scores_for_trial

_DEFAULT_OUT = "export/12jfpm_fbcca_fold01_val_s1"


def _emit_fbcca_config_header(out_dir: Path, banks: list[dict], cfg: FBCCAConfig) -> Path:
    n_bands = len(banks)
    max_sos = max(b["sos"].shape[0] for b in banks)
    bw = band_weights(n_bands)

    max_padlen = max(3 * (max(b["sos"].shape[0] * 2, 1)) for b in banks)

    lines = [
        "/* Auto-generated FBCCA config for 12JFPM — do not edit. */",
        "#ifndef FBCCA_CONFIG_GENERATED_H",
        "#define FBCCA_CONFIG_GENERATED_H",
        "",
        f"#define FBCCA_FS_HZ {cfg.sampling_rate_hz}",
        f"#define FBCCA_T {T}",
        f"#define FBCCA_C {C}",
        f"#define FBCCA_N_HARM {cfg.n_harmonics}",
        f"#define FBCCA_R {2 * cfg.n_harmonics}",
        f"#define FBCCA_NUM_CLASSES {len(FREQS)}",
        f"#define FBCCA_N_BANDS {n_bands}",
        f"#define FBCCA_N_SOS_MAX {max_sos}",
        f"#define FBCCA_REG {cfg.cca_reg}",
        "",
        f"static const int FBCCA_N_SOS_PER_BAND[{n_bands}] = {{",
        f"  {', '.join(str(b['sos'].shape[0]) for b in banks)}",
        "};",
        "",
        f"static const double FBCCA_BAND_WEIGHTS[{n_bands}] = {{",
        f"  {', '.join(str(w) for w in bw)}",
        "};",
        "",
        f"static const double FBCCA_SOS[{n_bands}][{max_sos}][6] = {{",
    ]
    for i, b in enumerate(banks):
        sos = b["sos"]
        n_s = sos.shape[0]
        lines.append(f"  {{ /* band {i} */")
        for s in range(max_sos):
            if s < n_s:
                vals = ", ".join(f"{v}" for v in sos[s])
            else:
                vals = "0, 0, 0, 1, 0, 0"
            comma = "," if s < max_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < n_bands - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")

    lines.append(f"static const double FBCCA_SOSFILT_ZI[{n_bands}][{max_sos}][2] = {{")
    for i, b in enumerate(banks):
        zi = b["zi"]
        n_s = zi.shape[0]
        lines.append(f"  {{ /* band {i} */")
        for s in range(max_sos):
            if s < n_s:
                vals = ", ".join(f"{v}" for v in zi[s])
            else:
                vals = "0, 0"
            comma = "," if s < max_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < n_bands - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")

    lines.append(f"static const double FBCCA_TARGET_HZ[{len(FREQS)}] = {{")
    lines.append(f"  {', '.join(str(f) for f in FREQS)}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* FBCCA_CONFIG_GENERATED_H */")
    lines.append("")
    p = out_dir / "fbcca_config_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export FBCCA constants for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FBCCA export -> {out_dir}")

    cfg = FBCCAConfig()
    bank_params = {
        "passband_hz": list(cfg.passband_hz),
        "stopband_hz": list(cfg.stopband_hz),
        "highcut_pass_hz": cfg.highcut_pass_hz,
        "highcut_stop_hz": cfg.highcut_stop_hz,
        "gpass_db": cfg.gpass_db,
        "gstop_db": cfg.gstop_db,
        "cheby_ripple_db": cfg.cheby_ripple_db,
    }
    banks = compute_cheby_filter_bank(bank_params)

    _emit_fbcca_config_header(out_dir, banks, cfg)
    print("  -> fbcca_config_generated.h")

    lut = build_reference_lut()
    emit_ref_lut_header(
        lut, out_dir / "fbcca_ref_lut_generated.h",
        "FBCCA_REF_LUT_GENERATED_H", "FBCCA_REF_LUT",
    )
    print("  -> fbcca_ref_lut_generated.h")

    npz = load_processed_npz(args.processed_npz)
    vectors = get_fold_test_vectors(npz)
    save_test_vectors(vectors, out_dir)
    print(f"  -> {len(vectors)} test vectors")

    refs = {}
    for i, v in enumerate(vectors):
        scores = fbcca_scores_for_trial(v, cfg)
        refs[f"idx{i}"] = {
            "scores": [float(s) for s in scores],
            "argmax": int(np.argmax(scores)),
        }
    write_host_reference(out_dir, refs)
    print("  -> host_reference.json")

    write_metadata(out_dir, "fbcca", {
        "n_bands": len(cfg.passband_hz),
        "cca_reg": cfg.cca_reg,
        "n_harmonics": cfg.n_harmonics,
    })
    print("  -> metadata.json")
    print("FBCCA export complete.")


if __name__ == "__main__":
    main()
