#!/usr/bin/env python3
"""
Export MSI constants + precomputed reference LUT for ESP32-S3 deployment.

MSI is training-free and needs only reference signals (same sin/cos as CCA/FBCCA)
plus the target frequencies.

Usage:
  python -m export.export_msi_constants [--out export/12jfpm_msi_fold01_val_s1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export.common import (
    FREQS,
    FS,
    N_HARMONICS,
    T,
    C,
    build_reference_lut,
    emit_ref_lut_header,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.benchmarks.msi_12jfpm import MSIConfig, msi_scores_for_trial

_DEFAULT_OUT = "export/12jfpm_msi_fold01_val_s1"


def _emit_msi_config_header(out_dir: Path, cfg: MSIConfig) -> Path:
    lines = [
        "/* Auto-generated MSI config for 12JFPM — do not edit. */",
        "#ifndef MSI_CONFIG_GENERATED_H",
        "#define MSI_CONFIG_GENERATED_H",
        "",
        f"#define MSI_FS_HZ {cfg.sampling_rate_hz}",
        f"#define MSI_T {T}",
        f"#define MSI_C {C}",
        f"#define MSI_N_HARM {cfg.n_harmonics}",
        f"#define MSI_R {2 * cfg.n_harmonics}",
        f"#define MSI_NUM_CLASSES {len(FREQS)}",
        "",
        f"static const double MSI_TARGET_HZ[{len(FREQS)}] = {{",
        f"  {', '.join(str(f) for f in FREQS)}",
        "};",
        "",
        "#endif /* MSI_CONFIG_GENERATED_H */",
        "",
    ]
    p = out_dir / "msi_config_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export MSI constants for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MSI export -> {out_dir}")

    cfg = MSIConfig()

    _emit_msi_config_header(out_dir, cfg)
    print("  -> msi_config_generated.h")

    lut = build_reference_lut()
    emit_ref_lut_header(
        lut, out_dir / "msi_ref_lut_generated.h",
        "MSI_REF_LUT_GENERATED_H", "MSI_REF_LUT",
    )
    print("  -> msi_ref_lut_generated.h")

    npz = load_processed_npz(args.processed_npz)
    vectors = get_fold_test_vectors(npz)
    save_test_vectors(vectors, out_dir)
    print(f"  -> {len(vectors)} test vectors")

    refs = {}
    for i, v in enumerate(vectors):
        scores = msi_scores_for_trial(v, cfg)
        refs[f"idx{i}"] = {
            "scores": [float(s) for s in scores],
            "argmax": int(np.argmax(scores)),
        }
    write_host_reference(out_dir, refs)
    print("  -> host_reference.json")

    write_metadata(out_dir, "msi", {"n_harmonics": cfg.n_harmonics})
    print("  -> metadata.json")
    print("MSI export complete.")


if __name__ == "__main__":
    main()
