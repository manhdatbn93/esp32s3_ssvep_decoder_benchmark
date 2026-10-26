#!/usr/bin/env python3
"""
Preprocess 12JFPM dataset and optionally verify against a reference artifact.

Usage:
  python -m src.training_entry.run_preprocess_12jfpm --config configs/protocol/preprocess_12jfpm.yaml
  python -m src.training_entry.run_preprocess_12jfpm --config configs/protocol/preprocess_12jfpm.yaml --verify-ref /path/to/processed.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.datasets.preprocess_12jfpm import (
    PreprocessConfig,
    config_from_yaml,
    load_processed_npz,
    run_preprocessing,
)


def parity_check(
    new_path: Path,
    ref_path: Path,
) -> dict:
    """Compare new processed.npz against a reference. Returns a report dict."""
    new = np.load(new_path, allow_pickle=True)
    ref = np.load(ref_path, allow_pickle=True)

    report: dict = {"status": "PASS", "details": {}}

    for key in ["X", "y", "subject"]:
        if key not in new or key not in ref:
            report["status"] = "FAIL"
            report["details"][key] = f"missing in {'new' if key not in new else 'ref'}"
            continue

        a, b = new[key], ref[key]

        if a.shape != b.shape:
            report["status"] = "FAIL"
            report["details"][key] = {
                "shape_new": list(a.shape),
                "shape_ref": list(b.shape),
                "error": "shape mismatch",
            }
            continue

        if key == "subject":
            match = all(str(x) == str(y) for x, y in zip(a, b))
            report["details"][key] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "all_match": match,
            }
            if not match:
                report["status"] = "FAIL"
            continue

        if a.dtype != b.dtype:
            report["details"][key] = {
                "dtype_new": str(a.dtype),
                "dtype_ref": str(b.dtype),
                "note": "dtype mismatch (comparing values anyway)",
            }

        exact = np.array_equal(a, b)
        max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        mean_abs_diff = float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))

        report["details"][key] = {
            "shape": list(a.shape),
            "dtype_new": str(a.dtype),
            "dtype_ref": str(b.dtype),
            "exact_equal": exact,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        }

        if not exact and max_abs_diff > 1e-5:
            report["status"] = "FAIL"

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess 12JFPM dataset")
    ap.add_argument(
        "--config", type=Path,
        default=Path("configs/protocol/preprocess_12jfpm.yaml"),
        help="YAML config file",
    )
    ap.add_argument(
        "--raw-data-root", type=Path, default=None,
        help="Folder containing s1.mat .. s10.mat (overrides data.raw_data_root)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory",
    )
    ap.add_argument(
        "--verify-ref", type=Path, default=None,
        help="Path to reference processed.npz for parity check",
    )
    args = ap.parse_args()

    if args.config.is_file():
        cfg = config_from_yaml(args.config)
    else:
        print(f"[WARN] Config not found: {args.config}; using defaults")
        cfg = PreprocessConfig()

    if args.raw_data_root:
        cfg.raw_data_root = args.raw_data_root
    if args.output_dir:
        cfg.output_dir = args.output_dir

    print(f"Running 12JFPM preprocessing...")
    print(f"  onset_sample_0 : {cfg.onset_sample_0}")
    print(f"  window_length  : {cfg.window_length}")
    print(f"  normalize      : {cfg.normalize}")
    print(f"  subjects       : {cfg.subjects}")
    print(f"  output_dir     : {cfg.output_dir or 'data/processed/12JFPM'}")

    meta = run_preprocessing(cfg)

    print(f"\nPreprocessing complete.")
    print(json.dumps(meta, indent=2))

    if args.verify_ref:
        ref = Path(args.verify_ref)
        if not ref.is_file():
            print(f"\n[WARN] Reference file not found: {ref}")
            return

        npz_path = Path(meta["output_file"])
        print(f"\nRunning parity check against: {ref}")
        report = parity_check(npz_path, ref)
        print(json.dumps(report, indent=2))

        if report["status"] == "PASS":
            print("\nPARITY CHECK: PASS")
        else:
            print("\nPARITY CHECK: FAIL")
            sys.exit(1)


if __name__ == "__main__":
    main()
