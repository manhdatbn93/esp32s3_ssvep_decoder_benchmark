#!/usr/bin/env python3
"""Prepare processed 12JFPM NPZ from local raw data."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/protocol/preprocess_12jfpm.yaml")
    ap.add_argument("--raw-data-root", type=Path, default=None, help="Folder containing s1.mat .. s10.mat")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--verify-ref", type=Path, default=None)
    args = ap.parse_args()
    cmd = [sys.executable, "-m", "src.training_entry.run_preprocess_12jfpm", "--config", str(args.config)]
    if args.raw_data_root:
        cmd.extend(["--raw-data-root", str(args.raw_data_root.resolve())])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir.resolve())])
    if args.verify_ref:
        cmd.extend(["--verify-ref", str(args.verify_ref.resolve())])
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
