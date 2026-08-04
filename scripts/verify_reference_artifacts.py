#!/usr/bin/env python3
"""Verify retained MCU reference artifacts under artifacts/reference/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts" / "reference",
    )
    ap.add_argument("--skip-checksum", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from src.artifacts.reference import verify_reference_artifacts

    report = verify_reference_artifacts(
        args.artifact_root,
        verify_checksum=not args.skip_checksum,
    )
    print(f"entries={report['n_entries']}")
    print(f"checkpoints={report['n_checkpoints']}")
    print(f"tflite={report['n_tflite']}")
    print(f"analytical={report['n_analytical']}")
    print(f"preprocessing={report['n_preprocessing']}")
    if report["errors"]:
        print(f"errors={len(report['errors'])}")
        for err in report["errors"][:20]:
            print(f"  - {err}")
        if len(report["errors"]) > 20:
            print(f"  ... {len(report['errors']) - 20} more")
        return 1
    print("OK reference artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
