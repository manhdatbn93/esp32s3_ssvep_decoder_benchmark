#!/usr/bin/env python3
"""Device replay helper for ESP32-S3 hardware."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=False, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument(
        "--artifact-set",
        choices=["reference", "generated"],
        default="generated",
    )
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/device"))
    ap.add_argument("--skip-checksum", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.artifact_set == "reference":
        if not args.backend or args.fold is None:
            raise SystemExit("--backend and --fold are required for reference replay")
        from src.artifacts.reference import resolve_analytical_bundle, resolve_tflite

        key = args.backend.strip().lower().replace("_", "-")
        if key in {"cca", "fbcca", "msi", "trca"}:
            bundle = resolve_analytical_bundle(
                key, args.fold, verify_checksum=not args.skip_checksum
            )
            print(
                f"OK reference replay {key} fold{args.fold:02d} "
                f"held_out={bundle['held_out_subject']} files={len(bundle['paths'])}"
            )
        else:
            parts = key.split("-")
            model = parts[0]
            precision = parts[1] if len(parts) > 1 else "f32"
            art = resolve_tflite(
                model, precision, args.fold, verify_checksum=not args.skip_checksum
            )
            print(
                f"OK reference replay {model}-{precision} fold{args.fold:02d} "
                f"held_out={art['held_out_subject']} file={art['path'].name}"
            )
        if args.dry_run:
            return
        print("Stage firmware inputs with scripts/generate_firmware_artifacts.py")
        print("Then build/flash firmware and use src/replay/replay_client.py")
        print("Saved runs default to results/replay/strict_loso_inner_val/.")
        print(f"Pass --outdir to write elsewhere, for example {args.output}")
        return

    print("See docs/firmware-and-replay.md and src/replay/replay_client.py")
    print("Do not commit Wi-Fi credentials.")
    if args.dataset_root:
        print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
