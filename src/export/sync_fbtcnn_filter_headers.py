#!/usr/bin/env python3
"""
Copy identical fbtcnn_filter_generated.h into every FBtCNN export directory.

The filter SOS coefficients are subject-independent; CMake includes
export/12jfpm_fbtcnn_{TARGET_FOLD_SUBJECT_TAG}/ for each build.

Usage (repo root):
  python -m export.sync_fbtcnn_filter_headers

Optionally regenerate first:
  python export/gen_fbtcnn_filter_header.py
  python -m export.sync_fbtcnn_filter_headers
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_STRICT_SUBDIR = "strict_loso_inner_val"


def _fbtcnn_export_dirs(ex: Path) -> list[Path]:
    """All FBtCNN export folders (legacy flat + strict_loso_inner_val tree)."""
    dirs: list[Path] = []
    for root in (ex, ex / _STRICT_SUBDIR):
        if not root.is_dir():
            continue
        dirs.extend(
            sorted(
                d for d in root.iterdir()
                if d.is_dir() and d.name.startswith("12jfpm_fbtcnn_")
            )
        )
    return dirs


def _canonical_header(ex: Path) -> Path | None:
    """Prefer legacy fold01_val_s1; else strict fold01_test_s1."""
    candidates = [
        ex / "12jfpm_fbtcnn_fold01_val_s1" / "fbtcnn_filter_generated.h",
        ex / _STRICT_SUBDIR / "12jfpm_fbtcnn_fold01_test_s1" / "fbtcnn_filter_generated.h",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen",
        action="store_true",
        help="Run gen_fbtcnn_filter_header.py before syncing.",
    )
    args = ap.parse_args()

    if args.regen:
        gen = _REPO / "export" / "gen_fbtcnn_filter_header.py"
        r = subprocess.run([sys.executable, str(gen)], cwd=str(_REPO))
        if r.returncode != 0:
            return r.returncode

    ex = _REPO / "export"
    canonical = _canonical_header(ex)
    if canonical is None:
        print("ERROR: canonical fbtcnn_filter_generated.h missing.", file=sys.stderr)
        print("Run: python export/gen_fbtcnn_filter_header.py", file=sys.stderr)
        print("  (writes export/12jfpm_fbtcnn_fold01_val_s1/ or copy to strict tree)", file=sys.stderr)
        return 1

    dest_dirs = _fbtcnn_export_dirs(ex)
    if not dest_dirs:
        print("No export/**/12jfpm_fbtcnn_* directories found.", file=sys.stderr)
        return 1

    n = 0
    for d in dest_dirs:
        dst = d / "fbtcnn_filter_generated.h"
        try:
            if canonical.resolve() == dst.resolve():
                print(f"SKIP (canonical) {d.name}/fbtcnn_filter_generated.h")
                n += 1
                continue
        except OSError:
            pass
        shutil.copy2(canonical, dst)
        n += 1
        print(f"OK {d.name}/fbtcnn_filter_generated.h")

    print(f"Synced {n} directories from {canonical.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
