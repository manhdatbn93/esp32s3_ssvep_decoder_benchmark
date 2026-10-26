#!/usr/bin/env python3
"""Maintainer tool: write MANIFEST.sha256 for the frozen evidence and expected values."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIRS = ("evidence", "expected")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_files(root: Path = ROOT) -> list[Path]:
    return sorted(p for d in FROZEN_DIRS for p in (root / d).rglob("*") if p.is_file())


def main() -> int:
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in frozen_files()]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} entries to MANIFEST.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
