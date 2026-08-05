#!/usr/bin/env python3
"""Public export entry. Default output under artifacts/generated/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neural import MODEL_REGISTRY
PROTECTED_TREES = (
    ROOT / "artifacts" / "archived_readonly",
    ROOT / "artifacts" / "reference",
)
SUPPORTED = {
    name: quant
    for name, quant in {
        "ccnn": ("f32", "int8"),
        "convca": ("f32",),
        "eegnet": ("f32", "int8"),
        "fbtcnn": ("f32", "int8"),
        "ssvepformer": ("f32",),
        "ssvepnet": ("f32",),
    }.items()
    if name in MODEL_REGISTRY
}


def refuse_protected_overwrite(path: Path) -> None:
    path = path.resolve()
    for protected in PROTECTED_TREES:
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            raise SystemExit(f"Refusing to write inside protected artifact tree: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SUPPORTED))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--quantization", choices=["f32", "int8"], default="f32")
    ap.add_argument("--output", type=Path, default=Path("artifacts/generated"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without exporting weights",
    )
    args = ap.parse_args()
    refuse_protected_overwrite(args.output)

    if args.quantization not in SUPPORTED[args.model]:
        raise SystemExit(
            f"Unsupported quantization {args.quantization} for model {args.model}. "
            f"Allowed: {', '.join(SUPPORTED[args.model])}"
        )
    if args.fold < 1 or args.fold > 10:
        raise SystemExit("--fold must be in 1..10")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")

    out = Path(args.output)
    if args.dry_run:
        ctor = MODEL_REGISTRY[args.model]
        print("dry-run ok")
        print("constructor:", ctor.__module__ + "." + ctor.__name__)
        print("model:", args.model)
        print("fold:", args.fold)
        print("checkpoint:", args.checkpoint)
        print("quantization:", args.quantization)
        print("output:", out)
        return

    print("Export arguments validated.")
    print("Invoke the model-specific exporter under src/export/ for", args.model)
    print("Default destination:", out)


if __name__ == "__main__":
    main()
