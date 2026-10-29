#!/usr/bin/env python3
"""Public training entry wrapper. Writes only under runs/."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TREES = (
    ROOT / "artifacts" / "archived_readonly",
    ROOT / "artifacts" / "reference",
)
_CONFIG = {
    "ccnn": ROOT / "configs/training/ccnn_12jfpm_strict_loso.yaml",
    "convca": ROOT / "configs/training/convca_12jfpm_strict_loso.yaml",
    "eegnet": ROOT / "configs/training/eegnet_12jfpm_strict_loso.yaml",
    "fbtcnn": ROOT / "configs/training/fbtcnn_12jfpm_strict_loso.yaml",
    "ssvepformer": ROOT / "configs/training/ssvepformer_12jfpm_strict_loso.yaml",
    "ssvepnet": ROOT / "configs/training/ssvepnet_12jfpm_strict_loso.yaml",
}


def training_results_arg(output_root: Path, model: str, fold: int | None) -> Path:
    """Directory for a multi-fold run, or one JSON file for a single fold.

    A single-fold path ends in ``.json``, which the strict LOSO runner writes
    directly. That avoids replacing ``runs/<model>/<model>_12jfpm.json``.
    """
    if fold is None:
        return output_root / model
    return output_root / model / "training" / f"fold{fold:02d}.json"


def build_train_command(
    *,
    model: str,
    fold: int | None,
    output_root: Path,
    dataset_root: Path,
    config: Path,
) -> list[str]:
    npz = dataset_root / "processed.npz"
    if not npz.is_file():
        npz = dataset_root if dataset_root.suffix == ".npz" else dataset_root / "processed" / "12JFPM" / "processed.npz"
    cmd = [
        sys.executable, "-m", "src.training_entry.run_dl_loso_12jfpm",
        "--model", model,
        "--config", str(config),
        "--protocol", "strict_loso_inner_val",
        "--checkpoint-dir", str(output_root / model / "checkpoints"),
        "--results-dir", str(training_results_arg(output_root, model, fold)),
    ]
    if npz.is_file():
        cmd.extend(["--npz", str(npz)])
    if fold is not None:
        cmd.extend(["--subjects", f"s{fold}"])
    return cmd


def refuse_protected_overwrite(path: Path) -> None:
    path = path.resolve()
    for protected in PROTECTED_TREES:
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            raise SystemExit(f"Refusing to write inside protected artifact tree: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(_CONFIG))
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--output-root", type=Path, default=Path("runs"))
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve config path and exit")
    args = ap.parse_args()
    refuse_protected_overwrite(args.output_root)

    config = _CONFIG[args.model]
    if not config.is_file():
        raise SystemExit(f"Missing config: {config}")
    if args.dry_run:
        print(f"OK {args.model}: {config.relative_to(ROOT).as_posix()}")
        return

    cmd = build_train_command(
        model=args.model,
        fold=args.fold,
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        config=config,
    )
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
