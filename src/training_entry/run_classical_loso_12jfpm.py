#!/usr/bin/env python3
"""Run LOSO evaluation for classical SSVEP backends on 12JFPM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation.cca_12jfpm import CCAConfig, cca_predict_batch
from src.evaluation.fbcca_12jfpm import FBCCAConfig, fbcca_predict_batch
from src.evaluation.msi_12jfpm import MSIConfig, msi_predict_batch
from src.evaluation.trca_12jfpm import TRCAConfig, trca_predict_fold
from src.evaluation.loso import LOSOResult, run_loso, run_loso_trainable, save_results

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_MAP = {
    "cca": _REPO / "configs/analysis/cca_12jfpm.yaml",
    "fbcca": _REPO / "configs/analysis/fbcca_12jfpm.yaml",
    "msi": _REPO / "configs/analysis/msi_12jfpm.yaml",
    "trca": _REPO / "configs/analysis/trca_12jfpm.yaml",
}
SUBJECTS = [f"s{i}" for i in range(1, 11)]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return doc


def _load_data(npz_path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if npz_path and npz_path.is_file():
        d = np.load(npz_path, allow_pickle=True)
        subjects = d["subjects"] if "subjects" in d.files else d["subject"]
        return d["X"], d["y"], subjects
    from src.datasets.preprocess_12jfpm import load_processed_npz
    return load_processed_npz()


def _cca_config(doc: dict) -> CCAConfig:
    block = doc.get("cca", {})
    band = block.get("single_band_hz", [8.0, 55.0])
    return CCAConfig(
        sampling_rate_hz=float(block.get("sampling_rate_hz", 256.0)),
        n_harmonics=int(block.get("n_harmonics", 3)),
        single_band_hz=(float(band[0]), float(band[1])),
        butterworth_order=int(block.get("butterworth_order", 4)),
        cca_reg=float(block.get("cca_reg", 1e-4)),
    )


def _fbcca_config(doc: dict) -> FBCCAConfig:
    block = doc.get("fbcca", {})
    return FBCCAConfig(
        sampling_rate_hz=float(block.get("sampling_rate_hz", 256.0)),
        n_harmonics=int(block.get("n_harmonics", 3)),
        cca_reg=float(block.get("cca_reg", 1e-4)),
        passband_hz=tuple(float(x) for x in block.get("passband_hz", FBCCAConfig().passband_hz)),
        stopband_hz=tuple(float(x) for x in block.get("stopband_hz", FBCCAConfig().stopband_hz)),
        highcut_pass_hz=float(block.get("highcut_pass_hz", 80.0)),
        highcut_stop_hz=float(block.get("highcut_stop_hz", 90.0)),
        gpass_db=float(block.get("gpass_db", 3.0)),
        gstop_db=float(block.get("gstop_db", 40.0)),
        cheby_ripple_db=float(block.get("cheby_ripple_db", 0.5)),
    )


def _msi_config(doc: dict) -> MSIConfig:
    block = doc.get("msi", {})
    return MSIConfig(
        sampling_rate_hz=float(block.get("sampling_rate_hz", 256.0)),
        n_harmonics=int(block.get("n_harmonics", 3)),
    )


def _trca_config(doc: dict) -> TRCAConfig:
    block = doc.get("trca", {})
    defaults = TRCAConfig()
    return TRCAConfig(
        sampling_rate_hz=float(block.get("sampling_rate_hz", defaults.sampling_rate_hz)),
        n_bands=int(block.get("n_bands", defaults.n_bands)),
        ensemble=bool(block.get("ensemble", defaults.ensemble)),
        passband_hz=tuple(float(x) for x in block.get("passband_hz", defaults.passband_hz)),
        stopband_hz=tuple(float(x) for x in block.get("stopband_hz", defaults.stopband_hz)),
        highcut_pass_hz=float(block.get("highcut_pass_hz", defaults.highcut_pass_hz)),
        highcut_stop_hz=float(block.get("highcut_stop_hz", defaults.highcut_stop_hz)),
        gpass_db=float(block.get("gpass_db", defaults.gpass_db)),
        gstop_db=float(block.get("gstop_db", defaults.gstop_db)),
        cheby_ripple_db=float(block.get("cheby_ripple_db", defaults.cheby_ripple_db)),
    )


def run_cca_loso(X, y, subjects, doc: dict) -> LOSOResult:
    cfg = _cca_config(doc)
    return run_loso(
        X, y, subjects,
        method_name="cca",
        predict_batch_fn=lambda x: cca_predict_batch(x, cfg),
        subject_order=SUBJECTS,
    )


def run_fbcca_loso(X, y, subjects, doc: dict) -> LOSOResult:
    cfg = _fbcca_config(doc)
    return run_loso(
        X, y, subjects,
        method_name="fbcca",
        predict_batch_fn=lambda x: fbcca_predict_batch(x, cfg),
        subject_order=SUBJECTS,
    )


def run_msi_loso(X, y, subjects, doc: dict) -> LOSOResult:
    cfg = _msi_config(doc)
    return run_loso(
        X, y, subjects,
        method_name="msi",
        predict_batch_fn=lambda x: msi_predict_batch(x, cfg),
        subject_order=SUBJECTS,
    )


def run_trca_loso(
    X, y, subjects, doc: dict, realization: str = "deployed_sos",
) -> LOSOResult:
    cfg = _trca_config(doc)
    return run_loso_trainable(
        X, y, subjects,
        method_name="trca",
        fit_predict_fn=lambda Xtr, ytr, Xte: trca_predict_fold(
            Xtr, ytr, Xte, cfg, realization=realization,
        ),
        subject_order=SUBJECTS,
    )


_DEFAULT_OUTPUT_DIR = Path("runs/classical_loso_12jfpm")


def _resolve_output_dir(cli_output_dir: Path | None, doc: dict | None) -> Path:
    """Prefer an explicit CLI directory, then YAML output.run_dir, then the default."""
    if cli_output_dir is not None:
        return Path(cli_output_dir)
    output = doc.get("output") if isinstance(doc, dict) else None
    if isinstance(output, dict):
        run_dir = output.get("run_dir")
        if run_dir:
            return Path(run_dir)
    return _DEFAULT_OUTPUT_DIR


def print_result(r: LOSOResult) -> None:
    print(f"\n{'='*60}")
    print(f"  {r.method.upper()} LOSO — 12JFPM")
    print(f"{'='*60}")
    print(f"  Mean accuracy : {r.mean_accuracy*100:.2f}% +/- {r.std_accuracy*100:.2f}%")
    print(f"  Total         : {r.total_correct}/{r.total_trials}")
    print(f"  Elapsed       : {r.elapsed_seconds:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Classical LOSO benchmarks on 12JFPM")
    ap.add_argument("--method", choices=["cca", "fbcca", "msi", "trca", "all"], default="all")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config (default: configs/analysis/<method>_12jfpm.yaml)")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Overrides YAML output.run_dir when set.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Load configs and construct method objects only")
    ap.add_argument(
        "--trca-realization",
        choices=["deployed_sos", "screening_ba"],
        default="deployed_sos",
        help=(
            "TRCA only. deployed_sos is the published SOS filter-bank reference. "
            "screening_ba is the historical transfer-function candidate."
        ),
    )
    args = ap.parse_args()

    methods = ["cca", "fbcca", "msi", "trca"] if args.method == "all" else [args.method]
    runners = {
        "cca": run_cca_loso,
        "fbcca": run_fbcca_loso,
        "msi": run_msi_loso,
        "trca": run_trca_loso,
    }

    if args.dry_run:
        for name in methods:
            path = args.config if (args.config and len(methods) == 1) else _CONFIG_MAP[name]
            doc = _load_yaml(path)
            if name == "cca":
                _cca_config(doc)
            elif name == "fbcca":
                _fbcca_config(doc)
            elif name == "msi":
                _msi_config(doc)
            else:
                _trca_config(doc)
            out = _resolve_output_dir(args.output_dir, doc)
            print(f"OK {name}: {path}")
            print(f"output-dir: {out}")
        return

    print("Loading 12JFPM processed data...")
    X, y, subjects = _load_data(args.npz)
    print(f"  X: {X.shape}, y: {y.shape}")

    for name in methods:
        path = args.config if (args.config and len(methods) == 1) else _CONFIG_MAP[name]
        doc = _load_yaml(path)
        if name == "trca":
            label = (
                "published SOS filter bank"
                if args.trca_realization == "deployed_sos"
                else "historical transfer-function screening candidate"
            )
            print(f"TRCA realization: {args.trca_realization} ({label})")
            result = run_trca_loso(X, y, subjects, doc, realization=args.trca_realization)
        else:
            result = runners[name](X, y, subjects, doc)
        print_result(result)
        out = _resolve_output_dir(args.output_dir, doc)
        save_results(result, out)


if __name__ == "__main__":
    main()
