#!/usr/bin/env python3
"""
Recompute canonical held-out s1 host-side classification accuracy for all
replay-aligned backends using the same trial contract as TCP replay:

  - Raw 12JFPM .mat for subject s1
  - Per trial: crop raw[38:550] -> [512, 8], channel z-score (_preprocess_zscore)
  - Trial order: class 0..11, trial 0..14 (180 trials) — matches replay_client.run_session

This is the definitive host baseline for comparison with on-device replay when
--compare uses HostReference with the same backend name (same TFLite artifact
for int8/f32).

Usage (repo root):
  python tools/replay/canonical_s1_host_accuracy.py
  python tools/replay/canonical_s1_host_accuracy.py --mat path/to/s1.mat

Outputs:
  results/canonical_s1_host_acc/host_s1_backend_accuracy.json
  results/canonical_s1_host_acc/host_s1_backend_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_REPLAY_DIR = Path(__file__).resolve().parent
for _p in (_REPO, _REPLAY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from host_reference import HostReference, _preprocess_zscore  # noqa: E402
from replay_client import (  # noqa: E402
    _build_trial_plan,
    _default_mat_path,
    extract_raw_trial,
)
from src.datasets.parse_12jfpm import read_subject  # noqa: E402

N_CLASSES = 12
N_TRIALS_PER_CLASS = 15

# Backend CLI names accepted by replay_client / replay_proto (decoder_id_from_name).
DEFAULT_BACKENDS = [
    "cca",
    "fbcca",
    "msi",
    "trca",
    "eegnet_f32",
    "eegnet_int8",
    "ccnn_f32",
    "ccnn_int8",
    "convca_f32",
    "fbtcnn_f32",
    "fbtcnn_int8",
]


def _git_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _variant_tag(backend: str) -> str:
    b = backend.lower()
    if "_int8" in b or b.endswith("_i8") or "int8" in b:
        return "int8"
    if "_f32" in b or "float32" in b or b in ("cca", "fbcca", "msi", "trca"):
        if b in ("cca", "fbcca", "msi", "trca"):
            return "classical"
        return "float32"
    return "unknown"


def eval_backend(
    backend: str,
    *,
    eeg: np.ndarray,
    subject_id: str,
    plan: list[tuple[int, int]],
) -> dict:
    """Return result dict; on failure include error string and zero trials."""
    out: dict = {
        "backend": backend,
        "variant": _variant_tag(backend),
        "n_trials": len(plan),
        "correct": 0,
        "accuracy": 0.0,
        "accuracy_pct": 0.0,
        "artifact_path": None,
        "status": "ok",
        "error": None,
        "comparable_to_replay_compare": True,
    }
    try:
        ref = HostReference(backend, subject=subject_id)
        out["artifact_path"] = ref.artifact_path
    except Exception as e:
        out["status"] = "failed_init"
        out["error"] = str(e)[:500]
        out["comparable_to_replay_compare"] = False
        return out

    correct = 0
    try:
        for class_id, trial_idx in plan:
            raw = extract_raw_trial(eeg, class_id, trial_idx)
            crop = raw[38:550, :].astype(np.float32)
            crop_z = _preprocess_zscore(crop)
            if ref.uses_indexed:
                scores = ref.run(crop_z, class_id=class_id, trial_idx=trial_idx)
            else:
                scores = ref.run(crop_z)
            pred = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            correct += int(pred == class_id)
    except Exception as e:
        ref.cleanup()
        out["status"] = "failed_eval"
        out["error"] = str(e)[:500]
        out["comparable_to_replay_compare"] = False
        return out

    ref.cleanup()
    n = len(plan)
    out["correct"] = correct
    out["accuracy"] = correct / n if n else 0.0
    out["accuracy_pct"] = round(100.0 * correct / n, 4) if n else 0.0

    if backend.strip().lower() == "trca":
        out["evaluation_note"] = (
            "HostReference('trca') returns rows from replay_reference.npz by "
            "(class_id, trial_idx); per-trial z-scored input from .mat is ignored. "
            "Accuracy is therefore exactly NPZ argmax vs label in class-major order. "
            "Classical LOSO JSON (training/run_classical_loso_12jfpm) may differ "
            "slightly (e.g. 101/180 vs 100/180 for s1) because that harness uses "
            "src/benchmarks/trca_12jfpm rather than export/export_trca_constants SOS "
            "fit/predict used to build the replay artifact."
        )
        out["matches_classical_loso_fold1"] = False

    return out


def _load_loso_fold1_s1(repo: Path) -> dict | None:
    """Optional cross-check: PyTorch LOSO summary fold 1 val s1."""
    p = repo / "results" / "dl_loso_12jfpm_summary.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return None
        rows = {}
        for entry in data:
            model = entry.get("model")
            for fold in entry.get("per_fold") or []:
                if fold.get("fold") == 1 and fold.get("val_subject") == "s1":
                    rows[model] = {
                        "correct": fold.get("correct"),
                        "n_trials": fold.get("n_trials"),
                        "accuracy": fold.get("accuracy"),
                    }
                    break
        return rows
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mat",
        type=Path,
        default=None,
        help="Path to s1.mat (default: discover like replay_client)",
    )
    ap.add_argument(
        "--backends",
        nargs="*",
        default=DEFAULT_BACKENDS,
        help="Backend names (replay CLI style)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=_REPO / "results" / "canonical_s1_host_acc",
        help="Output directory",
    )
    args = ap.parse_args()

    mat = args.mat
    if mat is None:
        mat = _default_mat_path()
    if mat is None or not mat.is_file():
        print("ERROR: need s1.mat (--mat or default path).", file=sys.stderr)
        return 1
    mat = mat.resolve()

    sid = mat.stem.lower()
    if not sid.startswith("s"):
        sid = "s" + sid.lstrip("s")

    rec = read_subject(sid, data_root=mat.parent)
    eeg = np.asarray(rec["eeg"])
    n_trials_per_class = eeg.shape[3]
    plan = _build_trial_plan(N_CLASSES, n_trials_per_class, max_trials=None)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for backend in args.backends:
        print(f"  Evaluating {backend}...", flush=True)
        results.append(
            eval_backend(backend, eeg=eeg, subject_id=sid, plan=plan)
        )

    doc = {
        "schema": "canonical_s1_host_accuracy_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short(),
        "subject": sid,
        "mat_file": str(mat),
        "n_trials": len(plan),
        "trial_order": "class_major: for c in 0..11, for t in 0..(n_trials_per_class-1)",
        "preprocess": {
            "onset_sample": 38,
            "window": 512,
            "normalize": "channel_zscore (host_reference._preprocess_zscore)",
            "matches_replay_client_compare": True,
        },
        "backends": results,
        "cross_check": {
            "loso_pytorch_fold1_val_s1": _load_loso_fold1_s1(_REPO),
            "note": (
                "LOSO JSON is PyTorch checkpoint val, not TFLite. "
                "Use only as rough consistency check for DL f32."
            ),
        },
        "documentation_refs": {
            "replay_compare": "tools/replay/replay_client.py",
            "host_reference": "tools/replay/host_reference.py",
            "quantization_log_table": "project_context/EXPERIMENT_LOG.md",
        },
    }

    json_path = outdir / "host_s1_backend_accuracy.json"
    json_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")

    csv_path = outdir / "host_s1_backend_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "backend", "variant", "n_trials", "correct", "accuracy_pct",
            "status", "artifact_path", "comparable_to_replay_compare", "error",
        ])
        for r in results:
            w.writerow([
                r["backend"],
                r["variant"],
                r["n_trials"],
                r["correct"],
                f'{r["accuracy_pct"]:.4f}',
                r["status"],
                r.get("artifact_path") or "",
                r["comparable_to_replay_compare"],
                r.get("error") or "",
            ])

    print(f"\nWrote {json_path.relative_to(_REPO)}")
    print(f"Wrote {csv_path.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
