#!/usr/bin/env python3
"""
Generate host-side TRCA reference scores for all trials of a held-out subject.

TRCA is training-based — the host reference must fit on the same fold's
training data, then predict all held-out trials using the SOS-parity
filtfilt path (matching the firmware's `trca_forward.c`).

This script produces an NPZ artifact that `host_reference.py` loads at
replay time for per-trial parity comparison.

Artifact path:
  export/12jfpm_trca_fold01_val_<subject>/replay_reference.npz

Contents:
  scores      — (180, 12) float32, score vector per trial
  argmax      — (180,)    int32,   predicted class per trial
  class_ids   — (180,)    int32,   true class per trial
  trial_idxs  — (180,)    int32,   within-class trial index per trial
  subject     — str,       held-out subject ID (e.g. "s1")
  fold        — int,       fold number (1-based)
  backend     — str,       "trca"
  n_classes   — int
  n_trials_per_class — int

Trial order matches replay_client's _build_trial_plan:
  for class_id in 0..11:
      for trial_idx in 0..14:
          (class_id, trial_idx)

Usage:
  python tools/replay/generate_trca_reference.py             # default: val=s1
  python tools/replay/generate_trca_reference.py --subject s2 --fold 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from export.export_trca_constants import (
    _apply_filter_bank_sos,
    _trca_fit_sos,
    _trca_predict_sos,
)
from export.common import compute_cheby_filter_bank, load_processed_npz
from src.benchmarks.trca_12jfpm import TRCAConfig

N_CLASSES = 12
N_TRIALS_PER_CLASS = 15



def generate_trca_reference(
    subject: str = "s1",
    fold: int = 1,
    processed_npz: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Generate and save TRCA reference for all trials of the held-out subject."""
    cfg = TRCAConfig()

    npz = load_processed_npz(processed_npz)
    X, y, subjects = npz["X"], npz["y"], npz.get("subjects", npz.get("subject"))

    mask_val = np.array([str(s) == subject for s in subjects])
    n_val = int(mask_val.sum())
    if n_val == 0:
        raise ValueError(f"Subject {subject!r} not found in dataset")

    X_train = X[~mask_val]
    y_train = y[~mask_val]

    bank_params = {
        "passband_hz": list(cfg.passband_hz),
        "stopband_hz": list(cfg.stopband_hz),
        "highcut_pass_hz": cfg.highcut_pass_hz,
        "highcut_stop_hz": cfg.highcut_stop_hz,
        "gpass_db": cfg.gpass_db,
        "gstop_db": cfg.gstop_db,
        "cheby_ripple_db": cfg.cheby_ripple_db,
    }
    banks = compute_cheby_filter_bank(bank_params)

    print(f"  Fitting TRCA on fold{fold:02d} (val={subject}), "
          f"{X_train.shape[0]} train trials, SOS-parity filtfilt...")
    t0 = time.time()
    templates, w_arr = _trca_fit_sos(X_train, y_train, cfg, banks)
    t_fit = time.time() - t0
    print(f"  Fit complete ({t_fit:.1f}s). Templates: {templates.shape}, W: {w_arr.shape}")

    X_val = X[mask_val]
    y_val = y[mask_val]

    eeg_4d = X_val.reshape(N_CLASSES, N_TRIALS_PER_CLASS, X_val.shape[1], X_val.shape[2])

    n_total = N_CLASSES * N_TRIALS_PER_CLASS
    all_scores = np.zeros((n_total, N_CLASSES), dtype=np.float32)
    all_argmax = np.zeros(n_total, dtype=np.int32)
    all_class_ids = np.zeros(n_total, dtype=np.int32)
    all_trial_idxs = np.zeros(n_total, dtype=np.int32)

    print(f"  Computing reference scores for {n_total} trials...")
    t0 = time.time()
    idx = 0
    for c in range(N_CLASSES):
        for t in range(N_TRIALS_PER_CLASS):
            trial_tc = eeg_4d[c, t]  # (T, C) — already cropped+z-scored in processed.npz
            trial_batch = trial_tc[np.newaxis, ...].astype(np.float64)  # (1, T, C)
            preds, score_rows = _trca_predict_sos(
                trial_batch, templates, w_arr, cfg, banks
            )
            scores = np.array(score_rows[0], dtype=np.float32)
            all_scores[idx] = scores
            all_argmax[idx] = int(np.argmax(scores))
            all_class_ids[idx] = c
            all_trial_idxs[idx] = t
            idx += 1

            if (idx % 30) == 0 or idx == n_total:
                print(f"    [{idx:3d}/{n_total}] ...", flush=True)

    t_pred = time.time() - t0
    print(f"  Prediction complete ({t_pred:.1f}s)")

    if out_dir is None:
        out_dir = _REPO / "export" / f"12jfpm_trca_fold{fold:02d}_val_{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "replay_reference.npz"
    np.savez_compressed(
        str(out_path),
        scores=all_scores,
        argmax=all_argmax,
        class_ids=all_class_ids,
        trial_idxs=all_trial_idxs,
        subject=subject,
        fold=fold,
        backend="trca",
        n_classes=N_CLASSES,
        n_trials_per_class=N_TRIALS_PER_CLASS,
    )
    print(f"  Saved: {out_path}")
    print(f"  Shape: scores={all_scores.shape}, argmax={all_argmax.shape}")

    n_correct = int((all_argmax == all_class_ids).sum())
    print(f"  Host TRCA accuracy: {n_correct}/{n_total} "
          f"({100*n_correct/n_total:.1f}%)")

    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate TRCA host-side replay reference (all trials)",
    )
    ap.add_argument("--subject", default="s1",
                    help="Held-out subject ID (default: s1)")
    ap.add_argument("--fold", type=int, default=1,
                    help="Fold number (default: 1)")
    ap.add_argument("--processed-npz", type=Path, default=None,
                    help="Path to processed.npz")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: export/12jfpm_trca_fold<N>_val_<subj>)")
    args = ap.parse_args()

    print(f"Generating TRCA replay reference for {args.subject} (fold {args.fold})...")
    generate_trca_reference(
        subject=args.subject,
        fold=args.fold,
        processed_npz=args.processed_npz,
        out_dir=args.out_dir,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
