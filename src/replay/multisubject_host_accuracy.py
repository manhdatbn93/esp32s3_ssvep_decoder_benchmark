#!/usr/bin/env python3
"""
Multi-subject host-side accuracy for the replay-aligned deployed subset.

Runs the same HostReference evaluation as canonical_s1_host_accuracy.py but
loops over a list of held-out subjects (default: s1..s10). For each
(subject, backend) pair it:

  - loads raw sN.mat through src.datasets.parse_12jfpm.read_subject
  - builds the canonical 180-trial plan (class-major)
  - crops raw[38:550], channel z-scores, runs HostReference
  - computes correct/180 for that subject+backend
  - writes per-subject JSON + CSV and a matrix summary across all subjects

Outputs (default: results/replay_multisubject/host_accuracy/):

  per_subject/host_<subject>_backend_accuracy.json
  per_subject/host_<subject>_backend_accuracy.csv
  host_accuracy_matrix.csv   (rows=backend, cols=s1..s10, mean, std)
  host_accuracy_summary.json (aggregate with means/stds)

For DL backends, the corresponding subject/fold TFLite artifact must have
been exported beforehand with:

  python -m tools.export.export_all_subjects --subjects s<N> --backends ...

TRCA also requires the per-subject replay_reference.npz (export orchestrator
generates it automatically).

Usage (repo root):

  python tools/replay/multisubject_host_accuracy.py                    # s1..s10
  python tools/replay/multisubject_host_accuracy.py --subjects s1 s2
  python tools/replay/multisubject_host_accuracy.py --backends cca fbcca msi trca
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
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

from canonical_s1_host_accuracy import (  # noqa: E402
    DEFAULT_BACKENDS,
    eval_backend,
)
from replay_client import _build_trial_plan  # noqa: E402
from src.datasets.parse_12jfpm import discover_subject_paths, read_subject  # noqa: E402
from export.subject_paths import all_subjects  # noqa: E402

N_CLASSES = 12


def _git_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _per_subject_doc(subject: str, mat_path: Path, results: list[dict],
                     n_trials: int) -> dict:
    return {
        "schema": "multisubject_host_accuracy_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short(),
        "subject": subject,
        "mat_file": str(mat_path),
        "n_trials": n_trials,
        "trial_order": "class_major: for c in 0..11, for t in 0..14",
        "preprocess": {
            "onset_sample": 38,
            "window": 512,
            "normalize": "channel_zscore (host_reference._preprocess_zscore)",
            "matches_replay_client_compare": True,
        },
        "backends": results,
    }


def _write_per_subject(outdir: Path, subject: str, mat_path: Path,
                       results: list[dict], n_trials: int) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    doc = _per_subject_doc(subject, mat_path, results, n_trials)

    json_path = outdir / f"host_{subject}_backend_accuracy.json"
    json_path.write_text(json.dumps(doc, indent=2, default=str) + "\n",
                         encoding="utf-8")

    csv_path = outdir / f"host_{subject}_backend_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "backend", "variant", "n_trials", "correct", "accuracy_pct",
            "status", "artifact_path", "comparable_to_replay_compare", "error",
        ])
        for r in results:
            w.writerow([
                r["backend"], r["variant"], r["n_trials"], r["correct"],
                f'{r["accuracy_pct"]:.4f}', r["status"],
                r.get("artifact_path") or "",
                r["comparable_to_replay_compare"], r.get("error") or "",
            ])
    return json_path, csv_path


def _write_matrix(outdir: Path, backends: list[str], subjects: list[str],
                  data: dict[tuple[str, str], dict]) -> Path:
    """
    Rows = backend; columns = s1..sN + mean + std (in %).
    Cell = accuracy_pct (or blank on failure).
    """
    p = outdir / "host_accuracy_matrix.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend"] + subjects + ["mean_pct", "std_pct", "n_ok"])
        for b in backends:
            row = [b]
            values = []
            for s in subjects:
                r = data.get((s, b))
                if r and r.get("status") == "ok":
                    pct = float(r["accuracy_pct"])
                    row.append(f"{pct:.4f}")
                    values.append(pct)
                else:
                    row.append("")
            if values:
                row.append(f"{statistics.fmean(values):.4f}")
                row.append(f"{(statistics.pstdev(values) if len(values) > 1 else 0.0):.4f}")
                row.append(str(len(values)))
            else:
                row.extend(["", "", "0"])
            w.writerow(row)
    return p


def _summarize(backends: list[str], subjects: list[str],
               data: dict[tuple[str, str], dict]) -> dict:
    per_backend: list[dict] = []
    for b in backends:
        vals = []
        failed = []
        for s in subjects:
            r = data.get((s, b))
            if r and r.get("status") == "ok":
                vals.append(float(r["accuracy_pct"]))
            else:
                err = (r or {}).get("error") or (r or {}).get("status") or "missing"
                failed.append({"subject": s, "error": err})
        per_backend.append({
            "backend": b,
            "n_subjects_ok": len(vals),
            "mean_pct": round(statistics.fmean(vals), 4) if vals else None,
            "std_pct": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4)
            if vals else None,
            "min_pct": round(min(vals), 4) if vals else None,
            "max_pct": round(max(vals), 4) if vals else None,
            "failed": failed,
        })

    per_subject: list[dict] = []
    for s in subjects:
        row = {"subject": s}
        for b in backends:
            r = data.get((s, b))
            row[b] = (round(float(r["accuracy_pct"]), 4)
                      if r and r.get("status") == "ok" else None)
        per_subject.append(row)

    return {
        "schema": "multisubject_host_accuracy_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short(),
        "subjects": subjects,
        "backends": backends,
        "per_backend": per_backend,
        "per_subject": per_subject,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Subjects (default s1..s10).")
    ap.add_argument("--backends", nargs="*", default=DEFAULT_BACKENDS,
                    help="Backend names (replay CLI style).")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Folder containing s1.mat..s10.mat. "
                         "Default: autodetect via src.datasets.parse_12jfpm.")
    ap.add_argument("--outdir", type=Path,
                    default=_REPO / "results" / "replay_multisubject" / "host_accuracy",
                    help="Output directory.")
    args = ap.parse_args()

    subjects = [s.strip().lower() for s in (args.subjects or list(all_subjects()))]
    backends = list(args.backends)

    avail = discover_subject_paths(args.data_root)
    missing = [s for s in subjects if s not in avail]
    if missing:
        print(f"ERROR: raw .mat not found for {missing}. "
              f"Data root used: {list(avail.values())[0].parent if avail else '?'}",
              file=sys.stderr)
        return 1

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    per_sub_dir = outdir / "per_subject"

    print("=" * 64)
    print(" Multi-subject host-side accuracy")
    print("=" * 64)
    print(f" Subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f" Backends ({len(backends)}): {', '.join(backends)}")
    print(f" Outdir  : {outdir.relative_to(_REPO)}")
    print("=" * 64)

    data: dict[tuple[str, str], dict] = {}
    for subject in subjects:
        mat_path = avail[subject]
        print(f"\n[{subject}] {mat_path.name}")
        rec = read_subject(subject, data_root=mat_path.parent)
        eeg = np.asarray(rec["eeg"])
        n_trials_per_class = eeg.shape[3]
        plan = _build_trial_plan(N_CLASSES, n_trials_per_class, None)

        subj_results = []
        for backend in backends:
            print(f"  -> {backend}", flush=True)
            r = eval_backend(backend, eeg=eeg, subject_id=subject, plan=plan)
            subj_results.append(r)
            data[(subject, backend)] = r
            if r.get("status") == "ok":
                print(f"     {r['correct']}/{r['n_trials']} "
                      f"({r['accuracy_pct']:.2f}%)")
            else:
                print(f"     {r['status']}: {(r.get('error') or '')[:140]}")

        _write_per_subject(per_sub_dir, subject, mat_path, subj_results, len(plan))

    matrix_path = _write_matrix(outdir, backends, subjects, data)
    summary = _summarize(backends, subjects, data)
    sj = outdir / "host_accuracy_summary.json"
    sj.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print("\nWrote:")
    print(f"  {per_sub_dir.relative_to(_REPO)}/*.json/.csv")
    print(f"  {matrix_path.relative_to(_REPO)}")
    print(f"  {sj.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
