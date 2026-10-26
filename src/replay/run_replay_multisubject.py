#!/usr/bin/env python3
"""
Multi-subject on-device replay orchestrator.

Drives the existing `replay_client.py` + `host_reference.py` pipeline —
unchanged in contract — across a matrix of (subject, backend) pairs.

Workflow per subject:

  1. Optional --build-firmware looks for the historical embedded/replay_fw
     directory. That directory is not in this public tree, so the flag does
     not build the published project. Use docs/firmware-and-replay.md.
  2. For each backend in the deployed subset, call `replay_client.py`
     with the correct raw .mat file for that subject. Outputs go under:
       results/replay_multisubject/<backend>/<tag>_<timestamp>/
     plus the symlink-compatible default that replay_client already writes.
  3. Collect per-run summaries and assemble an aggregate matrix.

The public measurement-profile build is docs/firmware-and-replay.md.
This helper does not stage artifacts or copy sdkconfig.measurement.

Usage (repo root):

  python src/replay/run_replay_multisubject.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REPLAY_DIR = Path(__file__).resolve().parent
for _p in (_REPO, _REPLAY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from export.protocol_cli import (  # noqa: E402
    DEFAULT_PROTOCOL,
    add_protocol_arg,
    apply_protocol_side_effects,
    replay_results_root,
)
from export.subject_paths import all_subjects, fold_for_subject, tag_for  # noqa: E402
from src.datasets.parse_12jfpm import discover_subject_paths  # noqa: E402

DEFAULT_BACKENDS = [
    "cca", "fbcca", "msi", "trca",
    "eegnet_f32", "eegnet_int8",
    "ccnn_f32", "ccnn_int8",
    "convca_f32",
    "fbtcnn_f32", "fbtcnn_int8",
]

# Wi-Fi/TCP benchmark replay firmware (timing); links smoke_fw components.
FIRMWARE_DIR = _REPO / "embedded" / "replay_fw"
STRICT_EXPORT_ROOT = _REPO / "export" / "strict_loso_inner_val"


def _git_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _build_and_flash(subject: str, fold: int, *,
                     build: bool, flash: bool, port: str | None,
                     protocol: str = DEFAULT_PROTOCOL) -> dict:
    """Invoke idf.py to (re)build and optionally flash the firmware."""
    tag = tag_for(subject, fold, protocol=protocol)
    info: dict = {"subject": subject, "tag": tag, "build": build, "flash": flash}
    if not FIRMWARE_DIR.is_dir():
        info["status"] = "firmware_dir_missing"
        info["path"] = str(FIRMWARE_DIR)
        return info

    targets = []
    if build:
        targets.append("build")
    if flash:
        targets.append("flash")
    if not targets:
        info["status"] = "skipped"
        return info

    cmd = ["idf.py", f"-DTARGET_FOLD_SUBJECT_TAG={tag}"]
    if flash and port:
        cmd += ["-p", port]
    cmd += targets

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(FIRMWARE_DIR),
            capture_output=True, text=True, timeout=1800,
        )
        info["returncode"] = proc.returncode
        info["elapsed_s"] = round(time.time() - t0, 1)
        info["status"] = "ok" if proc.returncode == 0 else "failed"
        info["stdout_tail"] = proc.stdout[-800:]
        info["stderr_tail"] = proc.stderr[-800:]
    except subprocess.TimeoutExpired:
        info["status"] = "timeout"
    except FileNotFoundError:
        info["status"] = "idf_not_found"
        info["note"] = "Run in an ESP-IDF environment (export.ps1/export.sh)."
    return info


def _run_replay(*, subject: str, mat: Path, backend: str, host: str,
                port: int, compare: bool, max_trials: int | None,
                outdir: Path, verbose: bool, protocol: str) -> dict:
    """Call replay_client.py once; return status + summary lookup."""
    outdir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{tag_for(subject, protocol=protocol)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cmd = [
        sys.executable, "-m", "tools.replay.replay_client",
        "--backend", backend,
        "--host", host,
        "--port", str(port),
        "--mat", str(mat),
        "--outdir", str(outdir / run_tag),
        "--run-tag", run_tag,
        "--protocol", protocol,
        "--no-aggregate",
    ]
    if not compare:
        cmd.append("--no-compare")
    if max_trials is not None:
        cmd += ["--max-trials", str(max_trials)]
    if verbose:
        cmd.append("--verbose")

    t0 = time.time()
    info: dict = {
        "subject": subject, "backend": backend, "cmd": cmd,
        "outdir": str(outdir / run_tag),
        "run_tag": run_tag,
    }
    try:
        proc = subprocess.run(
            cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=1200,
        )
        info["returncode"] = proc.returncode
        info["elapsed_s"] = round(time.time() - t0, 1)
        info["status"] = "ok" if proc.returncode == 0 else "failed"
        info["stdout_tail"] = proc.stdout[-1000:]
        info["stderr_tail"] = proc.stderr[-1000:]

        summary_path = outdir / run_tag / "run_summary.json"
        if summary_path.is_file():
            try:
                info["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as e:
                info["summary_parse_error"] = str(e)
    except subprocess.TimeoutExpired:
        info["status"] = "timeout"
        info["elapsed_s"] = round(time.time() - t0, 1)
    return info


def _extract_row(entry: dict) -> dict:
    """Condense a per-run replay summary to a single row for aggregation."""
    s = entry.get("summary") or {}
    return {
        "subject": entry.get("subject"),
        "backend": entry.get("backend"),
        "status": entry.get("status"),
        "n_trials": s.get("n_trials"),
        "device_correct": s.get("n_correct"),
        "device_accuracy_pct": s.get("accuracy_pct"),
        "parity_argmax_agree_pct": s.get("parity_argmax_agree_pct"),
        "parity_max_abs_diff": s.get("parity_max_abs_diff"),
        "parity_mean_abs_diff": s.get("parity_mean_abs_diff"),
        "avg_infer_ms": s.get("avg_infer_ms"),
        "avg_preproc_ms": s.get("avg_preproc_ms"),
        "run_dir": entry.get("outdir"),
        "elapsed_s": entry.get("elapsed_s"),
    }


def _aggregate(results: list[dict], out_root: Path) -> Path:
    rows = [_extract_row(r) for r in results]
    agg = {
        "schema": "replay_multisubject_matrix_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short(),
        "rows": rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "replay_multisubject_matrix.json"
    p.write_text(json.dumps(agg, indent=2, default=str) + "\n", encoding="utf-8")

    import csv as _csv
    cp = out_root / "replay_multisubject_matrix.csv"
    fieldnames = list(rows[0].keys()) if rows else ["subject", "backend", "status"]
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subjects", "--subject", nargs="*", default=None,
                    dest="subjects",
                    help="Subjects (default: s1..s10).")
    ap.add_argument("--backends", nargs="*", default=DEFAULT_BACKENDS,
                    help="Backends to replay (default: full deployed subset).")
    ap.add_argument("--host", default=os.environ.get("REPLAY_HOST"),
                    help="ESP32-S3 IP (env REPLAY_HOST). Required unless --dry-run.")
    ap.add_argument("--tcp-port", type=int, default=5555)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Folder containing s1.mat .. s10.mat (required if not under data/raw/12JFPM).")
    add_protocol_arg(ap)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output root (default: results/replay_multisubject/<protocol>).")
    ap.add_argument("--compare", action="store_true", default=True)
    ap.add_argument("--no-compare", dest="compare", action="store_false")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--build-firmware", action="store_true",
                    help="idf.py build with TARGET_FOLD_SUBJECT_TAG for each subject.")
    ap.add_argument("--flash", action="store_true",
                    help="idf.py flash after building.")
    ap.add_argument("--idf-port", default=None,
                    help="Serial port for idf.py flash (e.g. COM5 / /dev/ttyUSB0).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply_protocol_side_effects(args.protocol)

    subjects = [s.strip().lower() for s in (args.subjects or list(all_subjects()))]
    backends = list(args.backends)

    if not args.dry_run and not args.host:
        print("ERROR: --host is required (or set REPLAY_HOST).", file=sys.stderr)
        return 1

    try:
        avail = discover_subject_paths(args.data_root)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("  Example: --data-root D:/datasets/12JFPM_SSVEP-master/data", file=sys.stderr)
        return 1
    missing = [s for s in subjects if s not in avail]
    if missing:
        print(f"ERROR: raw .mat missing for {missing} under {avail.get('s1', Path('?')).parent}",
              file=sys.stderr)
        return 1

    out_root: Path = args.outdir or (
        _REPO / "results" / "replay_multisubject" / args.protocol
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(" Multi-subject on-device replay orchestrator")
    print("=" * 64)
    print(f" Subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f" Backends ({len(backends)}): {', '.join(backends)}")
    print(f" Device host   : {args.host or '— (dry run)'}")
    print(f" Build firmware: {args.build_firmware}   Flash: {args.flash}"
          f"   Port: {args.idf_port}")
    print(f" Compare       : {args.compare}    Dry run: {args.dry_run}")
    print(f" Protocol      : {args.protocol}")
    print(f" Output root   : {out_root.relative_to(_REPO)}")
    print(f" Firmware dir  : {FIRMWARE_DIR.relative_to(_REPO)}")
    print(f" Export root   : {STRICT_EXPORT_ROOT.relative_to(_REPO)}"
          if args.protocol == "strict_loso_inner_val"
          else f"(protocol {args.protocol})")
    print(f" Replay root   : {replay_results_root(_REPO, args.protocol).relative_to(_REPO)}")
    print(f" Firmware tag  : foldNN_test_sN (strict) or foldNN_val_sN (Version A)")
    print("=" * 64)

    results: list[dict] = []
    t_all = time.time()

    for subject in subjects:
        fold = fold_for_subject(subject)
        mat = avail[subject]
        print(f"\n[{tag_for(subject, fold, protocol=args.protocol)}]  mat={mat.name}")

        if args.dry_run:
            results.append({"stage": "dry_run", "subject": subject})
        else:
            fw_info = _build_and_flash(subject, fold,
                                       build=args.build_firmware,
                                       flash=args.flash,
                                       port=args.idf_port,
                                       protocol=args.protocol)
            fw_info["stage"] = "firmware"
            results.append(fw_info)
            print(f"  firmware: {fw_info.get('status')}"
                  f" ({fw_info.get('elapsed_s', '-')}s)")
            if (args.build_firmware or args.flash) and fw_info.get("status") != "ok":
                print("  -> skipping replays for this subject")
                continue

        for backend in backends:
            print(f"  -> replay {backend}", flush=True)
            if args.dry_run:
                results.append({"stage": "replay", "subject": subject,
                                "backend": backend, "status": "dry_run",
                                "outdir": str(out_root / backend)})
                continue
            info = _run_replay(
                subject=subject, mat=mat, backend=backend,
                host=args.host, port=args.tcp_port,
                compare=args.compare, max_trials=args.max_trials,
                outdir=out_root / backend, verbose=args.verbose,
                protocol=args.protocol,
            )
            info["stage"] = "replay"
            results.append(info)
            print(f"     {info.get('status')} ({info.get('elapsed_s', '-')}s)")

    replay_rows = [r for r in results if r.get("stage") == "replay"]
    matrix_path = _aggregate(replay_rows, out_root) if replay_rows else None

    report = {
        "schema": "replay_multisubject_run_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short(),
        "subjects": subjects,
        "backends": backends,
        "host": args.host,
        "build_firmware": args.build_firmware,
        "flash": args.flash,
        "dry_run": args.dry_run,
        "elapsed_s": round(time.time() - t_all, 1),
        "results": results,
    }
    rp = out_root / "replay_multisubject_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\nReport : {rp.relative_to(_REPO)}")
    if matrix_path:
        print(f"Matrix : {matrix_path.relative_to(_REPO)}")
        print(f"Matrix : {matrix_path.with_suffix('.csv').relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
