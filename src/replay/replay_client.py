#!/usr/bin/env python3
"""
PC client for 12JFPM dataset replay over TCP/Wi-Fi.

Connects to the ESP32-S3 replay firmware, streams raw trials from a
.mat file, and collects device scores/logits. By default also runs the
host-side Python reference backend to compute device-vs-host parity
(argmax agreement rate + max/mean absolute score difference).

Saves structured per-trial and per-run artifacts for analysis-ready analysis.

Usage (from repo root):
  # Replay with parity comparison (default --compare is ON):
  python tools/replay/replay_client.py --backend cca --host BOARD_IP_PLACEHOLDER --mat path/to/s1.mat

  Example of mat path:
  --mat "<PRIVATE_REPO_ROOT>\existing_work\data\raw\12JFPM\extracted\12JFPM_SSVEP-master\data\s1.mat"

  # Quick smoke (5 trials, no disk artifacts):
  python tools/replay/replay_client.py --backend cca --host BOARD_IP_PLACEHOLDER --max-trials 5 --no-save

  # Skip host comparison (accuracy only, no parity):
  python tools/replay/replay_client.py --backend cca --host ... --mat ... --no-compare

  # Aggregate only (no board):
  python tools/replay/aggregate_replay_results.py

Environment variables:
  REPLAY_WIFI_HOST  — board IP  (default BOARD_IP_PLACEHOLDER)
  REPLAY_WIFI_PORT  — TCP port  (default 3333)
  REPLAY_MAT        — path to s1.mat
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_TOOLS_REPLAY = Path(__file__).resolve().parent
for _p in (_REPO, _TOOLS_REPLAY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from replay_proto import (  # noqa: E402
    DECODER_NAMES,
    DecoderID,
    MODEL_IDENTITY_PACK,
    NormMode,
    PktType,
    REPLAY_MAGIC_U32,
    TrialMeta,
    decoder_id_from_name,
    pack_header,
    pack_start_run,
    pack_trial_meta,
    recv_exact,
    unpack_error_payload,
    unpack_header,
    unpack_model_identity,
    unpack_result_payload,
)
from model_identity import (  # noqa: E402
    ModelIdentityMismatchError,
    assert_eegnet_int8_identity,
)

from src.datasets.parse_12jfpm import read_subject  # noqa: E402

from aggregate_replay_results import run_aggregate  # noqa: E402
from host_reference import HostReference, _preprocess_zscore  # noqa: E402

_DEFAULT_HOST = os.environ.get("REPLAY_WIFI_HOST", "BOARD_IP_PLACEHOLDER")
_DEFAULT_PORT = int(os.environ.get("REPLAY_WIFI_PORT", "3333"))

_N_CLASSES = 12
_N_TRIALS_PER_CLASS = 15


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _default_mat_path() -> Path | None:
    candidates = [
        _REPO / "existing_work" / "data" / "raw" / "12JFPM" / "extracted"
        / "12JFPM_SSVEP-master" / "data" / "s1.mat",
        _REPO / "data" / "raw" / "12JFPM" / "extracted"
        / "12JFPM_SSVEP-master" / "data" / "s1.mat",
        _REPO / "data" / "raw" / "12JFPM" / "s1.mat",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def extract_raw_trial(eeg: np.ndarray, class_id: int, trial_idx: int) -> np.ndarray:
    """Raw segment [n_samples, 8] float32 — 12JFPM axes."""
    seg = np.asarray(eeg[class_id, :, :, trial_idx], dtype=np.float64).T
    return seg.astype(np.float32)


def _parity_summary(records: list[dict]) -> dict:
    """Compute parity metrics across trials (empty dict if no host comparison)."""
    compared = [r for r in records if "argmax_agree" in r]
    if not compared:
        return {}
    n_agree = sum(r["argmax_agree"] for r in compared)
    n_compared = len(compared)
    max_diffs = [r["max_abs_diff"] for r in compared]
    mean_diffs = [r["mean_abs_diff"] for r in compared]
    return {
        "parity_trials_compared": n_compared,
        "parity_argmax_agree": n_agree,
        "parity_argmax_agree_pct": round(100 * n_agree / n_compared, 2),
        "parity_max_abs_diff": round(max(max_diffs), 8),
        "parity_mean_abs_diff": round(float(np.mean(mean_diffs)), 8),
        "parity_worst_mean_diff": round(max(mean_diffs), 8),
    }


def _build_trial_plan(
    n_classes: int, n_trials_per_class: int, max_trials: int | None
) -> list[tuple[int, int]]:
    """Return list of (class_id, trial_idx) in dataset order."""
    plan = []
    for c in range(n_classes):
        for t in range(n_trials_per_class):
            plan.append((c, t))
    if max_trials is not None and max_trials < len(plan):
        plan = plan[:max_trials]
    return plan


def run_session(
    host: str,
    port: int,
    mat_path: Path,
    decoder_id: int,
    norm_mode: int = int(NormMode.CHANNEL_ZSCORE),
    max_trials: int | None = None,
    verbose: bool = False,
    host_ref: HostReference | None = None,
    backend_name: str | None = None,
) -> list[dict]:
    """Run one replay session. Returns list of per-trial result dicts."""
    sid = mat_path.stem.lower()
    if not sid.startswith("s"):
        sid = "s" + sid.lstrip("s")
    rec = read_subject(sid, data_root=mat_path.parent)
    eeg = np.asarray(rec["eeg"])

    n_classes = eeg.shape[0]
    n_trials = eeg.shape[3]
    plan = _build_trial_plan(n_classes, n_trials, max_trials)
    total_planned = len(plan)

    dec_name = DECODER_NAMES.get(decoder_id, f"decoder_{decoder_id}")
    records: list[dict] = []

    sock = socket.create_connection((host, port), timeout=30.0)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # HELLO
        sock.sendall(pack_header(int(PktType.HELLO), 0))
        rh = recv_exact(sock, 16)
        magic, _, _, ptype, _, plen = unpack_header(rh)
        if magic != REPLAY_MAGIC_U32 or ptype != int(PktType.HELLO_ACK):
            print(f"HELLO failed: magic=0x{magic:08x} type={ptype}", file=sys.stderr)
            return records
        if plen:
            recv_exact(sock, plen)

        # START_RUN
        body = pack_start_run(
            norm=norm_mode, onset=38, window=512, fs_hz=256,
            n_channels=8, decoder_id=decoder_id,
        )
        sock.sendall(pack_header(int(PktType.START_RUN), len(body)) + body)
        rh = recv_exact(sock, 16)
        magic, _, _, ptype, _, plen = unpack_header(rh)
        if ptype == int(PktType.ERROR):
            eb = recv_exact(sock, plen) if plen else b""
            ec, em = unpack_error_payload(eb)
            print(f"START_RUN error code={ec} msg={em!r}", file=sys.stderr)
            return records
        if ptype != int(PktType.START_ACK):
            print(f"Expected START_ACK, got type={ptype}", file=sys.stderr)
            return records

        fw_model_identity = None
        if plen:
            ack_payload = recv_exact(sock, plen)
            if plen >= MODEL_IDENTITY_PACK.size:
                try:
                    fw_model_identity = unpack_model_identity(ack_payload)
                except ValueError:
                    if verbose:
                        print(f"  START_ACK payload {plen} B (not model identity)",
                              flush=True)

        bkey = (backend_name or dec_name).strip().lower().replace("-", "_")
        if host_ref is not None and bkey in ("eegnet_int8", "eegnet_i8"):
            try:
                assert_eegnet_int8_identity(
                    host_ref, fw_model_identity, requested_backend=bkey,
                )
            except ModelIdentityMismatchError as e:
                print(str(e), file=sys.stderr)
                return records

        if verbose:
            print(f"Session: {dec_name} (id={decoder_id}) on {host}:{port}, "
                  f"{total_planned} trials planned")

        # TRIAL loop
        for idx, (c, tr) in enumerate(plan):
            raw = extract_raw_trial(eeg, c, tr)
            meta_bytes = pack_trial_meta(TrialMeta(
                subject=sid, class_id=c, trial_idx=tr,
                n_time=int(raw.shape[0]), n_channels=int(raw.shape[1]),
                fs_hz=256, normalize_mode=norm_mode,
            ))
            payload = meta_bytes + raw.tobytes(order="C")
            sock.sendall(
                pack_header(int(PktType.TRIAL_DATA), len(payload)) + payload
            )

            rh = recv_exact(sock, 16)
            magic, _, _, ptype, _, plen = unpack_header(rh)
            if ptype == int(PktType.ERROR):
                eb = recv_exact(sock, plen) if plen else b""
                ec, em = unpack_error_payload(eb)
                print(f"Trial c={c} tr={tr} ERROR code={ec} msg={em!r}",
                      file=sys.stderr)
                return records
            if ptype != int(PktType.RESULT):
                print(f"Expected RESULT, got type={ptype}", file=sys.stderr)
                return records

            res_pl = recv_exact(sock, plen)
            meta, res_dec, status, elapsed_us, logits = unpack_result_payload(res_pl)

            dev_logits = np.asarray(logits, dtype=np.float32)
            dev_argmax = int(np.argmax(dev_logits))
            correct = int(dev_argmax == c)
            rec_entry = {
                "trial_num": idx + 1,
                "subject": sid,
                "class_id": c,
                "trial_idx": tr,
                "true_label": c,
                "device_argmax": dev_argmax,
                "correct": correct,
                "device_logits": logits,
                "decoder_id": res_dec,
                "decoder_name": dec_name,
                "status": status,
                "elapsed_us": elapsed_us,
            }

            parity_str = ""
            if host_ref is not None and status == 0:
                try:
                    crop = raw[38:550, :].astype(np.float32)
                    crop_z = _preprocess_zscore(crop)
                    host_scores = host_ref.run(
                        crop_z, class_id=c, trial_idx=tr
                    )
                    host_argmax = int(np.argmax(host_scores))
                    diff = np.abs(dev_logits - host_scores)
                    max_diff = float(np.max(diff))
                    mean_diff = float(np.mean(diff))
                    agree = int(dev_argmax == host_argmax)
                    rec_entry["host_argmax"] = host_argmax
                    rec_entry["host_logits"] = host_scores.tolist()
                    rec_entry["argmax_agree"] = agree
                    rec_entry["max_abs_diff"] = round(max_diff, 8)
                    rec_entry["mean_abs_diff"] = round(mean_diff, 8)
                    tag = "AGREE" if agree else "DISAGREE"
                    parity_str = f" {tag} max_diff={max_diff:.8f}"
                except Exception as e:
                    if idx == 0:
                        print(f"  [warn] host reference failed: {e}", flush=True)
                        print(f"  [warn] disabling comparison for remaining trials",
                              flush=True)
                        host_ref = None

            records.append(rec_entry)

            match_str = "OK" if correct else "MISS"
            print(
                f"  [{idx+1:3d}/{total_planned}] c={c:2d} tr={tr:2d} "
                f"argmax={dev_argmax:2d} label={c:2d} {match_str} "
                f"{elapsed_us/1000:.1f}ms{parity_str}",
                flush=True,
            )

        # END_RUN
        sock.sendall(pack_header(int(PktType.END_RUN), 0))
        try:
            rh = recv_exact(sock, 16)
        except ConnectionError:
            pass
    finally:
        sock.close()

    return records


def compute_summary(
    records: list[dict],
    *,
    decoder_id: int,
    dec_name: str,
    subject: str,
    mat_path: str,
    host: str,
    port: int,
    norm_mode: int,
    max_trials: int | None,
    cli_args: dict,
) -> dict:
    """Compute run-level summary from trial records."""
    n_total = len(records)
    n_ok = sum(1 for r in records if r["status"] == 0)
    n_err = sum(1 for r in records if r["status"] != 0)
    n_correct = sum(1 for r in records if r["correct"])
    latencies = [r["elapsed_us"] for r in records if r["status"] == 0]

    return {
        "backend_name": dec_name,
        "backend_id": decoder_id,
        "subject": subject,
        "mat_file": mat_path,
        "dataset": "12JFPM",
        "host": host,
        "port": port,
        "norm_mode": norm_mode,
        "max_trials_requested": max_trials,
        "trials_total": n_total,
        "trials_ok": n_ok,
        "trials_err": n_err,
        "correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "accuracy_pct": round(100 * n_correct / n_total, 2) if n_total else 0.0,
        "latency_avg_us": round(np.mean(latencies), 1) if latencies else 0.0,
        "latency_min_us": int(min(latencies)) if latencies else 0,
        "latency_max_us": int(max(latencies)) if latencies else 0,
        "latency_std_us": round(float(np.std(latencies)), 1) if latencies else 0.0,
        "latency_avg_ms": round(np.mean(latencies) / 1000, 2) if latencies else 0.0,
        "latency_min_ms": round(min(latencies) / 1000, 2) if latencies else 0.0,
        "latency_max_ms": round(max(latencies) / 1000, 2) if latencies else 0.0,
        "preprocessing": "on-device crop+zscore",
        **_parity_summary(records),
        "git_hash": _git_hash(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "cli_args": cli_args,
    }


def save_artifacts(
    outdir: Path,
    records: list[dict],
    summary: dict,
) -> dict[str, Path]:
    """Save per-trial and run-level artifacts. Returns dict of saved paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # run_metadata.json (same as summary.json but separated for clarity)
    p = outdir / "summary.json"
    p.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    paths["summary"] = p

    # trial_results.csv
    has_parity = any("argmax_agree" in r for r in records)
    csv_fields = [
        "trial_num", "subject", "class_id", "trial_idx", "true_label",
        "device_argmax", "correct", "status", "elapsed_us",
        "decoder_id", "decoder_name",
    ]
    parity_fields = (
        ["host_argmax", "argmax_agree", "max_abs_diff", "mean_abs_diff"]
        if has_parity else []
    )
    logit_fields = [f"dev_logit_{i}" for i in range(12)]
    host_logit_fields = [f"host_logit_{i}" for i in range(12)] if has_parity else []
    all_fields = csv_fields + parity_fields + logit_fields + host_logit_fields
    p = outdir / "trial_results.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in csv_fields + parity_fields}
            for i, v in enumerate(r.get("device_logits", [])):
                row[f"dev_logit_{i}"] = f"{v:.8f}"
            for i, v in enumerate(r.get("host_logits", [])):
                row[f"host_logit_{i}"] = f"{v:.8f}"
            w.writerow(row)
    paths["csv"] = p

    # trial_results.jsonl
    p = outdir / "trial_results.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            line = dict(r)
            if "device_logits" in line:
                line["device_logits"] = [round(v, 8) for v in line["device_logits"]]
            if "host_logits" in line:
                line["host_logits"] = [round(v, 8) for v in line["host_logits"]]
            f.write(json.dumps(line, default=str) + "\n")
    paths["jsonl"] = p

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="12JFPM dataset replay client for ESP32-S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--host", default=_DEFAULT_HOST,
                    help="ESP32-S3 IP address")
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT,
                    help="TCP port")
    ap.add_argument("--mat", type=Path, default=None,
                    help="Path to s1.mat (or set REPLAY_MAT env)")
    ap.add_argument("--backend", required=True,
                    help="Backend name (e.g. cca, eegnet_f32, ccnn_int8)")
    ap.add_argument("--norm", type=int, default=int(NormMode.CHANNEL_ZSCORE),
                    help="0=none, 1=channel_zscore, 2=minmax")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="Max trials to replay (omit for all 180)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory for artifacts "
                         "(default: results/replay/<backend>/<subject>_<timestamp>)")
    ap.add_argument("--run-tag", default=None,
                    help="Custom run tag for output folder name")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--compare", action="store_true", default=True,
                    help="Run host-side reference to compute parity metrics (default: on)")
    ap.add_argument("--no-compare", dest="compare", action="store_false",
                    help="Disable host-side parity comparison")
    ap.add_argument("--no-save", action="store_true",
                    help="Skip saving artifacts (console-only mode)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip refreshing results/replay/aggregate.csv after the run")
    from export.protocol_cli import add_protocol_arg, apply_protocol_side_effects  # noqa: E402
    add_protocol_arg(ap)
    args = ap.parse_args()
    apply_protocol_side_effects(args.protocol)

    # Resolve mat file
    mat = args.mat
    if mat is None:
        env_mat = os.environ.get("REPLAY_MAT")
        if env_mat:
            mat = Path(env_mat)
        else:
            mat = _default_mat_path()
    if mat is None:
        print("No .mat file found. Pass --mat PATH or set REPLAY_MAT.",
              file=sys.stderr)
        return 1
    mat = mat.resolve()
    if not mat.is_file():
        print(f"Mat file not found: {mat}", file=sys.stderr)
        return 1

    decoder_id = decoder_id_from_name(args.backend)
    dec_name = DECODER_NAMES.get(decoder_id, args.backend)
    sid = mat.stem.lower()
    if not sid.startswith("s"):
        sid = "s" + sid.lstrip("s")
    trial_desc = "all 180" if args.max_trials is None else str(args.max_trials)

    print(f"{'='*50}")
    print(f"  Replay: {dec_name} (id={decoder_id})")
    print(f"  Host  : {args.host}:{args.port}")
    print(f"  Mat   : {mat.name}  Subject: {sid}")
    print(f"  Trials: {trial_desc}")
    print(f"  Compare: {'ON' if args.compare else 'OFF'}")
    print(f"{'='*50}")
    print()

    host_ref = None
    if args.compare:
        try:
            host_ref = HostReference(args.backend, subject=sid, protocol=args.protocol)
            mode = "indexed (precomputed)" if host_ref.uses_indexed else "live"
            print(f"  Host reference loaded for {args.backend} [{mode}]")
            if host_ref.artifact_path:
                print(f"  Artifact: {host_ref.artifact_path}")
        except Exception as e:
            print(f"  Host reference unavailable: {e}")
            print(f"  (continuing without parity comparison)")
        print()

    t_start = time.time()
    try:
        records = run_session(
            host=args.host,
            port=args.port,
            mat_path=mat,
            decoder_id=decoder_id,
            norm_mode=args.norm,
            max_trials=args.max_trials,
            verbose=args.verbose,
            host_ref=host_ref,
            backend_name=args.backend,
        )
    except ModelIdentityMismatchError as e:
        print(str(e), file=sys.stderr)
        return 2
    t_elapsed = time.time() - t_start

    if host_ref is not None:
        host_ref.cleanup()

    if not records:
        print("No results collected.", file=sys.stderr)
        return 1

    # CLI args dict for metadata
    cli_args_dict = {
        "host": args.host, "port": args.port, "mat": str(mat),
        "backend": args.backend, "norm": args.norm,
        "max_trials": args.max_trials, "verbose": args.verbose,
        "compare": args.compare,
    }

    summary = compute_summary(
        records,
        decoder_id=decoder_id,
        dec_name=dec_name,
        subject=sid,
        mat_path=str(mat),
        host=args.host,
        port=args.port,
        norm_mode=args.norm,
        max_trials=args.max_trials,
        cli_args=cli_args_dict,
    )
    summary["wall_time_s"] = round(t_elapsed, 2)

    # Print summary
    n_total = summary["trials_total"]
    n_correct = summary["correct"]
    print()
    print(f"{'='*50}")
    print(f"  Summary: {dec_name} / {sid}")
    print(f"  Trials : {n_total} total, "
          f"{summary['trials_ok']} ok, {summary['trials_err']} err")
    print(f"  Accuracy: {n_correct}/{n_total} "
          f"({summary['accuracy_pct']:.1f}%) [device vs true label]")
    if "parity_argmax_agree" in summary:
        p_n = summary["parity_trials_compared"]
        p_ok = summary["parity_argmax_agree"]
        print(f"  Parity : {p_ok}/{p_n} argmax agree "
              f"({summary['parity_argmax_agree_pct']:.1f}%) "
              f"[device vs host]")
        print(f"           max_abs_diff = {summary['parity_max_abs_diff']:.8f}")
        print(f"           mean_abs_diff = {summary['parity_mean_abs_diff']:.8f}")
    if summary["trials_ok"] > 0:
        print(f"  Latency: avg {summary['latency_avg_ms']:.1f} ms, "
              f"min {summary['latency_min_ms']:.1f} ms, "
              f"max {summary['latency_max_ms']:.1f} ms")
    print(f"  Wall   : {t_elapsed:.1f}s")
    print(f"{'='*50}")

    # Save artifacts
    if not args.no_save:
        if args.outdir:
            outdir = args.outdir
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = args.run_tag or f"{sid}_{ts}"
            from export.protocol_cli import replay_results_root  # noqa: E402
            outdir = replay_results_root(_REPO, args.protocol) / dec_name / tag
        saved = save_artifacts(outdir, records, summary)
        print()
        print(f"Artifacts saved to: {outdir}")
        for name, path in saved.items():
            print(f"  {name}: {path.name}")

    if not args.no_aggregate:
        from export.protocol_cli import replay_results_root  # noqa: E402
        run_aggregate(replay_results_root(_REPO, args.protocol), quiet=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
