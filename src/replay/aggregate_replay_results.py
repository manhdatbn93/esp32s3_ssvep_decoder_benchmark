#!/usr/bin/env python3
"""
Aggregate replay results across backends and subjects into paper-ready tables.

Reads all summary.json / trial_results.csv files under a results tree and
produces consolidated output for paper figures and tables.

Usage (from repo root):
  # Usually you do not need this alone. replay_client.py runs
  # aggregate automatically after each successful replay unless --no-aggregate.

  # Aggregate only (refresh table from disk):
  python src/replay/aggregate_replay_results.py

  # Custom source / output:
  python src/replay/aggregate_replay_results.py --srcdir results/replay --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _path_for_message(p: Path) -> str:
    """Prefer repo-relative path for readable, cross-platform messages."""
    try:
        return p.resolve().relative_to(_REPO.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def discover_summaries(srcdir: Path) -> list[Path]:
    """Find all summary.json files recursively."""
    return sorted(srcdir.rglob("summary.json"))


def load_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
        return None


def aggregate(summaries: list[dict]) -> list[dict]:
    """Flatten summaries into rows for a consolidated table."""
    rows = []
    for s in summaries:
        rows.append({
            "backend": s.get("backend_name", ""),
            "backend_id": s.get("backend_id", ""),
            "subject": s.get("subject", ""),
            "trials": s.get("trials_total", 0),
            "correct": s.get("correct", 0),
            "accuracy_pct": s.get("accuracy_pct", 0.0),
            "parity_agree": s.get("parity_argmax_agree", ""),
            "parity_agree_pct": s.get("parity_argmax_agree_pct", ""),
            "parity_max_diff": s.get("parity_max_abs_diff", ""),
            "parity_mean_diff": s.get("parity_mean_abs_diff", ""),
            "latency_avg_ms": s.get("latency_avg_ms", 0.0),
            "latency_min_ms": s.get("latency_min_ms", 0.0),
            "latency_max_ms": s.get("latency_max_ms", 0.0),
            "latency_std_us": s.get("latency_std_us", 0.0),
            "trials_ok": s.get("trials_ok", 0),
            "trials_err": s.get("trials_err", 0),
            "wall_time_s": s.get("wall_time_s", 0.0),
            "timestamp": s.get("timestamp", ""),
            "git_hash": s.get("git_hash", ""),
        })
    rows.sort(key=lambda r: (r["backend"], r["subject"]))
    return rows


def print_table(rows: list[dict]) -> None:
    """Print a concise aligned table to stdout."""
    if not rows:
        print("No results to display.")
        return

    has_parity = any(r.get("parity_agree_pct") not in (None, "") for r in rows)

    if has_parity:
        fmt = ("{backend:<14s} {subject:<5s} {trials:>6s} {correct:>7s} {acc:>8s}"
               " {par_pct:>10s} {par_diff:>12s} {avg:>10s} {min:>10s} {max:>10s}")
        hdr = fmt.format(
            backend="Backend", subject="Subj", trials="Trials",
            correct="Correct", acc="Acc%",
            par_pct="Parity%", par_diff="MaxDiff",
            avg="Avg(ms)", min="Min(ms)", max="Max(ms)",
        )
    else:
        fmt = "{backend:<14s} {subject:<5s} {trials:>6s} {correct:>7s} {acc:>8s} {avg:>10s} {min:>10s} {max:>10s}"
        hdr = fmt.format(
            backend="Backend", subject="Subj", trials="Trials",
            correct="Correct", acc="Acc%", avg="Avg(ms)", min="Min(ms)", max="Max(ms)",
        )

    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        kw = dict(
            backend=r["backend"],
            subject=r["subject"],
            trials=str(r["trials"]),
            correct=f'{r["correct"]}/{r["trials"]}',
            acc=f'{r["accuracy_pct"]:.1f}',
            avg=f'{r["latency_avg_ms"]:.1f}',
            min=f'{r["latency_min_ms"]:.1f}',
            max=f'{r["latency_max_ms"]:.1f}',
        )
        if has_parity:
            pp = r.get("parity_agree_pct")
            pd = r.get("parity_max_diff")
            kw["par_pct"] = f"{pp:.1f}" if pp not in (None, "") else "n/a"
            kw["par_diff"] = f"{pd:.8f}" if pd not in (None, "") else "n/a"
        print(fmt.format(**kw))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")


def run_aggregate(
    srcdir: Path | None = None,
    *,
    output: Path | None = None,
    json_path: Path | None = None,
    quiet: bool = False,
    banner: bool = True,
) -> int:
    """
    Scan srcdir for summary.json files, print table, write aggregate CSV/JSON.

    Callable from replay_client after a run. Returns 0 always (empty tree is OK).
    """
    root = (srcdir or (_REPO / "results" / "replay")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    paths = discover_summaries(root)
    if not paths:
        rel = _path_for_message(root)
        print(
            f"No summary.json files yet (search root: {rel}/).\n"
            f"  Run a replay first, e.g.:\n"
            f"    python tools/replay/replay_client.py --backend cca --host <IP> --mat <path/to/s1.mat>\n"
            f"  Per-run artifacts: {rel}/<backend>/<subject>_<timestamp>/summary.json",
        )
        return 0

    if banner and not quiet:
        print()
        print("--- Aggregate (all runs under results/replay/) ---")

    print(f"Found {len(paths)} summary file(s) under {_path_for_message(root)}/")
    summaries = []
    for p in paths:
        s = load_summary(p)
        if s:
            summaries.append(s)

    rows = aggregate(summaries)

    if not quiet:
        print()
        print_table(rows)
        print()

    out_csv = output or (root / "aggregate.csv")
    save_csv(rows, out_csv)
    print(f"Saved: {out_csv}")

    if json_path:
        save_json(rows, json_path)
        print(f"Saved: {json_path}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate replay results for paper tables/figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from export.protocol_cli import add_protocol_arg, replay_results_root  # noqa: E402
    add_protocol_arg(ap)
    ap.add_argument("--srcdir", type=Path, default=None,
                    help="Root directory containing replay result folders")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output CSV path (default: <srcdir>/aggregate.csv)")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also output as JSON")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress table printout")
    args = ap.parse_args()

    src = args.srcdir or replay_results_root(_REPO, args.protocol)
    out = args.output or (src / "aggregate.csv")
    return run_aggregate(
        src,
        output=out,
        json_path=args.json,
        quiet=args.quiet,
        banner=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
