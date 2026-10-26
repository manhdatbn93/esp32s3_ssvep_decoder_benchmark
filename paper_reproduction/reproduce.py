#!/usr/bin/env python3
"""Reproduce the paper's results tables and figures from the released evidence.

    python paper_reproduction/reproduce.py [--out DIR] [--no-figures]

Requires only the files in this directory (no raw EEG, no ESP32-S3 hardware).
Writes regenerated tables, trial records, figures, and ``verification_report.json``
to ``--out`` (default ``paper_reproduction/outputs``). Exit status is 0 only when
every check passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from repro import analysis, evidence, policies, resources  # noqa: E402
from repro.constants import (  # noqa: E402
    ALL_BACKENDS,
    EVIDENCE,
    EXPECTED,
    INT8_BACKENDS,
    MANIFEST,
    NEURAL_BACKENDS,
    QUARTILES,
    TABLE2_ORDER,
)

# (application image KiB, used arena KiB) hard-coded in the original policy analysis
# (tools/analysis/r21_policy_analysis.py, MEASURED_RESOURCE_PROFILE). Kept only to show
# that the policies used the same resource profile as the archived Table 5 evidence.
ORIGINAL_POLICY_RESOURCE_PROFILE = {
    "CCA": (960.9, None), "FBCCA": (965.3, None), "MSI": (961.6, None), "TRCA": (2743.1, None),
    "CCNN-f32": (1164.3, 40.6), "CCNN-int8": (1039.5, 15.4), "ConvCA-f32": (1023.0, 275.9),
    "EEGNet-f32": (1000.2, 163.7), "EEGNet-int8": (1000.2, 76.1), "FBtCNN-f32": (1711.6, 385.5),
    "FBtCNN-int8": (1179.4, 346.1),
}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, **details: Any) -> bool:
        details.pop("passed", None)
        self.checks.append({"check": name, "passed": bool(ok), **details})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        return ok

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def fmt2(v: float) -> str:
    return f"{v:.2f}"


def fmt_rel_l2(v: float) -> str:
    return f"{v:.3g}" if v >= 1e-3 else f"{v:.2e}"


def fmt_cos(v: float) -> str:
    return f"{v:.2e}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def verify_manifest(report: Report) -> None:
    lines = [x for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    listed, bad = set(), []
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = HERE / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    present = {p.relative_to(HERE).as_posix() for d in ("evidence", "expected")
               for p in (HERE / d).rglob("*") if p.is_file()}
    unlisted = sorted(present - listed)
    report.add("frozen evidence matches MANIFEST.sha256", not bad and not unlisted,
               n_listed=len(listed), mismatched_or_missing=bad[:20], unlisted=unlisted[:20])


def compare_table(report: Report, name: str, expected: dict, computed: dict[str, list[str]]) -> None:
    diffs = []
    for backend, values in expected["rows"].items():
        got = computed.get(backend)
        for col, e, g in zip(expected["columns"], values, got or []):
            if str(e) != str(g):
                diffs.append({"backend": backend, "column": col, "submitted": e, "regenerated": g})
        if got is None:
            diffs.append({"backend": backend, "missing": True})
    report.add(name, not diffs, n_cells=sum(len(v) for v in expected["rows"].values()), differences=diffs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out: Path = args.out
    report = Report()
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

    verify_manifest(report)

    # ------------------------------------------------------------ load evidence
    replay = evidence.load_replay()
    csv_mirror = evidence.load_replay_csv_mirror()
    stage1 = evidence.load_stage1()
    classical = evidence.load_classical_fidelity()
    summaries = evidence.load_replay_summaries()

    consistency = analysis.check_replay_consistency(replay, csv_mirror)
    report.add("replay evidence: 110 runs x 180 unique trials, stored argmax and CSV mirror consistent",
               consistency["passed"] and consistency["n_runs"] == 110 and consistency["n_trials"] == 19800,
               **consistency)
    summary_errors = []
    for (b, s), sm in summaries.items():
        trials = [t for t in replay[b] if t.subject == s]
        correct = sum(t.device_argmax == t.true_label for t in trials)
        agree = sum(t.device_argmax == t.host_argmax for t in trials)
        mean_us = sum(t.elapsed_us for t in trials) / len(trials)
        if (sm["trials_ok"], sm["correct"], sm["parity_argmax_agree"]) != (180, correct, agree) \
                or abs(sm["latency_avg_us"] - mean_us) > 0.051:
            summary_errors.append(f"{b}/{s}")
    report.add("run summaries agree with their trial records (count, correct, agreement, mean latency)",
               not summary_errors, mismatched_runs=summary_errors)
    cls_agree = analysis.classical_source_agreement(replay, classical)
    report.add("classical vectors: replay records equal classical fidelity record", cls_agree["passed"], **cls_agree)

    # ------------------------------------------------------------ Table 2
    t2 = analysis.table2(replay)
    t2_fmt = {b: [fmt2(r["device_acc_pct"]), fmt2(r["agreement_pct"]), fmt2(r["flip_pct"]),
                  fmt_rel_l2(r["mean_relative_l2"]), fmt_cos(r["cosine_distance"]), f"{r['latency_ms']:.1f}"]
              for b, r in t2.items()}
    compare_table(report, "Table 2 (11 realized backends) equals submitted values", expected["table2_realized_backends"], t2_fmt)

    # ------------------------------------------------------------ Stage join, Table 3, Table S12
    joined, join_info = analysis.join_stages(stage1, replay)
    report.add("Stage 1 and Stage 2 records align (12,600 neural trials; export host = replay host top-1)",
               join_info["passed"] and join_info["joined_rows"] == 12600, **join_info)
    derived_s2 = evidence.load_stage2_derived()
    by_key = {(t.backend, t.subject, t.trial_num): t for b in NEURAL_BACKENDS for t in replay[b]}
    s2_mismatch = 0
    for r in derived_s2:
        t = by_key.get((r["backend"], r["subject"], int(r["trial_num"])))
        if t is None or (int(r["host_prediction"]), int(r["device_prediction"])) != (t.host_argmax, t.device_argmax):
            s2_mismatch += 1
    report.add("frozen per_trial_fidelity.csv predictions equal the replay records",
               s2_mismatch == 0 and len(derived_s2) == 12600, rows=len(derived_s2), mismatches=s2_mismatch)
    t3 = analysis.table3(joined)
    t3_fmt = {b: [fmt2(r["stage1_agree_pct"]), fmt2(r["stage1_flip_pct"]), fmt2(r["stage2_agree_pct"]),
                  fmt2(r["stage2_flip_pct"]), r["larger_boundary"]] for b, r in t3.items()}
    compare_table(report, "Table 3 (two-stage neural fidelity) equals submitted values", expected["table3_two_stage"], t3_fmt)
    trans, trans_records = analysis.transitions(joined)
    compare_table(report, "Table S12 (transition decomposition) equals submitted counts",
                  expected["table_s12_transitions"], {b: [v[k] for k in analysis.TRANSITIONS] for b, v in trans.items()})

    # ------------------------------------------------------------ flips and quartiles
    flip_summary, flip_records = analysis.int8_flips(replay)
    ef = expected["int8_flips"]
    report.add("int8 Stage 2 flips: 107 total (67/34/6), 104 distinct trials, none on all three",
               flip_summary["total"] == ef["total"] and flip_summary["per_backend"] == ef["per_backend"]
               and flip_summary["distinct_trials"] == ef["distinct_trials"]
               and flip_summary["all_three_overlap"] == ef["all_three_overlap"], **flip_summary)
    vectors = analysis.trial_vectors(replay, classical)
    q_rows, q_info = analysis.margin_quartiles(vectors)
    q_by = {(r["backend"], r["host_margin_quartile"]): r for r in q_rows}
    q_ok = all(f"{q_by[(b, 'Q1')]['pct_of_backend_flips']:.1f}" == ef["q1_share_pct_1dp"][b]
               and q_by[(b, "Q2")]["n_flips"] == ef["q2_flips"][b]
               and q_by[(b, "Q3")]["n_flips"] + q_by[(b, "Q4")]["n_flips"] == ef["q3_q4_flips"]
               for b in INT8_BACKENDS)
    report.add("margin quartiles: Q1 shares 98.5/94.1/100.0 %, Q2 flips 1/2/0, no Q3/Q4 flips", q_ok,
               counts={b: [q_by[(b, q)]["n_trials"] for q in QUARTILES] for b in INT8_BACKENDS},
               flips={b: [q_by[(b, q)]["n_flips"] for q in QUARTILES] for b in INT8_BACKENDS}, **q_info)
    flip_q = {(r["backend"], r["subject"], r["trial_num"]): r for r in vectors}
    for f in flip_records:
        f["prediction_flip_recomputed"] = flip_q[(f["backend"], f["subject"], f["trial_num"])]["prediction_flip"]

    r21 = EVIDENCE / "r21_analysis_outputs"
    ref_fig = {(r["backend"], r["host_margin_quartile"]): r for r in evidence.read_csv(r21 / "figure3_margin_quartile_source_r21.csv")}
    fig_diffs = [
        {"group": f"{b}/{q}", "field": k, "regenerated": q_by[(b, q)][k], "archived": float(ref_fig[(b, q)][k])}
        for b in INT8_BACKENDS for q in QUARTILES
        for k in ("n_trials", "n_flips", "subject_mean_flip_rate", "bootstrap_ci_lo", "bootstrap_ci_hi")
        if not math.isclose(float(q_by[(b, q)][k]), float(ref_fig[(b, q)][k]), rel_tol=1e-12, abs_tol=1e-15)
    ]
    report.add("Figure 2 source (rates and 10,000-sample bootstrap CIs) equals archived r21 figure source",
               not fig_diffs, differences=fig_diffs)

    norm = analysis.normalized_summary(vectors)
    ref_norm = {r["backend"]: r for r in evidence.read_csv(r21 / "normalized_output_error_summary_r21.csv")}
    s13 = expected["table_s13_normalized_output_error"]
    s13_fmt = {}
    for r in norm:
        printed = s13["rows"][r["backend"]]
        s13_fmt[r["backend"]] = [
            printed[i] if f"{float(printed[i]):.6g}" == f"{r[k]:.6g}" else f"{r[k]:.6g}"
            for i, k in enumerate(("mean_relative_l2_error", "median_relative_l2_error", "p95_relative_l2_error"))
        ] + [str(r["near_zero_host_norm_count"])] + [
            fmt_cos(r[k]) for k in ("mean_cosine_distance", "median_cosine_distance", "p95_cosine_distance")]
    compare_table(report, "Table S13 (normalized output error) equals submitted values", s13, s13_fmt)
    norm_diffs = [
        {"backend": r["backend"], "field": k, "regenerated": r[k], "archived": float(ref_norm[r["backend"]][k])}
        for r in norm
        for k in ("mean_relative_l2_error", "median_relative_l2_error", "p95_relative_l2_error",
                  "mean_cosine_distance", "median_cosine_distance", "p95_cosine_distance")
        if not math.isclose(r[k], float(ref_norm[r["backend"]][k]), rel_tol=1e-9, abs_tol=0.0)
    ]
    report.add("normalized output error equals archived r21 summary (relative tolerance 1e-9)",
               not norm_diffs, differences=norm_diffs)

    # ------------------------------------------------------------ Table 5
    t5 = resources.table5()
    et5 = expected["table5_resources"]
    t5_fp = {b: [str(r["app_kib"]), str(r["code_kib"]), str(r["data_kib"]), str(r["diram_kib"])] for b, r in t5.items()}
    compare_table(report, "Table 5(a) firmware footprint (44 values) equals archived idf.py size evidence",
                  {"columns": et5["footprint_columns"], "rows": et5["footprint"]}, t5_fp)
    t5_rt = {b: [str(r["heap_drop_kib"]), str(r["min_free_stack_bytes"]),
                 "---" if r["arena_used_kib"] is None else str(r["arena_used_kib"])] for b, r in t5.items()}
    compare_table(report, "Table 5(b) runtime memory (29 values + 4 analytical dashes) equals archived UART/arena logs",
                  {"columns": et5["runtime_columns"], "rows": et5["runtime"]}, t5_rt)
    report.add("all seven neural arenas allocated in PSRAM",
               all(t5[b]["arena_region"] == et5["arena_region"] for b in NEURAL_BACKENDS))
    profile = resources.policy_resource_profile(t5)
    report.add("policy resource profile derived from Table 5 evidence equals the profile used by the original policy analysis",
               profile == ORIGINAL_POLICY_RESOURCE_PROFILE, derived=profile)

    # ------------------------------------------------------------ Table 4 policies
    p95 = {r["backend"]: r["p95_relative_l2_error"] for r in norm}
    backends = {
        b: policies.Backend(
            name=b, family="classical" if b in ("CCA", "FBCCA", "MSI", "TRCA") else "neural",
            precision="classical" if b in ("CCA", "FBCCA", "MSI", "TRCA") else b.rsplit("-", 1)[1],
            device_accuracy_pct=t2[b]["device_acc_pct"], agreement_pct=t2[b]["agreement_pct"],
            latency_ms=t2[b]["latency_ms"], app_kib=profile[b][0], arena_kib=profile[b][1],
            p95_relative_l2=p95[b])
        for b in ALL_BACKENDS
    }
    results = {v: policies.evaluate(backends, v) for v in policies.VARIANTS}
    for variant, res in results.items():
        diffs = []
        for r in res:
            e = expected["table4_policies"][r["policy_id"]]
            for k in ("admissible", "pareto", "selected"):
                if r[k] != e[k]:
                    diffs.append({"policy": r["policy_id"], "field": k, "submitted": e[k], "regenerated": r[k]})
            gates = {"device_accuracy_pct": "A", "agreement_pct": "G", "latency_ms": "L", "app_kib": "S",
                     "arena_kib": "M", "tflm_dependency": "TFLM"}
            got_gates = {gates[m]: t for m, _op, t in r["gates"]}
            if got_gates != {k: float(v) for k, v in e["gates"].items()}:
                diffs.append({"policy": r["policy_id"], "field": "gates", "submitted": e["gates"], "regenerated": got_gates})
        report.add(f"Table 4 policies ({variant} objectives): gates, admissible sets, Pareto sets, selections", not diffs,
                   differences=diffs)
    same = all(a[k] == b[k] for a, b in zip(results["as_described"], results["as_executed"])
               for k in ("admissible", "pareto", "selected"))
    report.add("policy outcomes identical with and without P95 relative L2 in the objectives", same)
    ref_sel = evidence.read_csv(r21 / "deployment_policy_selection_r21.csv")
    sel_ok = all(
        next(r for r in results["as_executed"] if r["policy_id"] == row["policy_id"])["selected"][row["priority_id"]]
        == row["selected_backend"] for row in ref_sel)
    report.add("selections equal archived r21 deployment_policy_selection", sel_ok and len(ref_sel) == 6)
    grid = policies.sensitivity_grid(backends)
    ref_grid = evidence.read_csv(r21 / "deployment_policy_sensitivity_r21.csv")
    grid_diffs = [(g["min_agreement_pct"], g["max_latency_ms"], g["selected_backend"], r["selected_backend"])
                  for g, r in zip(grid, ref_grid)
                  if (g["selected_backend"], g["admissible_backends"]) != (r["selected_backend"], r["admissible_backends"])
                  or float(r["min_agreement_pct"]) != g["min_agreement_pct"] or float(r["max_latency_ms"]) != g["max_latency_ms"]]
    report.add("5x5 policy sensitivity grid equals archived r21 grid", not grid_diffs and len(ref_grid) == 25,
               differences=grid_diffs)

    # ------------------------------------------------------------ outputs
    tables = out / "tables"
    write_csv(tables / "table2_realized_backends.csv",
              [{"backend": b, **t2[b], **dict(zip([f"{c}_printed" for c in expected["table2_realized_backends"]["columns"]], t2_fmt[b]))}
               for b in TABLE2_ORDER])
    write_csv(tables / "table3_two_stage_fidelity.csv", [{"backend": b, **t3[b]} for b in NEURAL_BACKENDS])
    write_csv(tables / "table_s12_transitions.csv", [{"backend": b, **trans[b]} for b in NEURAL_BACKENDS])
    write_csv(tables / "table4_policies.csv", [
        {"variant": v, "policy_id": r["policy_id"], "name": r["name"],
         "admissible": " | ".join(r["admissible"]), "pareto": " | ".join(r["pareto"]),
         "selected": " | ".join(f"{k}={s}" for k, s in r["selected"].items()),
         "pareto_objectives": " | ".join(r["pareto_objectives"]),
         "priorities": " || ".join(f"{k}={' | '.join(o)}" for k, o in r["priorities"].items())}
        for v, res in results.items() for r in res])
    write_csv(tables / "table4_policy_exclusions.csv",
              [{"policy_id": r["policy_id"], **x} for r in results["as_described"] for x in r["exclusions"]])
    write_csv(tables / "table_s16_policy_sensitivity.csv", grid)
    write_csv(tables / "table5_resource_profile.csv",
              [{"backend": b, **{k: (str(v) if v is not None else "") for k, v in t5[b].items()}} for b in ALL_BACKENDS])
    write_csv(tables / "normalized_output_error_summary.csv", norm)
    write_csv(tables / "figure2_margin_quartile_source.csv", q_rows)
    write_csv(tables / "figure3_latency_source.csv",
              [{"backend": b, "latency_ms": t2[b]["latency_ms"], "device_acc_pct": t2[b]["device_acc_pct"],
                "agreement_pct": t2[b]["agreement_pct"]} for b in NEURAL_BACKENDS])
    records = out / "records"
    write_csv(records / "neural_stage_transitions_per_trial.csv", trans_records)
    write_csv(records / "int8_stage2_flips.csv", flip_records)
    write_csv(records / "flip_kind_counts.csv",
              [{"backend": b, **Counter(f["kind"] for f in flip_records if f["backend"] == b)} for b in INT8_BACKENDS])

    if not args.no_figures:
        from repro import figures

        paths = figures.figure2_margin_quartile(q_rows, out / "figures" / "fig_margin_quartile")
        paths += figures.figure3_latency(t2, out / "figures")
        report.add("figures regenerated", all(p.is_file() and p.stat().st_size > 0 for p in paths),
                   files=[p.relative_to(out).as_posix() for p in paths])

    import matplotlib
    import numpy

    payload = {
        "status": "PASS" if report.passed else "FAIL",
        "n_checks": len(report.checks),
        "n_failed": sum(not c["passed"] for c in report.checks),
        "environment": {"python": platform.python_version(), "platform": platform.system(),
                        "numpy": numpy.__version__, "matplotlib": matplotlib.__version__},
        "checks": report.checks,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification_report.json").write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\n{payload['status']}: {payload['n_checks'] - payload['n_failed']}/{payload['n_checks']} checks passed")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
