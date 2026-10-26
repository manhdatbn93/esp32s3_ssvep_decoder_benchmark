"""Tables 2-3, Table S12 transitions, int8 flips, normalized output error, and margin quartiles."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from .constants import (
    ALL_BACKENDS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEEDS,
    CLASSICAL_BACKENDS,
    CLASSICAL_SERIALIZATION_TOLERANCE,
    EPSILON,
    INT8_BACKENDS,
    NEAR_ZERO_HOST_NORM_THRESHOLD,
    NEURAL_BACKENDS,
    QUARTILES,
    SUBJECTS,
    TRIALS_PER_SUBJECT,
)
from .evidence import ReplayTrial
from .metrics import (
    average_ranks,
    cosine_similarity,
    percentile,
    prediction_and_margin,
    quartile_from_rank,
    raw_vector_errors,
    relative_l2_error,
    subject_bootstrap_ci,
    vector_norm,
)


def subject_first_mean(rows: list[dict], key: str) -> float:
    """Mean of the ten held-out-subject means."""
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(float(row[key]))
    if set(by_subject) != set(SUBJECTS):
        raise ValueError("subject-first mean needs all ten held-out subjects")
    return statistics.mean(statistics.mean(v) for v in by_subject.values())


# ---------------------------------------------------------------- replay checks

def check_replay_consistency(replay: dict[str, list[ReplayTrial]], csv_mirror: dict) -> dict[str, Any]:
    """Recompute argmax from the stored vectors and compare JSONL with the CSV mirror."""
    argmax_errors = 0
    csv_mismatch = 0
    max_csv_abs_diff = 0.0
    for backend, trials in replay.items():
        for t in trials:
            if prediction_and_margin(t.host)[0] != t.host_argmax:
                argmax_errors += 1
            if prediction_and_margin(t.device)[0] != t.device_argmax:
                argmax_errors += 1
            dev_arg, host_arg, elapsed, dev, host = csv_mirror[(backend, t.subject, t.trial_num)]
            if (dev_arg, host_arg, elapsed) != (t.device_argmax, t.host_argmax, t.elapsed_us):
                csv_mismatch += 1
            diff = max(abs(a - b) for a, b in zip(dev + host, t.device + t.host))
            max_csv_abs_diff = max(max_csv_abs_diff, diff)
    return {
        "n_trials": sum(len(v) for v in replay.values()),
        "n_runs": len(replay) * len(SUBJECTS),
        "stored_vs_recomputed_argmax_errors": argmax_errors,
        "jsonl_vs_csv_field_mismatches": csv_mismatch,
        "jsonl_vs_csv_max_abs_vector_diff": max_csv_abs_diff,
        "passed": argmax_errors == 0 and csv_mismatch == 0 and max_csv_abs_diff < 5e-8,
    }


# ---------------------------------------------------------------- Table 2

def table2(replay: dict[str, list[ReplayTrial]]) -> dict[str, dict[str, float]]:
    out = {}
    for backend, trials in replay.items():
        rows = []
        for t in trials:
            cos = cosine_similarity(t.host, t.device)
            rows.append({
                "subject": t.subject,
                "acc": float(t.device_argmax == t.true_label),
                "agree": float(t.device_argmax == t.host_argmax),
                "rel_l2": relative_l2_error(t.host, t.device),
                "cos_dist": 1.0 - cos,
                "latency_ms": t.elapsed_us / 1000.0,
            })
        agree = 100.0 * subject_first_mean(rows, "agree")
        out[backend] = {
            "n_trials": len(rows),
            "device_acc_pct": 100.0 * subject_first_mean(rows, "acc"),
            "agreement_pct": agree,
            "flip_pct": 100.0 - agree,
            "mean_relative_l2": subject_first_mean(rows, "rel_l2"),
            "cosine_distance": subject_first_mean(rows, "cos_dist"),
            "latency_ms": subject_first_mean(rows, "latency_ms"),
        }
    return out


# ---------------------------------------------------------------- Stage 1 / Stage 2 join

def join_stages(stage1: list[dict], replay: dict[str, list[ReplayTrial]]) -> tuple[list[dict], dict[str, Any]]:
    s1 = {(r["backend"], r["subject"], r["class_id"], r["trial_idx"]): r for r in stage1}
    if len(s1) != len(stage1):
        raise ValueError("duplicate Stage 1 trial identities")
    joined, missing, label_errors, middle_mismatch = [], 0, 0, Counter()
    neural_keys = set()
    for backend in NEURAL_BACKENDS:
        for t in replay[backend]:
            key = (backend, t.subject, t.class_id, t.trial_idx)
            neural_keys.add(key)
            r = s1.get(key)
            if r is None:
                missing += 1
                continue
            if r["true_label"] != t.true_label or r["trial_num"] != t.trial_num:
                label_errors += 1
            if r["tflite_prediction"] != t.host_argmax:
                middle_mismatch[backend] += 1
            joined.append({
                "backend": backend,
                "subject": t.subject,
                "class_id": t.class_id,
                "trial_idx": t.trial_idx,
                "true_label": t.true_label,
                "checkpoint": r["pytorch_prediction"],
                "host_tflite_export": r["tflite_prediction"],
                "host_tflite_replay": t.host_argmax,
                "device": t.device_argmax,
            })
    extra = len(set(s1) - neural_keys)
    report = {
        "stage1_rows": len(stage1),
        "joined_rows": len(joined),
        "replay_rows_without_stage1": missing,
        "stage1_rows_without_replay": extra,
        "label_or_trial_number_mismatches": label_errors,
        "export_host_vs_replay_host_prediction_mismatches": dict(middle_mismatch),
        "passed": missing == 0 and extra == 0 and label_errors == 0 and not middle_mismatch,
    }
    return joined, report


def table3(joined: list[dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for backend in NEURAL_BACKENDS:
        rows = [r for r in joined if r["backend"] == backend]
        s1 = 100.0 * statistics.mean(r["checkpoint"] == r["host_tflite_export"] for r in rows)
        s2 = 100.0 * statistics.mean(r["host_tflite_replay"] == r["device"] for r in rows)
        f1, f2 = 100.0 - s1, 100.0 - s2
        boundary = "None" if f1 == 0 and f2 == 0 else ("Stage 1" if f1 > f2 else "Stage 2")
        out[backend] = {
            "n_trials": len(rows),
            "stage1_agree_pct": s1,
            "stage1_flip_pct": f1,
            "stage2_agree_pct": s2,
            "stage2_flip_pct": f2,
            "larger_boundary": boundary,
        }
    return out


TRANSITIONS = ("preserved", "export_created_retained", "device_created", "checkpoint_restored", "third_class")


def transition_path(checkpoint: int, host: int, device: int) -> str:
    if checkpoint == host == device:
        return "preserved"
    if checkpoint != host and device == host:
        return "export_created_retained"
    if checkpoint == host and device != host:
        return "device_created"
    if checkpoint != host and device == checkpoint:
        return "checkpoint_restored"
    return "third_class"


def transitions(joined: list[dict]) -> tuple[dict[str, dict[str, int]], list[dict]]:
    counts = {b: dict.fromkeys(TRANSITIONS, 0) for b in NEURAL_BACKENDS}
    records = []
    for r in joined:
        path = transition_path(r["checkpoint"], r["host_tflite_replay"], r["device"])
        counts[r["backend"]][path] += 1
        records.append({**r, "transition": path})
    return counts, records


# ---------------------------------------------------------------- int8 flips

def int8_flips(replay: dict[str, list[ReplayTrial]]) -> tuple[dict[str, Any], list[dict]]:
    flips = []
    for backend in INT8_BACKENDS:
        for t in replay[backend]:
            if t.host_argmax != t.device_argmax:
                host_ok, dev_ok = t.host_argmax == t.true_label, t.device_argmax == t.true_label
                kind = "harmful" if host_ok and not dev_ok else "beneficial" if dev_ok and not host_ok else "both_wrong"
                flips.append({
                    "backend": backend, "subject": t.subject, "class_id": t.class_id,
                    "trial_idx": t.trial_idx, "trial_num": t.trial_num, "true_label": t.true_label,
                    "host_prediction": t.host_argmax, "device_prediction": t.device_argmax,
                    "host_margin": prediction_and_margin(t.host)[1], "kind": kind,
                })
    sets = {b: {(f["subject"], f["class_id"], f["trial_idx"]) for f in flips if f["backend"] == b}
            for b in INT8_BACKENDS}
    c, e, f = (sets[b] for b in INT8_BACKENDS)
    summary = {
        "total": len(flips),
        "per_backend": {b: len(sets[b]) for b in INT8_BACKENDS},
        "distinct_trials": len(c | e | f),
        "pairwise_overlap": {
            "CCNN-int8&EEGNet-int8": len(c & e),
            "CCNN-int8&FBtCNN-int8": len(c & f),
            "EEGNet-int8&FBtCNN-int8": len(e & f),
        },
        "all_three_overlap": len(c & e & f),
        "kind": {b: dict(Counter(x["kind"] for x in flips if x["backend"] == b)) for b in INT8_BACKENDS},
    }
    return summary, flips


# ---------------------------------------------------------------- trial vectors (normalized metrics, quartiles)

def trial_vectors(replay: dict[str, list[ReplayTrial]], classical: list[dict]) -> list[dict]:
    """Classical vectors from the classical fidelity record, neural vectors from the replay JSONL."""
    rows = []

    def make(backend, family, subject, trial_num, class_id, trial_idx, true_label, host, device):
        hp, hm = prediction_and_margin(host)
        dp, _ = prediction_and_margin(device)
        mae, rmse, mx = raw_vector_errors(host, device)
        cos = cosine_similarity(host, device)
        return {
            "backend": backend, "family": family, "subject": subject, "trial_num": trial_num,
            "class_id": class_id, "trial_idx": trial_idx, "true_label": true_label,
            "host_prediction": hp, "device_prediction": dp, "prediction_flip": int(hp != dp),
            "host_margin": hm, "relative_l2_error": relative_l2_error(host, device),
            "cosine_similarity": cos, "cosine_distance": 1.0 - cos,
            "logit_mae": mae, "logit_rmse": rmse, "max_abs_logit_error": mx,
            "host_norm": vector_norm(host),
        }

    for r in classical:
        row = make(r["backend"], "classical", r["subject"], r["trial_num"], r["class_id"],
                   r["trial_idx"], r["true_label"], r["host"], r["device"])
        if row["prediction_flip"] != r["prediction_flip"]:
            raise ValueError(f"{r['backend']}/{r['subject']}/{r['trial_num']}: stored flip differs")
        if not math.isclose(row["host_margin"], r["host_margin"], rel_tol=0.0,
                            abs_tol=CLASSICAL_SERIALIZATION_TOLERANCE):
            raise ValueError(f"{r['backend']}/{r['subject']}/{r['trial_num']}: stored margin differs")
        rows.append(row)
    for backend in NEURAL_BACKENDS:
        for t in replay[backend]:
            rows.append(make(backend, "neural", t.subject, t.trial_num, t.class_id, t.trial_idx,
                             t.true_label, t.host, t.device))
    expected = len(ALL_BACKENDS) * len(SUBJECTS) * TRIALS_PER_SUBJECT
    if len(rows) != expected:
        raise ValueError(f"expected {expected} trial vectors, found {len(rows)}")
    return rows


def classical_source_agreement(replay: dict[str, list[ReplayTrial]], classical: list[dict]) -> dict[str, Any]:
    """Classical vectors in the replay JSONL versus the classical fidelity record."""
    index = {(r["backend"], r["subject"], r["trial_num"]): r for r in classical}
    worst = 0.0
    missing = 0
    for backend in CLASSICAL_BACKENDS:
        for t in replay[backend]:
            r = index.get((backend, t.subject, t.trial_num))
            if r is None:
                missing += 1
                continue
            worst = max(worst, max(abs(a - b) for a, b in zip(r["host"] + r["device"], t.host + t.device)))
    return {"missing": missing, "max_abs_diff": worst, "passed": missing == 0 and worst < 5e-8}


def normalized_summary(vectors: list[dict]) -> list[dict]:
    out = []
    for backend in ALL_BACKENDS:
        items = [r for r in vectors if r["backend"] == backend]
        rel = [r["relative_l2_error"] for r in items]
        cos = [r["cosine_similarity"] for r in items]
        cdist = [r["cosine_distance"] for r in items]
        out.append({
            "backend": backend,
            "n_trials": len(items),
            "mean_relative_l2_error": subject_first_mean(items, "relative_l2_error"),
            "median_relative_l2_error": statistics.median(rel),
            "p95_relative_l2_error": percentile(rel, 0.95),
            "mean_cosine_similarity": subject_first_mean(items, "cosine_similarity"),
            "median_cosine_similarity": statistics.median(cos),
            "p05_cosine_similarity": percentile(cos, 0.05),
            "mean_cosine_distance": subject_first_mean(items, "cosine_distance"),
            "median_cosine_distance": statistics.median(cdist),
            "p95_cosine_distance": percentile(cdist, 0.95),
            "near_zero_host_norm_count": sum(r["host_norm"] < NEAR_ZERO_HOST_NORM_THRESHOLD for r in items),
            "epsilon": EPSILON,
        })
    return out


def margin_quartiles(vectors: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Within backend and held-out subject average-rank quartiles; subject bootstrap per quartile."""
    annotated = []
    unequal_groups = 0
    for backend in INT8_BACKENDS:
        for subject in SUBJECTS:
            items = [r for r in vectors if r["backend"] == backend and r["subject"] == subject]
            ranks = average_ranks([r["host_margin"] for r in items])
            group = [{**r, "quartile": quartile_from_rank(k, len(items))} for r, k in zip(items, ranks)]
            if any(Counter(g["quartile"] for g in group)[q] != 45 for q in QUARTILES):
                unequal_groups += 1
            annotated.extend(group)

    figure_rows = []
    for backend in INT8_BACKENDS:
        items = [r for r in annotated if r["backend"] == backend]
        total_flips = sum(r["prediction_flip"] for r in items)
        for quartile in QUARTILES:
            q_items = [r for r in items if r["quartile"] == quartile]
            rates = []
            for subject in SUBJECTS:
                s_items = [r for r in q_items if r["subject"] == subject]
                rates.append(sum(r["prediction_flip"] for r in s_items) / len(s_items))
            seed = BOOTSTRAP_SEEDS[(backend, quartile)]
            lo, hi = subject_bootstrap_ci(rates, seed=seed)
            n_flips = sum(r["prediction_flip"] for r in q_items)
            mean_rate = statistics.mean(rates)
            figure_rows.append({
                "backend": backend,
                "host_margin_quartile": quartile,
                "n_subjects": len(rates),
                "n_trials": len(q_items),
                "n_flips": n_flips,
                "pooled_flip_rate": n_flips / len(q_items),
                "subject_mean_flip_rate": mean_rate,
                "subject_mean_flip_rate_pct": 100.0 * mean_rate,
                "bootstrap_ci_lo": lo,
                "bootstrap_ci_hi": hi,
                "bootstrap_ci_lo_pct": 100.0 * lo,
                "bootstrap_ci_hi_pct": 100.0 * hi,
                "backend_total_flips": total_flips,
                "pct_of_backend_flips": 100.0 * n_flips / total_flips,
                "bootstrap_seed": seed,
                "bootstrap_n": BOOTSTRAP_N,
            })
    info = {"groups": len(INT8_BACKENDS) * len(SUBJECTS), "groups_with_unequal_quartiles_due_to_ties": unequal_groups}
    return figure_rows, info
