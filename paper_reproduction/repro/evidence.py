"""Loaders for the released trial-level evidence, with identity and completeness checks."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    ALL_BACKENDS,
    CLASSICAL_BACKENDS,
    EVIDENCE,
    N_CLASSES,
    SUBJECTS,
    TRIALS_PER_CLASS,
    TRIALS_PER_SUBJECT,
)


@dataclass(frozen=True)
class ReplayTrial:
    backend: str
    subject: str
    run: str
    trial_num: int
    class_id: int
    trial_idx: int
    true_label: int
    host: tuple[float, ...]
    device: tuple[float, ...]
    host_argmax: int
    device_argmax: int
    elapsed_us: int

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.subject, self.class_id, self.trial_idx)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay_run_dir(backend: str, subject: str, evidence: Path = EVIDENCE) -> Path:
    runs = [p for p in (evidence / "replay" / backend).iterdir()
            if p.is_dir() and p.name.split("_", 1)[0] == subject]
    if len(runs) != 1:
        raise ValueError(f"{backend}/{subject}: expected exactly one replay run, found {len(runs)}")
    return runs[0]


def _validate_group(trials: list[ReplayTrial], label: str) -> None:
    if len(trials) != TRIALS_PER_SUBJECT:
        raise ValueError(f"{label}: expected {TRIALS_PER_SUBJECT} trials, found {len(trials)}")
    numbers = {t.trial_num for t in trials}
    identities = {(t.class_id, t.trial_idx) for t in trials}
    if numbers != set(range(1, TRIALS_PER_SUBJECT + 1)):
        raise ValueError(f"{label}: trial numbers are not 1..{TRIALS_PER_SUBJECT}")
    if identities != {(c, k) for c in range(N_CLASSES) for k in range(TRIALS_PER_CLASS)}:
        raise ValueError(f"{label}: (class_id, trial_idx) identities incomplete or duplicated")
    for t in trials:
        if t.trial_num != t.class_id * TRIALS_PER_CLASS + t.trial_idx + 1 or t.true_label != t.class_id:
            raise ValueError(f"{label}: trial {t.trial_num} identity mapping error")
        if len(t.host) != N_CLASSES or len(t.device) != N_CLASSES:
            raise ValueError(f"{label}: trial {t.trial_num} vector length")
        if not all(math.isfinite(v) for v in t.host + t.device):
            raise ValueError(f"{label}: trial {t.trial_num} non-finite output")


def load_replay(evidence: Path = EVIDENCE) -> dict[str, list[ReplayTrial]]:
    """All 110 replay runs (11 backends x 10 held-out subjects) from trial_results.jsonl."""
    out: dict[str, list[ReplayTrial]] = {}
    for backend in ALL_BACKENDS:
        rows: list[ReplayTrial] = []
        for subject in SUBJECTS:
            run = replay_run_dir(backend, subject, evidence)
            group = []
            with open(run / "trial_results.jsonl", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("status", 0) != 0 or r["subject"] != subject:
                        raise ValueError(f"{run}: unexpected status or subject in trial {r.get('trial_num')}")
                    group.append(ReplayTrial(
                        backend=backend,
                        subject=subject,
                        run=run.name,
                        trial_num=int(r["trial_num"]),
                        class_id=int(r["class_id"]),
                        trial_idx=int(r["trial_idx"]),
                        true_label=int(r["true_label"]),
                        host=tuple(float(v) for v in r["host_logits"]),
                        device=tuple(float(v) for v in r["device_logits"]),
                        host_argmax=int(r["host_argmax"]),
                        device_argmax=int(r["device_argmax"]),
                        elapsed_us=int(r["elapsed_us"]),
                    ))
            _validate_group(group, f"{backend}/{subject}")
            rows.extend(group)
        out[backend] = rows
    return out


def load_replay_csv_mirror(evidence: Path = EVIDENCE) -> dict[tuple, tuple]:
    """trial_results.csv values keyed by (backend, subject, trial_num), used to cross-check the JSONL."""
    out = {}
    for backend in ALL_BACKENDS:
        for subject in SUBJECTS:
            run = replay_run_dir(backend, subject, evidence)
            for r in read_csv(run / "trial_results.csv"):
                out[(backend, subject, int(r["trial_num"]))] = (
                    int(r["device_argmax"]),
                    int(r["host_argmax"]),
                    int(r["elapsed_us"]),
                    tuple(float(r[f"dev_logit_{k}"]) for k in range(N_CLASSES)),
                    tuple(float(r[f"host_logit_{k}"]) for k in range(N_CLASSES)),
                )
    return out


def load_replay_summaries(evidence: Path = EVIDENCE) -> dict[tuple[str, str], dict]:
    return {
        (backend, subject): json.loads((replay_run_dir(backend, subject, evidence) / "summary.json")
                                       .read_text(encoding="utf-8"))
        for backend in ALL_BACKENDS
        for subject in SUBJECTS
    }


def load_stage1(evidence: Path = EVIDENCE) -> list[dict]:
    """Checkpoint-versus-host-TFLite trial predictions (Stage 1) for the seven neural backends."""
    rows = []
    for r in read_csv(evidence / "fidelity" / "export_fidelity_per_trial.csv"):
        rows.append({
            "backend": r["backend"],
            "subject": r["subject"],
            "class_id": int(r["class_id"]),
            "trial_idx": int(r["trial_idx"]),
            "trial_num": int(r["trial_num"]),
            "true_label": int(r["true_label"]),
            "pytorch_prediction": int(r["pytorch_prediction"]),
            "tflite_prediction": int(r["tflite_prediction"]),
        })
    return rows


def load_stage2_derived(evidence: Path = EVIDENCE) -> list[dict]:
    return read_csv(evidence / "fidelity" / "per_trial_fidelity.csv")


def load_classical_fidelity(evidence: Path = EVIDENCE) -> list[dict]:
    rows = []
    for r in read_csv(evidence / "fidelity" / "classical_per_trial_fidelity.csv"):
        if r["backend"] not in CLASSICAL_BACKENDS:
            continue
        rows.append({
            "backend": r["backend"],
            "subject": r["subject"],
            "trial_num": int(r["trial_num"]),
            "class_id": int(r["class_id"]),
            "trial_idx": int(r["trial_idx"]),
            "true_label": int(r["true_label"]),
            "host": tuple(float(v) for v in r["host_scores"].split()),
            "device": tuple(float(v) for v in r["device_scores"].split()),
            "prediction_flip": int(r["prediction_flip"]),
            "host_margin": float(r["host_margin"]),
        })
    return rows
