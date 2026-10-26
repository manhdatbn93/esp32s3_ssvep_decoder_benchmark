"""Deployment-policy admissibility, Pareto sets, and lexicographic selection (Table 4, Tables S15-S16).

Two objective variants are evaluated on the same measured inputs. Reported
backend selections are the same for both.

* ``as_executed``: the submission-time analysis. This is the canonical objective
  list. It minimizes the empirical 95th percentile of relative L2 error among
  the Pareto objectives of P1, P3, P4, and P5, and uses that percentile as the
  second P3 priority.
* ``as_described``: a sensitivity check that omits the P95 relative L2 term.
  That shorter list is the one printed in the manuscript.

Both give identical admissible sets, Pareto sets, and selections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

VARIANTS = ("as_described", "as_executed")


@dataclass(frozen=True)
class Backend:
    name: str
    family: str
    precision: str
    device_accuracy_pct: float
    agreement_pct: float
    latency_ms: float
    app_kib: float
    arena_kib: float | None
    p95_relative_l2: float

    @property
    def tflm_dependency(self) -> float:
        return 1.0 if self.family == "neural" else 0.0

    def metric(self, name: str) -> float | None:
        value = getattr(self, name)
        return None if value is None else float(value)


@dataclass(frozen=True)
class Gate:
    metric: str
    op: str
    threshold: float

    def passes(self, b: Backend) -> bool:
        v = b.metric(self.metric)
        if v is None or not math.isfinite(v):
            return False
        return {"<=": v <= self.threshold, ">=": v >= self.threshold, "==": v == self.threshold}[self.op]


Objective = tuple[str, str]


@dataclass(frozen=True)
class Policy:
    policy_id: str
    name: str
    candidates: Callable[[Backend], bool]
    gates: tuple[Gate, ...]
    pareto: tuple[Objective, ...]
    priorities: tuple[tuple[str, tuple[Objective, ...]], ...]


A, G, L, S, M, P95 = ("device_accuracy_pct", "max"), ("agreement_pct", "max"), ("latency_ms", "min"), \
    ("app_kib", "min"), ("arena_kib", "min"), ("p95_relative_l2", "min")


def definitions(variant: str) -> tuple[Policy, ...]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    executed = variant == "as_executed"
    neural = lambda b: b.family == "neural"
    int8 = lambda b: neural(b) and b.precision == "int8"
    f32 = lambda b: neural(b) and b.precision == "f32"
    classical = lambda b: b.family == "classical"
    neural_obj = (A, G, P95, L, S, M) if executed else (A, G, L, S, M)
    return (
        Policy("P1", "High-fidelity sub-250-ms neural", neural,
               (Gate(A[0], ">=", 85.0), Gate(G[0], ">=", 99.5), Gate(L[0], "<=", 250.0),
                Gate(S[0], "<=", 1250.0), Gate(M[0], "<=", 128.0)),
               neural_obj, (("P1", (A, L, S)),)),
        Policy("P2", "Sub-100-ms neural processing", neural,
               (Gate(A[0], ">=", 80.0), Gate(G[0], ">=", 96.0), Gate(L[0], "<=", 100.0),
                Gate(S[0], "<=", 1120.0), Gate(M[0], "<=", 128.0)),
               (A, G, L, S, M), (("P2", (A, G, L)),)),
        Policy("P3", "Fidelity-first int8 trade-off", int8,
               (Gate(A[0], ">=", 80.0), Gate(G[0], ">=", 96.0), Gate(L[0], "<=", 2500.0),
                Gate(S[0], "<=", 1250.0), Gate(M[0], "<=", 384.0)),
               neural_obj, (("P3", (G, P95, A, L) if executed else (G, A, L)),)),
        Policy("P4", "Latency-relaxed analytical no-TFLM", classical,
               (Gate(A[0], ">=", 80.0), Gate(G[0], ">=", 98.0), Gate(L[0], "<=", 15000.0),
                Gate(S[0], "<=", 3000.0), Gate("tflm_dependency", "==", 0.0)),
               (A, G, P95, L, S) if executed else (A, G, L, S), (("P4", (L, S, A)),)),
        Policy("P5", "High-fidelity f32 trade-off", f32,
               (Gate(A[0], ">=", 85.0), Gate(G[0], ">=", 99.9), Gate(L[0], "<=", 2500.0),
                Gate(S[0], "<=", 1250.0), Gate(M[0], "<=", 320.0)),
               neural_obj, (("P5-A", (A, L)), ("P5-B", (L, A)))),
    )


def dominates(x: Backend, y: Backend, objectives: Sequence[Objective]) -> bool:
    no_worse, better = True, False
    for metric, direction in objectives:
        a, b = x.metric(metric), y.metric(metric)
        if a is None or b is None:
            raise ValueError(f"undefined objective {metric} for {x.name} or {y.name}")
        if direction == "max":
            no_worse &= a >= b
            better |= a > b
        else:
            no_worse &= a <= b
            better |= a < b
    return no_worse and better


def priority_key(b: Backend, objectives: Sequence[Objective]) -> tuple:
    return tuple(-b.metric(m) if d == "max" else b.metric(m) for m, d in objectives) + (b.name,)


def evaluate(backends: Mapping[str, Backend], variant: str) -> list[dict]:
    out = []
    for p in definitions(variant):
        candidates = sorted(n for n, b in backends.items() if p.candidates(b))
        failures, admissible = [], []
        for n in candidates:
            failed = [g for g in p.gates if not g.passes(backends[n])]
            if failed:
                failures += [{"backend": n, "metric": g.metric, "op": g.op, "threshold": g.threshold,
                              "observed": backends[n].metric(g.metric)} for g in failed]
            else:
                admissible.append(n)
        pareto = [n for n in admissible
                  if not any(dominates(backends[o], backends[n], p.pareto) for o in admissible if o != n)]
        selected = {pid: (min(pareto, key=lambda n: priority_key(backends[n], obj)) if pareto else None)
                    for pid, obj in p.priorities}
        out.append({
            "policy_id": p.policy_id,
            "name": p.name,
            "gates": [(g.metric, g.op, g.threshold) for g in p.gates],
            "pareto_objectives": [f"{m}:{d}" for m, d in p.pareto],
            "priorities": {pid: [f"{m}:{d}" for m, d in obj] for pid, obj in p.priorities},
            "candidates": candidates,
            "admissible": admissible,
            "pareto": pareto,
            "selected": selected,
            "exclusions": failures,
        })
    return out


AGREEMENT_THRESHOLDS = (96.0, 98.0, 99.0, 99.5, 100.0)
LATENCY_THRESHOLDS_MS = (100.0, 250.0, 1000.0, 2500.0, 15000.0)


def sensitivity_grid(backends: Mapping[str, Backend]) -> list[dict]:
    """Neural agreement-by-latency grid: accuracy >= 80 %, select max accuracy then min latency."""
    rows = []
    neural = {n: b for n, b in backends.items() if b.family == "neural"}
    for g in AGREEMENT_THRESHOLDS:
        for lat in LATENCY_THRESHOLDS_MS:
            adm = sorted(n for n, b in neural.items()
                         if b.device_accuracy_pct >= 80.0 and b.agreement_pct >= g and b.latency_ms <= lat)
            sel = min(adm, key=lambda n: priority_key(neural[n], (A, L))) if adm else "none"
            rows.append({"min_agreement_pct": g, "max_latency_ms": lat,
                         "admissible_backends": " | ".join(adm), "selected_backend": sel,
                         "n_admissible": len(adm)})
    return rows
