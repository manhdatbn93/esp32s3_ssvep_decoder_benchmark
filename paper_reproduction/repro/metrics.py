"""Output-vector metrics, margin ranking, and subject bootstrap (pure Python, deterministic)."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

from .constants import BOOTSTRAP_N, EPSILON, N_CLASSES


# Sums use math.fsum (exactly rounded). For f32 backends 1 - cosine similarity is
# about 1e-14, and naive summation order already changes its third significant figure.
def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in values))


def relative_l2_error(host: Sequence[float], device: Sequence[float]) -> float:
    """||device - host||_2 / max(||host||_2, 1e-12)."""
    diff = vector_norm(d - h for h, d in zip(host, device, strict=True))
    return diff / max(vector_norm(host), EPSILON)


def cosine_similarity(host: Sequence[float], device: Sequence[float]) -> float:
    dot = math.fsum(h * d for h, d in zip(host, device, strict=True))
    return dot / max(vector_norm(host) * vector_norm(device), EPSILON)


def raw_vector_errors(host: Sequence[float], device: Sequence[float]) -> tuple[float, float, float]:
    abs_diff = [abs(d - h) for h, d in zip(host, device, strict=True)]
    mae = statistics.mean(abs_diff)
    rmse = math.sqrt(statistics.mean(v * v for v in abs_diff))
    return mae, rmse, max(abs_diff)


def prediction_and_margin(values: Sequence[float]) -> tuple[int, float]:
    """Top-1 class (lowest index wins ties) and top-1 minus top-2 score."""
    if len(values) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} outputs, found {len(values)}")
    order = sorted(range(N_CLASSES), key=lambda i: (-values[i], i))
    return order[0], values[order[0]] - values[order[1]]


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def average_ranks(values: Sequence[float]) -> list[float]:
    """One-based ranks; exact ties share their average rank."""
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start
        while stop + 1 < len(indexed) and indexed[stop + 1][1] == indexed[start][1]:
            stop += 1
        average = (start + stop + 2) / 2.0
        for k in range(start, stop + 1):
            ranks[indexed[k][0]] = average
        start = stop + 1
    return ranks


def quartile_from_rank(rank: float, count: int) -> str:
    p = (rank - 0.5) / count
    if p <= 0.25:
        return "Q1"
    if p <= 0.50:
        return "Q2"
    if p <= 0.75:
        return "Q3"
    return "Q4"


def subject_bootstrap_ci(subject_rates: Sequence[float], *, seed: int) -> tuple[float, float]:
    """Percentile 95% CI of the mean over held-out subjects, resampled with replacement."""
    rng = random.Random(seed)
    n = len(subject_rates)
    samples = [
        statistics.mean(subject_rates[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_N)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)
