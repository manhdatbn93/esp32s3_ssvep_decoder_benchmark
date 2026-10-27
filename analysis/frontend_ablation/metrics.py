"""Feature, output, and decision metrics for one paired ablation trial.

Path A is the training-compatible frontend. Path B is the deployment-aligned
frontend. Relative errors use path A as the denominator. The floor is the
same ``1e-12`` used by the paper reproduction metrics. Top-1 ties resolve to
the lowest class index, matching ``prediction_and_margin``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from paper_reproduction.repro.metrics import (
    cosine_similarity,
    percentile,
    prediction_and_margin,
    relative_l2_error,
)

RELATIVE_L2_FLOOR = 1e-12


def _flat_float64(values: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("Empty feature tensor")
    return flat


def feature_pair_metrics(path_a: np.ndarray, path_b: np.ndarray) -> dict[str, float | int]:
    """Compare two model-input tensors. Shapes must already match."""
    if path_a.shape != path_b.shape:
        raise ValueError(f"Feature contract violation: {path_a.shape} vs {path_b.shape}")
    left = _flat_float64(path_a)
    right = _flat_float64(path_b)
    nonfinite = int((~np.isfinite(left)).sum() + (~np.isfinite(right)).sum())
    if nonfinite:
        return {
            "max_abs": math.nan,
            "mean_abs": math.nan,
            "relative_l2": math.nan,
            "cosine_distance": math.nan,
            "nonfinite_count": nonfinite,
        }
    diff = right - left
    denom = max(float(np.linalg.norm(left)), RELATIVE_L2_FLOOR)
    cosine_denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), RELATIVE_L2_FLOOR)
    cosine = float(np.dot(left, right) / cosine_denom)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "cosine_distance": float(1.0 - cosine),
        "nonfinite_count": 0,
    }


def output_pair_metrics(path_a: Sequence[float], path_b: Sequence[float]) -> dict[str, float]:
    """Compare two length-12 dequantized output vectors."""
    left = [float(v) for v in path_a]
    right = [float(v) for v in path_b]
    if len(left) != 12 or len(right) != 12:
        raise ValueError("Output vectors must have length 12")
    abs_diff = [abs(b - a) for a, b in zip(left, right, strict=True)]
    return {
        "max_abs": max(abs_diff),
        "mean_abs": math.fsum(abs_diff) / len(abs_diff),
        "relative_l2": relative_l2_error(left, right),
        "cosine_distance": 1.0 - cosine_similarity(left, right),
    }


def classify_flip(prediction_a: int, prediction_b: int, true_label: int) -> str:
    """Agree, harmful, beneficial, or both-wrong.

    Harmful means A is correct and B is not. Beneficial means B is correct
    and A is not. Both-wrong means the two predictions differ and neither
    matches the true label.
    """
    if prediction_a == prediction_b:
        return "agree"
    a_correct = prediction_a == true_label
    b_correct = prediction_b == true_label
    if a_correct and not b_correct:
        return "harmful"
    if b_correct and not a_correct:
        return "beneficial"
    return "both_wrong"


def top1_and_margin(values: Sequence[float]) -> tuple[int, float]:
    return prediction_and_margin(values)


def summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    """Mean, median, linearly interpolated P95, and maximum."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "maximum": math.nan}
    ordered = sorted(finite)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "n": len(finite),
        "mean": math.fsum(finite) / len(finite),
        "median": median,
        "p95": percentile(finite, 0.95),
        "maximum": ordered[-1],
    }
