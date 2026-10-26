"""Backend sets, trial layout, and fixed analysis parameters."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = PACKAGE_ROOT / "evidence"
EXPECTED = PACKAGE_ROOT / "expected" / "submitted_values.json"
MANIFEST = PACKAGE_ROOT / "MANIFEST.sha256"

SUBJECTS = tuple(f"s{i}" for i in range(1, 11))
CLASSICAL_BACKENDS = ("CCA", "FBCCA", "MSI", "TRCA")
NEURAL_BACKENDS = (
    "CCNN-f32",
    "CCNN-int8",
    "ConvCA-f32",
    "EEGNet-f32",
    "EEGNet-int8",
    "FBtCNN-f32",
    "FBtCNN-int8",
)
INT8_BACKENDS = ("CCNN-int8", "EEGNet-int8", "FBtCNN-int8")
ALL_BACKENDS = CLASSICAL_BACKENDS + NEURAL_BACKENDS
TABLE2_ORDER = ("CCA", "FBCCA", "MSI", "TRCA") + NEURAL_BACKENDS
QUARTILES = ("Q1", "Q2", "Q3", "Q4")

N_CLASSES = 12
TRIALS_PER_CLASS = 15
TRIALS_PER_SUBJECT = N_CLASSES * TRIALS_PER_CLASS

EPSILON = 1e-12
NEAR_ZERO_HOST_NORM_THRESHOLD = 1e-6
# The classical fidelity file stores vectors and margins with different decimal
# formatting; a margin recomputed from the stored vector differs by at most 9e-8.
CLASSICAL_SERIALIZATION_TOLERANCE = 1e-7

BOOTSTRAP_BASE_SEED = 20260625
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEEDS = {
    (backend, quartile): BOOTSTRAP_BASE_SEED + b * 10 + q
    for b, backend in enumerate(INT8_BACKENDS)
    for q, quartile in enumerate(QUARTILES)
}
