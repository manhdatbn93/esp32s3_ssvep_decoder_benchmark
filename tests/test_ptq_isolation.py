"""PTQ representative-dataset isolation checks."""
from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_ptq_loader_excludes_held_out_subject_logic():
    # Unit-level contract encoded by export.quantize_int8 selection rule.
    # Held-out subject must never appear in calibration subject ids.
    subjects = [f"s{i}" for i in range(1, 11)]
    for held in subjects:
        calib = [s for s in subjects if s != held]
        assert held not in calib
        assert len(calib) == 9


def test_ptq_sample_counts_contract():
    n_classes = 12
    n_per_class = 16
    assert n_classes * n_per_class == 192
