"""Canonical TRCA evaluation against the retained deployed-host replay.

The numerical check recomputes predictions from the processed 12JFPM NPZ.
It skips on a fresh clone that does not contain that dataset. Archived replay
records are the comparison target only; they are not used as model inputs.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_NPZ = Path(
    os.environ.get("SSVEP_PROCESSED_NPZ", ROOT / "data" / "processed" / "12JFPM" / "processed.npz")
)
needs_dataset = pytest.mark.skipif(
    not PROCESSED_NPZ.is_file(),
    reason="processed 12JFPM NPZ not available (set SSVEP_PROCESSED_NPZ)",
)


def test_published_trca_default_is_deployed_sos():
    from src.evaluation.trca_12jfpm import TRCA_REALIZATIONS, trca_predict_fold

    default = inspect.signature(trca_predict_fold).parameters["realization"].default
    assert default == "deployed_sos"
    assert TRCA_REALIZATIONS == ("deployed_sos", "screening_ba")
    with pytest.raises(ValueError):
        trca_predict_fold(np.zeros((1, 512, 8)), np.zeros(1), np.zeros((1, 512, 8)), realization="other")


@needs_dataset
def test_deployed_sos_matches_retained_host_and_device():
    if str(ROOT / "paper_reproduction") not in sys.path:
        sys.path.insert(0, str(ROOT / "paper_reproduction"))
    from repro import evidence
    from src.evaluation.trca_12jfpm import TRCAConfig, trca_predict_fold

    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    subjects = data["subjects"] if "subjects" in data.files else data["subject"]
    x, y = data["X"], data["y"]
    replay = {(t.subject, t.class_id, t.trial_idx): t for t in evidence.load_replay()["TRCA"]}
    cfg = TRCAConfig()

    n = correct = host_agree = device_agree = 0
    screening_correct = screening_diff = 0
    for subject_id in [f"s{i}" for i in range(1, 11)]:
        mask = np.array([str(s) == subject_id for s in subjects])
        y_test = y[mask]
        pred = trca_predict_fold(x[~mask], y[~mask], x[mask], cfg, realization="deployed_sos")
        screening = trca_predict_fold(x[~mask], y[~mask], x[mask], cfg, realization="screening_ba")
        for i, (label, guess, screen) in enumerate(zip(y_test, pred, screening)):
            class_id, trial_idx = divmod(int(i), 15)
            assert int(label) == class_id
            rec = replay[(subject_id, class_id, trial_idx)]
            n += 1
            correct += int(guess) == int(label)
            host_agree += int(guess) == rec.host_argmax
            device_agree += int(guess) == rec.device_argmax
            screening_correct += int(screen) == int(label)
            screening_diff += int(screen) != int(guess)

    assert n == 1800
    assert correct == 1502
    assert correct / n == pytest.approx(0.83444444, abs=1e-8)
    assert host_agree == 1800
    assert device_agree == 1800
    assert screening_correct == 1510
    assert screening_correct / n == pytest.approx(0.83888889, abs=1e-8)
    assert screening_diff == 45
