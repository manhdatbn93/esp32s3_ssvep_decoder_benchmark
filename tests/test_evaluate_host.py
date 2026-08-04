"""Host-evaluation interface dry-run checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.evaluation.neural_host import (
    TFLITE_MODELS,
    build_model,
    load_yaml,
    preprocess_trial_for_tflite,
    resolve_config,
)
from src.neural import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    return env


def test_classical_dry_run():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "classical",
            "--method",
            "all",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, r.stderr
    assert "OK cca" in r.stdout


def test_pytorch_dry_run_all_models():
    r = subprocess.run(
        [sys.executable, "scripts/evaluate_host.py", "--runtime", "pytorch", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, r.stderr + r.stdout
    for name in MODEL_REGISTRY:
        assert f"OK pytorch {name}" in r.stdout


def test_tflite_dry_run_adapters():
    r = subprocess.run(
        [sys.executable, "scripts/evaluate_host.py", "--runtime", "tflite", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, r.stderr + r.stdout
    for name in sorted(TFLITE_MODELS):
        assert f"OK tflite {name}" in r.stdout


def test_tflite_preprocess_shapes():
    trial = np.zeros((512, 8), dtype=np.float32)
    expected = {
        "eegnet": (1, 512, 8),
        "ssvepnet": (1, 512, 8),
        "convca": (1, 512, 8),
        "ccnn": (1, 1, 8, 220),
        "fbtcnn": (1, 4, 8, 512),
        "ssvepformer": (1, 8, 560),
    }
    for name, shape in expected.items():
        out = preprocess_trial_for_tflite(name, trial)
        assert tuple(out.shape) == shape


def test_build_model_from_each_config():
    for name in MODEL_REGISTRY:
        cfg = load_yaml(resolve_config(name))
        model = build_model(name, cfg)
        assert isinstance(model, MODEL_REGISTRY[name])
