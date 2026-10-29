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


def _output_dirs(stdout: str) -> list[Path]:
    found = []
    for line in stdout.splitlines():
        if line.startswith("output-dir:"):
            found.append(Path(line.split(":", 1)[1].strip()))
    return found


def _write_cca_config(path: Path, run_dir: Path | None) -> None:
    text = (
        "method: cca\n"
        "cca:\n"
        "  sampling_rate_hz: 256.0\n"
        "  n_harmonics: 3\n"
        "  single_band_hz: [8.0, 55.0]\n"
        "  butterworth_order: 4\n"
        "  cca_reg: 1.0e-4\n"
    )
    if run_dir is not None:
        text += f'output:\n  run_dir: "{run_dir.as_posix()}"\n'
    path.write_text(text, encoding="utf-8")


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )


def test_classical_output_dir_cli_overrides_yaml(tmp_path):
    cli_dir = tmp_path / "from_cli"
    yaml_dir = tmp_path / "from_yaml"
    cfg = tmp_path / "cca.yaml"
    _write_cca_config(cfg, yaml_dir)
    runner = _run_cmd(
        [
            sys.executable,
            "-m",
            "src.training_entry.run_classical_loso_12jfpm",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--output-dir",
            str(cli_dir),
            "--dry-run",
        ]
    )
    assert runner.returncode == 0, runner.stderr
    assert _output_dirs(runner.stdout) == [cli_dir]
    host = _run_cmd(
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "classical",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--output-dir",
            str(cli_dir),
            "--dry-run",
        ]
    )
    assert host.returncode == 0, host.stderr + host.stdout
    assert _output_dirs(host.stdout) == [cli_dir]
    assert not cli_dir.exists()
    assert not yaml_dir.exists()


def test_classical_output_alias_overrides_yaml(tmp_path):
    cli_dir = tmp_path / "from_alias"
    yaml_dir = tmp_path / "from_yaml"
    cfg = tmp_path / "cca.yaml"
    _write_cca_config(cfg, yaml_dir)
    host = _run_cmd(
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "classical",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--output",
            str(cli_dir),
            "--dry-run",
        ]
    )
    assert host.returncode == 0, host.stderr + host.stdout
    assert _output_dirs(host.stdout) == [cli_dir]
    assert not cli_dir.exists()
    assert not yaml_dir.exists()


def test_classical_output_dir_uses_yaml_when_cli_omitted(tmp_path):
    yaml_dir = tmp_path / "from_yaml"
    cfg = tmp_path / "cca.yaml"
    _write_cca_config(cfg, yaml_dir)
    runner = _run_cmd(
        [
            sys.executable,
            "-m",
            "src.training_entry.run_classical_loso_12jfpm",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--dry-run",
        ]
    )
    assert runner.returncode == 0, runner.stderr
    assert _output_dirs(runner.stdout) == [yaml_dir]
    host = _run_cmd(
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "classical",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--dry-run",
        ]
    )
    assert host.returncode == 0, host.stderr + host.stdout
    assert _output_dirs(host.stdout) == [yaml_dir]
    assert not yaml_dir.exists()


def test_classical_output_dir_defaults_when_unset(tmp_path):
    cfg = tmp_path / "cca.yaml"
    _write_cca_config(cfg, None)
    default = Path("runs/classical_loso_12jfpm")
    on_disk = ROOT / default
    before = on_disk.exists()
    json_before = (on_disk / "cca_loso_12jfpm.json").exists()
    runner = _run_cmd(
        [
            sys.executable,
            "-m",
            "src.training_entry.run_classical_loso_12jfpm",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--dry-run",
        ]
    )
    assert runner.returncode == 0, runner.stderr
    assert _output_dirs(runner.stdout) == [default]
    assert on_disk.exists() is before
    assert (on_disk / "cca_loso_12jfpm.json").exists() is json_before


def test_classical_dry_run_creates_no_output_files(tmp_path):
    cli_dir = tmp_path / "dry"
    cfg = tmp_path / "cca.yaml"
    _write_cca_config(cfg, tmp_path / "yaml_unused")
    before = sorted(p.name for p in tmp_path.iterdir())
    result = _run_cmd(
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "classical",
            "--method",
            "cca",
            "--config",
            str(cfg),
            "--output-dir",
            str(cli_dir),
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert _output_dirs(result.stdout) == [cli_dir]
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not cli_dir.exists()
