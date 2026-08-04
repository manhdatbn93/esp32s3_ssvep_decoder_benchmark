"""Reference artifact manifest, resolution, and overwrite guards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from src.artifacts.reference import (
    load_reference_manifest,
    resolve_analytical_bundle,
    resolve_checkpoint,
    resolve_tflite,
    verify_reference_artifacts,
)
from src.evaluation.neural_host import build_model, load_yaml, resolve_config
from src.neural import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "artifacts" / "reference"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    return env


def test_manifest_schema_and_counts():
    entries = load_reference_manifest(REF)
    assert len(entries) == 220
    ck = [e for e in entries if e["artifact_type"] == "checkpoint"]
    tf = [e for e in entries if e["artifact_type"] == "tflite"]
    assert len(ck) == 40
    assert len(tf) == 70
    ids = [e["artifact_id"] for e in entries]
    assert len(ids) == len(set(ids))
    banned = ("paper", "manuscript", "ojsc", "source_private_path", "release_storage_tier")
    blob = json.dumps(entries)
    for term in banned:
        assert term not in blob


def test_verify_reference_artifacts_script():
    r = subprocess.run(
        [sys.executable, "scripts/verify_reference_artifacts.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "checkpoints=40" in r.stdout
    assert "tflite=70" in r.stdout


def test_verify_helper_ok():
    report = verify_reference_artifacts(REF, verify_checksum=True)
    assert report["ok"], report["errors"][:5]


def test_checkpoint_resolution_and_forward():
    for model in ("ccnn", "convca", "eegnet", "fbtcnn"):
        art = resolve_checkpoint(model, 1, verify_checksum=True)
        assert art["held_out_subject"] == "s1"
        ckpt = torch.load(art["path"], map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        cfg = load_yaml(resolve_config(model))
        net = build_model(model, cfg)
        if model == "convca":
            if "templates" in ckpt:
                net.set_templates(ckpt["templates"])
            else:
                net.set_templates(torch.randn(12, 512, 8))
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(2, 512, 8))
        assert tuple(out.shape) == (2, 12)


def test_tflite_resolution_metadata_fold1():
    import tensorflow as tf

    cases = [
        ("ccnn", "f32", (1, 1, 8, 220)),
        ("ccnn", "int8", (1, 1, 8, 220)),
        ("convca", "f32", (1, 512, 8)),
        ("eegnet", "f32", (1, 512, 8)),
        ("eegnet", "int8", (1, 512, 8)),
        ("fbtcnn", "f32", (1, 4, 8, 512)),
        ("fbtcnn", "int8", (1, 4, 8, 512)),
    ]
    for model, precision, expected_in in cases:
        art = resolve_tflite(model, precision, 1, verify_checksum=True)
        interp = tf.lite.Interpreter(model_path=str(art["path"]))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        # onnx2tf may transpose to NHWC
        in_shape = tuple(inp["shape"])
        assert in_shape[0] == 1
        assert tuple(out["shape"])[-1] == 12 or tuple(out["shape"]) == (1, 12)
        assert np_prod(in_shape) == np_prod(expected_in)


def np_prod(shape) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n


def test_analytical_bundle_fold1():
    for backend in ("cca", "fbcca", "msi", "trca"):
        bundle = resolve_analytical_bundle(backend, 1, verify_checksum=True)
        assert bundle["held_out_subject"] == "s1"
        assert bundle["paths"]


def test_reference_host_and_staging_dry_runs():
    cmds = [
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "pytorch",
            "--backend",
            "eegnet",
            "--fold",
            "1",
            "--artifact-set",
            "reference",
            "--dry-run",
        ],
        [
            sys.executable,
            "scripts/evaluate_host.py",
            "--runtime",
            "tflite",
            "--backend",
            "eegnet",
            "--precision",
            "int8",
            "--fold",
            "1",
            "--artifact-set",
            "reference",
            "--dry-run",
        ],
        [
            sys.executable,
            "scripts/generate_firmware_artifacts.py",
            "--backend",
            "trca",
            "--fold",
            "1",
            "--artifact-set",
            "reference",
            "--dry-run",
        ],
        [
            sys.executable,
            "scripts/replay_device.py",
            "--backend",
            "eegnet-int8",
            "--fold",
            "1",
            "--artifact-set",
            "reference",
            "--dry-run",
        ],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=_env())
        assert r.returncode == 0, cmd[-1] + "\n" + r.stdout + r.stderr


def test_export_cannot_write_reference():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/export_tflite.py",
            "--model",
            "eegnet",
            "--fold",
            "1",
            "--checkpoint",
            str(REF / "checkpoints" / "eegnet" / "fold01" / "fold01_best.pt"),
            "--dataset-root",
            str(ROOT),
            "--manifest",
            str(ROOT / "manifests" / "loso" / "fold01.json"),
            "--output",
            "artifacts/reference/tflite/eegnet_f32/fold01",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert r.returncode != 0
    assert "Refusing to write" in (r.stderr + r.stdout)


def test_missing_artifact_fails():
    try:
        resolve_tflite("eegnet", "f32", 99)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_no_ssvepformer_ssvepnet_reference_checkpoints():
    entries = load_reference_manifest(REF)
    models = {e["model"] for e in entries if e["artifact_type"] == "checkpoint"}
    assert "ssvepformer" not in models
    assert "ssvepnet" not in models
    assert MODEL_REGISTRY["ssvepformer"] is not None
