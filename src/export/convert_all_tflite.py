#!/usr/bin/env python3
"""
Unified TFLite conversion + parity + operator inspection for all DL models.

Requires Python 3.10 venv with onnx2tf, tensorflow, torch, onnx, onnxruntime.
Run from repo root:
  $env:TF_ENABLE_ONEDNN_OPTS="0"; $env:PYTHONUTF8="1"
  .venv_tflite310\\Scripts\\python.exe -m export.convert_all_tflite
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from export.common import get_fold_test_vectors, load_processed_npz, write_host_reference, write_metadata
from export.protocol_cli import (
    DEFAULT_PROTOCOL,
    add_protocol_arg,
    add_test_subject_args,
    apply_protocol_side_effects,
    resolve_test_subject,
)
from export.subject_paths import (
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    assert_not_legacy_artifact,
    checkpoint_dir,
    export_dir,
    export_dir_name,
    fold_for_subject,
    legacy_checkpoint_path,
)

# Active fold / held-out subject. Overridden via --fold / --val-subject / --protocol.
ACTIVE_FOLD: int = 1
ACTIVE_VAL_SUBJECT: str = "s1"
ACTIVE_PROTOCOL: str = DEFAULT_PROTOCOL


def _ckpt_for(model: str) -> Path:
    """Per-fold checkpoint from protocol-aware checkpoints/<model>_.../."""
    path = checkpoint_dir(model, protocol=ACTIVE_PROTOCOL, repo=REPO) / (
        f"fold{ACTIVE_FOLD:02d}_best.pt"
    )
    assert_not_legacy_artifact(path, protocol=ACTIVE_PROTOCOL, context="checkpoint")

    if ACTIVE_PROTOCOL == PROTOCOL_STRICT_LOSO_INNER_VAL:
        legacy = legacy_checkpoint_path(model, ACTIVE_FOLD, repo=REPO)
        if path.is_file():
            return path
        if legacy.is_file():
            raise FileNotFoundError(
                f"Strict checkpoint missing: {path}\n"
                f"  Archived Version A checkpoint exists at {legacy} but is not loaded.\n"
                f"  Train with:\n"
                f"    python -m training.run_dl_loso_12jfpm --model {model} "
                f"--protocol strict_loso_inner_val --subjects {ACTIVE_VAL_SUBJECT}"
            )
        raise FileNotFoundError(
            f"Strict checkpoint not found: {path}\n"
            f"  Train with:\n"
            f"    python -m training.run_dl_loso_12jfpm --model {model} "
            f"--protocol strict_loso_inner_val --subjects {ACTIVE_VAL_SUBJECT}"
        )

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _train_mask(subjects) -> np.ndarray:
    """Boolean mask selecting TRAINING samples for the active val subject."""
    return np.array([str(s) != ACTIVE_VAL_SUBJECT for s in subjects])


def _onnx_export(model: nn.Module, dummy: torch.Tensor, path: Path,
                 input_names: list[str], output_names: list[str]) -> bool:
    try:
        torch.onnx.export(model, dummy, str(path),
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None, opset_version=18)
        return True
    except Exception as e:
        print(f"    ONNX export failed: {e}")
        return False


def _run_onnx2tf(onnx_path: Path, out_dir: Path, timeout: int = 300,
                  extra_args: list[str] | None = None) -> dict:
    """Run onnx2tf without int8 quantization (we do that separately)."""
    try:
        cmd = [sys.executable, "-m", "onnx2tf",
               "-i", str(onnx_path),
               "-o", str(out_dir)]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "TF_ENABLE_ONEDNN_OPTS": "0", "PYTHONUTF8": "1"},
        )
        tflite_files = list(out_dir.glob("*.tflite")) if out_dir.exists() else []
        if tflite_files:
            return {"success": True, "n_tflite": len(tflite_files)}
        if result.returncode == 0:
            return {"success": True, "n_tflite": 0, "note": "no tflite produced"}
        return {"success": False, "error": result.stderr[-500:] if result.stderr else "unknown"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)[:500]}


def _find_tflite_files(tf_dir: Path) -> dict[str, Path | None]:
    f32, i8 = None, None
    if not tf_dir.exists():
        return {"float32": None, "int8": None}
    for f in sorted(tf_dir.glob("*.tflite")):
        name = f.name.lower()
        if "integer_quant" in name:
            i8 = f
        elif "float32" in name:
            f32 = f
        elif "float16" in name:
            pass
        elif "dynamic_range" in name:
            pass
        elif f32 is None:
            f32 = f
    return {"float32": f32, "int8": i8}


def _find_dynamic_range_tflite(tf_dir: Path) -> Path | None:
    if not tf_dir.exists():
        return None
    for f in tf_dir.glob("*dynamic_range*.tflite"):
        return f
    return None


def _quantize_int8(saved_model_dir: Path, out_path: Path,
                   calib_data: list[np.ndarray]) -> bool:
    """Post-training int8 quantization using tf.lite.TFLiteConverter."""
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32

        def representative_dataset():
            for d in calib_data:
                yield [d.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        return True
    except Exception as e:
        print(f"    int8 quantization failed: {e}")
        return False


def _run_onnx_inference(onnx_path: Path, input_data: np.ndarray, input_name: str) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    return sess.run(None, {input_name: input_data.astype(np.float32)})[0]


def _run_tflite_inference(tflite_path: Path, input_data: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    expected_shape = tuple(inp_detail["shape"])
    data = input_data.astype(np.float32)

    if tuple(data.shape) != expected_shape:
        # onnx2tf transposes NCHW→NHWC; try transpose BEFORE reshape because
        # reshape merely reinterprets flat memory order while transpose
        # correctly moves axis values (they differ for non-singleton swaps).
        matched = False
        if data.ndim == 4 and len(expected_shape) == 4:
            data_t = np.transpose(data, (0, 2, 3, 1))
            if tuple(data_t.shape) == expected_shape:
                data = data_t
                matched = True
        if not matched and data.ndim == 3 and len(expected_shape) == 3:
            data_t = np.transpose(data, (0, 2, 1))
            if tuple(data_t.shape) == expected_shape:
                data = data_t
                matched = True
        if not matched:
            if np.prod(data.shape) == np.prod(expected_shape):
                data = data.reshape(expected_shape)
            else:
                raise ValueError(f"Shape mismatch: data {data.shape} vs expected {expected_shape}")

    inp_dtype = inp_detail["dtype"]
    if inp_dtype in (np.int8, np.uint8):
        scale, zp = inp_detail["quantization"]
        data = (data / scale + zp).clip(
            np.iinfo(inp_dtype).min, np.iinfo(inp_dtype).max
        ).astype(inp_dtype)
    else:
        data = data.astype(inp_dtype)

    interpreter.set_tensor(inp_detail["index"], data)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_detail["index"])

    out_dtype = out_detail["dtype"]
    if out_dtype in (np.int8, np.uint8):
        scale, zp = out_detail["quantization"]
        return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)


def _inspect_tflite_ops(tflite_path: Path) -> list[str]:
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        details = interpreter._get_ops_details()
        return sorted(set(d["op_name"] for d in details))
    except Exception:
        return ["(inspection failed)"]


def _parity_report(ref: np.ndarray, test: np.ndarray, label: str) -> dict:
    ref_f = ref.flatten().astype(np.float32)
    test_f = test.flatten().astype(np.float32)
    diff = np.abs(ref_f - test_f)
    return {
        "comparison": label,
        "argmax_match": bool(np.argmax(ref_f) == np.argmax(test_f)),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "ref_argmax": int(np.argmax(ref_f)),
        "test_argmax": int(np.argmax(test_f)),
    }


# ── Model handlers ──────────────────────────────────────────────────

def _handle_eegnet(vectors, out_dir):
    from src.neural.eegnet import EEGNet
    ckpt_path = _ckpt_for("eegnet")
    model = EEGNet(seq_len=512, embed_dim=8, num_classes=12,
                   F1=8, D=2, F2=16, dropout=0.25, kernel_length=64,
                   pool1_stride=4, pool2_stride=8)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 512, 8)
    onnx_path = out_dir / "eegnet.onnx"
    if not _onnx_export(model, dummy, onnx_path, ["input"], ["logits"]):
        return {"status": "onnx_failed"}

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "input",
            "input_shape": [1, 512, 8], "pt_refs": pt_refs, "preprocess": None,
            "preprocessing_note": "None — raw [B, T, C] input"}


def _handle_ccnn(vectors, out_dir):
    from src.neural.ccnn import CCNN, _complex_spectrum_features

    ckpt_path = _ckpt_for("ccnn")
    model = CCNN(seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.25)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class PostFFT(nn.Module):
        def __init__(self, full):
            super().__init__()
            self.layer1 = full.layer1
            self.layer2 = full.layer2
            self.classifier = full.classifier
        def forward(self, spec):
            return self.classifier(self.layer2(self.layer1(spec)))

    post = PostFFT(model)
    post.eval()
    dummy = torch.randn(1, 1, 8, 220)
    onnx_path = out_dir / "ccnn_postfft.onnx"
    if not _onnx_export(post, dummy, onnx_path, ["spectrum"], ["logits"]):
        return {"status": "onnx_failed"}

    def preprocess(raw_tc):
        t = torch.from_numpy(raw_tc).unsqueeze(0)
        return _complex_spectrum_features(t, 256.0).numpy()

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "spectrum",
            "input_shape": [1, 1, 8, 220], "pt_refs": pt_refs, "preprocess": preprocess,
            "preprocessing_note": "FFT outside model: [512,8] -> [1,1,8,220]"}


def _handle_ssvepformer(vectors, out_dir):
    from src.neural.ssvepformer import SSVEPformer, _ssvepformer_spectrum

    ckpt_path = _ckpt_for("ssvepformer")
    model = SSVEPformer(seq_len=512, embed_dim=8, num_classes=12,
                        sampling_rate_hz=256.0, depth=1, attention_kernel_length=31, dropout=0.5)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class PostFFT(nn.Module):
        def __init__(self, full):
            super().__init__()
            self.to_patch_embedding = full.to_patch_embedding
            self.transformer = full.transformer
            self.mlp_head = full.mlp_head
        def forward(self, spec):
            x = self.to_patch_embedding(spec)
            for attn_block, ff_block in self.transformer:
                x = attn_block(x) + x
                x = ff_block(x) + x
            return self.mlp_head(x)

    post = PostFFT(model)
    post.eval()
    dummy_raw = torch.randn(1, 512, 8)
    with torch.no_grad():
        spec_shape = _ssvepformer_spectrum(dummy_raw, 256.0).shape
    dummy_spec = torch.randn(*spec_shape)
    onnx_path = out_dir / "ssvepformer_postfft.onnx"
    if not _onnx_export(post, dummy_spec, onnx_path, ["spectrum"], ["logits"]):
        return {"status": "onnx_failed"}

    def preprocess(raw_tc):
        t = torch.from_numpy(raw_tc).unsqueeze(0)
        return _ssvepformer_spectrum(t, 256.0).numpy()

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "spectrum",
            "input_shape": list(spec_shape), "pt_refs": pt_refs, "preprocess": preprocess,
            "preprocessing_note": f"FFT outside model: [512,8] -> {list(spec_shape)}"}


def _handle_ssvepnet(vectors, out_dir):
    from src.neural.ssvepnet import SSVEPNet

    ckpt_path = _ckpt_for("ssvepnet")
    model = SSVEPNet(seq_len=512, embed_dim=8, num_classes=12, dropout=0.5)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 512, 8)
    onnx_path = out_dir / "ssvepnet.onnx"
    if not _onnx_export(model, dummy, onnx_path, ["input"], ["logits"]):
        return {"status": "onnx_failed"}

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "input",
            "input_shape": [1, 512, 8], "pt_refs": pt_refs, "preprocess": None,
            "preprocessing_note": "None — raw [B,T,C]. Contains Bi-LSTM."}


def _handle_fbtcnn(vectors, out_dir):
    from src.neural.fbtcnn import FBtCNN

    ckpt_path = _ckpt_for("fbtcnn")
    model = FBtCNN(seq_len=512, embed_dim=8, num_classes=12,
                   sampling_rate_hz=256.0, dropout=0.4)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class PostFilterBank(nn.Module):
        """Export wrapper: merges band dim into batch so sub_band runs once
        as a single forward pass — avoids the for-loop and zeros_like that
        produced invalid ONNX nodes."""
        def __init__(self, full):
            super().__init__()
            self.sub_band = full.sub_band
            self.final_conv = full.final_conv
            self.dense = full.dense
        def forward(self, x_bands):
            B, N, C, T = x_bands.shape
            feat = self.sub_band(x_bands.reshape(B * N, 1, C, T))
            _, Ch, H, W = feat.shape
            fused = feat.reshape(B, N, Ch, H, W).sum(dim=1)
            return self.dense(self.final_conv(fused))

    post = PostFilterBank(model)
    post.eval()
    dummy = torch.randn(1, 4, 8, 512)
    onnx_path = out_dir / "fbtcnn_postfb.onnx"
    if not _onnx_export(post, dummy, onnx_path, ["filtered_bands"], ["logits"]):
        return {"status": "onnx_failed"}

    def preprocess(raw_tc):
        x = torch.from_numpy(raw_tc).unsqueeze(0)
        x = x.transpose(1, 2).unsqueeze(1)
        bands = model._filter_bank(x)
        stacked = torch.stack(bands, dim=1)
        if stacked.dim() == 5:
            stacked = stacked.squeeze(2)
        return stacked.numpy()

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "filtered_bands",
            "input_shape": [1, 4, 8, 512], "pt_refs": pt_refs, "preprocess": preprocess,
            "preprocessing_note": "scipy.filtfilt 4-band filter bank outside model"}


def _handle_convca(vectors, out_dir):
    from src.neural.convca import ConvCA

    ckpt_path = _ckpt_for("convca")
    model = ConvCA(seq_len=512, embed_dim=8, num_classes=12,
                   dropout_sig=0.75, dropout_temp=0.15)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "templates" in ckpt:
            model.set_templates(ckpt["templates"])
    model.eval()

    if model._templates is None:
        npz = load_processed_npz()
        X, y = npz["X"], npz["y"]
        subjects = npz.get("subjects", npz.get("subject"))
        mask = _train_mask(subjects)
        templates = []
        for c in sorted(np.unique(y[mask])):
            templates.append(X[mask][y[mask] == c].mean(axis=0))
        model.set_templates(torch.from_numpy(np.stack(templates)).float())

    class ConvCABaked(nn.Module):
        """Export wrapper: explicit F.pad replaces padding='same' (avoids
        onnx2tf even-kernel-width bug), bmm replaces einsum for TFLite."""
        def __init__(self, full, templates):
            super().__init__()
            self.sig_w1 = full.sig_conv1.weight
            self.sig_b1 = full.sig_conv1.bias
            self.sig_w2 = full.sig_conv2.weight
            self.sig_b2 = full.sig_conv2.bias
            self.sig_w3 = full.sig_conv3.weight
            self.sig_b3 = full.sig_conv3.bias
            self.temp_w1 = full.temp_conv1.weight
            self.temp_b1 = full.temp_conv1.bias
            self.temp_w2 = full.temp_conv2.weight
            self.temp_b2 = full.temp_conv2.bias
            self.dense = full.dense
            self.register_buffer("templates", templates)

        def forward(self, x_btc):
            B = x_btc.shape[0]
            sig = x_btc.unsqueeze(1)
            temp = self.templates.permute(2, 1, 0).unsqueeze(0).expand(B, -1, -1, -1)

            sig = F.conv2d(F.pad(sig, [3, 4, 4, 4]), self.sig_w1, self.sig_b1)
            sig = F.conv2d(F.pad(sig, [3, 4, 0, 0]), self.sig_w2, self.sig_b2)
            sig = F.conv2d(sig, self.sig_w3, self.sig_b3)

            temp = F.conv2d(F.pad(temp, [0, 0, 4, 4]), self.temp_w1, self.temp_b1)
            temp = F.conv2d(F.pad(temp, [0, 0, 4, 4]), self.temp_w2, self.temp_b2)

            x = sig.squeeze(1)
            t = temp.squeeze(1)
            corr_xt = torch.bmm(t.transpose(1, 2), x).squeeze(-1)
            corr_xx = (x * x).sum(dim=1)
            corr_tt = (t * t).sum(dim=1)
            corr = corr_xt / (torch.sqrt(corr_tt) * torch.sqrt(corr_xx) + 1e-8)
            return self.dense(corr)

    baked = ConvCABaked(model, model._templates)
    baked.eval()
    dummy = torch.randn(1, 512, 8)
    onnx_path = out_dir / "convca_baked.onnx"
    if not _onnx_export(baked, dummy, onnx_path, ["input"], ["logits"]):
        return {"status": "onnx_failed"}

    pt_refs = []
    for v in vectors:
        with torch.no_grad():
            pt_refs.append(model(torch.from_numpy(v).unsqueeze(0)).numpy().flatten())

    return {"status": "onnx_ok", "onnx_path": onnx_path, "onnx_input_name": "input",
            "input_shape": [1, 512, 8], "pt_refs": pt_refs, "preprocess": None,
            "preprocessing_note": "Templates baked as buffer (fold-specific). Raw [B,T,C].",
            "onnx2tf_extra_args": ["-kat", "input"]}


HANDLERS = {
    "eegnet": _handle_eegnet,
    "ccnn": _handle_ccnn,
    "ssvepformer": _handle_ssvepformer,
    "ssvepnet": _handle_ssvepnet,
    "fbtcnn": _handle_fbtcnn,
    "convca": _handle_convca,
}


def process_model(name: str, vectors: list[np.ndarray], npz: dict) -> dict:
    print(f"\n{'='*60}")
    tag_label = "test" if ACTIVE_PROTOCOL == PROTOCOL_STRICT_LOSO_INNER_VAL else "val"
    print(f"  {name.upper()} (fold{ACTIVE_FOLD:02d}/{tag_label}_{ACTIVE_VAL_SUBJECT})")
    print(f"{'='*60}")

    out_dir = export_dir(
        name, ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, protocol=ACTIVE_PROTOCOL, repo=REPO,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    info = HANDLERS[name](vectors, out_dir)
    report = {"model": name, "out_dir": str(out_dir)}

    if info["status"] != "onnx_ok":
        report["status"] = info["status"]
        print(f"  RESULT: {info['status']}")
        return report

    onnx_path = info["onnx_path"]
    report["onnx_path"] = str(onnx_path)
    report["onnx_size_kb"] = round(onnx_path.stat().st_size / 1024, 1)
    report["input_shape"] = info["input_shape"]
    report["preprocessing_note"] = info["preprocessing_note"]
    print(f"  ONNX: {onnx_path.name} ({report['onnx_size_kb']} KB)")

    # ── onnx2tf ──
    tf_dir = out_dir / f"{name}_tf"
    onnx2tf_extra = info.get("onnx2tf_extra_args")
    print(f"  Converting via onnx2tf...")
    conv_result = _run_onnx2tf(onnx_path, tf_dir, timeout=600,
                                extra_args=onnx2tf_extra)
    report["onnx2tf_success"] = conv_result["success"]
    if not conv_result["success"]:
        report["onnx2tf_error"] = conv_result.get("error", "")[:300]
        print(f"  onnx2tf FAILED: {report['onnx2tf_error'][:200]}")
    else:
        print(f"  onnx2tf: SUCCESS")

    tflite_files = _find_tflite_files(tf_dir)

    # Copy float32
    if tflite_files["float32"]:
        dst = out_dir / f"{name}_float32.tflite"
        shutil.copy(tflite_files["float32"], dst)
        report["tflite_float32_path"] = str(dst)
        report["tflite_float32_kb"] = round(dst.stat().st_size / 1024, 1)
        print(f"  TFLite float32: {dst.name} ({report['tflite_float32_kb']} KB)")
    else:
        report["tflite_float32_path"] = None
        print(f"  TFLite float32: not produced")

    # ── int8: use onnx2tf's dynamic_range_quant or full_integer_quant if available ──
    if tflite_files.get("int8"):
        dst = out_dir / f"{name}_int8.tflite"
        shutil.copy(tflite_files["int8"], dst)
        report["tflite_int8_path"] = str(dst)
        report["tflite_int8_kb"] = round(dst.stat().st_size / 1024, 1)
        print(f"  TFLite int8: {dst.name} ({report['tflite_int8_kb']} KB)")
    else:
        dyn = _find_dynamic_range_tflite(tf_dir)
        if dyn:
            dst = out_dir / f"{name}_dynrange.tflite"
            shutil.copy(dyn, dst)
            report["tflite_dynrange_path"] = str(dst)
            report["tflite_dynrange_kb"] = round(dst.stat().st_size / 1024, 1)
            print(f"  TFLite dynamic-range: {dst.name} ({report['tflite_dynrange_kb']} KB)")
        report["tflite_int8_path"] = None
        print(f"  TFLite int8: not produced (use -oiqt with calibration data for full int8)")

    # ── Parity checks ──
    print(f"  Parity checks...")
    parity = []
    preprocess_fn = info.get("preprocess")

    for i, v in enumerate(vectors[:3]):
        pt_ref = info["pt_refs"][i]
        if preprocess_fn is not None:
            onnx_input = preprocess_fn(v)
        else:
            onnx_input = v[np.newaxis].astype(np.float32)

        try:
            onnx_out = _run_onnx_inference(onnx_path, onnx_input, info["onnx_input_name"])
            parity.append(_parity_report(pt_ref, onnx_out.flatten(), f"vec{i}_pt_vs_onnx"))
        except Exception as e:
            parity.append({"comparison": f"vec{i}_pt_vs_onnx", "error": str(e)[:200]})

        for quant in ["float32", "int8"]:
            tpath = report.get(f"tflite_{quant}_path")
            if tpath and Path(tpath).exists():
                try:
                    tfl_out = _run_tflite_inference(Path(tpath), onnx_input)
                    parity.append(_parity_report(pt_ref, tfl_out.flatten(),
                                                 f"vec{i}_pt_vs_tflite_{quant}"))
                except Exception as e:
                    parity.append({"comparison": f"vec{i}_pt_vs_tflite_{quant}",
                                   "error": str(e)[:200]})

    report["parity"] = parity
    for p in parity:
        tag = "PASS" if p.get("argmax_match", False) else "FAIL"
        if "error" in p:
            tag = "ERROR"
        maxd = p.get("max_abs_diff", "n/a")
        print(f"    {p['comparison']}: {tag} (max_diff={maxd})")

    # ── Operator inspection ──
    for quant in ["float32", "int8"]:
        tpath = report.get(f"tflite_{quant}_path")
        if tpath and Path(tpath).exists():
            ops = _inspect_tflite_ops(Path(tpath))
            report[f"tflite_{quant}_ops"] = ops
            print(f"  Ops ({quant}): {ops}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    report["status"] = "complete"

    (out_dir / "tflite_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    refs = {}
    for i, pt_ref in enumerate(info["pt_refs"]):
        refs[f"idx{i}"] = {"logits": [float(x) for x in pt_ref],
                           "argmax": int(np.argmax(pt_ref))}
    write_host_reference(out_dir, refs)
    meta_extra = {
        "protocol": ACTIVE_PROTOCOL,
        "outer_test_subject": ACTIVE_VAL_SUBJECT,
        "onnx_size_kb": report.get("onnx_size_kb"),
        "tflite_float32_kb": report.get("tflite_float32_kb"),
        "tflite_int8_kb": report.get("tflite_int8_kb"),
        "preprocessing": info["preprocessing_note"],
    }
    if name == "convca":
        subj = npz.get("subjects", npz.get("subject"))
        train_subs = sorted(
            set(str(s) for s in subj if str(s) != ACTIVE_VAL_SUBJECT)
        )
        meta_extra["template_subjects"] = train_subs
    write_metadata(out_dir, name, meta_extra)
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="TFLite conversion + parity + inspection (default: strict_loso_inner_val).",
    )
    add_test_subject_args(ap, default="s1")
    add_protocol_arg(ap)
    ap.add_argument("--fold", type=int, default=None,
                    help="Fold number (default: inferred from test subject).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model names; default: all DL models.")
    args = ap.parse_args()

    global ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, ACTIVE_PROTOCOL
    ACTIVE_PROTOCOL = args.protocol
    apply_protocol_side_effects(ACTIVE_PROTOCOL)
    ACTIVE_VAL_SUBJECT = resolve_test_subject(args)
    ACTIVE_FOLD = args.fold if args.fold is not None else fold_for_subject(ACTIVE_VAL_SUBJECT)
    # Patch common.CANONICAL_* so write_metadata embeds the correct fold/subject.
    from export import common as _common
    _common.CANONICAL_VAL_SUBJECT = ACTIVE_VAL_SUBJECT
    _common.CANONICAL_FOLD = ACTIVE_FOLD

    print("=" * 60)
    print("TFLite Conversion + Parity + inspection Pipeline")
    print(f"Python: {sys.version}")
    print(f"Repo:   {REPO}")
    print(f"Fold:   {ACTIVE_FOLD:02d}  held_out={ACTIVE_VAL_SUBJECT}  protocol={ACTIVE_PROTOCOL}")
    print("=" * 60)

    npz = load_processed_npz()
    vectors = get_fold_test_vectors(npz, val_subject=ACTIVE_VAL_SUBJECT)
    print(f"Loaded {len(vectors)} test vectors for {ACTIVE_VAL_SUBJECT}")

    all_models = ["eegnet", "ccnn", "ssvepformer", "convca", "fbtcnn", "ssvepnet"]
    models = args.models or [m for m in sys.argv[1:] if m in HANDLERS]
    if not models:
        models = all_models
    models = [m for m in models if m in HANDLERS]

    tag = export_dir_name("eegnet", ACTIVE_VAL_SUBJECT, ACTIVE_FOLD, protocol=ACTIVE_PROTOCOL)
    tag = tag.replace("12jfpm_eegnet_", "")
    summary_path = REPO / "export" / f"tflite_report_summary_{tag}.json"
    if ACTIVE_PROTOCOL == PROTOCOL_STRICT_LOSO_INNER_VAL:
        summary_path = (
            REPO / "export" / PROTOCOL_STRICT_LOSO_INNER_VAL / f"tflite_report_summary_{tag}.json"
        )
    if (
        not summary_path.exists()
        and ACTIVE_VAL_SUBJECT == "s1"
        and ACTIVE_FOLD == 1
        and ACTIVE_PROTOCOL == PROTOCOL_SUBJECT_AS_VAL
    ):
        # Preserve legacy path for s1.
        legacy = REPO / "export" / "tflite_report_summary.json"
        if legacy.exists():
            summary_path = legacy
    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        results = {}

    for name in models:
        try:
            results[name] = process_model(name, vectors, npz)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"model": name, "status": f"fatal: {e}"}
        summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, r in results.items():
        f32 = r.get("tflite_float32_kb", "—")
        i8 = r.get("tflite_int8_kb", "—")
        onnx_kb = r.get("onnx_size_kb", "—")
        parity_ok = all(p.get("argmax_match", False) for p in r.get("parity", [])
                        if "error" not in p)
        has_parity = len([p for p in r.get("parity", []) if "error" not in p]) > 0
        status = r.get("status", "unknown")
        ptag = "PASS" if (parity_ok and has_parity) else ("N/A" if not has_parity else "FAIL")
        print(f"  {name:15s} | ONNX {str(onnx_kb):>7s} KB | "
              f"f32 {str(f32):>7s} KB | i8 {str(i8):>7s} KB | "
              f"parity={ptag:4s} | {status}")

    print(f"\nFull report: {summary_path}")


if __name__ == "__main__":
    main()
