"""Host evaluation for neural PyTorch checkpoints and TFLite artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.dataset_12jfpm import split_by_subjects
from src.neural import MODEL_REGISTRY
from src.neural.ccnn import _complex_spectrum_features
from src.neural.fbtcnn import FBtCNN
from src.neural.ssvepformer import _ssvepformer_spectrum
from src.training.engine import evaluate_outer_test

ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = [f"s{i}" for i in range(1, 11)]

CONFIG_MAP = {
    "ccnn": ROOT / "configs/training/ccnn_12jfpm_strict_loso.yaml",
    "convca": ROOT / "configs/training/convca_12jfpm_strict_loso.yaml",
    "eegnet": ROOT / "configs/training/eegnet_12jfpm_strict_loso.yaml",
    "fbtcnn": ROOT / "configs/training/fbtcnn_12jfpm_strict_loso.yaml",
    "ssvepformer": ROOT / "configs/training/ssvepformer_12jfpm_strict_loso.yaml",
    "ssvepnet": ROOT / "configs/training/ssvepnet_12jfpm_strict_loso.yaml",
}

# Models with retained export handlers. Runtime may still fail if the graph
# cannot execute on the installed host interpreter.
TFLITE_MODELS = {
    "ccnn",
    "convca",
    "eegnet",
    "fbtcnn",
    "ssvepformer",
    "ssvepnet",
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return doc


def resolve_config(model: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    path = CONFIG_MAP.get(model)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"No training config for model {model!r}")
    return path


def load_processed(npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz, allow_pickle=True)
    subjects = d["subjects"] if "subjects" in d.files else d["subject"]
    return d["X"].astype(np.float32), d["y"].astype(np.int64), np.asarray(subjects)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_held_out_subject(*, fold: int | None, manifest: Path | None) -> str:
    if manifest is not None:
        doc = load_manifest(manifest)
        return str(doc["held_out_subject"])
    if fold is None:
        raise ValueError("Provide --fold or --manifest to select the held-out subject")
    if fold < 1 or fold > 10:
        raise ValueError("--fold must be in 1..10")
    return f"s{fold}"


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Supported: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    mcfg = cfg["model"]
    common = dict(
        seq_len=int(mcfg["seq_len"]),
        embed_dim=int(mcfg["embed_dim"]),
        num_classes=int(mcfg["num_classes"]),
    )
    if model_name == "eegnet":
        return MODEL_REGISTRY[model_name](
            **common,
            F1=int(mcfg.get("F1", 8)),
            D=int(mcfg.get("D", 2)),
            F2=int(mcfg.get("F2", 16)),
            dropout=float(mcfg.get("dropout", 0.25)),
            kernel_length=int(mcfg.get("kernel_length", 64)),
            pool1_stride=int(mcfg.get("pool1_stride", 4)),
            pool2_stride=int(mcfg.get("pool2_stride", 8)),
        )
    if model_name == "ccnn":
        return MODEL_REGISTRY[model_name](
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            dropout=float(mcfg.get("dropout", 0.25)),
        )
    if model_name == "ssvepformer":
        return MODEL_REGISTRY[model_name](
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            depth=int(mcfg.get("depth", 1)),
            attention_kernel_length=int(mcfg.get("attention_kernel_length", 31)),
            dropout=float(mcfg.get("dropout", 0.5)),
        )
    if model_name == "ssvepnet":
        return MODEL_REGISTRY[model_name](
            **common,
            dropout=float(mcfg.get("dropout", 0.5)),
        )
    if model_name == "fbtcnn":
        return MODEL_REGISTRY[model_name](
            **common,
            sampling_rate_hz=float(mcfg.get("sampling_rate_hz", 256.0)),
            dropout=float(mcfg.get("dropout", 0.4)),
        )
    if model_name == "convca":
        return MODEL_REGISTRY[model_name](
            **common,
            dropout_sig=float(mcfg.get("dropout_sig", 0.75)),
            dropout_temp=float(mcfg.get("dropout_temp", 0.15)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_class_templates(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    outer_test_subject: str,
    num_classes: int = 12,
) -> torch.Tensor:
    train_subs = [s for s in SUBJECTS if s != outer_test_subject]
    train_mask, _ = split_by_subjects(subjects, train_subs, [outer_test_subject])
    X_train = X[train_mask]
    y_train = y[train_mask]
    templates = [X_train[y_train == c].mean(axis=0) for c in range(num_classes)]
    return torch.from_numpy(np.stack(templates).astype(np.float32))


def prepare_pytorch_model(
    model_name: str,
    cfg: dict,
    checkpoint: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    held_out: str,
) -> torch.nn.Module:
    model = build_model(model_name, cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if model_name == "convca":
        if "templates" in ckpt:
            model.set_templates(ckpt["templates"])
        else:
            model.set_templates(
                compute_class_templates(
                    X, y, subjects, held_out, int(cfg["model"]["num_classes"])
                )
            )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_pytorch(
    model_name: str,
    *,
    checkpoint: Path,
    npz: Path,
    fold: int | None = None,
    manifest: Path | None = None,
    config: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    cfg_path = resolve_config(model_name, config)
    cfg = load_yaml(cfg_path)
    held_out = resolve_held_out_subject(fold=fold, manifest=manifest)
    X, y, subjects = load_processed(npz)

    model = prepare_pytorch_model(
        model_name, cfg, checkpoint, X=X, y=y, subjects=subjects, held_out=held_out
    )
    result = evaluate_outer_test(
        model,
        checkpoint,
        X,
        y,
        subjects,
        held_out,
        SUBJECTS,
        batch_size=batch_size,
    )
    summary = {
        "runtime": "pytorch",
        "model": model_name,
        "held_out_subject": held_out,
        "checkpoint": str(checkpoint),
        "config": str(cfg_path),
        "npz": str(npz),
        **result,
    }
    if output_dir is not None:
        _write_summary(output_dir, summary)
    return summary


def preprocess_trial_for_tflite(model_name: str, trial_tc: np.ndarray) -> np.ndarray:
    """Convert one raw [T, C] trial to the export-time TFLite entry tensor."""
    x = trial_tc.astype(np.float32)
    if x.ndim != 2 or x.shape != (512, 8):
        raise ValueError(f"Expected trial shape [512, 8], got {x.shape}")

    if model_name in {"eegnet", "ssvepnet", "convca"}:
        return x[np.newaxis, ...]

    if model_name == "ccnn":
        t = torch.from_numpy(x).unsqueeze(0)
        return _complex_spectrum_features(t, 256.0).numpy()

    if model_name == "ssvepformer":
        t = torch.from_numpy(x).unsqueeze(0)
        return _ssvepformer_spectrum(t, 256.0).numpy()

    if model_name == "fbtcnn":
        model = FBtCNN(
            seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.4
        )
        xb = torch.from_numpy(x).unsqueeze(0).transpose(1, 2).unsqueeze(1)
        bands = model._filter_bank(xb)
        stacked = torch.stack(bands, dim=1)
        if stacked.dim() == 5:
            stacked = stacked.squeeze(2)
        return stacked.numpy()

    raise ValueError(f"No TFLite preprocess adapter for {model_name!r}")


def run_tflite_logits(tflite_path: Path, input_data: np.ndarray) -> np.ndarray:
    """Run one TFLite invoke with export-compatible shape adaptation."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    expected_shape = tuple(inp_detail["shape"])
    data = input_data.astype(np.float32)

    if tuple(data.shape) != expected_shape:
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
                raise ValueError(
                    f"Shape mismatch: data {data.shape} vs expected {expected_shape}"
                )

    inp_dtype = inp_detail["dtype"]
    if inp_dtype in (np.int8, np.uint8):
        scale, zp = inp_detail["quantization"]
        data = (
            (data / scale + zp)
            .clip(np.iinfo(inp_dtype).min, np.iinfo(inp_dtype).max)
            .astype(inp_dtype)
        )
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


def evaluate_tflite(
    model_name: str,
    *,
    tflite: Path,
    npz: Path,
    fold: int | None = None,
    manifest: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if model_name not in TFLITE_MODELS:
        raise ValueError(
            f"TFLite host evaluation is not supported for {model_name!r}. "
            f"Supported: {', '.join(sorted(TFLITE_MODELS))}"
        )
    if not tflite.is_file():
        raise FileNotFoundError(f"TFLite artifact not found: {tflite}")

    held_out = resolve_held_out_subject(fold=fold, manifest=manifest)
    X, y, subjects = load_processed(npz)
    train_subs = [s for s in SUBJECTS if s != held_out]
    _, test_m = split_by_subjects(subjects, train_subs, [held_out])
    indices = np.where(test_m)[0]

    preds: list[int] = []
    labels: list[int] = []
    logits_rows: list[list[float]] = []
    errors: list[str] = []

    for i in indices:
        trial = X[i]
        label = int(y[i])
        try:
            inp = preprocess_trial_for_tflite(model_name, trial)
            logits = run_tflite_logits(tflite, inp).reshape(-1)
            pred = int(np.argmax(logits))
        except Exception as exc:  # noqa: BLE001 - surface host interpreter failures
            errors.append(f"index={int(i)}: {exc}")
            continue
        preds.append(pred)
        labels.append(label)
        logits_rows.append([float(v) for v in logits.tolist()])

    preds_a = np.asarray(preds, dtype=np.int64)
    labels_a = np.asarray(labels, dtype=np.int64)
    n = int(labels_a.size)
    correct = int((preds_a == labels_a).sum()) if n else 0
    summary: dict[str, Any] = {
        "runtime": "tflite",
        "model": model_name,
        "held_out_subject": held_out,
        "tflite": str(tflite),
        "npz": str(npz),
        "n_trials": n,
        "correct": correct,
        "accuracy": round(correct / n, 6) if n else 0.0,
        "n_errors": len(errors),
    }
    if errors:
        summary["errors_head"] = errors[:5]
        if n == 0:
            raise RuntimeError(
                "TFLite host evaluation produced no successful trials. "
                f"First error: {errors[0]}"
            )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "trial_outputs.npz",
            indices=indices.astype(np.int64),
            y_true=labels_a,
            y_pred=preds_a,
            logits=np.asarray(logits_rows, dtype=np.float32),
        )
        _write_summary(output_dir, summary)
    return summary


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
