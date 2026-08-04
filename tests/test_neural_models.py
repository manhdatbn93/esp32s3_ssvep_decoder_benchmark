"""Canonical neural package, registry, and input-contract checks."""

from __future__ import annotations

from pathlib import Path

import torch

from src.neural import MODEL_REGISTRY
from src.neural.ccnn import CCNN, _complex_spectrum_features
from src.neural.convca import ConvCA
from src.neural.eegnet import EEGNet
from src.neural.fbtcnn import FBtCNN
from src.neural.ssvepformer import SSVEPformer, _ssvepformer_spectrum
from src.neural.ssvepnet import SSVEPNet

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "ccnn": CCNN,
    "convca": ConvCA,
    "eegnet": EEGNet,
    "fbtcnn": FBtCNN,
    "ssvepformer": SSVEPformer,
    "ssvepnet": SSVEPNet,
}

# Shared processed-trial batch used by training DataLoader / PyTorch forward().
RAW_BTC = (2, 512, 8)

CTOR_KWARGS = {
    "ccnn": dict(
        seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.25
    ),
    "convca": dict(
        seq_len=512, embed_dim=8, num_classes=12, dropout_sig=0.75, dropout_temp=0.15
    ),
    "eegnet": dict(seq_len=512, embed_dim=8, num_classes=12),
    "fbtcnn": dict(
        seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0, dropout=0.4
    ),
    "ssvepformer": dict(
        seq_len=512, embed_dim=8, num_classes=12, sampling_rate_hz=256.0
    ),
    "ssvepnet": dict(seq_len=512, embed_dim=8, num_classes=12, dropout=0.5),
}


def test_registry_matches_canonical_classes():
    assert set(MODEL_REGISTRY) == set(EXPECTED)
    for name, cls in EXPECTED.items():
        assert MODEL_REGISTRY[name] is cls


def test_training_configs_map_to_registry():
    training = ROOT / "configs" / "training"
    names = sorted(p.stem.split("_12jfpm")[0] for p in training.glob("*_12jfpm_strict_loso.yaml"))
    assert names == sorted(MODEL_REGISTRY)
    for name in names:
        assert name in MODEL_REGISTRY


def test_pytorch_forward_accepts_raw_btc():
    """All six PyTorch models take processed trials [B, T, C] = [B, 512, 8]."""
    for name, cls in EXPECTED.items():
        torch.manual_seed(0)
        model = cls(**CTOR_KWARGS[name])
        model.eval()
        if name == "convca":
            model.set_templates(torch.randn(12, 512, 8))
        x = torch.randn(*RAW_BTC)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (RAW_BTC[0], 12), name


def test_ccnn_internal_spectrum_adapter():
    x = torch.randn(*RAW_BTC)
    spec = _complex_spectrum_features(x, 256.0)
    assert tuple(spec.shape) == (2, 1, 8, 220)
    model = CCNN(**CTOR_KWARGS["ccnn"]).eval()
    with torch.no_grad():
        y_raw = model(x)
        y_post = model.classifier(model.layer2(model.layer1(spec)))
    assert tuple(y_raw.shape) == (2, 12)
    assert torch.allclose(y_raw, y_post, atol=1e-5)


def test_ssvepformer_internal_spectrum_adapter():
    x = torch.randn(*RAW_BTC)
    spec = _ssvepformer_spectrum(x, 256.0)
    assert tuple(spec.shape) == (2, 8, 560)
    model = SSVEPformer(**CTOR_KWARGS["ssvepformer"]).eval()
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 12)


def test_fbtcnn_filter_bank_adapter():
    x = torch.randn(*RAW_BTC)
    model = FBtCNN(**CTOR_KWARGS["fbtcnn"]).eval()
    x_b1ct = x.transpose(1, 2).unsqueeze(1)
    bands = model._filter_bank(x_b1ct)
    stacked = torch.stack(bands, dim=1)
    if stacked.dim() == 5:
        stacked = stacked.squeeze(2)
    assert tuple(stacked.shape) == (2, 4, 8, 512)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 12)


def test_convca_requires_templates():
    model = ConvCA(**CTOR_KWARGS["convca"]).eval()
    x = torch.randn(*RAW_BTC)
    try:
        model(x)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    templates = torch.randn(12, 512, 8)
    with torch.no_grad():
        y = model.forward_with_templates(x, templates)
    assert tuple(y.shape) == (2, 12)


def test_export_onnx_input_shapes_match_handlers():
    """Exported ONNX/TFLite entry tensors differ from raw [B,512,8] for some models."""
    expected_export = {
        "eegnet": (1, 512, 8),
        "ssvepnet": (1, 512, 8),
        "convca": (1, 512, 8),
        "ccnn": (1, 1, 8, 220),
        "fbtcnn": (1, 4, 8, 512),
        "ssvepformer": (1, 8, 560),
    }
    raw = torch.randn(1, 512, 8)
    assert tuple(_complex_spectrum_features(raw, 256.0).shape) == expected_export["ccnn"]
    assert tuple(_ssvepformer_spectrum(raw, 256.0).shape) == expected_export["ssvepformer"]
    m = FBtCNN(**CTOR_KWARGS["fbtcnn"]).eval()
    xb = raw.transpose(1, 2).unsqueeze(1)
    bands = m._filter_bank(xb)
    stacked = torch.stack(bands, dim=1)
    if stacked.dim() == 5:
        stacked = stacked.squeeze(2)
    assert tuple(stacked.shape) == expected_export["fbtcnn"]
    for name in ("eegnet", "ssvepnet", "convca"):
        assert expected_export[name] == (1, 512, 8)


def test_no_src_models_package():
    assert not (ROOT / "src" / "models").exists()


def test_no_active_src_models_imports():
    banned = []
    needle = "src" + ".models"
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "test_neural_models.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle in text:
            banned.append(path.relative_to(ROOT).as_posix())
    assert banned == []


def test_no_wildcard_model_imports():
    hits = []
    for path in (ROOT / "src" / "neural").glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "import *" in text:
            hits.append(path.name)
    assert hits == []


def test_retained_neural_files_are_referenced():
    neural_files = {
        p.name
        for p in (ROOT / "src" / "neural").glob("*.py")
        if p.name != "__init__.py"
    }
    assert neural_files == {
        "ccnn.py",
        "convca.py",
        "eegnet.py",
        "fbtcnn.py",
        "ssvepformer.py",
        "ssvepnet.py",
    }
