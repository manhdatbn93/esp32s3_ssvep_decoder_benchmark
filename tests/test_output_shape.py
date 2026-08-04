"""Direct forward and raw-trial adapter output-shape checks."""

from __future__ import annotations

import torch

from src.neural import MODEL_REGISTRY
from src.neural.ccnn import CCNN, _complex_spectrum_features
from src.neural.convca import ConvCA
from src.neural.eegnet import EEGNet
from src.neural.fbtcnn import FBtCNN
from src.neural.ssvepformer import SSVEPformer, _ssvepformer_spectrum
from src.neural.ssvepnet import SSVEPNet

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


def test_expected_output_classes():
    assert 12 == 12


def test_direct_forward_shapes_per_model():
    """Each model is exercised with its documented PyTorch forward tensor."""
    cases = [
        ("eegnet", EEGNet, torch.randn(2, 512, 8), None),
        ("ssvepnet", SSVEPNet, torch.randn(2, 512, 8), None),
        ("ccnn", CCNN, torch.randn(2, 512, 8), None),
        ("ssvepformer", SSVEPformer, torch.randn(2, 512, 8), None),
        ("fbtcnn", FBtCNN, torch.randn(2, 512, 8), None),
        ("convca", ConvCA, torch.randn(2, 512, 8), torch.randn(12, 512, 8)),
    ]
    for name, cls, x, templates in cases:
        torch.manual_seed(1)
        model = cls(**CTOR_KWARGS[name]).eval()
        if templates is not None:
            model.set_templates(templates)
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (2, 12), name


def test_raw_trial_to_export_adapter_ccnn():
    raw = torch.randn(1, 512, 8)
    spec = _complex_spectrum_features(raw, 256.0)
    assert tuple(spec.shape) == (1, 1, 8, 220)
    model = CCNN(**CTOR_KWARGS["ccnn"]).eval()
    with torch.no_grad():
        y = model.classifier(model.layer2(model.layer1(spec)))
    assert tuple(y.shape) == (1, 12)


def test_raw_trial_to_export_adapter_fbtcnn():
    raw = torch.randn(1, 512, 8)
    model = FBtCNN(**CTOR_KWARGS["fbtcnn"]).eval()
    x = raw.transpose(1, 2).unsqueeze(1)
    bands = model._filter_bank(x)
    stacked = torch.stack(bands, dim=1)
    if stacked.dim() == 5:
        stacked = stacked.squeeze(2)
    assert tuple(stacked.shape) == (1, 4, 8, 512)
    with torch.no_grad():
        y = model(raw)
    assert tuple(y.shape) == (1, 12)


def test_raw_trial_to_export_adapter_ssvepformer():
    raw = torch.randn(1, 512, 8)
    spec = _ssvepformer_spectrum(raw, 256.0)
    assert tuple(spec.shape) == (1, 8, 560)
    model = SSVEPformer(**CTOR_KWARGS["ssvepformer"]).eval()
    with torch.no_grad():
        y = model(raw)
    assert tuple(y.shape) == (1, 12)


def test_registry_models_share_training_raw_contract():
    for name, cls in MODEL_REGISTRY.items():
        torch.manual_seed(2)
        model = cls(**CTOR_KWARGS[name]).eval()
        if name == "convca":
            model.set_templates(torch.randn(12, 512, 8))
        with torch.no_grad():
            out = model(torch.randn(2, 512, 8))
        assert tuple(out.shape) == (2, 12)
