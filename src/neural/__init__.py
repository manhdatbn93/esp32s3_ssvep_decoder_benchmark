"""Canonical neural SSVEP decoder implementations for 12JFPM."""

from __future__ import annotations

from .ccnn import CCNN
from .convca import ConvCA
from .eegnet import EEGNet
from .fbtcnn import FBtCNN
from .ssvepformer import SSVEPformer
from .ssvepnet import SSVEPNet

MODEL_REGISTRY = {
    "ccnn": CCNN,
    "convca": ConvCA,
    "eegnet": EEGNet,
    "fbtcnn": FBtCNN,
    "ssvepformer": SSVEPformer,
    "ssvepnet": SSVEPNet,
}

__all__ = [
    "CCNN",
    "ConvCA",
    "EEGNet",
    "FBtCNN",
    "SSVEPformer",
    "SSVEPNet",
    "MODEL_REGISTRY",
]
