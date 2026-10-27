"""Adapters that call the existing training and replay frontends.

Path A is the checkpoint / PTQ frontend. Path B is the recorded host-replay
frontend. Neither function normalizes. The caller must pass the same
already z-scored ``[512, 8]`` trial to both.
"""

from __future__ import annotations

import numpy as np
import torch

from src.neural.ccnn import (
    _FFT_FREQ_HI_HZ,
    _FFT_FREQ_LO_HZ,
    _FFT_RESOLUTION_HZ,
    _complex_spectrum_features,
)
from src.neural.fbtcnn import FBtCNN
from src.replay.host_reference import (
    _CCNN_BIN_HI,
    _CCNN_BIN_LO,
    _CCNN_N_BINS,
    _CCNN_N_FEATURES,
    _CCNN_NFFT,
    _CCNN_NORM_FACTOR,
    _FBTCNN_BANDS,
    _FBTCNN_FS,
    _ccnn_preprocess,
    _fbtcnn_preprocess,
)

SAMPLING_RATE_HZ = 256.0
WINDOW_SAMPLES = 512
N_CHANNELS = 8
CCNN_MODEL_INPUT_SHAPE = (1, N_CHANNELS, _CCNN_N_FEATURES, 1)
FBTCNN_MODEL_INPUT_SHAPE = (1, N_CHANNELS, WINDOW_SAMPLES, len(_FBTCNN_BANDS))


def ccnn_bin_specification() -> dict[str, int | float]:
    """Bin contract shared by the FFT helper and the Goertzel helper."""
    nfft = round(SAMPLING_RATE_HZ / _FFT_RESOLUTION_HZ)
    idx_lo = round(_FFT_FREQ_LO_HZ / _FFT_RESOLUTION_HZ)
    idx_hi = round(_FFT_FREQ_HI_HZ / _FFT_RESOLUTION_HZ) + 1
    return {
        "nfft": nfft,
        "bin_lo": idx_lo,
        "bin_hi": idx_hi,
        "n_bins": idx_hi - idx_lo,
        "n_features": 2 * (idx_hi - idx_lo),
        "norm_factor": WINDOW_SAMPLES / 2.0,
        "goertzel_nfft": _CCNN_NFFT,
        "goertzel_bin_lo": _CCNN_BIN_LO,
        "goertzel_bin_hi": _CCNN_BIN_HI,
        "goertzel_n_bins": _CCNN_N_BINS,
        "goertzel_n_features": _CCNN_N_FEATURES,
        "goertzel_norm_factor": _CCNN_NORM_FACTOR,
    }


def _as_trial(trial_tc: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial_tc)
    if trial.shape != (WINDOW_SAMPLES, N_CHANNELS):
        raise ValueError(f"Expected trial [512, 8], got {trial.shape}")
    if not np.isfinite(trial).all():
        raise ValueError("Trial contains a non-finite sample")
    return trial


def ccnn_fft_model_input(trial_tc: np.ndarray) -> np.ndarray:
    """Training-compatible FFT features as NHWC ``[1, 8, 220, 1]`` float32.

    Calls ``_complex_spectrum_features`` (``torch.fft.fft``). The transpose
    matches ``export.quantize_int8._calib_ccnn``.
    """
    trial = _as_trial(trial_tc).astype(np.float32, copy=False)
    spectrum = _complex_spectrum_features(torch.from_numpy(trial).unsqueeze(0), SAMPLING_RATE_HZ)
    model_input = np.transpose(spectrum.detach().cpu().numpy(), (0, 2, 3, 1))
    if model_input.shape != CCNN_MODEL_INPUT_SHAPE:
        raise ValueError(f"FFT model input shape {model_input.shape}")
    return np.ascontiguousarray(model_input, dtype=np.float32)


def ccnn_goertzel_model_input(trial_tc: np.ndarray) -> np.ndarray:
    """Recorded host-replay Goertzel features, ``[1, 8, 220, 1]`` float32."""
    model_input = _ccnn_preprocess(_as_trial(trial_tc))
    if model_input.shape != CCNN_MODEL_INPUT_SHAPE or model_input.dtype != np.float32:
        raise ValueError(f"Goertzel model input {model_input.shape} {model_input.dtype}")
    return model_input


_FBTCNN_MODEL: FBtCNN | None = None


def _fbtcnn_training_model() -> FBtCNN:
    """Reuse one module. ``_filter_bank`` does not read its parameters."""
    global _FBTCNN_MODEL
    if _FBTCNN_MODEL is None:
        _FBTCNN_MODEL = FBtCNN(
            seq_len=WINDOW_SAMPLES,
            embed_dim=N_CHANNELS,
            num_classes=12,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            dropout=0.4,
        )
    return _FBTCNN_MODEL


def fbtcnn_transfer_model_input(trial_tc: np.ndarray) -> np.ndarray:
    """Training-compatible ``butter`` + ``filtfilt`` features, ``[1, 8, 512, 4]``.

    Calls ``FBtCNN._filter_bank``. The stack and transpose match
    ``export.quantize_int8._calib_fbtcnn``.
    """
    trial = _as_trial(trial_tc).astype(np.float32, copy=False)
    model = _fbtcnn_training_model()
    x = torch.from_numpy(trial).unsqueeze(0).transpose(1, 2).unsqueeze(1)
    bands = model._filter_bank(x)
    stacked = torch.stack(bands, dim=1)
    if stacked.dim() == 5:
        stacked = stacked.squeeze(2)
    model_input = np.transpose(stacked.detach().cpu().numpy(), (0, 2, 3, 1))
    if model_input.shape != FBTCNN_MODEL_INPUT_SHAPE:
        raise ValueError(f"Transfer-function model input shape {model_input.shape}")
    return np.ascontiguousarray(model_input, dtype=np.float32)


def fbtcnn_sos_model_input(trial_tc: np.ndarray) -> np.ndarray:
    """Recorded host-replay SOS ``sosfiltfilt`` features, ``[1, 8, 512, 4]``."""
    model_input = _fbtcnn_preprocess(_as_trial(trial_tc))
    if model_input.shape != FBTCNN_MODEL_INPUT_SHAPE or model_input.dtype != np.float32:
        raise ValueError(f"SOS model input {model_input.shape} {model_input.dtype}")
    return model_input


def prepare_interpreter_input(feed: np.ndarray, input_detail: dict) -> np.ndarray:
    """Match ``HostReference._run_tflite`` dtype handling without reshaping.

    A shape mismatch is a contract violation. The float32 graphs are passed
    through. Integer input tensors are quantized with the tensor scale and
    zero-point, then clipped to the integer type.
    """
    expected = tuple(int(v) for v in input_detail["shape"])
    if tuple(feed.shape) != expected:
        raise ValueError(f"Model-input contract violation: {feed.shape} vs {expected}")
    dtype = np.dtype(input_detail["dtype"])
    if dtype == np.float32:
        return np.ascontiguousarray(feed, dtype=np.float32)
    if dtype in (np.int8, np.uint8):
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            raise ValueError("Input quantization scale is zero")
        info = np.iinfo(dtype)
        quantized = np.clip(np.round(feed / scale + zero_point), info.min, info.max)
        return np.ascontiguousarray(quantized, dtype=dtype)
    raise ValueError(f"Unsupported TFLite input dtype {dtype}")


def dequantize_output(raw: np.ndarray, output_detail: dict) -> np.ndarray:
    """Return a length-12 float32 vector using the host-reference rule."""
    dtype = np.dtype(output_detail["dtype"])
    values = np.asarray(raw)
    if dtype in (np.int8, np.uint8):
        scale, zero_point = output_detail["quantization"]
        values = (values.astype(np.float32) - zero_point) * scale
    vector = values.reshape(-1).astype(np.float32)
    if vector.shape != (12,):
        raise ValueError(f"Expected 12 outputs, got {vector.shape}")
    return vector


FRONTEND_BY_FAMILY = {
    "ccnn": (ccnn_fft_model_input, ccnn_goertzel_model_input),
    "fbtcnn": (fbtcnn_transfer_model_input, fbtcnn_sos_model_input),
}
