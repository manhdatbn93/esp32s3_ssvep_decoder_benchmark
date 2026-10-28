"""Specification tests for the frontend ablation. They do not replace the dataset run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as scipy_signal

from analysis.frontend_ablation.dataset import calibration_global_indices
from analysis.frontend_ablation.frontends import (
    CCNN_MODEL_INPUT_SHAPE,
    FBTCNN_MODEL_INPUT_SHAPE,
    ccnn_bin_specification,
    ccnn_fft_model_input,
    ccnn_goertzel_model_input,
    fbtcnn_sos_model_input,
    fbtcnn_transfer_model_input,
    prepare_interpreter_input,
)
from analysis.frontend_ablation.metrics import (
    classify_flip,
    summarize_numeric,
    top1_and_margin,
)
from src.replay.host_reference import _FBTCNN_BANDS, _fbtcnn_preprocess

ROOT = Path(__file__).resolve().parents[1]


def _trial(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((512, 8)).astype(np.float32)


def test_ccnn_selected_bin_specification_matches_both_frontends():
    spec = ccnn_bin_specification()
    assert spec["nfft"] == 874 == spec["goertzel_nfft"]
    assert spec["bin_lo"] == 10 == spec["goertzel_bin_lo"]
    assert spec["bin_hi"] == 120 == spec["goertzel_bin_hi"]
    assert spec["n_bins"] == 110
    assert spec["n_features"] == 220
    assert spec["norm_factor"] == 256.0 == spec["goertzel_norm_factor"]


def test_ccnn_real_imag_order_and_shape_agree_on_a_bin_cosine():
    nfft = 874
    bin_index = 10
    time = np.arange(512, dtype=np.float64)
    wave = np.cos(2.0 * np.pi * bin_index * time / nfft).astype(np.float32)
    trial = np.zeros((512, 8), dtype=np.float32)
    trial[:, 0] = wave
    fft_input = ccnn_fft_model_input(trial)
    goertzel_input = ccnn_goertzel_model_input(trial)
    assert fft_input.shape == goertzel_input.shape == CCNN_MODEL_INPUT_SHAPE
    fft_features = fft_input[0, 0, :, 0]
    goertzel_features = goertzel_input[0, 0, :, 0]
    assert abs(fft_features[0]) > abs(fft_features[110])
    assert abs(goertzel_features[0]) > abs(goertzel_features[110])
    assert fft_features[0] > 0 and goertzel_features[0] > 0


def test_frontends_do_not_normalize_and_are_deterministic():
    trial = _trial(1)
    for function in (
        ccnn_fft_model_input,
        ccnn_goertzel_model_input,
        fbtcnn_transfer_model_input,
        fbtcnn_sos_model_input,
    ):
        once = function(trial)
        twice = function(trial)
        scaled = function(trial * np.float32(2))
        np.testing.assert_array_equal(once, twice)
        np.testing.assert_allclose(scaled, once * np.float32(2), rtol=1e-5, atol=1e-5)


def test_fbtcnn_band_order_shape_and_scipy_default_pad_length():
    assert list(_FBTCNN_BANDS) == [(3, 14), (9, 26), (14, 38), (19, 50)]
    trial = _trial(2)
    transfer = fbtcnn_transfer_model_input(trial)
    sos = fbtcnn_sos_model_input(trial)
    assert transfer.shape == sos.shape == FBTCNN_MODEL_INPUT_SHAPE
    sos_coeff = scipy_signal.butter(6, [2 * 3 / 256, 2 * 14 / 256], btype="bandpass", output="sos")
    assert sos_coeff.shape[0] == 6
    series = trial[:, 0].astype(np.float64)
    default = scipy_signal.sosfiltfilt(sos_coeff, series)
    padded = scipy_signal.sosfiltfilt(sos_coeff, series, padlen=39)
    np.testing.assert_array_equal(default, padded)
    header = (ROOT / "artifacts/reference/preprocessing/fbtcnn/fold01/fbtcnn_filter_generated.h").read_text(encoding="utf-8")
    assert "#define FBTCNN_EDGE     39" in header
    assert "padlen" not in Path(_fbtcnn_preprocess.__code__.co_filename).read_text(encoding="utf-8").split("def _fbtcnn_preprocess", 1)[1].split("\ndef ", 1)[0]


def test_float32_and_int8_input_preparation():
    feed = np.zeros(CCNN_MODEL_INPUT_SHAPE, dtype=np.float32)
    float_detail = {"shape": CCNN_MODEL_INPUT_SHAPE, "dtype": np.float32, "quantization": (0.0, 0)}
    prepared = prepare_interpreter_input(feed, float_detail)
    assert prepared.dtype == np.float32
    int8_detail = {"shape": CCNN_MODEL_INPUT_SHAPE, "dtype": np.int8, "quantization": (0.25, -1)}
    quantized = prepare_interpreter_input(np.full(CCNN_MODEL_INPUT_SHAPE, 0.3, dtype=np.float32), int8_detail)
    assert quantized.dtype == np.int8
    assert quantized.flat[0] == np.int8(0)
    with pytest.raises(ValueError, match="contract"):
        prepare_interpreter_input(feed.reshape(1, 8, 1, 220), float_detail)


def test_fold_artifact_subject_identity():
    from src.artifacts.reference import resolve_tflite

    for model, precision in (("ccnn", "f32"), ("ccnn", "int8"), ("fbtcnn", "f32"), ("fbtcnn", "int8")):
        artifact = resolve_tflite(model, precision, 3, verify_checksum=True)
        assert artifact["held_out_subject"] == "s3"
        assert artifact["fold"] == 3


def test_calibration_selection_excludes_held_out_and_keeps_192():
    subjects = np.repeat([f"s{i}" for i in range(1, 11)], 180)
    y = np.concatenate([np.repeat(np.arange(12), 15) for _ in range(10)])
    chosen = calibration_global_indices(y, subjects, "s1")
    assert chosen.size == 192
    assert "s1" not in set(subjects[chosen].tolist())
    # Each class has 15 trials per subject, so the first 16 are 15 from s2 and one from s3.
    assert set(subjects[chosen].tolist()) == {"s2", "s3"}
    assert sum(int(y[i]) == 0 for i in chosen) == 16


def test_top1_tie_uses_lowest_index_and_flip_labels():
    assert top1_and_margin([1.0, 1.0, 0.0] + [0.0] * 9) == (0, 0.0)
    assert classify_flip(1, 1, 1) == "agree"
    assert classify_flip(1, 2, 1) == "harmful"
    assert classify_flip(2, 1, 1) == "beneficial"
    assert classify_flip(2, 3, 1) == "both_wrong"


def test_summary_percentiles_and_ordering():
    summary = summarize_numeric([1, 2, 3, 4, 100])
    assert summary["mean"] == 22
    assert summary["median"] == 3
    assert summary["maximum"] == 100
    assert summary["p95"] > 4
    rows = [("s2", 1, 0), ("s1", 2, 0), ("s1", 1, 1), ("s1", 1, 0)]
    assert sorted(rows) == [("s1", 1, 0), ("s1", 1, 1), ("s1", 2, 0), ("s2", 1, 0)]


def test_calling_frontends_does_not_change_replay_source_bytes():
    path = ROOT / "src/replay/host_reference.py"
    before = path.read_bytes()
    ccnn_goertzel_model_input(_trial(4))
    fbtcnn_sos_model_input(_trial(4))
    assert path.read_bytes() == before
