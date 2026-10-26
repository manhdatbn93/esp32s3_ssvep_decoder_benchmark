from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EEGNET_INT8 = ROOT / "artifacts" / "reference" / "tflite" / "eegnet_int8" / "fold01" / "eegnet_int8.tflite"


def _run(interp, x: np.ndarray) -> np.ndarray:
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    return interp.get_tensor(out["index"]).copy()


def test_host_reference_resolver_is_tensorflow_default():
    tf = pytest.importorskip("tensorflow")
    from src.replay.host_interpreter import HOST_REFERENCE_OP_RESOLVER, make_interpreter

    assert HOST_REFERENCE_OP_RESOLVER == "AUTO"
    x = np.random.default_rng(0).standard_normal((1, 8, 512)).astype(np.float32)
    default = _run(tf.lite.Interpreter(model_path=str(EEGNET_INT8)), x)
    explicit = _run(make_interpreter(EEGNET_INT8), x)
    np.testing.assert_array_equal(default, explicit)


def test_no_default_delegates_resolver_selects_different_kernels():
    pytest.importorskip("tensorflow")
    from src.replay.host_interpreter import NO_DEFAULT_DELEGATES_OP_RESOLVER, make_interpreter

    rng = np.random.default_rng(42)
    xs = [rng.standard_normal((1, 8, 512)).astype(np.float32) for _ in range(20)]
    auto = [_run(make_interpreter(EEGNET_INT8), x) for x in xs]
    builtin = [_run(make_interpreter(EEGNET_INT8, NO_DEFAULT_DELEGATES_OP_RESOLVER), x) for x in xs]
    assert any(not np.array_equal(a, b) for a, b in zip(auto, builtin))
