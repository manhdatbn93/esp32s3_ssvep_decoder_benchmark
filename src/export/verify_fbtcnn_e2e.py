#!/usr/bin/env python3
"""End-to-end FBtCNN verification: SOS preprocessing → TFLite → scores.

Confirms the SOS-based preprocessing (matching C impl) produces the same
argmax as the host reference when fed through the TFLite model.
"""
import numpy as np
from scipy import signal
from pathlib import Path
import json

try:
    import tensorflow as tf
except ImportError:
    tf = None

BANDS = [(3, 14), (9, 26), (14, 38), (19, 50)]
FS = 256.0
ORDER = 6
OUT_DIR = Path(__file__).resolve().parent / "12jfpm_fbtcnn_fold01_val_s1"
TFLITE_PATH = OUT_DIR / "fbtcnn_float32.tflite"


def preprocess_sos(raw_tc: np.ndarray) -> np.ndarray:
    """SOS-based preprocessing matching C implementation → NHWC [1,8,512,4]."""
    T, C = raw_tc.shape
    output = np.zeros((1, C, T, len(BANDS)), dtype=np.float32)
    for b_idx, (lo, hi) in enumerate(BANDS):
        sos = signal.butter(ORDER, [2 * lo / FS, 2 * hi / FS], btype="band", output="sos")
        zi = signal.sosfilt_zi(sos)
        edge = 3 * (2 * ORDER + 1)
        for c in range(C):
            ch = raw_tc[:, c].astype(np.float64)
            ext = np.zeros(T + 2 * edge)
            for i in range(edge):
                ext[i] = 2.0 * ch[0] - ch[edge - i]
            ext[edge:edge + T] = ch
            for i in range(edge):
                ext[edge + T + i] = 2.0 * ch[-1] - ch[-2 - i]
            z_fwd = zi * ext[0]
            y_fwd, _ = signal.sosfilt(sos, ext, zi=z_fwd)
            z_rev = zi * y_fwd[-1]
            y_rev, _ = signal.sosfilt(sos, y_fwd[::-1], zi=z_rev)
            output[0, c, :, b_idx] = y_rev[::-1][edge:edge + T].astype(np.float32)
    return output


def main():
    if tf is None:
        print("tensorflow not available, skipping TFLite verification")
        return

    with open(OUT_DIR / "host_reference.json") as f:
        host_ref = json.load(f)

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    print(f"TFLite input shape: {inp_details[0]['shape']}")
    print(f"TFLite input dtype: {inp_details[0]['dtype']}")

    for vec_idx in range(5):
        bin_path = OUT_DIR / f"smoke_input_idx{vec_idx}_float32.bin"
        if not bin_path.exists():
            print(f"  vec{vec_idx}: no .bin file, skip")
            continue

        raw_tc = np.fromfile(bin_path, dtype=np.float32).reshape(512, 8)
        preprocessed = preprocess_sos(raw_tc)  # [1, 8, 512, 4]

        interpreter.set_tensor(inp_details[0]['index'], preprocessed)
        interpreter.invoke()
        scores = interpreter.get_tensor(out_details[0]['index']).flatten()

        ref = host_ref[f"idx{vec_idx}"]
        ref_argmax = ref["argmax"]
        ref_logits = np.array(ref["logits"])
        pred = int(np.argmax(scores))
        max_diff = float(np.max(np.abs(scores - ref_logits)))

        status = "PASS" if pred == ref_argmax else "FAIL"
        print(f"  vec{vec_idx}: pred={pred:2d} exp={ref_argmax:2d} {status}  max_diff={max_diff:.6f}")


if __name__ == "__main__":
    main()
