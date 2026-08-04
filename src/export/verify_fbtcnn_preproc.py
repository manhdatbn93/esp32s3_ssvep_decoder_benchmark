#!/usr/bin/env python3
"""Verify FBtCNN C preprocessing would match Python host.

Loads the raw smoke vectors, applies the Python preprocessing, and
compares with what the C SOS-based approach should produce.
"""
import numpy as np
from scipy import signal
from pathlib import Path
import struct
import json

BANDS = [(3, 14), (9, 26), (14, 38), (19, 50)]
FS = 256.0
ORDER = 6

OUT_DIR = Path(__file__).resolve().parent / "12jfpm_fbtcnn_fold01_val_s1"


def python_preprocess_ba(raw_tc: np.ndarray) -> np.ndarray:
    """Exact Python host preprocessing (b,a form, scipy defaults)."""
    x = raw_tc.T[np.newaxis, np.newaxis, :, :]  # [1, 1, C=8, T=512]
    bands_out = []
    for lo, hi in BANDS:
        b, a = signal.butter(ORDER, [2 * lo / FS, 2 * hi / FS], btype="bandpass")
        y = signal.filtfilt(b, a, x, axis=-1)
        bands_out.append(y)
    stacked = np.stack(bands_out, axis=1)  # [1, 4, 1, 8, 512]
    if stacked.ndim == 5:
        stacked = stacked.squeeze(2)  # [1, 4, 8, 512]
    return stacked


def python_preprocess_sos(raw_tc: np.ndarray) -> np.ndarray:
    """SOS-based preprocessing matching C implementation."""
    T, C = raw_tc.shape
    n_bands = len(BANDS)
    output = np.zeros((C, T, n_bands), dtype=np.float64)
    for b_idx, (lo, hi) in enumerate(BANDS):
        sos = signal.butter(ORDER, [2 * lo / FS, 2 * hi / FS], btype="band", output="sos")
        zi = signal.sosfilt_zi(sos)
        edge = 3 * (2 * ORDER + 1)  # match scipy default padlen for (b,a)
        for c in range(C):
            ch = raw_tc[:, c].astype(np.float64)
            # Odd extension
            ext = np.zeros(T + 2 * edge)
            for i in range(edge):
                ext[i] = 2.0 * ch[0] - ch[edge - i]
            ext[edge:edge + T] = ch
            for i in range(edge):
                ext[edge + T + i] = 2.0 * ch[-1] - ch[-2 - i]
            # Forward pass
            z_fwd = zi * ext[0]
            y_fwd, _ = signal.sosfilt(sos, ext, zi=z_fwd)
            # Reverse pass
            z_rev = zi * y_fwd[-1]
            y_rev, _ = signal.sosfilt(sos, y_fwd[::-1], zi=z_rev)
            filtered = y_rev[::-1]
            output[c, :, b_idx] = filtered[edge:edge + T]
    return output


def main():
    smoke_header = OUT_DIR.parent / "generated_smoke_headers" / "smoke_vectors_fbtcnn.h"
    host_ref_path = OUT_DIR / "host_reference.json"
    with open(host_ref_path) as f:
        host_ref = json.load(f)

    n_vecs = len([k for k in host_ref if k.startswith("idx")])
    for vec_idx in range(min(3, n_vecs)):
        bin_path = OUT_DIR / f"smoke_input_idx{vec_idx}_float32.bin"
        if not bin_path.exists():
            print(f"  vec{vec_idx}: no .bin file, skip")
            continue
        raw_data = np.fromfile(bin_path, dtype=np.float32)
        raw_tc = raw_data.reshape(512, 8)

        # Python host approach (b,a)
        ba_out = python_preprocess_ba(raw_tc)  # [1, 4, 8, 512]

        # SOS approach (matches C impl)
        sos_out = python_preprocess_sos(raw_tc)  # [8, 512, 4]
        # Reshape to [1, 4, 8, 512] for comparison
        sos_nchw = sos_out.transpose(2, 0, 1)[np.newaxis]  # [1, 4, 8, 512]

        diff = np.abs(ba_out.astype(np.float64) - sos_nchw)
        print(f"  vec{vec_idx}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

    print("\nDone. If max_diff is near machine epsilon, C impl matches Python host.")


if __name__ == "__main__":
    main()
