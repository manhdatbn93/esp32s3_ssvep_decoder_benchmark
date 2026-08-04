#!/usr/bin/env python3
"""Generate FBtCNN filter-bank SOS coefficients as a C header.

Produces fbtcnn_filter_generated.h with:
  - SOS coefficients for 4 Butterworth bandpass filters (order 6)
  - sosfilt_zi initial conditions for each band
  - edge (padlen) value for scipy-compatible filtfilt

Usage:
  python export/gen_fbtcnn_filter_header.py
"""
import numpy as np
from scipy import signal
from pathlib import Path

BANDS = [(3, 14), (9, 26), (14, 38), (19, 50)]
FS = 256.0
ORDER = 6
N_BANDS = len(BANDS)

# Canonical s1 path — identical coefficients are synced to all foldNN_val_sN dirs
# via export/sync_fbtcnn_filter_headers.py (subject-independent Butterworth banks).
OUT_DIR = Path(__file__).resolve().parent / "12jfpm_fbtcnn_fold01_val_s1"
OUT_FILE = OUT_DIR / "fbtcnn_filter_generated.h"


def main():
    banks = []
    for lo, hi in BANDS:
        Wn = [2 * lo / FS, 2 * hi / FS]
        sos = signal.butter(ORDER, Wn, btype="band", output="sos")
        zi = signal.sosfilt_zi(sos)
        banks.append({"sos": sos, "zi": zi})

    n_sos = banks[0]["sos"].shape[0]
    # Match scipy.signal.filtfilt default padlen for (b,a) form:
    # padlen = 3 * max(len(a), len(b)) = 3 * (2*ORDER + 1)
    b_a_len = 2 * ORDER + 1  # len(b) = len(a) for bandpass butter
    edge = 3 * b_a_len

    lines = [
        "/* Auto-generated FBtCNN filter-bank constants — do not edit. */",
        "#ifndef FBTCNN_FILTER_GENERATED_H",
        "#define FBTCNN_FILTER_GENERATED_H",
        "",
        f"#define FBTCNN_N_BANDS  {N_BANDS}",
        f"#define FBTCNN_N_SOS    {n_sos}",
        f"#define FBTCNN_EDGE     {edge}   /* padlen for filtfilt odd extension */",
        f"#define FBTCNN_FS_HZ    {FS}",
        f"#define FBTCNN_ORDER    {ORDER}",
        "",
    ]

    lines.append(f"static const double FBTCNN_SOS[{N_BANDS}][{n_sos}][6] = {{")
    for i, b in enumerate(banks):
        lo, hi = BANDS[i]
        lines.append(f"  {{ /* band {i}: {lo}–{hi} Hz */")
        sos = b["sos"]
        for s in range(n_sos):
            vals = ", ".join(f"{v:.18e}" for v in sos[s])
            comma = "," if s < n_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < N_BANDS - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")

    lines.append(f"static const double FBTCNN_ZI[{N_BANDS}][{n_sos}][2] = {{")
    for i, b in enumerate(banks):
        lo, hi = BANDS[i]
        lines.append(f"  {{ /* band {i}: {lo}–{hi} Hz */")
        zi = b["zi"]
        for s in range(n_sos):
            vals = ", ".join(f"{v:.18e}" for v in zi[s])
            comma = "," if s < n_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < N_BANDS - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* FBTCNN_FILTER_GENERATED_H */")
    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {OUT_FILE}")
    print(f"  n_bands={N_BANDS}, n_sos={n_sos}, edge={edge}")
    for i, (lo, hi) in enumerate(BANDS):
        print(f"  Band {i}: {lo}–{hi} Hz")


if __name__ == "__main__":
    main()
