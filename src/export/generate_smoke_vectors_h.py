#!/usr/bin/env python3
"""
Generate C header files containing embedded smoke test vectors and expected
outputs for each export bundle. These headers are included by firmware backends
so test data lives in flash, not RAM.

Run from repo root:
  python -m export.generate_smoke_vectors_h
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T, C, N_CLASSES = 512, 8, 12


def _load_bin_vector(path: Path) -> list[float]:
    raw = path.read_bytes()
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw))


def _format_float_array(values: list[float], per_line: int = 8) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("  " + ", ".join(f"{v:.8e}f" for v in chunk) + ",")
    return "\n".join(lines)


def generate_header(bundle_dir: Path, backend_tag: str, score_key: str = "scores") -> str:
    """Generate a C header with smoke vectors and expected outputs.

    score_key: 'scores' for classical, 'logits' for DL models.
    """
    ref_path = bundle_dir / "host_reference.json"
    if not ref_path.exists():
        return None

    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    bin_files = sorted(bundle_dir.glob("smoke_input_idx*_float32.bin"))
    if not bin_files:
        return None

    n_vec = len(bin_files)
    tag = backend_tag.upper()

    lines = [
        f"/* Auto-generated smoke vectors for {backend_tag} — do not edit. */",
        f"#ifndef SMOKE_VECTORS_{tag}_H",
        f"#define SMOKE_VECTORS_{tag}_H",
        "",
        f"#define SMOKE_{tag}_N_VECTORS {n_vec}",
        f"#define SMOKE_{tag}_T {T}",
        f"#define SMOKE_{tag}_C {C}",
        f"#define SMOKE_{tag}_N_CLASSES {N_CLASSES}",
        "",
    ]

    argmax_vals = []
    for i in range(n_vec):
        entry = ref[f"idx{i}"]
        am = entry.get("argmax", entry.get("predicted_class", -1))
        argmax_vals.append(am)

    lines.append(f"static const int SMOKE_{tag}_EXPECTED_ARGMAX[{n_vec}] = {{")
    lines.append("  " + ", ".join(str(a) for a in argmax_vals))
    lines.append("};")
    lines.append("")

    has_scores = any(
        score_key in ref[f"idx{i}"] or "scores" in ref[f"idx{i}"] or "logits" in ref[f"idx{i}"]
        for i in range(n_vec)
    )
    lines.append(f"#define SMOKE_{tag}_HAS_EXPECTED_SCORES {1 if has_scores else 0}")
    lines.append("")

    if has_scores:
        lines.append(f"static const float SMOKE_{tag}_EXPECTED_SCORES[{n_vec}][{N_CLASSES}] = {{")
        for i in range(n_vec):
            entry = ref[f"idx{i}"]
            scores = entry.get(score_key, entry.get("scores", entry.get("logits", [])))
            if not scores:
                scores = [0.0] * N_CLASSES
            lines.append(f"  {{ /* idx{i} */")
            lines.append("    " + ", ".join(f"{v:.8e}f" for v in scores))
            lines.append("  },")
        lines.append("};")
        lines.append("")

    for i, bf in enumerate(bin_files):
        vec = _load_bin_vector(bf)
        lines.append(f"static const float SMOKE_{tag}_VEC{i}[{T * C}] = {{")
        lines.append(_format_float_array(vec, per_line=8))
        lines.append("};")
        lines.append("")

    lines.append(f"static const float *SMOKE_{tag}_VECTORS[{n_vec}] = {{")
    lines.append("  " + ", ".join(f"SMOKE_{tag}_VEC{i}" for i in range(n_vec)))
    lines.append("};")
    lines.append("")

    lines.append(f"#endif /* SMOKE_VECTORS_{tag}_H */")
    lines.append("")

    return "\n".join(lines)


BUNDLES = [
    ("12jfpm_cca_fold01_val_s1", "cca", "scores"),
    ("12jfpm_fbcca_fold01_val_s1", "fbcca", "scores"),
    ("12jfpm_msi_fold01_val_s1", "msi", "scores"),
    ("12jfpm_trca_fold01_val_s1", "trca", "scores"),
    ("12jfpm_eegnet_fold01_val_s1", "eegnet", "logits"),
    ("12jfpm_ccnn_fold01_val_s1", "ccnn", "logits"),
    ("12jfpm_convca_fold01_val_s1", "convca", "logits"),
    ("12jfpm_fbtcnn_fold01_val_s1", "fbtcnn", "logits"),
]


def main():
    export_dir = REPO / "export"
    out_dir = export_dir / "generated_smoke_headers"
    out_dir.mkdir(exist_ok=True)

    for bundle_name, tag, score_key in BUNDLES:
        bundle_dir = export_dir / bundle_name
        if not bundle_dir.exists():
            print(f"  SKIP {tag}: {bundle_dir} not found")
            continue

        header = generate_header(bundle_dir, tag, score_key)
        if header is None:
            print(f"  SKIP {tag}: no smoke vectors or host_reference.json")
            continue

        out_path = out_dir / f"smoke_vectors_{tag}.h"
        out_path.write_text(header, encoding="utf-8")
        print(f"  OK {tag}: {out_path.name}")

    print(f"\nHeaders written to {out_dir}")


if __name__ == "__main__":
    main()
