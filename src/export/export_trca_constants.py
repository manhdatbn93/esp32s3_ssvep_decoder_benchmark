#!/usr/bin/env python3
"""
Export TRCA fold-specific templates and spatial filters for ESP32-S3 deployment.

TRCA is training-based: templates and spatial filters are learned from
the training fold. This script fits TRCA on fold01 (val=s1) and exports
the fold-specific artifacts as C headers.

Usage:
  python -m export.export_trca_constants [--out export/12jfpm_trca_fold01_val_s1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from export import common as _common
from export.common import (
    FREQS,
    FS,
    N_HARMONICS,
    T,
    C,
    build_reference_lut,
    compute_cheby_filter_bank,
    emit_ref_lut_header,
    get_fold_test_vectors,
    load_processed_npz,
    repo_root,
    save_test_vectors,
    write_host_reference,
    write_metadata,
)
from src.benchmarks.trca_12jfpm import N_CLASSES, TRCAConfig, _trca_eigvec

_DEFAULT_OUT = "export/12jfpm_trca_fold01_val_s1"


def _apply_filter_bank_sos(X: np.ndarray, banks: list[dict]) -> np.ndarray:
    """
    Apply the same SOS filter bank as embedded `trca_forward.c`:
    scipy sosfiltfilt with padlen = 3 * n_sos (matches sosfilt_odd_ext edge length).

    X: (N, T, C) float — same layout as TRCAModel.
    Returns (n_bands, N, C, T) float64.
    """
    _n, t_len, _c = X.shape
    n_bands = len(banks)
    x_ct = X.transpose(0, 2, 1).astype(np.float64, copy=False)
    result = np.zeros((n_bands, X.shape[0], X.shape[2], t_len), dtype=np.float64)
    for i, b in enumerate(banks):
        sos = b["sos"]
        n_sos = int(sos.shape[0])
        padlen = 3 * n_sos
        result[i] = scipy_signal.sosfiltfilt(sos, x_ct, axis=-1, padlen=padlen)
    return result


def _trca_fit_sos(X_train: np.ndarray, y_train: np.ndarray, cfg: TRCAConfig, banks: list[dict]):
    """Fit templates and W using SOS filtfilt (MCU-parity). Returns (templates, W)."""
    fb = _apply_filter_bank_sos(X_train, banks)
    n_bands = cfg.n_bands
    _, _, n_c, n_t = fb.shape
    templates = np.zeros((N_CLASSES, n_bands, n_c, n_t), dtype=np.float64)
    w_arr = np.zeros((n_bands, N_CLASSES, n_c), dtype=np.float64)

    for cls in range(N_CLASSES):
        mask = y_train == cls
        for bi in range(n_bands):
            trials_ct = fb[bi, mask]
            templates[cls, bi] = trials_ct.mean(axis=0)
            trials_ctn = trials_ct.transpose(1, 2, 0)
            w_arr[bi, cls] = _trca_eigvec(trials_ctn)

    return templates, w_arr


def _trca_predict_sos(
    X_test: np.ndarray,
    templates: np.ndarray,
    w_arr: np.ndarray,
    cfg: TRCAConfig,
    banks: list[dict],
) -> tuple[np.ndarray, list[list[float]]]:
    """Predict labels and full score vectors (for smoke reference)."""
    fb = _apply_filter_bank_sos(X_test, banks)
    n_test = X_test.shape[0]
    n_bands = cfg.n_bands
    fb_coefs = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(n_bands)])
    preds = np.empty(n_test, dtype=np.int64)
    all_scores: list[list[float]] = []

    for trial in range(n_test):
        rho = np.zeros(N_CLASSES)
        for bi in range(n_bands):
            test_ct = fb[bi, trial]
            for cls in range(N_CLASSES):
                template = templates[cls, bi]
                w = w_arr[bi].T
                proj_test = (test_ct.T @ w).flatten()
                proj_template = (template.T @ w).flatten()
                r = np.corrcoef(proj_test, proj_template)[0, 1]
                if not np.isfinite(r):
                    r = 0.0
                rho[cls] += fb_coefs[bi] * r
        preds[trial] = int(np.argmax(rho))
        all_scores.append([float(x) for x in rho])

    return preds, all_scores


def _emit_trca_config_header(out_dir: Path, cfg: TRCAConfig, banks: list[dict]) -> Path:
    n_bands = cfg.n_bands
    max_sos = max(b["sos"].shape[0] for b in banks)
    fb_coefs = [(i + 1) ** (-1.25) + 0.25 for i in range(n_bands)]

    lines = [
        "/* Auto-generated TRCA config for 12JFPM — do not edit. */",
        "#ifndef TRCA_CONFIG_GENERATED_H",
        "#define TRCA_CONFIG_GENERATED_H",
        "",
        f"#define TRCA_FS_HZ {cfg.sampling_rate_hz}",
        f"#define TRCA_T {T}",
        f"#define TRCA_C {C}",
        f"#define TRCA_NUM_CLASSES {len(FREQS)}",
        f"#define TRCA_N_BANDS {n_bands}",
        f"#define TRCA_N_SOS_MAX {max_sos}",
        f"#define TRCA_ENSEMBLE 1",
        "",
        f"static const double TRCA_BAND_COEFS[{n_bands}] = {{",
        f"  {', '.join(f'{w:.16f}' for w in fb_coefs)}",
        "};",
        "",
        f"static const int TRCA_N_SOS_PER_BAND[{n_bands}] = {{",
        f"  {', '.join(str(b['sos'].shape[0]) for b in banks)}",
        "};",
        "",
        f"static const double TRCA_SOS[{n_bands}][{max_sos}][6] = {{",
    ]
    for i, b in enumerate(banks):
        sos = b["sos"]
        n_s = sos.shape[0]
        lines.append(f"  {{ /* band {i} */")
        for s in range(max_sos):
            if s < n_s:
                vals = ", ".join(f"{v}" for v in sos[s])
            else:
                vals = "0, 0, 0, 1, 0, 0"
            comma = "," if s < max_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < n_bands - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")

    lines.append(f"static const double TRCA_SOSFILT_ZI[{n_bands}][{max_sos}][2] = {{")
    for i, b in enumerate(banks):
        zi = b["zi"]
        n_s = zi.shape[0]
        lines.append(f"  {{ /* band {i} */")
        for s in range(max_sos):
            if s < n_s:
                vals = ", ".join(f"{v}" for v in zi[s])
            else:
                vals = "0, 0"
            comma = "," if s < max_sos - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if i < n_bands - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* TRCA_CONFIG_GENERATED_H */")
    lines.append("")
    p = out_dir / "trca_config_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _emit_trca_templates_header(out_dir: Path, templates: np.ndarray) -> Path:
    """templates: [n_classes, n_bands, C, T] float64."""
    n_cls, n_bands, n_c, n_t = templates.shape
    lines = [
        "/* Auto-generated TRCA templates (fold-specific) — do not edit. */",
        "#ifndef TRCA_TEMPLATES_GENERATED_H",
        "#define TRCA_TEMPLATES_GENERATED_H",
        "",
        f"#define TRCA_TMPL_N_CLASSES {n_cls}",
        f"#define TRCA_TMPL_N_BANDS {n_bands}",
        f"#define TRCA_TMPL_C {n_c}",
        f"#define TRCA_TMPL_T {n_t}",
        "",
        f"static const float TRCA_TEMPLATES[{n_cls}][{n_bands}][{n_c}][{n_t}] = {{",
    ]
    for cls in range(n_cls):
        lines.append(f"  {{ /* class {cls} */")
        for bi in range(n_bands):
            lines.append(f"    {{ /* band {bi} */")
            for ch in range(n_c):
                vals = ", ".join(f"{v:.8f}f" for v in templates[cls, bi, ch, :].astype(np.float32))
                comma = "," if ch < n_c - 1 else ""
                lines.append(f"      {{ {vals} }}{comma}")
            comma = "," if bi < n_bands - 1 else ""
            lines.append(f"    }}{comma}")
        comma = "," if cls < n_cls - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* TRCA_TEMPLATES_GENERATED_H */")
    lines.append("")
    p = out_dir / "trca_templates_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _emit_trca_spatial_filters_header(out_dir: Path, W: np.ndarray) -> Path:
    """W: [n_bands, n_classes, C] float64."""
    n_bands, n_cls, n_c = W.shape
    lines = [
        "/* Auto-generated TRCA spatial filters (fold-specific) — do not edit. */",
        "#ifndef TRCA_SPATIAL_FILTERS_GENERATED_H",
        "#define TRCA_SPATIAL_FILTERS_GENERATED_H",
        "",
        f"#define TRCA_SF_N_BANDS {n_bands}",
        f"#define TRCA_SF_N_CLASSES {n_cls}",
        f"#define TRCA_SF_C {n_c}",
        "",
        f"static const float TRCA_SPATIAL_W[{n_bands}][{n_cls}][{n_c}] = {{",
    ]
    for bi in range(n_bands):
        lines.append(f"  {{ /* band {bi} */")
        for cls in range(n_cls):
            vals = ", ".join(f"{v:.8f}f" for v in W[bi, cls, :].astype(np.float32))
            comma = "," if cls < n_cls - 1 else ""
            lines.append(f"    {{ {vals} }}{comma}")
        comma = "," if bi < n_bands - 1 else ""
        lines.append(f"  }}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* TRCA_SPATIAL_FILTERS_GENERATED_H */")
    lines.append("")
    p = out_dir / "trca_spatial_filters_generated.h"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Export TRCA constants for ESP32-S3")
    ap.add_argument("--out", type=Path, default=repo_root() / _DEFAULT_OUT)
    ap.add_argument("--processed-npz", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"TRCA export -> {out_dir}")

    cfg = TRCAConfig()

    npz = load_processed_npz(args.processed_npz)
    X, y, subjects = npz["X"], npz["y"], npz.get("subjects", npz.get("subject"))

    val_subject = _common.CANONICAL_VAL_SUBJECT
    print(f"  Fitting TRCA on fold{_common.CANONICAL_FOLD:02d} "
          f"(val={val_subject}) [SOS filtfilt, MCU-parity]...")
    mask_val = np.array([str(s) == val_subject for s in subjects])
    X_train = X[~mask_val]
    y_train = y[~mask_val]

    bank_params = {
        "passband_hz": list(cfg.passband_hz),
        "stopband_hz": list(cfg.stopband_hz),
        "highcut_pass_hz": cfg.highcut_pass_hz,
        "highcut_stop_hz": cfg.highcut_stop_hz,
        "gpass_db": cfg.gpass_db,
        "gstop_db": cfg.gstop_db,
        "cheby_ripple_db": cfg.cheby_ripple_db,
    }
    banks = compute_cheby_filter_bank(bank_params)

    templates, w_arr = _trca_fit_sos(X_train, y_train, cfg, banks)
    print(f"  Templates shape: {templates.shape}")
    print(f"  Spatial filters shape: {w_arr.shape}")

    _emit_trca_config_header(out_dir, cfg, banks)
    print("  -> trca_config_generated.h")

    _emit_trca_templates_header(out_dir, templates)
    print("  -> trca_templates_generated.h")

    _emit_trca_spatial_filters_header(out_dir, w_arr)
    print("  -> trca_spatial_filters_generated.h")

    vectors = get_fold_test_vectors(npz)
    save_test_vectors(vectors, out_dir)
    print(f"  -> {len(vectors)} test vectors")

    X_test = np.stack(vectors)
    preds, score_rows = _trca_predict_sos(X_test, templates, w_arr, cfg, banks)
    refs = {}
    for i in range(len(vectors)):
        refs[f"idx{i}"] = {
            "predicted_class": int(preds[i]),
            "scores": score_rows[i],
        }
    write_host_reference(out_dir, refs)
    print("  -> host_reference.json")

    outer_test = _common.CANONICAL_VAL_SUBJECT
    write_metadata(out_dir, "trca", {
        "n_bands": cfg.n_bands,
        "ensemble": cfg.ensemble,
        "fold_specific": True,
        "outer_test_subject": outer_test,
        "train_subjects": sorted(set(str(s) for s in subjects[~mask_val])),
        "protocol": "strict_loso_inner_val" if "_test_" in str(out_dir) else "subject_as_val",
    })
    print("  -> metadata.json")
    print("TRCA export complete.")


if __name__ == "__main__":
    main()
