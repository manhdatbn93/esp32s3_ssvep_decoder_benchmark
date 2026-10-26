"""Figure 2 (margin quartiles) and Figure 3 (latency versus accuracy and agreement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constants import INT8_BACKENDS, NEURAL_BACKENDS, QUARTILES  # noqa: E402

_PDF_METADATA = {"Creator": "paper_reproduction", "CreationDate": None, "ModDate": None}


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        options: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.02, "dpi": 300}
        if suffix == ".pdf":
            options["metadata"] = _PDF_METADATA
        fig.savefig(path, **options)
        paths.append(path)
    plt.close(fig)
    return paths


def figure2_margin_quartile(rows: list[dict], stem: Path) -> list[Path]:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = {"CCNN-int8": "#2166ac", "EEGNet-int8": "#b2182b", "FBtCNN-int8": "#404040"}
    markers = {"CCNN-int8": "o", "EEGNet-int8": "s", "FBtCNN-int8": "^"}
    offsets = {"CCNN-int8": -0.17, "EEGNet-int8": 0.0, "FBtCNN-int8": 0.17}
    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.17, right=0.98)
    max_ci = 0.0
    for backend in INT8_BACKENDS:
        by_q = {r["host_margin_quartile"]: r for r in rows if r["backend"] == backend}
        x = [i + offsets[backend] for i in range(len(QUARTILES))]
        y = [by_q[q]["subject_mean_flip_rate_pct"] for q in QUARTILES]
        lo = [y[i] - by_q[q]["bootstrap_ci_lo_pct"] for i, q in enumerate(QUARTILES)]
        hi = [by_q[q]["bootstrap_ci_hi_pct"] - y[i] for i, q in enumerate(QUARTILES)]
        max_ci = max(max_ci, *(by_q[q]["bootstrap_ci_hi_pct"] for q in QUARTILES))
        ax.errorbar(x, y, yerr=[lo, hi], fmt=markers[backend], linestyle="none", capsize=2.5,
                    capthick=0.8, elinewidth=0.9, markersize=4.5,
                    markerfacecolor="white" if backend == "FBtCNN-int8" else colors[backend],
                    markeredgewidth=0.9, color=colors[backend], label=backend, zorder=3)
        q1 = by_q["Q1"]
        ax.annotate(f"{q1['n_flips']}/{q1['n_trials']}", (x[0], y[0]), textcoords="offset points",
                    xytext=(4, 4), ha="left", va="bottom", fontsize=6, color=colors[backend])
    ax.set_xticks(range(len(QUARTILES)))
    ax.set_xticklabels(QUARTILES)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(-0.5, max(22.0, max_ci * 1.12))
    ax.set_xlabel("Host-margin quartile")
    ax.set_ylabel("Stage 2 host\u2013device flip rate (%)")
    ax.grid(axis="y", linestyle="-", linewidth=0.35, alpha=0.45, color="#999999")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    return _save(fig, stem)


_GROUPS = (("f32", "o", "Neural (f32)", "#2166ac"), ("int8", "^", "Neural (int8)", "#b2182b"))
_ACCURACY_LABELS = {
    "ConvCA-f32": (0, 5), "FBtCNN-int8": (0, -12), "FBtCNN-f32": (-2, 5), "CCNN-int8": (2, -13),
    "CCNN-f32": (0, 5), "EEGNet-int8": (6, 5), "EEGNet-f32": (0, 5),
}
_AGREEMENT_LABELS = {
    "CCNN-int8": ((-14, -6), "left", "top"), "EEGNet-int8": ((-14, 6), "left", "bottom"),
    "CCNN-f32": ((-12, -6), "left", "top"), "EEGNet-f32": ((14, -12), "right", "bottom"),
    "ConvCA-f32": ((15, 12), "right", "top"), "FBtCNN-f32": ((15, -6), "right", "top"),
    "FBtCNN-int8": ((-16, -8), "left", "top"),
}


def _latency_panel(t2: dict, key: str, ylabel: str, ylim: tuple[float, float], labels: dict, stem: Path):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.15, right=0.97)
    for precision, marker, _label, color in _GROUPS:
        names = [b for b in NEURAL_BACKENDS if b.endswith(precision)]
        ax.scatter([t2[b]["latency_ms"] for b in names], [t2[b][key] for b in names], s=75.0,
                   marker=marker, c=color, alpha=0.9, edgecolors="0.15", linewidths=0.45, zorder=2)
    for b in NEURAL_BACKENDS:
        spec = labels.get(b, (0, 5))
        offset, ha, va = (spec, "center", "bottom") if len(spec) == 2 and isinstance(spec[0], int) else spec
        ax.annotate(b, (t2[b]["latency_ms"], t2[b][key]), textcoords="offset points", xytext=offset,
                    ha=ha, va=va, fontsize=5.5, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25))
    ax.set_xscale("log")
    ax.set_xlim(45, 2.2e4)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Mean latency (ms, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.45, color="#999999")
    ax.grid(False, which="minor")
    ax.tick_params(which="both", direction="out", length=3)
    handles = [plt.Line2D([0], [0], marker=m, color="white", markerfacecolor=c, markeredgecolor="0.15",
                          markeredgewidth=0.45, label=lab, linestyle="None", markersize=6)
               for _p, m, lab, c in _GROUPS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
              columnspacing=0.9, handletextpad=0.35, borderaxespad=0.0, markerscale=0.75)
    return _save(fig, stem)


def figure3_latency(t2: dict, out_dir: Path) -> list[Path]:
    y_min = max(15.0, min(t2[b]["device_acc_pct"] for b in NEURAL_BACKENDS) - 8.0)
    paths = _latency_panel(t2, "device_acc_pct", "Device accuracy (%)", (y_min, 100.0), _ACCURACY_LABELS,
                           out_dir / "fig_latency_accuracy_neural")
    paths += _latency_panel(t2, "agreement_pct", "Host\u2013device agreement (%)", (94.8, 100.6),
                            _AGREEMENT_LABELS, out_dir / "fig_latency_fidelity_neural")
    return paths
