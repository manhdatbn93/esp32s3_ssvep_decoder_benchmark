"""
Canonical fold/subject → export directory naming.

Legacy (Version A) — subject-as-validation LOSO:
  export/12jfpm_<model>_fold{NN}_val_{subject}[_int8]

Strict outer LOSO with inner trial validation:
  export/strict_loso_inner_val/12jfpm_<model>_fold{NN}_test_{subject}[_int8]

Fold N is paired with outer held-out / test subject sN unless overridden.
"""
from __future__ import annotations

from pathlib import Path

PROTOCOL_SUBJECT_AS_VAL = "subject_as_val"
PROTOCOL_STRICT_LOSO_INNER_VAL = "strict_loso_inner_val"
STRICT_EXPORT_SUBDIR = "strict_loso_inner_val"

_SUBJECT_IDS: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 11))


def all_subjects() -> tuple[str, ...]:
    """Return the canonical 12JFPM subject id tuple, s1..s10."""
    return _SUBJECT_IDS


def fold_for_subject(subject: str) -> int:
    """Derive fold number from subject id under the LOSO convention."""
    s = subject.strip().lower()
    if not s.startswith("s") or not s[1:].isdigit():
        raise ValueError(f"Invalid subject id: {subject!r}")
    n = int(s[1:])
    if n < 1 or n > 10:
        raise ValueError(f"Subject index out of range (1..10): {subject!r}")
    return n


def tag_for(
    subject: str = "s1",
    fold: int | None = None,
    *,
    protocol: str = PROTOCOL_STRICT_LOSO_INNER_VAL,
) -> str:
    """Return fold tag string, e.g. 'fold01_val_s1' or 'fold01_test_s1'."""
    s = subject.strip().lower()
    f = fold if fold is not None else fold_for_subject(s)
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        return f"fold{f:02d}_test_{s}"
    if protocol == PROTOCOL_SUBJECT_AS_VAL:
        return f"fold{f:02d}_val_{s}"
    raise ValueError(f"Unknown protocol: {protocol!r}")


def export_dir_name(
    model: str,
    subject: str = "s1",
    fold: int | None = None,
    *,
    int8: bool = False,
    protocol: str = PROTOCOL_STRICT_LOSO_INNER_VAL,
) -> str:
    """Return relative export folder name under export/."""
    tag = tag_for(subject, fold, protocol=protocol)
    suffix = "_int8" if int8 else ""
    return f"12jfpm_{model}_{tag}{suffix}"


def export_dir(
    model: str,
    subject: str = "s1",
    fold: int | None = None,
    *,
    int8: bool = False,
    protocol: str = PROTOCOL_STRICT_LOSO_INNER_VAL,
    repo: Path | None = None,
) -> Path:
    """Absolute Path under <repo>/export/ for the given model+subject+variant."""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    name = export_dir_name(model, subject, fold, int8=int8, protocol=protocol)
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        return repo / "export" / STRICT_EXPORT_SUBDIR / name
    return repo / "export" / name


def checkpoint_dir_name(model: str, *, protocol: str = PROTOCOL_STRICT_LOSO_INNER_VAL) -> str:
    """Checkpoint directory name under checkpoints/."""
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        return f"{model}_strict_loso_inner_val_12jfpm"
    return f"{model}_loso_12jfpm"


def checkpoint_dir(
    model: str,
    *,
    protocol: str = PROTOCOL_STRICT_LOSO_INNER_VAL,
    repo: Path | None = None,
) -> Path:
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    return repo / "checkpoints" / checkpoint_dir_name(model, protocol=protocol)


def legacy_checkpoint_path(model: str, fold: int, repo: Path | None = None) -> Path:
    """Version A checkpoint path (must not be used under strict protocol)."""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    return repo / "checkpoints" / f"{model}_loso_12jfpm" / f"fold{fold:02d}_best.pt"


def legacy_export_dir(
    model: str, subject: str, fold: int | None = None, *, int8: bool = False, repo: Path | None = None,
) -> Path:
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    f = fold if fold is not None else fold_for_subject(subject)
    suffix = "_int8" if int8 else ""
    return repo / "export" / f"12jfpm_{model}_fold{f:02d}_val_{subject}{suffix}"


def is_strict_benchmark_protocol(protocol: str) -> bool:
    """True for strict_loso_inner_val (not Version A or diagnostic fixed200 variants)."""
    return protocol == PROTOCOL_STRICT_LOSO_INNER_VAL


def assert_not_legacy_artifact(path: Path, *, protocol: str, context: str = "") -> None:
    """Raise if a strict run resolved a Version A export/checkpoint path."""
    if not is_strict_benchmark_protocol(protocol):
        return
    parts = path.as_posix().lower()
    if "_val_s" in parts and "_test_s" not in parts:
        raise ValueError(
            f"Strict protocol must not use Version A path {path!s} {context}".strip()
        )
    if "/checkpoints/" in parts and "_loso_12jfpm/" in parts and "strict" not in parts:
        raise ValueError(
            f"Strict protocol must not load Version A checkpoint {path!s} {context}".strip()
        )
    if "fixed200" in parts or "_no_es_" in parts:
        raise ValueError(
            f"Strict benchmark protocol must not use diagnostic checkpoint path {path!s} "
            f"{context}".strip()
        )


def results_dir_strict(model: str, repo: Path | None = None) -> Path:
    """results/strict_loso_inner_val/ for per-model JSON summaries."""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    return repo / "results" / STRICT_EXPORT_SUBDIR / f"{model}_12jfpm.json"


def strict_diagnostic_checkpoint_dir(
    model: str,
    diagnostic_tag: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Non-destructive strict checkpoints: checkpoints/<model>_strict_loso_inner_val_12jfpm_<tag>/"""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    tag = diagnostic_tag.strip()
    if not tag:
        raise ValueError("diagnostic_tag must be non-empty")
    return repo / "checkpoints" / f"{checkpoint_dir_name(model)}_{tag}"


def strict_diagnostic_results_root(diagnostic_tag: str, *, repo: Path | None = None) -> Path:
    """Non-destructive strict results root: results/strict_loso_inner_val_<tag>/"""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    tag = diagnostic_tag.strip()
    if not tag:
        raise ValueError("diagnostic_tag must be non-empty")
    return repo / "results" / f"{STRICT_EXPORT_SUBDIR}_{tag}"


def strict_diagnostic_results_path(
    model: str,
    diagnostic_tag: str,
    *,
    repo: Path | None = None,
) -> Path:
    return strict_diagnostic_results_root(diagnostic_tag, repo=repo) / f"{model}_12jfpm.json"
