"""
Shared CLI helpers for LOSO protocol selection (Version A vs strict).

benchmark runs must use strict_loso_inner_val. subject_as_val is archived Version A.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from export.subject_paths import (
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    STRICT_EXPORT_SUBDIR,
)

VERSION_A_WARNING = (
    "WARNING: subject_as_val is archived Version A. It uses the LOSO left-out "
    "subject for deep-model early stopping/checkpoint selection and must NOT be "
    "used for benchmark results."
)

DEFAULT_PROTOCOL = PROTOCOL_STRICT_LOSO_INNER_VAL


def warn_if_subject_as_val(protocol: str) -> None:
    if protocol == PROTOCOL_SUBJECT_AS_VAL:
        print(VERSION_A_WARNING, file=sys.stderr)


def add_protocol_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_PROTOCOL,
) -> None:
    parser.add_argument(
        "--protocol",
        choices=[PROTOCOL_SUBJECT_AS_VAL, PROTOCOL_STRICT_LOSO_INNER_VAL],
        default=default,
        help=(
            f"LOSO protocol (default: {DEFAULT_PROTOCOL} for benchmark). "
            f"'{PROTOCOL_SUBJECT_AS_VAL}' is archived Version A only."
        ),
    )


def add_test_subject_args(parser: argparse.ArgumentParser, *, default: str = "s1") -> None:
    """--test-subject is the outer held-out subject; --val-subject is a legacy alias."""
    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--test-subject",
        dest="test_subject",
        default=None,
        help="Outer held-out test subject (strict LOSO), e.g. s1.",
    )
    grp.add_argument(
        "--val-subject",
        dest="test_subject",
        default=None,
        help="Deprecated alias for --test-subject (Version A naming).",
    )
    parser.set_defaults(test_subject=default)


def resolve_test_subject(args: Any) -> str:
    subj = getattr(args, "test_subject", None) or "s1"
    return str(subj).strip().lower()


def apply_protocol_side_effects(protocol: str) -> None:
    """Print warning when archived protocol is selected."""
    warn_if_subject_as_val(protocol)


def replay_results_root(repo: "Path | None" = None, protocol: str = DEFAULT_PROTOCOL):
    """Root directory for on-device replay session outputs."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
        return root / "results" / "replay" / STRICT_EXPORT_SUBDIR
    return root / "results" / "replay"
