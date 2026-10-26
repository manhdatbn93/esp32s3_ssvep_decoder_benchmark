#!/usr/bin/env python3
"""Maintainer tool: build the sanitized public evidence copies under ``evidence/``.

This script documents how ``paper_reproduction/evidence`` was produced. Users do
not need to run it. It reads the private development workspace and the private
resource-profiling folder (both passed as arguments), never writes to them, and
writes only under ``--out``.

Rules:

* Frozen files are verified against the development SHA-256 freeze
  (``docs/SUBMITTED_ARTIFACTS.sha256.json``) before they are copied.
* Text files are copied byte for byte except for the explicit replacements in
  ``RULES``: the private raw-data directory, the private development-repository
  root, private IPv4 addresses, and MAC addresses.
* Images are copied unchanged.
* Every output is listed in ``SANITIZATION_MAP.json`` with the original and
  public SHA-256 and the number of replacements per rule.
* The build fails if any absolute drive path, private IPv4 address, or MAC
  address remains in a public text copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import struct
import sys
from pathlib import Path

RAW_DATA_SUBDIR = ("existing_work", "data", "raw", "12JFPM", "extracted", "12JFPM_SSVEP-master", "data")
PLACEHOLDER_RAW = b"<12JFPM_RAW_DATA_DIR>"
PLACEHOLDER_DEV = b"<DEV_REPO>"
PLACEHOLDER_IP = b"<REDACTED_IP>"
PLACEHOLDER_MAC = b"<REDACTED_MAC>"

PRIVATE_IPV4 = re.compile(
    rb"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
MAC = re.compile(rb"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")
RESIDUAL_ABS_PATH = re.compile(rb"\b[A-Za-z]:(?:\\\\|\\|/)(?:[^\\/\s\"',|]+)")

STRICT_REPLAY_PREFIX = "results/replay/strict_loso_inner_val/"
FIDELITY_PREFIX = "results/fidelity/strict_loso_inner_val/"

ANALYTICAL_REPLAY = {"CCA", "FBCCA", "MSI", "TRCA"}
ANALYSIS_PUBLIC_NAMES = {
    "figure3_margin_quartile_source_r21.csv": "margin_quartile_figure_source.csv",
    "margin_quartile_integrity_r21.csv": "margin_quartile_integrity.csv",
    "normalized_output_error_summary_r21.csv": "normalized_output_error_summary.csv",
    "normalized_output_error_by_subject_r21.csv": "normalized_output_error_by_subject.csv",
    "raw_output_error_summary_r21.csv": "raw_output_error_summary.csv",
    "deployment_policy_definitions_r21.csv": "deployment_policy_definitions.csv",
    "deployment_policy_admissible_sets_r21.csv": "deployment_policy_admissible_sets.csv",
    "deployment_policy_pareto_sets_r21.csv": "deployment_policy_pareto_sets.csv",
    "deployment_policy_selection_r21.csv": "deployment_policy_selection.csv",
    "deployment_policy_exclusions_r21.csv": "deployment_policy_exclusions.csv",
    "deployment_policy_sensitivity_r21.csv": "deployment_policy_sensitivity.csv",
}


def public_replay_rel(rel: str) -> str:
    """Map a private replay path to its public path.

    Analytical subject folders drop the historical ``_strict_migrated`` suffix.
    Neural run folders keep their date stamps. The private source path is unchanged.
    """
    public_rel = "replay/" + rel[len(STRICT_REPLAY_PREFIX):]
    parts = public_rel.split("/")
    if len(parts) >= 3 and parts[1] in ANALYTICAL_REPLAY and parts[2].endswith("_strict_migrated"):
        parts[2] = parts[2][: -len("_strict_migrated")]
        public_rel = "/".join(parts)
    return public_rel


R21_OUTPUTS = (
    "review_support/figure3_margin_quartile_source_r21.csv",
    "review_support/margin_quartile_integrity_r21.csv",
    "review_support/normalized_output_error_summary_r21.csv",
    "review_support/normalized_output_error_by_subject_r21.csv",
    "review_support/raw_output_error_summary_r21.csv",
    "review_support/deployment/deployment_policy_definitions_r21.csv",
    "review_support/deployment/deployment_policy_admissible_sets_r21.csv",
    "review_support/deployment/deployment_policy_pareto_sets_r21.csv",
    "review_support/deployment/deployment_policy_selection_r21.csv",
    "review_support/deployment/deployment_policy_exclusions_r21.csv",
    "review_support/deployment/deployment_policy_sensitivity_r21.csv",
)
LAB_NOTE = "docs/results/on_device_resource_profile.md"
TABLE5_BACKEND_FILES = (
    "cca", "ccnn_f32", "ccnn_int8", "convca", "eegnet_f32", "eegnet_int8",
    "fbcca", "fbtcnn_f32", "fbtcnn_int8", "msi", "trca",
)
UART_PROFILE_NAMES = {"convca": "convca_f32"}
# Screenshot rows that show a local command line with user and workspace paths.
# The band is painted over in the public copy; all pixels below it are checked unchanged.
PNG_REDACTIONS = {"eegnet_f32.png": 40}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def root_variants(root: Path) -> list[bytes]:
    text = str(root.resolve())
    win = text.replace("/", "\\")
    variants = {win, win.replace("\\", "\\\\"), win.replace("\\", "/")}
    return sorted((v.encode() for v in variants), key=len, reverse=True)


class Sanitizer:
    def __init__(self, dev_root: Path) -> None:
        self.rules: list[tuple[str, re.Pattern[bytes], bytes]] = []
        raw_dir = dev_root.joinpath(*RAW_DATA_SUBDIR)
        for variant in root_variants(raw_dir):
            self.rules.append(("raw_data_dir", re.compile(re.escape(variant), re.I), PLACEHOLDER_RAW))
        for variant in root_variants(dev_root):
            self.rules.append(("dev_repo_root", re.compile(re.escape(variant), re.I), PLACEHOLDER_DEV))
        self.rules.append(("private_ipv4", PRIVATE_IPV4, PLACEHOLDER_IP))
        self.rules.append(("mac_address", MAC, PLACEHOLDER_MAC))

    def apply(self, data: bytes) -> tuple[bytes, dict[str, int]]:
        counts: dict[str, int] = {}
        for name, pattern, replacement in self.rules:
            data, n = pattern.subn(replacement, data)
            if n:
                counts[name] = counts.get(name, 0) + n
        return data, counts


def residual_private(data: bytes) -> list[str]:
    found = [m.group(0).decode("utf-8", "replace") for m in RESIDUAL_ABS_PATH.finditer(data)]
    found += [m.group(0).decode() for m in PRIVATE_IPV4.finditer(data)]
    found += [m.group(0).decode() for m in MAC.finditer(data)]
    return found


def wifi_credentials(sdkconfig: Path) -> list[bytes]:
    """Lower-cased SSID/password of the private sdkconfig; held in memory only, never logged."""
    if not sdkconfig.is_file():
        raise FileNotFoundError(f"{sdkconfig.name} of the development replay firmware is required")
    text = sdkconfig.read_text(encoding="utf-8")
    values = []
    for key in ("SSID", "PASSWORD"):
        match = re.search(rf'^CONFIG_REPLAY_WIFI_{key}="([^"]*)"', text, re.M)
        if match and len(match.group(1)) >= 4:
            values.append(match.group(1).lower().encode("utf-8"))
    return values


def check_structure(original: bytes, public: bytes, suffix: str) -> None:
    if suffix == ".csv":
        a = list(csv.reader(io.StringIO(original.decode("utf-8"))))
        b = list(csv.reader(io.StringIO(public.decode("utf-8"))))
        if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
            raise ValueError("CSV shape changed during sanitization")
        for x, y in zip(a, b):
            for u, v in zip(x, y):
                if u != v and b"<" not in v.encode():
                    raise ValueError("CSV cell changed without a placeholder")
    elif suffix == ".json":
        a, b = json.loads(original), json.loads(public)
        if set(a) != set(b):
            raise ValueError("JSON keys changed during sanitization")
    elif suffix == ".jsonl":
        a, b = original.splitlines(), public.splitlines()
        if len(a) != len(b):
            raise ValueError("JSONL line count changed")
        for x, y in zip(a, b):
            if x.strip() and set(json.loads(x)) != set(json.loads(y)):
                raise ValueError("JSONL keys changed during sanitization")


def png_text_chunks(data: bytes) -> list[str]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    chunks, pos = [], 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind in (b"tEXt", b"iTXt", b"zTXt", b"eXIf"):
            chunks.append(kind.decode())
        pos += 12 + length
    return chunks


def redact_top_band(data: bytes, rows: int) -> bytes:
    from PIL import Image, ImageChops, ImageDraw

    original = Image.open(io.BytesIO(data))
    original.load()
    redacted = original.copy()
    draw = ImageDraw.Draw(redacted)
    draw.rectangle((0, 0, redacted.width, rows - 1), fill=(30, 30, 30, 255))
    draw.text((6, 12), "[local build command and paths redacted]", fill=(200, 200, 200, 255))
    box = (0, rows, original.width, original.height)
    if ImageChops.difference(original.crop(box), redacted.crop(box)).getbbox() is not None:
        raise ValueError("redaction changed pixels outside the redacted band")
    buffer = io.BytesIO()
    redacted.save(buffer, format="PNG")
    return buffer.getvalue()


class Builder:
    def __init__(self, dev_root: Path, resource_root: Path, out: Path) -> None:
        self.dev_root = dev_root
        self.resource_root = resource_root
        self.out = out
        self.sanitizer = Sanitizer(dev_root)
        freeze = json.loads((dev_root / "docs" / "SUBMITTED_ARTIFACTS.sha256.json").read_text())
        self.freeze_commit = freeze["git_commit"]
        self.freeze_generated_at = freeze["generated_at"]
        self.freeze = {a["path"]: a for a in freeze["artifacts"]}
        self.entries: list[dict] = []
        self.credentials = wifi_credentials(dev_root / "embedded" / "replay_fw" / "sdkconfig")

    def _check_credentials(self, source_label: str, public: bytes) -> None:
        lowered = public.lower()
        if any(value in lowered for value in self.credentials):
            raise ValueError(f"{source_label}: development Wi-Fi credential found in public copy")

    def _write(self, public_rel: str, data: bytes) -> Path:
        path = self.out / public_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def copy_text(self, src: Path, source_label: str, public_rel: str, *, frozen_rel: str | None) -> None:
        stored = src.read_bytes()
        original_sha = sha256_bytes(stored)
        utf16 = stored[:2] in (b"\xff\xfe", b"\xfe\xff")
        original = stored.decode("utf-16").encode("utf-8") if utf16 else stored
        entry = {"source": source_label, "source_sha256": original_sha}
        if frozen_rel is not None:
            frozen = self.freeze[frozen_rel]
            if frozen["sha256"] != original_sha:
                raise ValueError(f"{frozen_rel}: SHA-256 differs from the development freeze")
            entry["freeze_class"] = frozen["class"]
            entry["freeze_sha256_verified"] = True
        public, counts = self.sanitizer.apply(original)
        residual = residual_private(public)
        if residual:
            raise ValueError(f"{source_label}: private strings remain after sanitization: {residual[:5]}")
        check_structure(original, public, src.suffix.lower())
        self._check_credentials(source_label, public)
        if utf16:
            public = public.decode("utf-8").encode("utf-16")
        self._write(public_rel, public)
        entry.update({
            "public_path": public_rel,
            "public_sha256": sha256_bytes(public),
            "transformed": public != stored,
            "replacements": counts,
            "encoding": "utf-16" if utf16 else "utf-8",
        })
        self.entries.append(entry)

    def copy_binary(self, src: Path, source_label: str, public_rel: str) -> None:
        data = src.read_bytes()
        text_chunks = png_text_chunks(data) if src.suffix.lower() == ".png" else []
        if text_chunks:
            raise ValueError(f"{source_label}: PNG carries metadata chunks {text_chunks}; review before release")
        public, replacements = data, {}
        band = PNG_REDACTIONS.get(src.name)
        if band:
            public = redact_top_band(data, band)
            replacements = {"png_top_band_redacted_rows": band}
        self._write(public_rel, public)
        self.entries.append({
            "source": source_label,
            "source_sha256": sha256_bytes(data),
            "public_path": public_rel,
            "public_sha256": sha256_bytes(public),
            "transformed": public != data,
            "replacements": replacements,
            "png_metadata_chunks": text_chunks,
        })

    def build(self) -> None:
        for rel, item in sorted(self.freeze.items()):
            if item["class"] == "raw_device_evidence" and rel.startswith(STRICT_REPLAY_PREFIX):
                public_rel = public_replay_rel(rel)
            elif item["class"] == "derived_fidelity" and rel.startswith(FIDELITY_PREFIX):
                public_rel = "fidelity/" + rel[len(FIDELITY_PREFIX):]
            else:
                continue
            self.copy_text(self.dev_root / rel, "<DEV_REPO>/" + rel, public_rel, frozen_rel=rel)

        for rel in R21_OUTPUTS:
            name = rel.split("/")[-1]
            self.copy_text(
                self.dev_root / rel, "<DEV_REPO>/" + rel,
                "submission_analysis/" + ANALYSIS_PUBLIC_NAMES[name], frozen_rel=None,
            )

        self.copy_text(self.dev_root / LAB_NOTE, "<DEV_REPO>/" + LAB_NOTE,
                       "table5/lab_note/on_device_resource_profile.md", frozen_rel=None)
        log_dir = self.resource_root / "peak_runtime_log"
        for name in TABLE5_BACKEND_FILES:
            self.copy_binary(log_dir / f"{name}.png", f"<RESOURCE_LOG_DIR>/{name}.png",
                             f"table5/idf_size_screenshots/{name}.png")
            self.copy_text(log_dir / f"uart_{name}.log", f"<RESOURCE_LOG_DIR>/uart_{name}.log",
                           f"table5/uart_replay_sessions/uart_{name}.log", frozen_rel=None)
            profile = f"uart_{UART_PROFILE_NAMES.get(name, name)}_profile.txt"
            self.copy_text(log_dir / profile, f"<RESOURCE_LOG_DIR>/{profile}",
                           f"table5/uart_replay_sessions/{profile}", frozen_rel=None)
        self.copy_text(self.resource_root / "tensor_arenatxt.txt", "<RESOURCE_ROOT>/tensor_arenatxt.txt",
                       "table5/smoke_arena/tensor_arenatxt.txt", frozen_rel=None)

    def write_map(self) -> None:
        totals: dict[str, int] = {}
        for entry in self.entries:
            for rule, n in entry["replacements"].items():
                totals[rule] = totals.get(rule, 0) + n
        payload = {
            "description": (
                "Original-to-public SHA-256 mapping for paper_reproduction/evidence. "
                "Originals are retained unchanged in the authors' private archive."
            ),
            "development_freeze": {
                "manifest": "<DEV_REPO>/docs/SUBMITTED_ARTIFACTS.sha256.json",
                "git_commit": self.freeze_commit,
                "generated_at": self.freeze_generated_at,
            },
            "rules": {
                "raw_data_dir": "private 12JFPM raw-data directory -> <12JFPM_RAW_DATA_DIR>",
                "dev_repo_root": "private development-repository root -> <DEV_REPO>",
                "private_ipv4": "RFC 1918 IPv4 address -> <REDACTED_IP> (ports kept)",
                "mac_address": "MAC address -> <REDACTED_MAC>",
            },
            "replacement_totals": totals,
            "n_files": len(self.entries),
            "n_transformed": sum(e["transformed"] for e in self.entries),
            "n_freeze_verified": sum(bool(e.get("freeze_sha256_verified")) for e in self.entries),
            "files": sorted(self.entries, key=lambda e: e["public_path"]),
        }
        (self.out / "SANITIZATION_MAP.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-root", type=Path, required=True, help="private development repository")
    parser.add_argument("--resource-root", type=Path, required=True,
                        help="folder containing peak_runtime_log/ and tensor_arenatxt.txt")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "evidence")
    parser.add_argument("--clean", action="store_true", help="remove generated evidence subfolders first")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    for root in (args.dev_root.resolve(), args.resource_root.resolve()):
        if out == root or root in out.parents:
            parser.error("--out must not be inside a private source folder")
    if args.clean:
        for sub in ("replay", "fidelity", "submission_analysis", "table5/idf_size_screenshots",
                    "table5/uart_replay_sessions", "table5/smoke_arena", "table5/lab_note"):
            shutil.rmtree(out / sub, ignore_errors=True)
    builder = Builder(args.dev_root.resolve(), args.resource_root.resolve(), out)
    builder.build()
    builder.write_map()
    print(f"wrote {len(builder.entries)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
