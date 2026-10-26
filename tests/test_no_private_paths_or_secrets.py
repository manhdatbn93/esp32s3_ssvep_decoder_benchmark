"""Public-tree hygiene: no Wi-Fi credentials, local paths, private IPs or MAC addresses.

Checks every file Git would publish (tracked plus untracked, not ignored). The test holds
no secret material: Wi-Fi settings are checked structurally against placeholders.
"""

from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

WIFI_PLACEHOLDERS = {"YOUR_WIFI_SSID", "YOUR_WIFI_PASSWORD"}
WIFI_ASSIGNMENT_RE = re.compile(
    r"""(?:CONFIG_)?REPLAY_WIFI_(?:SSID|PASSWORD)\b["']?\s*[=:]\s*["']([^"'\r\n]*)["']"""
)
KCONFIG_DEFAULT_RE = re.compile(r"""^\s*#?\s*default\s+"([^"]*)\"""", re.M)

PRIVATE_PATTERNS = {
    "windows_user_or_drive_path": re.compile(r"(?<![A-Za-z<])[A-Za-z]:[\\/]+(?:Users|WIP|Documents|OneDrive)\b", re.I),
    "posix_home_path": re.compile(r"(?<![\w.])/(?:home|Users)/(?!<)[A-Za-z0-9_.-]+/"),
    "private_ipv4": re.compile(
        r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])"
    ),
    "mac_address": re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])"),
}

SKIP_PARTS = {".git", "managed_components", "build", "generated", "__pycache__"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pth", ".tflite", ".npz", ".npy", ".bin", ".elf"}


def _public_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is required to list the public tree")
    paths = [ROOT / p for p in out.decode("utf-8").split("\0") if p]
    return [p for p in paths if p.is_file() and not SKIP_PARTS.intersection(p.relative_to(ROOT).parts)]


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def public_texts() -> dict[str, str]:
    texts = {}
    for path in _public_files():
        text = _read_text(path)
        if text is not None:
            texts[path.relative_to(ROOT).as_posix()] = text
    assert texts
    return texts


def test_wifi_credentials_are_placeholders(public_texts):
    found = {}
    for rel, text in public_texts.items():
        bad = [v for v in set(WIFI_ASSIGNMENT_RE.findall(text)) if v and v not in WIFI_PLACEHOLDERS]
        if bad:
            found[rel] = len(bad)
    assert not found, f"non-placeholder Wi-Fi values in {sorted(found)}"


def test_kconfig_wifi_defaults_are_placeholders():
    text = (ROOT / "firmware" / "replay_fw" / "main" / "Kconfig.projbuild").read_text(encoding="utf-8")
    for key in ("REPLAY_WIFI_SSID", "REPLAY_WIFI_PASSWORD"):
        block = re.search(rf"config {key}\b(.*?)(?=^\s*config |\Z)", text, re.S | re.M)
        assert block, key
        defaults = KCONFIG_DEFAULT_RE.findall(block.group(1))
        assert defaults and set(defaults) <= WIFI_PLACEHOLDERS, key


@pytest.mark.parametrize("name", sorted(PRIVATE_PATTERNS))
def test_no_private_metadata(public_texts, name):
    pattern = PRIVATE_PATTERNS[name]
    hits = {rel: len(pattern.findall(text)) for rel, text in public_texts.items() if pattern.search(text)}
    assert not hits, f"{name}: {hits}"
