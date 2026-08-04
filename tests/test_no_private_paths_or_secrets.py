from pathlib import Path
import re

_SECRET_PARTS = ["Mr" + ".Bee", "1s2s3s4s", "shinytree081"]
SECRET_RE = re.compile("|".join(re.escape(p) for p in _SECRET_PARTS), re.I)

def test_no_wifi_secrets_in_firmware():
    root = Path(__file__).resolve().parents[1]
    skip_parts = {"build", "managed_components", "generated"}
    for path in (root / "firmware").rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.name in {"sdkconfig", "sdkconfig.old"}:
            continue
        if path.suffix.lower() in {".c", ".h", ".defaults"} or path.name == "Kconfig.projbuild":
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not SECRET_RE.search(text), path
