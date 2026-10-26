import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_reproduction" / "reproduce.py"


def test_paper_reproduction_passes(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--no-figures", "--out", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    report = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    failed = [c["check"] for c in report["checks"] if not c["passed"]]
    assert not failed
    assert len(report["checks"]) >= 23
    assert (tmp_path / "tables" / "table2_realized_backends.csv").is_file()
