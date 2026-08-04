from pathlib import Path
import json

def test_class_order():
    root = Path(__file__).resolve().parents[1]
    man = json.loads((root / "manifests/loso/fold01.json").read_text(encoding="utf-8"))
    assert man["class_order"] == list(range(12))
