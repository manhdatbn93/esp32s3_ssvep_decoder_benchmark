from pathlib import Path
import json

def test_ten_folds_and_isolation():
    root = Path(__file__).resolve().parents[1]
    folds = sorted((root / "manifests/loso").glob("fold*.json"))
    assert len(folds) == 10
    for path in folds:
        man = json.loads(path.read_text(encoding="utf-8"))
        held = man["held_out_subject"]
        assert held not in man["outer_training_subjects"]
        opt = set(man["optimization_trial_ids"])
        val = set(man["validation_trial_ids"])
        test = set(man["test_trial_ids"])
        assert not opt & val
        assert not opt & test
        assert not val & test
        assert len(test) == 180
        assert len(man["class_order"]) == 12
