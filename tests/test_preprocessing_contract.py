from pathlib import Path
import json

def test_loso_manifest_preprocessing():
    root = Path(__file__).resolve().parents[1]
    man = json.loads((root / "manifests/loso/fold01.json").read_text(encoding="utf-8"))
    pre = man["preprocessing_configuration"]
    assert pre["slice"] == "38:550"
    assert pre["n_samples"] == 512
    assert pre["sampling_rate_hz"] == 256
    assert pre["normalization"] == "per_trial_per_channel_zscore"
