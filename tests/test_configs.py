"""Validate public configs/ inventory and loaders."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# Keys consumed by src/training_entry/run_dl_loso_12jfpm.py + training engine.
TRAINING_ALLOWED_TOP = {"seed", "data", "model", "train", "eval", "loso"}
TRAINING_ALLOWED_TRAIN = {
    "batch_size", "epochs", "lr", "weight_decay", "num_workers", "early_stopping_patience",
}
UNUSED_TRAINING_KEYS = {
    "data_augmentation", "class_weighting", "optimizer", "loss",
    "lr_scheduler", "checkpoint_rule", "early_stopping_metric", "split_seed_rule",
}

DOC_AND_SCRIPT_TEXT_FILES = [
    ROOT / "README.md",
        ROOT / "docs" / "training-and-export.md",
    ROOT / "docs" / "firmware-and-replay.md",
    ROOT / "docs" / "dataset.md",
        *sorted((ROOT / "scripts").glob("*.py")),
    ROOT / "src" / "training_entry" / "run_dl_loso_12jfpm.py",
    ROOT / "src" / "training_entry" / "run_classical_loso_12jfpm.py",
    ROOT / "src" / "training_entry" / "run_preprocess_12jfpm.py",
    ROOT / "src" / "contracts.py",
]


def _all_yaml() -> list[Path]:
    return sorted(CONFIGS.rglob("*.yaml"))


def _reference_blob() -> str:
    parts = []
    for path in DOC_AND_SCRIPT_TEXT_FILES:
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    # also scan scripts tree string literals for config paths
    for path in (ROOT / "src").rglob("*.py"):
        if "configs/" in path.read_text(encoding="utf-8", errors="ignore"):
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def test_every_config_is_referenced():
    blob = _reference_blob().replace("\\", "/")
    missing = []
    for path in _all_yaml():
        rel = path.relative_to(ROOT).as_posix()
        if rel not in blob and path.name not in blob:
            missing.append(rel)
    assert not missing, f"Unreferenced configs: {missing}"


def test_no_identical_duplicate_pairs():
    files = _all_yaml()
    by_hash: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        # compare effective YAML content
        data = yaml.safe_load(text)
        key = yaml.safe_dump(data, sort_keys=True)
        by_hash.setdefault(key, []).append(path.relative_to(ROOT).as_posix())
    dups = {k: v for k, v in by_hash.items() if len(v) > 1}
    assert not dups, f"Identical config pairs: {dups}"


def test_training_configs_have_no_unused_keys():
    for path in sorted((CONFIGS / "training").glob("*_12jfpm_strict_loso.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(doc) <= TRAINING_ALLOWED_TOP, path
        assert UNUSED_TRAINING_KEYS.isdisjoint(doc), path
        assert set(doc.get("train", {})) <= TRAINING_ALLOWED_TRAIN, path


def test_required_configs_load():
    from src.contracts import load_12jfpm_contract, contract_hardware_metadata

    c = load_12jfpm_contract(repo=ROOT)
    assert c.dataset_name == "12jfpm"
    assert c.window_samples == 512
    hw = contract_hardware_metadata()
    assert hw["dev_board"] == "esp32s3_devkitc_1_v1_1"

    for path in sorted((CONFIGS / "training").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "model" in doc and "train" in doc and "loso" in doc

    for path in sorted((CONFIGS / "analysis").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "method" in doc


def test_classical_configs_match_dataclass_defaults():
    from src.evaluation.cca_12jfpm import CCAConfig
    from src.evaluation.fbcca_12jfpm import FBCCAConfig
    from src.evaluation.msi_12jfpm import MSIConfig
    from src.evaluation.trca_12jfpm import TRCAConfig
    from src.training_entry.run_classical_loso_12jfpm import (
        _cca_config, _fbcca_config, _msi_config, _trca_config, _load_yaml, _CONFIG_MAP,
    )

    assert _cca_config(_load_yaml(_CONFIG_MAP["cca"])) == CCAConfig()
    assert _fbcca_config(_load_yaml(_CONFIG_MAP["fbcca"])) == FBCCAConfig()
    assert _msi_config(_load_yaml(_CONFIG_MAP["msi"])) == MSIConfig()
    assert _trca_config(_load_yaml(_CONFIG_MAP["trca"])) == TRCAConfig()


def test_no_legacy_short_training_aliases():
    legacy = list((CONFIGS / "training").glob("*_loso_12jfpm.yaml"))
    assert not legacy
    assert not list((CONFIGS / "analysis").glob("*_benchmark.yaml"))
    assert not (CONFIGS / "analysis" / "classical_12jfpm").exists()
