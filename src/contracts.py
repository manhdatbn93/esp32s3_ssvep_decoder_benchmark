"""
Dataset contract loader and validator.

Canonical contract definitions live in configs/contracts/*.yaml.
12JFPM strict benchmark paths remain in export.subject_paths (unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACTS_DIR_NAME = "contracts"

# Known contract_name aliases for lookup convenience.
_CONTRACT_ALIASES: dict[tuple[str, str], str] = {
    ("12jfpm", "12jfpm_fs256_T512_C8_N12"): "T512_FS256_C8_N12",
    ("wearable_ssvep", "wearable_ssvep_native250_wet_T500_FS250_C8_N12"): "native250_wet_T500_FS250_C8_N12",
    ("wearable_ssvep", "wearable_ssvep_resampled256_T512_FS256_C8_N12"): "resampled256_T512_FS256_C8_N12",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def contracts_dir(*, repo: Path | None = None) -> Path:
    root = repo_root() if repo is None else Path(repo)
    return root / "configs" / CONTRACTS_DIR_NAME


@dataclass(frozen=True)
class Contract:
    dataset_name: str
    display_name: str
    contract_name: str
    fs_hz: float
    window_samples: int
    channels: int
    n_classes: int
    crop_start: int
    crop_end: int
    resampled: bool
    processed_path: str
    notes: str
    electrode: str | None = None
    onset_sample_0: int | None = None
    source_file: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "display_name": self.display_name,
            "contract_name": self.contract_name,
            "fs_hz": self.fs_hz,
            "window_samples": self.window_samples,
            "channels": self.channels,
            "n_classes": self.n_classes,
            "crop_start": self.crop_start,
            "crop_end": self.crop_end,
            "electrode": self.electrode,
            "resampled": self.resampled,
            "processed_path": self.processed_path,
            "notes": self.notes,
            "onset_sample_0": self.onset_sample_0,
            "source_file": self.source_file,
        }


def _normalize_contract_name(dataset_name: str, contract_name: str) -> str:
    key = (dataset_name.strip().lower(), contract_name.strip())
    return _CONTRACT_ALIASES.get(key, contract_name.strip())


def _contract_from_dict(doc: dict[str, Any], *, source_file: Path | None = None) -> Contract:
    electrode = doc.get("electrode")
    if electrode is not None and str(electrode).lower() in ("null", "none", ""):
        electrode = None
    elif electrode is not None:
        electrode = str(electrode)

    return Contract(
        dataset_name=str(doc["dataset_name"]).strip().lower(),
        display_name=str(doc["display_name"]),
        contract_name=str(doc["contract_name"]).strip(),
        fs_hz=float(doc["fs_hz"]),
        window_samples=int(doc["window_samples"]),
        channels=int(doc["channels"]),
        n_classes=int(doc["n_classes"]),
        crop_start=int(doc["crop_start"]),
        crop_end=int(doc["crop_end"]),
        electrode=electrode,
        resampled=bool(doc["resampled"]),
        processed_path=str(doc["processed_path"]),
        notes=str(doc.get("notes", "")).strip(),
        onset_sample_0=int(doc["onset_sample_0"]) if doc.get("onset_sample_0") is not None else None,
        source_file=str(source_file) if source_file else None,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Contract YAML must be a mapping: {path}")
    return doc


def list_contract_files(*, repo: Path | None = None) -> list[Path]:
    root = contracts_dir(repo=repo)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.yaml"))


def _build_contract_index(*, repo: Path | None = None) -> dict[tuple[str, str], Contract]:
    index: dict[tuple[str, str], Contract] = {}
    for path in list_contract_files(repo=repo):
        doc = _load_yaml(path)
        contract = _contract_from_dict(doc, source_file=path)
        validate_contract_dict(contract.as_dict())
        key = (contract.dataset_name, contract.contract_name)
        if key in index:
            raise ValueError(f"Duplicate contract key {key!r} in {path}")
        index[key] = contract
    return index


def load_contract(
    dataset_name: str,
    contract_name: str,
    *,
    repo: Path | None = None,
) -> Contract:
    """Load and validate a contract by dataset_name + contract_name."""
    ds = dataset_name.strip().lower()
    cn = _normalize_contract_name(ds, contract_name)
    validate_dataset_contract_pair(ds, cn)
    index = _build_contract_index(repo=repo)
    key = (ds, cn)
    if key not in index:
        available = sorted(index.keys())
        raise KeyError(f"Unknown contract {key!r}. Available: {available}")
    return index[key]


def load_contract_from_file(path: Path | str) -> Contract:
    path = Path(path)
    doc = _load_yaml(path)
    contract = _contract_from_dict(doc, source_file=path)
    validate_contract_dict(contract.as_dict())
    return contract


def validate_dataset_contract_pair(dataset_name: str, contract_name: str) -> None:
    ds = dataset_name.strip().lower()
    cn = contract_name.strip()
    if not ds:
        raise ValueError("dataset_name must be non-empty")
    if not cn:
        raise ValueError("contract_name must be non-empty")
    if ds == "wearable_ssvep" and cn in ("wearable", "wearable_ssvep", "native250"):
        raise ValueError(
            f"Generic wearable contract name {cn!r} is not allowed; "
            "use an explicit contract_name such as native250_wet_T500_FS250_C8_N12"
        )


def validate_contract_dict(contract: dict[str, Any] | Contract) -> None:
    data = contract.as_dict() if isinstance(contract, Contract) else dict(contract)

    required = (
        "dataset_name", "display_name", "contract_name", "fs_hz", "window_samples",
        "channels", "n_classes", "crop_start", "crop_end", "resampled", "processed_path",
    )
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"Contract missing required fields: {missing}")

    validate_dataset_contract_pair(str(data["dataset_name"]), str(data["contract_name"]))

    ds = str(data["dataset_name"]).strip().lower()
    cn = str(data["contract_name"]).strip()
    fs = float(data["fs_hz"])
    t = int(data["window_samples"])
    c = int(data["channels"])
    n = int(data["n_classes"])
    crop_start = int(data["crop_start"])
    crop_end = int(data["crop_end"])
    resampled = bool(data["resampled"])

    if crop_end - crop_start != t:
        raise ValueError(
            f"Contract {cn}: crop window {crop_end - crop_start} != window_samples {t}"
        )

    if is_12jfpm_contract(data):
        if fs != 256 or t != 512 or c != 8 or n != 12:
            raise ValueError(f"12JFPM contract {cn} must be Fs=256, T=512, C=8, N=12")
        if resampled:
            raise ValueError(f"12JFPM contract {cn} must not be resampled")

    if is_native250_wearable(data):
        if fs != 250 or t != 500 or c != 8 or n != 12:
            raise ValueError(
                f"Wearable native250 contract {cn} must be Fs=250, T=500, C=8, N=12"
            )
        if resampled:
            raise ValueError(f"Wearable native250 contract {cn} must not be resampled")
        if crop_start != 125 or crop_end != 625:
            raise ValueError(f"Wearable native250 contract {cn} crop must be [125:625]")
        electrode = data.get("electrode")
        if electrode != "wet":
            raise ValueError(f"Wearable native250 contract {cn} electrode must be wet")
        pp = str(data["processed_path"])
        if "processed_native250/wet" not in pp.replace("\\", "/"):
            raise ValueError(
                f"Wearable native250 processed_path must point to processed_native250/wet/: {pp}"
            )

    if ds == "wearable_ssvep" and cn.startswith("resampled256"):
        if not resampled:
            raise ValueError(f"Wearable resampled256 contract {cn} must have resampled=true")
        if fs != 256 or t != 512:
            raise ValueError(f"Wearable resampled256 contract {cn} must be Fs=256, T=512")


def is_12jfpm_contract(contract: dict[str, Any] | Contract) -> bool:
    data = contract.as_dict() if isinstance(contract, Contract) else contract
    return str(data.get("dataset_name", "")).strip().lower() == "12jfpm"


def is_native250_wearable(contract: dict[str, Any] | Contract) -> bool:
    data = contract.as_dict() if isinstance(contract, Contract) else contract
    ds = str(data.get("dataset_name", "")).strip().lower()
    cn = str(data.get("contract_name", "")).strip()
    return ds == "wearable_ssvep" and cn == "native250_wet_T500_FS250_C8_N12"


def validate_fs(fs_hz: float, contract: dict[str, Any] | Contract) -> None:
    data = contract.as_dict() if isinstance(contract, Contract) else contract
    expected = float(data["fs_hz"])
    if abs(float(fs_hz) - expected) > 1e-6:
        raise ValueError(
            f"Fs mismatch for {data.get('contract_name')}: got {fs_hz}, expected {expected}"
        )


def validate_tensor_shape(X: np.ndarray, contract: dict[str, Any] | Contract) -> None:
    data = contract.as_dict() if isinstance(contract, Contract) else contract
    t = int(data["window_samples"])
    c = int(data["channels"])
    arr = np.asarray(X)
    if arr.ndim == 2:
        if arr.shape != (t, c):
            raise ValueError(f"Expected tensor shape ({t}, {c}), got {arr.shape}")
        return
    if arr.ndim == 3:
        if arr.shape[1:] != (t, c):
            raise ValueError(f"Expected tensor shape (N, {t}, {c}), got {arr.shape}")
        return
    raise ValueError(f"Expected 2D or 3D tensor, got ndim={arr.ndim} shape={arr.shape}")


def contract_hardware_metadata() -> dict[str, str]:
    """Hardware IDs from configs/firmware/esp32s3_devkitc_1_v1_1.yaml."""
    path = repo_root() / "configs" / "hardware" / "esp32s3_devkitc_1_v1_1.yaml"
    defaults = {
        "target_chip": "esp32s3",
        "dev_board": "esp32s3_devkitc_1_v1_1",
        "dev_board_display": "ESP32-S3-DevKitC-1 v1.1",
    }
    if not path.is_file():
        return defaults
    doc = _load_yaml(path)
    return {
        "target_chip": str(doc.get("target_chip", defaults["target_chip"])),
        "dev_board": str(doc.get("dev_board", defaults["dev_board"])),
        "dev_board_display": str(doc.get("dev_board_display", defaults["dev_board_display"])),
    }


# Convenience loaders for tests and scripts.
def load_12jfpm_contract(*, repo: Path | None = None) -> Contract:
    return load_contract("12jfpm", "T512_FS256_C8_N12", repo=repo)


def load_wearable_native250_wet_contract(*, repo: Path | None = None) -> Contract:
    return load_contract("wearable_ssvep", "native250_wet_T500_FS250_C8_N12", repo=repo)
