"""Deprecated: use src.contracts. Kept for backward compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.contracts import Contract, contract_hardware_metadata, load_contract, load_wearable_native250_wet_contract


@dataclass(frozen=True)
class ExperimentContract:
    """Thin wrapper around Contract with optional runtime metadata fields."""

    dataset_name: str
    contract_name: str
    fs_hz: float
    window_samples: int
    n_channels: int
    n_classes: int
    target_chip: str | None = None
    dev_board: str | None = None
    dev_board_display: str | None = None
    backend_name: str | None = None
    dtype_or_quant: str | None = None
    split_fold: str | None = None
    subject: str | None = None
    subset_id: str | None = None
    electrode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_contract(cls, contract: Contract, **overrides: Any) -> ExperimentContract:
        hw = contract_hardware_metadata()
        return cls(
            dataset_name=contract.dataset_name,
            contract_name=contract.contract_name,
            fs_hz=contract.fs_hz,
            window_samples=contract.window_samples,
            n_channels=contract.channels,
            n_classes=contract.n_classes,
            target_chip=overrides.get("target_chip", hw["target_chip"]),
            dev_board=overrides.get("dev_board", hw["dev_board"]),
            dev_board_display=overrides.get("dev_board_display", hw["dev_board_display"]),
            electrode=contract.electrode,
            subset_id=overrides.get("subset_id"),
            extra=overrides.get("extra", {}),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "contract_name": self.contract_name,
            "fs_hz": self.fs_hz,
            "window_samples": self.window_samples,
            "n_channels": self.n_channels,
            "n_classes": self.n_classes,
        }
        for key in (
            "target_chip", "dev_board", "dev_board_display", "backend_name",
            "dtype_or_quant", "split_fold", "subject", "subset_id", "electrode",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        out.update(self.extra)
        return out

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentContract:
        import yaml

        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        contract = load_contract(doc["dataset_name"], doc["contract_name"])
        inst = cls.from_contract(contract)
        ranking = doc.get("ranking", {})
        extra = {
            k: v
            for k, v in {
                "ranking_baseline": doc.get("ranking_baseline") or ranking.get("baseline_id"),
                "display_baseline": doc.get("display_baseline") or ranking.get("baseline_display"),
                "selection_formula": doc.get("selection_formula") or ranking.get("selection_formula"),
                "qc_status": doc.get("qc_status"),
            }.items()
            if v is not None
        }
        return cls(
            dataset_name=inst.dataset_name,
            contract_name=inst.contract_name,
            fs_hz=inst.fs_hz,
            window_samples=inst.window_samples,
            n_channels=inst.n_channels,
            n_classes=inst.n_classes,
            target_chip=doc.get("target_chip", inst.target_chip),
            dev_board=doc.get("dev_board", inst.dev_board),
            dev_board_display=doc.get("dev_board_display", inst.dev_board_display),
            backend_name=doc.get("backend_name") or ranking.get("baseline_id"),
            electrode=doc.get("electrode") if isinstance(doc.get("electrode"), str) else contract.electrode,
            subset_id=doc.get("subset_id"),
            extra=extra,
        )


CONTRACT_12JFPM = ExperimentContract.from_contract(load_contract("12jfpm", "T512_FS256_C8_N12"))


def load_contract_wearable_native250_wet() -> ExperimentContract:
    """Optional wearable contract; not part of the public 12JFPM package."""
    return ExperimentContract.from_contract(
        load_wearable_native250_wet_contract(),
        subset_id="selected10_native250_wet",
    )


# Backward-compatible name; resolved only when the wearable YAML is present.
try:
    CONTRACT_WEARABLE_NATIVE250_WET = load_contract_wearable_native250_wet()
except Exception:  # noqa: BLE001 — optional outside this package
    CONTRACT_WEARABLE_NATIVE250_WET = None  # type: ignore[assignment]
