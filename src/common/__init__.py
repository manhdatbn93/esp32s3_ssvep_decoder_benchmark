"""Backward-compatible re-exports. Prefer src.contracts and src.artifact_paths."""

from src.artifact_paths import (
    DEFAULT_DEV_BOARD,
    DEFAULT_DEV_BOARD_DISPLAY,
    DEFAULT_TARGET_CHIP,
    ArtifactPathRequest,
    build_paths,
    checkpoint_dir as training_checkpoint_dir,
    device_replay_logs_dir,
    device_replay_results_dir,
    embedded_generated_dir,
    export_artifact_dir,
    external_wearable_selection_dir,
    host_results_dir,
    legacy_strict_12jfpm_checkpoint_dir,
    legacy_strict_12jfpm_export_dir,
    repo_root,
)
from src.contracts import Contract, contract_hardware_metadata, load_contract

__all__ = [
    "ArtifactPathRequest",
    "Contract",
    "DEFAULT_DEV_BOARD",
    "DEFAULT_DEV_BOARD_DISPLAY",
    "DEFAULT_TARGET_CHIP",
    "build_paths",
    "contract_hardware_metadata",
    "device_replay_logs_dir",
    "device_replay_results_dir",
    "embedded_generated_dir",
    "export_artifact_dir",
    "external_wearable_selection_dir",
    "host_results_dir",
    "legacy_strict_12jfpm_checkpoint_dir",
    "legacy_strict_12jfpm_export_dir",
    "load_contract",
    "repo_root",
    "training_checkpoint_dir",
]
