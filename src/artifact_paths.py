"""
Dataset-aware artifact path builders for future export, embedded generation, and replay.

Legacy 12JFPM strict benchmark paths remain in export.subject_paths — do not replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET_CHIP = "esp32s3"
DEFAULT_DEV_BOARD = "esp32s3_devkitc_1_v1_1"
DEFAULT_DEV_BOARD_DISPLAY = "ESP32-S3-DevKitC-1 v1.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize(*parts: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in parts:
        s = str(part).strip().replace("\\", "/").strip("/")
        if not s:
            raise ValueError("Path component must be non-empty")
        out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class ArtifactPathRequest:
    dataset: str
    contract: str
    backend: str
    quant: str
    split_or_subject: str
    target_chip: str = DEFAULT_TARGET_CHIP
    dev_board: str = DEFAULT_DEV_BOARD
    run_id: str | None = None

    def metadata(self) -> dict[str, str]:
        meta = {
            "dataset": self.dataset,
            "contract": self.contract,
            "backend": self.backend,
            "quant": self.quant,
            "split_or_subject": self.split_or_subject,
            "target_chip": self.target_chip,
            "dev_board": self.dev_board,
        }
        if self.run_id is not None:
            meta["run_id"] = self.run_id
        return meta

    @classmethod
    def from_mapping(cls, fields: dict[str, Any]) -> ArtifactPathRequest:
        required = ("dataset", "contract", "backend", "quant", "split_or_subject")
        missing = [k for k in required if not fields.get(k)]
        if missing:
            raise ValueError(f"ArtifactPathRequest missing fields: {missing}")
        return cls(
            dataset=str(fields["dataset"]).strip().lower(),
            contract=str(fields["contract"]).strip(),
            backend=str(fields["backend"]).strip(),
            quant=str(fields["quant"]).strip(),
            split_or_subject=str(fields["split_or_subject"]).strip(),
            target_chip=str(fields.get("target_chip", DEFAULT_TARGET_CHIP)).strip(),
            dev_board=str(fields.get("dev_board", DEFAULT_DEV_BOARD)).strip(),
            run_id=str(fields["run_id"]).strip() if fields.get("run_id") else None,
        )


def checkpoint_dir(
    dataset: str,
    contract: str,
    backend: str,
    split_or_subject: str,
    *,
    repo: Path | None = None,
) -> Path:
    """checkpoints/<dataset>/<contract>/<backend>/<split_or_subject>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, sp = _sanitize(dataset, contract, backend, split_or_subject)
    return root / "checkpoints" / ds / cn / be / sp


def export_artifact_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    split_or_subject: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/<backend>/<quant>/<split_or_subject>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, dq, sp = _sanitize(dataset, contract, backend, quant, split_or_subject)
    return root / "export" / "artifacts" / ds / cn / be / dq / sp


def embedded_generated_dir(
    target_chip: str,
    dev_board: str,
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    *,
    repo: Path | None = None,
) -> Path:
    """embedded/generated/<target_chip>/<dev_board>/<dataset>/<contract>/<backend>/<quant>/"""
    root = repo_root() if repo is None else Path(repo)
    chip, board, ds, cn, be, dq = _sanitize(
        target_chip, dev_board, dataset, contract, backend, quant,
    )
    return root / "embedded" / "generated" / chip / board / ds / cn / be / dq


def device_replay_results_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    target_chip: str,
    dev_board: str,
    run_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/device_replay/<dataset>/<contract>/<backend>/<quant>/<chip>/<board>/<run_id>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, dq, chip, board, rid = _sanitize(
        dataset, contract, backend, quant, target_chip, dev_board, run_id,
    )
    return root / "results" / "device_replay" / ds / cn / be / dq / chip / board / rid


def device_replay_logs_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    target_chip: str,
    dev_board: str,
    run_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """logs/device_replay/<dataset>/<contract>/<backend>/<quant>/<chip>/<board>/<run_id>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, dq, chip, board, rid = _sanitize(
        dataset, contract, backend, quant, target_chip, dev_board, run_id,
    )
    return root / "logs" / "device_replay" / ds / cn / be / dq / chip / board / rid


def build_paths(req: ArtifactPathRequest, *, repo: Path | None = None) -> dict[str, Path]:
    """Build all canonical artifact paths for a single request."""
    kwargs = {"repo": repo}
    out = {
        "checkpoint": checkpoint_dir(
            req.dataset, req.contract, req.backend, req.split_or_subject, **kwargs,
        ),
        "export": export_artifact_dir(
            req.dataset, req.contract, req.backend, req.quant, req.split_or_subject, **kwargs,
        ),
        "embedded": embedded_generated_dir(
            req.target_chip, req.dev_board,
            req.dataset, req.contract, req.backend, req.quant,
            **kwargs,
        ),
    }
    if req.run_id:
        out["device_replay_results"] = device_replay_results_dir(
            req.dataset, req.contract, req.backend, req.quant,
            req.target_chip, req.dev_board, req.run_id, **kwargs,
        )
        out["device_replay_logs"] = device_replay_logs_dir(
            req.dataset, req.contract, req.backend, req.quant,
            req.target_chip, req.dev_board, req.run_id, **kwargs,
        )
    return out


def host_results_dir(
    dataset: str,
    contract: str,
    backend: str,
    split_or_subset: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/host/<dataset>/<contract>/<backend>/<split>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, sp = _sanitize(dataset, contract, backend, split_or_subset)
    return root / "results" / "host" / ds / cn / be / sp


def host_baseline_results_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    subset: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/host_baseline/<dataset>/<contract>/<backend>/<quant>/<subset>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, dq, sub = _sanitize(dataset, contract, backend, quant, subset)
    return root / "results" / "host_baseline" / ds / cn / be / dq / sub


def host_baseline_smoke_dir(
    dataset: str,
    contract: str,
    subset: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/host_baseline/<dataset>/<contract>/smoke/<subset>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, sub = _sanitize(dataset, contract, subset)
    return root / "results" / "host_baseline" / ds / cn / "smoke" / sub


def export_contract_dir(
    dataset: str,
    contract: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn = _sanitize(dataset, contract)
    return root / "export" / "artifacts" / ds / cn


def export_manifest_path(
    dataset: str,
    contract: str,
    manifest_name: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/<manifest_name>.json"""
    root = repo_root() if repo is None else Path(repo)
    name = manifest_name.strip()
    if not name.endswith(".json"):
        name = f"{name}.json"
    return export_contract_dir(dataset, contract, repo=root) / name


def export_preflight_dir(
    dataset: str,
    contract: str,
    target_chip: str,
    dev_board: str,
    subset: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/preflight/<chip>/<board>/<subset>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, chip, board, sub = _sanitize(dataset, contract, target_chip, dev_board, subset)
    return export_contract_dir(ds, cn, repo=root) / "preflight" / chip / board / sub


def export_reference_vectors_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    subset: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/reference_vectors/<backend>/<quant>/<subset>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be, dq, sub = _sanitize(dataset, contract, backend, quant, subset)
    return export_contract_dir(ds, cn, repo=root) / "reference_vectors" / be / dq / sub


def embedded_preflight_metadata_path(
    target_chip: str,
    dev_board: str,
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    *,
    repo: Path | None = None,
    header_name: str = "preflight_metadata.h",
) -> Path:
    """embedded/generated/<chip>/<board>/<dataset>/<contract>/<backend>/<quant>/<header>"""
    return embedded_generated_dir(
        target_chip, dev_board, dataset, contract, backend, quant, repo=repo,
    ) / header_name


def external_wearable_selection_dir(
    *,
    repo: Path | None = None,
    selection_id: str = "subject_selection_wet_cca_native250",
) -> Path:
    """results/external_wearable/<selection_id>/"""
    root = repo_root() if repo is None else Path(repo)
    sid = _sanitize(selection_id)[0]
    return root / "results" / "external_wearable" / sid


def legacy_strict_12jfpm_export_dir(
    model: str,
    subject: str = "s1",
    *,
    int8: bool = False,
    repo: Path | None = None,
) -> Path:
    """Delegate to export.subject_paths.export_dir (unchanged strict 12JFPM path)."""
    from export.subject_paths import export_dir

    return export_dir(model, subject, int8=int8, repo=repo)


def legacy_strict_12jfpm_checkpoint_dir(
    model: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Delegate to export.subject_paths.checkpoint_dir (unchanged strict 12JFPM path)."""
    from export.subject_paths import checkpoint_dir as legacy_checkpoint_dir

    return legacy_checkpoint_dir(model, repo=repo)


def wearable_loso_split_name(test_subject: str) -> str:
    """selected10_loso_test_<SUBJECT> split token for wearable DL LOSO."""
    sid = str(test_subject).strip().upper()
    return f"selected10_loso_test_{sid}"


def training_plan_dir(
    dataset: str,
    contract: str,
    relative: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/training_plan/<dataset>/<contract>/<relative>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, rel = _sanitize(dataset, contract, relative)
    return root / "results" / "training_plan" / ds / cn / rel


def wearable_loso_checkpoint_dir(
    dataset: str,
    contract: str,
    backend: str,
    test_subject: str,
    *,
    repo: Path | None = None,
) -> Path:
    """checkpoints/<dataset>/<contract>/<backend>/selected10_loso_test_<SUBJECT>/"""
    split = wearable_loso_split_name(test_subject)
    return checkpoint_dir(dataset, contract, backend, split, repo=repo)


def wearable_loso_checkpoint_metadata_path(
    dataset: str,
    contract: str,
    backend: str,
    test_subject: str,
    *,
    repo: Path | None = None,
    metadata_name: str = "fold_metadata.json",
) -> Path:
    """Fold planning metadata under dataset-aware checkpoint path."""
    return wearable_loso_checkpoint_dir(
        dataset, contract, backend, test_subject, repo=repo,
    ) / metadata_name


def wearable_loso_checkpoint_best_path(
    dataset: str,
    contract: str,
    backend: str,
    test_subject: str,
    *,
    repo: Path | None = None,
    checkpoint_name: str = "best.pt",
) -> Path:
    """Future best checkpoint path for wearable DL LOSO fold."""
    return wearable_loso_checkpoint_dir(
        dataset, contract, backend, test_subject, repo=repo,
    ) / checkpoint_name


def wearable_train_smoke_dir(
    dataset: str,
    contract: str,
    backend: str,
    test_subject: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/training_plan/<dataset>/<contract>/train_smoke/<backend>/selected10_loso_test_<SUBJECT>/"""
    split = wearable_loso_split_name(test_subject)
    return training_plan_dir(dataset, contract, f"train_smoke/{backend}/{split}", repo=repo)


def wearable_dl_loso_results_dir(
    dataset: str,
    contract: str,
    backend: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/dl_loso/<dataset>/<contract>/<backend>/selected10_loso/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be = _sanitize(dataset, contract, backend)
    return root / "results" / "dl_loso" / ds / cn / be / "selected10_loso"


def wearable_dl_loso_contract_dir(
    dataset: str,
    contract: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/dl_loso/<dataset>/<contract>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn = _sanitize(dataset, contract)
    return root / "results" / "dl_loso" / ds / cn


def wearable_classical_loso_results_dir(
    dataset: str,
    contract: str,
    method: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/classical_loso/<dataset>/<contract>/<method>/selected10_loso/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, me = _sanitize(dataset, contract, method)
    return root / "results" / "classical_loso" / ds / cn / me / "selected10_loso"


def wearable_classical_loso_contract_dir(
    dataset: str,
    contract: str,
    *,
    repo: Path | None = None,
) -> Path:
    """results/classical_loso/<dataset>/<contract>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn = _sanitize(dataset, contract)
    return root / "results" / "classical_loso" / ds / cn


def wearable_loso_checkpoint_final_path(
    dataset: str,
    contract: str,
    backend: str,
    test_subject: str,
    *,
    repo: Path | None = None,
    checkpoint_name: str = "final.pt",
) -> Path:
    """Official fixed-epoch checkpoint for wearable DL LOSO fold."""
    return wearable_loso_checkpoint_dir(
        dataset, contract, backend, test_subject, repo=repo,
    ) / checkpoint_name


def wearable_dl_export_dir(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    test_subject: str,
    *,
    repo: Path | None = None,
) -> Path:
    """export/artifacts/<dataset>/<contract>/<backend>/<quant>/selected10_loso_test_<SUBJECT>/"""
    ds = str(dataset).strip().lower()
    cn = str(contract).strip()
    if ds != "wearable_ssvep":
        raise ValueError(f"wearable_dl_export_dir requires dataset wearable_ssvep, got {dataset!r}")
    if not cn or cn != "native250_wet_T500_FS250_C8_N12":
        raise ValueError(
            f"wearable_dl_export_dir requires contract native250_wet_T500_FS250_C8_N12, got {contract!r}"
        )
    cn_lc = cn.lower()
    if "fs256" in cn_lc and "resampled256" not in cn_lc:
        raise ValueError(f"wearable_dl_export_dir rejects Fs=256 contract name: {contract!r}")
    if "t512" in cn_lc or "fs256_t512" in cn_lc:
        raise ValueError(f"wearable_dl_export_dir rejects T=512 contract name: {contract!r}")

    split = wearable_loso_split_name(test_subject)
    out = export_artifact_dir(dataset, contract, backend, quant, split, repo=repo)
    text = out.as_posix().lower()
    if "export/strict_loso_inner_val" in text:
        raise ValueError(f"wearable_dl_export_dir must not use strict_loso_inner_val: {out}")
    if "/export/12jfpm_" in text:
        raise ValueError(f"wearable_dl_export_dir must not use legacy 12JFPM export path: {out}")
    return out


def wearable_dl_export_metadata_path(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    test_subject: str,
    *,
    repo: Path | None = None,
    metadata_name: str = "export_metadata.json",
) -> Path:
    """export/artifacts/.../export_metadata.json"""
    return wearable_dl_export_dir(
        dataset, contract, backend, quant, test_subject, repo=repo,
    ) / metadata_name


def processed_data_dir(
    processed_path: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Resolve contract processed_path (repo-relative) to the containing directory.

    Contract files may use per-subject templates such as ``.../wet/Sxxx.npz``.
    """
    root = repo_root() if repo is None else Path(repo)
    rel = str(processed_path).strip().replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("processed_path must be non-empty")
    if rel.lower().endswith(".npz"):
        rel = str(Path(rel).parent).replace("\\", "/")
    return root / rel


def wearable_dl_training_logs_dir(
    dataset: str,
    contract: str,
    backend: str,
    *,
    repo: Path | None = None,
) -> Path:
    """logs/dl_loso/<dataset>/<contract>/<backend>/"""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be = _sanitize(dataset, contract, backend)
    return root / "logs" / "dl_loso" / ds / cn / be


def wearable_dl_backend_manifest_path(
    dataset: str,
    contract: str,
    backend: str,
    *,
    repo: Path | None = None,
    manifest_name: str = "run_manifest.json",
) -> Path:
    """Per-backend training manifest under results/dl_loso/<dataset>/<contract>/<backend>/."""
    root = repo_root() if repo is None else Path(repo)
    ds, cn, be = _sanitize(dataset, contract, backend)
    name = manifest_name.strip()
    if not name.endswith(".json"):
        name = f"{name}.json"
    return root / "results" / "dl_loso" / ds / cn / be / name


def wearable_dl_export_model_path(
    dataset: str,
    contract: str,
    backend: str,
    quant: str,
    test_subject: str,
    *,
    repo: Path | None = None,
    model_name: str = "model.tflite",
) -> Path:
    """Exported model file under wearable DL export directory."""
    return wearable_dl_export_dir(
        dataset, contract, backend, quant, test_subject, repo=repo,
    ) / model_name
