# ESP32-S3 SSVEP Decoder Benchmark

Host and ESP32-S3 workflows for classical and neural steady-state visually evoked potential (SSVEP) decoders on the 12JFPM benchmark. The repository trains models, exports TFLite, evaluates them on the host, stages firmware inputs, and replays trials to an ESP32-S3. Retained checkpoints and models under `artifacts/reference/` stay read-only.

## Supported decoders

| Decoder | What it supports |
|---|---|
| CCA, FBCCA, MSI, TRCA | Host evaluation and firmware, with retained analytical bundles |
| CCNN, EEGNet, FBtCNN | Training, float32 and int8 TFLite export, host evaluation, and firmware |
| ConvCA | Training, float32 TFLite export, host evaluation, and firmware |
| SSVEPformer, SSVEPNet | Host training and float32 export only |

Select the model, classical method, precision, and fold with the existing command-line arguments. Fold `N` holds out subject `sN`.

## Getting started

The full host workflow uses Python 3.10. From the repository root:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r environment\requirements-all.txt
python scripts\setup_dataset.py
```

`setup_dataset.py` downloads any missing 12JFPM subject files and writes `data/processed/12JFPM/processed.npz`. Regenerating the archived benchmark tables does not need that dataset. Use `paper_reproduction/requirements.txt` and [paper_reproduction/README.md](paper_reproduction/README.md).

## Workflow

Dataset preparation comes first. Classical evaluation can start from the processed dataset. Neural backends continue through training and export when that decoder supports them.

1. Prepare 12JFPM. See [docs/dataset.md](docs/dataset.md).
2. Evaluate a classical method, or train a neural model with `scripts/train_neural.py`.
3. Export TFLite with `scripts/export_tflite.py` for models that have an export path.
4. Evaluate on the host with `scripts/evaluate_host.py`. `--runtime stage1` compares a PyTorch checkpoint with a host TFLite file on the same held-out trials.
5. Stage one backend and one fold with `scripts/generate_firmware_artifacts.py`. The default directory is `firmware/generated/`. The command replaces that directory. It does not compile firmware.
6. Build, flash, and replay on an ESP32-S3. Replay compares host TFLite with the device. That comparison is Stage 2.

SSVEPformer and SSVEPNet stop before firmware staging. Classical methods skip neural training and TFLite export. Command details are in [docs/training-and-export.md](docs/training-and-export.md) and [docs/firmware-and-replay.md](docs/firmware-and-replay.md).

## Layout

| Path | Contents |
|---|---|
| `src/` | Host implementation |
| `scripts/` | Command-line entry points |
| `configs/` | Training, analysis, protocol, and hardware configuration |
| `firmware/` | ESP32-S3 replay firmware |
| `artifacts/reference/` | Retained read-only artifacts |
| `artifacts/generated/` | Local TFLite exports |
| `manifests/loso/` | Leave-one-subject-out fold manifests |
| `runs/` | Training checkpoints and logs |
| `outputs/` | Host and device evaluation outputs |
| `docs/` | Workflow guides |

## Requirements

- Python 3.10 and `environment/requirements-all.txt` for training, export, and host evaluation. Archived-result reproduction uses `paper_reproduction/requirements.txt`.
- ESP-IDF 5.4.0 to build firmware. Pins are in [environment/firmware_versions.md](environment/firmware_versions.md).
- Git LFS, so retained binary artifacts are downloaded with the clone.

Commands above use Windows PowerShell. The same flags work on other shells with forward slashes.

## Documentation

- [docs/dataset.md](docs/dataset.md)
- [docs/training-and-export.md](docs/training-and-export.md)
- [docs/firmware-and-replay.md](docs/firmware-and-replay.md)
- [paper_reproduction/README.md](paper_reproduction/README.md)
- [firmware/README.md](firmware/README.md)

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Third-party ESP-IDF components under `firmware/replay_fw/managed_components/` retain their own licenses (typically Apache 2.0).
