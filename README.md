# ESP32-S3 SSVEP Decoder Benchmark

## Overview

This repository provides practical tools for 12JFPM SSVEP decoder work on host and ESP32-S3:

- dataset preparation and validation
- analytical and neural host evaluation
- neural model training
- TensorFlow Lite export and post-training quantization (PTQ)
- retained fold-specific reference artifacts
- ESP32-S3 artifact staging
- clean ESP-IDF firmware builds
- host-side replay helpers and device evaluation entry points

Retraining or re-exporting may produce different weights or TFLite binaries. Exact reference firmware builds use the retained artifacts under `artifacts/reference/`.

## Supported workflows

- Prepare and validate the processed 12JFPM NPZ
- Train six PyTorch models under strict outer LOSO
- Evaluate classical methods and neural models on host
- Export float32 TFLite and run int8 PTQ for supported models
- Verify retained reference artifacts (220 checksum entries)
- Stage one backend and fold into `firmware/generated/`
- Configure and build `firmware/replay_fw` with ESP-IDF
- Resolve matching replay inputs and write device outputs under `outputs/device/`

## Supported decoders

| Decoder or backend | Train | Classical host | PyTorch host | TFLite export | Retained reference artifact | TFLite host | Firmware staging | Clean ESP-IDF build | Physical replay |
|---|---|---|---|---|---|---|---|---|---|
| CCA | No | Yes | No | No | Yes (analytical) | No | Yes | Yes (fold 1) | Not revalidated |
| FBCCA | No | Yes | No | No | Yes (analytical) | No | Yes | Yes (fold 1) | Not revalidated |
| MSI | No | Yes | No | No | Yes (analytical) | No | Yes | Yes (fold 1) | Not revalidated |
| TRCA | No | Yes | No | No | Yes (analytical) | No | Yes | Yes (fold 1) | Not revalidated |
| CCNN-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| CCNN-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| ConvCA-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| EEGNet-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| EEGNet-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| FBtCNN-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| FBtCNN-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes | Yes (fold 1) | Not revalidated |
| SSVEPformer | Yes | No | Yes | Yes (f32) | Not supplied | Host only | No | No | No |
| SSVEPNet | Yes | No | Yes | Yes (f32) | Not supplied | Host only | No | No | No |

Canonical PyTorch implementations live under `src/neural/`. SSVEPformer and SSVEPNet are host-side only in this package.

## Repository structure

| Path | Contents |
|---|---|
| `configs/` | Training, analysis, protocol, and hardware YAML |
| `src/` | Host implementation |
| `scripts/` | Public command wrappers |
| `firmware/` | ESP32-S3 replay firmware sources and components |
| `firmware/generated/` | Staged firmware build inputs (ignored by Git) |
| `firmware/replay_fw/` | ESP-IDF project |
| `artifacts/reference/` | Retained read-only checkpoints, TFLite, analytical bundles |
| `artifacts/generated/` | User-created exports |
| `manifests/loso/` | Ten LOSO fold manifests |
| `runs/` | Training checkpoints and logs |
| `outputs/host/` | Host evaluation outputs |
| `outputs/device/` | Device replay outputs |
| `environment/` | Dependency pins |
| `tests/` | Unit and contract tests |
| `docs/` | Workflow documentation |

## Requirements

- Python 3.10+ with packages from `environment/requirements-*.txt`
- ESP-IDF 5.4.0 for firmware builds (see `environment/firmware_versions.md`)
- Git LFS when cloning retained binary reference artifacts
- A local 12JFPM raw dataset for preparation, training, and evaluation paths that need data

## Quick start

Primary tested shell: Windows PowerShell.

```powershell
python -m pip install -r environment/requirements-analysis.txt
python scripts\prepare_12jfpm.py --config configs\protocol\preprocess_12jfpm.yaml --output-dir <DATASET_ROOT>
python scripts\validate_dataset.py --processed-npz <DATASET_ROOT>\processed.npz
python scripts\verify_reference_artifacts.py
python scripts\train_neural.py --model eegnet --dataset-root <DATASET_ROOT> --output-root runs --dry-run
python scripts\evaluate_host.py --runtime classical --method cca --dry-run
python scripts\generate_firmware_artifacts.py --backend eegnet-int8 --fold 1 --artifact-set reference
cd firmware\replay_fw
idf.py set-target esp32s3
idf.py build
```

POSIX equivalents use the same flags with forward slashes.

## Reference artifacts

`artifacts/reference/` holds retained fold-specific artifacts used for host checks and firmware staging:

- 40 PyTorch checkpoints (CCNN, ConvCA, EEGNet, FBtCNN × 10 folds)
- 70 TFLite models across seven realized neural variants × 10 folds
- analytical bundles for CCA, FBCCA, MSI, and TRCA across ten folds
- FBtCNN preprocessing headers for ten folds
- 220 checksum entries in `artifacts/reference/checksums.sha256`

```powershell
python scripts\verify_reference_artifacts.py
```

Training, export, and firmware staging refuse to overwrite this tree. Staging writes only under `firmware/generated/`.

## Host evaluation

```powershell
python scripts\evaluate_host.py --runtime classical --method cca --dataset-root <DATASET_ROOT> --output-dir outputs\host\cca
python scripts\evaluate_host.py `
  --runtime pytorch `
  --backend eegnet `
  --fold 1 `
  --artifact-set reference `
  --dataset-root <DATASET_ROOT> `
  --output outputs\host\eegnet\fold01
python scripts\evaluate_host.py `
  --runtime tflite `
  --backend eegnet `
  --precision int8 `
  --fold 1 `
  --artifact-set reference `
  --dataset-root <DATASET_ROOT> `
  --output outputs\host\eegnet_int8\fold01
```

Host outputs belong under `outputs/host/`. See `docs/training-and-export.md`.

## New training and export

```powershell
python scripts\train_neural.py --model eegnet --fold 1 --dataset-root <DATASET_ROOT> --output-root runs
python scripts\export_tflite.py `
  --model eegnet `
  --fold 1 `
  --checkpoint <CHECKPOINT_PATH> `
  --dataset-root <DATASET_ROOT> `
  --manifest manifests\loso\fold01.json `
  --quantization f32 `
  --output artifacts\generated\eegnet\fold01
```

New runs write under `runs/` and `artifacts/generated/`. They are not retained reference artifacts. See `docs/training-and-export.md`.

## ESP32-S3 firmware

Validated fold-1 clean builds cover all eleven retained MCU backends after independent staging into an empty `firmware/generated/` directory. Int8 staging is self-contained.

```powershell
python scripts\generate_firmware_artifacts.py --backend eegnet-int8 --fold 1 --artifact-set reference
cd firmware\replay_fw
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

After a successful build, the image can be flashed with the standard ESP-IDF flash command. Physical flashing and end-to-end device replay were not part of the clean-build validation described here.

Details: `docs/firmware-and-replay.md` and `firmware/README.md`.

## Output directories

| Directory | Role |
|---|---|
| `artifacts/reference/` | Retained read-only artifacts |
| `artifacts/generated/` | New user exports |
| `runs/` | Training checkpoints and logs |
| `outputs/host/` | Host evaluation outputs |
| `outputs/device/` | Device replay outputs |
| `firmware/generated/` | Staged firmware build inputs |
| `firmware/replay_fw/build/` | ESP-IDF build output |

Generated runtime directories are ignored by Git where appropriate. Do not edit files under `artifacts/reference/`.

## Documentation

- `docs/dataset.md` — dataset preparation and LOSO manifests
- `docs/training-and-export.md` — training, export, PTQ, host evaluation
- `docs/firmware-and-replay.md` — staging, ESP-IDF build, flash, replay
- `firmware/README.md` — firmware project entry point
- `environment/firmware_versions.md` — toolchain pins

## License

This repository has not yet been released under an open-source license.
