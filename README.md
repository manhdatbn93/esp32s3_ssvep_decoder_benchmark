# ESP32-S3 SSVEP Decoder Benchmark

Host and ESP32-S3 workflows for classical and neural steady-state visually evoked potential (SSVEP) decoders on the 12JFPM benchmark.

The platform supports:

- Classical and neural SSVEP decoders.
- Host-side evaluation.
- Model training and TFLite export.
- ESP32-S3 firmware integration.
- Host-to-device replay and evaluation.
- Reproduction of retained benchmark results.

The three quick starts below are independent. None of them modifies the retained reference artifacts.

## Quick start

### A. Reproduce benchmark results

Regenerate the benchmark tables and figures from retained evaluation records. This does not need the 12JFPM dataset or an ESP32-S3 board.

```powershell
python -m pip install -r paper_reproduction/requirements.txt
python paper_reproduction/reproduce.py
```

A passing run prints `PASS: 24/24 checks passed`. See [paper_reproduction/README.md](paper_reproduction/README.md).

### B. Host evaluation

12JFPM is not redistributed. Obtain it from its authors, then prepare it and evaluate one backend. The TRCA command uses the default realization, `deployed_sos`. Other realization options are in [docs/training-and-export.md](docs/training-and-export.md).

```powershell
python -m pip install -r environment/requirements-analysis.txt
python scripts\prepare_12jfpm.py --config configs\protocol\preprocess_12jfpm.yaml --raw-data-root <RAW_DATA_ROOT> --output-dir <DATASET_ROOT>
python scripts\evaluate_host.py --runtime classical --method trca --dataset-root <DATASET_ROOT> --output-dir outputs\host\trca
```

CCA, FBCCA, and MSI use the same command with `--method cca`, `fbcca`, or `msi`. Dataset checks and LOSO manifests are in [docs/dataset.md](docs/dataset.md).

### C. Firmware build

Stage one retained backend and compile it with ESP-IDF 5.4.0. This confirms that the sources build. It does not flash a board.

```powershell
python scripts\generate_firmware_artifacts.py --backend msi --fold 1 --artifact-set reference
cd firmware\replay_fw
Copy-Item sdkconfig.measurement sdkconfig -Force
idf.py build
```

`sdkconfig.defaults` selects 240 MHz and performance optimization (`-O2`). `sdkconfig.measurement` is a retained measurement-oriented build profile, and it also selects 240 MHz and `-O2`. An existing `sdkconfig` takes precedence over the defaults, so ESP-IDF does not reapply `sdkconfig.defaults` on top of it. Do not run `idf.py set-target` after the copy above: that command replaces `sdkconfig`. Delete `build/` before building a different backend in the same tree.

Flashing and replay are in [docs/firmware-and-replay.md](docs/firmware-and-replay.md).

## Supported decoders

| Decoder | Role in this repository |
|---|---|
| CCA, FBCCA, MSI, TRCA | Classical host evaluation and firmware, with retained analytical bundles |
| CCNN-f32, CCNN-int8 | Train, TFLite export, host evaluation, and firmware |
| ConvCA-f32 | Train, TFLite export, host evaluation, and firmware |
| EEGNet-f32, EEGNet-int8 | Train, TFLite export, host evaluation, and firmware |
| FBtCNN-f32, FBtCNN-int8 | Train, TFLite export, host evaluation, and firmware |
| SSVEPformer, SSVEPNet | Host-side screening only. No reference artifact is included, and neither backend is staged for firmware |

## Repository structure

| Path | Contents |
|---|---|
| `configs/` | Training, analysis, protocol, and hardware configuration |
| `src/` | Host implementation |
| `scripts/` | Command-line entry points |
| `firmware/` | ESP32-S3 replay firmware |
| `artifacts/reference/` | Retained read-only checkpoints, TFLite models, and analytical bundles |
| `artifacts/generated/` | Exports created by a local training run |
| `manifests/loso/` | Leave-one-subject-out fold manifests |
| `paper_reproduction/` | Regeneration of retained benchmark tables and figures |
| `runs/` | Training checkpoints and logs |
| `outputs/` | Host and device evaluation outputs |
| `environment/` | Dependency pins |
| `tests/` | Tests |
| `docs/` | Workflow guides |

Training and export write under `runs/` and `artifacts/generated/`. They do not overwrite `artifacts/reference/`.

## Requirements

- Python 3.10 or newer. Analysis dependencies are in `environment/requirements-analysis.txt`. Reproduction dependencies are in `paper_reproduction/requirements.txt`.
- ESP-IDF 5.4.0 for firmware builds. Pins are in [environment/firmware_versions.md](environment/firmware_versions.md).
- Git LFS, so the retained binary artifacts are downloaded with the clone.
- A local 12JFPM dataset for preparation, training, and host evaluation. Reproduction from archived records does not need it.

Commands above are for Windows PowerShell. The same flags work on other shells with forward slashes.

## Documentation

- [docs/dataset.md](docs/dataset.md) — dataset preparation and fold manifests
- [docs/training-and-export.md](docs/training-and-export.md) — training, TFLite export, and host evaluation
- [docs/firmware-and-replay.md](docs/firmware-and-replay.md) — staging, build, flash, and replay
- [paper_reproduction/README.md](paper_reproduction/README.md) — regenerating retained benchmark results
- [firmware/README.md](firmware/README.md) — firmware project
- [environment/firmware_versions.md](environment/firmware_versions.md) — toolchain pins

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Third-party ESP-IDF components under `firmware/replay_fw/managed_components/` retain their own licenses (typically Apache 2.0).
