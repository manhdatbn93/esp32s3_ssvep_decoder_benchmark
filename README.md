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
python scripts/verify_reference_artifacts.py
python paper_reproduction/reproduce.py
python paper_reproduction/verify_frontend_ablation.py
```

The three checks verify retained artifacts, regenerate the archived tables and figures, and check the stored frontend comparison. See [paper_reproduction/README.md](paper_reproduction/README.md).

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

Copy `sdkconfig.measurement` before building, and do not run `idf.py set-target` afterward. Delete `build/` before building a different backend in the same tree. Flashing and replay are in [docs/firmware-and-replay.md](docs/firmware-and-replay.md).

### D. EEGNet fold 1, from a fresh clone

This path was verified for EEGNet, strict leave-one-subject-out fold 1 (held-out subject `s1`) only. It is not a claim that every model and every fold has been reproduced.

Stage 1 compares the PyTorch checkpoint with host TFLite. Stage 2 compares host TFLite with ESP32-S3 firmware. The steps below verify Stage 1 and stage firmware inputs. They do not compile, flash, or replay the board.

From the repository root, with the new virtual environment activated:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r environment\requirements-all.txt
python scripts\setup_dataset.py
python scripts\train_neural.py --model eegnet --fold 1 --dataset-root data\processed\12JFPM --output-root runs
python scripts\export_tflite.py --model eegnet --fold 1 --checkpoint runs\eegnet\checkpoints\fold01_best.pt --dataset-root data\processed\12JFPM --manifest manifests\loso\fold01.json --quantization f32 --output artifacts\generated\eegnet\fold01
python scripts\export_tflite.py --model eegnet --fold 1 --checkpoint runs\eegnet\checkpoints\fold01_best.pt --dataset-root data\processed\12JFPM --manifest manifests\loso\fold01.json --quantization int8 --output artifacts\generated\eegnet\int8\fold01
python scripts\evaluate_host.py --runtime pytorch --model eegnet --fold 1 --checkpoint runs\eegnet\checkpoints\fold01_best.pt --npz data\processed\12JFPM\processed.npz --output-dir outputs\host\eegnet\fold01\pytorch
python scripts\evaluate_host.py --runtime tflite --model eegnet --precision f32 --fold 1 --tflite artifacts\generated\eegnet\fold01\eegnet_float32.tflite --npz data\processed\12JFPM\processed.npz --output-dir outputs\host\eegnet\fold01\tflite_f32
python scripts\evaluate_host.py --runtime tflite --model eegnet --precision int8 --fold 1 --tflite artifacts\generated\eegnet\int8\fold01\eegnet_int8.tflite --npz data\processed\12JFPM\processed.npz --output-dir outputs\host\eegnet\fold01\tflite_int8
python scripts\evaluate_host.py --runtime stage1 --model eegnet --precision f32 --fold 1 --manifest manifests\loso\fold01.json --checkpoint runs\eegnet\checkpoints\fold01_best.pt --tflite artifacts\generated\eegnet\fold01\eegnet_float32.tflite --npz data\processed\12JFPM\processed.npz --output-dir outputs\host\eegnet\fold01\stage1_f32
python scripts\evaluate_host.py --runtime stage1 --model eegnet --precision int8 --fold 1 --manifest manifests\loso\fold01.json --checkpoint runs\eegnet\checkpoints\fold01_best.pt --tflite artifacts\generated\eegnet\int8\fold01\eegnet_int8.tflite --npz data\processed\12JFPM\processed.npz --output-dir outputs\host\eegnet\fold01\stage1_int8
python scripts\generate_firmware_artifacts.py --backend eegnet-f32 --fold 1 --artifact-set generated --tflite artifacts\generated\eegnet\fold01\eegnet_float32.tflite --output artifacts\generated\eegnet\fold01\firmware
python scripts\generate_firmware_artifacts.py --backend eegnet-int8 --fold 1 --artifact-set generated --tflite artifacts\generated\eegnet\int8\fold01\eegnet_int8.tflite --output artifacts\generated\eegnet\int8\fold01\firmware
```

`setup_dataset.py` downloads any missing official subject files and writes `data/processed/12JFPM/processed.npz`. Training writes `runs/eegnet/checkpoints/fold01_best.pt`. Each export writes `eegnet_float32.tflite` or `eegnet_int8.tflite` plus an export report. Stage 1 writes `summary.json` and `trial_outputs.npz`.

Omitting `--output` on `generate_firmware_artifacts.py` stages into `firmware/generated/` and deletes that directory first. One staging root holds one backend. The two `--output` directories above keep float32 and int8 apart. Staging copies the TFLite file and writes CMake and identity files. It is not an ESP-IDF build. `vectors/smoke_vectors_eegnet.h` is a compile stub: `SMOKE_EEGNET_N_VECTORS` is 0 and the vectors are zeros.

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
| `analysis/` | Additional checks that do not replace the retained replay |
| `runs/` | Training checkpoints and logs |
| `outputs/` | Host and device evaluation outputs |
| `environment/` | Dependency pins |
| `tests/` | Tests |
| `docs/` | Workflow guides |

Training and export write under `runs/` and `artifacts/generated/`. They do not overwrite `artifacts/reference/`.

## Requirements

- Python 3.10. The full host workflow, including training and TFLite export, installs from `environment/requirements-all.txt`. Analysis-only dependencies remain in `environment/requirements-analysis.txt`. Reproduction from archived records uses `paper_reproduction/requirements.txt`.
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
