# ESP32-S3 SSVEP Decoder Benchmark

This repository accompanies the OJ-CS paper on deploying 12JFPM SSVEP decoders on an ESP32-S3. Three workflows are independent. Workflow A regenerates the published tables and figures. Workflow B evaluates decoders on a host from a locally prepared dataset. Workflow C builds the replay firmware. None of them requires editing the retained reference artifacts.

## Workflow A — published results

Regenerate the manuscript tables and figures from the archived trial-level evidence. This does not need the raw EEG and does not need a board.

```powershell
python -m pip install -r paper_reproduction/requirements.txt
python paper_reproduction/reproduce.py
```

A passing run reports 24/24 checks. `--no-figures` skips matplotlib and reports 23/23. Outputs go to `paper_reproduction/outputs/` (ignored by Git). `paper_reproduction/reference_outputs/` is a saved passing run.

The published TRCA accuracy, 83.44% (1,502/1,800), is read from the archived host and device trial records. Those two records agree on all 1,800 trials. Workflow A does not refit TRCA.

See `paper_reproduction/README.md` for provenance, the measurement profile, and the policy-objective note.

## Workflow B — independent host evaluation

Prepare the 12JFPM release locally (it is not redistributed; see `docs/dataset.md`), then evaluate a decoder on the host. The published TRCA command is:

```powershell
python -m pip install -r environment/requirements-analysis.txt
python scripts\prepare_12jfpm.py --config configs\protocol\preprocess_12jfpm.yaml --raw-data-root <RAW_DATA_ROOT> --output-dir <DATASET_ROOT>
python scripts\validate_dataset.py --processed-npz <DATASET_ROOT>\processed.npz
python scripts\evaluate_host.py --runtime classical --method trca --dataset-root <DATASET_ROOT> --output-dir outputs\host\trca
```

That command uses `deployed_sos`, the second-order-section filter bank shared with the deployed host reference and the ESP32-S3 firmware. On the ten held-out subjects it scores 1,502/1,800 (83.44%).

The historical transfer-function candidate is a different realization. It is not the published result:

```powershell
python scripts\evaluate_host.py --runtime classical --method trca --trca-realization screening_ba --dataset-root <DATASET_ROOT> --output-dir outputs\host\trca_screening
```

`screening_ba` scores 1,510/1,800 (83.89%) and disagrees with `deployed_sos` on 45 trials. CCA, FBCCA, and MSI have no such switch. Their host commands are the same pattern with `--method cca`, `fbcca`, or `msi`.

Other host commands:

```powershell
python scripts\verify_reference_artifacts.py
python scripts\evaluate_host.py --runtime pytorch --backend eegnet --fold 1 --artifact-set reference --dataset-root <DATASET_ROOT> --output outputs\host\eegnet\fold01
python scripts\evaluate_host.py --runtime tflite --backend eegnet --precision int8 --fold 1 --artifact-set reference --dataset-root <DATASET_ROOT> --output outputs\host\eegnet_int8\fold01
```

`verify_reference_artifacts.py` checks 220 retained checksum entries. Host outputs belong under `outputs/host/`. Training and export of a new run are documented in `docs/training-and-export.md` and write under `runs/` and `artifacts/generated/`, not into `artifacts/reference/`.

## Workflow C — firmware build and replay

Build one staged backend with ESP-IDF 5.4.0. A fresh build confirms that the sources compile. It does not replace the April resource measurements in Table 5, and it is not flashed by the commands below.

```powershell
python scripts\generate_firmware_artifacts.py --backend msi --fold 1 --artifact-set reference
cd firmware\replay_fw
copy sdkconfig.measurement sdkconfig
idf.py build
```

`sdkconfig.defaults` selects 240 MHz and performance optimization (`-O2`). ESP-IDF applies it when `sdkconfig` is absent and does not reapply it over an existing `sdkconfig`. `sdkconfig.measurement` is the complete May neural-replay configuration, with Wi-Fi placeholders, and it also selects 240 MHz and `-O2`. Copy that file to the gitignored `sdkconfig` and build when reproducing the May firmware environment. Do not run `idf.py set-target` after that copy: set-target replaces the file. If another backend was configured in the same tree, delete `build/` before the next build. Staging writes only under `firmware/generated/`.

Component versions are pinned in `firmware/replay_fw/dependencies.lock` (esp-nn 1.2.1, esp-tflite-micro 1.3.5, esp-dsp 1.8.0). Do not run `idf.py update-dependencies`.

Flashing and live replay are optional and require a board. They are not needed to check the published numbers. See `docs/firmware-and-replay.md`.

## Supported decoders

| Decoder or backend | Train | Classical host | PyTorch host | TFLite export | Retained reference artifact | TFLite host | Firmware staging |
|---|---|---|---|---|---|---|---|
| CCA | No | Yes | No | No | Yes (analytical) | No | Yes |
| FBCCA | No | Yes | No | No | Yes (analytical) | No | Yes |
| MSI | No | Yes | No | No | Yes (analytical) | No | Yes |
| TRCA | No | Yes | No | No | Yes (analytical) | No | Yes |
| CCNN-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| CCNN-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| ConvCA-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| EEGNet-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| EEGNet-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| FBtCNN-f32 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| FBtCNN-int8 | Yes | No | Yes | Yes | Yes | Yes | Yes |
| SSVEPformer | Yes | No | Yes | No | Not supplied | Host only | No |
| SSVEPNet | Yes | No | Yes | No | Not supplied | Host only | No |

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
| `paper_reproduction/` | Manuscript table and figure regeneration |
| `runs/` | Training checkpoints and logs |
| `outputs/host/` | Host evaluation outputs |
| `outputs/device/` | Device replay outputs when `--outdir` points here. The client default is `results/replay/strict_loso_inner_val/` |
| `environment/` | Dependency pins |
| `tests/` | Unit and contract tests |
| `docs/` | Workflow documentation |

## Requirements

- Python 3.10+ with packages from `environment/requirements-*.txt` for Workflow B
- ESP-IDF 5.4.0 for Workflow C (`environment/firmware_versions.md`)
- Git LFS when cloning retained binary reference artifacts
- A local 12JFPM dataset only for preparation, training, and evaluation paths that need EEG

Primary tested shell: Windows PowerShell. POSIX equivalents use the same flags with forward slashes.

## Reference artifacts

`artifacts/reference/` holds retained fold-specific artifacts used for host checks and firmware staging:

- 40 PyTorch checkpoints (CCNN, ConvCA, EEGNet, FBtCNN × 10 folds)
- 70 TFLite models across seven realized neural variants × 10 folds
- analytical bundles for CCA, FBCCA, MSI, and TRCA across ten folds
- FBtCNN preprocessing headers for ten folds
- 220 checksum entries in `artifacts/reference/checksums.sha256`

Training, export, and firmware staging refuse to overwrite this tree.

## Documentation

- `docs/dataset.md` — dataset preparation, citation, and LOSO manifests
- `docs/training-and-export.md` — training, export, PTQ, host evaluation, TRCA realizations
- `docs/firmware-and-replay.md` — staging, ESP-IDF build, flash, replay, measurement profile
- `paper_reproduction/README.md` — regenerating the paper's tables and figures
- `firmware/README.md` — firmware project entry point
- `environment/firmware_versions.md` — toolchain pins

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

Third-party ESP-IDF components under `firmware/replay_fw/managed_components/` retain their own licenses (typically Apache 2.0).
