# Firmware and device replay

## Target hardware and toolchain

- Board: ESP32-S3 DevKitC-1 v1.1
- Module: ESP32-S3-WROOM-2-N32R8V
- Flash: 32 MB
- PSRAM: 8 MB (OPI)
- Metadata: `configs/hardware/esp32s3_devkitc_1_v1_1.yaml`
- Toolchain pins: `environment/firmware_versions.md`

Validated clean builds used ESP-IDF 5.4.0 and Xtensa GCC 14.2.0. Keep `firmware/replay_fw/dependencies.lock` when you need the pinned component versions from that file (`esp-tflite-micro` 1.3.5, `esp-nn` 1.2.1, `esp-dsp` 1.8.0). Deleting the lock can resolve newer registry versions.

`sdkconfig.defaults` enables 32 MB flash and OPI PSRAM. It does not pin CPU frequency or `-O2`. Successful clean builds with those defaults used the ESP-IDF configure defaults of 160 MHz CPU and the default compiler optimization setting.

## Architecture

```text
artifacts/reference/
  → scripts/generate_firmware_artifacts.py
  → firmware/generated/
  → firmware/replay_fw/
  → idf.py build
```

Staging selects one backend and one fold, verifies the selected reference artifact, and writes only the required firmware build inputs under `firmware/generated/`. Firmware CMake consumes that directory. Reference artifacts remain read-only.

Typical staged contents:

- `selected_backend.cmake` / `selected_backend.json`
- `backend_identity.c` / `backend_identity.h`
- `vectors/` stub headers for optional boot self-check compilation
- one backend stage directory with analytical headers or the retained TFLite file

Switching backend or fold requires restaging. Each int8 staging path is self-contained. No sibling f32 stage directory is required. A clean ESP-IDF build is recommended when switching backend families.

CMake fails with a clear error when `selected_backend.cmake` is missing.

## Staging commands

Analytical (tested):

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend cca `
  --fold 1 `
  --artifact-set reference
```

Neural float32 (tested):

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-f32 `
  --fold 1 `
  --artifact-set reference
```

Neural int8 (tested):

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set reference
```

Accepted backend spellings include `cca`, `fbcca`, `msi`, `trca`, `ccnn-f32`, `ccnn-int8`, `convca-f32`, `eegnet-f32`, `eegnet-int8`, `fbtcnn-f32`, and `fbtcnn-int8`.

Dry-run:

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set reference `
  --dry-run
```

Generated-mode staging can consume a user-supplied TFLite path for non-reference builds. Generated outputs still belong under ignored staging or export directories, not under `artifacts/reference/`.

## Normal ESP-IDF build

```powershell
cd firmware\replay_fw
idf.py set-target esp32s3
idf.py build
```

## Clean validation build

Use this sequence for clean validation or after incompatible configuration changes. It is not required before every incremental rebuild.

```powershell
cd firmware\replay_fw
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item sdkconfig -Force -ErrorAction SilentlyContinue
Remove-Item sdkconfig.old -Force -ErrorAction SilentlyContinue
idf.py set-target esp32s3
idf.py build
```

Fold-1 clean builds were validated for all eleven retained MCU backends. Each backend was staged independently into an empty `firmware/generated/` directory. Reference checksums remained unchanged. Ten-fold firmware compilation was not individually repeated in that pass.

### Fold-1 clean-build image sizes

Values below are fold-1 clean-build `Total image size` figures reported by `idf.py size` after successful builds. They are packaging sizes, not benchmark or publication metrics. Neural sizes can change slightly when managed-component versions differ. Keep `dependencies.lock` for the pinned stack in `environment/firmware_versions.md`.

| Backend | Total image size (bytes) |
|---|---:|
| CCA | 919028 |
| FBCCA | 923432 |
| MSI | 919712 |
| TRCA | 2744000 |
| EEGNet-f32 | 965220 |
| EEGNet-int8 | 964920 |
| CCNN-f32 | 1132240 |
| CCNN-int8 | 1004400 |
| ConvCA-f32 | 987484 |
| FBtCNN-f32 | 1692600 |
| FBtCNN-int8 | 1147624 |

Confirmatory clean builds that retained `dependencies.lock` reproduced CCA at 919028 bytes and built EEGNet-f32 and EEGNet-int8 at 960696 and 960416 bytes.

## Flash and monitor

```powershell
idf.py -p <PORT> flash monitor
```

Replace `<PORT>` with the board serial port. Rebuild after changing the staged backend or fold so the flashed image matches the intended replay.

After a successful build, the image can be flashed with the standard ESP-IDF flash command. Physical flashing was not part of the clean-build validation described here.

Set Wi-Fi credentials locally through menuconfig. Public defaults are placeholders only. Do not commit credentials.

## Replay

Transport used by the included device client is TCP over Wi-Fi (`src/replay/replay_client.py`). Serial or USB CDC are used for ESP-IDF flash and monitor, not as the trial replay transport in the current client.

Host helper:

```powershell
python scripts\replay_device.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set reference `
  --dataset-root <DATASET_ROOT> `
  --output outputs\device\eegnet_int8\fold01 `
  --dry-run
```

With `--artifact-set reference`, `--backend` and `--fold` are required. The helper resolves the retained artifact mapping. Full TCP replay still depends on a flashed firmware image and `src/replay/replay_client.py`.

Device firmware and host replay must match on:

- backend
- precision
- fold
- held-out subject
- class order
- preprocessing contract

Replay covers 180 trials per held-out subject with deterministic class-by-trial order. Write device outputs under `outputs/device/`. Reference artifacts are not modified.

### Replay validation status

- Replay client availability: the helper, packet workflow, and TCP client sources are included.
- Physical end-to-end replay: not revalidated after the current firmware packaging and clean-build validation.

## Device output evaluation

```powershell
python scripts\evaluate_device.py
```

This entry point currently prints a placeholder message. It does not aggregate device metrics in this package.

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

## Troubleshooting

### No staged backend

Symptom: CMake reports that staged backend metadata is missing.

Action: run `scripts/generate_firmware_artifacts.py` for the intended backend and fold before `idf.py set-target` or `idf.py build`.

### Stale build cache after changing backend

Action: remove `build`, generated `sdkconfig`, and `sdkconfig.old`, then rerun `idf.py set-target esp32s3` and `idf.py build`.

### Backend or fold mismatch

Action: restage the intended backend and fold, rebuild, then use matching replay arguments.

### Reference checksum failure

Action: do not continue building. Restore the affected reference file from the published repository or release package.

### Dataset path missing

Action: provide the prepared 12JFPM dataset path required by the selected host or replay command.

### Serial port failure

Action: confirm the connected ESP32-S3 port and close other programs using it.
