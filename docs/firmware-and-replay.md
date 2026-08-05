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

Staging writes `firmware/generated/selected_backend.cmake`. The build reads that file and selects the backend and fold tag from it. Do not keep a stale `-DTARGET_BACKEND=...` value from an earlier backend.

Example for MSI fold 2:

```powershell
# from repository root
python scripts\generate_firmware_artifacts.py `
  --backend msi `
  --fold 2 `
  --artifact-set reference

cd firmware\replay_fw
idf.py set-target esp32s3
idf.py build
```

If the previous build used a different backend, either pass the matching backend once:

```powershell
idf.py -DTARGET_BACKEND=msi build
```

or clear the ESP-IDF/CMake cache before building:

```powershell
idf.py fullclean
idf.py set-target esp32s3
idf.py build
```

Optional `-DTARGET_BACKEND` is only a consistency check. It must equal the staged backend in `selected_backend.cmake`. Prefer letting staging select the backend and omitting `-DTARGET_BACKEND` after a clean configure.

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

### Fold-1 clean-build packaging sizes

Fold-1 clean-build `Total image size` from `idf.py size` after successful packaging builds. These are packaging totals only. They can change if ESP-IDF components, `sdkconfig`, or `dependencies.lock` differ. Keep `dependencies.lock` for the pinned stack in `environment/firmware_versions.md`.

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

Set Wi-Fi credentials locally before flashing. Public defaults are placeholders only. Do not commit credentials.

```powershell
cd firmware\replay_fw
idf.py menuconfig
# Example path: Component config → SSVEP Replay → Wi-Fi SSID / password
idf.py -p <PORT> flash monitor
```

Replace `<PORT>` with the board serial port (for example `COM5` on Windows). Rebuild after changing the staged backend or fold so the flashed image matches the intended replay.

Healthy boot for MSI fold 2 looks like:

```text
replay_wifi: got ip: <BOARD_IP>
replay: listening on port 3333
replay: Available backends:
replay:   [ 2] MSI
replay:   1 backend(s) ready
replay: Waiting for client connection...
```

Note the assigned station IP from the serial log. The PC and board must be on the same LAN. Serial or USB CDC is only for flash and monitor. Trial replay uses TCP over Wi-Fi.

## End-to-end MSI smoke (fold 2 / s2)

This is the shortest path to confirm staged artifacts, firmware identity, Wi-Fi, and host–device parity for MSI.

Prerequisites:

1. 12JFPM raw subject file for the held-out subject of the staged fold. Fold 2 holds out `s2`, so use `s2.mat`.
2. Python deps for the host client (`numpy`, `scipy`, and the repo `src` tree).
3. Board already flashed with the MSI fold-2 image and listening on port `3333`.

Confirm staging before flash:

```powershell
Get-Content firmware\generated\selected_backend.json
# expect backend=msi, fold=2, held_out_subject=s2
```

Optional host-side preflight (no board required):

```powershell
python scripts\replay_device.py `
  --backend msi `
  --fold 2 `
  --artifact-set reference `
  --dry-run
```

Live smoke (5 trials, console only):

```powershell
python src\replay\replay_client.py `
  --backend msi `
  --host <BOARD_IP> `
  --port 3333 `
  --mat <RAW_DATA_ROOT>\s2.mat `
  --max-trials 5 `
  --no-save `
  --no-aggregate
```

A healthy smoke run reports:

- `Replay: MSI (id=2)`
- host reference loaded for `msi`
- each trial `AGREE` under device-vs-host parity
- `Parity : 5/5 argmax agree (100.0%)`
- `trials_err` = 0

Device label accuracy on a 5-trial slice can be low. That does not indicate a firmware mismatch when parity is 100%. Parity is the config/firmware check. Accuracy needs the full 180-trial held-out subject run.

Full held-out subject replay with saved artifacts:

```powershell
python src\replay\replay_client.py `
  --backend msi `
  --host <BOARD_IP> `
  --port 3333 `
  --mat <RAW_DATA_ROOT>\s2.mat `
  --outdir outputs\device\msi\fold02_s2
```

## Replay contract

Transport used by the included device client is TCP over Wi-Fi (`src/replay/replay_client.py`).

Host helper dry-run (artifact resolve / checksum only):

```powershell
python scripts\replay_device.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set reference `
  --dry-run
```

With `--artifact-set reference`, `--backend` and `--fold` are required. `--dataset-root` is not required for `--dry-run`. Full TCP replay uses `src/replay/replay_client.py` against a flashed board.

Device firmware and host replay must match on:

- backend
- precision
- fold
- held-out subject (`fold N` → `sN.mat`)
- class order
- preprocessing contract (default `--norm 1` = per-channel z-score)

Replay covers 180 trials per held-out subject with deterministic class-by-trial order. Write device outputs under `outputs/device/` unless you pass `--outdir`. Reference artifacts are not modified.

### Replay validation status

- Replay client availability: the helper, packet workflow, and TCP client sources are included.
- Physical MSI smoke (fold 2 / `s2`, 5 trials): validated on-device with 100% device-vs-host argmax agreement and near-zero score differences.

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

Symptom: CMake reports `TARGET_BACKEND=cca does not match staged backend msi` (or another pair).

Cause: the ESP-IDF/CMake cache still holds an older `-DTARGET_BACKEND` value while `firmware/generated/selected_backend.cmake` was restaged for a different backend.

Action (pick one):

```powershell
# Option A: match the currently staged backend
idf.py -DTARGET_BACKEND=msi build

# Option B: drop the stale cache and rebuild from staging metadata
idf.py fullclean
idf.py set-target esp32s3
idf.py build
```

For a full clean validation, also remove `build`, `sdkconfig`, and `sdkconfig.old` as shown above.

### Backend or fold mismatch

Action: restage the intended backend and fold with `scripts/generate_firmware_artifacts.py`, confirm `firmware/generated/selected_backend.cmake` shows the expected `SSVEP_IDF_BACKEND` and fold tag, rebuild, then use matching replay arguments.

### Reference checksum failure

Action: do not continue building. Restore the affected reference file from the published repository or release package.

### Dataset path missing

Action: for live TCP replay, pass `--mat` to the held-out subject `.mat` for the staged fold (fold `N` → `sN.mat`). See `docs/dataset.md` for the expected raw layout.

### Board not reachable

Symptom: client hangs or fails to connect after the board prints `Waiting for client connection...`.

Action:

1. Confirm the serial log shows `got ip` and `listening on port 3333`.
2. Use that IP in `--host` (DHCP can change after reboot).
3. Confirm the PC is on the same LAN/subnet.
4. Confirm the flashed backend matches the client `--backend` (serial lists ready backends, for example `[ 2] MSI`).

### Wi-Fi associates then disconnects (no IP)

Symptom: serial shows `connecting to <SSID> ...`, then repeated `disconnected reason=..., retry`, and never `got ip`.

Notes:

- Boot lines such as `CCNN-int8 ... No boot vectors — SKIP` are normal for neural images without embedded smoke vectors. That is not the Wi-Fi failure.
- A healthy join must reach `replay_wifi: got ip: ...` before TCP replay can start.

Action:

1. Re-enter SSID and password in `idf.py menuconfig` under Replay WiFi TCP, then rebuild and flash.
2. Confirm the AP is 2.4 GHz (this log used channel 11). ESP32-S3 STA does not join 5 GHz-only SSIDs.
3. Prefer a WPA2 or WPA2/WPA3 mixed personal network. Avoid captive-portal / enterprise auth for this firmware.
4. After rebuilding with current `wifi_connect.c`, read the printed `reason=` code:
   - `15` / `204`: 4-way handshake timeout (often wrong password or unstable AP)
   - `201`: AP not found
   - `2` / `205`: auth/connection fail
5. Move the board closer to the AP or try a phone hotspot as a control.

### Backend or subject mismatch at identity check

Symptom: client exits with a model-identity or backend mismatch error.

Action: restage the intended backend/fold, rebuild and flash, then replay with the matching `--backend` and held-out `--mat`.

### Serial port failure

Action: confirm the connected ESP32-S3 port and close other programs using it.
