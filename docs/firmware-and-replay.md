# Firmware and device replay

## Target hardware and toolchain

- Board: ESP32-S3 DevKitC-1 v1.1
- Module: ESP32-S3-WROOM-2-N32R8V
- Flash: 32 MB
- PSRAM: 8 MB (OPI)
- Metadata: `configs/hardware/esp32s3_devkitc_1_v1_1.yaml`
- Toolchain pins: `environment/firmware_versions.md`

Validated clean builds used ESP-IDF 5.4.0 and Xtensa GCC 14.2.0. Keep `firmware/replay_fw/dependencies.lock` when you need the pinned component versions from that file (`esp-tflite-micro` 1.3.5, `esp-nn` 1.2.1, `esp-dsp` 1.8.0). Deleting the lock can resolve newer registry versions.

`sdkconfig.defaults` enables 32 MB flash, OPI PSRAM, the custom partition table, 240 MHz, and performance optimization (`-O2`). ESP-IDF uses it when `sdkconfig` is absent. It does not overwrite an existing `sdkconfig`.

## Measurement profile (`sdkconfig.measurement`)

`firmware/replay_fw/sdkconfig.measurement` is the complete ESP-IDF configuration that was present in the replay firmware project at the start of the May 2026 neural strict LOSO replay campaign (configuration file dated 23 May 2026, neural replay runs 23–24 May 2026, firmware commit `31fb64b`). Only the Wi-Fi SSID and password were replaced by the Kconfig placeholders. Key settings:

| Setting | Value |
|---|---|
| ESP-IDF (`CONFIG_IDF_INIT_VERSION`) | 5.4.0 |
| CPU frequency | 240 MHz (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`) |
| Compiler optimization | performance (`CONFIG_COMPILER_OPTIMIZATION_PERF`, `-O2`) |
| Power management | disabled (`CONFIG_PM_ENABLE` not set) |
| Flash | 32 MB octal (`CONFIG_ESPTOOLPY_OCT_FLASH`, `CONFIG_ESPTOOLPY_FLASHMODE_OPI`), 80 MHz. The flasher mode string is `dout` because auto-detect is enabled |
| PSRAM | octal, 80 MHz |
| FreeRTOS tick | 1000 Hz |
| Replay options | TCP port 3333, boot smoke test and `RES_PROFILE` resource logging enabled |

To build with this profile, copy it to `sdkconfig` and build. The file already selects ESP32-S3. `idf.py set-target` replaces an existing `sdkconfig` with a new default configuration, so do not run it after this copy. Enter local Wi-Fi credentials only if you will flash a board; the placeholders compile.

```powershell
cd firmware\replay_fw
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item sdkconfig.measurement sdkconfig -Force
idf.py build
idf.py menuconfig           # optional, before flashing: set Wi-Fi SSID / password locally; do not commit them
```

Scope and limits:

- The file documents the configuration of the seven neural device replays (70 runs) reported in the paper. The retained file is the public copy of that profile.
- The four analytical device replays (CCA, FBCCA, MSI, TRCA; 40 runs) were recorded on 13–21 April 2026 with earlier firmware commits (`24bbad9`, `49f0c55`, `6d808e3`) and later filed under the strict LOSO evidence tree. Their build configuration was not archived, so this file is not claimed to describe them. Resource-profile limits are in [the Table 5 records](../paper_reproduction/evidence/table5/README.md).
- The firmware-footprint and runtime-memory values in the paper's resource table come from an earlier per-backend profiling snapshot (16–26 April 2026, legacy fold-1 models). The configuration of those April builds was not archived. It is not claimed that the April builds used this file. Definitions and limits are in `paper_reproduction/evidence/table5/README.md`.
- A clean compile with this profile confirms that a backend builds. It is not a Table 5 measurement and was not flashed. Image size from a new compile is not a substitute for the archived screenshots: the application image embeds the build time. A configure that starts from `sdkconfig.defaults` also selects 240 MHz and `-O2`, and it is not this full profile.

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

The default staging root is `firmware/generated/`. The command deletes that selected directory before writing, so do not point it at a directory that holds anything else. One root contains one backend. Stage float32 and int8 into separate directories when both must be kept:

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-f32 `
  --fold 1 `
  --artifact-set generated `
  --tflite artifacts\generated\eegnet\fold01\eegnet_float32.tflite `
  --output artifacts\generated\eegnet\fold01\firmware
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set generated `
  --tflite artifacts\generated\eegnet\int8\fold01\eegnet_int8.tflite `
  --output artifacts\generated\eegnet\int8\fold01\firmware
```

Those commands copy the TFLite file and write `selected_backend.cmake`, `selected_backend.json`, `staging_meta.json`, `backend_identity.c`, `backend_identity.h`, and `vectors/smoke_vectors_eegnet.h`. EEGNet float32 is stored as `12jfpm_eegnet_fold01_test_s1/eegnet_float32.tflite`. EEGNet int8 is stored as `12jfpm_eegnet_fold01_test_s1_int8/eegnet_int8.tflite`. The smoke header is a compile stub: `SMOKE_EEGNET_N_VECTORS` is 0 and the vectors are zeros. It is not a numerical smoke test. Staging is not `idf.py build`, and it does not flash or replay the board.

Stage 1 is the host comparison of a PyTorch checkpoint with host TFLite, documented in [training and export](training-and-export.md). Stage 2 is the comparison of that host TFLite result with ESP32-S3 firmware during replay. The EEGNet fold-1 host verification did not run Stage 2.

## Build after staging

Staging writes `firmware/generated/selected_backend.cmake`. The build reads that file and selects the backend and fold from it. Use the measurement profile above. Do not run `idf.py set-target` after copying `sdkconfig.measurement`: that command replaces `sdkconfig` with one generated from `sdkconfig.defaults`. Both select 240 MHz and `-O2`. The defaults file is not the measurement profile. Delete `build/` when switching backends.

```powershell
# from repository root, after staging
cd firmware\replay_fw
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item sdkconfig.measurement sdkconfig -Force
idf.py build
```

Optional `-DTARGET_BACKEND` is only a consistency check. It must equal the staged backend. Prefer omitting it.

All eleven published backends compile with this profile. A successful compile is a build check. It is not a Table 5 measurement, and this document does not publish image sizes from those compiles.

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

Replay covers 180 trials per held-out subject with deterministic class-by-trial order. If `--outdir` is omitted and results are saved, `replay_client.py` writes under `results/replay/strict_loso_inner_val/<decoder>/<subject>_<timestamp>`. Pass `--outdir`, as in the example above, to write under `outputs/device/` instead. Reference artifacts are not modified. The live byte-identity hard check runs only for EEGNet-int8. Other backends are identified by staging checksums and by host/device score comparison.

### Replay validation status

- Replay client availability: the helper, packet workflow, and TCP client sources are included.
- Physical MSI smoke (fold 2 / `s2`, 5 trials): a development smoke with device-vs-host argmax agreement on those five trials. It is not the 180-trial strict replay reported in the paper.

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
| `outputs/device/` | Device replay outputs when `replay_client.py --outdir` points here |
| `results/replay/strict_loso_inner_val/` | Default directory when `replay_client.py` saves a run and `--outdir` is omitted |
| `firmware/generated/` | Staged firmware build inputs |
| `firmware/replay_fw/build/` | ESP-IDF build output |

## Troubleshooting

### No staged backend

Symptom: CMake reports that staged backend metadata is missing.

Action: run `scripts/generate_firmware_artifacts.py` for the intended backend and fold before `idf.py build`.

### Stale build cache after changing backend

Symptom: CMake reports `TARGET_BACKEND=cca does not match staged backend msi` (or another pair).

Cause: the ESP-IDF/CMake cache still holds an older `-DTARGET_BACKEND` value while `firmware/generated/selected_backend.cmake` was restaged for a different backend.

Action (pick one):

```powershell
# Option A: match the currently staged backend
idf.py -DTARGET_BACKEND=msi build

# Option B: drop the stale cache and rebuild from staging metadata
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item sdkconfig.measurement sdkconfig -Force
idf.py build
```

Do not run `idf.py set-target` after that copy. It replaces `sdkconfig` with a file generated from `sdkconfig.defaults`, which is not the measurement profile.

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

### EEGNet-int8 runtime identity mismatch

Symptom: `replay_client.py` exits with `MODEL_IDENTITY_MISMATCH`.

That hard check compares host and firmware EEGNet-int8 size, CRC-32, and the first and last 16 bytes. It does not run for the other ten backends. Those backends are checked when artifacts are staged and by host/device score comparison.

Action: restage EEGNet-int8 for the intended fold, rebuild, flash, and replay with the matching `--backend` and held-out `--mat`.

### Serial port failure

Action: confirm the connected ESP32-S3 port and close other programs using it.
