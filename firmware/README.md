# Firmware

`replay_fw/` is the ESP32-S3 TCP replay application.
`components/` holds classical backends, neural backends, preprocessing, common helpers, and `ssvep_dispatch`.

## Workflow

```powershell
python scripts\generate_firmware_artifacts.py `
  --backend eegnet-int8 `
  --fold 1 `
  --artifact-set reference
cd firmware\replay_fw
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

Staging writes only under `firmware/generated/` and does not modify `artifacts/reference/`.
The build reads `firmware/generated/selected_backend.cmake` and fails clearly when staging is missing.

When switching backends, clear a stale CMake `TARGET_BACKEND` cache value:

```powershell
idf.py fullclean
idf.py set-target esp32s3
idf.py build
```

Or pass the matching backend once, for example `idf.py -DTARGET_BACKEND=msi build`.

Fold-1 clean builds were validated for all eleven retained MCU backends after independent staging. Int8 staging does not require a sibling f32 stage directory.

After flash, the board joins Wi-Fi and listens on TCP port `3333`. Live replay uses `src/replay/replay_client.py` with the held-out subject `.mat` for the staged fold. Example MSI fold-2 smoke:

```powershell
python src\replay\replay_client.py `
  --backend msi `
  --host <BOARD_IP> `
  --mat <RAW_DATA_ROOT>\s2.mat `
  --max-trials 5 `
  --no-save `
  --no-aggregate
```

Set Wi-Fi credentials locally. Public defaults are placeholders only.

Build and replay documentation: `docs/firmware-and-replay.md`.
Toolchain versions: `environment/firmware_versions.md`.
