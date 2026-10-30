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
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item sdkconfig.measurement sdkconfig -Force
idf.py build
idf.py -p <PORT> flash monitor
```

`sdkconfig.defaults` selects 240 MHz and performance optimization (`-O2`) for a build that has no `sdkconfig` yet. `sdkconfig.measurement` is the retained neural-replay configuration and also selects 240 MHz and `-O2`. Do not run `idf.py set-target` after copying it: that replaces `sdkconfig`. Enter Wi-Fi credentials locally before flashing. Placeholders compile.

The command above, which omits `--output`, writes `firmware/generated/` and does not modify `artifacts/reference/`. The build reads `selected_backend.cmake` from that directory unless `-DEXPORT_ARTIFACT_ROOT` selects another staging directory. Pipeline paths are in `docs/firmware-and-replay.md`.

When switching backends, delete `build/` and copy `sdkconfig.measurement` again, then `idf.py build`. Int8 staging does not require a sibling f32 stage directory. A successful compile is not a Table 5 measurement.

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
