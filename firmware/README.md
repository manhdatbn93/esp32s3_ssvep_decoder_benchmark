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

Fold-1 clean builds were validated for all eleven retained MCU backends after independent staging. Int8 staging does not require a sibling f32 stage directory.

After a successful build, the image can be flashed with the standard ESP-IDF flash command. Physical flashing and live device replay were not part of that clean-build validation.

Set Wi-Fi credentials locally. Public defaults are placeholders only.

Build and replay documentation: `docs/firmware-and-replay.md`.
Toolchain versions: `environment/firmware_versions.md`.
