# Firmware toolchain versions

Target stack for the ESP32-S3 replay firmware:

- Board: ESP32-S3 DevKitC-1 v1.1
- Module: ESP32-S3-WROOM-2-N32R8V
- Flash: 32 MB
- PSRAM: 8 MB (OPI)
- ESP-IDF: 5.4.0
- Compiler family: xtensa-esp-elf GCC 14.2.0

Pinned component versions from the committed `firmware/replay_fw/dependencies.lock`:

- espressif/esp-tflite-micro: 1.3.5
- espressif/esp-nn: 1.2.1
- espressif/esp-dsp: 1.8.0

`firmware/components/dl_backends/idf_component.yml` declares `espressif/esp-tflite-micro` with version `*`. Keep `dependencies.lock` for the pins above. A clean component resolve without the lock file may select newer registry versions.

`firmware/replay_fw/sdkconfig.defaults` sets 32 MB flash, OPI PSRAM, custom partitions, and related replay options. It does not pin CPU frequency or force `-O2`. Successful clean builds that started from those defaults used the ESP-IDF configure defaults of 160 MHz CPU and the default compiler optimization setting.

Confirm versions from the local ESP-IDF install, from `firmware/replay_fw/dependencies.lock`, and from the generated `sdkconfig` before publishing a local binary.

Fold-1 compile checks for all eleven retained MCU backends have been run from this public tree after staging into `firmware/generated/` and copying `sdkconfig.measurement` to `sdkconfig`. Those checks confirm compilation. They are not Table 5 measurements, they were not flashed, and they are not the 160 MHz configure that ESP-IDF writes when `sdkconfig.defaults` is used alone. See `docs/firmware-and-replay.md`.
