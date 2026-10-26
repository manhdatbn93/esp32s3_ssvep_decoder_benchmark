# On-device resource profile (replay firmware)

This document defines **measured** vs **configured** embedded resource metrics for the
canonical **TCP replay** path (`embedded/replay_fw/`), i.e. the same on-device
verification workflow used for held-out subject **s1** (12JFPM replay client).

## Enable / disable

- Kconfig: `CONFIG_REPLAY_RESOURCE_PROFILE` (menu *Replay WiFi TCP*), default **on**.
- Also set in `embedded/replay_fw/sdkconfig.defaults`.

## UART log format (`RES_PROFILE`)

During each replay **session** (from `START_RUN` through `END_RUN`), the `replay_tcp`
task emits lines containing `RES_PROFILE` with **space-separated `key=value` fields**
(all heap sizes are **bytes** at sample time unless noted).

| Phase | When | Purpose |
|-------|------|---------|
| `pre_backend_init` | After prior decoder cleanup, **before** `backend->init()` | Baseline heap before backend allocations |
| `post_backend_init` | Immediately after successful `init()` | Heap after init (e.g. TFLM arena malloc) |
| `pre_first_infer` | After preprocess, before first `run()` | Heap before first forward |
| `post_first_infer` | After first successful `run()` | Heap after one forward (trial payload still allocated) |
| `session_summary` | On `END_RUN`, before decoder cleanup | Session rollup + stack high-water |

**Per-trial:** After **every** successful forward, the firmware samples internal /
SPIRAM free to update the **session minimum** (no extra UART line).

### Session summary fields (measured)

- `int_free_now`, `spiram_free_now` — instantaneous free at session end.
- `sess_baseline_int` — internal free at `pre_backend_init`.
- `sess_min_int` — **minimum** internal free observed from `pre_backend_init`
  through the last successful forward sample (session-scoped).
- `sess_est_peak_drop_int` — `sess_baseline_int - sess_min_int` (approximate
  **lower bound** on transient internal consumption during the session; other
  tasks and ISR noise are not isolated).
- `sess_min_spiram` — session minimum SPIRAM free (relevant when trial buffers
  or arenas use PSRAM).
- `stack_hwm_words` — field name is historical; value is
  `uxTaskGetStackHighWaterMark(NULL)` for the **replay_tcp** task. On **ESP-IDF
  for ESP32-S3**, this API returns the high-water mark in **bytes** (unused stack
  at the deepest usage), not FreeRTOS “words”. Compare to `REPLAY_TCP_STACK_BYTES`
  (16 KiB) in `embedded/replay_fw/main/replay_main.c`.

### Boot-global fields (measured, not session-scoped)

Phases `pre_backend_init` … `post_first_infer` also log:

- `int_min_ever_boot` — `heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)`
  since **chip boot** (can be lowered by Wi-Fi, boot smoke, or earlier sessions).
- `spiram_min_ever_boot` — same for SPIRAM.

Do **not** confuse these with per-session minima.

## Tensor arena / scratch (**configured**, not measured)

TFLite Micro uses a **compile-time arena size** passed to `tflite_ctx_create`.
These are **budgets**, not measured peak tensor usage:

| Backend | Arena / scratch (configured) | Source |
|---------|------------------------------|--------|
| EEGNet-int8 | 384 KiB | `embedded/smoke_fw/components/dl_backends/eegnet_smoke.c` (`EEGNET_ARENA_SIZE`) |
| EEGNet-f32 | 256 KiB | same |
| CCNN-int8 / CCNN-f32 | 512 KiB | `ccnn_smoke.c` (`CCNN_ARENA_SIZE`) |
| ConvCA-f32 | 512 KiB | `convca_smoke.c` (`CONVCA_ARENA_SIZE`) |
| FBtCNN-int8 / f32 | 768 KiB | `fbtcnn_smoke.c` (`FBTCNN_ARENA_SIZE`) + separate **64 KiB** feature buffer (PSRAM preferred) |

Classical backends (CCA, FBCCA, MSI, TRCA) use **static** `.bss` / `.rodata` and
stack; there is no TFLM arena. Peak stack for those paths is **not** fully
characterized here — use `stack_hwm_words` on replay sessions as a **lower
bound** on headroom for the TCP task’s stack (16 KiB configured in
`replay_main.c`).

## Build-time artifacts (**measured** after `idf.py build`)

From the **same** `idf.py` build as the on-device run:

1. `idf.py size` — total / DRAM / IRAM / flash-ROM breakdown for the app.
2. `idf.py size-components` — per-archive contributions.
3. Optional: `build/*.bin` byte sizes (recorded by helper scripts).

**Scripts (repo):**

- `tools/embedded/record_firmware_build_metrics.ps1` (Windows)
- `tools/embedded/record_firmware_build_metrics.sh` (Unix)

Outputs default to `results/firmware_metrics/<timestamp>/` (gitignored parent
`results/` except `.gitkeep`; copy summaries into this doc or a paper repo if
needed).

**Partition table:** `embedded/replay_fw/partitions.csv` defines the **factory**
app partition size (declared **15 MiB** at `0x10000`) — not the same as the
`.bin` file length.

## Canonical workflow (s1 replay + resource capture)

1. Build **one** replay image with exactly the backends you need (flash size
   grows with all symbols embedded), e.g. FBCCA-only:

   ```text
   cd embedded/replay_fw
   idf.py build -DSMOKE_ENABLE_FBCCA=ON
   ```

2. Record build sizes:

   ```text
   pwsh ../../tools/embedded/record_firmware_build_metrics.ps1
   ```

3. Flash, run replay client for subject s1 (180 trials), **save UART** to a log
   file (or use your usual serial capture).

4. Extract `RES_PROFILE` lines:

   ```text
   python ../../tools/embedded/parse_res_profile_from_log.py uart.log > res_profile.tsv
   ```

5. Fill the paper table below: **one row per backend**, **one build** per row
   (apples-to-apples for `idf.py size`).

## Recorded sample: EEGNet-int8 (2026-04, single-backend replay image)

**Evidence (runtime):** `tools/embedded/uart_eegnet_int8_profile.txt` (and full
`tools/embedded/uart_eegnet_int8.log`). Replay firmware listed only
`[5] EEGNet-int8`; session included full TCP replay after Wi-Fi association.

**Evidence (build):** `idf.py` application size summary (Memory Type Usage
Summary) from the **same** image as the UART capture — user-recorded figures below
(save `idf.py size` text under `results/firmware_metrics/…` for long-term
archival if desired).

### Build / static (from `idf.py size`, same image)

| Item | Value (B) | Notes |
|------|-----------|--------|
| Total image size (reported) | **1,024,250** | May exceed padded `.bin` on disk |
| Flash code (`.text`) | **694,728** | |
| Flash data (total) | **219,244** | incl. `.rodata` **218,988**, `.appdesc` 256 |
| DIRAM used (total) | **176,696** | `.bss` 82,680; `.text` 73,579; `.data` 20,288 |
| IRAM used | **16,383** / 16,384 | |

Factory partition budget remains `embedded/replay_fw/partitions.csv` (15 MiB app
slot at `0x10000`).

### Runtime heap / stack (`session_summary` + phases)

| Quantity | Value | Source field / note |
|----------|-------|---------------------|
| Session est. peak internal drop | **652 B** | `sess_est_peak_drop_int` |
| Session min internal free | **198,015 B** | `sess_min_int` |
| Internal free @ session end | **198,143 B** | `int_free_now` |
| PSRAM free @ session end | **7,919,000 B** | `spiram_free_now` |
| Session min PSRAM free | **7,883,156 B** | `sess_min_spiram` |
| Stack high-water (unused @ deepest) | **13,456 B** | `stack_hwm_words` in log; **bytes** on ESP-IDF |
| Min internal free since boot | **192,971 B** | `int_min_ever_boot` (not session-scoped) |
| TFLM arena (configured) | **393,216 B** (384 KiB) | UART: `arena=393216`; init-time `heap_caps` allocation |
| Embedded `.tflite` (binary) | **10,984 B** | UART: `model=10984 bytes` (~10.7 KiB int8 artifact) |

Phases `pre_backend_init` … `post_first_infer` for this capture are in
`uart_eegnet_int8_profile.txt` (full free-heap snapshots).

## Paper-ready table template

**Rule:** Replace `—` only with values **copied from captured logs / idf output**
for that backend’s build and session. Do not infer.

| Backend | Peak heap (est.) | Min free heap (session) | Internal free @ summary | PSRAM free @ summary | Stack HWM | Arena / configured scratch | App image (B) | Flash / partition note | Evidence |
|---------|------------------|-------------------------|-------------------------|----------------------|-----------|----------------------------|---------------|------------------------|----------|
| EEGNet-int8 | **652 B** (`sess_est_peak_drop_int`) | **198,015 B** | **198,143 B** | **7,919,000 B** | **13,456 B** unused (ESP-IDF HWM) | **384 KiB** TFLM arena; **10,984 B** int8 model in image | **1,024,250** (idf total image) | Factory **15 MiB** slot `partitions.csv` | `tools/embedded/uart_eegnet_int8_profile.txt`, `uart_eegnet_int8.log`, `idf.py size` |
| CCNN-int8 | … | … | … | … | … | 512 KiB TFLM arena + 1760×`float` static features (`ccnn_preprocess.h`) | … | … | … |
| ConvCA-f32 | … | … | … | … | … | 512 KiB TFLM arena | … | … | … |
| FBCCA | … | … | … | … | … | — (no TFLM arena) | … | … | … |
| TRCA | … | … | … | … | … | — | … | … | … |

### Limitations (for the paper)

- **Peak heap** in the table is **`sess_est_peak_drop_int`**, a session-scoped
  estimate from free heap samples on the replay task; it is **not** a hardware
  heap tracer and does not isolate other tasks (Wi-Fi, IDLE).
- **Stack** is reported only for the **replay_tcp** task, not for other FreeRTOS
  tasks (e.g. Wi-Fi).
- **Multi-backend images:** `idf.py size` reflects **all** embedded backends in
  that build; for smallest flash per backend, build **per-backend** images.

## Suggested manuscript text

**Methodology (one sentence):**  
*On-device resource usage was sampled in the replay firmware using ESP-IDF heap
caps APIs at decoder init and around the first inference, with session-level
minima and replay-task stack high-water reported over UART (`RES_PROFILE` lines),
and build-level flash/DRAM totals recorded with `idf.py size` for the matching
firmware image.*

**Results (one sentence):**  
*Table~X summarizes these measurements alongside compile-time TFLite Micro arena
budgets declared in the smoke backends.*

**Limitations (one sentence):**  
*Reported heap extrema bound but do not isolate concurrent system activity, and
classical decoders may consume additional stack depth in helper routines not
captured outside the replay task’s stack watermark.*
