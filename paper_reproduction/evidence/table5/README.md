# Table 5 records

## Overview

This directory holds the resource-profiling records used to regenerate the firmware-footprint and runtime-memory table. `paper_reproduction/reproduce.py` reconstructs every numeric cell from these files: 44 footprint values and 29 runtime-memory values. The four `---` arena cells are the analytical backends, which allocate no TFLite Micro arena.

Device latency and trial-level fidelity are not taken from this directory. They come from the strict LOSO replay records in `../replay/`.

## Measurement scope

Firmware footprint and memory values are representative per-backend snapshots. They do not necessarily come from the same firmware binaries as the strict LOSO replay.

| | Resource profiling | Strict LOSO replay |
| --- | --- | --- |
| Firmware | Representative single-backend builds. Footprint, heap, and stack come from `replay_fw` builds; arena workspace comes from separate `smoke_fw` runs. These builds are not the strict-replay image. | `replay_fw` at commit `31fb64b` |
| Models | Profiled neural outputs were matched to fold-1 host references. The profiling logs do not record a fold tag. For all seven neural backends, byte size, tensor shapes, data types, quantization-parameter counts, operator list, and operator wiring match the strict LOSO models of all 10 folds. | One strict LOSO model per held-out subject |
| Sessions | One profiled replay session per backend (subject s1, 180 trials), or one smoke run per neural backend for the arena | 10 subjects × 180 trials per backend |
| Quantities | Firmware footprint and runtime memory | Device latency and trial-level host–device records |

The size and operator match supports reading the embedded-model bytes and the TFLM arena plan as representative of those graphs. It does not make the heap drop or the stack high-water mark a measurement of the strict LOSO firmware. Those two quantities also depend on the profiling-build configuration and on other tasks in the session, including Wi-Fi.

## Evidence files

`reproduce.py` reads the transcription, the UART session summaries, and the arena log.

| Path | What it records |
| --- | --- |
| `idf_size_screenshots/<backend>.png` | `idf.py size` screenshot. Flash and static DRAM come from the total row. |
| `idf_size_transcription.csv` | Byte transcription of those 11 screenshots. |
| `uart_replay_sessions/uart_<backend>.log` | UART log of the profiled session. |
| `uart_replay_sessions/uart_<backend>_profile.txt` | `RES_PROFILE phase=session_summary` block from the same session. ConvCA uses `uart_convca_f32_profile.txt`. |
| `smoke_arena/tensor_arenatxt.txt` | `TFLM_ARENA backend=... cap=... used=... region=PSRAM` lines from the seven smoke runs. |

The public copy of `idf_size_screenshots/eegnet_f32.png` has its top 40 pixel rows painted over. The build header showed a local path. Pixels below that band are unchanged, and the transcribed byte counts are the original measurement values. The screenshots are the only footprint record.

## Metric definitions

KiB values are bytes/1024, rounded half-up to one decimal place.

### Firmware footprint

Application image, flash code, flash data, and static DRAM (DIRAM) are the total-row byte counts in each `idf.py size` screenshot, as transcribed in `idf_size_transcription.csv`.

### Heap drop

On the `RES_PROFILE phase=session_summary` line:

```text
sess_est_peak_drop_int = sess_baseline_int - sess_min_int
```

`sess_baseline_int` is the internal free heap at `pre_backend_init`. `sess_min_int` is the lowest internal free heap sampled from that point through the last successful forward. The difference is a session-scoped lower bound on transient internal consumption. It is not a hardware memory tracer, it does not isolate Wi-Fi or other tasks, and it is not the boot-wide minimum (`int_min_ever_boot`).

### Stack high-water mark

`stack_hwm_words` is the field name in the UART record. The value is `uxTaskGetStackHighWaterMark(NULL)` for the `replay_tcp` task. On ESP-IDF for the ESP32-S3 that call returns unused stack in bytes at the deepest use, not FreeRTOS words. It is the minimum free stack of that task, compared with the configured `REPLAY_TCP_STACK_BYTES` of 16 KiB. It is not a stack measurement for the whole firmware image. For the analytical backends it is a lower bound on remaining headroom in that 16 KiB stack.

### Used tensor-arena workspace and reserved capacity

Neural rows use `used` and `cap` from the `TFLM_ARENA` lines. All seven arenas were allocated in PSRAM. `used` is the workspace occupied by the interpreter plan. `cap` is the reserved capacity: CCNN-f32 and CCNN-int8 512 KiB, ConvCA-f32 512 KiB, EEGNet-f32 256 KiB, EEGNet-int8 384 KiB, and FBtCNN-f32 and FBtCNN-int8 768 KiB. The resource table reports the used size. Analytical backends have no arena.

## Limitations

Each value comes from one build and one session per backend. The results are not across-fold averages or worst-case guarantees.

The complete profiling-build configuration was not retained. Compiler optimization and CPU frequency of the profiling builds cannot be independently established from the retained configuration file.

Rebuilding the current firmware does not reproduce these image-size measurements. The application image embeds the build time, and a new compile is not a measurement of the archived screenshots.
