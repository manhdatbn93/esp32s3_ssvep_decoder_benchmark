# Table 5 evidence provenance

Table 5 of the submitted manuscript is a **representative pre-release resource
snapshot**. Every one of its 73 numeric values (44 footprint, 29 runtime memory)
is transcribed from a record in this directory and is regenerated exactly by
`paper_reproduction/reproduce.py`. The four "---" arena cells (CCA, FBCCA, MSI,
TRCA) correspond to analytical backends, which allocate no TFLite Micro arena.

These records were **not** produced by the firmware image that generated the
strict LOSO replay evidence in `../replay/`. The two campaigns differ as follows.

| | Table 5 snapshot (this directory) | Strict LOSO replay (`../replay/`, neural backends) |
|---|---|---|
| Date | 16-17 Apr 2026 (footprint, heap, stack); 26 Apr 2026 (arena used) | 23-24 May 2026 |
| Firmware | `replay_fw` (footprint, heap, stack) and `smoke_fw` (arena), built from uncommitted pre-release working trees | `replay_fw` at commit 31fb64b |
| Builds | One single-backend build per backend | Configuration archived as `firmware/replay_fw/sdkconfig.measurement` |
| Models | Fold-1 models of an earlier subject-as-validation protocol (held-out subject s1) | Strict LOSO fold models (one per test subject) |
| Sessions | One profiled replay session (subject s1, 180 trials) or one smoke run per backend | 10 subjects x 180 trials per backend |

## Why the snapshot describes the evaluated models

Footprint, heap and arena usage depend on the graph structure and model size, not on
the weight values. For all seven neural backends, the fold-1 models used here have the
same byte size, tensor shapes, data types, quantization-parameter counts, operator list
and operator wiring as the strict fold models of all 10 folds. Table 5 therefore
transfers to the evaluated models by **structural equivalence**. It is not a direct
measurement of the strict builds.

The CCNN-int8 and FBtCNN-int8 models were re-exported after 17 Apr with unchanged size
and structure. Their footprint, heap and stack values describe a structurally identical
predecessor. Latency is never taken from this snapshot; all latency values in the
paper come from `../replay/`.

## Configuration

Supported by the records:

- **ESP-IDF 5.4.** The build screenshot header shows a `replay_fw` build from an
  ESP-IDF 5.4 Python environment (the header is redacted in the public copy of
  `eegnet_f32.png`; see `../SANITIZATION_MAP.json`).
- **TFLite Micro arena capacity** for each backend, logged as `cap=` in
  `smoke_arena/tensor_arenatxt.txt` and equal to the firmware arena defines.

Not recorded:

- The `sdkconfig` of the April `replay_fw` builds was not archived. Their CPU frequency
  and compiler optimisation level are **not verified**. The April s1 replay latencies
  are within about 2 % of the strict-campaign means measured at 240 MHz, which is
  indirect evidence only.
- No `idf.py size` text output, map file, ELF or build directory from April survives.
  The screenshots are the only footprint record.

## Files

| Path | Table 5 quantity | Record |
|---|---|---|
| `idf_size_screenshots/<backend>.png` | App image (total image size), flash code, flash data, DIRAM used | `idf.py size` "Memory Type Usage Summary" terminal screenshot, one per backend |
| `idf_size_transcription.csv` | same | Byte values transcribed from the 11 screenshots |
| `uart_replay_sessions/uart_<backend>_profile.txt` | Heap drop (`sess_est_peak_drop_int`), minimum free stack (`stack_hwm_words`, bytes on ESP-IDF) | `RES_PROFILE phase=session_summary` line of the replay session |
| `uart_replay_sessions/uart_<backend>.log` | (context) | Full UART log of the same session |
| `smoke_arena/tensor_arenatxt.txt` | TFLM arena used; arena capacity; arena region | `TFLM_ARENA backend=... cap=... used=... region=PSRAM` lines of seven smoke runs |
| `lab_note/on_device_resource_profile.md` | (definitions) | Original lab note defining the measured fields |

Allocated arena capacity (not printed in the submitted Table 5): CCNN-f32 and CCNN-int8
512 KiB, ConvCA-f32 512 KiB, EEGNet-f32 256 KiB, EEGNet-int8 384 KiB, FBtCNN-f32 and
FBtCNN-int8 768 KiB. All seven arenas were allocated in PSRAM.

Unit conversion in Table 5: KiB = bytes / 1024, rounded half up to one decimal place.

## Limitations

- Each value comes from one build and one session per backend; there are no repeats.
- Heap drop is a session-scoped estimate from free-heap samples on the replay task. It
  does not isolate Wi-Fi or other tasks.
- Stack is reported for the replay task only.
- Rebuilding the current firmware will not necessarily reproduce these byte counts; the
  rebuilt values of `docs/firmware-and-replay.md` use `sdkconfig.defaults` and are not
  Table 5 values.
