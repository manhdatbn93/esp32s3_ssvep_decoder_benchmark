# Paper reproduction

This directory regenerates the results tables and figures of the submitted OJ-CS
manuscript from the archived trial-level device evidence. It needs neither the raw
12JFPM EEG data nor an ESP32-S3 board.

```powershell
python -m pip install -r paper_reproduction/requirements.txt
python paper_reproduction/reproduce.py
```

The run takes well under a minute. Use `--no-figures` to skip matplotlib (standard
library only) and `--out DIR` to change the output folder (default
`paper_reproduction/outputs/`, ignored by Git). The exit code is 0 only if every check
passes. `outputs/verification_report.json` lists each check with its details and the
Python and package versions. `reference_outputs/` holds the output of one such run for
inspection without running anything.

## What is reproduced

| Paper item | Regenerated from | Output |
|---|---|---|
| Table 2 (11 realized backends: accuracy, host-device agreement, flips, latency) | `evidence/replay/` trial records | `tables/table2_realized_backends.csv` |
| Table 3 (two-stage neural fidelity) | Stage 1 record + `evidence/replay/` | `tables/table3_two_stage_fidelity.csv` |
| Table S12 (transition decomposition) | same | `tables/table_s12_transitions.csv`, `records/neural_stage_transitions_per_trial.csv` |
| 107 int8 Stage 2 flips (67/34/6) over 104 distinct trials | `evidence/replay/` | `records/int8_stage2_flips.csv`, `records/flip_kind_counts.csv` |
| Figure 2 and margin-quartile statistics (10,000-sample subject bootstrap) | `evidence/replay/` | `figures/fig_margin_quartile.*`, `tables/figure2_margin_quartile_source.csv` |
| Table S13 (normalized output error) | output vectors in `evidence/replay/` and `evidence/fidelity/classical_per_trial_fidelity.csv` | `tables/normalized_output_error_summary.csv` |
| Figure 3 (latency vs agreement and accuracy) | Table 2 | `figures/fig_latency_*_neural.*`, `tables/figure3_latency_source.csv` |
| Table 5 (firmware footprint and runtime memory) | `evidence/table5/` | `tables/table5_resource_profile.csv` |
| Table 4 (deployment policies) and Table S16 (5 x 5 sensitivity grid) | Tables 2, S13 and 5 | `tables/table4_policies.csv`, `tables/table4_policy_exclusions.csv`, `tables/table_s16_policy_sensitivity.csv` |

Every regenerated value is compared at its printed precision with
`expected/submitted_values.json`, which is transcribed from the submitted LaTeX source.
Values are also compared at full precision with the analysis outputs archived at
submission (`evidence/submission_analysis/`, `evidence/fidelity/`). The code computes
everything from the trial records and never reads the expected values to produce a
result.

### Not recomputed here

- **Stage 1 (checkpoint to exported TFLite on host).** Recomputing it requires the raw
  EEG. The per-trial Stage 1 predictions are read from the frozen record
  `evidence/fidelity/export_fidelity_per_trial.csv`. To recompute them, prepare 12JFPM
  (`docs/dataset.md`) and run the host evaluation in `docs/training-and-export.md`.
- **Device measurements.** Latency, device outputs and resource values are archived
  measurements. Re-measuring them requires flashing a board
  (`docs/firmware-and-replay.md`).

## Evidence and provenance

| Evidence | Campaign | Firmware and models |
|---|---|---|
| `evidence/replay/<neural backend>/` (70 runs) | Strict LOSO replay, 23-24 May 2026 | `replay_fw` at commit 31fb64b, configuration `firmware/replay_fw/sdkconfig.measurement`; strict LOSO fold models (`artifacts/reference/`) |
| `evidence/replay/<CCA, FBCCA, MSI, TRCA>/s1` … `s10` (40 runs) | Analytical replay, 13-21 Apr 2026 | `replay_fw` at commits 24bbad9, 49f0c55 and 6d808e3; firmware configuration not archived. The runs were filed under the strict-protocol tree later (marked `version_a_device_replay_migrated` in each `summary.json`). Public folders are named by held-out subject only |
| `evidence/table5/` | Representative pre-release resource snapshot, 16-26 Apr 2026 | Single-backend builds of uncommitted pre-release firmware, earlier-protocol fold-1 models; see `evidence/table5/PROVENANCE.md` |
| `evidence/fidelity/` | Host-side fidelity records frozen at submission | - |
| `evidence/submission_analysis/` | Analysis outputs used for Tables 4 and S13 and Figure 2 at submission | - |

Each replay run folder holds `summary.json`, `trial_results.jsonl` (one record per
trial, with the full host and device output vectors) and `trial_results.csv`, as
written by the replay client.

### Measurement configuration

`firmware/replay_fw/sdkconfig.measurement` is the `sdkconfig` used for the
23-24 May neural replay, with only the Wi-Fi SSID and password replaced by
placeholders:

| File | SHA-256 | Bytes |
|---|---|---|
| Development `embedded/replay_fw/sdkconfig` (not published) | `F3D4941E25BDEA3DB585DE799CAD75D734767390FD8420D6C38D8132B210DEAE` | 71,497 |
| `firmware/replay_fw/sdkconfig.measurement` | `5925198ACB9A7389FE9416A57370A90DCF9FD36E85F56CD6B073375FF839B146` | 71,515 |

### Sanitization

The public evidence is a copy of the frozen submission evidence. The only changes
remove private metadata:

- local paths to the raw dataset and the development repository are replaced by
  `<12JFPM_RAW_DATA_DIR>` and `<DEV_REPO>`;
- private IPv4 and MAC addresses are replaced by `<REDACTED_IP>` and `<REDACTED_MAC>`;
- the top 40 pixel rows of `evidence/table5/idf_size_screenshots/eegnet_f32.png`
  (terminal header with local paths) are painted over; the size summary below it is
  unchanged.

No numeric value is altered. `evidence/SANITIZATION_MAP.json` records the SHA-256 of
each original and public file and the number of replacements per rule.
`MANIFEST.sha256` covers every file in `evidence/` and `expected/`. `reproduce.py`
verifies it first.

## Notes for readers of the manuscript

- **Policy objectives.** The canonical objective list is the one used by the
  submission-time analysis (`as_executed` in `repro/policies.py`). It minimizes
  the empirical 95th percentile of relative L2 error among the Pareto objectives
  of P1, P3, P4 and P5, and uses that percentile as the second P3 priority. The
  objective list printed in the manuscript omits that term (`as_described`).
  That shorter list is retained as a sensitivity check. On these measurements
  the two lists produce identical gates, admissible sets, Pareto sets, selections
  and the 5×5 sensitivity grid. Reported backend selections are not changed.
- **Archived filenames.** `evidence/submission_analysis/margin_quartile_figure_source.csv`
  is the source of manuscript Figure 2 (margin quartiles). Its filename at
  submission was `figure3_margin_quartile_source_r21.csv`. The contents are unchanged.
- **Frozen development paths.** `evidence/fidelity/fidelity_report.md` still cites
  `results/replay/strict_loso_inner_val/...`. That sentence is part of the frozen
  record and is not rewritten. The public replay files are under `evidence/replay/`.
- **Subject order in `evidence/replay/aggregate.csv`.** Rows are stored in lexical
  subject order (`s1`, `s10`, `s2`, …). Analysis code addresses subjects by id,
  not by row position.
- **Maintainer-only rebuild.** `tools/build_public_evidence.py` copies the private
  archive into `evidence/`. It is not part of Workflow A. A reviewer regenerates
  the tables with `reproduce.py` and does not need the private archive.
- **Cosine distance precision.** For f32 backends, 1 - cosine similarity is about
  1e-14. The third significant figure then depends on how the dot product and norms
  are summed. The code uses exactly rounded summation (`math.fsum`), which reproduces
  the printed Table S13 values and the archived outputs.
- **Figure 3 styling.** Figure 3 is re-rendered from Table 2; label offsets are set by
  hand and may differ slightly from the submitted PDF.

## Maintainer tools

- `tools/build_public_evidence.py` rebuilds `evidence/` from the private development
  tree. It verifies the submission SHA-256 freeze, applies the sanitization rules and
  fails if any local path, IP or MAC address remains.
- `tools/write_manifest.py` rewrites `MANIFEST.sha256`.

The raw 12JFPM dataset is not redistributed. Obtain it from its original publisher.
