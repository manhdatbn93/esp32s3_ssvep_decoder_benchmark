# Benchmark Results Reproduction

This directory regenerates the retained benchmark tables and figures from archived trial-level evidence. It does not retrain models and it does not repeat device measurements.

## Requirements

- Python 3.10 or newer.
- `matplotlib`, pinned in [requirements.txt](requirements.txt), when figures are rendered. `--no-figures` uses the Python standard library only.
- The files already in this directory, including `evidence/`, `expected/`, and `MANIFEST.sha256`.

Reproduction from this archived evidence needs neither the raw 12JFPM EEG dataset nor an ESP32-S3 board.

## How to run

From the repository root:

```powershell
python -m pip install -r paper_reproduction/requirements.txt
python paper_reproduction/reproduce.py
```

Options:

| Option | Effect |
|---|---|
| `--no-figures` | Skip figure rendering and the figure check. A passing run then reports 23/23. |
| `--out DIR` | Write outputs to `DIR` instead of `paper_reproduction/outputs/`. |

The process exit code is 0 only when every check passes. `outputs/` is ignored by Git. [reference_outputs/](reference_outputs/) is a saved passing run and can be inspected without running the script.

## Expected outputs

Default directory: `paper_reproduction/outputs/`.

| Result | Output |
|---|---|
| Table 2. Realized backends: accuracy, agreement, flips, output deviation, latency | `tables/table2_realized_backends.csv` |
| Table 3. Two-stage neural fidelity | `tables/table3_two_stage_fidelity.csv` |
| Table 4. Deployment policies, both objective formulations | `tables/table4_policies.csv`, `tables/table4_policy_exclusions.csv` |
| Table 5. Firmware footprint and runtime memory | `tables/table5_resource_profile.csv` |
| Transition decomposition | `tables/table_s12_transitions.csv`, `records/neural_stage_transitions_per_trial.csv` |
| Normalized output error | `tables/normalized_output_error_summary.csv` |
| Policy sensitivity grid | `tables/table_s16_policy_sensitivity.csv` |
| Figure 2. Margin-quartile flip rates and bootstrap intervals | `figures/fig_margin_quartile.*`, `tables/figure2_margin_quartile_source.csv` |
| Figure 3. Latency against accuracy and agreement | `figures/fig_latency_*_neural.*`, `tables/figure3_latency_source.csv` |
| Int8 decision changes | `records/int8_stage2_flips.csv`, `records/flip_kind_counts.csv` |

`verification_report.json` in the output directory has `status`, `n_checks`, `n_failed`, the Python environment, and a `checks` list. Each check has a name and `passed`. **24/24 PASS** means `status` is `PASS`, `n_checks` is 24, `n_failed` is 0, and the console printed `PASS: 24/24 checks passed`. That count includes the figure check. With `--no-figures` the same evidence checks report 23/23.

The script compares regenerated values with `expected/submitted_values.json` at printed precision, and with the archived analysis records at full precision. It computes results from the trial records. It does not read the expected file in order to produce a result. `reproduce.py` checks `MANIFEST.sha256` before those comparisons.

## Reproduction scope

| What the script does | What it does not do |
|---|---|
| Recomputes summary tables, policy outcomes, and figures from retained trial-level host and device records | Retrain neural networks |
| Reuses archived Stage 1 predictions in `evidence/fidelity/export_fidelity_per_trial.csv` | Recompute checkpoint-to-export predictions. That requires the raw EEG and a host evaluation run |
| Reuses archived device outputs, latencies, and resource measurements | Flash a board or collect new device measurements |
| Checks that the two policy formulations agree on this evidence | Run a new host evaluation. Host commands are in [docs/training-and-export.md](../docs/training-and-export.md) |

To recompute Stage 1, prepare 12JFPM as described in [docs/dataset.md](../docs/dataset.md) and follow the host evaluation steps in [docs/training-and-export.md](../docs/training-and-export.md). New device measurements require a board and the procedure in [docs/firmware-and-replay.md](../docs/firmware-and-replay.md).

## Important interpretation notes

**Policy objectives.** The reproduction package supports the full policy objective formulation and a reduced variant. In `tables/table4_policies.csv` the rows are labeled `as_executed` and `as_described`. The full formulation minimizes the empirical 95th percentile of trial-level relative L2 error among the Pareto objectives of profiles P1, P3, P4, and P5, and uses that percentile as P3's second priority, after agreement. The reduced formulation omits that objective. On the retained benchmark data the two formulations produce identical admissible sets, Pareto sets, and backend selections. Relative L2 error is the deviation from each backend's matched host output. It is not a common-scale score for comparing decoder families.

**Arena workspace.** Table 5 reports used tensor-arena workspace. Reserved PSRAM capacity is a different quantity: 512 KiB for CCNN and ConvCA, 256 KiB for EEGNet-f32, 384 KiB for EEGNet-int8, and 768 KiB for FBtCNN. Analytical backends allocate no tensor arena. A policy limit on used workspace is separate from these reserved capacities.

**Measurement scope.** Replay latency and the resource profile come from different measurement setups. The resource values are one build and one session per backend, not a fold mean or a maximum, and they are not paired with the replay latencies. The firmware configuration used for the analytical replay was not archived. The retained neural-replay profile is [sdkconfig.measurement](../firmware/replay_fw/sdkconfig.measurement). Model identity, capacities, and the limits of the resource snapshot are in [evidence/table5/README.md](evidence/table5/README.md).

**Numerical precision.** For floating-point backends, cosine distance is near 1e-14. The third significant figure depends on summation order. The code uses exactly rounded summation (`math.fsum`), which matches the archived values. Figure 3 is redrawn from Table 2. Its label placement is set in the plotting code and can differ slightly from an earlier rendering of the same data.

## Further documentation

- [docs/dataset.md](../docs/dataset.md) — obtaining and preparing 12JFPM
- [docs/training-and-export.md](../docs/training-and-export.md) — training, export, and host evaluation
- [docs/firmware-and-replay.md](../docs/firmware-and-replay.md) — firmware build and replay
- [evidence/table5/README.md](evidence/table5/README.md) — resource-profile metadata and limits
- [MANIFEST.sha256](MANIFEST.sha256) — checksums of `evidence/` and `expected/`
