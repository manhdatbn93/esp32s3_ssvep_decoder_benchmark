# Frontend ablation

This module measures one controlled substitution. The frozen TFLite graph,
interpreter, trial, and top-1 rule stay fixed. Only the external EEG frontend
changes.

It is not Stage 1 and it is not Stage 2.

- Path A uses the training and int8-calibration frontend: CCNN
  `torch.fft.fft` through `_complex_spectrum_features`, or FBtCNN
  transfer-function `butter` + `filtfilt` through `FBtCNN._filter_bank`.
- Path B uses the recorded host-replay frontend: CCNN float64 Goertzel through
  `_ccnn_preprocess`, or FBtCNN Butterworth SOS `sosfiltfilt` through
  `_fbtcnn_preprocess`.
- Both paths then use `tf.lite.Interpreter` with resolver `AUTO`, the same
  setting as `src/replay/host_interpreter.py`.

The processed 12JFPM array is already cropped at onset 38 to 512 samples and
channel-wise z-scored once. The ablation does not z-score again. Raw MAT files
are not required when that processed array is supplied, and synthetic trials
are never substituted for the dataset run.

## Command

From the repository root, with the processed array available outside Git:

```text
python -m analysis.frontend_ablation.run_ablation ^
  --dataset-root <directory-containing-processed.npz> ^
  --artifact-root artifacts/reference ^
  --output-dir <directory-outside-this-repository> ^
  --all-folds ^
  --verify-archive
```

Useful limits:

```text
python -m analysis.frontend_ablation.run_ablation --dataset-root <dir> --output-dir <dir> --backend ccnn_int8 --fold 1 --smoke-test --verify-archive
```

`--smoke-test` keeps the first two trial indices of every class. `--all-folds`
evaluates all ten held-out subjects. Repeat `--backend` to select
`ccnn_f32`, `ccnn_int8`, `fbtcnn_f32`, or `fbtcnn_int8`.

## Environment

Python 3.10 or newer, NumPy, SciPy, PyTorch, and TensorFlow 2.21. The
interpreter is created by `make_interpreter` with `OpResolverType.AUTO`.
`TF_ENABLE_ONEDNN_OPTS` is left at the process default and recorded in
`EXPERIMENT_MANIFEST.json`. No path supplied on the command line is written
into that manifest.

## Outputs

The output directory receives:

- `per_trial_diagnostics.csv` — scalar errors and decisions, not EEG or features
- `table_a_feature_fidelity.csv` through `table_e_archive_parity.csv`
- `RESULT_SUMMARY.csv` and `SUBJECT_LEVEL_SUMMARY.csv`
- `flip_trials.csv`, `ptq_frontend_feature_summary.csv`, `ptq_overlap.json`
- `historical_stage_check.json` — archived Stage 1 and Stage 2 recomputation
- `EXPERIMENT_MANIFEST.json` — commit, versions, model hashes, and protected-file hashes

## Metrics

Relative L2 is `||B - A|| / max(||A||, 1e-12)`. Feature norms use float64.
Output norms use the paper reproduction helpers. P95 is the linear percentile
used by those helpers. Cosine distance is one minus cosine similarity.

Top-1 ties use the lowest class index. A flip is harmful when A is correct and
B is not, beneficial when B is correct and A is not, and both-wrong when the
predictions differ and both are incorrect. Accuracy delta is B minus A, in
percentage points.

Path B archive error compares the recomputed deployment-frontend output with
the stored `host_logits`. JSON storage is decimal text, so this error includes
serialization as well as any interpreter difference. Top-1 agreement is the
baseline gate. A small vector residual does not by itself mean the decisions
differ.

## What this does not mean

Stage 1 compares the checkpoint with the recorded exported-host reference.
Stage 2 compares that reference with the ESP32-S3. This ablation compares two
host frontends on one frozen graph. It does not requantize, so it cannot
isolate the causal effect of the calibration frontend. A larger int8
sensitivity is an interaction observed on the retained graph.

The PTQ summary recomputes the historical selection: 16 processed trials per
class from the nine training subjects, 192 trials, held-out subject excluded.
It reports feature differences and overlap with the 10 percent inner-validation
split (`seed + fold`). It does not write a new TFLite file.

Protected model, evidence, and firmware files are hashed before and after the
run. A change aborts the run.
