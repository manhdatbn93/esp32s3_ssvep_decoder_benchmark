# Frontend ablation

This module compares two EEG frontends on one frozen TFLite graph. The interpreter, trial, and top-1 rule stay fixed. Only the frontend changes.

It is a host-only comparison. It is not a Stage 1 or Stage 2 replay, and it does not retrain or requantize.

- Path A is the training and int8-calibration frontend. CCNN uses `torch.fft.fft` through `_complex_spectrum_features`. FBtCNN uses transfer-function `butter` and `filtfilt` through `FBtCNN._filter_bank`.
- Path B is the recorded host-replay frontend. CCNN uses float64 Goertzel through `_ccnn_preprocess`. FBtCNN uses Butterworth SOS `sosfiltfilt` through `_fbtcnn_preprocess`.
- Both paths then use `tf.lite.Interpreter` with resolver `AUTO`, the same setting as `src/replay/host_interpreter.py`.

The processed 12JFPM array is already cropped at onset 38 to 512 samples and channel-wise z-scored once. This run does not z-score again. Raw MAT files are not required when that array is supplied.

## Run

From the repository root, with the processed array available outside Git:

```text
python -m analysis.frontend_ablation.run_ablation --dataset-root <directory-containing-processed.npz> --artifact-root artifacts/reference --output-dir <directory-outside-this-repository> --all-folds --verify-archive
```

`--smoke-test` keeps the first two trial indices of every class. Repeat `--backend` to select `ccnn_f32`, `ccnn_int8`, `fbtcnn_f32`, or `fbtcnn_int8`. Repeat `--fold` to limit the held-out subjects. `--all-folds` evaluates all ten.

Python 3.10 or newer, NumPy, SciPy, PyTorch, and TensorFlow are required. The run records its interpreter and package versions in `EXPERIMENT_MANIFEST.json`. Paths supplied on the command line are not written into that file.

The verified tables are stored in [paper_reproduction/reference_outputs/frontend_ablation/](../../paper_reproduction/reference_outputs/frontend_ablation/). Check those files without the EEG dataset:

```text
python paper_reproduction/verify_frontend_ablation.py
```

## Metrics

Relative L2 is `||B - A|| / max(||A||, 1e-12)`. Feature norms use float64. Output norms and the linear P95 use the paper-reproduction helpers. Top-1 ties use the lowest class index. A flip is harmful when A is correct and B is not, beneficial when B is correct and A is not, and both-wrong when the predictions differ and both miss the label.

The archive check compares the deployment frontend with the stored host outputs. Top-1 agreement is the baseline. A small vector residual can remain when the stored logits are decimal text.

The calibration summary recomputes the historical selection of 192 training-pool trials and its overlap with the inner-validation split. It does not write a new TFLite file. Protected model, evidence, and firmware files are hashed before and after the run. A change aborts the run.

Limitations of the stored comparison are in the [reference-output README](../../paper_reproduction/reference_outputs/frontend_ablation/README.md).
