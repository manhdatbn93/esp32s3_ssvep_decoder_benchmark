# Training and export

## Canonical models

Six PyTorch implementations under `src/neural/`:

- CCNN
- ConvCA
- EEGNet
- FBtCNN
- SSVEPformer
- SSVEPNet

All six accept raw processed batches shaped `[B, 512, 8]` (batch × time × channels). Model-specific adapters run inside the forward path or export path.

Configs: `configs/training/*_12jfpm_strict_loso.yaml`.

## Training

PowerShell (tested dry-run form resolves the config and exits):

```powershell
python scripts\train_neural.py `
  --model eegnet `
  --fold 1 `
  --dataset-root <DATASET_ROOT> `
  --output-root runs `
  --dry-run
```

Full training (requires the prepared dataset):

```powershell
python scripts\train_neural.py `
  --model eegnet `
  --fold 1 `
  --dataset-root <DATASET_ROOT> `
  --output-root runs
```

`--model` choices: `ccnn`, `convca`, `eegnet`, `fbtcnn`, `ssvepformer`, `ssvepnet`.

POSIX:

```bash
python scripts/train_neural.py \
  --model eegnet \
  --fold 1 \
  --dataset-root <DATASET_ROOT> \
  --output-root runs
```

Outputs are written under `runs/`. Training refuses to write under `artifacts/reference/`.

New training runs may differ from supplied reference checkpoints even when the same seed and config are used.

### Fixed training-engine behavior

Confirmed by the current strict-LOSO training path:

- strict outer LOSO across ten folds
- one held-out subject per fold (`sN` for fold `N`)
- inner validation fraction `0.10` of the outer-training pool
- base seed `42` in the canonical YAML files
- fold split seed = `seed + fold_index`
- Adam optimizer
- unweighted cross-entropy
- no data augmentation
- no class weighting
- no learning-rate scheduler
- early stopping on inner-validation loss
- retained checkpoint selected by highest inner-validation accuracy
- checkpoint file name `foldNN_best.pt` with `model_state_dict` and fold metadata

ConvCA requires fold-specific templates. The training and export paths register templates for that model. Call sites that invoke `forward()` must set templates first.

Executable YAML fields include `seed`, `data.processed_npz`, `model.*`, `train.*`, `eval.batch_size`, and `loso.inner_val_fraction`.

## Host evaluation

Runtimes: `classical`, `pytorch`, `tflite`, and `stage1`.

### Classical

```powershell
python scripts\evaluate_host.py `
  --runtime classical `
  --method cca `
  --dataset-root <DATASET_ROOT> `
  --output-dir outputs\host\cca
```

`--method` choices: `cca`, `fbcca`, `msi`, `trca`, `all`.

TRCA has two filter-bank realizations. The default, `deployed_sos`, is the second-order-section filter bank used by the published host reference and by the ESP32-S3 firmware. On the ten held-out subjects it scores 1,502 of 1,800 trials (83.44%) and agrees with the retained device predictions on every trial. `screening_ba` is the historical transfer-function candidate used during method screening. It scores 1,510 of 1,800 trials (83.89%) and is not the result reported for the deployed system.

```powershell
python scripts\evaluate_host.py `
  --runtime classical `
  --method trca `
  --dataset-root <DATASET_ROOT> `
  --output-dir outputs\host\trca
python scripts\evaluate_host.py `
  --runtime classical `
  --method trca `
  --trca-realization screening_ba `
  --dataset-root <DATASET_ROOT> `
  --output-dir outputs\host\trca_screening
```

The first command is the published TRCA evaluation. The second is the historical screening candidate. CCA, FBCCA, and MSI are unchanged.

### PyTorch (retained checkpoint)

```powershell
python scripts\evaluate_host.py `
  --runtime pytorch `
  --backend eegnet `
  --fold 1 `
  --artifact-set reference `
  --dataset-root <DATASET_ROOT> `
  --output outputs\host\eegnet\fold01
```

### TFLite (retained artifact)

```powershell
python scripts\evaluate_host.py `
  --runtime tflite `
  --backend eegnet `
  --precision int8 `
  --fold 1 `
  --artifact-set reference `
  --dataset-root <DATASET_ROOT> `
  --output outputs\host\eegnet_int8\fold01
```

Dry-run forms validate argument resolution without requiring a dataset:

```powershell
python scripts\evaluate_host.py --runtime classical --method all --dry-run
python scripts\evaluate_host.py --runtime pytorch --model eegnet --dry-run
python scripts\evaluate_host.py --runtime tflite --model eegnet --dry-run
```

Outputs belong under `outputs/host/`. They are local evaluation outputs only.

### Stage 1: checkpoint versus host TFLite

Stage 1 compares one PyTorch checkpoint with one host TFLite file on the same held-out trials. Stage 2, in [firmware and replay](firmware-and-replay.md), compares host TFLite with ESP32-S3 firmware. This section does not run the device.

The comparison below was run for EEGNet fold 1 only. It is not a result for the other models or folds.

```powershell
python scripts\evaluate_host.py `
  --runtime stage1 `
  --model eegnet `
  --precision f32 `
  --fold 1 `
  --manifest manifests\loso\fold01.json `
  --checkpoint runs\eegnet\checkpoints\fold01_best.pt `
  --tflite artifacts\generated\eegnet\fold01\eegnet_float32.tflite `
  --npz data\processed\12JFPM\processed.npz `
  --output-dir outputs\host\eegnet\fold01\stage1_f32
python scripts\evaluate_host.py `
  --runtime stage1 `
  --model eegnet `
  --precision int8 `
  --fold 1 `
  --manifest manifests\loso\fold01.json `
  --checkpoint runs\eegnet\checkpoints\fold01_best.pt `
  --tflite artifacts\generated\eegnet\int8\fold01\eegnet_int8.tflite `
  --npz data\processed\12JFPM\processed.npz `
  --output-dir outputs\host\eegnet\fold01\stage1_int8
```

Each directory receives `summary.json` and `trial_outputs.npz`. The summary records PyTorch accuracy, TFLite accuracy, top-1 agreement, flip count and dataset-row indices, logit mean absolute error, maximum absolute logit difference, and mean per-trial L2 distance. Both runtimes use the same held-out indices and labels. A fold, subject, or precision mismatch stops the command. Matching accuracies do not by themselves mean every trial agrees; the flip count is the trial-level check.

### TFLite host interpreter

Exported-host references (TFLite on host, compared with device outputs) use TensorFlow 2.21.0 `tf.lite.Interpreter` with the default op resolver (`OpResolverType.AUTO`), which applies the XNNPACK delegate. `src/replay/host_interpreter.py` holds this setting; TFLite host evaluation and the replay host reference use it. With this configuration the retained int8 host outputs are reproduced bit-exactly.

`OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES` selects the reference kernels without XNNPACK. It changes some int8 host outputs (for EEGNet-int8, on 9 of 10 subjects), so it does not reproduce the retained references. Passing `experimental_delegates=[]` does not disable the default delegate. `python src\replay\diag_xnnpack.py` compares the two settings for the fold-1 EEGNet artifacts on seeded synthetic inputs.

## Export and PTQ

### Float32 export

```powershell
python scripts\export_tflite.py `
  --model eegnet `
  --fold 1 `
  --checkpoint artifacts\reference\checkpoints\eegnet\fold01\fold01_best.pt `
  --dataset-root <DATASET_ROOT> `
  --manifest manifests\loso\fold01.json `
  --quantization f32 `
  --output artifacts\generated\eegnet\fold01
```

Generated exports must be written under `artifacts/generated/`. Export refuses to overwrite `artifacts/reference/`.

Export runs in the pinned conversion environment (`environment/tflite_conversion_py310.txt`). Float32 export of CCNN, EEGNet and FBtCNN needs only the checkpoint. ConvCA float32 export and all int8 exports also need `--dataset-root`: ConvCA for its fold templates, int8 for the calibration trials. Each export writes `<model>_<quantization>_export_report.json` next to the `.tflite`, recording the checkpoint and output SHA-256 and the TFLite input and output shapes.

`--dry-run` validates arguments without exporting weights.

Supported export quantizations:

- CCNN, EEGNet, FBtCNN: `f32` and `int8`
- ConvCA, SSVEPformer, SSVEPNet: `f32` only

### Int8 PTQ

```powershell
python scripts\export_tflite.py `
  --model eegnet `
  --fold 1 `
  --checkpoint artifacts\reference\checkpoints\eegnet\fold01\fold01_best.pt `
  --dataset-root <DATASET_ROOT> `
  --manifest manifests\loso\fold01.json `
  --quantization int8 `
  --output artifacts\generated\eegnet\int8\fold01
```

Representative-dataset PTQ contract (confirmed by the current implementation and `scripts/verify_ptq_isolation.py`):

- representative samples are processed trials
- 192 trials per fold (16 per class)
- samples come from the outer-training subject pool
- the held-out subject is excluded
- the process is post-training quantization, not quantization-aware training
- external TFLite I/O remains float32 for the retained int8 artifacts

```powershell
python scripts\verify_ptq_isolation.py
```

Re-exporting the same retained checkpoint may not create a byte-identical TFLite file. Exact reference firmware builds use the retained TFLite artifact directly.

## TFLite input contracts

Common PyTorch host input remains `[B, 512, 8]`. Exported TFLite graphs use backend-specific entry tensors. Do not send one raw `[B, 512, 8]` tensor unchanged to every TFLite artifact.

Validated retained TFLite shapes:

| Backend | TFLite input | TFLite output |
|---|---|---|
| CCNN-f32 | `[1, 8, 220, 1]` | `[1, 12]` float32 |
| CCNN-int8 | `[1, 8, 220, 1]` | `[1, 12]` float32 |
| ConvCA-f32 | `[1, 512, 8]` | `[1, 12]` float32 |
| EEGNet-f32 | `[1, 8, 512]` | `[1, 12]` float32 |
| EEGNet-int8 | `[1, 8, 512]` | `[1, 12]` float32 |
| FBtCNN-f32 | `[1, 8, 512, 4]` | `[1, 12]` float32 |
| FBtCNN-int8 | `[1, 8, 512, 4]` | `[1, 12]` float32 |

Notes:

- CCNN spectral preprocessing is outside the retained TFLite graph
- FBtCNN receives four filter-bank channels
- ConvCA uses fold-specific template information associated with the retained export
- EEGNet receives channel-major TFLite input
- retained int8 artifacts expose float32 external I/O (internal quantized ops do not imply int8 API tensors)

## Reference artifact verification

```powershell
python scripts\verify_reference_artifacts.py
```

Optional: `--skip-checksum`, `--artifact-root`.

The verifier checks:

- manifest schema and unique artifact IDs
- expected checkpoint count (40)
- expected TFLite count (70)
- ten-fold completeness for retained models and backends
- held-out-subject mapping against LOSO manifests
- file sizes and SHA256 digests
- checkpoint and TFLite presence
- analytical bundle completeness for CCA, FBCCA, MSI, and TRCA
- FBtCNN preprocessing header completeness

Current verified inventory: 40 checkpoints, 70 TFLite files, 220 reference files covered by checksum verification. These are inventory counts, not benchmark scores.
