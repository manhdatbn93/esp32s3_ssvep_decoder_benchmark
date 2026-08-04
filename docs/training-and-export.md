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

Three runtimes: `classical`, `pytorch`, `tflite`.

### Classical

```powershell
python scripts\evaluate_host.py `
  --runtime classical `
  --method cca `
  --dataset-root <DATASET_ROOT> `
  --output-dir outputs\host\cca
```

`--method` choices: `cca`, `fbcca`, `msi`, `trca`, `all`.

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
