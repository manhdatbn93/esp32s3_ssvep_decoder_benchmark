# Dataset setup

The raw 12JFPM dataset is not distributed in this repository. Obtain it from the dataset authors cited in the original 12JFPM publication and provide a local path.

## Expected raw layout

Place subject files in one directory:

```text
<RAW_DATA_ROOT>/
  s1.mat
  s2.mat
  ...
  s10.mat
```

Each file exposes an `eeg` array with shape `[12, 8, 1114, 15]` (classes × channels × samples × trials per class) at a released epoch sampling rate of 256 Hz.

Set `data.raw_data_root` in `configs/protocol/preprocess_12jfpm.yaml`, or pass an output directory and configure the preparer for your local raw root before running preparation.

## Processed contract

After preparation, trials match:

| Item | Value |
|---|---|
| Released epoch sampling rate | 256 Hz |
| Crop | samples `38:550` |
| Samples per trial | 512 |
| Channels | 8 |
| Trial tensor | `[512, 8]` (time × channels) |
| Batched PyTorch tensor | `[B, 512, 8]` |
| Normalization | per-trial per-channel z-score |
| Classes | 12 (`0` through `11`) |
| Subjects | `s1` through `s10` |
| Trials per subject | 180 |

Default processed output path: `data/processed/12JFPM/processed.npz` (override with `--output-dir`).

Contract file: `configs/contracts/12jfpm_fs256_T512_C8_N12.yaml`.

## Prepare

PowerShell:

```powershell
python scripts\prepare_12jfpm.py `
  --config configs\protocol\preprocess_12jfpm.yaml `
  --output-dir <DATASET_ROOT>
```

Optional:

- `--output-dir` — processed output directory (writes `processed.npz` and metadata)
- `--verify-ref` — optional reference NPZ path for preparer verification

POSIX:

```bash
python scripts/prepare_12jfpm.py \
  --config configs/protocol/preprocess_12jfpm.yaml \
  --output-dir <DATASET_ROOT>
```

Preparation requires the user-supplied raw dataset.

## Validate

```powershell
python scripts\validate_dataset.py --processed-npz <DATASET_ROOT>\processed.npz
```

The validator checks that `X` has trailing shape `(512, 8)`, that ten subjects are present, and that twelve class labels are present.

## LOSO manifests

Fold manifests live under `manifests/loso/fold01.json` through `fold10.json`.

Held-out mapping is fold index `N` → subject `sN` (fold 1 holds out `s1`, fold 10 holds out `s10`).

Each manifest includes:

- fold index
- held-out subject
- 180 ordered `test_trial_ids`
- class order `0` through `11`
- preprocessing configuration (`38:550`, 512 samples, per-trial per-channel z-score)
- outer training subjects and related split metadata
- fold-matched artifact expectations where applicable

Point training and evaluation tools at the processed NPZ with `--dataset-root` (directory containing `processed.npz`, or the NPZ path itself where accepted) or `--npz`.

## Replay ordering

Device replay reconstructs a deterministic class-by-trial sequence from the held-out subject data (class `0`..`11`, trial `0`..`14`). That sequence is fixed and is not based on filesystem discovery order.

The current replay client does not read `test_trial_ids` from the LOSO manifest. For the standard 12JFPM processed layout, the reconstructed class-by-trial order matches the stable preparation order used when building the processed NPZ.
