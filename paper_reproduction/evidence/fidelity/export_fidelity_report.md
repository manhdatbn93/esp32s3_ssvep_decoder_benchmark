# Strict LOSO Export Fidelity Report

Stage 1 (this report): PyTorch checkpoint vs exported host TFLite.
Stage 2 (existing): host TFLite vs ESP32-S3/TFLM device replay.

Trial source: outer-test folds from `data/processed/12JFPM/processed.npz` (class-major order, 180 trials/subject; matches strict replay trial plan).

## A. Does f32 TFLite preserve PyTorch decisions?

- **CCNN-f32**: 100.00% argmax agreement (1800 trials), mean logit MAE 6.55836e-06, max error 7.24792e-05.
- **ConvCA-f32**: 100.00% argmax agreement (1800 trials), mean logit MAE 2.71548e-07, max error 2.86102e-06.
- **EEGNet-f32**: 100.00% argmax agreement (1800 trials), mean logit MAE 8.13132e-07, max error 9.53674e-06.
- **FBtCNN-f32**: 100.00% argmax agreement (1800 trials), mean logit MAE 0.000277503, max error 0.00274205.

## B. Does int8 introduce most deviation before device replay?

- **CCNN-int8**: export/quant flip 1.56%, device flip 3.72%.
- **EEGNet-int8**: export/quant flip 9.89%, device flip 1.89%.
- **FBtCNN-int8**: export/quant flip 0.78%, device flip 0.33%.

## C. Are remaining mismatches mainly device realization?

- **CCNN-f32**: device-only flips 0, export-only flips 0, both 0 → dominant stage: `none`.
- **CCNN-int8**: device-only flips 52, export-only flips 13, both 15 → dominant stage: `device_realization`.
- **ConvCA-f32**: device-only flips 0, export-only flips 0, both 0 → dominant stage: `none`.
- **EEGNet-f32**: device-only flips 0, export-only flips 0, both 0 → dominant stage: `none`.
- **EEGNet-int8**: device-only flips 14, export-only flips 158, both 20 → dominant stage: `export_quantization`.
- **FBtCNN-f32**: device-only flips 0, export-only flips 0, both 0 → dominant stage: `none`.
- **FBtCNN-int8**: device-only flips 2, export-only flips 10, both 4 → dominant stage: `export_quantization`.

## D. Which stage causes more decision change?

| Backend | Export flip % | Device flip % | Export-only | Device-only | Dominant |
|---|---:|---:|---:|---:|---|
| CCNN-f32 | 0.00 | 0.00 | 0 | 0 | none |
| CCNN-int8 | 1.56 | 3.72 | 13 | 52 | device_realization |
| ConvCA-f32 | 0.00 | 0.00 | 0 | 0 | none |
| EEGNet-f32 | 0.00 | 0.00 | 0 | 0 | none |
| EEGNet-int8 | 9.89 | 1.89 | 158 | 14 | export_quantization |
| FBtCNN-f32 | 0.00 | 0.00 | 0 | 0 | none |
| FBtCNN-int8 | 0.78 | 0.33 | 10 | 2 | export_quantization |

## E. Manuscript-safe wording

- **CCNN-f32**: Safe to state that float32 TFLite export preserved checkpoint argmax on all 1800 outer-test trials (agreement 100.00%).
- **CCNN-int8**: int8 export disagrees with checkpoint on 1.56% of trials; attribute primary accuracy delta to quantization unless device flips dominate.
- **ConvCA-f32**: Safe to state that float32 TFLite export preserved checkpoint argmax on all 1800 outer-test trials (agreement 100.00%).
- **EEGNet-f32**: Safe to state that float32 TFLite export preserved checkpoint argmax on all 1800 outer-test trials (agreement 100.00%).
- **EEGNet-int8**: int8 export disagrees with checkpoint on 9.89% of trials; attribute primary accuracy delta to quantization unless device flips dominate.
- **FBtCNN-f32**: Safe to state that float32 TFLite export preserved checkpoint argmax on all 1800 outer-test trials (agreement 100.00%).
- **FBtCNN-int8**: int8 quantization preserved ≥99% checkpoint decisions before device replay (99.22% export agreement).
- **CCNN-f32** full chain: no export or device argmax flips observed.
- **CCNN-int8** full chain: after faithful export, remaining prediction changes are dominated by on-device execution (device-only flips 52).
- **ConvCA-f32** full chain: no export or device argmax flips observed.
- **EEGNet-f32** full chain: no export or device argmax flips observed.
- **EEGNet-int8** full chain: decision changes are dominated by export/quantization (export-only flips 158 vs device-only 14).
- **FBtCNN-f32** full chain: no export or device argmax flips observed.
- **FBtCNN-int8** full chain: decision changes are dominated by export/quantization (export-only flips 10 vs device-only 2).
- General: report output-level fidelity in two stages (checkpoint→host TFLite, host TFLite→device); avoid claiming bit-exact equivalence unless f32 agreement is 100%.
