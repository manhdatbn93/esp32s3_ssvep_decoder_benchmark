# Strict LOSO Host-Device Fidelity Report

Source: `results/replay/strict_loso_inner_val/<backend>/<subject_timestamp>/trial_results.jsonl`.

## Backend Summary

| Backend | Precision | Trials | Agreement | Flip rate | Mean MAE | Max error | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCNN-f32 | f32 | 1800 | 100.00% | 0.00% | 0.000212002 | 0.00209427 | 161.23 |
| CCNN-int8 | int8 | 1800 | 96.28% | 3.72% | 0.644819 | 4.6953 | 74.11 |
| ConvCA-f32 | f32 | 1800 | 100.00% | 0.00% | 2.07518e-07 | 2.39e-06 | 2118.08 |
| EEGNet-f32 | f32 | 1800 | 100.00% | 0.00% | 2.75108e-07 | 3.82e-06 | 1118.84 |
| EEGNet-int8 | int8 | 1800 | 98.11% | 1.89% | 0.0177333 | 0.156153 | 69.16 |
| FBtCNN-f32 | f32 | 1800 | 100.00% | 0.00% | 1.82828e-06 | 1.907e-05 | 12970.05 |
| FBtCNN-int8 | int8 | 1800 | 99.67% | 0.33% | 0.0485334 | 0.484936 | 2138.89 |

## f32 Backends With 100% Agreement

- CCNN-f32: 1800 trials, max error 0.00209427
- ConvCA-f32: 1800 trials, max error 2.39e-06
- EEGNet-f32: 1800 trials, max error 3.82e-06
- FBtCNN-f32: 1800 trials, max error 1.907e-05

## int8 Backends With Disagreement

- CCNN-int8: agreement 96.28%, flip rate 3.72%, max error 4.6953
- EEGNet-int8: agreement 98.11%, flip rate 1.89%, max error 0.156153
- FBtCNN-int8: agreement 99.67%, flip rate 0.33%, max error 0.484936

## Margin-Aware Mismatch Pattern

Highest flip-rate backend/margin bins:

| Backend | Margin bin | Trials | Flip rate | Mean host margin |
|---|---:|---:|---:|---:|
| CCNN-int8 | 0.10-0.25 | 19 | 57.89% | 0.179052 |
| CCNN-int8 | <=0.01 | 18 | 55.56% | 0 |
| CCNN-int8 | 0.25-0.50 | 50 | 42.00% | 0.332432 |
| FBtCNN-int8 | <=0.01 | 14 | 21.43% | 0 |
| CCNN-int8 | 0.50-1.00 | 75 | 16.00% | 0.724623 |
| EEGNet-int8 | <=0.01 | 211 | 8.53% | 0 |
| EEGNet-int8 | 0.05-0.10 | 192 | 8.33% | 0.0709273 |
| FBtCNN-int8 | 0.10-0.25 | 38 | 5.26% | 0.183388 |
| CCNN-int8 | 1.00-2.00 | 120 | 4.17% | 1.49723 |
| FBtCNN-int8 | 0.25-0.50 | 47 | 2.13% | 0.374224 |
| CCNN-int8 | >2.00 | 1518 | 0.53% | 16.1819 |
| CCNN-f32 | 0.01-0.05 | 8 | 0.00% | 0.0230277 |

Interpretation: if high flip rates concentrate in the smallest host-margin bins, deployment-induced flips are mostly near decision boundaries. If large-margin bins also flip, the backend needs deeper quantization or implementation investigation.

## Latency vs Fidelity

| Backend | Agreement | Mean latency (ms) |
|---|---:|---:|
| EEGNet-int8 | 98.11% | 69.16 |
| CCNN-int8 | 96.28% | 74.11 |
| CCNN-f32 | 100.00% | 161.23 |
| EEGNet-f32 | 100.00% | 1118.84 |
| ConvCA-f32 | 100.00% | 2118.08 |
| FBtCNN-int8 | 99.67% | 2138.89 |
| FBtCNN-f32 | 100.00% | 12970.05 |

## Main-Paper Evidence Suitability

- CCNN-f32: suitable as output-level fidelity evidence (agreement 100.00%, latency 161.23 ms).
- EEGNet-f32: suitable as output-level fidelity evidence (agreement 100.00%, latency 1118.84 ms).
- ConvCA-f32: suitable as output-level fidelity evidence (agreement 100.00%, latency 2118.08 ms).
- FBtCNN-f32: suitable as output-level fidelity evidence (agreement 100.00%, latency 12970.05 ms).
- FBtCNN-int8: suitable as output-level fidelity evidence (agreement 99.67%, latency 2138.89 ms).
- EEGNet-int8: suitable as output-level fidelity evidence (agreement 98.11%, latency 69.16 ms).
- CCNN-int8: suitable as output-level fidelity evidence (agreement 96.28%, latency 74.11 ms).

Caution: this report supports output-level fidelity, not layer/operator-level sensitivity.
