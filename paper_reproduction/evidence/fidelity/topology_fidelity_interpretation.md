# Topology-Fidelity Interpretation

Hardware: ESP32-S3 DevKitC-1 v1.1.

## Tier 1: Directly Supported By Current Output-Level Results

- All evaluated f32 neural backends reproduce host top-1 decisions exactly in strict replay.
- Int8 backends show backend-dependent host-device disagreement.
- Host and device replay scores are compared after output dequantization when the TFLite output tensor is quantized.
- Prediction flips are margin-aware: EEGNet-int8 and FBtCNN-int8 flips mostly occur in low-margin bins, while CCNN-int8 also has a small number of high-margin flips.

## Tier 2: Architecture-Level Interpretations

- FBtCNN's four-band preprocessing and multi-branch temporal CNN are consistent with high latency, but not with high mismatch in the current results.
- CCNN-int8 is a low-latency operating point with the largest observed int8 score drift and argmax disagreement.
- EEGNet-int8 shows that small logit drift can still change decisions when the host top-1/top-2 margin is small.
- ConvCA-f32 contains correlation/template reductions, but current strict replay only directly supports exact f32 host-device argmax fidelity.

## Tier 3: Not Yet Supported Without Layer-Wise Evidence

- The exact layer or operator causing any int8 mismatch.
- Whether a specific convolution, dense layer, depthwise layer, batch-normalization fold, or reduction causes a particular flip.
- Whether ESP-NN optimized kernels or TFLM reference kernels are responsible for any backend-specific drift.

## Direct Answers

A. Can we identify a specific layer that causes mismatch? No, not from current output-level replay artifacts.
B. Direct layer evidence: none currently available.
C. Supported topology-level explanations: CCNN-int8 combines spectral preprocessing and dense reductions with the largest drift (0.6448 MAE); FBtCNN's filterbank/multi-branch path explains latency better than mismatch; EEGNet-int8 flips align strongly with low margins.
D. Strongest mismatch: CCNN-int8, with 96.28% agreement and 3.72% flip rate.
E. Most stable int8 backend: FBtCNN-int8, with 99.67% agreement and 0.33% flip rate.
F. Margin is currently a stronger direct explanation than layer type because it is measured per trial; layer type remains an architecture-level hypothesis.
G. Proving layer/operator causality needs host TFLite preserve-all-tensors diagnostics and ESP32-S3 firmware instrumentation to dump selected intermediate tensors.
H. Paper now: report output-level fidelity, int8 disagreement, margin-aware flips, latency tradeoffs, and avoid layer-causal wording.
I. Future work: layer-wise PyTorch/TFLite/TFLM activation comparison and targeted firmware instrumentation.

## Layer-Wise Diagnostic Feasibility

- PyTorch f32 vs TFLite f32 on host: possible with a small host-side script, but requires careful checkpoint and preprocessing alignment.
- PyTorch f32 vs TFLite int8/dequantized on host: possible with TensorFlow `experimental_preserve_all_tensors=True`, subject to tensor access and layout alignment.
- TFLite f32 vs TFLite int8 on host: possible now with a small host-side script using preserved tensors.
- TFLite int8 host vs ESP32-S3 int8 device: requires ESP32-S3 firmware instrumentation to expose intermediate tensors.
- Current firmware invokes TFLM through `tflite_ctx_invoke`; intermediate tensors are not returned by the TCP protocol.

Firmware instrumentation would need to locate selected tensors after `Invoke()`, dequantize them when needed, and stream compact summaries or sampled tensor values over serial/TCP. This is useful for future work, but risky for the current revision because it changes firmware behavior and adds substantial logging overhead.
