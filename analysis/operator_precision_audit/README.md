# Operator precision audit

Static inspection of the 30 retained int8 TFLite graphs for CCNN, EEGNet, and FBtCNN. The script reads FlatBuffer operators and tensor types. It does not run inference, load EEG, or modify the graphs.

From the repository root:

```powershell
python analysis/operator_precision_audit/audit_tflite_ops.py
```

Outputs in this directory:

- `operator_inventory_30_models.csv` — one row per operator in every fold.
- `operator_precision_summary.csv` — one row per model, including SHA-256 and conversion metadata.
- `OPERATOR_PRECISION_AUDIT.md` — the written findings.

The script exits 0 when every retained graph is classified. The classes are quantized numerical computation, floating-point numerical computation, precision conversion, and shape or layout movement. `Quantize` and `Dequantize` are conversions. A float32 graph interface does not by itself make an internal operator floating-point.
