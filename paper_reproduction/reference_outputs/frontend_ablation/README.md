# Frontend ablation reference outputs

These files are the stored host-only comparison of the training-compatible frontend and the deployment-aligned frontend on the retained CCNN and FBtCNN graphs. They do not replace the Stage 1 or Stage 2 replay evidence in `paper_reproduction/evidence/`.

The comparison covers four backends, ten LOSO folds, and the held-out trials of each fold. The graph, interpreter, trial, and top-1 rule stay fixed. The implementation and the meaning of the metrics are in [analysis/frontend_ablation/README.md](../../../analysis/frontend_ablation/README.md).

`EXPERIMENT_MANIFEST.json` and `RELEASE_MANIFEST.sha256.json` are the provenance and checksum records. The CSV and JSON tables give feature deviation, output deviation, top-1 stability, subject summaries, archive agreement, changed trials, the historical stage join, and the recorded calibration overlap. Trial diagnostics contain scalar errors and decisions, not EEG samples.

## Verification

From the repository root, without the EEG dataset:

```powershell
python paper_reproduction/verify_frontend_ablation.py
```

An optional argument selects another results directory. The check compares file checksums with the release manifest and checks the stored trial counts, prediction changes, archive agreement, and historical stage join.

A full rerun needs the processed dataset and is described in the analysis README.

## Limitations

This comparison does not establish device-side intermediate-feature parity. It does not isolate quantization as the cause of a Stage 1 or Stage 2 disagreement, because the int8 graphs were not requantized. Historical Stage 1, Stage 2, and checkpoint-to-device records are left unchanged.
