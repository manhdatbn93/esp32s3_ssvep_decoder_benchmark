# Paper Reproduction

## Overview

This directory regenerates the retained benchmark tables and figures from archived trial-level evidence. It does not retrain models or repeat device measurements. The raw 12JFPM recordings and an ESP32-S3 board are not required.

## Quick Start

From the repository root:

```powershell
python -m pip install -r paper_reproduction/requirements.txt
python scripts/verify_reference_artifacts.py
python paper_reproduction/reproduce.py
python paper_reproduction/verify_frontend_ablation.py
```

Python 3.10 or newer is required. `reproduce.py` exits 0 only when every check passes. `--no-figures` skips figure rendering and uses the standard library only. `--out DIR` writes somewhere other than `outputs/`.

## Contents

- `evidence/` holds the archived trial records, fidelity tables, and resource logs.
- `expected/` holds the submitted values used for comparison.
- `reference_outputs/` is a saved passing run of the table and figure regeneration.
- [reference_outputs/frontend_ablation/](reference_outputs/frontend_ablation/) holds the host-only frontend comparison.
- `outputs/` is the local regeneration directory and is ignored by Git.
- `MANIFEST.sha256` checksums `evidence/` and `expected/`.

## Scope

Verification recomputes summaries from the retained records. It reuses the archived checkpoint-to-export predictions and device measurements. A new host evaluation needs the processed dataset ([dataset preparation](../docs/dataset.md), [training and export](../docs/training-and-export.md)). New device measurements need a board ([firmware and replay](../docs/firmware-and-replay.md)).

The frontend check reads the stored comparison and does not repeat inference. A full rerun is described in [the analysis README](../analysis/frontend_ablation/README.md).

The policy table compares two objective formulations, labeled `as_executed` and `as_described`. On this evidence they select the same backends. Relative L2 is the deviation from each backend's own host output. Floating-point summaries use exactly rounded summation so they match the archived values. Resource-profile limits are in [evidence/table5/README.md](evidence/table5/README.md).
