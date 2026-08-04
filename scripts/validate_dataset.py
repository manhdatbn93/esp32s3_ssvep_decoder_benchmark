#!/usr/bin/env python3
"""Validate 12JFPM processed metadata against the benchmark preprocessing contract."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-npz", type=Path, required=True)
    args = ap.parse_args()
    npz = np.load(args.processed_npz, allow_pickle=True)
    X = npz["X"]
    assert X.ndim == 3 and X.shape[1:] == (512, 8), X.shape
    subjects = [str(s) for s in npz["subjects"]]
    assert len(set(subjects)) == 10
    y = npz["y"]
    assert len(np.unique(y)) == 12
    print(json.dumps({"n_trials": int(len(y)), "subjects": sorted(set(subjects)), "shape": list(X.shape)}, indent=2))

if __name__ == "__main__":
    main()
