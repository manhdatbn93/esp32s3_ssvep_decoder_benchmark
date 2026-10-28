"""Run the controlled frontend ablation and write aggregate tables.

The historical Stage 1 and Stage 2 files are read only. Path B is checked
against the archived host replay vectors before A-versus-B is interpreted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis.frontend_ablation.dataset import (
    SUBJECTS,
    calibration_global_indices,
    load_processed_trials,
)
from analysis.frontend_ablation.frontends import (
    FRONTEND_BY_FAMILY,
    dequantize_output,
    prepare_interpreter_input,
)
from analysis.frontend_ablation.metrics import (
    classify_flip,
    feature_pair_metrics,
    output_pair_metrics,
    summarize_numeric,
    top1_and_margin,
)
from paper_reproduction.repro.analysis import join_stages, table3
from paper_reproduction.repro.evidence import load_replay, load_stage1
from src.artifacts.reference import resolve_tflite
from src.replay.host_interpreter import HOST_REFERENCE_OP_RESOLVER, make_interpreter
from src.training.engine import strict_fold_train_inner_indices

REPO = Path(__file__).resolve().parents[2]
BACKENDS = ("ccnn_f32", "ccnn_int8", "fbtcnn_f32", "fbtcnn_int8")
EVIDENCE_NAME = {
    "ccnn_f32": "CCNN-f32",
    "ccnn_int8": "CCNN-int8",
    "fbtcnn_f32": "FBtCNN-f32",
    "fbtcnn_int8": "FBtCNN-int8",
}
PROTECTED = (
    "src/neural/ccnn.py",
    "src/neural/fbtcnn.py",
    "src/replay/host_reference.py",
    "src/replay/host_interpreter.py",
    "src/evaluation/neural_host.py",
    "src/export/quantize_int8.py",
    "firmware/components/dl_backends/ccnn_preprocess.c",
    "firmware/components/dl_backends/fbtcnn_preprocess.c",
    "paper_reproduction/evidence/fidelity/export_fidelity_per_trial.csv",
    "paper_reproduction/reference_outputs/tables/table3_two_stage_fidelity.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(backend: str) -> str:
    return "ccnn" if backend.startswith("ccnn") else "fbtcnn"


def _precision(backend: str) -> str:
    return "int8" if backend.endswith("int8") else "f32"


def protected_hashes() -> dict[str, str]:
    hashes = {rel: _sha256(REPO / rel) for rel in PROTECTED}
    for backend in BACKENDS:
        model = _family(backend)
        for fold in range(1, 11):
            art = resolve_tflite(model, _precision(backend), fold, verify_checksum=True)
            hashes[f"tflite/{backend}/fold{fold:02d}"] = art["sha256"]
    return hashes


def _git_head() -> str:
    head = REPO / ".git" / "HEAD"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = REPO / ".git" / text.split(" ", 1)[1].strip()
        return ref.read_text(encoding="utf-8").strip()
    return text


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _invoke(interpreter, feed: np.ndarray) -> np.ndarray:
    details_in = interpreter.get_input_details()[0]
    details_out = interpreter.get_output_details()[0]
    prepared = prepare_interpreter_input(feed, details_in)
    interpreter.set_tensor(details_in["index"], prepared)
    interpreter.invoke()
    return dequantize_output(interpreter.get_tensor(details_out["index"]), details_out)


def _interpreter_record(interpreter) -> dict:
    details_in = interpreter.get_input_details()[0]
    details_out = interpreter.get_output_details()[0]
    delegates = getattr(interpreter, "_delegates", None)
    names = []
    if delegates:
        names = [type(item).__name__ for item in delegates]
    return {
        "op_resolver": HOST_REFERENCE_OP_RESOLVER,
        "input_dtype": str(np.dtype(details_in["dtype"])),
        "output_dtype": str(np.dtype(details_out["dtype"])),
        "input_shape": [int(v) for v in details_in["shape"]],
        "output_shape": [int(v) for v in details_out["shape"]],
        "input_quantization": [float(v) for v in details_in.get("quantization", (0, 0))],
        "output_quantization": [float(v) for v in details_out.get("quantization", (0, 0))],
        "delegate_objects_exposed": names,
    }


def historical_stage_check() -> dict:
    """Recompute published neural Stage 1 / Stage 2 rates from archived files."""
    joined, report = join_stages(load_stage1(), load_replay())
    rates = {}
    for backend, row in table3(joined).items():
        rows = [item for item in joined if item["backend"] == backend]
        n = len(rows)
        chain = sum(item["checkpoint"] == item["device"] for item in rows)
        rates[backend] = {
            "n": n,
            "stage1_count": sum(item["checkpoint"] == item["host_tflite_export"] for item in rows),
            "stage2_count": sum(item["host_tflite_replay"] == item["device"] for item in rows),
            "chain_count": chain,
            "stage1_percent": row["stage1_agree_pct"],
            "stage2_percent": row["stage2_agree_pct"],
            "chain_percent": 100.0 * chain / n if n else float("nan"),
        }
    return {"join": report, "rates": rates}


def _ptq_overlap(bundle: dict, held_out: str, fold: int) -> dict:
    y = bundle["y"]
    subjects = bundle["subjects"]
    calib = set(int(v) for v in calibration_global_indices(y, subjects, held_out))
    _train, inner, split_seed = strict_fold_train_inner_indices(
        y, subjects, held_out, list(SUBJECTS), 0.10, 42, fold,
    )
    inner_set = set(int(v) for v in inner)
    test_rows = {row for subject, _fold, _c, _t, row in bundle["identities"] if subject == held_out}
    return {
        "held_out_subject": held_out,
        "fold": fold,
        "n_calibration": len(calib),
        "n_inner_validation": len(inner_set),
        "split_seed": int(split_seed),
        "calibration_inner_overlap": len(calib & inner_set),
        "calibration_test_overlap": len(calib & test_rows),
    }


def _feature_cache_key(family: str, row: int) -> tuple[str, int]:
    return family, row


def run(args: argparse.Namespace) -> dict:
    import tensorflow as tf

    before = protected_hashes()
    bundle = load_processed_trials(Path(args.dataset_root))
    backends = list(args.backend) if args.backend else list(BACKENDS)
    unknown = [name for name in backends if name not in BACKENDS]
    if unknown:
        raise ValueError(f"Unknown backend {unknown}. Choose from {BACKENDS}")
    folds = [1] if args.smoke_test and not args.fold else (list(args.fold) if args.fold else list(range(1, 11)))
    if args.all_folds:
        folds = list(range(1, 11))
    trial_limit = 2 if args.smoke_test else None

    evidence = load_replay() if args.verify_archive else {}
    identities = [
        item for item in bundle["identities"]
        if item[1] in folds and (trial_limit is None or item[3] < trial_limit)
    ]
    identities.sort(key=lambda item: (item[1], item[2], item[3], item[0]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    interpreters = {}
    interpreter_meta = {}
    per_trial = []

    for backend in backends:
        family = _family(backend)
        path_a, path_b = FRONTEND_BY_FAMILY[family]
        archived = {}
        if args.verify_archive:
            archived = {
                (trial.subject, trial.class_id, trial.trial_idx): trial
                for trial in evidence[EVIDENCE_NAME[backend]]
            }
        for fold in folds:
            art = resolve_tflite(family, _precision(backend), fold, artifact_root=args.artifact_root, verify_checksum=True)
            if art["held_out_subject"] != f"s{fold}":
                raise ValueError(f"Fold {fold} artifact subject is {art['held_out_subject']}")
            interpreter = make_interpreter(art["path"], HOST_REFERENCE_OP_RESOLVER)
            interpreter.allocate_tensors()
            interpreters[(backend, fold)] = interpreter
            interpreter_meta[f"{backend}/fold{fold:02d}"] = {
                "sha256": art["sha256"],
                "relative_path": art["relative_path"],
                **_interpreter_record(interpreter),
            }
        for subject, fold, class_id, trial_idx, row in identities:
            if (backend, fold) not in interpreters:
                continue
            cache_key = _feature_cache_key(family, row)
            if cache_key not in feature_cache:
                trial = bundle["X"][row]
                feature_cache[cache_key] = (path_a(trial), path_b(trial))
            features_a, features_b = feature_cache[cache_key]
            feat = feature_pair_metrics(features_a, features_b)
            logits_a = _invoke(interpreters[(backend, fold)], features_a)
            logits_b = _invoke(interpreters[(backend, fold)], features_b)
            pred_a, margin_a = top1_and_margin(logits_a.tolist())
            pred_b, margin_b = top1_and_margin(logits_b.tolist())
            out = output_pair_metrics(logits_a.tolist(), logits_b.tolist())
            record = {
                "backend": backend,
                "evidence_backend": EVIDENCE_NAME[backend],
                "subject": subject,
                "fold": fold,
                "class_id": class_id,
                "trial_idx": trial_idx,
                "true_label": class_id,
                "feature_shape": "x".join(str(v) for v in features_a.shape),
                "feature_dtype": str(features_a.dtype),
                "feature_max_abs": feat["max_abs"],
                "feature_mean_abs": feat["mean_abs"],
                "feature_relative_l2": feat["relative_l2"],
                "feature_cosine_distance": feat["cosine_distance"],
                "feature_nonfinite": feat["nonfinite_count"],
                "output_max_abs": out["max_abs"],
                "output_mean_abs": out["mean_abs"],
                "output_relative_l2": out["relative_l2"],
                "output_cosine_distance": out["cosine_distance"],
                "prediction_a": pred_a,
                "prediction_b": pred_b,
                "margin_a": margin_a,
                "margin_b": margin_b,
                "flip_kind": classify_flip(pred_a, pred_b, class_id),
                "correct_a": int(pred_a == class_id),
                "correct_b": int(pred_b == class_id),
            }
            if args.verify_archive:
                ref = archived[(subject, class_id, trial_idx)]
                ref_pred, _ref_margin = top1_and_margin(ref.host)
                archive_error = output_pair_metrics(logits_b.tolist(), ref.host)
                record.update({
                    "archive_host_argmax": ref.host_argmax,
                    "archive_argmax_from_stored_logits": ref_pred,
                    "path_b_matches_archive_argmax": int(pred_b == ref.host_argmax),
                    "archive_output_max_abs": archive_error["max_abs"],
                    "archive_output_relative_l2": archive_error["relative_l2"],
                })
            per_trial.append(record)
        print(f"finished {backend}: {sum(r['backend'] == backend for r in per_trial)} trials", flush=True)

    ptq_rows = []
    ptq_overlap = []
    if not args.smoke_test:
        families = sorted({_family(name) for name in backends})
        for fold in folds:
            held_out = f"s{fold}"
            overlap = _ptq_overlap(bundle, held_out, fold)
            ptq_overlap.append(overlap)
            rows = calibration_global_indices(bundle["y"], bundle["subjects"], held_out)
            if trial_limit is not None:
                rows = rows[:trial_limit]
            for family in families:
                path_a, path_b = FRONTEND_BY_FAMILY[family]
                metrics = []
                for row in rows:
                    if int(bundle["subjects"][row] == held_out):
                        raise ValueError("PTQ row is a held-out test trial")
                    left, right = path_a(bundle["X"][row]), path_b(bundle["X"][row])
                    metrics.append(feature_pair_metrics(left, right))
                summary = summarize_numeric([item["relative_l2"] for item in metrics])
                ptq_rows.append({
                    "family": family,
                    "fold": fold,
                    "held_out_subject": held_out,
                    "n": len(metrics),
                    "mean_relative_l2": summary["mean"],
                    "p95_relative_l2": summary["p95"],
                    "max_abs": max(item["max_abs"] for item in metrics),
                    "nonfinite": sum(item["nonfinite_count"] for item in metrics),
                    "calibration_test_overlap": overlap["calibration_test_overlap"],
                    "calibration_inner_overlap": overlap["calibration_inner_overlap"],
                })

    after = protected_hashes()
    if before != after:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f"Protected artifact hash changed: {changed}")

    tables = _tables(per_trial, args.verify_archive)
    stage = historical_stage_check()
    _write_outputs(output_dir, per_trial, tables, ptq_rows, ptq_overlap, stage, interpreter_meta, bundle, args, before)
    manifest = json.loads((output_dir / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["tensorflow"] = tf.__version__
    (output_dir / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"trials": len(per_trial), "output_dir": str(output_dir)}, indent=2), flush=True)
    return {"per_trial": per_trial, "tables": tables, "stage": stage}


def _tables(rows: list[dict], verify_archive: bool) -> dict[str, list[dict]]:
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    feature_rows = []
    output_rows = []
    decision_rows = []
    subject_rows = []
    archive_rows = []
    for backend in BACKENDS:
        group = by_backend.get(backend, [])
        if not group:
            continue
        feat = summarize_numeric([row["feature_relative_l2"] for row in group])
        out = summarize_numeric([row["output_relative_l2"] for row in group])
        flips = [row for row in group if row["flip_kind"] != "agree"]
        feature_rows.append({
            "model": backend,
            "n": len(group),
            "mean_relative_l2": feat["mean"],
            "median_relative_l2": feat["median"],
            "p95_relative_l2": feat["p95"],
            "max_abs_error": max(row["feature_max_abs"] for row in group),
            "nonfinite_count": sum(row["feature_nonfinite"] for row in group),
        })
        output_rows.append({
            "backend": backend,
            "n": len(group),
            "mean_relative_output_l2": out["mean"],
            "p95_relative_output_l2": out["p95"],
            "max_abs_logit_error": max(row["output_max_abs"] for row in group),
        })
        decision_rows.append({
            "backend": backend,
            "n": len(group),
            "agreement_percent": 100.0 * (len(group) - len(flips)) / len(group),
            "flip_count": len(flips),
            "harmful": sum(row["flip_kind"] == "harmful" for row in group),
            "beneficial": sum(row["flip_kind"] == "beneficial" for row in group),
            "both_wrong": sum(row["flip_kind"] == "both_wrong" for row in group),
            "accuracy_a": 100.0 * sum(row["correct_a"] for row in group) / len(group),
            "accuracy_b": 100.0 * sum(row["correct_b"] for row in group) / len(group),
            "delta_accuracy_percentage_points": (
                100.0 * sum(row["correct_b"] for row in group) / len(group)
                - 100.0 * sum(row["correct_a"] for row in group) / len(group)
            ),
        })
        for subject in SUBJECTS:
            sub = [row for row in group if row["subject"] == subject]
            if not sub:
                continue
            sub_feat = summarize_numeric([row["feature_relative_l2"] for row in sub])
            subject_rows.append({
                "backend": backend,
                "subject": subject,
                "n": len(sub),
                "flips": sum(row["flip_kind"] != "agree" for row in sub),
                "accuracy_a": 100.0 * sum(row["correct_a"] for row in sub) / len(sub),
                "accuracy_b": 100.0 * sum(row["correct_b"] for row in sub) / len(sub),
                "mean_relative_feature_error": sub_feat["mean"],
            })
        if verify_archive:
            matched = sum(row["path_b_matches_archive_argmax"] for row in group)
            archive_rows.append({
                "backend": backend,
                "n": len(group),
                "archived_host_top1_matches": matched,
                "archived_host_agreement_percent": 100.0 * matched / len(group),
                "maximum_archived_vector_error": max(row["archive_output_max_abs"] for row in group),
                "mean_archived_relative_l2": summarize_numeric(
                    [row["archive_output_relative_l2"] for row in group]
                )["mean"],
            })
    return {
        "feature": feature_rows,
        "output": output_rows,
        "decision": decision_rows,
        "subject": subject_rows,
        "archive": archive_rows,
    }


def _write_outputs(output_dir, per_trial, tables, ptq_rows, ptq_overlap, stage, interpreter_meta, bundle, args, hashes) -> None:
    _write_csv(output_dir / "per_trial_diagnostics.csv", per_trial)
    _write_csv(output_dir / "RESULT_SUMMARY.csv", tables["decision"])
    _write_csv(output_dir / "SUBJECT_LEVEL_SUMMARY.csv", tables["subject"])
    _write_csv(output_dir / "table_a_feature_fidelity.csv", tables["feature"])
    _write_csv(output_dir / "table_b_output_fidelity.csv", tables["output"])
    _write_csv(output_dir / "table_c_top1_stability.csv", tables["decision"])
    _write_csv(output_dir / "table_d_subject_level.csv", tables["subject"])
    _write_csv(output_dir / "table_e_archive_parity.csv", tables["archive"])
    _write_csv(output_dir / "ptq_frontend_feature_summary.csv", ptq_rows)
    flips = [row for row in per_trial if row["flip_kind"] != "agree"]
    _write_csv(output_dir / "flip_trials.csv", flips)
    (output_dir / "historical_stage_check.json").write_text(json.dumps(stage, indent=2), encoding="utf-8")
    (output_dir / "ptq_overlap.json").write_text(json.dumps(ptq_overlap, indent=2), encoding="utf-8")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(),
        "smoke_test": bool(args.smoke_test),
        "verify_archive": bool(args.verify_archive),
        "backends": list(args.backend) if args.backend else list(BACKENDS),
        "command": _public_command(sys.argv),
        "dataset": {
            "name": "12JFPM",
            "file": bundle["npz_name"],
            "sha256": _sha256(Path(args.dataset_root) if Path(args.dataset_root).is_file() else _npz_from_root(Path(args.dataset_root))),
            "meta": bundle["meta"],
            "normalization": "processed channel z-score applied once by the dataset builder; ablation does not z-score again",
            "raw_mat_files": "not present in the local workspaces; processed trials were used",
        },
        "interpreter": interpreter_meta,
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        },
        "protected_sha256": hashes,
        "output_sha256": {},
    }
    (output_dir / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_hashes = {}
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "EXPERIMENT_MANIFEST.json":
            output_hashes[path.name] = _sha256(path)
    manifest["output_sha256"] = output_hashes
    (output_dir / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _public_command(argv: list[str]) -> list[str]:
    """Drop local paths so the manifest can be reviewed without publishing them."""
    redacted = []
    hide_value = False
    for arg in argv:
        if hide_value:
            redacted.append("<path>")
            hide_value = False
            continue
        redacted.append(arg)
        hide_value = arg in {"--dataset-root", "--artifact-root", "--output-dir"}
    return redacted


def _npz_from_root(root: Path) -> Path:
    from analysis.frontend_ablation.dataset import resolve_processed_npz
    return resolve_processed_npz(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-versus-deployment frontend ablation on frozen TFLite graphs.")
    parser.add_argument("--dataset-root", required=True, help="Directory or processed.npz for 12JFPM. No synthetic fallback.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Retained artifact root. Defaults to artifacts/reference.")
    parser.add_argument("--output-dir", required=True, help="Directory for new ablation tables. Existing paper results are not written here by the pipeline.")
    parser.add_argument("--backend", action="append", default=[], help="Repeat for ccnn_f32, ccnn_int8, fbtcnn_f32, fbtcnn_int8.")
    parser.add_argument("--fold", type=int, action="append", default=[], help="Repeat for a 1-based LOSO fold.")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate folds 1..10.")
    parser.add_argument("--verify-archive", action="store_true", help="Compare Path B with archived host replay vectors.")
    parser.add_argument("--smoke-test", action="store_true", help="Fold 1, first two trials, unless --fold is set.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.fold and not args.all_folds and not args.smoke_test:
        raise SystemExit("Specify --fold, --all-folds, or --smoke-test.")
    try:
        run(args)
    except FileNotFoundError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc


if __name__ == "__main__":
    main()
