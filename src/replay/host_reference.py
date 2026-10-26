"""
Host-side reference inference for replay parity checks.

For each backend, computes the same scores/logits that the PC Python
implementation would produce on a preprocessed trial [512, 8] float32.

Classical backends call src/benchmarks/*.py directly.
DL backends run the exported TFLite model via tf.lite.Interpreter,
with model-specific preprocessing matching the firmware contract:
  - EEGNet: transpose [T,C] -> [C,T]
  - CCNN:   Goertzel FFT -> [C, 220] spectrum features
  - ConvCA: direct feed (templates baked into model)
  - FBtCNN: 4-band Butterworth filtfilt -> [C, T, 4]
TRCA uses precomputed reference NPZ (generate with generate_trca_reference.py).

Usage:
    ref = HostReference("cca")
    host_scores = ref.run(trial_tc)   # [12] float32
    ref.cleanup()

    # TRCA needs subject to locate the correct precomputed reference:
    ref = HostReference("trca", subject="s1")
    host_scores = ref.run(trial_tc, class_id=0, trial_idx=0)  # [12] float32
    ref.cleanup()
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_REPLAY = Path(__file__).resolve().parent
for _p in (_REPO, _SRC, _REPLAY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from export.protocol_cli import DEFAULT_PROTOCOL, warn_if_subject_as_val  # noqa: E402
from export.subject_paths import (  # noqa: E402
    PROTOCOL_STRICT_LOSO_INNER_VAL,
    PROTOCOL_SUBJECT_AS_VAL,
    assert_not_legacy_artifact,
    export_dir,
    export_dir_name,
    fold_for_subject,
    tag_for,
)

from model_identity import (  # noqa: E402
    format_crc32,
    identity_from_file,
    log_model_identity,
)
from host_interpreter import HOST_REFERENCE_OP_RESOLVER, make_interpreter  # noqa: E402


def _preprocess_zscore(trial_tc: np.ndarray) -> np.ndarray:
    """Channel-wise z-score — mirrors on-device preprocess_trial (CHANNEL_ZSCORE)."""
    x = trial_tc.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return ((x - mean) / std).astype(np.float32)


# ── Export folder + tflite filename conventions ─────────────────────
#
# Float32 models live in:  export/12jfpm_<model>_fold{NN}_val_{sN}/<model>_float32.tflite
# Int8 models live in:     export/12jfpm_<model>_fold{NN}_val_{sN}_int8/<model>_int8.tflite
#
# The resolver composes those at runtime using export.subject_paths so
# the same HostReference class works for any subject/fold without duplication.

# key -> (base_model, variant, tflite_filename)
_TFLITE_SPEC: dict[str, tuple[str, str, str]] = {
    "eegnet_f32":     ("eegnet", "f32",  "eegnet_float32.tflite"),
    "eegnet_float32": ("eegnet", "f32",  "eegnet_float32.tflite"),
    "eegnet_int8":    ("eegnet", "int8", "eegnet_int8.tflite"),
    "eegnet_i8":      ("eegnet", "int8", "eegnet_int8.tflite"),
    "ccnn_f32":       ("ccnn",   "f32",  "ccnn_float32.tflite"),
    "ccnn_float32":   ("ccnn",   "f32",  "ccnn_float32.tflite"),
    "ccnn_int8":      ("ccnn",   "int8", "ccnn_int8.tflite"),
    "ccnn_i8":        ("ccnn",   "int8", "ccnn_int8.tflite"),
    "convca_f32":     ("convca", "f32",  "convca_float32.tflite"),
    "convca_float32": ("convca", "f32",  "convca_float32.tflite"),
    "fbtcnn_f32":     ("fbtcnn", "f32",  "fbtcnn_float32.tflite"),
    "fbtcnn_float32": ("fbtcnn", "f32",  "fbtcnn_float32.tflite"),
    "fbtcnn_int8":    ("fbtcnn", "int8", "fbtcnn_int8.tflite"),
    "fbtcnn_i8":      ("fbtcnn", "int8", "fbtcnn_int8.tflite"),
}

# Model family determines which preprocessing to apply before TFLite inference.
_MODEL_FAMILY: dict[str, str] = {
    "eegnet_f32": "eegnet", "eegnet_float32": "eegnet",
    "eegnet_int8": "eegnet", "eegnet_i8": "eegnet",
    "ccnn_f32": "ccnn", "ccnn_float32": "ccnn",
    "ccnn_int8": "ccnn", "ccnn_i8": "ccnn",
    "convca_f32": "convca", "convca_float32": "convca",
    "fbtcnn_f32": "fbtcnn", "fbtcnn_float32": "fbtcnn",
    "fbtcnn_int8": "fbtcnn", "fbtcnn_i8": "fbtcnn",
}


# ── CCNN FFT preprocessing (Goertzel, mirrors ccnn_preprocess.c) ───

_CCNN_NFFT = 874
_CCNN_INPUT_T = 512
_CCNN_INPUT_C = 8
_CCNN_BIN_LO = 10
_CCNN_BIN_HI = 120
_CCNN_N_BINS = _CCNN_BIN_HI - _CCNN_BIN_LO  # 110
_CCNN_N_FEATURES = 2 * _CCNN_N_BINS  # 220
_CCNN_NORM_FACTOR = _CCNN_INPUT_T / 2.0  # 256.0


def _ccnn_preprocess(trial_tc: np.ndarray) -> np.ndarray:
    """Goertzel DFT matching firmware ccnn_preprocess.c.

    Input:  [512, 8] float32 (time-channel, z-scored)
    Output: [1, 8, 220, 1] float32 (NHWC for TFLite)
    """
    tc = trial_tc.astype(np.float64)
    T, C = tc.shape
    assert T == _CCNN_INPUT_T and C == _CCNN_INPUT_C

    omegas = 2.0 * math.pi * np.arange(_CCNN_BIN_LO, _CCNN_BIN_HI) / _CCNN_NFFT
    cos_w = np.cos(omegas)
    sin_w = np.sin(omegas)
    coeff = 2.0 * cos_w

    output = np.zeros((C, _CCNN_N_FEATURES), dtype=np.float64)

    for c in range(C):
        s1 = np.zeros(_CCNN_N_BINS)
        s2 = np.zeros(_CCNN_N_BINS)

        for n in range(T):
            x = tc[n, c]
            s0 = x + coeff * s1 - s2
            s2 = s1
            s1 = s0

        for n in range(T, _CCNN_NFFT):
            s0 = coeff * s1 - s2
            s2 = s1
            s1 = s0

        real = (cos_w * s1 - s2) / _CCNN_NORM_FACTOR
        imag = (sin_w * s1) / _CCNN_NORM_FACTOR
        output[c, :_CCNN_N_BINS] = real
        output[c, _CCNN_N_BINS:] = imag

    return output.reshape(1, C, _CCNN_N_FEATURES, 1).astype(np.float32)


# ── FBtCNN filter-bank preprocessing (mirrors fbtcnn_preprocess.c) ──

_FBTCNN_T = 512
_FBTCNN_C = 8
_FBTCNN_BANDS = [(3, 14), (9, 26), (14, 38), (19, 50)]
_FBTCNN_FS = 256.0


def _fbtcnn_preprocess(trial_tc: np.ndarray) -> np.ndarray:
    """4-band Butterworth filtfilt matching firmware fbtcnn_preprocess.c.

    Input:  [512, 8] float32 (time-channel, z-scored)
    Output: [1, 8, 512, 4] float32 (NHWC for TFLite)
    """
    from scipy import signal as scipy_signal

    tc = trial_tc.astype(np.float64)
    T, C = tc.shape
    assert T == _FBTCNN_T and C == _FBTCNN_C

    output = np.zeros((C, T, len(_FBTCNN_BANDS)), dtype=np.float64)

    for bi, (lo, hi) in enumerate(_FBTCNN_BANDS):
        sos = scipy_signal.butter(
            6, [2 * lo / _FBTCNN_FS, 2 * hi / _FBTCNN_FS], btype="bandpass", output="sos",
        )
        for c in range(C):
            ch = tc[:, c]
            filtered = scipy_signal.sosfiltfilt(sos, ch)
            output[c, :, bi] = filtered

    return output.reshape(1, C, T, len(_FBTCNN_BANDS)).astype(np.float32)


# ── EEGNet preprocessing (transpose only) ──────────────────────────

def _eegnet_preprocess(trial_tc: np.ndarray) -> np.ndarray:
    """Transpose [T, C] -> [1, C, T] matching firmware eegnet_smoke.c."""
    return trial_tc.astype(np.float32).T[np.newaxis, ...]  # [1, C, T]


# ── ConvCA preprocessing (none — direct feed) ──────────────────────

def _convca_preprocess(trial_tc: np.ndarray) -> np.ndarray:
    """Direct feed [1, T, C] — templates baked into model."""
    return trial_tc.astype(np.float32)[np.newaxis, ...]  # [1, T, C]


_PREPROCESS_FN = {
    "eegnet": _eegnet_preprocess,
    "ccnn": _ccnn_preprocess,
    "convca": _convca_preprocess,
    "fbtcnn": _fbtcnn_preprocess,
}


class HostReference:
    """Dispatch host-side inference for a given backend name."""

    def __init__(self, backend_name: str, *,
                 subject: str = "s1", fold: int | None = None,
                 protocol: str = DEFAULT_PROTOCOL):
        key = backend_name.strip().lower().replace("-", "_").replace(" ", "_")
        self._key = key
        self._name = backend_name
        # Derive fold from subject under the LOSO convention unless explicit.
        if fold is None:
            try:
                fold = fold_for_subject(subject)
            except Exception:
                fold = 1
        self._subject = subject
        self._fold = fold
        self._protocol = protocol
        if protocol == PROTOCOL_SUBJECT_AS_VAL:
            warn_if_subject_as_val(protocol)
        self._runner = None
        self._tflite_interp = None
        self._trca_ref = None
        self._uses_indexed = False
        self._artifact_path: str | None = None
        self._invoke_count = 0
        self.model_path: str | None = None
        self.model_size: int | None = None
        self.model_sha256: str | None = None
        self.model_crc32: int | None = None
        self.model_first16: str | None = None
        self.model_last16: str | None = None

        if key == "cca":
            from src.benchmarks.cca_12jfpm import cca_scores_for_trial
            self._runner = lambda tc: cca_scores_for_trial(tc)
            self._artifact_path = "live: src.benchmarks.cca_12jfpm"

        elif key == "fbcca":
            from src.benchmarks.fbcca_12jfpm import fbcca_scores_for_trial
            self._runner = lambda tc: fbcca_scores_for_trial(tc)
            self._artifact_path = "live: src.benchmarks.fbcca_12jfpm"

        elif key == "msi":
            from src.benchmarks.msi_12jfpm import msi_scores_for_trial
            self._runner = lambda tc: msi_scores_for_trial(tc)
            self._artifact_path = "live: src.benchmarks.msi_12jfpm"

        elif key == "trca":
            self._init_trca(subject, fold, protocol=protocol)

        elif key in _TFLITE_SPEC:
            self._init_tflite(key, subject=subject, fold=fold, protocol=protocol)

        else:
            raise ValueError(
                f"No host reference available for backend {backend_name!r}. "
                f"Supported: cca, fbcca, msi, trca, eegnet_f32/i8, ccnn_f32/i8, "
                f"convca_f32, fbtcnn_f32/i8"
            )

    def _init_trca(self, subject: str, fold: int, *, protocol: str) -> None:
        """Load precomputed TRCA reference NPZ for the held-out subject."""
        ref_dir = export_dir("trca", subject, fold, protocol=protocol, repo=_REPO)
        ref_path = ref_dir / "replay_reference.npz"
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"TRCA replay reference not found: {ref_path}\n"
                f"  Generate it with:\n"
                f"    python tools/replay/generate_trca_reference.py "
                f"--subject {subject} --fold {fold}"
            )
        d = np.load(str(ref_path), allow_pickle=True)
        scores = d["scores"]  # (180, 12)
        class_ids = d["class_ids"]  # (180,)
        trial_idxs = d["trial_idxs"]  # (180,)

        self._trca_ref = {}
        for i in range(len(class_ids)):
            key = (int(class_ids[i]), int(trial_idxs[i]))
            self._trca_ref[key] = scores[i].astype(np.float32)

        self._uses_indexed = True
        self._runner = self._run_trca_indexed
        self._trca_ref_path = ref_path
        self._artifact_path = str(ref_path)

    def _run_trca_indexed(self, trial_tc: np.ndarray,
                          class_id: int = -1, trial_idx: int = -1) -> np.ndarray:
        key = (class_id, trial_idx)
        if key not in self._trca_ref:
            raise KeyError(
                f"No TRCA reference for class_id={class_id}, trial_idx={trial_idx}. "
                f"Reference has {len(self._trca_ref)} entries."
            )
        return self._trca_ref[key]

    def _find_tflite(self, key: str, *, subject: str, fold: int,
                     protocol: str = DEFAULT_PROTOCOL) -> Path:
        """Locate the correct .tflite file for subject/fold using subject_paths."""
        base_model, variant, filename = _TFLITE_SPEC[key]
        tflite_path = export_dir(
            base_model, subject, fold, int8=(variant == "int8"),
            protocol=protocol, repo=_REPO,
        ) / filename
        assert_not_legacy_artifact(tflite_path, protocol=protocol, context="host reference")
        folder = export_dir_name(base_model, subject, fold,
                                 int8=(variant == "int8"), protocol=protocol)
        if not tflite_path.is_file():
            raise FileNotFoundError(
                f"TFLite model not found: {tflite_path}\n"
                f"  Expected: export/{folder}/{filename}\n"
                f"  Re-run the export pipeline for this subject/backend:\n"
                f"    python -m tools.export.export_all_subjects "
                f"--subjects {subject} --backends {base_model}"
                f"{'_int8' if variant == 'int8' else ''}"
            )
        return tflite_path

    def _init_tflite(self, key: str, *, subject: str = "s1", fold: int = 1,
                     protocol: str = DEFAULT_PROTOCOL) -> None:
        tflite_path = self._find_tflite(key, subject=subject, fold=fold, protocol=protocol)
        base_model, variant, filename = _TFLITE_SPEC[key]
        if key in ("eegnet_int8", "eegnet_i8"):
            if filename != "eegnet_int8.tflite":
                raise ValueError(f"EEGNet-int8 host path must use eegnet_int8.tflite, got {filename}")
            if variant != "int8":
                raise ValueError("EEGNet-int8 host resolver variant must be int8")
            norm_path = str(tflite_path.resolve()).replace("\\", "/")
            if protocol == PROTOCOL_STRICT_LOSO_INNER_VAL:
                expected_tag = tag_for(subject, fold, protocol=protocol)
                if f"{expected_tag}_int8/eegnet_int8.tflite" not in norm_path:
                    raise ValueError(
                        f"EEGNet-int8 strict path mismatch:\n  got: {tflite_path}\n"
                        f"  expected tag {expected_tag} and eegnet_int8.tflite under strict_loso_inner_val"
                    )
            if "_val_s" in norm_path and "_test_s" not in norm_path:
                raise ValueError(f"EEGNet-int8 must not use legacy fold*_val_s* path: {tflite_path}")
            if "float32" in norm_path:
                raise ValueError(f"EEGNet-int8 host path must not be float32: {tflite_path}")
        if key in ("eegnet_f32", "eegnet_float32"):
            if "eegnet_int8" in str(tflite_path) or filename == "eegnet_int8.tflite":
                raise ValueError(f"EEGNet-f32 host path must not be int8: {tflite_path}")

        ident = identity_from_file(tflite_path)
        self.model_path = ident["path"]
        self.model_size = ident["size"]
        self.model_sha256 = ident["sha256"]
        self.model_crc32 = ident["crc32"]
        self.model_first16 = ident["first16"]
        self.model_last16 = ident["last16"]
        log_model_identity(
            "[host_ref:model]",
            ident,
            backend=key,
            subject=subject,
            fold=fold,
            protocol=protocol,
        )

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "tensorflow is required for DL host reference. "
                "Install with: pip install tensorflow"
            )

        family = _MODEL_FAMILY[key]
        self._preprocess_fn = _PREPROCESS_FN[family]
        self._family = family

        self.op_resolver = HOST_REFERENCE_OP_RESOLVER
        self._tflite_interp = make_interpreter(tflite_path, self.op_resolver)
        self._tflite_interp.allocate_tensors()
        self._tflite_path = tflite_path
        self._artifact_path = str(tflite_path)
        self._runner = self._run_tflite

    @property
    def model_crc32_hex(self) -> str | None:
        if self.model_crc32 is None:
            return None
        return format_crc32(self.model_crc32)

    def _run_tflite(self, trial_tc: np.ndarray) -> np.ndarray:
        feed = self._preprocess_fn(trial_tc)

        interp = self._tflite_interp
        inp_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]

        expected_shape = tuple(inp_detail["shape"])
        if feed.shape != expected_shape:
            feed = feed.reshape(expected_shape)

        if self._invoke_count == 0:
            flat = feed.flatten()
            print(f"  [host_ref] invoke#1: input[0..3] = "
                  f"{flat[0]:.6f} {flat[1]:.6f} {flat[2]:.6f} {flat[3]:.6f}")
            print(f"  [host_ref] in_dtype={inp_detail['dtype']} "
                  f"out_dtype={out_detail['dtype']} "
                  f"in_quant={inp_detail.get('quantization')} "
                  f"out_quant={out_detail.get('quantization')}")

        if inp_detail["dtype"] in (np.int8, np.uint8):
            s, z = inp_detail["quantization"]
            feed_q = np.clip(np.round(feed / s + z),
                             np.iinfo(inp_detail["dtype"]).min,
                             np.iinfo(inp_detail["dtype"]).max).astype(inp_detail["dtype"])
            if self._invoke_count == 0:
                flat_q = feed_q.flatten()
                print(f"  [host_ref] invoke#1: quant scale={s:.8f} zp={z}  "
                      f"q[0..3]={flat_q[0]} {flat_q[1]} {flat_q[2]} {flat_q[3]}")
            feed = feed_q

        interp.set_tensor(inp_detail["index"], feed)
        interp.invoke()
        raw_out = interp.get_tensor(out_detail["index"])

        if out_detail["dtype"] in (np.int8, np.uint8):
            s, z = out_detail["quantization"]
            raw_out = (raw_out.astype(np.float32) - z) * s

        result = raw_out.flatten().astype(np.float32)

        if self._invoke_count == 0:
            print(f"  [host_ref] invoke#1: out[0..3] = "
                  f"{result[0]:.6f} {result[1]:.6f} {result[2]:.6f} {result[3]:.6f}")

        self._invoke_count += 1
        return result

    @property
    def available(self) -> bool:
        return self._runner is not None

    @property
    def name(self) -> str:
        return self._name

    @property
    def artifact_path(self) -> str | None:
        """Resolved path or description of the host reference artifact."""
        return self._artifact_path

    @property
    def uses_indexed(self) -> bool:
        """True for backends (like TRCA) that use precomputed per-trial references."""
        return self._uses_indexed

    def run(self, trial_tc: np.ndarray, *,
            class_id: int = -1, trial_idx: int = -1) -> np.ndarray:
        """Run host reference. Returns [12] float32.

        For most backends, trial_tc is the preprocessed [512, 8] input.
        For TRCA (indexed mode), class_id and trial_idx select the
        precomputed reference; trial_tc is ignored.
        """
        if self._runner is None:
            raise RuntimeError(f"No host reference for {self._name}")
        if self._uses_indexed:
            scores = self._runner(trial_tc, class_id=class_id, trial_idx=trial_idx)
        else:
            scores = self._runner(trial_tc)
        return np.asarray(scores, dtype=np.float32).ravel()

    def cleanup(self):
        self._tflite_interp = None
        self._trca_ref = None
        self._runner = None
