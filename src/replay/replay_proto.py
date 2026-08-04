"""
Binary framing for dataset replay — keep in sync with
embedded/replay_fw/main/replay_proto.h (magic, version, types, struct layouts).

Protocol version 2.0 — supports 11 backends (CCA through FBtCNN-int8).

START_ACK may carry an optional replay_model_identity_payload_t (116 B) for
embedded DL backends (EEGNet f32/int8) so the host can verify firmware model bytes.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

REPLAY_MAGIC_U32 = 0x52425032  # 'RBP2'
REPLAY_VERSION_MAJOR = 2
REPLAY_VERSION_MINOR = 0

# Wire header: magic u32, ver_maj u16, ver_min u16, type u16, flags u16, payload_len u32
HEADER_PACK = struct.Struct("<IHHHHI")
HEADER_SIZE = HEADER_PACK.size  # 16 bytes


class PktType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    START_RUN = 3
    START_ACK = 4
    TRIAL_DATA = 5
    RESULT = 6
    END_RUN = 7
    END_ACK = 8
    ERROR = 9


class NormMode(IntEnum):
    NONE = 0
    CHANNEL_ZSCORE = 1
    GLOBAL_MINMAX = 2


class DecoderID(IntEnum):
    CCA = 0
    FBCCA = 1
    MSI = 2
    TRCA = 3
    EEGNET_F32 = 4
    EEGNET_I8 = 5
    CCNN_F32 = 6
    CCNN_I8 = 7
    CONVCA_F32 = 8
    FBTCNN_F32 = 9
    FBTCNN_I8 = 10


DECODER_NAMES: dict[int, str] = {
    DecoderID.CCA: "CCA",
    DecoderID.FBCCA: "FBCCA",
    DecoderID.MSI: "MSI",
    DecoderID.TRCA: "TRCA",
    DecoderID.EEGNET_F32: "EEGNet-f32",
    DecoderID.EEGNET_I8: "EEGNet-int8",
    DecoderID.CCNN_F32: "CCNN-f32",
    DecoderID.CCNN_I8: "CCNN-int8",
    DecoderID.CONVCA_F32: "ConvCA-f32",
    DecoderID.FBTCNN_F32: "FBtCNN-f32",
    DecoderID.FBTCNN_I8: "FBtCNN-int8",
}


def decoder_id_from_name(name: str) -> int:
    """Resolve a human-friendly backend name to a wire decoder ID."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    _MAP = {
        "cca": DecoderID.CCA,
        "fbcca": DecoderID.FBCCA,
        "msi": DecoderID.MSI,
        "trca": DecoderID.TRCA,
        "eegnet_f32": DecoderID.EEGNET_F32,
        "eegnet_float32": DecoderID.EEGNET_F32,
        "eegnet_int8": DecoderID.EEGNET_I8,
        "eegnet_i8": DecoderID.EEGNET_I8,
        "ccnn_f32": DecoderID.CCNN_F32,
        "ccnn_float32": DecoderID.CCNN_F32,
        "ccnn_int8": DecoderID.CCNN_I8,
        "ccnn_i8": DecoderID.CCNN_I8,
        "convca_f32": DecoderID.CONVCA_F32,
        "convca_float32": DecoderID.CONVCA_F32,
        "fbtcnn_f32": DecoderID.FBTCNN_F32,
        "fbtcnn_float32": DecoderID.FBTCNN_F32,
        "fbtcnn_int8": DecoderID.FBTCNN_I8,
        "fbtcnn_i8": DecoderID.FBTCNN_I8,
    }
    if key not in _MAP:
        raise ValueError(
            f"Unknown backend name {name!r}. "
            f"Expected one of: {', '.join(sorted(_MAP))}"
        )
    return int(_MAP[key])


def pack_header(pkt_type: int, payload_len: int, flags: int = 0) -> bytes:
    return HEADER_PACK.pack(
        REPLAY_MAGIC_U32,
        REPLAY_VERSION_MAJOR,
        REPLAY_VERSION_MINOR,
        pkt_type,
        flags,
        payload_len,
    )


def unpack_header(buf: bytes) -> tuple:
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"need {HEADER_SIZE} bytes for header, got {len(buf)}")
    return HEADER_PACK.unpack(buf[:HEADER_SIZE])


# replay_start_payload_t — 16 bytes
START_RUN_PACK = struct.Struct("<B B H H B B 8s")


def pack_start_run(
    norm: int = int(NormMode.CHANNEL_ZSCORE),
    onset: int = 38,
    window: int = 512,
    fs_hz: int = 256,
    n_channels: int = 8,
    decoder_id: int = 0,
) -> bytes:
    return START_RUN_PACK.pack(
        norm & 0xFF,
        onset & 0xFF,
        window & 0xFFFF,
        fs_hz & 0xFFFF,
        n_channels & 0xFF,
        decoder_id & 0xFF,
        b"\0" * 8,
    )


# replay_trial_meta_t — 22 bytes
TRIAL_META_PACK = struct.Struct("<8sBBHBBHB5s")


@dataclass
class TrialMeta:
    subject: str
    class_id: int
    trial_idx: int
    n_time: int
    n_channels: int
    fs_hz: int
    normalize_mode: int


def pack_trial_meta(m: TrialMeta) -> bytes:
    subj = m.subject.encode("ascii")[:8].ljust(8, b"\0")
    return TRIAL_META_PACK.pack(
        subj,
        m.class_id & 0xFF,
        m.trial_idx & 0xFF,
        m.n_time & 0xFFFF,
        m.n_channels & 0xFF,
        0,
        m.fs_hz & 0xFFFF,
        m.normalize_mode & 0xFF,
        b"\0" * 5,
    )


def unpack_trial_meta(b: bytes) -> TrialMeta:
    subj, cid, tid, nt, nc, _r0, fs, norm, _r1 = TRIAL_META_PACK.unpack(b[:22])
    return TrialMeta(
        subject=subj.split(b"\0")[0].decode("ascii", errors="replace"),
        class_id=cid,
        trial_idx=tid,
        n_time=nt,
        n_channels=nc,
        fs_hz=fs,
        normalize_mode=norm,
    )


# replay_result_payload_t: 22 (meta) + 1 + 1 + 2 + 4 + 48 = 78 bytes
RESULT_EXTRA_PACK = struct.Struct("<BBHI12f")


def unpack_result_payload(payload: bytes) -> tuple[TrialMeta, int, int, int, list[float]]:
    """Returns (meta, decoder_id, status, elapsed_us, logits[12])."""
    meta = unpack_trial_meta(payload[:22])
    dec_id, status, _res, elapsed_us, *logits = RESULT_EXTRA_PACK.unpack(
        payload[22 : 22 + RESULT_EXTRA_PACK.size]
    )
    return meta, dec_id, status, elapsed_us, list(logits)


# replay_model_identity_payload_t — 104 bytes (optional START_ACK body)
MODEL_IDENTITY_PACK = struct.Struct("<BBBBII16s16s24s48s")


@dataclass
class ModelIdentityPayload:
    requested_decoder_id: int
    actual_decoder_id: int
    input_dtype: int
    output_dtype: int
    model_size: int
    model_crc32: int
    first16: bytes
    last16: bytes
    embed_tag: str
    symbol: str


def unpack_model_identity(payload: bytes) -> ModelIdentityPayload:
    if len(payload) < MODEL_IDENTITY_PACK.size:
        raise ValueError(
            f"model identity payload need {MODEL_IDENTITY_PACK.size} bytes, got {len(payload)}"
        )
    (req, act, in_dt, out_dt, size, crc, f16, l16, tag_b, sym_b) = MODEL_IDENTITY_PACK.unpack(
        payload[: MODEL_IDENTITY_PACK.size]
    )
    return ModelIdentityPayload(
        requested_decoder_id=req,
        actual_decoder_id=act,
        input_dtype=in_dt,
        output_dtype=out_dt,
        model_size=size,
        model_crc32=crc,
        first16=f16,
        last16=l16,
        embed_tag=tag_b.split(b"\0", 1)[0].decode("ascii", errors="replace"),
        symbol=sym_b.split(b"\0", 1)[0].decode("ascii", errors="replace"),
    )


def unpack_error_payload(payload: bytes) -> tuple[int, str]:
    if len(payload) < 4:
        return -1, ""
    code = struct.unpack_from("<I", payload, 0)[0]
    tail = payload[4:100] if len(payload) >= 100 else payload[4:]
    msg = tail.split(b"\0", 1)[0].decode("ascii", errors="replace")
    return int(code), msg


def recv_exact(sock, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        try:
            b = sock.recv(n - got)
        except OSError as e:
            raise ConnectionError(f"recv failed: {e!r}") from e
        if not b:
            raise ConnectionError("peer closed")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)
