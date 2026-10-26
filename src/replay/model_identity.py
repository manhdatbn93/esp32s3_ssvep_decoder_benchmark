"""TFLite model file identity (CRC32/SHA256/byte probes) — zlib-compatible CRC32."""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import TypedDict


class ModelIdentity(TypedDict):
    path: str
    size: int
    sha256: str
    crc32: int
    first16: str
    last16: str


def crc32_bytes(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def identity_from_bytes(data: bytes, *, path: str = "") -> ModelIdentity:
    n = len(data)
    return {
        "path": path,
        "size": n,
        "sha256": hashlib.sha256(data).hexdigest(),
        "crc32": crc32_bytes(data),
        "first16": data[:16].hex() if n else "",
        "last16": data[-16:].hex() if n >= 16 else data.hex(),
    }


def identity_from_file(path: Path) -> ModelIdentity:
    data = path.read_bytes()
    return identity_from_bytes(data, path=str(path.resolve()))


def format_crc32(crc: int) -> str:
    return f"0x{crc:08x}"


def log_model_identity(prefix: str, ident: ModelIdentity, *, backend: str = "",
                       subject: str = "", fold: int = 0, protocol: str = "") -> None:
    p = prefix
    if backend:
        print(f"{p} backend={backend}")
    if subject or fold or protocol:
        print(f"{p} subject={subject} fold={fold} protocol={protocol}")
    print(f"{p} path={ident['path']}")
    print(f"{p} size={ident['size']}")
    print(f"{p} sha256={ident['sha256']}")
    print(f"{p} crc32={format_crc32(ident['crc32'])}")
    print(f"{p} first16={ident['first16']}")
    print(f"{p} last16={ident['last16']}")


def identities_match(host: ModelIdentity, fw: ModelIdentity) -> bool:
    return (
        host["size"] == fw["size"]
        and host["crc32"] == fw["crc32"]
        and host["first16"] == fw["first16"]
        and host["last16"] == fw["last16"]
    )


class ModelIdentityMismatchError(RuntimeError):
    """Host and firmware embedded TFLite bytes differ."""


def assert_eegnet_int8_identity(
    host_ref,
    fw_payload,
    *,
    requested_backend: str = "eegnet_int8",
) -> None:
    """
    Hard-fail if host HostReference and firmware START_ACK identity disagree.
    fw_payload: replay_proto.ModelIdentityPayload or None.
    """
    key = requested_backend.strip().lower().replace("-", "_")
    if key not in ("eegnet_int8", "eegnet_i8"):
        return

    if fw_payload is None:
        raise ModelIdentityMismatchError(
            "MODEL_IDENTITY_MISMATCH: firmware sent no START_ACK model identity.\n"
            "  Restage EEGNet-int8 for the intended fold, then rebuild and flash. "
            "See docs/firmware-and-replay.md."
        )

    from replay_proto import DecoderID  # local import avoids cycle at module load

    if fw_payload.requested_decoder_id != int(DecoderID.EEGNET_I8):
        raise ModelIdentityMismatchError(
            f"MODEL_IDENTITY_MISMATCH: requested_decoder_id="
            f"{fw_payload.requested_decoder_id} (expected EEGNET_I8={int(DecoderID.EEGNET_I8)})"
        )
    if fw_payload.actual_decoder_id != int(DecoderID.EEGNET_I8):
        raise ModelIdentityMismatchError(
            f"MODEL_IDENTITY_MISMATCH: actual_decoder_id="
            f"{fw_payload.actual_decoder_id} (expected EEGNET_I8)"
        )
    if "float32" in fw_payload.symbol.lower():
        raise ModelIdentityMismatchError(
            f"MODEL_IDENTITY_MISMATCH: firmware symbol is f32, not int8: {fw_payload.symbol!r}"
        )
    if "int8" not in fw_payload.symbol.lower():
        raise ModelIdentityMismatchError(
            f"MODEL_IDENTITY_MISMATCH: firmware symbol missing int8: {fw_payload.symbol!r}"
        )

    host_crc = host_ref.model_crc32
    host_size = host_ref.model_size
    host_f16 = host_ref.model_first16
    host_l16 = host_ref.model_last16
    fw_crc = fw_payload.model_crc32
    fw_size = fw_payload.model_size
    fw_f16 = fw_payload.first16.hex()
    fw_l16 = fw_payload.last16.hex()

    print(f"HOST_INT8_CRC32 = {format_crc32(host_crc)}")
    print(f"FW_INT8_CRC32   = {format_crc32(fw_crc)}")

    if (
        host_size == fw_size
        and host_crc == fw_crc
        and host_f16 == fw_f16
        and host_l16 == fw_l16
    ):
        print("MODEL_IDENTITY_OK: host and firmware use the same eegnet_int8 bytes")
        return

    lines = [
        "MODEL_IDENTITY_MISMATCH:",
        f"  host_crc32 = {format_crc32(host_crc)}",
        f"  fw_crc32   = {format_crc32(fw_crc)}",
        f"  host_size  = {host_size}",
        f"  fw_size    = {fw_size}",
        f"  host_first16 = {host_f16}",
        f"  fw_first16   = {fw_f16}",
        f"  host_last16  = {host_l16}",
        f"  fw_last16    = {fw_l16}",
        f"  fw_symbol    = {fw_payload.symbol}",
        f"  fw_tag       = {fw_payload.embed_tag}",
        "  Restage EEGNet-int8 for the intended fold, then rebuild and flash. "
        "See docs/firmware-and-replay.md.",
    ]
    raise ModelIdentityMismatchError("\n".join(lines))
