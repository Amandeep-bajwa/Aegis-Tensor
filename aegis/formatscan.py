"""Header-level checks for .safetensors files.

The safetensors library already rejects malformed structure: overlapping or
out-of-range offsets, trailing bytes and truncated headers. What it accepts
without complaint is free-form data inside the JSON header: the ``__metadata__``
map, tensor names and whitespace. Each of these can carry a hidden payload, so
this module reads the header directly and checks those channels.
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from aegis.secrets import find_secrets

# Real metadata is a few short strings such as a format tag or a generator name.
MAX_METADATA_VALUE_BYTES = 1024
MAX_METADATA_TOTAL_BYTES = 4096
# Padding for 8-byte alignment is at most 7 bytes. Anything beyond a small slack is unexplained.
MAX_HEADER_SLACK_BYTES = 16
MAX_TENSOR_NAME_BYTES = 256


@dataclass
class FormatFinding:
    severity: str  # SUSPICIOUS
    channel: str  # metadata | header_whitespace | tensor_name
    where: str  # pinpoints the key, tensor or byte count
    message: str
    treatment: str  # sanitize (strip metadata or rewrite header) | quarantine


@dataclass
class FormatScanResult:
    path: str
    header_bytes: int = 0
    findings: list[FormatFinding] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.findings)


def scan_safetensors_header_bytes(raw: bytes, header_len: int, label: str = "") -> FormatScanResult:
    """Check parsed safetensors header bytes for data hidden outside tensor bytes."""
    result = FormatScanResult(path=label, header_bytes=header_len)
    full = json.loads(raw)
    header = dict(full)
    metadata = header.pop("__metadata__", None) or {}

    total = 0
    for key, value in metadata.items():
        size = len(str(value).encode("utf-8"))
        total += size
        if size > MAX_METADATA_VALUE_BYTES:
            result.findings.append(
                FormatFinding(
                    "SUSPICIOUS",
                    "metadata",
                    f"__metadata__['{key}']",
                    f"metadata value is {size} bytes (limit {MAX_METADATA_VALUE_BYTES}) - possible hidden payload",
                    "sanitize",
                )
            )
    if total > MAX_METADATA_TOTAL_BYTES and not any(f.channel == "metadata" for f in result.findings):
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "metadata",
                "__metadata__",
                f"metadata totals {total} bytes (limit {MAX_METADATA_TOTAL_BYTES})",
                "sanitize",
            )
        )

    for key, value in metadata.items():
        for kind, masked in find_secrets(str(value)):
            result.findings.append(
                FormatFinding(
                    "SUSPICIOUS",
                    "secret",
                    f"__metadata__['{key}']",
                    f"{kind} in metadata ({masked}) - leaked credential; tell the owner privately and have it revoked",
                    "sanitize",
                )
            )

    canonical = json.dumps(full, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    slack = header_len - len(canonical)
    if slack > MAX_HEADER_SLACK_BYTES:
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_whitespace",
                "header",
                f"{slack} bytes of unexplained whitespace or padding in the JSON header - a bit-level channel",
                "sanitize",
            )
        )

    for name in header:
        size = len(name.encode("utf-8"))
        if size > MAX_TENSOR_NAME_BYTES:
            result.findings.append(
                FormatFinding(
                    "SUSPICIOUS",
                    "tensor_name",
                    name[:40] + "...",
                    f"tensor name is {size} bytes (limit {MAX_TENSOR_NAME_BYTES}) - possible hidden payload",
                    "quarantine",
                )
            )
    return result


def scan_safetensors_header(path: str | Path) -> FormatScanResult:
    """Check the safetensors header for data hidden outside the tensor bytes."""
    path = Path(path)
    result = FormatScanResult(path=str(path))
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            return result
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)

    return scan_safetensors_header_bytes(raw, header_len, str(path))
