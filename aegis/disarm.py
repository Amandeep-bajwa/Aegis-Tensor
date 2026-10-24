"""Disarm low-order mantissa bits of a model's float tensors.

Steganographic payloads hidden in the least significant mantissa bits of float
weights survive any check that looks only at the bytes, and a random-looking
payload cannot be told apart from trained weights at all. Rounding those bits
away destroys the payload without having to find it, and it changes each weight
by at most a few units in the last place. The attacker is assumed to know the
scan thresholds, so this is the recommended treatment for every file, including
files with no findings.

The number of bits that can be removed safely depends on the dtype. bfloat16
has only 7 explicit mantissa bits, so a width that is harmless for float32
would destroy a bfloat16 model. Requested widths are clamped to a per-dtype
safe maximum, and a width that would reach the exponent is always refused.

Works on raw bit patterns with numpy, so PyTorch is not needed. Supported files:
safetensors (F32, F16, BF16 tensors) and GGUF (F32, F16, BF16 tensors; quantized
blocks cannot be treated this way and are reported). Torch tensors are accepted by
``disarm_tensor`` and ``disarm_state`` for in-memory use.
"""

import json
import mmap
import os
import secrets
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class _ByDtype(dict):
    """Per-dtype table keyed by name ("float32"); torch dtypes are accepted as keys too."""

    def __getitem__(self, dtype):
        return super().__getitem__(dtype_name(dtype))

    def __contains__(self, dtype):
        return super().__contains__(dtype_name(dtype))


# Number of explicit mantissa bits per floating-point dtype.
MANTISSA_BITS = _ByDtype(float32=23, float16=10, bfloat16=7)
# Largest width removed without allow_unsafe. Rounding k of m mantissa bits changes a
# weight by at most 2**(k - m - 1) of its value: about 0.4% (f32), 0.8% (f16), 1.6% (bf16).
# benchmarks/disarm_impact.py measures what these bounds cost in accuracy.
SAFE_MAX_LOW_BITS = _ByDtype(float32=16, float16=4, bfloat16=2)
# Width requested when the user gives none; clamped per dtype by effective_low_bits.
DEFAULT_LOW_BITS = 8

_WIDTH = {"float32": 32, "float16": 16, "bfloat16": 16}
_UINT = {32: np.uint32, 16: np.uint16}
_SAFETENSORS_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}
_GGUF_DTYPES = {0: "float32", 1: "float16", 30: "bfloat16"}


def dtype_name(dtype) -> str:
    """ "float32" for torch.float32, "float32", "F32"; unknown dtypes are returned as text."""
    text = str(dtype).replace("torch.", "")
    return _SAFETENSORS_DTYPES.get(text, text)


@dataclass
class DisarmReport:
    tensors_disarmed: int
    tensors_skipped: int
    low_bits: int
    max_relative_change: float
    effective_low_bits: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    format: str = "safetensors"
    untreated: dict[str, int] = field(default_factory=dict)  # dtype -> tensors that could not be rounded


def max_relative_change(dtype, low_bits: int) -> float:
    """Worst-case relative change of a normal value when ``low_bits`` are rounded away."""
    return 2.0 ** (low_bits - MANTISSA_BITS[dtype] - 1)


def effective_low_bits(dtype, requested: int, allow_unsafe: bool = False) -> int:
    """Width actually applied to ``dtype``: clamped to the safe maximum unless allow_unsafe.

    Raises ValueError when the width would reach the exponent, even with allow_unsafe.
    """
    if dtype not in MANTISSA_BITS:
        raise ValueError(f"Unsupported dtype for disarm: {dtype}")
    if requested < 1:
        raise ValueError("low_bits must be at least 1")
    if requested >= MANTISSA_BITS[dtype]:
        if not allow_unsafe:
            return SAFE_MAX_LOW_BITS[dtype]
        raise ValueError(
            f"low_bits={requested} would round away all {MANTISSA_BITS[dtype]} mantissa bits of {dtype_name(dtype)} "
            "and change the exponent; refused"
        )
    if allow_unsafe:
        return requested
    return min(requested, SAFE_MAX_LOW_BITS[dtype])


def default_width(dtype, requested: int = DEFAULT_LOW_BITS) -> int:
    """Width applied to ``dtype`` when ``requested`` is asked for without allow_unsafe."""
    return effective_low_bits(dtype, requested)


# --- Bit rounding ------------------------------------------------------------------


def round_bits(bits: np.ndarray, dtype, low_bits: int) -> np.ndarray:
    """Round raw float bit patterns to nearest with the lowest ``low_bits`` mantissa bits cleared.

    ``bits`` holds the patterns as uint32 (float32) or uint16 (float16, bfloat16). A carry
    out of the mantissa correctly bumps the exponent. Infinities and NaNs are left unchanged,
    and a value that would round up to infinity is truncated instead.
    """
    name = dtype_name(dtype)
    if name not in _WIDTH:
        raise ValueError(f"Unsupported dtype for disarm: {dtype}")
    mantissa = MANTISSA_BITS[name]
    if not 0 < low_bits < mantissa:
        raise ValueError(
            f"low_bits must be in 1..{mantissa - 1} for {name}; removing every mantissa bit would change the exponent"
        )
    width = _WIDTH[name]
    full = (1 << width) - 1
    wide = bits.astype(np.uint64)
    mask = np.uint64((1 << low_bits) - 1)
    rounded = (wide + np.uint64(1 << (low_bits - 1))) & ~mask & np.uint64(full)
    exponent = np.uint64(((1 << (width - 1 - mantissa)) - 1) << mantissa)
    special = (wide & exponent) == exponent
    overflow = ((rounded & exponent) == exponent) & ~special
    rounded = np.where(overflow, wide & ~mask, rounded)
    return np.where(special, wide, rounded).astype(_UINT[width])


def _as_float32(bits: np.ndarray, dtype) -> np.ndarray:
    name = dtype_name(dtype)
    if name == "float32":
        return bits.view(np.float32)
    if name == "float16":
        return bits.view(np.float16).astype(np.float32)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _relative_change(before: np.ndarray, after: np.ndarray, dtype) -> float:
    old, new = _as_float32(before, dtype), _as_float32(after, dtype)
    finite = np.isfinite(old) & (old != 0)
    if not finite.any():
        return 0.0
    return float((np.abs(new[finite] - old[finite]) / np.abs(old[finite])).max())


class _Treatment:
    """Applies per-dtype widths and accumulates the report."""

    def __init__(self, low_bits: int, allow_unsafe: bool, fmt: str):
        self.low_bits, self.allow_unsafe = low_bits, allow_unsafe
        self.report = DisarmReport(0, 0, low_bits, 0.0, format=fmt)

    def width(self, dtype) -> int:
        name = dtype_name(dtype)
        bits = effective_low_bits(name, self.low_bits, self.allow_unsafe)
        if name not in self.report.effective_low_bits:
            self.report.effective_low_bits[name] = bits
            if bits < self.low_bits:
                self.report.warnings.append(
                    f"{name}: low_bits clamped from {self.low_bits} to {bits} "
                    f"(safe maximum; worst-case change {max_relative_change(name, bits):.2%} per weight). "
                    "Pass allow_unsafe to go further."
                )
            elif bits > SAFE_MAX_LOW_BITS[name]:
                self.report.warnings.append(
                    f"{name}: low_bits={bits} is above the safe maximum {SAFE_MAX_LOW_BITS[name]}; "
                    f"worst-case change {max_relative_change(name, bits):.2%} per weight"
                )
        return bits

    def apply(self, bits: np.ndarray, dtype) -> np.ndarray:
        out = round_bits(bits, dtype, self.width(dtype))
        self.report.max_relative_change = max(self.report.max_relative_change, _relative_change(bits, out, dtype))
        self.report.tensors_disarmed += 1
        return out

    def skip(self, dtype: str, untreated: bool = False) -> None:
        self.report.tensors_skipped += 1
        if untreated:
            self.report.untreated[dtype] = self.report.untreated.get(dtype, 0) + 1


# --- In-memory (torch) -------------------------------------------------------------


def _torch_bits(tensor):
    import torch

    view = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}[tensor.dtype]
    width = 32 if tensor.dtype == torch.float32 else 16
    return tensor.detach().contiguous().view(view).cpu().numpy().view(_UINT[width])


def _from_bits(bits: np.ndarray, like):
    import torch

    signed = bits.view(np.int32 if bits.dtype == np.uint32 else np.int16)
    return torch.from_numpy(signed.copy()).view(like.dtype).reshape(like.shape).to(like.device)


def disarm_tensor(tensor, low_bits: int):
    """Torch tensor with its lowest ``low_bits`` mantissa bits rounded away (see round_bits)."""
    if dtype_name(tensor.dtype) not in _WIDTH:
        raise ValueError(f"Unsupported dtype for disarm: {tensor.dtype}")
    return _from_bits(round_bits(_torch_bits(tensor), tensor.dtype, low_bits), tensor)


def disarm_state(
    tensors: dict[str, Any], low_bits: int = DEFAULT_LOW_BITS, allow_unsafe: bool = False
) -> tuple[dict[str, Any], DisarmReport]:
    """Disarm every float tensor in a torch state dict, applying a per-dtype effective width."""
    treatment = _Treatment(low_bits, allow_unsafe, "state_dict")
    out = {}
    for name, tensor in tensors.items():
        if dtype_name(tensor.dtype) not in _WIDTH:
            out[name] = tensor
            treatment.skip(dtype_name(tensor.dtype))
            continue
        out[name] = _from_bits(treatment.apply(_torch_bits(tensor), tensor.dtype), tensor)
    return out, treatment.report


# --- Files -------------------------------------------------------------------------


def default_disarm_path(source: str | Path) -> Path:
    """model.safetensors -> model.disarmed.safetensors, next to the input."""
    source = Path(source)
    return source.with_name(f"{source.stem}.disarmed{source.suffix}")


def disarm_file(
    source: str | Path,
    destination: str | Path,
    low_bits: int = DEFAULT_LOW_BITS,
    strip_metadata: bool = False,
    allow_unsafe: bool = False,
) -> DisarmReport:
    """Write a copy of ``source`` with low mantissa bits rounded away.

    safetensors: the header is re-serialised compactly, which also removes any
    whitespace channel; ``strip_metadata`` drops ``__metadata__``. GGUF: float
    tensors are rounded in place in the copy and everything else is kept byte for
    byte. Quantized GGUF tensors cannot be treated by rounding and are listed in
    ``report.untreated``. ``low_bits`` is clamped per dtype to ``SAFE_MAX_LOW_BITS``
    unless ``allow_unsafe``.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("destination must differ from the source")
    with open(source, "rb") as handle:
        head = handle.read(4)
    if head == b"GGUF" and strip_metadata:
        raise ValueError("--strip-metadata is not supported for GGUF: its metadata holds the tokenizer")
    # The output is written under a temporary name and renamed only when every tensor was
    # treated, so a failure never leaves a partly disarmed file under the "disarmed" name,
    # and never replaces an earlier good output.
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.partial")
    try:
        if head == b"GGUF":
            report = _disarm_gguf(source, partial, _Treatment(low_bits, allow_unsafe, "gguf"))
        else:
            treatment = _Treatment(low_bits, allow_unsafe, "safetensors")
            report = _disarm_safetensors(source, partial, strip_metadata, treatment)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return report


def _disarm_safetensors(source: Path, destination: Path, strip_metadata: bool, treatment: _Treatment) -> DisarmReport:
    with open(source, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) < 8:
            raise ValueError("not a safetensors file: too short")
        (header_len,) = struct.unpack("<Q", data[:8])
        try:
            header = json.loads(bytes(data[8 : 8 + header_len]))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"not a safetensors file: {error}") from error
        body = 8 + header_len
        if strip_metadata:
            header.pop("__metadata__", None)
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)
        tensors = sorted(
            ((name, info) for name, info in header.items() if name != "__metadata__"),
            key=lambda item: item[1]["data_offsets"][0],
        )
        with open(destination, "wb") as out:
            out.write(struct.pack("<Q", len(blob)) + blob)
            position = 0
            for name, info in tensors:
                start, end = info["data_offsets"]
                if start != position or body + end > len(data):
                    raise ValueError(f"tensor {name}: data offsets are not contiguous or run past the end")
                raw = data[body + start : body + end]
                dtype = _SAFETENSORS_DTYPES.get(info["dtype"])
                if dtype is None:
                    out.write(raw)
                    treatment.skip(info["dtype"])
                else:
                    bits = np.frombuffer(raw, dtype=_UINT[_WIDTH[dtype]])
                    out.write(treatment.apply(bits, dtype).tobytes())
                position = end
    return treatment.report


def _disarm_gguf(source: Path, destination: Path, treatment: _Treatment) -> DisarmReport:
    from aegis.modelformats import parse_gguf_header

    shutil.copyfile(source, destination)
    with open(destination, "r+b") as handle, mmap.mmap(handle.fileno(), 0) as data:
        metadata, infos, header_end = parse_gguf_header(data)
        alignment = metadata.get("general.alignment") or 32
        data_start = -(-header_end // alignment) * alignment
        for name, dims, ggml_type, offset in infos:
            dtype = _GGUF_DTYPES.get(ggml_type)
            if dtype is None:
                treatment.skip(f"ggml type {ggml_type}", untreated=ggml_type not in (24, 25, 26, 27))
                continue
            count = int(np.prod(dims, dtype=np.uint64)) if dims else 1
            start = data_start + offset
            end = start + count * (_WIDTH[dtype] // 8)
            if end > len(data):
                raise ValueError(f"tensor {name} points past the end of the file")
            bits = np.frombuffer(data[start:end], dtype=_UINT[_WIDTH[dtype]])  # a copy: the mmap must close
            data[start:end] = treatment.apply(bits, dtype).tobytes()
        data.flush()
    if treatment.report.untreated:
        total = sum(treatment.report.untreated.values())
        treatment.report.warnings.append(
            f"{total} quantized tensors were left as they are: their low-order bits are the weights, "
            "so rounding cannot remove a payload from them"
        )
    return treatment.report
