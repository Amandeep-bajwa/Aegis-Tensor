"""Disarm low-order mantissa bits of a safetensors model.

Steganographic payloads hidden in the least significant mantissa bits of
float weights survive any check that looks only at the bytes. Rounding those
bits away destroys the payload without having to find it, and it changes
each weight by at most a few units in the last place.

The number of bits that can be removed safely depends on the dtype. bfloat16
has only 7 explicit mantissa bits, so a width that is harmless for float32
would destroy a bfloat16 model. Requested widths are clamped to a per-dtype
safe maximum, and a width that would reach the exponent is always refused.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import torch
from safetensors.torch import load_file, save_file

# Number of explicit mantissa bits per floating-point dtype.
MANTISSA_BITS = {
    torch.float32: 23,
    torch.float16: 10,
    torch.bfloat16: 7,
}
# Largest width removed without allow_unsafe. Rounding k of m mantissa bits changes a
# weight by at most 2**(k - m - 1) of its value: about 0.4% (f32), 0.8% (f16), 1.6% (bf16).
# benchmarks/disarm_impact.py measures what these bounds cost in accuracy.
SAFE_MAX_LOW_BITS = {
    torch.float32: 16,
    torch.float16: 4,
    torch.bfloat16: 2,
}


@dataclass
class DisarmReport:
    tensors_disarmed: int
    tensors_skipped: int
    low_bits: int
    max_relative_change: float
    effective_low_bits: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def max_relative_change(dtype: torch.dtype, low_bits: int) -> float:
    """Worst-case relative change of a normal value when ``low_bits`` are rounded away."""
    return 2.0 ** (low_bits - MANTISSA_BITS[dtype] - 1)


def effective_low_bits(dtype: torch.dtype, requested: int, allow_unsafe: bool = False) -> int:
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
            f"low_bits={requested} would round away all {MANTISSA_BITS[dtype]} mantissa bits of {dtype} "
            "and change the exponent; refused"
        )
    if allow_unsafe:
        return requested
    return min(requested, SAFE_MAX_LOW_BITS[dtype])


def disarm_tensor(tensor: torch.Tensor, low_bits: int) -> torch.Tensor:
    """Round ``tensor`` to nearest with its lowest ``low_bits`` mantissa bits cleared.

    Rounding is done on the raw bit pattern, so a carry out of the mantissa
    correctly bumps the exponent. Infinities and NaNs are left unchanged. Widths
    that cover every mantissa bit are refused: they would rewrite the exponent.
    """
    if tensor.dtype not in MANTISSA_BITS:
        raise ValueError(f"Unsupported dtype for disarm: {tensor.dtype}")
    if not 0 < low_bits < MANTISSA_BITS[tensor.dtype]:
        raise ValueError(
            f"low_bits must be in 1..{MANTISSA_BITS[tensor.dtype] - 1} for {tensor.dtype}; "
            "removing every mantissa bit would change the exponent"
        )

    if tensor.dtype == torch.float32:
        width, int_dtype = 32, torch.int32
    else:
        width, int_dtype = 16, torch.int16

    # Work in int64 so the rounding carry cannot overflow, then wrap back to signed.
    bits = tensor.contiguous().view(int_dtype).to(torch.int64) & ((1 << width) - 1)
    half = 1 << (low_bits - 1)
    mask = (1 << low_bits) - 1
    rounded = (bits + half) & ~mask & ((1 << width) - 1)

    # Keep inf and NaN bit patterns exactly as they were.
    exponent_all_ones = ((1 << (width - 1 - MANTISSA_BITS[tensor.dtype])) - 1) << MANTISSA_BITS[tensor.dtype]
    special = (bits & exponent_all_ones) == exponent_all_ones
    rounded = torch.where(special, bits, rounded)

    signed = torch.where(rounded >= (1 << (width - 1)), rounded - (1 << width), rounded)
    return signed.to(int_dtype).view(tensor.dtype).reshape(tensor.shape)


def disarm_state(
    tensors: Dict[str, torch.Tensor],
    low_bits: int = 8,
    allow_unsafe: bool = False,
) -> "tuple[Dict[str, torch.Tensor], DisarmReport]":
    """Disarm every float tensor in a state dict, applying a per-dtype effective width."""
    out: Dict[str, torch.Tensor] = {}
    report = DisarmReport(0, 0, low_bits, 0.0)
    for name, tensor in tensors.items():
        if tensor.dtype not in MANTISSA_BITS:
            out[name] = tensor
            report.tensors_skipped += 1
            continue
        bits = effective_low_bits(tensor.dtype, low_bits, allow_unsafe)
        dtype_name = str(tensor.dtype).replace("torch.", "")
        if dtype_name not in report.effective_low_bits:
            report.effective_low_bits[dtype_name] = bits
            if bits < low_bits:
                report.warnings.append(
                    f"{dtype_name}: low_bits clamped from {low_bits} to {bits} "
                    f"(safe maximum; worst-case change {max_relative_change(tensor.dtype, bits):.2%} per weight). "
                    "Pass allow_unsafe to go further."
                )
            elif bits > SAFE_MAX_LOW_BITS[tensor.dtype]:
                report.warnings.append(
                    f"{dtype_name}: low_bits={bits} is above the safe maximum "
                    f"{SAFE_MAX_LOW_BITS[tensor.dtype]}; worst-case change "
                    f"{max_relative_change(tensor.dtype, bits):.2%} per weight"
                )
        new = disarm_tensor(tensor, bits)
        original = tensor.float()
        finite = torch.isfinite(original) & (original != 0)
        if finite.any():
            rel = ((new.float() - original).abs()[finite] / original.abs()[finite]).max().item()
            report.max_relative_change = max(report.max_relative_change, rel)
        report.tensors_disarmed += 1
        out[name] = new
    return out, report


def disarm_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    low_bits: int = 8,
    strip_metadata: bool = False,
    allow_unsafe: bool = False,
) -> DisarmReport:
    """Write a copy of ``source`` with low mantissa bits rounded away.

    ``low_bits`` is clamped per dtype to ``SAFE_MAX_LOW_BITS`` unless ``allow_unsafe``;
    the report records the width applied to each dtype and any clamping. Non-float
    tensors are copied unchanged. Metadata is preserved unless ``strip_metadata``
    is set, which removes it so a payload hidden there is gone.
    """
    source, destination = Path(source), Path(destination)
    tensors: Dict[str, torch.Tensor] = load_file(str(source))
    metadata = {} if strip_metadata else _read_metadata(source)
    disarmed, report = disarm_state(tensors, low_bits, allow_unsafe)
    save_file(disarmed, str(destination), metadata=metadata)
    return report


def _read_metadata(path: Path) -> Dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})
