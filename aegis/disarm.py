"""Disarm low-order mantissa bits of a safetensors model.

Steganographic payloads hidden in the least significant mantissa bits of
float weights survive any check that looks only at the bytes. Rounding those
bits away destroys the payload without having to find it, and it changes
each weight by at most a few units in the last place.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch
from safetensors.torch import load_file, save_file

# Number of explicit mantissa bits per floating-point dtype.
MANTISSA_BITS = {
    torch.float32: 23,
    torch.float16: 10,
    torch.bfloat16: 7,
}


@dataclass
class DisarmReport:
    tensors_disarmed: int
    tensors_skipped: int
    low_bits: int
    max_relative_change: float


def disarm_tensor(tensor: torch.Tensor, low_bits: int) -> torch.Tensor:
    """Round ``tensor`` to nearest with its lowest ``low_bits`` mantissa bits cleared.

    Rounding is done on the raw bit pattern, so a carry out of the mantissa
    correctly bumps the exponent. Infinities and NaNs are left unchanged.
    """
    if tensor.dtype not in MANTISSA_BITS:
        raise ValueError(f"Unsupported dtype for disarm: {tensor.dtype}")
    if not 0 < low_bits <= MANTISSA_BITS[tensor.dtype]:
        raise ValueError(
            f"low_bits must be in 1..{MANTISSA_BITS[tensor.dtype]} for {tensor.dtype}"
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


def disarm_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    low_bits: int = 8,
    strip_metadata: bool = False,
) -> DisarmReport:
    """Write a copy of ``source`` with low mantissa bits rounded away.

    Non-floating-point tensors are copied unchanged. Metadata is preserved unless
    ``strip_metadata`` is set, which removes it so a payload hidden there is gone.
    """
    source, destination = Path(source), Path(destination)
    tensors: Dict[str, torch.Tensor] = load_file(str(source))
    metadata = {} if strip_metadata else _read_metadata(source)

    disarmed: Dict[str, torch.Tensor] = {}
    changed_tensors = 0
    skipped = 0
    max_rel = 0.0
    for name, tensor in tensors.items():
        if tensor.dtype not in MANTISSA_BITS:
            disarmed[name] = tensor
            skipped += 1
            continue

        new = disarm_tensor(tensor, low_bits)
        original = tensor.float()
        scale = original.abs().clamp_min(torch.finfo(torch.float32).tiny)
        rel = ((new.float() - original).abs() / scale).max().item()
        max_rel = max(max_rel, rel)
        changed_tensors += 1
        disarmed[name] = new

    save_file(disarmed, str(destination), metadata=metadata)
    return DisarmReport(
        tensors_disarmed=changed_tensors,
        tensors_skipped=skipped,
        low_bits=low_bits,
        max_relative_change=max_rel,
    )


def _read_metadata(path: Path) -> Dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})
