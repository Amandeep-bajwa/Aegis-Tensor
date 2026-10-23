"""Tests for mantissa-bit disarming."""

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from aegis.disarm import (
    MANTISSA_BITS,
    SAFE_MAX_LOW_BITS,
    disarm_file,
    disarm_tensor,
    effective_low_bits,
)


def test_float32_low_bits_are_cleared():
    values = torch.tensor([1.0000001, 3.1415927, -0.02, 123.456], dtype=torch.float32)
    out = disarm_tensor(values, low_bits=8)
    low_mask = torch.tensor((1 << 8) - 1, dtype=torch.int32)
    assert torch.all((out.view(torch.int32) & low_mask) == 0)


def test_disarm_changes_values_by_at_most_half_ulp_of_kept_bits():
    rng = np.random.default_rng(0)
    values = torch.from_numpy(rng.normal(0, 0.02, 10_000).astype(np.float32))
    out = disarm_tensor(values, low_bits=8)
    rel = ((out - values).abs() / values.abs()).max().item()
    # Rounding error is at most half of 2**8 ulps of a 23-bit mantissa.
    assert rel <= 2 ** (8 - 23)


def test_inf_and_nan_are_preserved():
    values = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32)
    out = disarm_tensor(values, low_bits=4)
    assert torch.isinf(out[0]) and out[0] > 0
    assert torch.isinf(out[1]) and out[1] < 0
    assert torch.isnan(out[2])


@pytest.mark.parametrize("dtype,bits", [(torch.float16, 9), (torch.bfloat16, 6)])
def test_half_precision_dtypes(dtype, bits):
    values = torch.linspace(-2, 2, 257).to(dtype)
    out = disarm_tensor(values, low_bits=bits)
    assert out.dtype == dtype
    assert out.shape == values.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rounding_every_mantissa_bit_is_refused(dtype):
    values = torch.linspace(-2, 2, 33).to(dtype)
    with pytest.raises(ValueError, match="exponent"):
        disarm_tensor(values, low_bits=MANTISSA_BITS[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_effective_low_bits_clamps_to_safe_maximum(dtype):
    assert effective_low_bits(dtype, 100) == SAFE_MAX_LOW_BITS[dtype]
    assert effective_low_bits(dtype, 1) == 1
    assert effective_low_bits(dtype, SAFE_MAX_LOW_BITS[dtype] + 1, allow_unsafe=True) == SAFE_MAX_LOW_BITS[dtype] + 1
    with pytest.raises(ValueError, match="exponent"):
        effective_low_bits(dtype, MANTISSA_BITS[dtype], allow_unsafe=True)


def test_bfloat16_file_with_default_width_is_clamped_not_rejected(tmp_path):
    src, dst = tmp_path / "bf16.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(1024).to(torch.bfloat16)}, str(src))
    report = disarm_file(src, dst)  # default low_bits=8 exceeds bf16's 7 mantissa bits
    assert report.effective_low_bits == {"bfloat16": SAFE_MAX_LOW_BITS[torch.bfloat16]}
    assert any("clamped from 8" in w for w in report.warnings)
    out = load_file(str(dst))["w"]
    assert torch.all((out.view(torch.int16) & 0b11) == 0)
    assert report.max_relative_change <= 2.0 ** (SAFE_MAX_LOW_BITS[torch.bfloat16] - 7 - 1)


def test_mixed_dtype_file_gets_per_dtype_widths(tmp_path):
    src, dst = tmp_path / "mixed.safetensors", tmp_path / "out.safetensors"
    save_file({
        "a": torch.randn(256),
        "b": torch.randn(256).to(torch.float16),
        "c": torch.randn(256).to(torch.bfloat16),
    }, str(src))
    report = disarm_file(src, dst, low_bits=8)
    assert report.effective_low_bits == {"float32": 8, "float16": 4, "bfloat16": 2}


def test_float32_default_width_needs_no_warning(tmp_path):
    src, dst = tmp_path / "f32.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(256)}, str(src))
    report = disarm_file(src, dst)
    assert report.warnings == []


def test_allow_unsafe_goes_past_safe_maximum_with_a_warning(tmp_path):
    src, dst = tmp_path / "bf16.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(256).to(torch.bfloat16)}, str(src))
    report = disarm_file(src, dst, low_bits=5, allow_unsafe=True)
    assert report.effective_low_bits == {"bfloat16": 5}
    assert any("above the safe maximum" in w for w in report.warnings)
    with pytest.raises(ValueError, match="exponent"):
        disarm_file(src, dst, low_bits=7, allow_unsafe=True)


def test_rejects_invalid_low_bits():
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.float32), low_bits=0)
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.float16), low_bits=11)
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.int64), low_bits=2)


def test_disarm_file_destroys_lsb_payload_and_keeps_metadata(tmp_path):
    rng = np.random.default_rng(1)
    clean = torch.from_numpy(rng.normal(0, 0.02, 65_536).astype(np.float32))
    payload_bits = torch.from_numpy(rng.integers(0, 16, 65_536, dtype=np.int32))
    infected = (clean.view(torch.int32) & ~0xF) | payload_bits
    infected = infected.view(torch.float32)

    src = tmp_path / "infected.safetensors"
    dst = tmp_path / "disarmed.safetensors"
    save_file({"w": infected}, str(src), metadata={"generator": "test"})

    report = disarm_file(src, dst, low_bits=4)
    recovered = load_file(str(dst))["w"]

    assert report.tensors_disarmed == 1
    assert report.low_bits == 4
    assert torch.all((recovered.view(torch.int32) & 0xF) == 0)
    assert report.max_relative_change <= 2 ** (4 - 23)

    from safetensors import safe_open

    with safe_open(str(dst), framework="pt") as handle:
        assert handle.metadata() == {"generator": "test"}


def test_disarm_file_skips_integer_tensors(tmp_path):
    src = tmp_path / "mixed.safetensors"
    dst = tmp_path / "out.safetensors"
    save_file(
        {"w": torch.ones(8, dtype=torch.float32), "idx": torch.arange(8, dtype=torch.int64)},
        str(src),
    )
    report = disarm_file(src, dst, low_bits=8)
    out = load_file(str(dst))
    assert report.tensors_skipped == 1
    assert torch.equal(out["idx"], torch.arange(8, dtype=torch.int64))
