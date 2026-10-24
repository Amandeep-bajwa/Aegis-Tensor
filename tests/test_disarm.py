"""Tests for mantissa-bit disarming."""

import numpy as np
import pytest

from aegis import TORCH_AVAILABLE
from aegis.disarm import (
    MANTISSA_BITS,
    SAFE_MAX_LOW_BITS,
    disarm_file,
    disarm_tensor,
    effective_low_bits,
)

if TORCH_AVAILABLE:
    import torch
    from safetensors.torch import load_file, save_file

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@needs_torch
def test_float32_low_bits_are_cleared():
    values = torch.tensor([1.0000001, 3.1415927, -0.02, 123.456], dtype=torch.float32)
    out = disarm_tensor(values, low_bits=8)
    low_mask = torch.tensor((1 << 8) - 1, dtype=torch.int32)
    assert torch.all((out.view(torch.int32) & low_mask) == 0)


@needs_torch
def test_disarm_changes_values_by_at_most_half_ulp_of_kept_bits():
    rng = np.random.default_rng(0)
    values = torch.from_numpy(rng.normal(0, 0.02, 10_000).astype(np.float32))
    out = disarm_tensor(values, low_bits=8)
    rel = ((out - values).abs() / values.abs()).max().item()
    # Rounding error is at most half of 2**8 ulps of a 23-bit mantissa.
    assert rel <= 2 ** (8 - 23)


@needs_torch
def test_inf_and_nan_are_preserved():
    values = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32)
    out = disarm_tensor(values, low_bits=4)
    assert torch.isinf(out[0]) and out[0] > 0
    assert torch.isinf(out[1]) and out[1] < 0
    assert torch.isnan(out[2])


@needs_torch
@pytest.mark.parametrize("dtype_name,bits", [("float16", 9), ("bfloat16", 6)])
def test_half_precision_dtypes(dtype_name, bits):
    dtype = getattr(torch, dtype_name)
    values = torch.linspace(-2, 2, 257).to(dtype)
    out = disarm_tensor(values, low_bits=bits)
    assert out.dtype == dtype
    assert out.shape == values.shape


@needs_torch
@pytest.mark.claim("C09")
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_rounding_every_mantissa_bit_is_refused(dtype_name):
    dtype = getattr(torch, dtype_name)
    values = torch.linspace(-2, 2, 33).to(dtype)
    with pytest.raises(ValueError, match="exponent"):
        disarm_tensor(values, low_bits=MANTISSA_BITS[dtype])


@needs_torch
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_effective_low_bits_clamps_to_safe_maximum(dtype_name):
    dtype = getattr(torch, dtype_name)
    assert effective_low_bits(dtype, 100) == SAFE_MAX_LOW_BITS[dtype]
    assert effective_low_bits(dtype, 1) == 1
    assert effective_low_bits(dtype, SAFE_MAX_LOW_BITS[dtype] + 1, allow_unsafe=True) == SAFE_MAX_LOW_BITS[dtype] + 1
    with pytest.raises(ValueError, match="exponent"):
        effective_low_bits(dtype, MANTISSA_BITS[dtype], allow_unsafe=True)


@needs_torch
def test_bfloat16_file_with_default_width_is_clamped_not_rejected(tmp_path):
    src, dst = tmp_path / "bf16.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(1024).to(torch.bfloat16)}, str(src))
    report = disarm_file(src, dst)  # default low_bits=8 exceeds bf16's 7 mantissa bits
    assert report.effective_low_bits == {"bfloat16": SAFE_MAX_LOW_BITS[torch.bfloat16]}
    assert any("clamped from 8" in w for w in report.warnings)
    out = load_file(str(dst))["w"]
    assert torch.all((out.view(torch.int16) & 0b11) == 0)
    assert report.max_relative_change <= 2.0 ** (SAFE_MAX_LOW_BITS[torch.bfloat16] - 7 - 1)


@needs_torch
def test_mixed_dtype_file_gets_per_dtype_widths(tmp_path):
    src, dst = tmp_path / "mixed.safetensors", tmp_path / "out.safetensors"
    save_file(
        {
            "a": torch.randn(256),
            "b": torch.randn(256).to(torch.float16),
            "c": torch.randn(256).to(torch.bfloat16),
        },
        str(src),
    )
    report = disarm_file(src, dst, low_bits=8)
    assert report.effective_low_bits == {"float32": 8, "float16": 4, "bfloat16": 2}


@needs_torch
def test_float32_default_width_needs_no_warning(tmp_path):
    src, dst = tmp_path / "f32.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(256)}, str(src))
    report = disarm_file(src, dst)
    assert report.warnings == []


@needs_torch
def test_allow_unsafe_goes_past_safe_maximum_with_a_warning(tmp_path):
    src, dst = tmp_path / "bf16.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(256).to(torch.bfloat16)}, str(src))
    report = disarm_file(src, dst, low_bits=5, allow_unsafe=True)
    assert report.effective_low_bits == {"bfloat16": 5}
    assert any("above the safe maximum" in w for w in report.warnings)
    with pytest.raises(ValueError, match="exponent"):
        disarm_file(src, dst, low_bits=7, allow_unsafe=True)


@needs_torch
def test_rejects_invalid_low_bits():
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.float32), low_bits=0)
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.float16), low_bits=11)
    with pytest.raises(ValueError):
        disarm_tensor(torch.zeros(4, dtype=torch.int64), low_bits=2)


@needs_torch
@pytest.mark.claim("C09")
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


@needs_torch
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


# --- Format coverage and edge cases (numpy core, no torch needed) -------------------


def test_largest_finite_values_do_not_round_up_to_infinity():
    from aegis.disarm import round_bits

    largest = np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32)
    out = round_bits(largest.view(np.uint32), "float32", 16).view(np.float32)
    assert np.all(np.isfinite(out))
    assert np.all((out.view(np.uint32) & 0xFFFF) == 0)


@needs_torch
def test_disarm_rewrites_a_whitespace_padded_header_compactly(tmp_path):
    import json
    import struct

    src, dst = tmp_path / "padded.safetensors", tmp_path / "out.safetensors"
    save_file({"w": torch.randn(256)}, str(src))
    raw = src.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    padded = raw[8 : 8 + n].rstrip() + b" " * 4096  # a 4 KiB whitespace channel
    src.write_bytes(struct.pack("<Q", len(padded)) + padded + raw[8 + n :])
    disarm_file(src, dst)
    (m,) = struct.unpack("<Q", dst.read_bytes()[:8])
    assert m < 200 and json.loads(dst.read_bytes()[8 : 8 + m])["w"]["dtype"] == "F32"


@pytest.mark.claim("C09")
def test_gguf_float_tensors_are_disarmed_and_everything_else_is_kept(tmp_path):
    from aegis.modelformats import parse_gguf_header
    from tests.test_model_formats import CLEAN_TEMPLATE, gguf_bytes

    rng = np.random.default_rng(3)
    f32 = rng.normal(0, 0.02, 4096).astype(np.float32)
    f32 = ((f32.view(np.uint32) & ~np.uint32(0xFF)) | rng.integers(0, 256, 4096, dtype=np.uint32)).view(np.float32)
    f16 = rng.normal(0, 0.02, 4096).astype(np.float16)
    q4 = rng.integers(0, 256, 2304, dtype=np.uint8).tobytes()  # Q4_0 blocks
    src, dst = tmp_path / "m.gguf", tmp_path / "m.disarmed.gguf"
    src.write_bytes(
        gguf_bytes(
            [("general.alignment", 32), ("tokenizer.chat_template", CLEAN_TEMPLATE)],
            [("norm", [4096], 0, f32.tobytes()), ("emb", [4096], 1, f16.tobytes()), ("blk.0.ffn", [4096], 2, q4)],
        )
    )
    report = disarm_file(src, dst)
    assert report.format == "gguf" and report.tensors_disarmed == 2
    assert report.effective_low_bits == {"float32": 8, "float16": 4}
    assert report.untreated == {"ggml type 2": 1} and "quantized" in report.warnings[-1]

    out = dst.read_bytes()
    metadata, infos, end = parse_gguf_header(out)
    assert metadata["tokenizer.chat_template"] == CLEAN_TEMPLATE
    start = -(-end // 32) * 32
    norm = np.frombuffer(out, np.uint32, 4096, start + infos[0][3])
    emb = np.frombuffer(out, np.uint16, 4096, start + infos[1][3])
    assert np.all(norm & 0xFF == 0) and np.all(emb & 0xF == 0)
    assert out[start + infos[2][3] : start + infos[2][3] + len(q4)] == q4


def test_disarm_runs_without_torch(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file as save_numpy

    src = tmp_path / "m.safetensors"
    save_numpy({"w": np.random.default_rng(0).standard_normal(256).astype(np.float32)}, str(src))
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from aegis.disarm import disarm_file\n"
        f"r = disarm_file({str(src)!r}, {str(tmp_path / 'o.safetensors')!r})\n"
        "assert r.tensors_disarmed == 1\n"
    )
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr


# --- The disarm guarantee at the default width -----------------------------------------
# Whatever is stored in the bits disarm rounds away is gone afterwards: those bits are zero, and
# the payload can move the kept value by at most one unit in the last kept place (the rounding
# carry). This holds for every supported float dtype, without PyTorch.

_BITS = {"float32": (np.uint32, 32), "float16": (np.uint16, 16), "bfloat16": (np.uint16, 16)}


def _weight_bits(dtype_name: str, n: int, seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)
    if dtype_name == "float32":
        return values.view(np.uint32)
    if dtype_name == "float16":
        return values.astype(np.float16).view(np.uint16)
    return (values.view(np.uint32) >> 16).astype(np.uint16)  # bfloat16: the top half of float32


@pytest.mark.claim("C09")
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_default_width_leaves_nothing_of_a_low_bit_payload(dtype_name):
    from aegis.disarm import default_width, round_bits

    uint, _ = _BITS[dtype_name]
    width = default_width(dtype_name)
    low = uint((1 << width) - 1)
    clean = _weight_bits(dtype_name, 20_000, 0)
    payload = np.random.default_rng(1).integers(0, 1 << width, clean.size, dtype=np.uint64).astype(uint)
    stuffed = (clean & ~low) | payload

    disarmed = round_bits(stuffed, dtype_name, width)
    assert not np.any(disarmed & low), "payload bits survived"
    baseline = round_bits(clean & ~low, dtype_name, width)
    carry = np.abs(disarmed.astype(np.int64) - baseline.astype(np.int64))
    assert carry.max() <= 1 << width, "the payload moved a kept value by more than one rounding step"


@pytest.mark.claim("C09")
def test_files_disarmed_at_the_default_width_keep_no_low_bit_payload(tmp_path):
    from safetensors.numpy import save_file as save_numpy

    from aegis.disarm import default_width
    from tests.test_model_formats import gguf_bytes

    rng = np.random.default_rng(2)
    f32 = _weight_bits("float32", 4096, 3) | rng.integers(0, 1 << 16, 4096, dtype=np.uint32)
    f16 = _weight_bits("float16", 4096, 4) | rng.integers(0, 1 << 4, 4096, dtype=np.uint16)
    save_numpy({"a": f32.view(np.float32), "b": f16.view(np.float16)}, str(tmp_path / "m.safetensors"))
    (tmp_path / "m.gguf").write_bytes(
        gguf_bytes([("general.alignment", 32)], [("a", [4096], 0, f32.tobytes()), ("b", [4096], 1, f16.tobytes())])
    )
    for name in ("m.safetensors", "m.gguf"):
        out = tmp_path / f"out-{name}"
        disarm_file(tmp_path / name, out)
        # Both formats store "a" then "b" contiguously at the end of the file (16384 is aligned).
        tail = out.read_bytes()[-(4096 * 4 + 4096 * 2) :]
        a = np.frombuffer(tail[: 4096 * 4], dtype=np.uint32)
        b = np.frombuffer(tail[4096 * 4 :], dtype=np.uint16)
        assert not np.any(a & np.uint32((1 << default_width("float32")) - 1)), name
        assert not np.any(b & np.uint16((1 << default_width("float16")) - 1)), name


# --- A failed disarm leaves nothing behind -------------------------------------------
# The output name says "disarmed", so a half-written or half-treated file must never be left there.


def _clean_safetensors(path):
    from safetensors.numpy import save_file as save_numpy

    save_numpy({"w": np.random.default_rng(0).normal(0, 0.02, 4096).astype(np.float32)}, str(path))
    return path


def test_refused_safetensors_disarm_leaves_no_output(tmp_path):
    src, out = _clean_safetensors(tmp_path / "m.safetensors"), tmp_path / "m.disarmed.safetensors"
    with pytest.raises(ValueError, match="exponent"):
        disarm_file(src, out, low_bits=23, allow_unsafe=True)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [src]


def test_failed_gguf_disarm_leaves_no_partly_disarmed_copy(tmp_path):
    from tests.test_model_formats import gguf_bytes

    rng = np.random.default_rng(1)
    first, second = (rng.normal(0, 0.02, 4096).astype(np.float32).tobytes() for _ in range(2))
    full = gguf_bytes([("general.alignment", 32)], [("a.weight", [4096], 0, first), ("b.weight", [4096], 0, second)])
    src, out = tmp_path / "m.gguf", tmp_path / "m.disarmed.gguf"
    src.write_bytes(full[:-1024])  # the second tensor now points past the end of the file
    with pytest.raises(ValueError, match="past the end"):
        disarm_file(src, out)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [src]


def test_failed_disarm_keeps_an_existing_output_unchanged(tmp_path):
    src, out = _clean_safetensors(tmp_path / "m.safetensors"), tmp_path / "m.disarmed.safetensors"
    disarm_file(src, out)
    good = out.read_bytes()
    with pytest.raises(ValueError):
        disarm_file(src, out, low_bits=23, allow_unsafe=True)
    assert out.read_bytes() == good
