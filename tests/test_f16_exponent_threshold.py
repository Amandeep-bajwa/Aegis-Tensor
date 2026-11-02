"""Float16 gets its own exponent-lane threshold (issue D1, docs/calibration.md proposal P1).

The float16 high byte holds the sign, five exponent bits and two mantissa bits, so clean
float16 tensors score higher on the exponent-lane rule than float32 or bfloat16 ones. On 54
clean public models, 37 float16 attention bias vectors (Pythia, BLOOM) scored 6.51-6.85 and
were flagged at the general threshold of 6.5. These tests pin the float16 threshold without
weakening detection of overwritten floats.
"""

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE, scan_safetensors, scan_tensor_bytes
from aegis.cli import app
from aegis.severity import EXIT_CLEAN, EXIT_FINDINGS
from aegis.thresholds import DEFAULT_THRESHOLDS

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
runner = CliRunner()


def _wide_exponent_bias(seed: int = 0, n: int = 3072) -> np.ndarray:
    """A float16 bias vector shaped like the three false positives: 2,304-3,072 elements whose
    magnitudes span about four decades, so the sign/exponent byte is unusually varied."""
    rng = np.random.default_rng(seed)
    magnitudes = 10 ** rng.uniform(-4.0, 0.5, n)
    return (magnitudes * rng.choice([-1.0, 1.0], n)).astype(np.float16)


def _wide_exponent_values(dtype: str, seed: int = 0, n: int = 3072) -> bytes:
    """float32 or bfloat16 values whose exponent lane measures about 6.8 bits."""
    rng = np.random.default_rng(seed)
    values = (10 ** rng.uniform(-30.0, 3.0, n) * rng.choice([-1.0, 1.0], n)).astype(np.float32)
    if dtype == "F32":
        return values.tobytes()
    return (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()  # BF16 = top half of F32


def _scan_one(path, tensors: dict) -> dict:
    save_file(tensors, str(path))
    return {r.name: r for r in scan_safetensors(str(path))}


@needs_core
def test_f16_bias_vector_with_wide_exponents_is_not_flagged(tmp_path):
    result = _scan_one(tmp_path / "m.safetensors", {"attn.bias": _wide_exponent_bias()})["attn.bias"]

    # The fixture sits in the measured false-positive band: above the general threshold,
    # below the float16 one. Without this, the test could pass for the wrong reason.
    assert DEFAULT_THRESHOLDS["exponent"] < result.exponent_lane_entropy < DEFAULT_THRESHOLDS["exponent_f16"]
    assert result.out_of_range_fraction == 0.0
    assert result.rules == []
    assert not result.is_suspicious


@needs_core
def test_f16_fully_overwritten_tensor_is_still_flagged(tmp_path):
    blob = np.random.default_rng(1).integers(0, 256, 2 * 65536, dtype=np.uint8).view(np.float16)
    result = _scan_one(tmp_path / "m.safetensors", {"layer.weight": blob})["layer.weight"]

    assert result.exponent_lane_entropy > DEFAULT_THRESHOLDS["exponent_f16"]
    assert "exponent_lane" in result.rules or "out_of_range" in result.rules
    assert result.is_suspicious


@needs_core
def test_f16_one_percent_overwrite_is_still_flagged(tmp_path):
    # 1% of the floats replaced with random bytes. The out-of-range rule catches this, and
    # D1 must not change it. The tensor is large (2^20 elements) because the margin is thin
    # for float16: only about 11% of random float16 words exceed the magnitude cap, so a 1%
    # overwrite measures about 0.0012 against the 0.001 threshold.
    rng = np.random.default_rng(4)
    n = 1 << 20
    words = rng.normal(0, 0.02, n).astype(np.float16).view(np.uint8).reshape(-1, 2).copy()
    idx = rng.choice(n, n // 100, replace=False)
    words[idx] = rng.integers(0, 256, (idx.size, 2), dtype=np.uint8)
    blob = words.reshape(-1).view(np.float16)

    result = _scan_one(tmp_path / "m.safetensors", {"layer.weight": blob})["layer.weight"]

    assert "out_of_range" in result.rules
    assert result.out_of_range_fraction > 0.001
    assert result.is_suspicious


@needs_core
@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_f32_and_bf16_still_use_the_general_exponent_threshold(dtype):
    result = scan_tensor_bytes("attn.bias", dtype, [3072], _wide_exponent_values(dtype))

    assert DEFAULT_THRESHOLDS["exponent"] < result.exponent_lane_entropy < DEFAULT_THRESHOLDS["exponent_f16"]
    assert "exponent_lane" in result.rules


@needs_core
def test_f16_threshold_is_a_separate_keyword_argument():
    data = _wide_exponent_bias().tobytes()

    default = scan_tensor_bytes("attn.bias", "F16", [3072], data)
    strict = scan_tensor_bytes("attn.bias", "F16", [3072], data, exponent_f16_entropy_threshold=6.5)
    general_only = scan_tensor_bytes("attn.bias", "F16", [3072], data, exponent_entropy_threshold=5.0)

    assert "exponent_lane" not in default.rules
    assert "exponent_lane" in strict.rules
    assert "exponent_lane" not in general_only.rules  # the general threshold no longer governs F16


@needs_core
def test_cli_exponent_threshold_f16_option_is_applied(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"attn.bias": _wide_exponent_bias()}, str(path), metadata={"format": "pt"})

    default = runner.invoke(app, ["scan", str(path), "--engines", "none"])
    strict = runner.invoke(app, ["scan", str(path), "--engines", "none", "--exponent-threshold-f16", "6.5"])

    assert default.exit_code == EXIT_CLEAN, default.output
    assert strict.exit_code == EXIT_FINDINGS, strict.output
