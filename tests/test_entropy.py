"""Tests for Shannon Entropy and Benford's Law in aegis_core."""

import numpy as np
import pytest

from aegis import (
    CORE_AVAILABLE,
    benford_law_mad,
    mantissa_bit_plane_entropy,
    shannon_entropy_bytes,
    shannon_entropy_chunked,
)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_shannon_entropy_uniform_bytes():
    """Uniform byte distribution should produce entropy close to 8.0 bits/byte."""
    # 256 unique bytes repeated uniformly
    uniform_bytes = bytes([i % 256 for i in range(256 * 100)])
    entropy = shannon_entropy_bytes(uniform_bytes)
    assert abs(entropy - 8.0) < 1e-3, f"Expected ~8.0, got {entropy}"


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_shannon_entropy_zero_bytes():
    """Identical bytes should produce 0.0 entropy."""
    zero_bytes = bytes([0] * 10000)
    assert shannon_entropy_bytes(zero_bytes) == 0.0


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_shannon_entropy_chunked_detects_localized_payload():
    """Chunked entropy should expose a high-entropy region in low-entropy data."""
    localized = bytes([0] * 1024) + bytes(range(256)) + bytes([0] * 1024)
    assert shannon_entropy_chunked(localized, 256) == 8.0
    assert shannon_entropy_chunked(localized, 0) == 0.0


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_mantissa_bit_plane_entropy_ignores_sign_and_exponent():
    """Only mantissa bits should affect the bit-plane entropy values."""
    values = np.array([1.0, -1.0, 2.0, -2.0, 4.0, -4.0], dtype=np.float32)
    bit_0_entropy, bit_1_entropy = mantissa_bit_plane_entropy(values.tobytes())
    assert bit_0_entropy == 0.0
    assert bit_1_entropy == 0.0


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_benford_law_ideal_vs_anomalous():
    """Benford's Law MAD should be low for logarithmic distributions and high for uniform."""
    # Ideal Benford distribution
    samples = []
    total = 5000
    for d in range(1, 10):
        p_d = np.log10(1 + 1 / d)
        count = round(p_d * total)
        samples.extend([float(d) + 0.1 * i for i in range(count)])

    benford_mad = benford_law_mad(samples)
    assert benford_mad < 0.01, f"Expected MAD < 0.01 for Benford data, got {benford_mad}"

    # Non-Benford (uniform leading digits)
    uniform_samples = [float((i % 9) + 1) for i in range(4500)]
    uniform_mad = benford_law_mad(uniform_samples)
    assert uniform_mad > 0.03, f"Expected MAD > 0.03 for uniform data, got {uniform_mad}"
