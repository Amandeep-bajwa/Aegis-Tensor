"""End-to-end validation for synthetic steganography fixtures."""

import pytest

from aegis import CORE_AVAILABLE, scan_safetensors
from tests.generate_stego_model import generate_models


@pytest.fixture(scope="module")
def fixture_paths(tmp_path_factory):
    return generate_models(tmp_path_factory.mktemp("stego"))


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_clean_weights_are_not_flagged(fixture_paths):
    clean_path, _ = fixture_paths
    results = scan_safetensors(str(clean_path))
    assert results
    assert all(not result.is_suspicious for result in results)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_overwritten_floats_are_flagged_as_structural(fixture_paths):
    _, infected_path = fixture_paths
    results = {r.name: r for r in scan_safetensors(str(infected_path))}
    blob = results["layer_1.weight"]
    assert blob.is_suspicious
    assert any("overwritten" in reason or "encrypted" in reason for reason in blob.anomaly_reasons)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_structured_low_bit_payload_is_flagged(fixture_paths):
    _, infected_path = fixture_paths
    results = {r.name: r for r in scan_safetensors(str(infected_path))}
    text = results["layer_2.weight"]
    assert text.is_suspicious
    assert text.low_plane_entropy is not None and text.low_plane_entropy < 7.5
    assert any("structured" in reason for reason in text.anomaly_reasons)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_random_low_bit_payload_is_a_known_blind_spot(fixture_paths):
    """Random bits in the low mantissa look like trained weights, so the static scan
    cannot separate them. This test documents that limit: `aegis disarm` handles it."""
    _, infected_path = fixture_paths
    results = {r.name: r for r in scan_safetensors(str(infected_path))}
    random_lsb = results["layer_3.weight"]
    assert not random_lsb.is_suspicious
    assert random_lsb.low_plane_entropy > 7.9
