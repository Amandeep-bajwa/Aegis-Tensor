"""Threat matrix: one test per in-scope threat. See docs/THREAT_MATRIX.md.

Each test checks that the threat is (a) detected, (b) pinpointed to a tensor,
header key, global or layer, and (c) where a treatment exists, that the
treatment removes the threat. Malicious pickles are scanned, never loaded,
except in the test that proves an unsafe load is refused.
"""

import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.numpy import save_file
from safetensors.torch import load_file
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE, TORCH_AVAILABLE, scan_safetensors
from aegis.cli import app
from aegis.disarm import disarm_file
from aegis.formatscan import scan_safetensors_header
from aegis.picklescan import scan_pickle_file

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
runner = CliRunner()


def _weights(seed: int = 0, n: int = 1 << 16) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)


def _write(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def _rewrite_header(path: Path, mutate) -> None:
    """Apply `mutate` to the parsed JSON header and rewrite the file, keeping the data bytes."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    header = mutate(header)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + n :])


def _tensors(path: Path) -> dict:
    return {r.name: r for r in scan_safetensors(str(path))}


# --- Tensor-level threats -----------------------------------------------------


@pytest.mark.claim("C06")
@needs_core
def test_t01_structured_lsb_payload_is_pinpointed_and_disarmed(tmp_path):
    payload = np.unpackbits(np.frombuffer(b"connect 10.0.0.5:4444 /bin/sh " * 200, np.uint8), bitorder="little")
    w = _weights(1, payload.size)
    words = w.view(np.uint32).copy() & ~np.uint32(1)
    words |= payload.astype(np.uint32)
    infected = _write(tmp_path / "m.safetensors", {"layer.weight": words.view(np.float32)})

    before = _tensors(infected)["layer.weight"]
    assert before.is_suspicious
    assert any("structured" in r for r in before.anomaly_reasons)

    cleaned = tmp_path / "disarmed.safetensors"
    disarm_file(infected, cleaned, low_bits=8)
    assert not _tensors(cleaned)["layer.weight"].is_suspicious


@pytest.mark.claim("C09", "L01")
@needs_core
def test_t02_random_lsb_payload_is_a_documented_blind_spot_and_disarm_removes_it(tmp_path):
    rng = np.random.default_rng(2)
    w = _weights(2)
    random_bits = rng.integers(0, 16, w.size, dtype=np.uint32)  # 4-bit random payload per weight
    words = (w.view(np.uint32) & ~np.uint32(15)) | random_bits
    infected = _write(tmp_path / "m.safetensors", {"layer.weight": words.view(np.float32)})

    assert not _tensors(infected)["layer.weight"].is_suspicious  # static scan cannot see it

    cleaned = tmp_path / "disarmed.safetensors"
    disarm_file(infected, cleaned, low_bits=8)
    recovered = load_file(str(cleaned))["layer.weight"].view(torch.int32).numpy() & 15
    assert np.all(recovered == 0)  # payload gone


@pytest.mark.claim("C05")
@needs_core
def test_t03_overwritten_tensor_is_flagged_and_quarantined(tmp_path):
    blob = np.random.default_rng(3).integers(0, 256, 4 * 65536, dtype=np.uint8).view(np.float32)
    path = _write(tmp_path / "m.safetensors", {"layer.weight": blob, "ok.weight": _weights(3)})
    results = _tensors(path)
    assert results["layer.weight"].is_suspicious
    assert not results["ok.weight"].is_suspicious


@pytest.mark.claim("C05")
@needs_core
def test_t04_partial_overwrite_of_one_percent_is_flagged(tmp_path):
    rng = np.random.default_rng(4)
    arr = _weights(4, 1 << 20).view(np.uint8).reshape(-1, 4)
    idx = rng.choice(arr.shape[0], arr.shape[0] // 100, replace=False)
    arr[idx] = rng.integers(0, 256, (idx.size, 4), dtype=np.uint8)
    path = _write(tmp_path / "m.safetensors", {"layer.weight": arr.reshape(-1).view(np.float32)})
    result = _tensors(path)["layer.weight"]
    assert result.is_suspicious
    assert result.out_of_range_fraction > 0.001


@needs_core
def test_t08_clean_model_has_no_findings_in_any_channel(tmp_path):
    path = _write(
        tmp_path / "clean.safetensors", {f"l{i}.weight": _weights(i) for i in range(4)}, metadata={"format": "pt"}
    )
    assert all(not r.is_suspicious for r in scan_safetensors(str(path)))
    assert not scan_safetensors_header(path).is_suspicious


# --- Container and header channels --------------------------------------------


@pytest.mark.claim("C07")
def test_t05_payload_in_metadata_is_flagged_and_stripped(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(5)}, metadata={"note": "A" * 200_000})
    header = scan_safetensors_header(path)
    assert header.is_suspicious
    assert header.findings[0].channel == "metadata"
    assert "__metadata__['note']" in header.findings[0].where

    cleaned = tmp_path / "stripped.safetensors"
    disarm_file(path, cleaned, low_bits=8, strip_metadata=True)
    assert not scan_safetensors_header(cleaned).is_suspicious


@pytest.mark.claim("C07")
def test_t06_whitespace_channel_in_header_is_flagged(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(6)})
    _rewrite_header(path, lambda h: h)  # re-serialise to the canonical form
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = raw[8 : 8 + n].rstrip(b" ")
    padded = header + b" " * 5000
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + raw[8 + n :])
    result = scan_safetensors_header(path)
    assert result.is_suspicious
    assert result.findings[0].channel == "header_whitespace"


@pytest.mark.claim("C07")
def test_t07_oversized_tensor_name_is_flagged_for_quarantine(tmp_path):
    long_name = "x" * 600
    path = _write(tmp_path / "m.safetensors", {long_name: _weights(7)})
    result = scan_safetensors_header(path)
    assert result.is_suspicious
    assert result.findings[0].treatment == "quarantine"


@needs_core
def test_t09_truncated_file_is_rejected_not_silently_scanned(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(9)})
    path.write_bytes(path.read_bytes()[:-1000])
    with pytest.raises(Exception):
        scan_safetensors(str(path))


# --- Pickle-container threats --------------------------------------------------


class _Exec:
    """Pickles as os.system(...). The payload only writes a marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


@pytest.mark.claim("C02")
def test_t10_os_system_pickle_is_critical_and_pinpointed(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"w": _Exec(tmp_path / "pwned")}))
    result = scan_pickle_file(path)
    assert result.is_critical
    assert any("system" in f.message for f in result.findings)


@pytest.mark.claim("C02")
def test_t11_builtins_eval_pickle_is_critical(tmp_path):
    path = tmp_path / "eval.pkl"
    path.write_bytes(b"\x80\x02cbuiltins\neval\nq\x00(X\x0c\x00\x00\x00print(1)q\x01tq\x02Rq\x03.")
    assert scan_pickle_file(path).is_critical


@pytest.mark.claim("C03")
def test_t12_malformed_pickle_is_reported_not_skipped(tmp_path):
    clean = pickle.dumps({"w": [1.0] * 100})
    path = tmp_path / "broken.pkl"
    path.write_bytes(clean[: len(clean) // 2])
    result = scan_pickle_file(path)
    assert result.is_suspicious


@pytest.mark.claim("C01", "C12")
@needs_torch
def test_t13_unsafe_pickle_is_never_executed_without_trust(tmp_path):
    """The payload writes a marker file. Fuzzing must refuse to load it, so no marker appears."""
    marker = tmp_path / "pwned"
    model_path = tmp_path / "evil.pt"
    model_path.write_bytes(pickle.dumps(_Exec(marker)))
    result = runner.invoke(app, ["fuzz", str(model_path), "-s", "1,10", "-n", "2"])
    assert not marker.exists()
    assert result.exit_code != 0


@needs_torch
def test_t14_torchscript_is_refused_with_explanation(tmp_path):
    model = torch.jit.script(nn.Linear(4, 2))
    path = tmp_path / "m.pt"
    torch.jit.save(model, str(path))
    result = runner.invoke(app, ["fuzz", str(path), "-s", "1,4", "-n", "2"])
    assert result.exit_code != 0
    assert "TorchScript" in result.output


# --- Behavioural threats (sleeper / trigger backdoors) -------------------------


class _ThresholdBackdoor(nn.Module):
    """Dormant trigger: a large input on feature 0 amplifies the hidden layer by 50x."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        h = self.fc1(x)
        if (x[:, 0] > 8.0).any():
            h = h * 50.0
        return self.fc2(h)


@needs_torch
def test_t15_trigger_backdoor_is_detected_and_pinpointed_to_its_layer():
    from aegis import DynamicTrojanFuzzer

    torch.manual_seed(0)
    report = DynamicTrojanFuzzer(_ThresholdBackdoor().eval(), spike_threshold=4.0).run_fuzzing(
        input_shape=(2, 10), num_iterations=12, seed=0
    )
    assert report.suspected_trojan
    assert "fc2" in report.suspicious_layers
    assert "fc1" not in report.suspicious_layers  # the gate acts after fc1


@needs_torch
def test_t16_clean_network_is_not_flagged_at_any_input_scale():
    from aegis import DynamicTrojanFuzzer

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).eval()
    report = DynamicTrojanFuzzer(model, spike_threshold=4.0).run_fuzzing(input_shape=(2, 10), num_iterations=40, seed=0)
    assert not report.suspected_trojan
    assert report.spike_ratio < 4.0


# --- Out of scope for a weights-and-container scanner --------------------------
# Data poisoning (needs the training set), adversarial examples at inference,
# membership inference and model extraction (need query access), and LLM prompt
# injection (a runtime property of the deployed system). docs/THREAT_MATRIX.md
# records them as out of scope, so they have no tests here.


@pytest.mark.claim("C05")
@needs_core
def test_t17_nan_and_inf_injection_is_flagged(tmp_path):
    """Non-finite weights break inference (NaN propagation, overflow). Caught by the out-of-range rule."""
    w = _weights(17, 1 << 16)
    idx = np.random.default_rng(17).choice(w.size, w.size // 200, replace=False)
    w[idx[: len(idx) // 2]] = np.nan
    w[idx[len(idx) // 2 :]] = np.inf
    path = _write(tmp_path / "m.safetensors", {"layer.weight": w})
    result = _tensors(path)["layer.weight"]
    assert result.is_suspicious
    assert result.out_of_range_fraction > 0.001


@needs_core
def test_t18_scan_output_names_the_metadata_payload_and_its_treatment(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(18)}, metadata={"note": "B" * 50_000})
    result = runner.invoke(app, ["scan", str(path)])
    assert "__metadata__['note']" in result.output
    assert "--strip-metadata" in result.output
    assert "Recommended Treatment" in result.output


@needs_core
@pytest.mark.xfail(
    strict=True,
    reason="Known gap: tensors under 256 elements are below the sample floor, so a payload "
    "split into many tiny tensors is not flagged. Strict, so closing the gap will be noticed.",
)
@pytest.mark.claim("L02")
def test_t19_payload_split_into_tiny_tensors_is_flagged(tmp_path):
    rng = np.random.default_rng(19)
    tiny = {f"chunk.{i}": rng.integers(0, 256, 4 * 64, dtype=np.uint8).view(np.float32) for i in range(64)}
    path = _write(tmp_path / "m.safetensors", tiny)
    assert any(r.is_suspicious for r in scan_safetensors(str(path)))
