"""Exit-code contract for `aegis scan`: 0 = nothing at or above --fail-on, 1 = findings, 2 = error."""

import json
import pickle
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE
from aegis.cli import app
from aegis.severity import EXIT_CLEAN, EXIT_ERROR, EXIT_FINDINGS

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
runner = CliRunner()


def _weights(seed: int = 0, n: int = 1 << 16) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)


def _scan(*args):
    return runner.invoke(app, ["scan", *map(str, args)])


@pytest.fixture
def clean_model(tmp_path):
    path = tmp_path / "clean.safetensors"
    save_file({"a.weight": _weights(1), "b.weight": _weights(2)}, str(path), metadata={"format": "pt"})
    return path


@pytest.fixture
def overwritten_model(tmp_path):
    """High severity: one tensor is random bytes."""
    path = tmp_path / "overwritten.safetensors"
    blob = np.random.default_rng(3).integers(0, 256, 4 * 65536, dtype=np.uint8).view(np.float32)
    save_file({"a.weight": _weights(1), "b.weight": blob}, str(path))
    return path


@pytest.fixture
def whitespace_model(tmp_path):
    """Medium severity only: unexplained whitespace in the header, clean tensors."""
    path = tmp_path / "whitespace.safetensors"
    save_file({"a.weight": _weights(4)}, str(path))
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    padded = raw[8 : 8 + n].rstrip(b" ") + b" " * 4096
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + raw[8 + n :])
    return path


# --- Exit 0 -------------------------------------------------------------------


@pytest.mark.claim("C16")
@needs_core
def test_clean_model_exits_0(clean_model):
    result = _scan(clean_model)
    assert result.exit_code == EXIT_CLEAN, result.output


@needs_core
def test_findings_below_fail_on_exit_0(whitespace_model):
    result = _scan(whitespace_model, "--fail-on", "high")
    assert result.exit_code == EXIT_CLEAN, result.output
    assert "Header Channel Findings" in result.output  # still reported


@needs_core
def test_low_findings_do_not_fail_at_default_medium(clean_model):
    # Benford on normally distributed weights deviates, giving a low-severity finding only.
    result = _scan(clean_model, "--benford-threshold", "0.0001")
    assert result.exit_code == EXIT_CLEAN, result.output


def test_clean_pickle_exits_0(tmp_path):
    path = tmp_path / "clean.pkl"
    path.write_bytes(pickle.dumps({"w": [1.0, 2.0]}))
    assert _scan(path).exit_code == EXIT_CLEAN


# --- Exit 1 -------------------------------------------------------------------


@pytest.mark.claim("C16")
@needs_core
def test_high_finding_exits_1_at_default(overwritten_model):
    result = _scan(overwritten_model)
    assert result.exit_code == EXIT_FINDINGS, result.output


@needs_core
def test_high_finding_exits_1_at_fail_on_high(overwritten_model):
    assert _scan(overwritten_model, "--fail-on", "high").exit_code == EXIT_FINDINGS


@needs_core
def test_high_finding_exits_0_at_fail_on_critical(overwritten_model):
    assert _scan(overwritten_model, "--fail-on", "critical").exit_code == EXIT_CLEAN


@needs_core
def test_medium_finding_exits_1_at_default(whitespace_model):
    assert _scan(whitespace_model).exit_code == EXIT_FINDINGS


@needs_core
def test_medium_from_tensor_rule_exits_1_at_default(clean_model):
    # Lowering the pooled-entropy threshold below clean weights (about 7.36) yields a medium finding.
    assert _scan(clean_model, "--entropy-threshold", "7.0").exit_code == EXIT_FINDINGS


@needs_core
def test_low_finding_exits_1_at_fail_on_low(clean_model):
    result = _scan(clean_model, "--benford-threshold", "0.0001", "--fail-on", "low")
    assert result.exit_code == EXIT_FINDINGS, result.output


@pytest.mark.claim("C16")
def test_critical_pickle_exits_1_even_at_fail_on_critical(tmp_path):
    path = tmp_path / "eval.pkl"
    path.write_bytes(b"\x80\x02cbuiltins\neval\nq\x00(X\x0c\x00\x00\x00print(1)q\x01tq\x02Rq\x03.")
    assert _scan(path, "--fail-on", "critical").exit_code == EXIT_FINDINGS


def test_malformed_pickle_is_a_finding_not_an_error(tmp_path):
    clean = pickle.dumps({"w": [1.0] * 50})
    path = tmp_path / "broken.pkl"
    path.write_bytes(clean[: len(clean) // 2])
    assert _scan(path).exit_code == EXIT_FINDINGS  # medium at default


@needs_core
def test_fail_on_is_case_insensitive(overwritten_model):
    assert _scan(overwritten_model, "--fail-on", "HIGH").exit_code == EXIT_FINDINGS


# --- Exit 2 -------------------------------------------------------------------


@pytest.mark.claim("C16")
def test_missing_file_exits_2(tmp_path):
    assert _scan(tmp_path / "nope.safetensors").exit_code == EXIT_ERROR


@pytest.mark.claim("C16")
@needs_core
def test_corrupt_safetensors_exits_2(tmp_path):
    path = tmp_path / "corrupt.safetensors"
    save_file({"w": _weights(5)}, str(path))
    path.write_bytes(path.read_bytes()[:-1000])
    assert _scan(path).exit_code == EXIT_ERROR


def test_unknown_format_is_a_finding_not_an_error(tmp_path):
    # Routing is by magic bytes: a non-safetensors file named .safetensors goes to the
    # container scanner, where an unknown format is a medium finding (exit 1).
    path = tmp_path / "garbage.safetensors"
    path.write_bytes(b"\x00this is not a model file at all")
    result = _scan(path)
    assert result.exit_code == EXIT_FINDINGS, result.output
    assert "unrecognised file format" in result.output


def test_invalid_fail_on_value_exits_2(tmp_path):
    path = tmp_path / "clean.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    assert _scan(path, "--fail-on", "severe").exit_code == EXIT_ERROR


# --- JSON report carries the severity -------------------------------------------


@needs_core
def test_json_report_includes_max_severity_and_rules(overwritten_model, tmp_path):
    out = tmp_path / "report.json"
    result = _scan(overwritten_model, "-o", out)
    assert result.exit_code == EXIT_FINDINGS
    report = json.loads(out.read_text())
    assert report["schema_version"] == "1.1.0"
    assert report["max_severity"] == "high"
    assert report["fail_on"] == "medium"
    file_entry = report["files"][0]
    blob = next(t for t in file_entry["tensors"] if t["name"] == "b.weight")
    assert "exponent_lane" in blob["rules"]
    assert blob["severity"] == "high"
    assert any(f["rule"] == "AEGIS-T002" and f["location"]["tensor"] == "b.weight" for f in file_entry["findings"])


# --- Errors are exit 2, never a traceback ------------------------------------------


def test_scanning_a_directory_is_an_error_not_a_traceback(tmp_path):
    result = _scan(tmp_path)
    assert result.exit_code == EXIT_ERROR
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_disarm_into_its_own_input_is_refused(clean_model):
    before = clean_model.read_bytes()
    result = runner.invoke(app, ["disarm", str(clean_model), str(clean_model)])
    assert result.exit_code == 1
    assert "Output must be a different file from the input" in result.output
    assert clean_model.read_bytes() == before


def test_disarm_width_reaching_the_exponent_is_refused_even_with_allow_unsafe(clean_model, tmp_path):
    out = tmp_path / "out.safetensors"
    result = runner.invoke(app, ["disarm", str(clean_model), str(out), "--low-bits", "23", "--allow-unsafe"])
    assert result.exit_code == 1
    assert "exponent" in result.output
    assert not out.exists()


def test_report_bundle_with_an_unknown_engine_is_an_error(clean_model, tmp_path):
    result = runner.invoke(
        app, ["report-bundle", str(clean_model), "--engines", "no-such-engine", "-o", str(tmp_path / "b")]
    )
    assert result.exit_code == EXIT_ERROR
    assert not (tmp_path / "b").exists()


def test_report_bundle_on_an_unscannable_path_is_an_error(tmp_path):
    result = runner.invoke(app, ["report-bundle", str(tmp_path), "-o", str(tmp_path / "b")])
    assert result.exit_code == EXIT_ERROR
    assert "Scan failed" in result.output
