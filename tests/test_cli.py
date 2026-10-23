import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aegis.cli import app, get_risk_badge
from aegis.fuzzer import TrojanScanReport, FuzzIterationReport, LayerActivationStat


runner = CliRunner()
ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\([a-zA-Z]")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Helper to run Aegis CLI in a subprocess with robust UTF-8 encoding and ANSI stripping."""
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    res = subprocess.run(
        [sys.executable, "-m", "aegis.cli", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    # Strip any ANSI escape sequences that Rich/Typer emits on Linux/macOS terminals
    res.stdout = ANSI_REGEX.sub("", res.stdout)
    res.stderr = ANSI_REGEX.sub("", res.stderr)
    return res


def test_cli_version():
    """Verify that `aegis --version` runs without error and outputs version."""
    result = run_cli("--version")
    assert result.returncode == 0
    assert "Aegis-Tensor" in result.stdout
    assert "0.1.0" in result.stdout


def test_cli_doctor():
    """Verify that `aegis doctor` runs diagnostics successfully."""
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "Environment Diagnostics" in result.stdout


def test_cli_help():
    """Verify that `aegis --help` displays all available subcommands."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "scan" in result.stdout
    assert "fuzz" in result.stdout
    assert "doctor" in result.stdout


def test_cli_scan_help():
    """Verify that `aegis scan --help` displays required options and flags."""
    result = run_cli("scan", "--help")
    assert result.returncode == 0
    assert "entropy-threshold" in result.stdout
    assert "benford-threshold" in result.stdout
    assert "output-json" in result.stdout


def test_cli_fuzz_help():
    """Verify that `aegis fuzz --help` displays all required arguments and flags."""
    result = run_cli("fuzz", "--help")
    assert result.returncode == 0
    assert "spike-threshold" in result.stdout
    assert "iterations" in result.stdout
    assert "input-shape" in result.stdout
    assert "output-json" in result.stdout


def test_cli_fuzz_missing_model(tmp_path):
    """Verify that `aegis fuzz` exits with error code 1 when model file is missing."""
    missing_file = tmp_path / "nonexistent.pt"
    result = run_cli("fuzz", str(missing_file))
    assert result.returncode == 1
    assert "File Not Found" in result.stdout or "not found" in result.stdout.lower()


def test_cli_fuzz_safetensors_guidance(tmp_path):
    """Verify that providing .safetensors to `aegis fuzz` yields helpful guidance."""
    st_file = tmp_path / "model.safetensors"
    st_file.touch()

    result = run_cli("fuzz", str(st_file))
    assert result.returncode == 1
    assert "Safetensors weight container detected" in result.stdout
    assert "aegis scan" in result.stdout


def test_risk_badge_formatting():
    """Verify that security risk badges return expected formatted markup."""
    critical = get_risk_badge("CRITICAL")
    suspicious = get_risk_badge("SUSPICIOUS")
    clean = get_risk_badge("CLEAN")

    assert "CRITICAL MALWARE" in critical
    assert "SUSPICIOUS ANOMALY" in suspicious
    assert "CLEAN" in clean


def test_cli_scan_missing_model(tmp_path):
    """Verify that `aegis scan` exits with error code 2 (scan could not run) when the model file is missing."""
    missing_file = tmp_path / "missing.safetensors"
    result = run_cli("scan", str(missing_file))
    assert result.returncode == 2
    assert "not found" in result.stdout.lower()


def test_cli_fuzz_mocked_execution(tmp_path):
    """Verify full end-to-end execution of `aegis fuzz` with rich reporting and JSON export."""
    dummy_model_file = tmp_path / "model.pt"
    dummy_model_file.touch()
    json_out = tmp_path / "fuzz_results.json"

    fake_report = TrojanScanReport(
        model_name="MockTransformer",
        num_fuzz_samples=5,
        baseline_l_inf=1.5,
        peak_fuzzed_l_inf=9.0,
        spike_ratio=6.0,
        suspected_trojan=True,
        suspicious_layers=["transformer.layer.2"],
        iteration_reports=[
            FuzzIterationReport(
                iteration=1,
                input_tag="step_1",
                layer_stats={
                    "transformer.layer.1": LayerActivationStat("transformer.layer.1", 2.0, 0.5, 0.1),
                    "transformer.layer.2": LayerActivationStat("transformer.layer.2", 9.0, 1.2, 0.4),
                },
                max_l_inf=9.0,
                highest_layer="transformer.layer.2",
            )
        ],
    )

    mock_torch = MagicMock()
    mock_torch.zeros.return_value = MagicMock()
    mock_torch.randn.return_value = MagicMock()
    # The CLI checks isinstance(model, torch.jit.ScriptModule), which needs a real class.
    mock_torch.jit.ScriptModule = type("ScriptModule", (), {})
    mock_torch.load.return_value = MagicMock()

    with patch("aegis.cli.TORCH_AVAILABLE", True), \
         patch.dict("sys.modules", {"torch": mock_torch}), \
         patch("aegis.cli.DynamicTrojanFuzzer") as mock_fuzzer_cls:
        
        mock_fuzzer = MagicMock()
        mock_fuzzer.run_fuzzing.return_value = fake_report
        mock_fuzzer_cls.return_value = mock_fuzzer

        res = runner.invoke(app, [
            "fuzz",
            str(dummy_model_file),
            "--iterations", "5",
            "--spike-threshold", "4.0",
            "--output-json", str(json_out),
        ])

        assert res.exit_code == 0
        assert "Dynamic Introspection Report" in res.stdout
        assert "transformer.layer.2" in res.stdout
        assert "SPIKE DETECTED" in res.stdout
        assert "CRITICAL MALWARE" in res.stdout or "SUSPICIOUS ANOMALY" in res.stdout
        assert json_out.exists()

        data = json.loads(json_out.read_text())
        assert data["spike_ratio"] == 6.0
        assert data["suspected_trojan"] is True
        assert "transformer.layer.2" in data["suspicious_layers"]
        assert "iteration_reports" in data
        assert data["iteration_reports"][0]["highest_layer"] == "transformer.layer.2"
        assert data["iteration_reports"][0]["layer_stats"]["transformer.layer.2"]["l_inf_norm"] == 9.0
