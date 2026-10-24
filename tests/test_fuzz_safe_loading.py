"""`aegis fuzz` model loading: safetensors into a trusted architecture, and sandboxed pickles."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aegis import TORCH_AVAILABLE
from aegis.cli import app

if TORCH_AVAILABLE:
    import torch
    from safetensors.torch import save_file

    from tests.fuzz_fixtures import CleanMLP, SystemCallOnLoad, ThresholdBackdoor

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
runner = CliRunner()
ROOT = Path(__file__).resolve().parent.parent


def _sandbox_available() -> bool:
    from aegis.sandbox import SandboxUnavailable, sandbox_prefix

    try:
        sandbox_prefix(tempfile.gettempdir())
        return True
    except SandboxUnavailable:
        return False


needs_sandbox = pytest.mark.skipif(not _sandbox_available(), reason="no sandbox-exec or unprivileged unshare")
macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="filesystem and exec confinement is macOS-only")


def _fuzz(*args):
    return runner.invoke(app, ["fuzz", *map(str, args)])


# --- Safetensors into a trusted architecture -----------------------------------


@needs_torch
@pytest.mark.claim("C12")
def test_backdoor_in_safetensors_is_found_without_pickle(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "backdoor.safetensors"
    save_file(ThresholdBackdoor().state_dict(), str(path))
    out = tmp_path / "report.json"
    result = _fuzz(
        path, "--arch", "tests.fuzz_fixtures:ThresholdBackdoor", "--input-shape", "2,10", "-n", "12", "-o", out
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["suspected_trojan"]
    assert "fc2" in report["suspicious_layers"]


@needs_torch
def test_clean_safetensors_model_is_nominal(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "clean.safetensors"
    save_file(CleanMLP().state_dict(), str(path))
    out = tmp_path / "report.json"
    result = _fuzz(path, "--arch", "tests.fuzz_fixtures:CleanMLP", "--input-shape", "2,10", "-n", "40", "-o", out)
    assert result.exit_code == 0, result.output
    assert not json.loads(out.read_text())["suspected_trojan"]


@needs_torch
def test_state_dict_pt_loads_weights_only_into_arch(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "backdoor.pt"
    torch.save(ThresholdBackdoor().state_dict(), path)
    result = _fuzz(path, "--arch", "tests.fuzz_fixtures:ThresholdBackdoor", "--input-shape", "2,10", "-n", "12")
    assert result.exit_code == 0, result.output
    assert "fc2" in result.output


@needs_torch
def test_wrong_architecture_is_rejected(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"unrelated.weight": torch.zeros(3, 3)}, str(path))
    result = _fuzz(path, "--arch", "tests.fuzz_fixtures:CleanMLP", "--input-shape", "1,10")
    assert result.exit_code == 1
    assert "match the architecture" in result.output


@needs_torch
def test_strict_weights_rejects_partial_match(tmp_path):
    path = tmp_path / "m.safetensors"
    state = CleanMLP().state_dict()
    state.pop("net.4.bias")
    save_file(state, str(path))
    assert _fuzz(path, "--arch", "tests.fuzz_fixtures:CleanMLP", "--input-shape", "1,10", "-n", "2").exit_code == 0
    strict = _fuzz(path, "--arch", "tests.fuzz_fixtures:CleanMLP", "--input-shape", "1,10", "--strict-weights")
    assert strict.exit_code == 1


@needs_torch
def test_arch_without_input_shape_or_vocab_is_an_error(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file(CleanMLP().state_dict(), str(path))
    result = _fuzz(path, "--arch", "tests.fuzz_fixtures:CleanMLP")
    assert result.exit_code == 1
    assert "--input-shape" in result.output


@needs_torch
@pytest.mark.claim("C01", "C12")
def test_full_module_pickle_is_refused_without_trust(tmp_path):
    marker = tmp_path / "MARKER"
    path = tmp_path / "evil.pt"
    torch.save(SystemCallOnLoad(str(marker)), path)
    result = _fuzz(path, "--input-shape", "1,10")
    assert result.exit_code == 1
    assert "--trust-pickle" in result.output
    assert not marker.exists()


# --- Sandboxed --trust-pickle ----------------------------------------------------


@needs_torch
@pytest.mark.claim("C13")
@needs_sandbox
def test_trusted_pickle_is_fuzzed_in_the_sandbox(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "backdoor.pt"
    torch.save(ThresholdBackdoor(), path)
    out = tmp_path / "report.json"
    result = _fuzz(path, "--trust-pickle", "--input-shape", "2,10", "-n", "12", "-o", out)
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["suspected_trojan"] and "fc2" in report["suspicious_layers"]


@pytest.mark.claim("C13")
@needs_sandbox
def test_sandbox_blocks_network_at_the_os_level(tmp_path):
    from aegis.sandbox import run_sandboxed

    response = run_sandboxed({"action": "probe", "outside_path": str(tmp_path / "x"), "python_block": False}, 120)
    assert response["probe"]["network"].startswith("blocked")


@pytest.mark.claim("C14")
@needs_sandbox
@macos_only
def test_sandbox_confines_writes_and_blocks_shell(tmp_path):
    from aegis.sandbox import run_sandboxed

    outside = tmp_path / "outside.txt"
    response = run_sandboxed({"action": "probe", "outside_path": str(outside), "python_block": False}, 120)
    assert response["probe"]["write_outside"].startswith("blocked")
    assert response["probe"]["exec_shell"].startswith("blocked")
    assert not outside.exists()


@needs_torch
@pytest.mark.claim("C14")
@needs_sandbox
@macos_only
def test_payload_cannot_act_on_the_host_even_when_trusted(tmp_path):
    """The pickle runs os.system('touch MARKER') on load; the sandbox stops it."""
    marker = tmp_path / "MARKER"
    path = tmp_path / "evil.pt"
    torch.save(SystemCallOnLoad(str(marker)), path)
    _fuzz(path, "--trust-pickle", "--input-shape", "1,10", "-n", "2")
    assert not marker.exists()
