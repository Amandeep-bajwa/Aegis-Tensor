"""Verdict levels, optional engines, report bundles and credential detection.

Engine tests need Fickling and ModelScan (`pip install -e .[engines]`). Malicious fixtures
would only create a marker file, and every test checks the marker is never created.
"""

import json
import os
import pickle

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from aegis import TORCH_AVAILABLE
from aegis.cli import app

if TORCH_AVAILABLE:
    import torch

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
runner = CliRunner()


class _Touch:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.system, (f"touch {self.path}",))


@pytest.fixture
def marker(tmp_path):
    path = tmp_path / "MARKER"
    yield path
    assert not path.exists(), "a payload executed"


def _weights(seed=0):
    return np.random.default_rng(seed).normal(0, 0.02, 1 << 16).astype(np.float32)


def _scan(*args, engines="none"):
    result = runner.invoke(app, ["scan", *map(str, args), "--format", "json", "--engines", engines])
    return result.exit_code, json.loads(result.stdout)


# --- Verdicts -------------------------------------------------------------------------


@pytest.mark.claim("C17")
def test_verdict_levels_and_next_steps(tmp_path, marker):
    clean, tampered, evil = tmp_path / "clean.safetensors", tmp_path / "t.safetensors", tmp_path / "evil.pkl"
    save_file({"w": _weights()}, str(clean))
    blob = np.random.default_rng(1).integers(0, 256, 4 << 16, dtype=np.uint8).view(np.float32)
    save_file({"w": blob}, str(tampered))
    evil.write_bytes(pickle.dumps(_Touch(marker)))
    _, document = _scan(clean, tampered, evil)
    verdicts = {os.path.basename(f["path"]): f["verdict"] for f in document["files"]}

    assert verdicts["clean.safetensors"]["level"] == "no_known_findings"
    assert "Disarm recommended" in verdicts["clean.safetensors"]["next_step"]
    assert verdicts["clean.safetensors"]["command"].startswith("aegis disarm")

    assert verdicts["t.safetensors"]["level"] == "suspicious"
    assert verdicts["t.safetensors"]["command"].startswith("aegis disarm")

    assert verdicts["evil.pkl"]["level"] == "critical"
    assert "Do not load" in verdicts["evil.pkl"]["next_step"]
    assert verdicts["evil.pkl"]["disclosure"] == "platform"
    assert document["verdict"]["level"] == "critical"


@needs_torch
def test_clean_pickle_suggests_a_safer_format_to_the_owner(tmp_path):
    path = tmp_path / "ok.pt"
    torch.save({"w": torch.randn(8)}, path)
    _, document = _scan(path)
    verdict = document["files"][0]["verdict"]
    assert verdict["level"] == "no_known_findings" and verdict["disclosure"] == "owner"
    assert "safetensors" in verdict["next_step"]


# --- Credentials in metadata ---------------------------------------------------------

FAKE_TOKEN = "hf_" + "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3f"


@pytest.mark.claim("C18")
def test_leaked_token_in_metadata_is_reported_masked(tmp_path):
    path = tmp_path / "leak.safetensors"
    save_file({"w": _weights()}, str(path), metadata={"note": f"uploaded with {FAKE_TOKEN}"})
    code, document = _scan(path)
    entry = document["files"][0]
    finding = next(f for f in entry["findings"] if f["rule"] == "AEGIS-H004")
    assert code == 1 and FAKE_TOKEN not in json.dumps(document)
    assert "hf_aB3" in finding["message"] and entry["verdict"]["disclosure"] == "owner"


@pytest.mark.claim("C18")
def test_leaked_key_in_gguf_metadata_is_reported(tmp_path):
    from tests.test_model_formats import gguf_bytes

    path = tmp_path / "leak.gguf"
    path.write_bytes(gguf_bytes([("general.description", "key AKIA" + "ABCDEFGHIJKLMNOP")], []))
    _, document = _scan(path)
    assert [f["rule"] for f in document["files"][0]["findings"]] == ["AEGIS-H004"]


# --- Optional engines ------------------------------------------------------------------

needs_engines = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(n) for n in ("fickling", "modelscan")),
    reason="install the engines extra: pip install -e .[engines]",
)


@needs_torch
@needs_engines
@pytest.mark.claim("C19")
def test_engines_fold_into_one_verdict(tmp_path, marker):
    evil = tmp_path / "evil.pt"
    torch.save({"w": torch.randn(4), "x": _Touch(marker)}, evil)
    clean = tmp_path / "clean.pt"
    torch.save({"w": torch.randn(4)}, clean)
    _, document = _scan(evil, clean, engines="fickling,modelscan")
    by_name = {os.path.basename(f["path"]): f for f in document["files"]}
    engines = {e["name"]: e for e in by_name["evil.pt"]["engines"]}
    assert engines["fickling"]["worst"] == "critical" and engines["modelscan"]["worst"] == "critical"
    assert by_name["evil.pt"]["verdict"]["level"] == "critical"
    assert by_name["clean.pt"]["verdict"]["level"] == "no_known_findings"


@needs_engines
@pytest.mark.claim("C19")
def test_renamed_file_is_still_given_to_modelscan(tmp_path, marker):
    path = tmp_path / "weights.dat"
    path.write_bytes(pickle.dumps(_Touch(marker)))
    _, document = _scan(path, engines="modelscan")
    assert document["files"][0]["engines"][0] == {
        "name": "modelscan",
        "version": document["files"][0]["engines"][0]["version"],
        "ran": True,
        "error": None,
        "worst": "critical",
    }


@pytest.mark.claim("C19")
def test_disagreement_is_a_finding_in_both_directions():
    from aegis.engines import EngineIssue, EngineResult
    from aegis.findings import _make, from_engines

    engine_alarm = EngineResult("modelscan", "x", ran=True, issues=[EngineIssue("critical", "ModelScan: bad")])
    found = from_engines("m.pt", [engine_alarm], [])
    assert [(f.rule, f.severity) for f in found] == [("AEGIS-E001", "critical"), ("AEGIS-E002", "medium")]

    aegis_alarm = [_make("AEGIS-P001", "imports os.system", {"file": "m.pt"})]
    quiet = EngineResult("fickling", "x", ran=True)
    assert [(f.rule, f.severity) for f in from_engines("m.pt", [quiet], aegis_alarm)] == [("AEGIS-E002", "low")]


def test_unknown_or_missing_engine_is_an_error(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"w": _weights()}, str(path))
    assert runner.invoke(app, ["scan", str(path), "--engines", "nope"]).exit_code == 2


# --- Report bundle ------------------------------------------------------------------------


@pytest.mark.claim("C20")
def test_report_bundle_packages_hashes_source_and_findings_without_the_model(tmp_path, marker):
    import hashlib

    snapshot = tmp_path / "hub" / "models--acme--chat" / "snapshots" / ("ab" * 20)
    snapshot.mkdir(parents=True)
    evil = snapshot / "model.pkl"
    evil.write_bytes(pickle.dumps(_Touch(marker)))
    out = tmp_path / "bundle"
    result = runner.invoke(app, ["report-bundle", str(evil), "-o", str(out), "--engines", "none"])
    assert result.exit_code == 0, result.output

    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["files"][0]
    assert manifest["sent"] is False and manifest["disclosure_route"] == "platform"
    assert entry["sha256"] == hashlib.sha256(evil.read_bytes()).hexdigest()
    assert entry["source"] == {
        "repo_url": "https://huggingface.co/acme/chat",
        "revision": "ab" * 20,
        "path_in_repo": "model.pkl",
    }
    report = (out / "REPORT.md").read_text()
    assert "Do not contact the uploader" in report and "Nothing has been sent" in report
    assert sorted(p.name for p in out.iterdir()) == ["REPORT.md", "findings.json", "manifest.json"]
    assert evil.read_bytes() not in b"".join(p.read_bytes() for p in out.iterdir())


def test_available_lists_only_installed_engines(monkeypatch):
    from aegis import engines

    monkeypatch.setattr(engines, "ENGINES", ("aegis-no-such-engine",))
    assert engines.available() == {}


def test_engine_failure_is_a_finding_and_is_shown(capsys):
    from aegis.cli import _render_engines
    from aegis.engines import EngineIssue, EngineResult
    from aegis.findings import from_engines

    failed = EngineResult("modelscan", "0.8", ran=True, error="timed out")
    flagged = EngineResult("fickling", "0.1", ran=True, issues=[EngineIssue("high", "suspicious call")])
    findings = from_engines("m.pt", [failed], [])
    assert [f.rule for f in findings] == ["AEGIS-E003"]
    assert "modelscan: timed out" in findings[0].message

    _render_engines({"engines": [failed, flagged]})
    shown = capsys.readouterr().out
    assert "timed out" in shown
    assert "high" in shown and "1 issue(s)" in shown


def test_package_imports_without_the_compiled_core(tmp_path):
    """Without aegis_core, the package still imports and says the core is not built."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; sys.modules['aegis.aegis_core'] = None\n"
        "import aegis\n"
        "assert not aegis.CORE_AVAILABLE and aegis.scan_safetensors is None\n"
    )
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
