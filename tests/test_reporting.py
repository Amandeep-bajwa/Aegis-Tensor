"""Findings schema, SARIF output and taxonomy references.

tests/data/sarif-schema-2.1.0.json is the official OASIS schema, vendored from
oasis-tcs/sarif-spec @ 1742a253 (sarif-2.1/schema/sarif-schema-2.1.0.json) so the
test runs offline.
"""

import json
import os
import pickle
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE
from aegis.atlas import ATLAS_RELEASE, ATLAS_TECHNIQUES
from aegis.cli import app
from aegis.findings import OWASP, RULE_BY_CODE, RULES

ROOT = Path(__file__).resolve().parent.parent
FINDINGS_SCHEMA = json.loads((ROOT / "docs" / "schema" / "findings-v1.schema.json").read_text())
SARIF_SCHEMA = json.loads((ROOT / "tests" / "data" / "sarif-schema-2.1.0.json").read_text())
needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
runner = CliRunner()


def _weights(seed=0, n=1 << 16):
    return np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    """A clean model, an overwritten tensor plus a metadata payload, and a code-execution pickle.

    Paths are relative to the working directory, as they would be in a repository checkout.
    """
    monkeypatch.chdir(tmp_path)
    save_file({"w": _weights(1)}, "clean.safetensors")
    blob = np.random.default_rng(2).integers(0, 256, 4 * 65536, dtype=np.uint8).view(np.float32)
    save_file({"ok": _weights(3), "bad": blob}, "tampered.safetensors", metadata={"note": "Z" * 20000})

    class _Exec:
        def __reduce__(self):
            return (os.system, ("echo never-run",))

    Path("evil.pkl").write_bytes(pickle.dumps(_Exec()))
    return ["clean.safetensors", "tampered.safetensors", "evil.pkl"]


def _invoke(*args):
    return runner.invoke(app, ["scan", *map(str, args)])


# --- Taxonomy integrity ----------------------------------------------------------


def test_every_atlas_reference_exists_in_the_pinned_release():
    for rule_id, entry in RULES.items():
        for technique in entry["atlas"]:
            assert technique in ATLAS_TECHNIQUES, f"{rule_id} references unknown {technique}"


def test_owasp_references_are_known_identifiers():
    assert OWASP == {
        "ML06": "ML06:2023 ML Supply Chain Attacks",
        "ML10": "ML10:2023 Model Poisoning",
        "LLM03": "LLM03:2025 Supply Chain",
        "LLM04": "LLM04:2025 Data and Model Poisoning",
    }
    for entry in RULES.values():
        assert set(entry["owasp"]) <= set(OWASP)


def test_every_scanner_code_maps_to_exactly_one_rule():
    scanner_codes = {
        # Rust tensor rules (TensorScanResult.rules)
        "pooled_entropy",
        "exponent_lane",
        "out_of_range",
        "low_plane",
        "benford",
        "unreadable",
        # Header channels (FormatFinding.channel)
        "metadata",
        "header_whitespace",
        "tensor_name",
        "secret",
        # Container scanner (PickleFinding.code)
        "dangerous_import",
        "unknown_import",
        "unresolved_import",
        "unresolved_callable",
        "extension_registry",
        "malformed_pickle",
        "zip_tampered",
        "no_payload",
        "container_unreadable",
        "archive_7z",
        "malformed_npy",
        "npy_object",
        "unknown_format",
        # GGUF, Keras and ONNX scanners (aegis/modelformats.py)
        "gguf_template",
        "gguf_template_unverifiable",
        "gguf_malformed",
        "keras_lambda",
        "keras_custom_object",
        "onnx_external_path",
        "onnx_external_data",
        "onnx_custom_op",
        "format_unavailable",
        # Optional engines (aegis/engines.py)
        "engine_finding",
        "engine_disagreement",
        "engine_error",
        # Fuzzer
        "activation_spike",
        # Hub remote scanner (aegis/hub.py)
        "trust_remote_code",
        "pickle_needs_download",
        "header_unreadable",
    }
    assert scanner_codes == set(RULE_BY_CODE)
    assert len(RULE_BY_CODE) == len(RULES)


def test_container_scanner_only_emits_known_codes():
    source = (ROOT / "aegis" / "picklescan.py").read_text() + (ROOT / "aegis" / "modelformats.py").read_text()
    import re

    emitted = set(re.findall(r'code="([a-z_0-9]+)"', source)) | {"dangerous_import", "unknown_import"}
    assert emitted <= set(RULE_BY_CODE)


# --- JSON findings document ---------------------------------------------------------


@pytest.mark.claim("C15")
@needs_core
def test_json_output_validates_against_schema(model_files):
    result = _invoke(*model_files, "--format", "json")
    assert result.exit_code == 1
    document = json.loads(result.stdout)  # stdout is pure JSON: no banner, no Rich markup
    jsonschema.Draft202012Validator(
        FINDINGS_SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    ).validate(document)
    assert document["taxonomies"]["atlas"] == ATLAS_RELEASE
    assert document["max_severity"] == "critical"
    by_file = {entry["path"]: entry for entry in document["files"]}
    assert by_file["clean.safetensors"]["findings"] == []
    rules = {f["rule"] for f in by_file["tampered.safetensors"]["findings"]}
    assert {"AEGIS-T002", "AEGIS-H001"} <= rules
    pickle_finding = by_file["evil.pkl"]["findings"][0]
    assert pickle_finding["rule"] == "AEGIS-P001" and pickle_finding["severity"] == "critical"
    assert pickle_finding["location"]["offset"] >= 0
    assert {"id": "AML.T0011.000", "name": "Unsafe AI Artifacts"} in pickle_finding["references"]["atlas"]


@needs_core
def test_finding_ids_are_stable_across_runs(model_files):
    def ids():
        document = json.loads(_invoke(*model_files, "-f", "json").stdout)
        return sorted(f["id"] for e in document["files"] for f in e["findings"])

    assert ids() == ids()


@needs_core
def test_treatment_commands_are_attached(model_files):
    document = json.loads(_invoke("tampered.safetensors", "-f", "json").stdout)
    metadata = next(f for f in document["files"][0]["findings"] if f["rule"] == "AEGIS-H001")
    assert metadata["treatment"]["action"] == "strip"
    assert "--strip-metadata" in metadata["treatment"]["command"]


# --- SARIF --------------------------------------------------------------------------


@pytest.mark.claim("C15")
@needs_core
def test_sarif_output_validates_against_the_official_schema(model_files):
    result = _invoke(*model_files, "--format", "sarif")
    assert result.exit_code == 1
    sarif = json.loads(result.stdout)
    jsonschema.Draft4Validator(SARIF_SCHEMA).validate(sarif)
    run = sarif["runs"][0]
    assert sarif["version"] == "2.1.0"
    rule_ids = [rule["id"] for rule in run["tool"]["driver"]["rules"]]
    assert {"AEGIS-T002", "AEGIS-H001", "AEGIS-P001"} <= set(rule_ids)
    for res in run["results"]:
        assert run["tool"]["driver"]["rules"][res["ruleIndex"]]["id"] == res["ruleId"]
        uri = res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert not uri.startswith("/")  # relative to the repository root, as code scanning requires
        assert "aegisFindingId/v1" in res["partialFingerprints"]
        assert float(res["properties"]["security-severity"]) > 0
    critical = next(r for r in run["results"] if r["ruleId"] == "AEGIS-P001")
    assert critical["level"] == "error"


@needs_core
def test_sarif_reports_unscannable_files_as_tool_notifications(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_file({"w": _weights(1)}, "clean.safetensors")
    result = _invoke("clean.safetensors", "missing.safetensors", "--format", "sarif")
    assert result.exit_code == 2
    sarif = json.loads(result.stdout)
    jsonschema.Draft4Validator(SARIF_SCHEMA).validate(sarif)
    invocation = sarif["runs"][0]["invocations"][0]
    assert invocation["executionSuccessful"] is False
    assert "missing.safetensors" in invocation["toolExecutionNotifications"][0]["message"]["text"]


# --- Multi-file exit codes ---------------------------------------------------------


@needs_core
def test_multiple_files_take_the_worst_result(model_files):
    assert _invoke("clean.safetensors").exit_code == 0
    assert _invoke("clean.safetensors", "evil.pkl").exit_code == 1
    assert _invoke("clean.safetensors", "nope.safetensors").exit_code == 2


@needs_core
def test_text_format_is_unchanged_by_default(model_files):
    result = _invoke("clean.safetensors")
    assert "Static Analysis Report" in result.stdout
    assert "Model Clean" in result.stdout


# --- Fuzz findings use the same finding object ----------------------------------------


@pytest.mark.claim("C15")
def test_fuzz_findings_validate_against_the_finding_definition(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file as save_torch

    from tests.fuzz_fixtures import ThresholdBackdoor

    torch.manual_seed(0)
    path = tmp_path / "backdoor.safetensors"
    save_torch(ThresholdBackdoor().state_dict(), str(path))
    out = tmp_path / "fuzz.json"
    result = runner.invoke(
        app,
        [
            "fuzz",
            str(path),
            "--arch",
            "tests.fuzz_fixtures:ThresholdBackdoor",
            "--input-shape",
            "2,10",
            "-n",
            "12",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    findings = json.loads(out.read_text())["findings"]
    assert findings and findings[0]["rule"] == "AEGIS-B001"
    finding_schema = {**FINDINGS_SCHEMA["$defs"]["finding"], "$defs": FINDINGS_SCHEMA["$defs"]}
    for finding in findings:
        jsonschema.Draft202012Validator(finding_schema).validate(finding)
