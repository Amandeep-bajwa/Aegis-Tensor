"""Versioned findings: one record per issue, shared by every scanner and output format.

A finding has a stable rule ID, severity, confidence, a precise location, the evidence
behind it, the treatment, and taxonomy references (MITRE ATLAS, pinned in aegis/atlas.py;
OWASP ML and LLM Top 10). The JSON document follows docs/schema/findings-v1.schema.json,
and to_sarif() produces SARIF 2.1.0 for GitHub code scanning.

Confidence values are priors set from the synthetic evaluations in docs/THREAT_MATRIX.md.
They are not yet calibrated on real models (see docs/calibration.md).
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.atlas import ATLAS_RELEASE, ATLAS_TECHNIQUES

SCHEMA_VERSION = "1.1.0"
SCHEMA_ID = "https://github.com/KrishG7/Aegis-Tensor/blob/main/docs/schema/findings-v1.schema.json"

OWASP = {
    "ML06": "ML06:2023 ML Supply Chain Attacks",
    "ML10": "ML10:2023 Model Poisoning",
    "LLM03": "LLM03:2025 Supply Chain",
    "LLM04": "LLM04:2025 Data and Model Poisoning",
}

# SARIF level and GitHub security-severity per severity.
SARIF_LEVEL = {"critical": "error", "high": "error", "medium": "warning", "low": "note"}
SECURITY_SEVERITY = {"critical": "9.5", "high": "8.0", "medium": "5.5", "low": "2.0"}

_STEGO = {"atlas": ["AML.T0018.002", "AML.T0010.003"], "owasp": ["ML06", "LLM03"]}
_CONTAINER = {"atlas": ["AML.T0011.000", "AML.T0010.003"], "owasp": ["ML06", "LLM03"]}
_BACKDOOR = {"atlas": ["AML.T0018.000", "AML.T0058"], "owasp": ["ML10", "LLM04"]}

# rule id -> catalog entry. `treatment` is the action; commands are filled in per finding.
RULES: dict[str, dict[str, Any]] = {
    "AEGIS-T001": {
        "code": "pooled_entropy",
        "name": "high-tensor-entropy",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Whole-tensor entropy above threshold",
        "treatment": "inspect",
        **_STEGO,
    },
    "AEGIS-T002": {
        "code": "exponent_lane",
        "name": "overwritten-float-exponents",
        "severity": "high",
        "confidence": 0.9,
        "title": "Sign/exponent bytes are near-uniform: float values overwritten",
        "treatment": "quarantine",
        **_STEGO,
    },
    "AEGIS-T003": {
        "code": "out_of_range",
        "name": "implausible-float-values",
        "severity": "high",
        "confidence": 0.9,
        "title": "Non-finite or implausibly large values: float values overwritten",
        "treatment": "quarantine",
        **_STEGO,
    },
    "AEGIS-T004": {
        "code": "low_plane",
        "name": "structured-low-mantissa-bits",
        "severity": "high",
        "confidence": 0.8,
        "title": "Structured data in the least significant mantissa bits",
        "treatment": "disarm",
        **_STEGO,
    },
    "AEGIS-T005": {
        "code": "benford",
        "name": "benford-deviation",
        "severity": "low",
        "confidence": 0.2,
        "title": "Leading digits deviate from Benford's law",
        "treatment": "inspect",
        **_STEGO,
    },
    "AEGIS-T006": {
        "code": "unreadable",
        "name": "unreadable-tensor",
        "severity": "high",
        "confidence": 0.7,
        "title": "Tensor data could not be read",
        "treatment": "quarantine",
        **_STEGO,
    },
    "AEGIS-H001": {
        "code": "metadata",
        "name": "oversized-metadata",
        "severity": "high",
        "confidence": 0.7,
        "title": "Oversized __metadata__ value: possible hidden payload",
        "treatment": "strip",
        **_STEGO,
    },
    "AEGIS-H002": {
        "code": "header_whitespace",
        "name": "header-whitespace-channel",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Unexplained whitespace in the safetensors header",
        "treatment": "strip",
        **_STEGO,
    },
    "AEGIS-H003": {
        "code": "tensor_name",
        "name": "oversized-tensor-name",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Oversized tensor name: possible hidden payload",
        "treatment": "quarantine",
        **_STEGO,
    },
    "AEGIS-H004": {
        "code": "secret",
        "name": "credential-in-metadata",
        "severity": "medium",
        "confidence": 0.8,
        "title": "Credential (API key, token or private key) left in model metadata",
        "treatment": "strip",
        "atlas": ["AML.T0010.003"],
        "owasp": ["ML06", "LLM03"],
    },
    "AEGIS-H005": {
        "code": "trust_remote_code",
        "name": "hub-trust-remote-code",
        "severity": "medium",
        "confidence": 0.8,
        "title": "Model repository requires trust_remote_code=True",
        "treatment": "inspect",
        "atlas": ["AML.T0010.003"],
        "owasp": ["ML06", "LLM03"],
    },
    "AEGIS-H006": {
        "code": "pickle_needs_download",
        "name": "hub-pickle-needs-download",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Pickle checkpoint requires full download to scan",
        "treatment": "refuse",
        "atlas": ["AML.T0010.003"],
        "owasp": ["ML06", "LLM03"],
    },
    "AEGIS-H007": {
        "code": "header_unreadable",
        "name": "hub-header-unreadable",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Remote model header could not be read",
        "treatment": "quarantine",
        "atlas": ["AML.T0010.003"],
        "owasp": ["ML06", "LLM03"],
    },
    "AEGIS-P001": {
        "code": "dangerous_import",
        "name": "pickle-code-execution",
        "severity": "critical",
        "confidence": 0.95,
        "title": "Pickle imports a code-execution primitive",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P002": {
        "code": "unknown_import",
        "name": "pickle-unknown-global",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Pickle imports a global outside the checkpoint allowlist",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-P003": {
        "code": "unresolved_import",
        "name": "pickle-obfuscated-import",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Pickle import could not be resolved statically",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P004": {
        "code": "extension_registry",
        "name": "pickle-extension-registry",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Pickle resolves a global through the copyreg extension registry",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P005": {
        "code": "malformed_pickle",
        "name": "malformed-pickle",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Malformed pickle stream (scan continued past it)",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P006": {
        "code": "zip_tampered",
        "name": "tampered-zip",
        "severity": "medium",
        "confidence": 0.7,
        "title": "Zip entry tampered (flags, CRC or central directory)",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P007": {
        "code": "archive_7z",
        "name": "7z-repackaged-checkpoint",
        "severity": "medium",
        "confidence": 0.7,
        "title": "Checkpoint repackaged as 7z",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P008": {
        "code": "npy_object",
        "name": "numpy-object-array",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Object-dtype numpy array requires unpickling",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-P009": {
        "code": "malformed_npy",
        "name": "malformed-npy",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Malformed .npy header",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-P010": {
        "code": "no_payload",
        "name": "empty-archive",
        "severity": "medium",
        "confidence": 0.4,
        "title": "Archive contains no pickle or numpy payload",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-P011": {
        "code": "container_unreadable",
        "name": "unreadable-container",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Container could not be read",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-P012": {
        "code": "unknown_format",
        "name": "unknown-format",
        "severity": "medium",
        "confidence": 0.4,
        "title": "Unrecognised model file format",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-P013": {
        "code": "unresolved_callable",
        "name": "pickle-runtime-callable",
        "severity": "critical",
        "confidence": 0.8,
        "title": "Pickle calls a callable built at runtime",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-G001": {
        "code": "gguf_template",
        "name": "chat-template-injection",
        "severity": "critical",
        "confidence": 0.8,
        "title": "GGUF chat template reaches Python internals (Jinja template injection)",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-G002": {
        "code": "gguf_malformed",
        "name": "malformed-gguf",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Malformed GGUF structure",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-G003": {
        "code": "gguf_template_unverifiable",
        "name": "chat-template-unverifiable",
        "severity": "medium",
        "confidence": 0.4,
        "title": "GGUF chat template cannot be verified statically",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-K001": {
        "code": "keras_lambda",
        "name": "keras-lambda-layer",
        "severity": "high",
        "confidence": 0.9,
        "title": "Keras Lambda layer carries Python bytecode",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-K002": {
        "code": "keras_custom_object",
        "name": "keras-custom-object",
        "severity": "medium",
        "confidence": 0.6,
        "title": "Keras model requires a custom (non-Keras) object",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-X001": {
        "code": "onnx_external_path",
        "name": "onnx-external-data-escape",
        "severity": "high",
        "confidence": 0.8,
        "title": "ONNX external data path leaves the model directory",
        "treatment": "refuse",
        **_CONTAINER,
    },
    "AEGIS-X002": {
        "code": "onnx_external_data",
        "name": "onnx-external-data",
        "severity": "low",
        "confidence": 0.3,
        "title": "ONNX tensor stored in an external file (not scanned)",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-X003": {
        "code": "onnx_custom_op",
        "name": "onnx-custom-operator",
        "severity": "medium",
        "confidence": 0.6,
        "title": "ONNX graph uses a custom operator domain",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-F001": {
        "code": "format_unavailable",
        "name": "format-not-inspected",
        "severity": "medium",
        "confidence": 0.3,
        "title": "Format recognised but not inspected (optional library missing)",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-E001": {
        "code": "engine_finding",
        "name": "engine-finding",
        "severity": "medium",
        "confidence": 0.7,
        "title": "Optional engine (Fickling or ModelScan) reported an issue",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-E002": {
        "code": "engine_disagreement",
        "name": "engine-disagreement",
        "severity": "medium",
        "confidence": 0.5,
        "title": "Aegis and an optional engine disagree about this file",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-E003": {
        "code": "engine_error",
        "name": "engine-error",
        "severity": "low",
        "confidence": 0.3,
        "title": "Optional engine could not check this file",
        "treatment": "inspect",
        **_CONTAINER,
    },
    "AEGIS-B001": {
        "code": "activation_spike",
        "name": "backdoor-activation-spike",
        "severity": "high",
        "confidence": 0.6,
        "title": "Layer activation spikes under fuzzing (possible backdoor)",
        "treatment": "inspect",
        **_BACKDOOR,
    },
}
RULE_BY_CODE = {entry["code"]: rule_id for rule_id, entry in RULES.items()}


@dataclass
class Finding:
    rule: str
    severity: str
    confidence: float
    message: str
    location: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    treatment: dict[str, Any] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            key = (
                f"{self.rule}|{self.location.get('file')}|{self.location.get('tensor')}|"
                f"{self.location.get('entry')}|{self.location.get('offset')}|{self.location.get('header_key')}"
            )
            self.id = hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def code(self) -> str:
        return RULES[self.rule]["code"]


def _make(
    rule: str,
    message: str,
    location: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    command: str | None = None,
    severity: str | None = None,
) -> Finding:
    entry = RULES[rule]
    return Finding(
        rule=rule,
        severity=severity or entry["severity"],
        confidence=entry["confidence"],
        message=message,
        location={k: v for k, v in location.items() if v is not None},
        evidence={k: v for k, v in (evidence or {}).items() if v is not None},
        treatment={"action": entry["treatment"], **({"command": command} if command else {})},
        references={
            "atlas": [{"id": t, "name": ATLAS_TECHNIQUES[t]} for t in entry["atlas"]],
            "owasp": [OWASP[o] for o in entry["owasp"]],
        },
    )


def _disarm_command(path: str, strip: bool = False) -> str:
    from aegis.disarm import default_disarm_path

    return f"aegis disarm {path} {default_disarm_path(path)}" + (" --strip-metadata" if strip else "")


# --- Builders ------------------------------------------------------------------------


def from_safetensors(path: str, tensor_results, header_result=None) -> list[Finding]:
    findings = []
    for result in tensor_results:
        # The Rust core appends a rule and its reason together, so the lists always pair up.
        for rule_code, reason in zip(result.rules, result.anomaly_reasons, strict=True):
            rule = RULE_BY_CODE.get(rule_code, "AEGIS-T001")
            command = _disarm_command(path) if RULES[rule]["treatment"] == "disarm" else None
            findings.append(
                _make(
                    rule,
                    reason,
                    {"file": path, "tensor": result.name},
                    evidence={
                        "dtype": result.dtype,
                        "elements": result.num_elements,
                        "entropy": round(result.entropy, 4),
                        "exponent_lane_entropy": _round(result.exponent_lane_entropy),
                        "low_plane_entropy": _round(result.low_plane_entropy),
                        "out_of_range_fraction": _round(result.out_of_range_fraction, 6),
                        "benford_mad": round(result.benford_mad, 4) if rule_code == "benford" else None,
                    },
                    command=command,
                )
            )
    for finding in header_result.findings if header_result is not None else []:
        rule = RULE_BY_CODE[finding.channel]
        location = {"file": path}
        if finding.channel == "metadata":
            location["header_key"] = finding.where
        elif finding.channel == "tensor_name":
            location["tensor"] = finding.where
        else:
            location["header_key"] = "header"
        command = _disarm_command(path, strip=True) if RULES[rule]["treatment"] == "strip" else None
        findings.append(_make(rule, finding.message, location, command=command))
    return findings


def from_container(path: str, container_result) -> list[Finding]:
    findings = []
    for finding in container_result.findings:
        rule = RULE_BY_CODE.get(finding.code, "AEGIS-P012")
        entry, offset = _split_location(finding.location)
        severity = "critical" if finding.severity == "CRITICAL" else RULES[rule]["severity"]
        findings.append(
            _make(
                rule,
                finding.message,
                {"file": path, "entry": entry, "offset": offset},
                evidence={"format": container_result.format},
                severity=severity,
            )
        )
    return findings


def from_fuzz(path: str, report) -> list[Finding]:
    return [
        _make(
            "AEGIS-B001",
            f"layer {a.layer_name} ({a.region}, {a.depth}) reached {a.spike_ratio:.1f}x its clean activation "
            f"under input '{a.trigger_tag}'",
            {"file": path, "layer": a.layer_name},
            evidence={
                "spike_ratio": round(a.spike_ratio, 3),
                "z_score": round(a.z_score, 3),
                "peak_l_inf": round(a.peak_l_inf, 4),
                "baseline_max_l_inf": round(a.baseline_max_l_inf, 4),
                "trigger": a.trigger_tag,
                "region": a.region,
            },
        )
        for a in report.layer_anomalies
    ]


def from_engines(path: str, results, aegis_findings: list[Finding]) -> list[Finding]:
    """Engine issues as findings (E001), plus a finding wherever an engine and Aegis disagree (E002)."""
    findings = []
    aegis_worst = max_severity(aegis_findings)
    for result in results:
        for issue in result.issues:
            entry, offset = _split_location(issue.location) if issue.location else (None, None)
            findings.append(
                _make(
                    "AEGIS-E001",
                    issue.message,
                    {"file": path, "entry": entry or issue.location or None, "offset": offset},
                    evidence={"engine": result.engine, "engine_version": result.version},
                    severity=issue.severity,
                )
            )
        if result.error:
            findings.append(
                _make(
                    "AEGIS-E003",
                    f"{result.engine}: {result.error}",
                    {"file": path},
                    evidence={"engine": result.engine, "engine_version": result.version},
                )
            )
        if not result.ran:
            continue
        engine_worst = result.worst

        def rank(level: str | None) -> int:
            return -1 if level is None else SEVERITY_RANK[level]

        if rank(engine_worst) >= SEVERITY_RANK["high"] and rank(aegis_worst) < SEVERITY_RANK["high"]:
            findings.append(
                _make(
                    "AEGIS-E002",
                    f"{result.engine} rates this file {engine_worst}; Aegis's own checks rate it "
                    f"{aegis_worst or 'clean'}. Treat it as {engine_worst} until the difference is understood.",
                    {"file": path},
                    evidence={"engine": result.engine, "engine_worst": engine_worst, "aegis_worst": aegis_worst},
                    severity="medium",
                )
            )
        elif rank(aegis_worst) >= SEVERITY_RANK["high"] and rank(engine_worst) < SEVERITY_RANK["medium"]:
            findings.append(
                _make(
                    "AEGIS-E002",
                    f"Aegis rates this file {aegis_worst}; {result.engine} found nothing. "
                    f"{result.engine} may not cover this threat (see docs/THREAT_MATRIX.md).",
                    {"file": path},
                    evidence={"engine": result.engine, "engine_worst": engine_worst, "aegis_worst": aegis_worst},
                    severity="low",
                )
            )
    return findings


def _split_location(location: str) -> tuple[str | None, int | None]:
    """'archive.pt:archive/data.pkl@35' -> ('archive/data.pkl', 35)."""
    entry, offset = location, None
    if "@" in location:
        entry, _, tail = location.rpartition("@")
        offset = int(tail) if tail.isdigit() else None
    return (entry.split(":", 1)[1] if ":" in entry else None), offset


def _round(value, digits: int = 4):
    return None if value is None else round(value, digits)


# --- Documents -----------------------------------------------------------------------

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def max_severity(findings: list[Finding]) -> str | None:
    return max((f.severity for f in findings), key=SEVERITY_RANK.__getitem__, default=None)


def to_json_document(files: list[dict[str, Any]], fail_on: str, tool_version: str) -> dict[str, Any]:
    """``files`` entries: {path, format, findings: [Finding], error?: str, tensors?: [...]}."""
    all_findings = [f for entry in files for f in entry["findings"]]
    return {
        "$schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "aegis-tensor", "version": tool_version},
        "taxonomies": {"atlas": dict(ATLAS_RELEASE), "owasp": sorted(OWASP.values())},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fail_on": fail_on,
        "max_severity": max_severity(all_findings),
        **({"verdict": _overall([e["verdict"] for e in files])} if all(e.get("verdict") for e in files) else {}),
        "files": [
            {
                "path": entry["path"],
                "format": entry.get("format"),
                "max_severity": max_severity(entry["findings"]),
                **({"verdict": entry["verdict"]} if entry.get("verdict") else {}),
                **(
                    {
                        "engines": [
                            {"name": r.engine, "version": r.version, "ran": r.ran, "error": r.error, "worst": r.worst}
                            for r in entry["engines"]
                        ]
                    }
                    if entry.get("engines")
                    else {}
                ),
                **({"error": entry["error"]} if entry.get("error") else {}),
                "findings": [asdict(f) for f in entry["findings"]],
                **({"tensors": entry["tensors"]} if entry.get("tensors") is not None else {}),
            }
            for entry in files
        ],
    }


def _overall(verdicts):
    from aegis.verdict import overall

    return overall(verdicts)


def to_sarif(files: list[dict[str, Any]], tool_version: str) -> dict[str, Any]:
    """SARIF 2.1.0 for GitHub code scanning. File paths should be relative to the repo root."""
    used = sorted({f.rule for entry in files for f in entry["findings"]})
    rules = []
    for rule_id in used:
        entry = RULES[rule_id]
        atlas = ", ".join(f"{t} {ATLAS_TECHNIQUES[t]}" for t in entry["atlas"])
        owasp = ", ".join(OWASP[o] for o in entry["owasp"])
        rules.append(
            {
                "id": rule_id,
                "name": entry["name"],
                "shortDescription": {"text": entry["title"]},
                "fullDescription": {"text": f"{entry['title']}. MITRE ATLAS: {atlas}. OWASP: {owasp}."},
                "help": {"text": f"Treatment: {entry['treatment']}. See docs/THREAT_MATRIX.md."},
                "defaultConfiguration": {"level": SARIF_LEVEL[entry["severity"]]},
                "properties": {
                    "tags": ["security", "ai-supply-chain"]
                    + list(entry["atlas"])
                    + [OWASP[o].split(" ")[0] for o in entry["owasp"]],
                    "security-severity": SECURITY_SEVERITY[entry["severity"]],
                    "precision": "high"
                    if entry["confidence"] >= 0.8
                    else "medium"
                    if entry["confidence"] >= 0.5
                    else "low",
                },
            }
        )
    rule_index = {rule["id"]: i for i, rule in enumerate(rules)}
    results = []
    for entry in files:
        for f in entry["findings"]:
            logical = [
                {"name": f.location[key], "kind": kind}
                for key, kind in (
                    ("tensor", "member"),
                    ("entry", "module"),
                    ("layer", "member"),
                    ("header_key", "member"),
                )
                if key in f.location
            ]
            message = f.message + (f" Treatment: {f.treatment['command']}" if f.treatment.get("command") else "")
            results.append(
                {
                    "ruleId": f.rule,
                    "ruleIndex": rule_index[f.rule],
                    "level": SARIF_LEVEL[f.severity],
                    "message": {"text": message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": Path(f.location["file"]).as_posix()},
                                "region": {"startLine": 1},
                            },
                            **({"logicalLocations": logical} if logical else {}),
                        }
                    ],
                    "partialFingerprints": {"aegisFindingId/v1": f.id},
                    "properties": {
                        "security-severity": SECURITY_SEVERITY[f.severity],
                        "confidence": f.confidence,
                        "evidence": f.evidence,
                        "treatment": f.treatment,
                        **({"offset": f.location["offset"]} if "offset" in f.location else {}),
                    },
                }
            )
    invocation_errors = [
        {"level": "error", "message": {"text": f"{entry['path']}: {entry['error']}"}}
        for entry in files
        if entry.get("error")
    ]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Aegis-Tensor",
                        "version": tool_version,
                        "informationUri": "https://github.com/KrishG7/Aegis-Tensor",
                        "rules": rules,
                    }
                },
                "invocations": [
                    {
                        "executionSuccessful": not invocation_errors,
                        **({"toolExecutionNotifications": invocation_errors} if invocation_errors else {}),
                    }
                ],
                "results": results,
            }
        ],
    }
