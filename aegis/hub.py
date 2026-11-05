"""Remote model inspection via HTTP range requests on Hugging Face Hub.

Allows inspecting model headers, metadata, secrets, and repository structure before
downloading full model checkpoints over the network.
"""

from __future__ import annotations

import http.client
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from aegis.findings import (
    RULE_BY_CODE,
    Finding,
    _make,
)
from aegis.formatscan import (
    FormatFinding,
    FormatScanResult,
    scan_safetensors_header_bytes,
)

# safetensors format caps the header length at 100 MB.
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

# Hard cap for GGUF header inspection via dynamic range growth.
MAX_GGUF_HEADER_BYTES = 64 * 1024 * 1024
INITIAL_GGUF_CHUNK = 1 * 1024 * 1024

PICKLE_EXTENSIONS = (".bin", ".pt", ".pth", ".pkl", ".ckpt")
MODEL_EXTENSIONS = (".safetensors", ".gguf", *PICKLE_EXTENSIONS)


def _http_get(
    url: str,
    headers: dict[str, str] | None = None,
    limit: int | None = None,
    timeout: float = 30,
) -> tuple[int, bytes]:
    """Single network access gateway for all Hub operations.

    Tests monkeypatch this function and make no live network calls.

    When *limit* is given, at most ``limit + 1`` bytes are read from the
    response body. Reading one byte over the limit lets callers detect when a
    server has ignored the ``Range`` header and is sending the whole file.
    """
    req = urllib.request.Request(url, headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(limit + 1) if limit is not None else resp.read()
            return resp.status, body
    except urllib.error.HTTPError as e:
        try:
            body = e.read(limit + 1) if limit is not None else e.read()
        except (OSError, http.client.HTTPException):
            body = b""
        return e.code, body
    except (OSError, http.client.HTTPException) as e:
        reason = getattr(e, "reason", e)
        return 0, str(reason).encode("utf-8")


@dataclass
class HubRepoInfo:
    repo: str
    sha: str
    revision: str
    files: list[str]
    findings: list[Finding] = field(default_factory=list)


def resolve(repo: str, revision: str = "main") -> HubRepoInfo:
    """Resolve a Hugging Face model repository revision to its commit SHA and file list."""
    encoded_repo = urllib.parse.quote(repo, safe="/")
    encoded_rev = urllib.parse.quote(revision, safe="")
    api_url = f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_rev}"
    status, body = _http_get(api_url, headers={"User-Agent": "aegis-tensor"})
    if status != 200:
        if status == 0:
            err = body.decode("utf-8", errors="replace")
            raise ValueError(f"could not reach Hugging Face: {err}" if err else "could not reach Hugging Face")
        raise ValueError(f"Hugging Face API returned status {status} for {repo}@{revision}")

    data = json.loads(body.decode("utf-8"))
    sha = data.get("sha", "")
    siblings = data.get("siblings", [])
    files = [s["rfilename"] for s in siblings if isinstance(s, dict) and "rfilename" in s]

    findings: list[Finding] = []

    # Check for trust_remote_code requirement
    has_py_files = any(f.endswith(".py") for f in files)
    auto_map = data.get("config", {}).get("auto_map") if isinstance(data.get("config"), dict) else None

    if has_py_files or auto_map:
        findings.append(
            _make(
                RULE_BY_CODE["trust_remote_code"],
                "model requires trust_remote_code=True (loading it runs custom Python code from the repository)",
                {"file": repo},
            )
        )

    return HubRepoInfo(
        repo=repo,
        sha=sha,
        revision=revision,
        files=files,
        findings=findings,
    )


def _network_or_range_error(filename: str, status: int, body: bytes, stage: str = "header prefix") -> FormatScanResult:
    """Return a finding when an HTTP request fails or does not return 206."""
    result = FormatScanResult(path=filename)
    if status == 0:
        err = body.decode("utf-8", errors="replace")
        message = f"could not reach Hugging Face: {err}" if err else "could not reach Hugging Face"
    else:
        message = f"HTTP range requests not supported by server (status {status})"
    result.findings.append(
        FormatFinding(
            "SUSPICIOUS",
            "header_unreadable",
            "header",
            message,
            "quarantine",
        )
    )
    return result


def fetch_safetensors_header(repo: str, sha: str, filename: str) -> FormatScanResult:
    """Fetch and scan the header of a remote safetensors file using two HTTP range requests."""
    encoded_repo = urllib.parse.quote(repo, safe="/")
    encoded_sha = urllib.parse.quote(sha, safe="")
    encoded_filename = urllib.parse.quote(filename, safe="/")
    file_url = f"https://huggingface.co/{encoded_repo}/resolve/{encoded_sha}/{encoded_filename}"

    # Request 1: 8-byte length prefix
    status, prefix = _http_get(file_url, headers={"Range": "bytes=0-7"}, limit=8)
    if status != 206:
        return _network_or_range_error(filename, status, prefix, stage="header prefix")

    if len(prefix) != 8:
        result = FormatScanResult(path=filename)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"failed to read safetensors header length prefix (got {len(prefix)} bytes)",
                "quarantine",
            )
        )
        return result

    (header_len,) = struct.unpack("<Q", prefix)

    # Refuse zero-length headers without issuing a second request
    if header_len == 0:
        result = FormatScanResult(path=filename, header_bytes=0)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                "safetensors header length is 0 (refusing to fetch)",
                "quarantine",
            )
        )
        return result

    # Refuse oversized headers without issuing a second request.
    # The length field is attacker-controlled; the safetensors format caps it at 100 MB.
    if header_len > MAX_SAFETENSORS_HEADER_BYTES:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"safetensors header length {header_len} exceeds 100 MB limit (refusing to fetch)",
                "quarantine",
            )
        )
        return result

    # Request 2: header bytes
    status, raw = _http_get(file_url, headers={"Range": f"bytes=8-{7 + header_len}"}, limit=header_len)
    if status != 206:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        if status == 0:
            err = raw.decode("utf-8", errors="replace")
            message = f"could not reach Hugging Face: {err}" if err else "could not reach Hugging Face"
        else:
            message = f"HTTP range requests not supported for header bytes (status {status})"
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                message,
                "quarantine",
            )
        )
        return result

    if len(raw) != header_len:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"failed to read complete safetensors header bytes (expected {header_len}, got {len(raw)})",
                "quarantine",
            )
        )
        return result

    # Validate that raw bytes decode as a valid JSON object.
    # scan_safetensors_header_bytes expects valid JSON dict and will raise otherwise.
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            result = FormatScanResult(path=filename, header_bytes=header_len)
            result.findings.append(
                FormatFinding(
                    "SUSPICIOUS",
                    "header_unreadable",
                    "header",
                    f"safetensors header is not a JSON object (got {type(parsed).__name__})",
                    "quarantine",
                )
            )
            return result
        return scan_safetensors_header_bytes(raw, header_len, filename)
    except (ValueError, TypeError) as e:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"safetensors header could not be parsed: {e}",
                "quarantine",
            )
        )
        return result


def fetch_gguf_header(repo: str, sha: str, filename: str) -> Any:
    """Fetch and scan the header of a remote GGUF file using growing HTTP range requests."""
    from aegis.modelformats import (
        ModelFormatResult,
        _check_secrets,
        _check_template,
        chat_templates,
        parse_gguf_header,
    )

    result = ModelFormatResult(path=filename, format="gguf")
    encoded_filename = urllib.parse.quote(filename, safe="/")
    file_url = f"https://huggingface.co/{repo}/resolve/{sha}/{encoded_filename}"
    data = b""
    chunk_size = INITIAL_GGUF_CHUNK

    while True:
        offset = len(data)
        if offset >= MAX_GGUF_HEADER_BYTES:
            result.add(
                "SUSPICIOUS",
                f"GGUF header exceeds {MAX_GGUF_HEADER_BYTES // (1024 * 1024)} MB limit",
                filename,
                code="gguf_malformed",
            )
            return result

        fetch_size = min(chunk_size, MAX_GGUF_HEADER_BYTES - offset)
        end_byte = offset + fetch_size - 1
        status, chunk = _http_get(file_url, headers={"Range": f"bytes={offset}-{end_byte}"})
        if status != 206 or not chunk:
            result.add(
                "SUSPICIOUS",
                f"failed to read GGUF header bytes (HTTP {status})",
                filename,
                code="gguf_malformed",
            )
            return result

        data += chunk

        try:
            metadata, _infos, _end = parse_gguf_header(data)
            break
        except (ValueError, struct.error) as error:
            err_msg = str(error)
            if "truncated" not in err_msg and "past the end" not in err_msg:
                result.add("SUSPICIOUS", f"malformed GGUF ({error})", filename, code="gguf_malformed")
                return result
            if len(chunk) < fetch_size:
                result.add("SUSPICIOUS", f"truncated GGUF file ({error})", filename, code="gguf_malformed")
                return result
            chunk_size = min(chunk_size * 2, 16 * 1024 * 1024)

    for key, value in chat_templates(metadata).items():
        _check_template(value, result, f"{filename}:{key}")
    _check_secrets(metadata, result, filename)
    return result


def scan_remote_file(repo: str, sha: str, filename: str) -> Any:
    """Scan a remote repository file without downloading full weights."""
    lower_name = filename.lower()
    if lower_name.endswith(PICKLE_EXTENSIONS):
        result = FormatScanResult(path=filename)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "pickle_needs_download",
                filename,
                "pickle format: full download required to scan; prefer the safetensors version",
                "refuse",
            )
        )
        return result

    if lower_name.endswith(".safetensors"):
        return fetch_safetensors_header(repo, sha, filename)

    if lower_name.endswith(".gguf"):
        return fetch_gguf_header(repo, sha, filename)

    return FormatScanResult(path=filename)


def parse_hf_target(target: str) -> tuple[str, str]:
    """Parse 'hf://org/repo[@revision]' into (repo, revision)."""
    raw = target[len("hf://") :] if target.startswith("hf://") else target
    if "@" in raw:
        repo, revision = raw.split("@", 1)
    else:
        repo, revision = raw, "main"
    return repo.strip("/"), revision


def scan_remote_repo(target: str) -> list[dict[str, Any]]:
    """Inspect all remote model files in a Hugging Face repository."""
    from aegis.findings import _make, from_container, from_safetensors
    from aegis.modelformats import ModelFormatResult
    from aegis.verdict import file_verdict

    repo, revision = parse_hf_target(target)
    hub_info = resolve(repo, revision)
    candidates = [f for f in hub_info.files if f.lower().endswith(MODEL_EXTENSIONS)]
    repo_findings = [
        _make("AEGIS-H005", f.message, {"file": f"hf://{hub_info.repo}@{hub_info.sha}"}) for f in hub_info.findings
    ]
    if not candidates:
        p = f"hf://{hub_info.repo}@{hub_info.sha}"
        entry = {
            "path": p,
            "format": None,
            "kind": "format",
            "findings": repo_findings,
            "tensors": [],
            "verdict": file_verdict(p, None, repo_findings),
        }
        return [entry]

    outcomes = []
    for filename in candidates:
        file_path = f"hf://{hub_info.repo}@{hub_info.sha}/{filename}"
        res = scan_remote_file(hub_info.repo, hub_info.sha, filename)
        is_gguf = isinstance(res, ModelFormatResult)
        fmt = "gguf" if is_gguf else ("pickle" if filename.lower().endswith(PICKLE_EXTENSIONS) else "safetensors")
        kind = "format" if is_gguf else ("container" if fmt == "pickle" else "safetensors")
        findings = (from_container(file_path, res) if is_gguf else from_safetensors(file_path, [], res)) + [
            _make("AEGIS-H005", f.message, {"file": file_path}) for f in hub_info.findings
        ]
        outcome: dict[str, Any] = {
            "path": file_path,
            "format": fmt,
            "kind": kind,
            "findings": findings,
            "tensors": [],
            "verdict": file_verdict(file_path, fmt, findings),
        }
        if kind in ("format", "container"):
            outcome["container"] = res
        else:
            outcome["header"], outcome["tensor_results"] = res, []
        outcomes.append(outcome)
    return outcomes
