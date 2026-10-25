"""Remote model inspection via HTTP range requests on Hugging Face Hub.

Allows inspecting model headers, metadata, secrets, and repository structure before
downloading full model checkpoints over the network.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

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

PICKLE_EXTENSIONS = (".bin", ".pt", ".pth", ".pkl", ".ckpt")


def _http_get(
    url: str,
    headers: dict[str, str] | None = None,
    limit: int | None = None,
) -> tuple[int, bytes]:
    """Single network access gateway for all Hub operations.

    Tests monkeypatch this function and make no live network calls.
    """
    req = urllib.request.Request(url, headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            body = resp.read(limit + 1) if limit is not None else resp.read()
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read(limit + 1) if limit is not None else e.read()
        return e.code, body
    except urllib.error.URLError as e:
        return 0, str(e.reason).encode("utf-8")


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


def _call_http_get(
    url: str,
    headers: dict[str, str] | None = None,
    limit: int | None = None,
) -> tuple[int, bytes]:
    try:
        return _http_get(url, headers=headers, limit=limit)
    except TypeError:
        status, body = _http_get(url, headers=headers)
        if limit is not None and len(body) > limit:
            body = body[: limit + 1]
        return status, body


def fetch_safetensors_header(repo: str, sha: str, filename: str) -> FormatScanResult:
    """Fetch and scan the header of a remote safetensors file using two HTTP range requests."""
    encoded_filename = urllib.parse.quote(filename, safe="/")
    file_url = f"https://huggingface.co/{repo}/resolve/{sha}/{encoded_filename}"

    # Request 1: 8-byte length prefix
    status, prefix = _call_http_get(file_url, headers={"Range": "bytes=0-7"}, limit=8)
    if status != 206:
        result = FormatScanResult(path=filename)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"HTTP range requests not supported by server (status {status})",
                "quarantine",
            )
        )
        return result

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

    # Refuse oversized or zero-length headers without issuing a second request
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
    status, raw = _call_http_get(file_url, headers={"Range": f"bytes=8-{7 + header_len}"}, limit=header_len)
    if status != 206:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "header_unreadable",
                "header",
                f"HTTP range requests not supported for header bytes (status {status})",
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

    return scan_safetensors_header_bytes(raw, header_len, filename)


def scan_remote_file(repo: str, sha: str, filename: str) -> FormatScanResult:
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

    return FormatScanResult(path=filename)
