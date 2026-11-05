"""Remote model inspection via HTTP range requests on Hugging Face Hub.

Allows inspecting model headers, metadata, secrets, and repository structure before
downloading full model checkpoints over the network.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from aegis.formatscan import (
    FormatFinding,
    FormatScanResult,
    scan_safetensors_header_bytes,
)

# safetensors format caps the header length at 100 MB.
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

PICKLE_EXTENSIONS = (".bin", ".pt", ".pth", ".pkl", ".ckpt")


def _http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    """Single network access gateway for all Hub operations.

    Tests monkeypatch this function and make no live network calls.
    """
    req = urllib.request.Request(url, headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@dataclass
class HubFinding:
    severity: str
    code: str
    where: str
    message: str
    treatment: str = "inspect"


@dataclass
class HubRepoInfo:
    repo: str
    sha: str
    revision: str
    files: list[str]
    findings: list[HubFinding] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def resolve(repo: str, revision: str = "main") -> HubRepoInfo:
    """Resolve a Hugging Face model repository revision to its commit SHA and file list."""
    api_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
    status, body = _http_get(api_url, headers={"User-Agent": "aegis-tensor"})
    if status != 200:
        raise ValueError(f"Hugging Face API returned status {status} for {repo}@{revision}")

    data = json.loads(body.decode("utf-8"))
    sha = data.get("sha", "")
    siblings = data.get("siblings", [])
    files = [s["rfilename"] for s in siblings if isinstance(s, dict) and "rfilename" in s]

    findings: list[HubFinding] = []

    # Check for trust_remote_code requirement
    has_py_files = any(f.endswith(".py") for f in files)
    auto_map = data.get("config", {}).get("auto_map") if isinstance(data.get("config"), dict) else None

    if has_py_files or auto_map:
        findings.append(
            HubFinding(
                severity="SUSPICIOUS",
                code="trust_remote_code",
                where=repo,
                message=(
                    "model requires trust_remote_code=True (loading it runs custom Python code from the repository)"
                ),
                treatment="inspect",
            )
        )

    return HubRepoInfo(
        repo=repo,
        sha=sha,
        revision=revision,
        files=files,
        findings=findings,
    )


def fetch_safetensors_header(repo: str, sha: str, filename: str) -> FormatScanResult:
    """Fetch and scan the header of a remote safetensors file using two HTTP range requests."""
    file_url = f"https://huggingface.co/{repo}/resolve/{sha}/{filename}"

    # Request 1: 8-byte length prefix
    status, prefix = _http_get(file_url, headers={"Range": "bytes=0-7"})
    if status not in (200, 206) or len(prefix) < 8:
        result = FormatScanResult(path=filename)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "metadata",
                "header",
                f"failed to read safetensors header length prefix (HTTP {status})",
                "quarantine",
            )
        )
        return result

    (header_len,) = struct.unpack("<Q", prefix)

    # Refuse oversized headers without issuing a second request
    if header_len > MAX_SAFETENSORS_HEADER_BYTES or header_len <= 0:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "metadata",
                "header",
                f"safetensors header length {header_len} exceeds 100 MB limit (refusing to fetch)",
                "quarantine",
            )
        )
        return result

    # Request 2: header bytes
    status, raw = _http_get(file_url, headers={"Range": f"bytes=8-{7 + header_len}"})
    if status not in (200, 206) or len(raw) < header_len:
        result = FormatScanResult(path=filename, header_bytes=header_len)
        result.findings.append(
            FormatFinding(
                "SUSPICIOUS",
                "metadata",
                "header",
                f"failed to read complete safetensors header bytes (HTTP {status})",
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
                "metadata",
                filename,
                "pickle format: full download required to scan; prefer the safetensors version",
                "refuse",
            )
        )
        return result

    if lower_name.endswith(".safetensors"):
        return fetch_safetensors_header(repo, sha, filename)

    return FormatScanResult(path=filename)
