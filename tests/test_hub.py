"""Tests for remote model scanning via HTTP range requests (aegis.hub).

Every test replaces `aegis.hub._http_get` with a fake (monkeypatch). No test touches the network
unless explicitly invoked with `-m network`.
"""

import http.server
import json
import struct
import threading

import numpy as np
import pytest
from safetensors.numpy import save_file

from aegis.formatscan import scan_safetensors_header
from aegis.hub import (
    _http_get,
    fetch_safetensors_header,
    resolve,
    scan_remote_file,
)


def _weights(n=1024):
    return np.random.default_rng(42).normal(0, 0.02, n).astype(np.float32)


def test_resolve_pins_revision_to_commit_sha(monkeypatch):
    """resolve() calls the HF API and pins the revision to the commit SHA."""
    expected_sha = "9e2c81d85642a8b38914b1021481a79d01234567"
    api_payload = {
        "sha": expected_sha,
        "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "config.json"}],
    }

    def fake_get(url, headers=None):
        assert "api/models/test-org/test-model/revision/main" in url
        return 200, json.dumps(api_payload).encode("utf-8")

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    info = resolve("test-org/test-model", revision="main")
    assert info.sha == expected_sha
    assert "model.safetensors" in info.files
    assert info.repo == "test-org/test-model"


def test_resolve_raises_on_http_error(monkeypatch):
    """resolve() raises ValueError when the Hugging Face API returns non-200."""
    monkeypatch.setattr("aegis.hub._http_get", lambda url, headers=None: (404, b"Not Found"))
    with pytest.raises(ValueError, match="Hugging Face API returned status 404"):
        resolve("nonexistent/model")


def test_safetensors_header_is_fetched_with_two_range_requests(monkeypatch, tmp_path):
    """Assert exactly the Range: bytes=0-7 request and then the header-range request, and nothing else."""
    test_file = tmp_path / "model.safetensors"
    save_file({"w": _weights()}, test_file, metadata={"tag": "safe"})
    file_bytes = test_file.read_bytes()
    (header_len,) = struct.unpack("<Q", file_bytes[:8])

    calls = []

    def fake_get(url, headers=None):
        headers = headers or {}
        range_header = headers.get("Range")
        calls.append((url, range_header))
        if range_header == "bytes=0-7":
            return 206, file_bytes[:8]
        if range_header == f"bytes=8-{7 + header_len}":
            return 206, file_bytes[8 : 8 + header_len]
        pytest.fail(f"Unexpected Range request: {range_header}")

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = fetch_safetensors_header("test-org/model", "abc1234", "model.safetensors")
    assert len(calls) == 2
    assert calls[0][1] == "bytes=0-7"
    assert calls[1][1] == f"bytes=8-{7 + header_len}"
    assert result.header_bytes == header_len
    assert not result.is_suspicious


def test_oversized_header_length_is_refused_without_fetching(monkeypatch):
    """Length field 2^40. Expect a finding and no second request."""
    calls = []
    oversized_length = 1 << 40

    def fake_get(url, headers=None):
        headers = headers or {}
        calls.append((url, headers.get("Range")))
        return 206, struct.pack("<Q", oversized_length)

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = fetch_safetensors_header("test-org/model", "abc1234", "model.safetensors")
    assert len(calls) == 1
    assert calls[0][1] == "bytes=0-7"
    assert result.is_suspicious
    assert any("exceeds 100 MB" in f.message for f in result.findings)
    assert any(f.channel == "header_unreadable" for f in result.findings)


def test_zero_header_length_is_refused_without_fetching(monkeypatch):
    """Length field 0. Expect a finding and no second request."""
    calls = []

    def fake_get(url, headers=None, limit=None):
        headers = headers or {}
        calls.append((url, headers.get("Range")))
        return 206, struct.pack("<Q", 0)

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = fetch_safetensors_header("test-org/model", "abc1234", "model.safetensors")
    assert len(calls) == 1
    assert calls[0][1] == "bytes=0-7"
    assert result.is_suspicious
    assert any("header length is 0" in f.message for f in result.findings)
    assert any(f.channel == "header_unreadable" for f in result.findings)


def test_safetensors_server_ignoring_range_is_refused(monkeypatch):
    """Server that answers 200 instead of 206 is flagged as not supporting range requests."""
    calls = []

    def fake_get(url, headers=None, limit=None):
        calls.append((url, limit))
        return 200, b"this-is-a-full-file-not-range"

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = fetch_safetensors_header("test-org/model", "abc1234", "model.safetensors")
    assert len(calls) == 1
    assert calls[0][1] == 8
    assert result.is_suspicious
    assert any("range requests not supported" in f.message.lower() for f in result.findings)


def test_metadata_payload_found_remotely_matches_local_scan(monkeypatch, tmp_path):
    """The same file scanned locally and through the fake hub gives the same header findings."""
    test_file = tmp_path / "payload.safetensors"
    save_file({"w": _weights()}, test_file, metadata={"hidden_note": "X" * 15000})

    local_result = scan_safetensors_header(test_file)
    assert local_result.is_suspicious

    file_bytes = test_file.read_bytes()
    (header_len,) = struct.unpack("<Q", file_bytes[:8])

    def fake_get(url, headers=None):
        headers = headers or {}
        range_header = headers.get("Range")
        if range_header == "bytes=0-7":
            return 206, file_bytes[:8]
        if range_header == f"bytes=8-{7 + header_len}":
            return 206, file_bytes[8 : 8 + header_len]
        return 404, b""

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    remote_result = fetch_safetensors_header("test-org/model", "abc1234", "payload.safetensors")
    assert remote_result.is_suspicious
    assert len(remote_result.findings) == len(local_result.findings)
    assert remote_result.findings[0].channel == local_result.findings[0].channel
    assert remote_result.findings[0].where == local_result.findings[0].where
    assert remote_result.findings[0].message == local_result.findings[0].message


def test_leaked_token_in_remote_metadata_is_masked(monkeypatch, tmp_path):
    """Leaked credentials in remote metadata are detected and masked in findings."""
    raw_token = "hf_0123456789abcdef0123456789abcdef01"  # noqa: S105
    test_file = tmp_path / "leaked.safetensors"
    save_file({"w": _weights()}, test_file, metadata={"api_key": raw_token})
    file_bytes = test_file.read_bytes()
    (header_len,) = struct.unpack("<Q", file_bytes[:8])

    def fake_get(url, headers=None):
        headers = headers or {}
        range_header = headers.get("Range")
        if range_header == "bytes=0-7":
            return 206, file_bytes[:8]
        if range_header == f"bytes=8-{7 + header_len}":
            return 206, file_bytes[8 : 8 + header_len]
        return 404, b""

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = fetch_safetensors_header("test-org/model", "abc1234", "leaked.safetensors")
    assert result.is_suspicious
    secret_findings = [f for f in result.findings if f.channel == "secret"]
    assert len(secret_findings) == 1
    msg = secret_findings[0].message
    assert "Hugging Face token in metadata" in msg
    assert raw_token not in msg  # plaintext token must never be exposed
    assert "hf_" in msg


def test_pickle_file_is_reported_not_downloaded(monkeypatch):
    """Pickle files are reported as requiring full download with zero range requests."""
    calls = []

    def fake_get(url, headers=None):
        calls.append(url)
        return 200, b"should-not-be-called"

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result = scan_remote_file("test-org/model", "abc1234", "pytorch_model.bin")
    assert len(calls) == 0
    assert result.is_suspicious
    assert any("pickle format: full download required to scan" in f.message for f in result.findings)
    assert any(f.channel == "pickle_needs_download" for f in result.findings)


def test_trust_remote_code_repo_gets_a_finding(monkeypatch):
    """Repositories with .py files or auto_map in config.json receive a trust_remote_code finding."""
    # Case 1: Repo has Python files
    api_payload_py = {
        "sha": "1234567890abcdef1234567890abcdef12345678",
        "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "modeling_custom.py"}],
    }

    def fake_get_py(url, headers=None):
        return 200, json.dumps(api_payload_py).encode("utf-8")

    monkeypatch.setattr("aegis.hub._http_get", fake_get_py)
    info_py = resolve("test-org/custom-model", revision="main")
    assert len(info_py.findings) == 1
    assert info_py.findings[0].code == "trust_remote_code"
    assert info_py.findings[0].rule == "AEGIS-H005"
    assert "custom Python code" in info_py.findings[0].message

    # Case 2: config.json has auto_map entry
    api_payload_automap = {
        "sha": "abcdef1234567890abcdef1234567890abcdef12",
        "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "config.json"}],
        "config": {"auto_map": {"AutoModel": "modeling_remote.RemoteModel"}},
    }

    def fake_get_automap(url, headers=None):
        return 200, json.dumps(api_payload_automap).encode("utf-8")

    monkeypatch.setattr("aegis.hub._http_get", fake_get_automap)
    info_automap = resolve("test-org/automap-model", revision="main")
    assert len(info_automap.findings) == 1
    assert info_automap.findings[0].code == "trust_remote_code"
    assert info_automap.findings[0].rule == "AEGIS-H005"


def test_fetch_safetensors_header_prefix_error(monkeypatch):
    """fetch_safetensors_header reports failure if prefix fetch fails or is truncated."""
    monkeypatch.setattr("aegis.hub._http_get", lambda url, headers=None, limit=None: (500, b""))
    result = fetch_safetensors_header("test-org/model", "abc", "model.safetensors")
    assert result.is_suspicious
    assert any(f.channel == "header_unreadable" for f in result.findings)
    assert any("failed to read" in f.message or "not supported" in f.message for f in result.findings)


def test_fetch_safetensors_header_body_error(monkeypatch):
    """fetch_safetensors_header reports failure if header bytes fetch fails or is truncated."""

    def fake_get(url, headers=None, limit=None):
        headers = headers or {}
        if headers.get("Range") == "bytes=0-7":
            return 206, struct.pack("<Q", 500)
        return 500, b""

    monkeypatch.setattr("aegis.hub._http_get", fake_get)
    result = fetch_safetensors_header("test-org/model", "abc", "model.safetensors")
    assert result.is_suspicious
    assert any(f.channel == "header_unreadable" for f in result.findings)
    assert any("failed to read" in f.message or "not supported" in f.message for f in result.findings)


def test_scan_remote_file_safetensors_and_other(monkeypatch, tmp_path):
    """scan_remote_file routes safetensors to range fetch and leaves unrecognized files clean."""
    test_file = tmp_path / "model.safetensors"
    save_file({"w": _weights()}, test_file)
    file_bytes = test_file.read_bytes()
    (header_len,) = struct.unpack("<Q", file_bytes[:8])

    def fake_get(url, headers=None):
        headers = headers or {}
        range_header = headers.get("Range")
        if range_header == "bytes=0-7":
            return 206, file_bytes[:8]
        if range_header == f"bytes=8-{7 + header_len}":
            return 206, file_bytes[8 : 8 + header_len]
        return 404, b""

    monkeypatch.setattr("aegis.hub._http_get", fake_get)

    result_st = scan_remote_file("test-org/model", "abc", "model.safetensors")
    assert not result_st.is_suspicious

    result_other = scan_remote_file("test-org/model", "abc", "README.md")
    assert not result_other.is_suspicious


def test_http_get_local_server():
    """Verify _http_get execution against a real local socket (no external network)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ok":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"hello-hub")
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not-found")

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    status, body = _http_get(f"http://127.0.0.1:{port}/ok")
    assert status == 200
    assert body == b"hello-hub"

    thread2 = threading.Thread(target=server.handle_request)
    thread2.start()
    status404, body404 = _http_get(f"http://127.0.0.1:{port}/missing")
    assert status404 == 404
    assert body404 == b"not-found"
    server.server_close()


@pytest.mark.network
def test_network_fetch_real_hub():
    """Optional real-network check (excluded by default, run with `pytest -m network`)."""
    status, body = _http_get("https://huggingface.co/api/models/bert-base-uncased/revision/main")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert "sha" in data
