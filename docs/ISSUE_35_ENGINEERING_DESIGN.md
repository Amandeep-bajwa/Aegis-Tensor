# Issue #35 Engineering Design: Remote Model Scanning via HTTP Range Requests

**Issue**: [#35 [A1] `aegis scan hf://org/repo[@revision]`: scan before downloading](https://github.com/KrishG7/Aegis-Tensor/issues/35)  
**Branch**: `feat/hf-range-safetensors` (PR a)  
**Author**: Amandeep-Bajwa (`amnibajwa2006@gmail.com`)  
**Status**: In Progress  

---

## 1. Executive Summary & Motivation

Before downloading gigabytes of model weights over the internet, a security practitioner or developer must evaluate whether the model is safe to pull and run locally. Traditional security workflows require downloading entire multi-gigabyte checkpoints before inspecting headers, embedded metadata, chat templates, or code-execution triggers.

`safetensors` files store an 8-byte little-endian unsigned 64-bit integer at the beginning of the file specifying the header byte-length, followed immediately by a UTF-8 JSON header containing tensor metadata and optional `__metadata__`. By utilizing HTTP Range Requests (`Range: bytes=0-7` followed by `Range: bytes=8-{7 + header_len}`), Aegis-Tensor can inspect safetensors metadata, check for hidden steganographic payloads or leaked tokens, verify structural alignment, and flag supply-chain threats without downloading tensor payload weights.

This document describes the engineering design for **PR (a)** of Issue #35: implementing the standard-library HTTP range-fetch engine, repository revision resolution, safe safetensors header inspection, pickle refusal, and `trust_remote_code` detection.

---

## 2. Security Invariants & Threat Model

Every design decision adheres strictly to the invariants defined in `CLAUDE.md` and `docs/DECISIONS.md`:

1. **Attacker-Controlled Header Length Capping (100 MB Hard Limit)**:
   - *Threat*: The first 8 bytes of a safetensors file are controlled by the remote uploader. An attacker could craft a file where the header length is set to $2^{40}$ or $2^{63}-1$, triggering an out-of-memory denial-of-service (DoS) or astronomical bandwidth consumption.
   - *Mitigation*: The safetensors specification caps headers at 100 MB (`104,857,600` bytes). Aegis inspects the 8-byte prefix and **immediately refuses** any header specifying a length $> 100\text{ MB}$ or $< 0$, creating a finding without issuing a second HTTP request.
2. **Never Download or Load Pickle Files**:
   - *Threat*: PyTorch `.bin`, `.pt`, `.pth`, `.pkl`, and `.ckpt` files rely on Python `pickle`, which can execute arbitrary bytecode upon deserialization.
   - *Mitigation*: Remote pickle files cannot be safely inspected via simple byte-range headers because pickle streams and zip directories require deep container traversal. Aegis refuses to download pickle files remotely and immediately emits a finding informing the user that pickle formats require a full download to inspect and that the safetensors format should be preferred.
3. **Detection of `trust_remote_code` Requirements**:
   - *Threat*: Hugging Face models containing custom architectures can execute arbitrary remote Python code during model instantiation if the user passes `trust_remote_code=True`.
   - *Mitigation*: When resolving a repository, Aegis inspects the repository file list and `config.json`. If any Python files (`.py`) exist or if `config.json` contains an `auto_map` dictionary, Aegis records a finding that the model requires `trust_remote_code` (loading it runs repository code).
4. **Masking of Leaked Secrets**:
   - *Threat*: Remote uploaders may accidentally leave API tokens, Hugging Face user access tokens, or private keys in the `__metadata__` block of a model header.
   - *Mitigation*: All metadata values are scanned using `aegis.secrets.find_secrets`. Any discovered secrets are masked (e.g. `hf_...1234`) in the finding message so credentials are never logged or exposed in plaintext.
5. **Zero External Runtime Dependencies**:
   - *Design Choice*: All HTTP network operations use Python's built-in `urllib.request` standard library. No third-party network libraries (e.g., `requests`, `aiohttp`, `httpx`) are added to runtime dependencies.
6. **Network-Isolated Testing via Deterministic Monkeypatching**:
   - *Design Choice*: All network calls in `aegis/hub.py` funnel through a single entry point: `_http_get(url: str, headers: dict) -> tuple[int, bytes]`. Standard test suites monkeypatch `_http_get` with in-memory responses and never touch the live internet. Real-network tests are marked with `@pytest.mark.network` and excluded by default (`-m "not network"`).

---

## 3. Architecture & Component Design

```
+---------------------------------------------------------------------------------+
|                                 aegis/hub.py                                    |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   |   _http_get(url: str, headers: dict[str, str]) -> tuple[int, bytes]      |   |
|   +-------------------------------------------------------------------------+   |
|                                    ^                                            |
|                                    |                                            |
|          +-------------------------+-------------------------+                  |
|          |                                                   |                  |
|   +------+----------------------------+   +------------------+--------------+   |
|   | resolve(repo, revision)           |   | fetch_safetensors_header(...)   |   |
|   |  - API call /revision/{rev}       |   |  - Step 1: Range bytes=0-7      |   |
|   |  - Resolves commit SHA            |   |  - Step 2: Validate <= 100 MB   |   |
|   |  - Checks .py and auto_map        |   |  - Step 3: Range bytes=8-{7+len}|   |
|   |    for trust_remote_code          |   |  - Step 4: Scan header bytes    |   |
|   +-----------------------------------+   +------------------+--------------+   |
+--------------------------------------------------------------|------------------+
                                                               v
                                   +------------------------------------------+
                                   |         aegis/formatscan.py              |
                                   |                                          |
                                   |  scan_safetensors_header_bytes(          |
                                   |      raw: bytes,                         |
                                   |      header_len: int,                    |
                                   |      label: str                          |
                                   |  ) -> FormatScanResult                   |
                                   +------------------------------------------+
```

### 3.1. `aegis/hub.py`

* `_http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]`:
  Uses `urllib.request.Request` with configured headers (including `Range`). Traps `urllib.error.HTTPError` and returns `(e.code, e.read())`.
* `resolve(repo: str, revision: str = "main") -> HubRepoInfo`:
  Queries `https://huggingface.co/api/models/{repo}/revision/{revision}`.
  Extracts `sha` and `files` (`[s["rfilename"] for s in siblings]`).
  If any file ends with `.py` or if `config.json` contains `auto_map`, appends a `HubFinding` warning of `trust_remote_code`.
* `fetch_safetensors_header(repo: str, sha: str, filename: str) -> FormatScanResult`:
  1. Requests `Range: bytes=0-7` from `https://huggingface.co/{repo}/resolve/{sha}/{filename}`.
  2. Unpacks `<Q` uint64 header length.
  3. If length $> 100\text{ MB}$ (`104,857,600` bytes) or $< 0$, returns a `FormatScanResult` with an `oversized_header` finding and aborts without issuing a second request.
  4. Requests `Range: bytes=8-{7 + header_len}`.
  5. Passes raw bytes to `scan_safetensors_header_bytes`.
* `scan_remote_file(repo: str, sha: str, filename: str) -> FormatScanResult`:
  Inspects the file type based on extension:
  - If safetensors: delegates to `fetch_safetensors_header`.
  - If pickle (`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`): produces `pickle format: full download required to scan; prefer the safetensors version` with zero network requests.

### 3.2. Refactoring `aegis/formatscan.py`

To ensure zero code duplication between local and remote scanning:
1. Extract core parsing and validation logic from `scan_safetensors_header` into:
   `scan_safetensors_header_bytes(raw: bytes, header_len: int, label: str = "") -> FormatScanResult`
2. `scan_safetensors_header(path: str | Path)` reads the initial 8 bytes, extracts `header_len`, reads `raw`, and forwards directly to `scan_safetensors_header_bytes(raw, header_len, str(path))`.
3. All existing local format tests continue passing with zero behavioral divergence.

---

## 4. Test Suite Strategy (`tests/test_hub.py`)

In accordance with test-driven development:
1. `test_resolve_pins_revision_to_commit_sha`:
   Mocks API response with `sha="abcd1234ef..."`; asserts resolved SHA is reported.
2. `test_safetensors_header_is_fetched_with_two_range_requests`:
   Asserts exactly two calls to `_http_get`: first with `Range: bytes=0-7`, second with `Range: bytes=8-{7+header_len}`.
3. `test_oversized_header_length_is_refused_without_fetching`:
   Prefix specifies $2^{40}$; asserts single call to `_http_get` and finding generated.
4. `test_metadata_payload_found_remotely_matches_local_scan`:
   Creates a safetensors file locally with a metadata payload, mocks remote range fetches with identical bytes, and verifies findings match exactly.
5. `test_leaked_token_in_remote_metadata_is_masked`:
   Remote metadata contains active API token (`hf_...`); asserts finding generated and plaintext token is masked.
6. `test_pickle_file_is_reported_not_downloaded`:
   Passes `pytorch_model.bin`; asserts zero range requests and finding emitted.
7. `test_trust_remote_code_repo_gets_a_finding`:
   Repo containing `modeling_custom.py` or `config.json` with `auto_map`; asserts finding emitted.
8. `test_optional_real_network_fetch`:
   Marked `@pytest.mark.network`, skipped by default.

---

## 5. Implementation Roadmap

1. Write `docs/ISSUE_35_ENGINEERING_DESIGN.md` (this file).
2. Configure `tests/conftest.py` and `pyproject.toml` to register `network` marker and exclude it by default.
3. Write `tests/test_hub.py` with the 7 mandatory tests + 1 network test. Run pytest to watch them fail (TDD).
4. Refactor `aegis/formatscan.py` to expose `scan_safetensors_header_bytes`.
5. Implement `aegis/hub.py`.
6. Run pytest, ruff, mypy, and `python scripts/claims.py --write`.
7. Verify all CI gates pass and PR changed lines remain well under 400 lines.
