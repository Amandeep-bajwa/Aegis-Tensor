# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Every PR adds a line under Unreleased.

## [Unreleased]

### Changed
- float16 tensors use their own exponent-lane threshold, 7.2 (`--exponent-threshold-f16`); float32
  and bfloat16 keep 6.5. This clears the 37 float16 attention-bias false positives found in
  calibration (docs/calibration.md, P1). The low-plane floor stays at 7.5, and the one DETR tensor
  below it is recorded as an accepted false positive (P2).

### Added
- docs/KNOWN_BYPASSES.md: every known bypass and limit, with its evidence and where a fix is tracked.
- Tests of the disarm guarantee at the default width for float32, float16 and bfloat16, in
  safetensors and GGUF files: the rounded-away bits hold nothing afterwards.
- Quality gates in CI: ruff, mypy, cargo fmt and clippy, a coverage floor, diff coverage on PRs,
  a wheel smoke test, a no-PyTorch test job, Conventional Commit PR titles and a check that code
  changes come with test changes.
- Negative tests in `tests/test_claims.py` for each check in `scripts/claims.py`: out-of-date
  README claims, claims without tests, unregistered markers, duplicate IDs and execution without
  optional dependencies (#37).
- Full claims and benchmark register validation in the CI `lint` job (#37).
- Remote Hugging Face model inspection via standard-library HTTP range requests (`aegis.hub`): inspect
  safetensors headers and metadata up to a 100 MB ceiling, pin revisions to commit SHAs, refuse pickle
  downloads, and flag repositories requiring `trust_remote_code`.

### Changed
- `benchmarks/disarm_impact.py` imports `torch` and `transformers` inside functions rather than at
  module level, so `scripts/claims.py` runs without PyTorch (#37).

### Security
- The pickle scanner no longer trusts every `torch.*` and `numpy.*` global. A pickle calling
  `torch.hub.load`, `torch.utils.cpp_extension.load_inline`, `torch.jit`, `torch.ops.load_library`,
  a nested `torch.load`, or a `numpy.testing` helper that runs code used to get no finding; these
  are now critical. Only the globals that `torch.save` and numpy files use are allowed, and any
  other `torch`/`numpy` global is reported as unrecognised.
- The pickle scanner tracks calls and their results. Calling a callable that another call built at
  runtime (for example `operator.attrgetter("system")(obj)(...)`), even through the memo, is
  critical (rule AEGIS-P013). Pickles embedded in PyTorch storages (`torch.storage._load_from_bytes`)
  are scanned too, up to a nesting limit.
- Legacy tar checkpoints: every pickle in the `storages` and `tensors` members is scanned, not only
  the first. Members are read record by record as PyTorch reads them, so raw tensor bytes are never
  parsed as opcodes; a member that cannot be read that way is reported and still scanned.
- `aegis disarm` writes its output under a temporary name and renames it only after every tensor
  was treated. Before, a failure part-way through left a file under the "disarmed" name; for GGUF
  it was a full copy in which some tensors were never disarmed, and a failure could also replace an
  earlier good output.

### Fixed
- `aegis fuzz` refuses an unsupported file before asking for PyTorch.
- `fuzz_model` raises a clear error when given neither an input shape nor a vocabulary size.

## [0.1.0] - unreleased

> **Note.** 0.1.0 was never tagged. This section records what was on `main` before the v0.2.0
> sprint started (the first commit is 2026-09-08).

### Added
- Static scanner for model checkpoint files supporting safetensors, GGUF, PyTorch pickle (.pt, .pth,
  .bin, .pkl), Keras HDF5 (.h5), and ONNX.
- Rust core engine with PyO3 bindings for tensor mantissa bit analysis, chunk entropy scanning, and
  Benford's Law distribution analysis.
- GGUF scanner verifying tensor metadata, quantized block alignment, and Jinja chat template AST
  security analysis.
- Disarm treatment (`aegis disarm`) rounding away low-order mantissa bits in safetensors and GGUF
  files to neutralize hidden steganographic payloads without PyTorch.
- Dynamic fuzzer (`aegis fuzz`) comparing activations against clean baselines at matched input
  magnitudes with sandboxed loading.
- Interactive terminal dashboard and report export.
- Standardized findings schema supporting JSON output and SARIF 2.1.0 reports for CI/CD integration.
- Claims register (`scripts/claims.py` and `docs/claims.json`) mechanically linking README claims
  to automated tests and benchmarks.
- Real-world calibration harness on popular Hugging Face models measuring per-rule false-positive
  rates.
- CLI exit-code contract: 0 clean, 1 findings above threshold, 2 unreadable file or usage error.
- Integration with optional scanning engines (Fickling and ModelScan, via `aegis/engines.py`)
  folded into the final verdict.
- Single-line verdict with a recommended next step printed after every scan (`aegis scan`).
- Leaked-credential detection in model metadata (API keys, tokens, private keys).
- `aegis report-bundle`: writes a folder for review (REPORT.md, findings.json, manifest.json with hashes and source); never sends anything and never copies the model file.
- `aegis doctor`: check the runtime environment and report which optional capabilities are available.

### Security
- Static pickle opcode scanner simulating Python pickle stack and memo, detecting dangerous
  globals, and resyncing past malformed bytes without executing code.
- Clamping of disarm low-bit widths to per-dtype safe maximums (float32 16 bits, float16 4 bits,
  bfloat16 2 bits) and strict refusal of exponent bits.
- Automated sandbox enforcement on Linux (`unshare -rn`) and macOS (`sandbox-exec`) for untrusted code execution during fuzzing.
