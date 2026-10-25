# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Every PR adds a line under Unreleased. The history before this file existed is being
backfilled (issue A3).

## [Unreleased]

### Added
- docs/KNOWN_BYPASSES.md: every known bypass and limit, with its evidence and where a fix is tracked.
- Tests of the disarm guarantee at the default width for float32, float16 and bfloat16, in
  safetensors and GGUF files: the rounded-away bits hold nothing afterwards.
- Quality gates in CI: ruff, mypy, cargo fmt and clippy, a coverage floor, diff coverage on PRs,
  a wheel smoke test, a no-PyTorch test job, Conventional Commit PR titles and a check that code
  changes come with test changes.
- Automated release workflow (.github/workflows/release.yml) building and smoke-testing abi3 wheels
  for Linux x86_64, macOS arm64, macOS x86_64 and sdist on v* tags (#36).

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
