# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Every PR adds a line under Unreleased. The history before this file existed is being
backfilled (issue A3).

## [Unreleased]

### Added
- Quality gates in CI: ruff, mypy, cargo fmt and clippy, a coverage floor, diff coverage on PRs,
  a wheel smoke test, a no-PyTorch test job, Conventional Commit PR titles and a check that code
  changes come with test changes.

### Security
- The pickle scanner no longer trusts every `torch.*` and `numpy.*` global. A pickle calling
  `torch.hub.load`, `torch.utils.cpp_extension.load_inline`, `torch.jit`, `torch.ops.load_library`,
  a nested `torch.load`, or a `numpy.testing` helper that runs code used to get no finding; these
  are now critical. Only the globals that `torch.save` and numpy files use are allowed, and any
  other `torch`/`numpy` global is reported as unrecognised.

### Fixed
- `aegis fuzz` refuses an unsupported file before asking for PyTorch.
- `fuzz_model` raises a clear error when given neither an input shape nor a vocabulary size.
