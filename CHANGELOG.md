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

### Fixed
- `aegis fuzz` refuses an unsupported file before asking for PyTorch.
- `fuzz_model` raises a clear error when given neither an input shape nor a vocabulary size.
