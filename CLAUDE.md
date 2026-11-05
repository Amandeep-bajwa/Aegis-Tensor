# CLAUDE.md

Instructions for AI assistants (and people) working in this repository. Read this,
[docs/DECISIONS.md](docs/DECISIONS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything.

Aegis answers one question: **is this model file safe to load and run?** It reads model files
without loading them and gives one verdict with a next step.

## Build, test, lint

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release                   # builds the Rust core (src/lib.rs -> aegis.aegis_core)
pip install -e ".[dev,fuzzer,engines]"      # drop `fuzzer` to work without PyTorch

pytest tests/                               # torch-only tests skip without torch
AEGIS_REQUIRE_CLAIMS=1 pytest tests/        # as CI runs it: a skipped claim test fails
pytest tests/ --cov=aegis                   # CI enforces a coverage floor and 90% on changed lines

cargo fmt --all --check
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features

ruff check . && ruff format --check . && mypy aegis
python scripts/claims.py                    # claims register vs README vs evidence
```

After changing `src/lib.rs`, run `maturin develop --release` again before the Python tests.

## Security invariants (never break these)

- Never unpickle, `torch.load` without `weights_only=True`, render a template, or import code from a
  scanned file in-process. Unpickling happens only in the sandboxed subprocess behind `--trust-pickle`
  (`aegis/sandbox.py`, `aegis/sandbox_worker.py`). Jinja templates are parsed, never rendered.
- Test fixtures that model an attack are benign: at most they create a marker file in `tmp_path`, and
  the test asserts the marker never appears.
- Tests make no network calls.
- Output never calls a file "safe". The best verdict is NO KNOWN FINDINGS.
- `aegis disarm` never modifies its input; it writes a new file and refuses a destination equal to
  the source.

## Engineering rules

- Plan before any change that touches more than one file. One concern per PR. There is no limit on
  PR size: a PR is judged on quality, and CONTRIBUTING.md lists exactly what gets one accepted.
- Write the test first and watch it fail. Conventional Commit PR titles.
- Never weaken, skip or delete a test, and never loosen a detection threshold, to make CI pass.
  Threshold changes need calibration evidence (`benchmarks/calibrate.py`, docs/calibration.md) and
  the lead's approval.
- A known gap is a strict `xfail` test plus a row in docs/THREAT_MATRIX.md, never a silent skip.
- Every README claim lives in `docs/claims.json` with a test marked `@pytest.mark.claim("<id>")`.
  Changing the rules or disarm code changes the fingerprints in `benchmarks/fingerprint.py`; the
  benchmark behind a claim must then be re-run (`python scripts/claims.py --write` afterwards).
- No new runtime dependency without approval in the issue. Update CHANGELOG.md under Unreleased.
