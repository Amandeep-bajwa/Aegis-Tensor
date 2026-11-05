<!--
Title: a Conventional Commit, for example `fix(picklescan): resolve callables built by REDUCE`.
Read CLAUDE.md, CONTRIBUTING.md and your issue before opening this PR. PRs that skip a section are sent back.
-->

Closes #

## What this changes

<!-- Plain English, two to five sentences: what was wrong or missing, and what is different now. -->

## How I tested it

<!-- The exact commands you ran and their result counts, for example:
pytest tests/                        -> 212 passed, 1 skipped, 1 xfailed
cargo test --no-default-features     -> 21 passed
ruff check . && ruff format --check . && mypy aegis -> clean
Name the new tests, and confirm each one failed before your change. -->

```text

```

## Fresh-session AI review

<!-- Open a new AI session (Claude Code, Gemini, ...) with no memory of writing this code, point it at
CLAUDE.md, the issue and this diff, and ask it to find bugs, missing tests and broken rules.
Summarise what it found and what you did about each point. -->

## Checklist

- [ ] The title is a Conventional Commit (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`, `perf:`, `style:`).
- [ ] This PR covers one issue only, and every change in it is needed for that issue.
- [ ] I wrote the tests first, and they fail without the change.
- [ ] No test was weakened, skipped or deleted, and no detection threshold was loosened.
- [ ] Nothing from a scanned file is loaded, unpickled, rendered or imported in-process.
- [ ] Tests make no network calls, and every malicious fixture is benign (at most it creates a marker file).
- [ ] No new runtime dependency, unless the issue approves it.
- [ ] Output never calls a file "safe".
- [ ] `docs/claims.json`, `docs/THREAT_MATRIX.md`, `README.md` and `CHANGELOG.md` are updated where this changes what Aegis claims or does.
- [ ] The branch is rebased on the latest `main`.
