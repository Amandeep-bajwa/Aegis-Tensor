# Contributing to Aegis-Tensor

This page tells you exactly what gets a pull request (PR) accepted. If you follow it, the PR
will pass review the first time or need only small changes. If something here is unclear, ask
in your issue before you start.

Read these first, in this order:

1. [CLAUDE.md](CLAUDE.md): the build commands, the security rules and the engineering rules.
2. [docs/DECISIONS.md](docs/DECISIONS.md): why Aegis works the way it does. To change a
   decision, open an issue that proposes the change and its evidence.
3. Your issue. It says what to build, which tests to write first and when you are done.

## What a PR is judged on

**Quality, not size.** There is no limit on how many lines a PR changes. Write as much code
and as many tests as the change needs to be correct. A reviewer asks these questions, roughly
in this order:

1. **Does it do what the issue says?** Every item under "Done when" is true, and you can show it.
2. **Is it correct on every path a user can reach?** This includes the error paths: a network
   failure, a malformed file, a timeout, an empty input. The right result is a finding or exit
   code 2 with a clear message, never a traceback.
3. **Do the tests prove it?** Each new behaviour has a test, and each test fails if the code it
   covers is removed (see "Tests" below).
4. **Is it true?** Every statement in the PR description, the README, the CHANGELOG and the
   docs matches what the code does. Check each one against the code before you write it.
5. **Is it only what was asked for?** Unrelated edits go in their own PR.
6. **Does it fit the project?** It reuses the existing types and helpers instead of making
   parallel ones, and it follows the security rules.

Size still matters in one way: a reviewer must be able to understand the PR. Keep one concern
per PR, explain it well in the description, and keep unrelated changes out.

## Set up

```bash
git clone https://github.com/KrishG7/Aegis-Tensor.git && cd Aegis-Tensor
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release                   # builds the Rust core
pip install -e ".[dev,fuzzer,engines]"      # drop `fuzzer` to work without PyTorch
```

After you change `src/lib.rs`, run `maturin develop --release` again before the Python tests.
`pip install -e ".[dev]"` installs the same pinned `ruff` and `mypy` that CI uses, so local
results match CI.

## The workflow

1. **Pull the latest `main`** and create a branch from it: `git switch main && git pull &&
   git switch -c <type>/<short-name>`. Always branch from `main`, never from another
   contributor's branch (see "Stacked work" below).
2. **Write the tests first.** Write the tests your issue lists, run them, and watch them fail
   for the right reason. Then write the code until they pass.
3. **Run every check locally** before you push (see "Run these before you push").
4. **Open the PR into `main`.** Fill in every section of the PR template. A PR that skips a
   section is sent back.
5. **Respond to review.** Push new commits to the same branch (the PR is squashed when it
   merges, so extra commits are fine). Reply to every comment, and resolve a conversation only
   when you have fixed it or agreed on the answer. Then ping the reviewer.
6. **If `main` moves**, click **Update branch** on the PR and wait for CI to run again.
7. **A maintainer squash-merges it** once it is approved and green. The branch is deleted
   automatically.

The lead maintainer aims to review within 24 hours.

## Run these before you push

```bash
pytest tests/                                # torch-only tests skip without torch
AEGIS_REQUIRE_CLAIMS=1 pytest tests/         # as CI runs it: a skipped claim test fails
pytest tests/ --cov=aegis                    # check the coverage of the lines you changed

ruff check . && ruff format --check . && mypy aegis
python scripts/claims.py                     # claims register vs README vs evidence

cargo fmt --all --check                      # only when you changed Rust
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features
```

`ruff format .` fixes formatting for you. If a command fails on `main` before you changed
anything, say so in the PR instead of working around it.

## What must be true before a PR can merge

GitHub blocks the merge button until all of these are true.

**Nine CI checks pass.** Each one runs on every PR into `main`:

| Check | What it runs | The usual fix when it fails |
|---|---|---|
| `rust` | `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` | Run those three commands and fix what they print. |
| `lint` | `ruff check`, `ruff format --check`, `mypy aegis` | `ruff check --fix .` and `ruff format .`; add type annotations where mypy asks. |
| `test (py3.10)` | The full test suite on Python 3.10 | Reproduce with `pytest tests/` on Python 3.10. |
| `test (py3.12)` | The full suite with a coverage floor of 88%, and **diff coverage of at least 90%** on the lines your PR adds or changes | Add tests for the uncovered lines (the log lists them). Never lower the floor. |
| `test (no torch)` | The suite without PyTorch. Torch tests skip; nothing may fail at collection | Skip torch-only tests when torch is missing (see `needs_torch` in `tests/test_disarm.py`) and import torch inside functions, not at module level. |
| `macos` | The suite on macOS | Look for Linux-only assumptions (paths, sandboxing, shell commands). |
| `wheel` | Builds the wheel, installs it in a fresh environment and runs `aegis` from outside the repo | Check that new files are packaged and that nothing relies on the repo checkout. |
| `pr-title` | The PR title is a Conventional Commit | See "PR title" below. |
| `tests-touched` | A PR that changes `aegis/` or `src/` must also change something under `tests/` | Add a test. Only KrishG7 can waive this, with the `no-test-needed` label. |

**The branch is up to date with `main`.** Click **Update branch** if GitHub says it is not.

**A code owner approves.** The lead maintainer approves every PR (`.github/CODEOWNERS`);
changes to `aegis/fuzzer.py` also need Daksh. If you push new commits after an approval, the
approval is dismissed and you must ask again.

**Every review conversation is resolved.**

CI being green is necessary, not sufficient. The reviewer reads the change, and everything in
"What a PR is judged on" applies.

## PR title

The title becomes the commit message on `main`, so write it carefully. Format:
`type(scope): what changed`, for example `feat(hub): scan safetensors headers before download`.
Use one of `feat`, `fix`, `test`, `docs`, `ci`, `chore`, `refactor`, `perf` or `style`.

## PR description

Use the template and be factual:

- **Part of #N** if the issue needs more than one PR, **Closes #N** only on the PR that
  finishes it. A `Closes` line closes the issue when the PR merges, so do not write it on a PR
  that leaves work undone.
- **What this changes:** what was wrong or missing, and what is different now.
- **How I tested it:** the exact commands you ran and their results. Name the new tests and
  confirm that each one failed before your change.
- **Evidence for anything that needs a run.** If the issue asks for a run on your fork or a
  benchmark result, link a run that actually succeeded, and check that it did (open the link,
  read every job). A cancelled or partly green run is not evidence.
- **Fresh-session AI review:** if you used an AI assistant to write the code, review the diff in
  a new session that did not write it, and summarise what it found and what you did. You are
  responsible for every line in the PR, whoever or whatever wrote it.

## Tests

- **Write them first and watch them fail.** A test that never failed proves nothing.
- **A test must be able to fail.** Before you push, break the code the test covers (remove the
  check, return early) and run the test. If it still passes, it is too weak. Typical weak tests:
  `assert "a" in x or "b" in x or "c" in x` (passes if any one is present), a value that is read
  but never asserted, and a test that only checks the exit code.
- **Cover every path a user can reach,** not only the one you used while developing. If a
  command has text, JSON and SARIF output, or handles several file kinds, test each combination
  you changed. Test the error paths too (network error, bad input, empty input).
- **Never weaken, skip or delete a test, and never loosen a detection threshold, to make CI
  pass.** If something cannot pass properly, stop and explain it in the PR. Threshold changes
  need calibration evidence (`benchmarks/calibrate.py`, `docs/calibration.md`) and the lead's
  approval.
- **A known gap is a strict `xfail` test plus a row in `docs/THREAT_MATRIX.md`,** never a silent
  skip.
- **No network.** Tests never touch the network. Replace network access with a fake (for
  example, route all network access through one function and replace that function in the test).
- **Benign fixtures only.** A fixture that models an attack may at most create a marker file in
  `tmp_path`, and the test asserts that the marker never appears.
- **Do not change production code to suit a test.** If the code needs a fallback only because a
  test fake is too limited (for example catching `TypeError` to cope with a fake that lacks an
  argument), fix the fake.

## Claims

Every statement the README makes about what Aegis does is a *claim*, and every claim is backed
by a test.

1. Add an entry to `docs/claims.json` with the next free ID (`C22`, `C23`, ...) and a plain
   statement. Known limits use IDs like `L03` and `"kind": "limit"`.
2. Mark at least one test with `@pytest.mark.claim("C22")`.
3. Run `python scripts/claims.py --write`. It regenerates the README claims table.
4. `python scripts/claims.py` must report no problems. In CI, a skipped claim test is a failure.

If your PR adds behaviour that users will rely on, add the claims for it in the same PR. Do not
change the wording of existing claims unless your issue says so, and use no marketing words
("enterprise", "very-high precision", "military-grade", "zero-download").

**The rules fingerprint.** A change to `src/lib.rs`, `aegis/formatscan.py`,
`aegis/modelformats.py`, `aegis/templatescan.py` or `aegis/thresholds.py` changes the rules
fingerprint (`benchmarks/fingerprint.py`), so the calibration behind claim C10 goes stale. In
that PR, run `python scripts/claims.py --write`. It marks C10 pending, which is honest. A change
to `aegis/disarm.py` does the same to claim C11 and needs `benchmarks/disarm_impact.py` re-run.

## Security rules

These come from [CLAUDE.md](CLAUDE.md); breaking one always means a rejected PR.

- Never unpickle, `torch.load` without `weights_only=True`, render a template or import code
  from a scanned file in-process. Unpickling happens only in the sandboxed subprocess behind
  `--trust-pickle`. Jinja templates are parsed, never rendered.
- Output never calls a file "safe". The best verdict is NO KNOWN FINDINGS.
- `aegis disarm` never modifies its input.
- No new runtime dependency unless your issue approves it.
- Findings use the project's own types in `aegis/findings.py`. A new kind of finding gets its
  own rule there (with a stable ID, severity and treatment). Do not reuse a rule whose meaning
  does not match, because the rule decides what users see in JSON and SARIF.

## Keeping the repository consistent

- Update `CHANGELOG.md` under *Unreleased*, in plain words, and check each line against the
  code. Describe what the code does, not what you hope it does.
- Update `docs/THREAT_MATRIX.md` when you change what Aegis detects or misses.
- Update the README and docs when behaviour changes. Do not describe a feature that does not
  exist yet.
- Do not edit files unrelated to your issue. If you find a separate problem, open an issue.

## Stacked work (a PR that depends on another PR)

CI and branch protection only apply to PRs into `main`. A PR that targets another contributor's
branch gets no CI at all, so it can look "clean" without being checked. Therefore:

- Open every PR into `main`.
- If PR (b) needs PR (a), wait until (a) is merged, then update your branch from `main` and
  open (b). While you wait, keep (b) as a draft or work locally.
- Put `Part of #N` on (a) and `Closes #N` on (b) only.

## Reporting a security problem

If you find a way to get a malicious model past Aegis, do not open a public issue or PR. Use
**Security → Report a vulnerability** on the repository ([SECURITY.md](SECURITY.md)). Test files
for it must be benign, and the report should say which file kind and which rule it slips past.

## Checklist before you ask for review

Copy this into your PR and tick it honestly.

- [ ] I read CLAUDE.md, docs/DECISIONS.md and my issue, and every "Done when" item is true.
- [ ] My branch comes from the latest `main`, and the PR targets `main`.
- [ ] The title is a Conventional Commit; the body uses `Part of #N` or `Closes #N` correctly.
- [ ] I wrote the tests first and saw them fail. I broke the code on purpose and saw them fail.
- [ ] Error paths and every output format or file kind I touched are tested.
- [ ] `pytest`, `ruff check`, `ruff format --check`, `mypy aegis` and `python scripts/claims.py`
      pass locally (and the Rust checks, if I changed Rust).
- [ ] New and changed lines are at least 90% covered.
- [ ] No test was weakened, skipped or deleted, and no threshold was loosened.
- [ ] No production code exists only to make a test pass.
- [ ] Claims for the new behaviour are in `docs/claims.json`, each with a marked test.
- [ ] CHANGELOG, docs and README are updated, and every sentence in them is true.
- [ ] The PR contains only what the issue asks for.
- [ ] Any run or result the issue asks for is linked, and I checked that it succeeded.
- [ ] I reviewed the diff in a fresh AI session (if I used AI) and addressed what it found.
