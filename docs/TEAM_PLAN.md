# v0.2.0 Sprint Plan

Sprint: Saturday 24 October to Monday 9 November 2026, 23:59 IST. Tag `v0.2.0` on 10 November.
Milestone: **v0.2.0**. Every issue below is created from this file, so this file is the source
of truth; if an issue and this file disagree, fix the file and the issue together.

Every due date is end of day, IST.

## Team

| Who | GitHub | Role this sprint | Reviews |
|---|---|---|---|
| Krish | @KrishG7 | Lead. Security-critical work, the release, and every review | Every PR |
| Daksh | @7dxksh7 | The heaviest work: compute, large downloads, long runs | Krish's PRs and anything touching `aegis/fuzzer.py` |
| Amandeep | @Amandeep-bajwa | Medium, tightly specified tasks (CI, integration) | — |
| Himanshu | @HimanshuChhillar | Small tasks that the release cannot ship without, including Rust he knows | — |

Mukul (@Mukul09800) owns issue #10 (docs and presentation), which is outside this sprint.

## Rules

1. **Read [CLAUDE.md](../CLAUDE.md), [CONTRIBUTING.md](../CONTRIBUTING.md),
   [docs/DECISIONS.md](DECISIONS.md) and your issue before writing code.** CONTRIBUTING.md lists
   exactly what gets a PR accepted.
2. **Wait for the quality-gates PR (K0) to merge before opening a PR.** You can read code and
   write tests before then. After K0, `main` is protected: CI must be green, a code owner must
   approve, and every review conversation must be resolved.
3. **One concern per PR. There is no limit on PR size; a PR is judged on quality.** Conventional
   Commit titles, for example `feat(hub): scan safetensors headers before download`.
4. **Tests first.** Write the tests in your issue, watch them fail, then make them pass. Diff
   coverage on a PR must be at least 90%.
5. **Never weaken, skip or delete a test, and never loosen a detection threshold, to make CI
   pass.** If something cannot pass properly, stop and say so in the PR.
6. **Every claim needs a test.** A README claim lives in `docs/claims.json` with a test marked
   `@pytest.mark.claim("<id>")`. No marketing words ("enterprise", "very-high precision",
   "military-grade") in code or docs.
7. **Benign fixtures, no network.** A malicious fixture may at most create a marker file in
   `tmp_path`. Tests never touch the network.
8. **The scan fingerprint.** A change to `src/lib.rs`, `aegis/formatscan.py`,
   `aegis/modelformats.py`, `aegis/templatescan.py` or `aegis/thresholds.py` changes the rules
   fingerprint (`benchmarks/fingerprint.py`), so the calibration behind claim C10 goes stale. In
   that PR, run `python scripts/claims.py --write`: it marks C10 pending, which is honest. D4
   makes the final calibration run that claims C10 for the release. A change to
   `aegis/disarm.py` does the same to claim C11 and needs `benchmarks/disarm_impact.py` re-run.
9. **Review within 24 hours.** Krish reviews every PR and sends back any PR that skips the
   template.

## Timeline

| Date | Due |
|---|---|
| Sat 24 Oct | Quality-gates PR opened (K0) |
| Mon 26 Oct | K0: gates merged, `main` protected |
| Sat 31 Oct | D1 |
| Sun 1 Nov | K1 |
| Mon 2 Nov | H1 |
| Tue 3 Nov | A1 |
| Wed 4 Nov | H2 |
| Thu 5 Nov | D2 |
| Fri 6 Nov | D3, A2 |
| Sat 7 Nov | K2, A3 |
| Sun 8 Nov | D4, H3 |
| Mon 9 Nov | K3 (release ready); H3 re-run after the final merges |
| Tue 10 Nov | Tag `v0.2.0` |

## Dependencies

- **K0 blocks every PR.**
- **`src/lib.rs` merges in order: D1 → H1 → D2.** All three edit the tensor rules, so each one
  rebases on the one before.
- **D4 runs after D1, H1, D2 and A1(a) have merged.** Each of those changes the scan
  fingerprint, so D4's calibration run is the one that claims C10 for the release.
- **A2 blocks K3** (the release needs the wheels). **A3 blocks K3** (the CHANGELOG).
- **D3 feeds K3.** The release notes quote its results.
- **H3 runs twice**: on 8 Nov, and again on 9 Nov after the final merges.
- **K1 and K2 come before K3.** The release notes link docs/KNOWN_BYPASSES.md (K2).

## Issues

---

### [K0] Merge the quality gates and turn on branch protection (due 26 Oct)

**Owner:** @KrishG7 · **Labels:** `owner:krish`, `priority:critical`, `area:ci`

#### What is this, simply?
Think of `main` as the school's front door. Right now anyone with a key can walk straight in.
The gates PR adds a guard who checks every visitor: tests pass, code is tidy, and a second
person has looked. This issue gets the guard installed and then locks the side doors.

#### Why it matters
Until `main` is protected, a broken or untested change can land directly and every teammate
then builds on it. Every other issue waits for this one.

#### Before you start
- Branch: `chore/quality-gates` (the PR already exists).
- Read: `.github/workflows/ci.yml`, `.github/workflows/pr-checks.yml`, `scripts/protect_main.sh`.
- Depends on: nothing.

#### Step by step
1. Ask @7dxksh7 to review the gates PR. Resolve every comment.
2. Squash-merge it once CI is green.
3. Run `bash scripts/protect_main.sh KrishG7/Aegis-Tensor`.
4. Check: `gh api repos/KrishG7/Aegis-Tensor/branches/main/protection` lists the nine required checks.
5. Prove it: `git push origin HEAD:main` from a scratch commit must be rejected. Delete the scratch commit.
6. Post "gates are up" in the team chat with a link to the v0.2.0 tracker.

#### Tests you must write first
None: this issue is configuration. The PR it merges adds `tests/test_check_tests_touched.py`.

#### Done when
- [ ] The gates PR is merged after an approving review from @7dxksh7.
- [ ] Branch protection lists the checks `rust`, `lint`, `test (py3.10)`, `test (py3.12)`, `test (no torch)`, `macos`, `wheel`, `pr-title`, `tests-touched`.
- [ ] A direct push to `main` is rejected (paste the error into this issue).
- [ ] The repo allows squash merges only and deletes branches on merge.

#### Don't
- Don't set `enforce_admins` to true until the release is out; it would block hotfixes during the sprint.
- Don't merge the gates PR with a red check.

#### Depends on / blocks
Blocks every other issue in this milestone.

---

### [K1] Resolve pickle callables built at runtime, and scan every pickle in legacy tar members (due 1 Nov)

**Owner:** @KrishG7 · **Labels:** `owner:krish`, `priority:critical`, `area:pickle`

**Status:** implemented in PRs #22, #23 and #25.

#### What is this, simply?
A pickle file is a list of instructions for rebuilding an object, like a recipe. Our scanner
reads the recipe without cooking it. Today it recognises "fetch `os.system` and call it". It
does not follow "make something with step 3, then call whatever step 3 made". That is like a
recipe whose dangerous ingredient is assembled out of two harmless ones. A second gap: in old
PyTorch tar files, the `storages` and `tensors` members are read only up to their first
pickle, so a payload hidden in a later pickle in the same member is never read.

#### Why it matters
Both are documented gaps (docs/THREAT_MATRIX.md, "Known gaps", item 3). An attacker who reads
our code can use either one to hide a code-execution payload in a file we pass.

#### Before you start
- Branch: `fix/pickle-runtime-callables`.
- Read: `aegis/picklescan.py`, especially `_step`, `_scan_pickle_stream` and `_scan_tar`;
  `tests/test_pickle_bypasses.py`.
- Depends on: K0.

#### Step by step
1. In `_step`, stop treating the results of `REDUCE`, `NEWOBJ`, `NEWOBJ_EX`, `OBJ`, `INST` and
   `BUILD` as plain `_UNKNOWN`. Push a symbolic "call result" object that records which global
   was called, so that a later `REDUCE` on that result can name the whole chain (for example
   `builtins.getattr(...) -> call`).
2. When `REDUCE` (or `NEWOBJ`) is applied to a callable that is not a recorded allowlisted
   global, report a CRITICAL finding with code `unresolved_callable` and a message naming the
   chain. Allowlisted torch rebuild helpers, which call their own results, must stay clean.
3. In `_scan_tar`, scan every pickle in the `storages` and `tensors` members. Today it passes
   `max_pickles=1`; replace that with a scan that walks the whole member. Raw storage bytes
   between pickles must not be reported as malformed pickles (keep `report_malformed=False` for
   those regions).
4. Add the new finding code to `aegis/findings.py` (`RULES`, with ATLAS and OWASP references),
   to `docs/THREAT_MATRIX.md` (new row; move gap 3 out of "Known gaps") and to `docs/claims.json`.

#### Tests you must write first
All payloads are benign: at most `touch <tmp_path>/MARKER`, and the `marker` fixture asserts it never appears.
- `test_callable_returned_by_reduce_and_then_called_is_critical`: a pickle that calls
  `builtins.getattr(builtins, "eval")` and then calls the result. Expect a CRITICAL
  `unresolved_callable` finding whose message names `getattr`.
- `test_reduce_chain_through_memo_is_resolved`: the same chain with the intermediate callable
  stored with `BINPUT` and fetched with `BINGET`. Expect the same finding.
- `test_clean_torch_checkpoint_still_has_no_findings`: `torch.save(nn.Linear(4, 2).state_dict())`
  in both zip and legacy formats. Expect no findings.
- `test_payload_in_second_pickle_of_legacy_storages_member_is_found`: a legacy tar checkpoint
  whose `storages` member holds a clean pickle, raw bytes, then a pickle importing `os.system`.
  Expect CRITICAL.
- `test_raw_storage_bytes_are_not_reported_as_malformed`: a clean legacy tar checkpoint with
  large storages. Expect no `malformed_pickle` finding.

#### Done when
- [ ] Every test above was seen failing before the change and passes after it.
- [ ] CI is fully green; diff coverage ≥ 90%.
- [ ] `docs/claims.json`, `docs/THREAT_MATRIX.md` and `CHANGELOG.md` are updated.
- [ ] The PR changes only what this issue asks for.

#### Don't
- Don't unpickle anything, even "just to compare".
- Don't add `getattr`, `apply` or similar to an allowlist to make a clean file pass; find out
  why the clean file calls it.

#### Depends on / blocks
Depends on K0. Blocks K3.

---

### [K2] Known bypasses page and disarm guarantee tests (due 7 Nov)

**Owner:** @KrishG7 · **Labels:** `owner:krish`, `priority:high`, `area:docs`

**Status:** implemented in PR #26.

#### What is this, simply?
A list of what Aegis does not catch, so users read "no known findings" correctly, plus tests that
`aegis disarm` clears every rounded-away bit at its default width.

#### Done when
- [ ] PR #26 merged: docs/KNOWN_BYPASSES.md, linked from docs/THREAT_MATRIX.md.
- [ ] The disarm guarantee tests pass in CI for float32, float16 and bfloat16, in safetensors and GGUF.
- [ ] CI fully green; CHANGELOG updated.

#### Depends on / blocks
Depends on K0 and K1. Blocks K3 (the release notes link docs/KNOWN_BYPASSES.md).

---

### [K3] Release v0.2.0 (due 9 Nov, tag on 10 Nov)

**Owner:** @KrishG7 · **Labels:** `owner:krish`, `priority:critical`, `area:ci`

#### What is this, simply?
Shipping day. Like closing the lid on a school project: everything is finished, checked,
labelled, and the label tells the truth, including what the project can't do.

#### Why it matters
A release is what users install. Its notes must match the evidence exactly.

#### Before you start
- Branch: `chore/release-0.2.0`.
- Read: `docs/claims.json`, `CHANGELOG.md`, `docs/KNOWN_BYPASSES.md`, `.github/workflows/release.yml` (A2).
- Depends on: every other issue in the milestone.

#### Step by step
1. Check that every issue in the milestone is closed, or moved out with a reason.
2. `python scripts/claims.py` passes with every claim claimed, including C10 (from D4) and C11.
3. Bump the version to `0.2.0` in `pyproject.toml`, `Cargo.toml` and `aegis/__init__.py`.
4. Finish `CHANGELOG.md`: move Unreleased to `[0.2.0] - 2026-11-10`.
5. Merge, then tag `v0.2.0` on 10 Nov. The release workflow (A2) builds and attaches the wheels.
6. Write the release notes: what is new, the claims table, and a plain list of what Aegis does
   not catch (known gaps, limits L01 and L02 if still open, and `docs/KNOWN_BYPASSES.md`).

#### Tests you must write first
None new. `tests/test_claims.py` must pass with no pending claim.

#### Done when
- [ ] Every claim in `docs/claims.json` is claimed and passing; none is pending.
- [ ] Wheels for Linux x86_64, macOS arm64 and macOS x86_64 plus an sdist are attached to the release.
- [ ] The release notes state what Aegis does not catch.
- [ ] H3's re-run on 9 Nov passed.

#### Don't
- Don't tag with a pending claim. If C10 can't be claimed, the release notes say so and the README shows it pending.

#### Depends on / blocks
Depends on every other issue in the milestone.

---

### [D1] Fix calibration false positives and claim C10 (due 31 Oct)

**Owner:** @7dxksh7 · **Labels:** `owner:daksh`, `priority:critical`, `heavy`, `area:tensor-rules`, `area:benchmarks`

#### What is this, simply?
A smoke alarm that goes off every time someone makes toast gets switched off, and then it is no
help in a real fire. On 54 clean public models, 4 set our alarm off. Three are float16 bias
vectors whose sign/exponent byte looks "too random" to the exponent-lane rule; one is a DETR
weight with oddly regular low bits. This issue makes the alarm stop reacting to toast without
making it deaf to smoke.

#### Why it matters
False positives teach users to ignore Aegis. Claim C10 (the false-positive rate) cannot be shown
in the README until this is fixed and re-measured.

#### Before you start
- Branch: `fix/f16-exponent-threshold`.
- Read: docs/calibration.md (section "Proposals", P1 and P2); `src/lib.rs` (`Thresholds`,
  `analyze_tensor`, `scan_safetensors`, `scan_tensor_bytes`); `aegis/thresholds.py`;
  `aegis/modelformats.py` (`_tensor_stats`); `benchmarks/calibrate.py`; `scripts/claims.py`.
- Depends on: K0. Merge before H1 and D2 (they edit the same Rust code).
- Note: the uncommitted 10-model results on Krish's machine are not a starting point. Run from
  the committed lock.

#### Step by step
1. From `benchmarks/results/calibration.json`, list the float16 exponent-lane values of clean
   tensors and check proposal P1's number. 7.2 is the proposal; pick the threshold from the data
   (largest clean value plus a stated margin), and show the calculation in the PR.
2. Add a float16-specific exponent threshold:
   - `aegis/thresholds.py`: a new `"exponent_f16"` entry;
   - `src/lib.rs`: a field in `Thresholds`, a new keyword argument on `scan_safetensors` and
     `scan_tensor_bytes`, and the dtype check in `analyze_tensor` (F16 uses it; F32 and BF16
     keep `exponent`);
   - `aegis/cli.py`: a `--exponent-threshold-f16` option next to `--exponent-threshold`;
   - `aegis/modelformats.py` (`_tensor_stats`) and `benchmarks/calibrate.py`: pass it through.
3. DETR (proposal P2): either fix the low-plane rule, with evidence that the detection margin
   holds, or keep the floor and record `facebook/detr-resnet-50` as an accepted false positive
   in docs/calibration.md with its measured value (5.94) and the reason.
4. Re-run calibration from the pinned lock: `python benchmarks/calibrate.py` (use `--evict` if
   disk is short). Then run `python scripts/claims.py --write`, which moves C10 from pending to
   claimed once the results match the current rules.
5. Update docs/calibration.md (the proposals section says what was adopted) and
   docs/THREAT_MATRIX.md "Known gaps", item 1.

#### Tests you must write first
- `test_f16_bias_vector_with_wide_exponents_is_not_flagged`: a float16 tensor built to the
  measured shape of the three false positives (exponent lane between 6.5 and the new
  threshold). Expect no finding.
- `test_f16_fully_overwritten_tensor_is_still_flagged`: a float16 tensor of random bytes. Expect
  `exponent_lane` (or `out_of_range`).
- `test_f16_one_percent_overwrite_is_still_flagged`: 1% of a float16 tensor overwritten with
  random bytes. Expect `out_of_range`. This proves the margin against partial overwrites did not
  shrink.
- `test_f32_and_bf16_still_use_the_general_exponent_threshold`: a float32 tensor at exponent
  lane 6.8 is still flagged.
- In Rust: a unit test that `analyze_tensor` picks the F16 threshold only for `Dtype::F16`.

#### Done when
- [ ] The threshold is derived from the measured distribution, with the numbers in the PR.
- [ ] Every test in `tests/test_threat_matrix.py` still passes unchanged.
- [ ] The DETR case is fixed or documented with evidence.
- [ ] `python scripts/claims.py` passes and C10 is claimed, not pending.
- [ ] CI is fully green; diff coverage ≥ 90%.
- [ ] docs/calibration.md, docs/THREAT_MATRIX.md, `docs/claims.json` and CHANGELOG.md updated.
- [ ] Krish approved the threshold change in review.
- [ ] The PR changes only what this issue asks for.

#### Don't
- Don't raise the general `exponent` threshold (6.5); only float16 gets its own value.
- Don't lower the low-plane floor to clear DETR without showing that base64-like payloads
  (about 6 bits per byte) are still caught.
- Don't hand-edit `benchmarks/results/calibration.json` or the values in `docs/claims.json`.

#### Depends on / blocks
Depends on K0. Blocks H1, D2 and D4 (the same Rust code), and K3 (C10).

---

### [D2] Format-aware rule for packed quantized tensors (due 5 Nov)

**Owner:** @7dxksh7 · **Labels:** `owner:daksh`, `priority:high`, `heavy`, `area:tensor-rules`, `area:benchmarks`

#### What is this, simply?
A quantized model squeezes several small numbers into one bigger one, like packing eight
4-bit numbers into each 32-bit box. Our rules look at the boxes and see a jumble, so the
only rule that applies is "is the whole tensor too random?", and real 4-bit models sit just
0.15 bits under that line. We need to unpack the boxes and look at the numbers inside.

#### Why it matters
Quantized models (GPTQ, AWQ, bitsandbytes 4-bit, FP8) are what most people download for local
use. A payload inside them is covered by one rule with almost no margin (docs/calibration.md, P3).

#### Before you start
- Branch: `feat/quantized-unpacking` (PR 1), then `feat/quantized-rules` (PR 2).
- Read: docs/calibration.md, P3; `src/lib.rs` (`analyze_tensor`, `parse_dtype`,
  `scan_safetensors`); `benchmarks/calibrate.py` (`MODELS`, the "quantized" family).
- Depends on: K0 and D1, then H1 (rebase on both; same file).

#### Step by step
**PR 1, unpacking only (no new findings):**
1. In `src/lib.rs`, add unpacking functions that read packed values in place from the memory map
   (no copy of the whole file):
   - GPTQ/AWQ `qweight` int32 tensors: eight 4-bit values per word, or four 8-bit values for 8-bit
     variants (read the `bits` value from the repo's `quantize_config.json` or `config.json`,
     passed in from Python);
   - bitsandbytes 4-bit `uint8`: two 4-bit values per byte;
   - FP8 (`F8_E4M3`): one value per byte, with its sign/exponent bits.
2. Expose per-tensor statistics of the unpacked values (value histogram entropy, plus the
   lane and plane statistics that make sense for each format) on `TensorScanResult`, without
   any new rule yet.
3. Rust unit tests for each unpacker on hand-built words.

**PR 2, rules and calibration:**
4. Run the unpacked statistics over the quantized models in `benchmarks/calibrate.py` (add more
   to the lock: at least three GPTQ, two AWQ, two bnb-4bit and two FP8 repos). Choose thresholds
   from the clean distributions, with a margin, as in D1.
5. Add the rules, threat-matrix rows (packed payload per format) and claims.

#### Tests you must write first
- Rust: `unpack_gptq_int4_word_gives_eight_values`, `unpack_bnb_byte_gives_two_values`,
  `fp8_e4m3_sign_exponent_extraction`, each on hand-built inputs with known answers.
- Python: `test_payload_in_packed_int4_tensor_is_flagged` (random bytes replacing a realistic
  4-bit `qweight`), `test_clean_quantized_tensor_is_not_flagged` (a `qweight` built from a
  realistic weight distribution), and the same pair for bnb-4bit and FP8.

#### Done when
- [ ] Two PRs (unpacking, then rules plus calibration).
- [ ] Unpacking reads from the memory map; no tensor is copied whole.
- [ ] False-positive rates per quantized format are published in docs/calibration.md.
- [ ] Threat-matrix rows and claims added; CHANGELOG updated.
- [ ] CI is fully green; diff coverage ≥ 90%.

#### Don't
- Don't add a rule in PR 1.
- Don't load any model with transformers or bitsandbytes to unpack it; read the bytes.

#### Depends on / blocks
Depends on K0, D1, H1. Blocks D4.

---

### [D3] Run the backdoor benchmark for real (due 6 Nov)

**Owner:** @7dxksh7 · **Labels:** `owner:daksh`, `priority:high`, `heavy`, `area:fuzzer`, `area:benchmarks`

#### What is this, simply?
We built a test track (`benchmarks/backdoor_eval.py`) but never drove a car round it. This issue
runs the race: our fuzzer against two well-known detectors, on models with planted backdoors
and on clean "twin" models trained the same way, and writes down who wins each lap, honestly.

#### Why it matters
T15 (trigger backdoors) is the only behavioural threat we claim to detect. Without measured TPR
and FPR the claim is a hope. If we lose to Neural Cleanse, users deserve to know.

#### Before you start
- Branch: `bench/backdoor-eval`.
- Read: the docstring and `main()` of `benchmarks/backdoor_eval.py`; `aegis/fuzzer.py`
  (`DynamicTrojanFuzzer`, `build_auto_fuzz_suite`).
- Environment: Python 3.12 with torch, tensorflow, `adversarial-robustness-toolbox` and aegis
  (`pip install -e ".[bench,fuzzer]"` plus tensorflow). Apple Silicon uses `--device mps`.
- Depends on: K0.

#### Step by step
1. Smoke run first: `python benchmarks/backdoor_eval.py --seeds 1 --clean-seeds 1 --epochs 2 --nc-steps 20`.
2. Full run with the defaults: BadNets, Blended, WaNet and clean-label, three seeds each, plus
   clean twins trained with the same recipe. The script resumes if interrupted.
3. Commit `benchmarks/results/backdoor_eval.json` and the generated `docs/backdoor_eval.md`.
   Do not commit `benchmarks/results/backdoor_models/`.
4. In docs/backdoor_eval.md, report TPR, FPR and runtime per method and attack. Implants that
   failed (ASR below `--min-asr`) are listed, not counted.
5. Update docs/THREAT_MATRIX.md (T15) and `docs/claims.json` to match the data exactly. If the
   fuzzer does badly on an attack, say so.

#### Tests you must write first
- `test_backdoor_results_match_the_report`: loads `benchmarks/results/backdoor_eval.json` and
  checks that every TPR/FPR in docs/backdoor_eval.md is computed from it (no hand-typed
  numbers).
- If you add a claim: a `@pytest.mark.claim` test that re-derives the claimed numbers from the
  results file, following `test_calibration_claim_matches_current_results_and_rules`.

#### Done when
- [ ] Full run committed (results JSON plus report), with seeds, epochs and versions recorded.
- [ ] TPR, FPR and runtime per method and attack, including where we lose.
- [ ] THREAT_MATRIX T15 and claims match the data; CHANGELOG updated.
- [ ] CI is fully green.

#### Don't
- Don't tune the fuzzer's threshold on the test models and then report results on the same
  models.
- Don't drop an attack or a seed because the result looks bad.

#### Depends on / blocks
Depends on K0. Feeds K3.

---

### [D4] Extend calibration to models over 3 GB and more GGUF quantizations (due 8 Nov)

**Owner:** @7dxksh7 · **Labels:** `owner:daksh`, `priority:high`, `heavy`, `area:benchmarks`

#### What is this, simply?
So far we only tested Aegis on small and medium models, like testing a bridge with cars but
never with a lorry. People run 7B to 70B models locally, and mostly as GGUF. This issue tests
the lorries and measures how much memory and time a scan takes as models grow.

#### Why it matters
Our false-positive claim (C10) only covers what we measured. This is also the final
calibration run, under the rules everyone else changed this sprint, so it is the one that makes
C10 claimable for the release.

#### Before you start
- Branch: `bench/calibration-large`.
- Read: `benchmarks/calibrate.py` (`MODELS`, `GGUF_FULL`, `GGUF_HEADER_ONLY`, `--max-gb`,
  `--evict`); `benchmarks/calibration_lock.json`; docs/calibration.md.
- Depends on: D1, H1, D2 and A1(a) merged (all change the scan fingerprint).

#### Step by step
1. Add larger LLMs (above 3 GB, at least four families) to `MODELS`, and GGUF Q4_K_M, Q5_K_M and
   Q8_0 files from at least four families to `GGUF_FULL`. Pin them in the lock (first run
   without `--refresh-lock` pins new entries).
2. Run with `--max-gb` raised and `--evict`, so each repo is deleted after scanning. Files are
   memory-mapped, never loaded whole.
3. Record peak memory (RSS) and wall-clock scan time per model, and add a table of scan time and
   peak memory against model size to docs/calibration.md.
4. Run `python scripts/claims.py --write` and confirm C10 is claimed with the new totals.

#### Tests you must write first
- `test_scan_memory_is_bounded_by_tensor_not_file_size`: scan a generated multi-tensor
  safetensors file of a few hundred MB and assert peak RSS grows by much less than the file size
  (use `resource.getrusage`; skip on platforms without it).

#### Done when
- [ ] Models over 3 GB and GGUF Q4/Q5/Q8 from several families are pinned and scanned.
- [ ] False-positive rates published per set; peak memory and scan time per size in docs/calibration.md.
- [ ] C10 is claimed under the final rules; `python scripts/claims.py` passes.
- [ ] CI is fully green; CHANGELOG updated.

#### Don't
- Don't start before D1, H1, D2 and A1(a) have merged: the run would be stale on arrival.
- Don't load models with transformers to "check" them; scan the files.

#### Depends on / blocks
Depends on D1, H1, D2, A1. Blocks K3.

---

### [A1] `aegis scan hf://org/repo[@revision]`: scan before downloading (due 3 Nov)

**Owner:** @Amandeep-bajwa · **Labels:** `owner:amandeep`, `priority:high`, `area:hub`, `area:cli`

#### What is this, simply?
Before you eat a sandwich from a stranger you can read the label on the wrapper. A safetensors
file has a label at the front (an 8-byte length, then a JSON header), and a GGUF file has one
too (its metadata, including the chat template). Both can be fetched without downloading the
whole file, using an HTTP "range request" that asks the server for only the first few bytes.
This issue lets a user check a Hugging Face model's labels before downloading gigabytes.

#### Why it matters
Today a user must download the model before Aegis can look at it. The header, metadata, chat
template and credential checks only need the first few kilobytes. Your closed PR #13 had this
idea (`aegis/hf_inspector.py`); thank you. This issue rebuilds it on the current findings and
verdict code.

#### Before you start
- Branches: `feat/hf-range-safetensors` (PR a), then `feat/hf-gguf-cli` (PR b).
- Read: CLAUDE.md and docs/DECISIONS.md; `aegis/formatscan.py` (`scan_safetensors_header`);
  `aegis/modelformats.py` (`parse_gguf_header`, `chat_templates`, `_check_template`,
  `_check_secrets`); `benchmarks/calibrate.py` (`scan_gguf_header`, which already grows a
  range-fetched buffer until the GGUF header parses); `aegis/findings.py` and
  `aegis/verdict.py`; the scan command in `aegis/cli.py`; closed PR #13 for ideas only.
- Depends on: K0.

#### Step by step
**PR (a): range fetch plus safetensors**
1. Create `aegis/hub.py` with a single function that does all network access, so tests can
   replace it: `_http_get(url: str, headers: dict) -> tuple[int, bytes]`. Use `urllib.request`
   from the standard library. **No new runtime dependency.**
2. `resolve(repo, revision)`: call `https://huggingface.co/api/models/{repo}/revision/{revision}`
   and return the commit SHA and the file list. Default revision `main`, always resolved to a SHA,
   and report the SHA in the output.
3. If the repo has `.py` files or its `config.json` has an `auto_map` entry, add a finding that the
   model needs `trust_remote_code` (loading it runs the repo's code).
4. `fetch_safetensors_header(repo, sha, filename)`: range-request bytes 0-7, read the
   little-endian length, refuse anything over 100 MB (the length is attacker-controlled; the
   safetensors format caps headers at 100 MB), then range-request the header.
5. Refactor `scan_safetensors_header` so its checks run on header bytes
   (`scan_safetensors_header_bytes(raw: bytes, header_len: int, label: str)`), and the
   existing path-based function calls it. Existing tests must pass unchanged.
6. Pickle files (`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`): add the finding "pickle format: full
   download required to scan; prefer the safetensors version". Never download them.
7. Because `aegis/formatscan.py` is in the rules fingerprint, run `python scripts/claims.py --write` (C10 goes pending; D4 re-claims it).

**PR (b): GGUF plus CLI and README**
8. `fetch_gguf_header(repo, sha, filename)`: grow a range-fetched buffer until
   `parse_gguf_header` succeeds, as `scan_gguf_header` in `benchmarks/calibrate.py` does, with a
   hard cap (for example 64 MB). Run `_check_template` on every chat template and `_check_secrets`
   on the metadata.
9. `aegis scan hf://org/repo[@revision]` in `aegis/cli.py`: one result per remote file, through
   the same findings, verdict, JSON and SARIF code as local files. Exit codes as for local files.
10. README section "Scanning before you download", with the limits stated: only headers are
    checked; tensor rules need the file; pickles need a full download.

#### Tests you must write first
Every test replaces `aegis.hub._http_get` with a fake (monkeypatch). No test touches the network.
- `test_resolve_pins_revision_to_commit_sha`: fake API response; expect the SHA in the result.
- `test_safetensors_header_is_fetched_with_two_range_requests`: assert exactly the `Range:
  bytes=0-7` request and then the header-range request, and nothing else.
- `test_oversized_header_length_is_refused_without_fetching`: length field 2^40. Expect a
  finding and no second request.
- `test_metadata_payload_found_remotely_matches_local_scan`: the same file scanned locally and
  through the fake hub gives the same header findings.
- `test_leaked_token_in_remote_metadata_is_masked`.
- `test_pickle_file_is_reported_not_downloaded`: assert no range request for the `.bin` file.
- `test_trust_remote_code_repo_gets_a_finding`.
- PR (b): `test_gguf_template_injection_found_from_header_only`,
  `test_gguf_header_growth_stops_at_cap`, `test_cli_hf_scan_json_validates_against_schema`.
- One optional real-network test marked `@pytest.mark.network`, excluded by default (register the
  marker in `tests/conftest.py` and add `-m "not network"` to the default pytest options).

#### Done when
- [ ] Two PRs: (a), then (b) after (a) has merged.
- [ ] No new runtime dependency; all network access goes through `aegis/hub.py:_http_get`.
- [ ] No test uses the network unless marked `network` (excluded by default).
- [ ] Remote and local scans of the same header give the same findings.
- [ ] Claims for the new behaviour in `docs/claims.json`, each with a test; README and CHANGELOG updated.
- [ ] CI is fully green; diff coverage ≥ 90%.

#### Don't
- Don't copy `aegis/hf_inspector.py` from PR #13 wholesale; rebuild on `aegis/findings.py`.
- Don't trust the header length field; cap it before fetching.
- Don't download pickle files, and never load anything.
- No marketing words ("zero-download", "very-high precision") and no claim without a test.

#### Prompt for your AI assistant
```text
You are working in the Aegis-Tensor repository. First read CLAUDE.md and docs/DECISIONS.md, then
GitHub issue [A1] (pasted below). Work on PR (a) only: range fetch plus safetensors.

Rules: plan first and show me the plan (files, functions, tests) before writing code. Write the
tests listed in the issue first and run them to see them fail. All network access goes through a
single function aegis/hub.py:_http_get(url, headers) -> (status, body) using urllib; tests replace
it with monkeypatch and never touch the network. No new runtime dependencies. Reuse
aegis/formatscan.py by refactoring scan_safetensors_header so its checks run on header bytes;
existing tests must keep passing unchanged. Cap the header length at 100 MB before fetching. Never
download or load pickle files. Do not change any detection threshold or any existing test without
explaining why in the PR. Run: pytest tests/, ruff check .,
ruff format --check ., mypy aegis, python scripts/claims.py --write. Every claim needs a test.
Do not use words like "zero-download" or "very-high precision".

[paste issue A1 here]
```

#### Depends on / blocks
Depends on K0. PR (a) blocks D4 (scan fingerprint).

---

### [A2] Release workflow (due 6 Nov)

**Owner:** @Amandeep-bajwa · **Labels:** `owner:amandeep`, `priority:high`, `area:ci`

#### What is this, simply?
A wheel is a ready-to-install package, like a flat-pack desk with the screws already in the
right bag. Users without Rust need one per computer type. This workflow builds them
automatically when we tag a release, tries each one, and attaches them to the release page.

#### Why it matters
Without wheels, every user needs Rust and a compiler. K3 (the release) cannot ship without this.

#### Before you start
- Branch: `ci/release-workflow`.
- Read: the `wheel` job in `.github/workflows/ci.yml` (copy its smoke test); `pyproject.toml`
  (`[tool.maturin]`); `Cargo.toml` (`pyo3` with `abi3-py310`, so one wheel per platform covers
  Python 3.10+).
- Depends on: K0.

#### Step by step
1. Create `.github/workflows/release.yml`, triggered on tags matching `v*`, with
   `permissions: contents: write` only on the job that publishes.
2. Build abi3 wheels with `PyO3/maturin-action` for Linux x86_64 (manylinux), macOS arm64 and
   macOS x86_64, plus an sdist.
3. Smoke-test every wheel on its own platform in a fresh venv, from a temp directory: `aegis
   --version`, `aegis doctor`, and `aegis scan` on a generated safetensors file (same steps as
   the `wheel` job).
4. Attach every wheel and the sdist to the GitHub release for the tag.
5. Test on your fork: push tag `v0.0.0-test`, link the successful run in the PR, then delete
   the tag and the test release from your fork.

#### Tests you must write first
- `tests/test_release_workflow.py`: parse `.github/workflows/release.yml` with `yaml.safe_load`
  and assert it triggers only on `v*` tags, builds the three targets plus an sdist, smoke-tests
  each wheel before publishing, and grants `contents: write` only to the publishing job. (Add
  `pyyaml` to the dev extra if it is not already installed.)

#### Done when
- [ ] A green run on your fork for `v0.0.0-test` is linked in the PR, with all four artifacts attached.
- [ ] Every wheel is smoke-tested before the publish step runs.
- [ ] CI is fully green; CHANGELOG updated.

#### Don't
- Don't publish to PyPI in this issue.
- Don't push test tags to `KrishG7/Aegis-Tensor`; use your fork.
- Don't give the whole workflow write permissions.

#### Prompt for your AI assistant
```text
You are working in the Aegis-Tensor repository. Read CLAUDE.md and docs/DECISIONS.md first, then
issue [A2] below. Plan first and show me the plan. Write tests/test_release_workflow.py first (it
parses the workflow YAML and checks the trigger, targets, smoke tests and permissions), run it to
see it fail, then write .github/workflows/release.yml. Use PyO3/maturin-action to build abi3
wheels for linux x86_64 (manylinux), macos arm64 and macos x86_64, plus an sdist. Copy the smoke
test from the `wheel` job in .github/workflows/ci.yml. Only the publishing job gets
`contents: write`. Do not publish to PyPI. Do not touch any other
workflow. Every claim in the PR description needs evidence (a link to the green run on my fork).

[paste issue A2 here]
```

#### Depends on / blocks
Depends on K0. Blocks K3.

---

### [A3] CHANGELOG.md history and claims tooling tests (due 7 Nov)

**Owner:** @Amandeep-bajwa · **Labels:** `owner:amandeep`, `priority:medium`, `area:docs`, `area:ci`

#### What is this, simply?
A changelog is the project's diary: what changed in each version, in plain words. The gates PR
started it with an empty "Unreleased" page; this issue fills in everything since 0.1.0 from the
git history. It also tests the tool that keeps the README's claims table honest, so it can't
quietly stop working.

#### Why it matters
Users read the changelog to decide whether to upgrade. `scripts/claims.py` protects every claim
in the README, but nothing tests that it actually catches a mismatch.

#### Before you start
- Branch: `docs/changelog-and-claims-tests`.
- Read: `CHANGELOG.md`; `git log --oneline v0.1.0..main` (or from the first commit if there is no
  tag); `scripts/claims.py` (`check`, `render_section`, `collect_markers`);
  `tests/test_claims.py`.
- Note: `test_every_claim_has_evidence_and_the_readme_matches_the_register` already fails CI
  when the README table and `docs/claims.json` disagree. This issue adds tests that prove the
  checker catches each kind of mistake, and runs the full checker in CI.
- Depends on: K0.

#### Step by step
1. Backfill `CHANGELOG.md` (Keep a Changelog format) from the git log: group by Added, Changed,
   Fixed and Security, in plain words, with no marketing language.
2. In `tests/test_claims.py`, add negative tests that run `claims.check` against a temporary copy
   of the register and README (monkeypatch `claims.REGISTER`, `claims.README` and
   `claims.TESTS` to `tmp_path` copies).
3. Add a `lint`-job step to `.github/workflows/ci.yml` that runs `python scripts/claims.py`
   (the full check, including benchmark staleness). Today it crashes without torch:
   `disarm_values()` imports `benchmarks/disarm_impact.py`, which imports torch at module level.
   Move the torch and transformers imports in `benchmarks/disarm_impact.py` into the functions that
   use them, so the lint job (which has no torch) can run the script.

#### Tests you must write first
- `test_check_reports_readme_out_of_date`: edit one claim's text in the copied register, keep the
  old README. Expect "README claims section is out of date".
- `test_check_reports_claim_without_a_test`: add a claim id with no marked test. Expect "has no
  test marked".
- `test_check_reports_marker_for_unknown_claim`: a copied test file marks `C99`. Expect "not in
  docs/claims.json".
- `test_check_reports_duplicate_ids`.
- `test_claims_script_runs_without_optional_dependencies`: run `python scripts/claims.py` in a
  subprocess with `sys.modules` entries for torch and huggingface_hub set to None (as
  `test_disarm_runs_without_torch` does for torch).

#### Done when
- [ ] CHANGELOG.md covers every user-visible change since 0.1.0.
- [ ] Each negative test fails if the corresponding check in `scripts/claims.py` is removed.
- [ ] The lint job runs `python scripts/claims.py`.
- [ ] CI is fully green; diff coverage ≥ 90%.

#### Don't
- Don't change claim wording in `docs/claims.json` in this PR.
- No marketing words in the changelog.

#### Prompt for your AI assistant
```text
You are working in the Aegis-Tensor repository. Read CLAUDE.md and docs/DECISIONS.md first, then
issue [A3] below. Plan first and show me the plan. Part 1: backfill CHANGELOG.md in Keep a
Changelog format from `git log`, in plain words, grouped as Added / Changed / Fixed / Security,
with no marketing language. Part 2: write the negative tests listed in the issue first, in
tests/test_claims.py, using tmp_path copies of docs/claims.json, README.md and a test file, with
monkeypatch on claims.REGISTER, claims.README and claims.TESTS. Each test must fail if the matching
check in scripts/claims.py is deleted; prove that by deleting it temporarily. Then add
`python scripts/claims.py` to the lint job in .github/workflows/ci.yml, after making sure it
runs without torch (benchmarks/disarm_impact.py imports torch at module level today; move it into
the functions that use it). Do not change docs/claims.json wording, any detection
threshold or any existing test.

[paste issue A3 here]
```

#### Depends on / blocks
Depends on K0. Blocks K3.

---

### [H1] Close the T19 gap: payloads split across tiny tensors (due 2 Nov)

**Owner:** @HimanshuChhillar · **Labels:** `owner:himanshu`, `priority:critical`, `area:tensor-rules`

#### What is this, simply?
Our exponent-lane rule only looks at tensors with at least 256 numbers, because a smaller
sample is too noisy, like judging a coin as unfair after three flips. An attacker can exploit
that by cutting a payload into many tiny tensors, each one too small to be checked. The fix is
to pour all the tiny tensors of the same type into one pile and check the pile.

#### Why it matters
This is a documented gap with a strict-xfail test (`test_t19_payload_split_into_tiny_tensors_is_flagged`
in `tests/test_threat_matrix.py`) and a known limit in the README (L02). Anyone who reads
`MIN_LANE_ELEMENTS` in `src/lib.rs` knows how to use it.

#### Before you start
- Branch: `fix/pool-small-tensors`.
- Read: CLAUDE.md and docs/DECISIONS.md; `src/lib.rs` (`MIN_LANE_ELEMENTS`, `analyze_tensor`,
  `scan_safetensors`, `TensorScanResult`); `aegis/findings.py` (`from_safetensors`);
  `tests/test_threat_matrix.py` (the T19 test); `docs/claims.json` (L02).
- Depends on: K0 and D1 (rebase on D1; it edits the same Rust code).

#### Step by step
1. In `scan_safetensors`, after the parallel per-tensor pass, collect the float tensors (F32,
   F16, BF16) with fewer than `MIN_LANE_ELEMENTS` elements, grouped by dtype, in tensor-name
   order.
2. For each dtype group whose combined element count reaches `MIN_LANE_ELEMENTS`, concatenate
   their bytes and run `analyze_tensor` on the pool, with a name such as
   `pooled small F32 tensors (64)`.
3. Add a field `pooled_from: Vec<String>` to `TensorScanResult` (with `#[pyo3(get)]`): the names of
   the contributing tensors for a pooled result, empty for a normal tensor. Push the pooled
   result only if it has findings.
4. In `aegis/findings.py` (`from_safetensors`), put `pooled_from` into the finding's evidence, so
   the report says which tensors contributed.
5. Remove the `xfail` from the T19 test. Update T19 in docs/THREAT_MATRIX.md to `tested`, remove
   limit L02 from `docs/claims.json` (and its marker on the test), and add a claim with the T19
   test as evidence. Run `python scripts/claims.py --write`.
6. Ask Daksh to run calibration on the small-tensor-heavy models (or run
   `python benchmarks/calibrate.py --only <repos> --sets safetensors`) and report any new false
   positive in the PR.

#### Tests you must write first
- The existing T19 test, minus `xfail`: 64 tensors of 64 random-byte float32 values. Expect a finding.
- `test_pooled_finding_names_every_contributing_tensor`: the same file; the finding's evidence
  lists all 64 names.
- `test_small_clean_tensors_pooled_are_not_flagged`: 64 small tensors of realistic weights
  (`np.random.default_rng(0).normal(0, 0.02, 64)`), plus a few layer-norm-like vectors near 1.0.
  Expect no finding.
- `test_pools_are_per_dtype`: 40 small F16 and 40 small F32 tensors of random bytes, each group
  above 256 elements in total. Expect two pooled results.
- Rust: a unit test that a pool below `MIN_LANE_ELEMENTS` elements is not analysed.

#### Done when
- [ ] The T19 test passes without `xfail`; L02 is gone and T19 is `tested`.
- [ ] Calibration on small-tensor-heavy models shows no new false positive (numbers in the PR).
- [ ] CI is fully green; diff coverage ≥ 90%; `cargo clippy -D warnings` clean.
- [ ] docs/THREAT_MATRIX.md, `docs/claims.json` and CHANGELOG.md updated.

#### Don't
- Don't lower `MIN_LANE_ELEMENTS`; small samples are too noisy on their own.
- Don't change any threshold to make the clean-pool test pass. If clean pools trip the rule,
  stop and post the numbers in the issue.
- GGUF tensors go through `scan_tensor_bytes` one at a time and are not pooled by this change.
  Say so in the T19 row rather than claiming more.

#### Prompt for your AI assistant
```text
You are working in the Aegis-Tensor repository. Read CLAUDE.md and docs/DECISIONS.md first, then
issue [H1] below. Plan first and show me the plan before editing. Write the tests listed in the
issue first (Python in tests/test_threat_matrix.py, Rust in the tests module of src/lib.rs) and
run them to see them fail. Then, in src/lib.rs scan_safetensors, after the per-tensor pass, pool
F32/F16/BF16 tensors with fewer than MIN_LANE_ELEMENTS elements by dtype (in name order),
run analyze_tensor on each pool that reaches MIN_LANE_ELEMENTS, and record the contributing
tensor names in a new TensorScanResult field `pooled_from`. Pass it into the finding evidence in
aegis/findings.py from_safetensors. Do not change MIN_LANE_ELEMENTS or any threshold, and do not
weaken any test. After editing Rust, run `maturin develop --release` before pytest. Run:
cargo fmt --all --check; cargo clippy --all-targets --no-default-features -- -D warnings;
cargo test --no-default-features; pytest tests/; ruff check .; mypy aegis;
python scripts/claims.py --write.

[paste issue H1 here]
```

#### Depends on / blocks
Depends on K0 and D1. Blocks D2 (rebase) and D4.

---

### [H2] Fix documentation drift and verify citations (due 4 Nov)

**Owner:** @HimanshuChhillar · **Labels:** `owner:himanshu`, `priority:medium`, `area:docs`

#### What is this, simply?
Some of our documents describe the project as it was planned in September, like an old map
of a town that has since grown new roads. People follow old maps and get lost. This issue
labels the old maps as history, checks that the numbers we quote from research papers really
are in those papers, and runs every command in the README to make sure it still works.

#### Why it matters
A wrong figure attributed to a paper is the kind of mistake that costs a security tool its
credibility. A README command that fails is the first thing a new user meets.

#### Before you start
- Branch: `docs/drift-and-citations`.
- Read: README.md; docs/THREAT_MATRIX.md (section "Precedents"); docs/RESEARCH_COMPENDIUM.md,
  section 2.2 and the reference list.
- Depends on: K0.

#### Step by step
1. Add this banner at the top of docs/ARCHITECTURE.md, docs/TECH_STACK.md, docs/PHASES.md,
   docs/REQUIREMENTS.md, docs/TEAM_ALLOCATION.md and docs/ISSUE_7_ENGINEERING_DESIGN.md:
   `> **Design history.** This document records an earlier plan. For current behaviour see the README, docs/THREAT_MATRIX.md and docs/DECISIONS.md.`
2. Open the original papers:
   - EvilModel (Wang, Liu and Cui, arXiv:2107.08415, and the later IEEE version);
   - StegoNet (Liu et al., ACSAC 2020).
   Check every figure attributed to them in docs/RESEARCH_COMPENDIUM.md, section 2.2: AlexNet
   size, 36.9 MB of malware, "less than 1%" accuracy loss, 58 antivirus engines, the four
   embedding techniques, and survival of transfer learning. Fix each one to what the paper says,
   with a page or section reference, or remove it. Do the same for "MalDNet (Kim et al., 2022)";
   if the paper can't be found, remove it.
3. In docs/THREAT_MATRIX.md, "Precedents": move EvilModel and StegoNet to "checked" once
   verified, and fix the T01 row if a citation changes.
4. Run every command in the README exactly as written, on a fresh clone. Fix any that fail; if a
   failure is a code bug, open an issue instead and link it.

#### Tests you must write first
- `tests/test_docs.py::test_history_documents_carry_the_banner`: each of the six files starts with
  the banner.
- `tests/test_docs.py::test_readme_relative_links_exist`: every relative link in README.md and
  docs/*.md points to a file that exists.

#### Done when
- [ ] Six banners added; both tests pass.
- [ ] Every EvilModel, StegoNet and MalDNet figure is verified (with page or section) or removed;
      the PR lists each figure and its outcome.
- [ ] Every README command was run; the PR lists each with its result.
- [ ] CI is fully green; CHANGELOG updated.

#### Don't
- Don't rewrite the history documents; only add the banner.
- Don't cite a figure you didn't find in the paper itself (not a blog post or summary).

#### Prompt for your AI assistant
```text
You are working in the Aegis-Tensor repository. Read CLAUDE.md and docs/DECISIONS.md first, then
issue [H2] below. Plan first. Write tests/test_docs.py first (the banner test and the relative-link
test) and run it to see it fail. Then add the exact banner text from the issue to the six files,
without editing anything else in them. For the citations, do not rely on your own memory: I will
paste the relevant text from each paper. List every figure from docs/RESEARCH_COMPENDIUM.md
section 2.2 and docs/THREAT_MATRIX.md "Precedents" in a table (figure, source paper, verified
yes/no, page), and only keep figures I confirm from the paper text. Then help me run every README
command and record the results. Do not change any code or test outside tests/test_docs.py.

[paste issue H2 here]
```

#### Depends on / blocks
Depends on K0.

---

### [H3] Fresh-install test and example Action check (due 8 Nov; re-run 9 Nov after the final merges)

**Owner:** @HimanshuChhillar · **Labels:** `owner:himanshu`, `priority:high`, `area:docs`, `area:ci`

#### What is this, simply?
Pretend you have never seen this project. Take a clean computer, follow only the README, and
time every step with a stopwatch. Wherever you get stuck, a real user would too. Then try the
GitHub Action we tell people to copy, and check that it works on a real repository.

#### Why it matters
The README promises a quick start. If it takes 40 minutes or needs help from us, people give up
before they ever scan a file. The release (K3) waits for this to pass.

#### Before you start
- Branch: `docs/fresh-install-fixes`.
- Read: README.md ("Quick start"); `examples/github-actions/aegis-model-scan.yml`.
- You need: a clean macOS (Apple Silicon) user account or machine, and a clean Ubuntu (a VM or a
  GitHub Codespace).
- Depends on: K0. Run again on 9 Nov after the final merges.

#### Step by step
1. On each machine, follow only the README, from `git clone` to `aegis doctor` and a first
   `aegis scan`. Time each step and write down every snag (missing tool, unclear wording, error).
2. Fix the README directly for anything that is a documentation problem. Open an issue for
   anything that is a code bug, and link it.
3. The quickstart must take under 5 minutes with no outside help, not counting the one-time Rust
   install. If it can't, say what blocks it.
4. Create a throwaway public repository with a small benign model
   (`python tests/generate_stego_model.py` creates clean and test models). Copy
   `examples/github-actions/aegis-model-scan.yml` into it, open a PR that adds the model, and link
   the run.
5. The example workflow triggers only on `.safetensors`, `.pt`, `.pth`, `.bin`, `.pkl`, `.ckpt`,
   `.npy` and `.npz`. Add `.gguf`, `.onnx`, `.keras` and `.h5` (Aegis scans them, and GGUF is what
   most local users run), and check the run again with a `.gguf` file.

#### Tests you must write first
- `tests/test_docs.py::test_example_workflow_triggers_on_every_scanned_format`: parse the example
  workflow with `yaml.safe_load` and assert its `paths` include every extension above.

#### Done when
- [ ] A timing table for both machines is in the PR.
- [ ] README fixed; code bugs filed as issues and linked.
- [ ] A link to a green run of the example Action on a throwaway repo, including a `.gguf` file.
- [ ] The 9 Nov re-run is recorded in a comment on this issue.
- [ ] CI is fully green; CHANGELOG updated.

#### Don't
- Don't use your existing development machine or environment for the timing; it hides problems.
- Don't put a malicious model in the throwaway repo; use the benign generated ones.

#### Prompt for your AI assistant
```text
You are helping me test the Aegis-Tensor README on a clean machine. Read CLAUDE.md, then issue [H3]
below. Do not run commands for me from your own knowledge of the project: I will paste each README
step's output, and you help me record the time, the result and any snag in a table. For each snag,
decide with me whether it is a README problem (we fix the text) or a code bug (we draft a GitHub
issue). Then help me write tests/test_docs.py::test_example_workflow_triggers_on_every_scanned_format
first, see it fail, and add .gguf, .onnx, .keras and .h5 to the paths in
examples/github-actions/aegis-model-scan.yml. Do not change any other code or test.

[paste issue H3 here]
```

#### Depends on / blocks
Depends on K0. Blocks K3.

---

### [T0] v0.2.0 release tracker

**Owner:** @KrishG7 · Pinned · **Labels:** `owner:krish`, `priority:critical`

Every issue in the milestone, by due date (end of day IST). Tick an item when its PR merges.

- [ ] K0: Merge the quality gates and turn on branch protection (26 Oct)
- [ ] D1: Fix calibration false positives and claim C10 (31 Oct)
- [ ] K1: Resolve pickle callables built at runtime, and scan every pickle in legacy tar members (1 Nov)
- [ ] H1: Close the T19 gap: payloads split across tiny tensors (2 Nov)
- [ ] A1: `aegis scan hf://`: scan before downloading (3 Nov)
- [ ] H2: Fix documentation drift and verify citations (4 Nov)
- [ ] D2: Format-aware rule for packed quantized tensors (5 Nov)
- [ ] D3: Run the backdoor benchmark for real (6 Nov)
- [ ] A2: Release workflow (6 Nov)
- [ ] K2: Known bypasses page and disarm guarantee tests (7 Nov)
- [ ] A3: CHANGELOG.md history and claims tooling tests (7 Nov)
- [ ] D4: Extend calibration to models over 3 GB and more GGUF quantizations (8 Nov)
- [ ] H3: Fresh-install test and example Action check (8 Nov, re-run 9 Nov)
- [ ] K3: Release v0.2.0 (9 Nov, tag 10 Nov)

The rules, timeline and dependencies are in docs/TEAM_PLAN.md.
