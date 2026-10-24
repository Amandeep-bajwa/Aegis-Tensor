# Known Bypasses and Limits

What Aegis does not catch, stated plainly, so "no known findings" is read correctly. Each entry
names the threat class, what Aegis does about it today, the evidence, and where a fix is tracked.
Rows refer to [docs/THREAT_MATRIX.md](THREAT_MATRIX.md).

This page lists only classes that are already public in this repository. **A new bypass is a
security report: use private vulnerability reporting ([SECURITY.md](../SECURITY.md)), not an issue
or a pull request.** It is added here once a fix or a documented limit is released.

The attacker is assumed to know every rule and threshold. That is why `aegis disarm` is the
recommended treatment for every supported file, including files with no findings.

## Tensor data

| Class | What Aegis does | Evidence | Tracked in |
|---|---|---|---|
| Random-looking payload in the lowest mantissa bits (T02) | Not detectable by any static statistic. `aegis disarm` removes it: at the default width the rounded-away bits are zero afterwards, and the payload can move a kept value by at most one rounding step | `test_t02_random_lsb_payload_is_a_documented_blind_spot_and_disarm_removes_it`, `test_default_width_leaves_nothing_of_a_low_bit_payload`, `test_files_disarmed_at_the_default_width_keep_no_low_bit_payload` (claims L01, C09) | Limit by design |
| Payload in mantissa bits above the disarm width, shaped to stay under every public threshold | Neither detected nor removed. Disarm keeps those bits, because rounding them away costs model quality (docs/disarm_impact.md) | Thresholds in `aegis/thresholds.py`; quality cost per width in docs/disarm_impact.md | Capacity not yet measured (K2) |
| Payload split across tensors of fewer than 256 elements (T19) | Not flagged: below the sample floor of the exponent-lane rule | Strict xfail `test_t19_payload_split_into_tiny_tensors_is_flagged` (claim L02) | H1 |
| Payload inside packed quantized tensors (GPTQ/AWQ int32, bitsandbytes 4-bit, FP8) | Only the pooled-entropy rule applies, with little margin. Disarm leaves quantized GGUF tensors unchanged and reports them, because their low bits are the weights | docs/calibration.md, proposal P3 | D2 |
| Small GGUF tensors | GGUF tensors are checked one at a time, so the T19 limit applies to GGUF too | `aegis/modelformats.py` (`_tensor_stats`) | H1 (safetensors first) |

## Pickle and containers

| Class | What Aegis does | Evidence | Tracked in |
|---|---|---|---|
| Import whose name cannot be resolved statically (STACK_GLOBAL on computed operands) | Reported as medium (`unresolved_import`), not critical, because the target is unknown. A call chain that starts from such a value is not escalated further | `test_unresolvable_stack_global_is_reported` (C03) | Limit: treat medium pickle findings as "do not load" until checked |
| Behaviour of allowed classes | Globals from `collections`, `copyreg`, `_codecs`, the torch tensor rebuild helpers, `torch.nn.modules` and `torch.nn.functional` are trusted by name. What their constructors or `__setstate__` do with attacker-chosen arguments is not analysed | `aegis/picklescan.py` (`_checkpoint_global`); `tests/test_pickle_allowlist.py` | Limit |
| Anything a pickle does once it is unpickled | `aegis fuzz --trust-pickle` unpickles in a sandbox; a malicious file can still forge its own fuzzing report there | docs/FUZZ_SAFETY.md (claims C13, C14) | Limit by design |

## Chat templates

| Class | What Aegis does | Evidence | Tracked in |
|---|---|---|---|
| Templates checked without jinja2 installed | Text matching only, reported as `unverifiable` | `test_without_jinja2_a_text_match_is_used_and_marked_unverifiable` | jinja2 is a required dependency; this is a fallback |

## Behavioural backdoors

| Class | What Aegis does | Evidence | Tracked in |
|---|---|---|---|
| Trigger backdoors (T15) | `aegis fuzz` detects magnitude-keyed triggers and points to the layer. Other trigger types are not yet measured. There is no repair step | `test_t15_trigger_backdoor_is_detected_and_pinpointed_to_its_layer` | D3 (benchmark), Known gaps in THREAT_MATRIX |
| Backdoors that need a running application (prompt injection, jailbreaks) | Out of scope | THREAT_MATRIX section F | Use a red-teaming tool such as garak or PyRIT |
