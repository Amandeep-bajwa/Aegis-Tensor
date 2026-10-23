# Threat Matrix

What Aegis-Tensor detects, where it points, and what it can treat. Each in-scope row
is backed by a test in `tests/test_threat_matrix.py`. Out-of-scope rows are listed
so their absence is a decision, not an oversight.

**Status key:** `tested` = covered by a passing test · `gap` = a known, strict-xfail
test documents it · `limit` = detected only in part, with the limit stated · `out` = out of scope.

**Treatments:** `disarm` = `aegis disarm` rounds away low mantissa bits · `strip` =
`aegis disarm --strip-metadata` · `quarantine` = the tensor must come from another source ·
`refuse` = do not load the file.

## A. Model artifact: tensor data

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T01 | Structured payload in low mantissa bits (text or code) | EvilModel, StegoNet (LSB) | Low-bit plane entropy < 7.5 | Tensor name | `disarm` | tested |
| T02 | Random payload in low mantissa bits | MaleficNet-style encrypted payloads | Not statically separable from trained weights | — | `disarm` (removes it) | limit |
| T03 | Whole tensor overwritten with random bytes | Naive encrypted-blob injection | Sign/exponent lane entropy > 6.5, pooled entropy | Tensor name | `quarantine` | tested |
| T04 | Partial overwrite (about 1% of floats) | Targeted tampering | Out-of-range fraction > 0.1% | Tensor name + fraction | `quarantine` | tested |
| T08 | Clean model | — | No finding on any channel | — | — | tested (no false positives) |
| T17 | NaN / Inf injection | Inference-breaking tampering | Out-of-range fraction | Tensor name | `quarantine` | tested |
| T19 | Payload split into tensors under 256 elements | Evasion of the sample floor | Not flagged | — | — | gap (strict xfail) |

Known limits: the lane rule and the out-of-range rule assume trained weights stay below
1e4 in magnitude. Tensors with genuinely larger values need a per-model threshold. Partial
overwrites under about 1% of floats can evade the lane rule, but the out-of-range rule
catches them.

## B. Model artifact: container and header

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T05 | Payload in `__metadata__` | Container-level smuggling | Metadata value > 1 KiB or total > 4 KiB | Metadata key | `strip` | tested |
| T06 | Bit channel in header whitespace | Covert channel | More than 16 bytes of unexplained slack | Header | `strip` | tested |
| T07 | Oversized tensor name | Covert channel | Name > 256 bytes | Tensor name | `quarantine` | tested |
| T09 | Truncated or corrupt file | Parser confusion | Library rejects; scan fails loudly | File | `refuse` | tested |

The safetensors library itself rejects trailing bytes and overlapping offsets. Both were
probed and confirmed.

## C. Model artifact: pickle containers (.pt, .pth, .bin, .pkl)

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T10 | Code execution via imported `os`/`posix` callable | Pickle RCE; nullifAI (Hugging Face, Feb 2025) | Opcode scan against allowlist | Global name | `refuse` | tested |
| T11 | `builtins.eval` / `exec` / `__import__` | Pickle RCE | Opcode scan | Global name | `refuse` | tested |
| T12 | Malformed pickle (evades loaders that stop at first error) | nullifAI broken-pickle evasion | Stream parse error reported | Byte offset not yet given | `refuse` | tested |
| T13 | Loading an unsafe pickle through `aegis fuzz` | Loader-side RCE | `weights_only=True` by default; payload never runs | — | `refuse` | tested |
| — | Obfuscated import via memo or get opcodes | Pickle evasion | Only direct and `STACK_GLOBAL` with literal strings | — | — | limit |

## D. Behavioural backdoors (sleeper and trigger)

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T14 | TorchScript model cannot be inspected | Tooling gap | Refused with explanation | — | — | tested |
| T15 | Trigger-activated amplification (magnitude-keyed) | BadNets, trojaning attacks | Magnitude-matched activation ratio | Layer | `inspect` only | tested |
| T16 | Clean network false alarm at any input scale | Tooling risk | Must stay below threshold | — | — | tested |

Treatment for T15 is detection and localisation only. This tool does not yet prune or
retrain a trojaned layer. Triggers that do not act on activation magnitude, such as
semantic or token-level triggers, are not covered by the fuzzer's trigger families.

## E. Out of scope for a weights-and-container scanner

| Threat | Why out of scope | Where to address it |
|---|---|---|
| Data poisoning | Needs the training data and pipeline | Data provenance and training-pipeline controls |
| Adversarial examples at inference | Needs runtime inputs and a threat model per use case | Adversarial robustness testing against the deployed model |
| Membership inference, model extraction | Needs query access and privacy analysis | Query rate limits, differential privacy |
| Prompt injection, insecure output handling (LLM apps) | A property of the deployed application, not the weights | Application-level input and output controls |
| Supply-chain identity (who published the model) | Needs signing infrastructure, not file analysis | Model signing and provenance attestation |

## Known gaps, in priority order

1. **Calibration on real checkpoints.** All thresholds are validated on synthetic and small trained models. Real Hugging Face checkpoints are needed to measure false-positive rates.
2. **Sub-256-element tensors** (T19) are not covered. A split-payload attacker can use this.
3. **Obfuscated pickle imports** (memo and get opcodes) are not resolved.
4. **Trojan treatment.** Detection exists for T15, but there's no fine-prune or retrain step.
5. **Random low-bit payloads** (T02) are not detectable statically. Only the removal treatment works.

## Precedents

Checked against sources during the literature review: MaleficNet, nullifAI, the Anthropic probes, and Goldwasser et al.
Taken from the project compendium and not yet re-checked: EvilModel and StegoNet, whose
figures should be verified before they are cited. BadNets is cited from general knowledge.

- EvilModel (2021) and StegoNet (ACSAC 2020) for LSB embedding in weights.
- MaleficNet (arXiv 2403.03593) for spread-spectrum payloads.
- ReversingLabs report on nullifAI (Feb 2025) for broken-pickle evasion of picklescan.
- BadNets (2017) and trojaning attacks for trigger-based backdoors.
- Anthropic, "Sleeper Agents" (2024), and the linear defection probes that followed.
- Goldwasser et al. (FOCS 2022) on undetectable backdoors, and the STOC 2025 oblivious-defense follow-up.

Coverage here is mapped to the public threat categories in MITRE ATLAS and the OWASP LLM
Top 10. Those frameworks list the full threat landscape, including out-of-scope items.
