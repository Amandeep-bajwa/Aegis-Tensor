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
| T12 | Malformed pickle, payload before or after the broken region | nullifAI broken-pickle evasion | Scan resyncs past malformed bytes and reports every global | Entry and byte offset | `refuse` | tested |
| T13 | Loading an unsafe pickle through `aegis fuzz` | Loader-side RCE | `weights_only=True` by default; payload never runs | — | `refuse` | tested |
| T20 | Callable assembled from memoized strings (MEMOIZE/PUT then GET then STACK_GLOBAL) | Scanner evasion | Static stack and memo simulation | Byte offset | `refuse` | tested |
| T21 | Pickle renamed to another extension | Scanner evasion | Routing by magic bytes, not extension | File | `refuse` | tested |
| T22 | Checkpoint repackaged as 7z | nullifAI | 7z flagged; contents scanned with py7zr | Archive entry and offset | `refuse` | tested |
| T23 | Zip entry with encryption flag set, bad CRC, or damaged central directory | Breaks Python zip readers while PyTorch still loads | Entries read from local headers; tampering reported | Archive entry | `refuse` | tested |
| T24 | Object-dtype numpy array (`.npy`/`.npz`) carrying a pickle | Pickle smuggling | Header parsed; object payload scanned | Array and offset | `refuse` | tested |
| T25 | Unrecognised container format | Scanner evasion | Unknown format is a finding; leading bytes still opcode-scanned | File | inspect | tested |

## D. Behavioural backdoors (sleeper and trigger)

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T14 | TorchScript model cannot be inspected | Tooling gap | Refused with explanation | — | — | tested |
| T15 | Trigger-activated amplification (magnitude-keyed) | BadNets, trojaning attacks | Magnitude-matched activation ratio | Layer | `inspect` only | tested |
| T16 | Clean network false alarm at any input scale | Tooling risk | Must stay below threshold | — | — | tested |

Treatment for T15 is detection and localisation only. This tool does not yet prune or
retrain a trojaned layer. Triggers that do not act on activation magnitude, such as
semantic or token-level triggers, are not covered by the fuzzer's trigger families.

## E. Other model formats (GGUF, Keras, ONNX)

Tensor statistics for these formats come from the same Rust rules as safetensors (`scan_tensor_bytes`).

| ID | Threat | Precedent | Detected by | Pinpoints | Treatment | Status |
|---|---|---|---|---|---|---|
| T26 | Jinja template injection in a GGUF chat template | CVE-2024-34359 (llama-cpp-python) | Dunder access, `attr` filter, module or eval calls in `tokenizer.chat_template*` | Metadata key | `refuse` | tested |
| T27 | Overwritten float tensor in a GGUF file | Weight steganography | Tensor rules on F32/F16/BF16/F64 tensors | Tensor name | `quarantine` | tested |
| T28 | Malformed or truncated GGUF | Parser confusion | Bounds-checked parse; reported as a finding | File | inspect | tested |
| T29 | Keras `Lambda` layer carrying Python bytecode | Keras code execution on load without safe_mode | Config JSON walk | Layer name | `refuse` | tested |
| T30 | Keras custom object from a third-party module | Code execution through imports | `module` / `registered_name` outside Keras | Layer name | inspect | tested |
| T31 | ONNX external data path leaving the model directory | Arbitrary file read by the loader | Absolute, `..` or `~` locations | Initializer | `refuse` | tested |
| T32 | ONNX custom operator domain | Native code via a custom-op library | Node domain outside the standard ONNX domains | Node | inspect | tested |
| T33 | Overwritten tensor in ONNX initializers or Keras/HDF5 weights | Weight steganography | Tensor rules | Tensor name | `quarantine` | tested |

Limits:
- Quantized GGUF tensors (Q4/Q5/Q8/K-quants) are packed blocks, so the float rules do not apply to them and they are skipped.
- ONNX files are recognised by the `.onnx` extension plus their protobuf header, because ONNX has no magic number.
- Template checks are pattern-based. A template that renders safely in a sandboxed environment can still be flagged when it uses dunder names.

## F. Out of scope for a weights-and-container scanner

| Threat | Why out of scope | Where to address it |
|---|---|---|
| Data poisoning | Needs the training data and pipeline | Data provenance and training-pipeline controls |
| Adversarial examples at inference | Needs runtime inputs and a threat model per use case | Adversarial robustness testing against the deployed model |
| Membership inference, model extraction | Needs query access and privacy analysis | Query rate limits, differential privacy |
| Prompt injection, insecure output handling (LLM apps) | A property of the deployed application, not the weights | Application-level input and output controls |
| Supply-chain identity (who published the model) | Needs signing infrastructure, not file analysis | Model signing and provenance attestation |

## Known gaps, in priority order

1. **Calibration findings** (docs/calibration.md: 54 public models, 18,113 tensors).
   - At default thresholds, 4 of 54 models (7.4%) raise false positives.
   - Three of them are float16 bias vectors tripping the exponent-lane rule; proposal P1 is a float16 threshold of 7.2.
   - The fourth is one DETR weight with structured low bits.
   - Payloads inside **packed quantized tensors** (GPTQ/AWQ int32, bitsandbytes 4-bit uint8, FP8) are only covered by the pooled-entropy rule, with little margin. A format-aware unpacking rule is needed (proposal P3).
   - Models over 3 GB were not calibrated.
2. **Sub-256-element tensors** (T19) are not covered. A split-payload attacker can use this.
3. **Pickle imports built at runtime** (for example a callable produced by another REDUCE and then called) cannot be resolved statically. They surface as unresolved STACK_GLOBAL or as the importing global, not as the final callable. Legacy tar `storages`/`tensors` members are scanned only up to their first pickle.
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
