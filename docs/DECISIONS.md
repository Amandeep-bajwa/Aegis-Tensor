# Design Decisions

The current design, recorded so that changes to it are deliberate. Each decision cites where it
is implemented or stated. To change one, open an issue that proposes the change and its evidence.

## D1. One question

Aegis answers one question: **is this model file safe to load and run?** Every feature is judged
by whether it helps answer that question for a file a user is about to load (README, SECURITY.md).

## D2. Threat model

- **Attacker:** anyone who can publish or modify a model file: an uploader, a compromised account
  or mirror, or an insider. The attacker knows every rule and threshold, because the code is public.
- **Victim:** a person who downloads and runs open-weight models locally. Local runtimes mostly load
  GGUF; Mac users also pull MLX models, which are safetensors.
- **Consequence:** a payload can be tuned to sit just under any statistical threshold, so detection
  is not the safety net. `aegis disarm` is recommended for every supported file, including files
  with no findings (docs/THREAT_MATRIX.md, "Threat model").

## D3. Build, integrate, or leave out

- **Build:** checks that take a model file as input (tensor statistics, header checks, pickle
  simulation, chat-template analysis, format-specific checks).
- **Integrate:** established file scanners run as optional engines inside the Aegis verdict
  (Fickling and ModelScan, `aegis/engines.py`). A disagreement with Aegis is itself a finding.
- **Out of scope:** anything that needs a running application or the training pipeline: prompt
  injection and jailbreaks (use garak or PyRIT), data poisoning, adversarial examples at inference,
  membership inference and extraction, and publisher identity (docs/THREAT_MATRIX.md, section F).

## D4. Supported formats

The scanner is chosen from a file's leading bytes, never its extension:

- **safetensors** (including MLX): tensor rules on float tensors, header checks, credentials.
- **GGUF:** chat templates parsed (never rendered) with every attribute and item lookup traced;
  F32/F16/BF16 tensor rules; metadata credentials.
- **PyTorch and other pickles** (raw, zip, legacy tar, numpy, 7z): the pickle stack and memo are
  simulated statically and imports checked against an allowlist. Nothing is unpickled.
- **Keras** (`.keras`, HDF5) and **ONNX:** `Lambda` layers, custom objects, external-data paths and
  custom operators; unquantized tensors get the tensor rules.

## D5. Verdicts and exit codes

- Each file gets one verdict (`aegis/verdict.py`):
  - **CRITICAL: do not load** for a code-execution payload or a file that loads code;
  - **SUSPICIOUS** for a statistical or structural anomaly;
  - **NO KNOWN FINDINGS** otherwise.
- Aegis never calls a file safe.
- Exit codes: `0` means no finding at or above `--fail-on` (default `medium`), `1` means findings,
  and `2` means a file could not be scanned.
- Output formats: text, JSON (versioned schema, `docs/schema/findings-v1.schema.json`) and
  SARIF 2.1.0. Findings carry MITRE ATLAS and OWASP references.

## D6. Disarm policy

- `aegis disarm` rounds away the lowest mantissa bits of every F32, F16 and BF16 tensor, which
  destroys payloads hidden there whether or not they were detected.
- Widths are clamped per dtype to a safe maximum (float32 16, float16 4, bfloat16 2), and a width
  that would reach the exponent is always refused. `--allow-unsafe` goes past the maximum with a
  warning.
- Disarm writes a new file and never modifies its input. Safetensors headers are rewritten
  compactly; `--strip-metadata` drops `__metadata__`. Quantized GGUF tensors are left unchanged
  and reported, because their low-order bits are the weights.
- The quality cost is measured, not assumed (docs/disarm_impact.md).

## D7. Coordinated disclosure

- Aegis never sends a report itself; `aegis report-bundle` prepares one for the user to review.
- A clearly malicious file goes to the platform first (for Hugging Face, the Report option and
  security@huggingface.co). Do not contact the uploader, and do not name the repository publicly
  until the platform has acted.
- A probably accidental problem (an insecure pickle, a leaked secret) goes privately to the owner.
  When unsure, treat it as malicious.
- Vulnerabilities in Aegis itself are reported through GitHub private vulnerability reporting
  (SECURITY.md).

## D8. Claims are backed by tests

- Every claim in the README comes from `docs/claims.json` and is proved by the tests marked
  `@pytest.mark.claim("<id>")`.
- Benchmark claims are checked against committed results and the rules fingerprint they were
  produced under (`benchmarks/fingerprint.py`). A claim whose benchmark is stale is shown as
  pending, not claimed.
- In CI (`AEGIS_REQUIRE_CLAIMS=1`), a skipped claim test is a failure.
- Known gaps and limits are stated as strict `xfail` tests and threat-matrix rows (README "Claims",
  docs/THREAT_MATRIX.md "Known gaps").
