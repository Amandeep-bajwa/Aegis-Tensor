# 🛡️ Aegis-Tensor

**Aegis answers one question: is this model file safe to load and run?**

It reads safetensors, GGUF, PyTorch and other pickle checkpoints, numpy, Keras and ONNX files without
loading them. It reports what it finds as one verdict with a next step:

| Verdict | Meaning | Next step |
|---|---|---|
| **CRITICAL: do not load** | A code-execution payload, or a file that loads code | Quarantine it. Report it to the platform, not the uploader ([SECURITY.md](SECURITY.md)) |
| **SUSPICIOUS** | A statistical or structural anomaly | `aegis disarm`, or get the file from another source |
| **NO KNOWN FINDINGS** | Nothing Aegis knows how to detect | `aegis disarm` is still recommended (below) |

**Threat model.** The attacker is anyone who can publish or modify a model file (an uploader, a compromised
account or mirror, an insider) and knows all of Aegis's rules. The victim downloads and runs open-weight
models locally. Because the rules are public, a payload can be tuned to sit just under any threshold, so
detection is not the safety net. `aegis disarm` removes anything hidden in the low-order bits whether or not
it was detected. See [docs/THREAT_MATRIX.md](docs/THREAT_MATRIX.md).

---

## Claims

Every claim below is proved by the tests or the benchmark it links to, and CI fails if one of them fails.
The table is generated from [docs/claims.json](docs/claims.json) by `python scripts/claims.py --write`. A
capability that cannot be tested is not listed.

<!-- claims:start -->

| ID | Claim | Evidence |
|---|---|---|
| C01 | Scanning never runs code from the file being scanned. Every malicious test fixture would create a marker file if executed, and each test checks that the marker is still absent afterwards. | [`test_full_module_pickle_is_refused_without_trust`](tests/test_fuzz_safe_loading.py)<br>[`test_memo_assembled_os_system_is_resolved`](tests/test_pickle_bypasses.py)<br>[`test_renamed_extension_is_routed_by_magic_bytes`](tests/test_pickle_bypasses.py)<br>[`test_7z_wrapped_payload_is_inspected`](tests/test_pickle_bypasses.py)<br>[`test_broken_stream_after_payload`](tests/test_pickle_bypasses.py)<br>[`test_template_analysis_never_renders`](tests/test_template_scan.py)<br>[`test_t13_unsafe_pickle_is_never_executed_without_trust`](tests/test_threat_matrix.py) |
| C02 | Detects pickle imports of code-execution primitives (for example `os.system`, `builtins.eval`, `torch.hub.load`, `torch.utils.cpp_extension.load_inline`) in raw pickles, PyTorch zip and legacy tar checkpoints, numpy object arrays and 7z archives, whatever the file extension. | [`test_torch_and_numpy_entry_points_that_run_code_are_critical`](tests/test_pickle_allowlist.py)<br>[`test_protocol_0_global_is_detected`](tests/test_pickle_bypasses.py)<br>[`test_renamed_extension_is_routed_by_magic_bytes`](tests/test_pickle_bypasses.py)<br>[`test_7z_wrapped_payload_is_inspected`](tests/test_pickle_bypasses.py)<br>[`test_npy_object_array_payload_is_detected`](tests/test_pickle_bypasses.py)<br>[`test_payload_in_legacy_torch_object_pickle_is_found`](tests/test_pickle_bypasses.py)<br>[`test_payload_in_legacy_tar_checkpoint_is_found`](tests/test_pickle_bypasses.py)<br>[`test_t10_os_system_pickle_is_critical_and_pinpointed`](tests/test_threat_matrix.py)<br>[`test_t11_builtins_eval_pickle_is_critical`](tests/test_threat_matrix.py) |
| C03 | Resolves callables assembled from memoized strings (memo PUT/GET plus STACK_GLOBAL). It keeps scanning past malformed bytes and tampered zip entries, and reports every global it sees. | [`test_memo_assembled_os_system_is_resolved`](tests/test_pickle_bypasses.py)<br>[`test_long_binput_and_get_variants_are_resolved`](tests/test_pickle_bypasses.py)<br>[`test_unresolvable_stack_global_is_reported`](tests/test_pickle_bypasses.py)<br>[`test_payload_behind_junk_prefix_is_found_by_best_effort_scan`](tests/test_pickle_bypasses.py)<br>[`test_broken_stream_after_payload`](tests/test_pickle_bypasses.py)<br>[`test_every_global_is_reported_across_a_malformed_region`](tests/test_pickle_bypasses.py)<br>[`test_zip_with_encryption_flag_set_is_still_scanned`](tests/test_pickle_bypasses.py)<br>[`test_zip_with_bad_crc_is_still_scanned`](tests/test_pickle_bypasses.py)<br>[`test_zip_with_damaged_central_directory_falls_back_to_local_headers`](tests/test_pickle_bypasses.py)<br>[`test_t12_malformed_pickle_is_reported_not_skipped`](tests/test_threat_matrix.py) |
| C04 | Flags Jinja template injection in GGUF chat templates, including the bypasses an attacker who knows the rules would try. These are names built at render time; names hidden in variables, namespaces or macro arguments; names reached through `attr`, `map` or `str.format`; and names taken from `eos_token`. | [`test_gguf_template_injection_is_critical`](tests/test_model_formats.py)<br>[`test_template_injection_bypasses_are_flagged`](tests/test_template_scan.py)<br>[`test_obfuscated_gguf_template_is_critical`](tests/test_template_scan.py) |
| C05 | Flags float tensors whose values were overwritten, either whole or in about 1% of values, and NaN/Inf injection. This covers safetensors, GGUF, ONNX and Keras/HDF5 files. | [`test_gguf_overwritten_float_tensor_gets_tensor_rules`](tests/test_model_formats.py)<br>[`test_legacy_h5_model_config_and_weights_are_checked`](tests/test_model_formats.py)<br>[`test_onnx_external_paths_custom_ops_and_tensor_rules`](tests/test_model_formats.py)<br>[`test_t03_overwritten_tensor_is_flagged_and_quarantined`](tests/test_threat_matrix.py)<br>[`test_t04_partial_overwrite_of_one_percent_is_flagged`](tests/test_threat_matrix.py)<br>[`test_t17_nan_and_inf_injection_is_flagged`](tests/test_threat_matrix.py) |
| C06 | Flags structured data (text or code) hidden in the least significant mantissa bits. | [`test_t01_structured_lsb_payload_is_pinpointed_and_disarmed`](tests/test_threat_matrix.py) |
| C07 | Flags payloads in safetensors `__metadata__`, in header whitespace and in tensor names. | [`test_t05_payload_in_metadata_is_flagged_and_stripped`](tests/test_threat_matrix.py)<br>[`test_t06_whitespace_channel_in_header_is_flagged`](tests/test_threat_matrix.py)<br>[`test_t07_oversized_tensor_name_is_flagged_for_quarantine`](tests/test_threat_matrix.py) |
| C08 | Flags Keras `Lambda` layers and non-Keras custom objects. Flags ONNX external-data paths that leave the model directory, and ONNX custom operators. | [`test_keras_lambda_and_custom_objects_are_flagged`](tests/test_model_formats.py)<br>[`test_legacy_h5_model_config_and_weights_are_checked`](tests/test_model_formats.py)<br>[`test_onnx_external_paths_custom_ops_and_tensor_rules`](tests/test_model_formats.py) |
| C09 | `aegis disarm` removes any payload stored in the rounded-away low mantissa bits, whether or not a scan detected it. It refuses any width that would reach the exponent. | [`test_rounding_every_mantissa_bit_is_refused`](tests/test_disarm.py)<br>[`test_disarm_file_destroys_lsb_payload_and_keeps_metadata`](tests/test_disarm.py)<br>[`test_gguf_float_tensors_are_disarmed_and_everything_else_is_kept`](tests/test_disarm.py)<br>[`test_t02_random_lsb_payload_is_a_documented_blind_spot_and_disarm_removes_it`](tests/test_threat_matrix.py) |
| C10 | _Pending: not claimed until its benchmark has run under the current code (see [calibration.md](docs/calibration.md))._ | [calibration.md](docs/calibration.md)<br>[`test_calibration_claim_matches_current_results_and_rules`](tests/test_claims.py) |
| C11 | At its default width, `aegis disarm` changed ViT ImageNet top-1 by +0.00 points, BERT SST-2 accuracy by +0.00 points, GPT-2 WikiText perplexity by +0.00 (float32) and +0.10 (float16), and Qwen2.5-0.5B perplexity by +0.12 (bfloat16). | [disarm_impact.md](docs/disarm_impact.md)<br>[`test_disarm_claim_matches_current_results_and_code`](tests/test_claims.py) |
| C12 | `aegis fuzz` loads weights from safetensors into an architecture you supply, so no pickle runs. A pickled full module is refused unless you pass `--trust-pickle`. | [`test_backdoor_in_safetensors_is_found_without_pickle`](tests/test_fuzz_safe_loading.py)<br>[`test_full_module_pickle_is_refused_without_trust`](tests/test_fuzz_safe_loading.py)<br>[`test_t13_unsafe_pickle_is_never_executed_without_trust`](tests/test_threat_matrix.py) |
| C13 | With `--trust-pickle`, the module is unpickled in a subprocess with no network access. | [`test_trusted_pickle_is_fuzzed_in_the_sandbox`](tests/test_fuzz_safe_loading.py)<br>[`test_sandbox_blocks_network_at_the_os_level`](tests/test_fuzz_safe_loading.py) |
| C14 | On macOS, a `--trust-pickle` load also cannot write outside a temporary directory or start other programs. _(platform: darwin)_ | [`test_sandbox_confines_writes_and_blocks_shell`](tests/test_fuzz_safe_loading.py)<br>[`test_payload_cannot_act_on_the_host_even_when_trusted`](tests/test_fuzz_safe_loading.py) |
| C15 | `--format json` output validates against `docs/schema/findings-v1.schema.json`, and `--format sarif` output validates against the official SARIF 2.1.0 schema. | [`test_json_output_validates_against_schema`](tests/test_reporting.py)<br>[`test_sarif_output_validates_against_the_official_schema`](tests/test_reporting.py)<br>[`test_fuzz_findings_validate_against_the_finding_definition`](tests/test_reporting.py) |
| C16 | Exit codes: 0 means no finding at or above `--fail-on`, 1 means findings, and 2 means a file could not be scanned. | [`test_clean_model_exits_0`](tests/test_cli_exit_codes.py)<br>[`test_high_finding_exits_1_at_default`](tests/test_cli_exit_codes.py)<br>[`test_critical_pickle_exits_1_even_at_fail_on_critical`](tests/test_cli_exit_codes.py)<br>[`test_missing_file_exits_2`](tests/test_cli_exit_codes.py)<br>[`test_corrupt_safetensors_exits_2`](tests/test_cli_exit_codes.py) |
| C17 | Every scanned file gets one verdict with a next step. CRITICAL (do not load) means a code-execution payload or a file that loads code. SUSPICIOUS means an anomaly: disarm, or get the file elsewhere. NO KNOWN FINDINGS still recommends disarm. Each verdict says how to disclose a confirmed problem. | [`test_verdict_levels_and_next_steps`](tests/test_verdict_engines_bundle.py) |
| C18 | Flags credentials (API keys, tokens, private keys) left in safetensors or GGUF metadata, showing only a masked form. | [`test_leaked_token_in_metadata_is_reported_masked`](tests/test_verdict_engines_bundle.py)<br>[`test_leaked_key_in_gguf_metadata_is_reported`](tests/test_verdict_engines_bundle.py) |
| C19 | If Fickling or ModelScan is installed, Aegis runs it on the same file and folds its result into the Aegis verdict. A disagreement between Aegis and an engine is reported as a finding, in either direction. | [`test_engines_fold_into_one_verdict`](tests/test_verdict_engines_bundle.py)<br>[`test_renamed_file_is_still_given_to_modelscan`](tests/test_verdict_engines_bundle.py)<br>[`test_disagreement_is_a_finding_in_both_directions`](tests/test_verdict_engines_bundle.py) |
| C20 | `aegis report-bundle` writes a folder for you to review: findings, SHA-256 hashes, repository URL and revision. It never sends anything and never copies the model file. | [`test_report_bundle_packages_hashes_source_and_findings_without_the_model`](tests/test_verdict_engines_bundle.py) |
| L01 | **Known limit:** Random bits hidden in the low mantissa bits cannot be detected by any static statistic. Only `aegis disarm` removes them. | [`test_t02_random_lsb_payload_is_a_documented_blind_spot_and_disarm_removes_it`](tests/test_threat_matrix.py) |
| L02 | **Known limit:** Payloads split across tensors of fewer than 256 elements are not detected. | [`test_t19_payload_split_into_tiny_tensors_is_flagged`](tests/test_threat_matrix.py) |

<!-- claims:end -->

---

## Quick start

Requirements: Python 3.10+, [Rust](https://rustup.rs/) and [maturin](https://github.com/PyO3/maturin).

```bash
git clone https://github.com/KrishG7/Aegis-Tensor.git
cd Aegis-Tensor
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release          # builds the Rust core
pip install -e ".[engines]"        # optional: Fickling and ModelScan as extra engines
aegis doctor
```

## Scanning

```bash
aegis scan model.gguf                                  # text report and verdict
aegis scan a.safetensors b.pt --format json            # findings document (docs/schema/findings-v1.schema.json)
aegis scan model.safetensors --format sarif > aegis.sarif   # SARIF 2.1.0 for GitHub code scanning
```

The scanner is chosen from each file's leading bytes, never its extension. What each format gets:

- **safetensors**: tensor rules on every float tensor, plus header checks (oversized `__metadata__`,
  whitespace in the header, oversized tensor names, credentials).
- **GGUF**: chat templates are parsed with Jinja's parser, never rendered, and every attribute and item
  lookup is traced. F32/F16/BF16 tensors get the tensor rules. Metadata is checked for credentials.
- **PyTorch and other pickles** (raw, zip, legacy tar, numpy, 7z): the pickle stack and memo are simulated
  statically, and every imported global is checked against an allowlist. Nothing is unpickled.
- **Keras** (`.keras`, HDF5) and **ONNX**: `Lambda` layers, custom objects, external-data paths and custom
  operators are flagged, and unquantized tensors get the tensor rules.

The tensor rules flag three things:
- float values that were overwritten (a near-uniform sign/exponent byte lane, non-finite or implausibly
  large values);
- structured data in the least significant mantissa bits;
- whole-tensor entropy above `--entropy-threshold`.

The thresholds are in [`aegis/thresholds.py`](aegis/thresholds.py), and their false-positive rates on
public models are in [docs/calibration.md](docs/calibration.md).

**Optional engines.** If Fickling or ModelScan is installed, `aegis scan` runs it on the same file and folds
the result into the verdict. A disagreement with Aegis is itself a finding. Choose engines with
`--engines auto|none|fickling,modelscan`.

**Exit codes:**
- `0`: no finding at or above `--fail-on` (default `medium`);
- `1`: there are findings;
- `2`: a file could not be scanned.

A ready-to-copy pull-request workflow is in
[`examples/github-actions/aegis-model-scan.yml`](examples/github-actions/aegis-model-scan.yml).

## Disarming

```bash
aegis disarm model.safetensors              # writes model.disarmed.safetensors
aegis disarm model.gguf                     # F32/F16/BF16 tensors; quantized blocks are reported
```

Disarm rounds away the lowest mantissa bits of every F32, F16 and BF16 tensor. That destroys payloads hidden
there without needing to find them. It does not need PyTorch.

- **Widths:** `--low-bits` is clamped per dtype to a safe maximum (float32 16, float16 4, bfloat16 2), and a
  width that would reach the exponent is always refused.
- **safetensors:** the header is rewritten compactly, which removes any whitespace channel.
- **GGUF:** quantized tensors are left unchanged and reported, because their low-order bits *are* the
  weights.
- **Quality cost:** measured in [docs/disarm_impact.md](docs/disarm_impact.md).

## Reporting what you find

```bash
aegis report-bundle model.pkl     # writes ./aegis-report-model-<timestamp>/ for you to review
```

Aegis never reports anything itself. The bundle holds three files:
- a draft `REPORT.md` that says where the report should go;
- `manifest.json`, with file hashes, the repository URL and the revision;
- `findings.json`.

Clearly malicious files go to the platform first. Accidental problems, such as an insecure pickle format or
a leaked key, go privately to the owner. [SECURITY.md](SECURITY.md) has the full process.

## Backdoor fuzzing

```bash
aegis fuzz model.safetensors --hf-config bert-base-uncased                    # architecture from a HF config
aegis fuzz model.safetensors --arch mypkg.models:Net --input-shape 1,3,32,32    # your own class
aegis fuzz untrusted.pt --trust-pickle --input-shape 1,10                     # pickled module, sandboxed
```

Weights are loaded from safetensors into an architecture you trust, so no code from the file runs.
`--trust-pickle` unpickles inside a sandboxed subprocess. Residual risk is in
[docs/FUZZ_SAFETY.md](docs/FUZZ_SAFETY.md).

## Out of scope

Aegis checks files. Jailbreaks, prompt injection and other behaviour of a *running* chatbot or agent need a
red-teaming tool such as garak or PyRIT. Data poisoning needs the training pipeline.

---

## Architecture

| Part | What it does |
|---|---|
| Rust core (`src/lib.rs`, `aegis_core`) | Memory-maps safetensors files and computes the tensor statistics in parallel (rayon). Exposed to Python with PyO3 |
| `aegis/picklescan.py` | Static pickle and container analysis |
| `aegis/modelformats.py`, `aegis/templatescan.py` | GGUF, Keras, ONNX scanners and chat-template analysis |
| `aegis/disarm.py` | Bit rounding for safetensors and GGUF |
| `aegis/findings.py`, `aegis/verdict.py` | Findings schema, SARIF, verdicts, ATLAS and OWASP references |
| `aegis/engines.py`, `aegis/bundle.py` | Optional engines and report bundles |
| `aegis/fuzzer.py`, `aegis/modelload.py`, `aegis/sandbox.py` | Backdoor fuzzing and safe model loading |
| `benchmarks/` | Calibration on public models, disarm quality impact, backdoor detection |

Further documents:
- [Threat matrix](docs/THREAT_MATRIX.md)
- [Research compendium](docs/RESEARCH_COMPENDIUM.md)
- [Requirements](docs/REQUIREMENTS.md)
- [Architecture specification](docs/ARCHITECTURE.md)
- [Phases](docs/PHASES.md)
- [Team allocation](docs/TEAM_ALLOCATION.md)

## 👥 Engineering team

| Contributor | Focus Area | GitHub |
| :--- | :--- | :--- |
| **Krish Gupta** | Lead Architect, Rust Core & FFI Bridge | [@KrishG7](https://github.com/KrishG7) |
| **Himanshu Chhillar** | Cryptanalysis, Shannon Entropy & Benford's Law | [@HimanshuChhillar](https://github.com/HimanshuChhillar) |
| **Daksh Dahiya** | Dynamic Activation Fuzzing & Hook Instrumentation | [@7dxksh7](https://github.com/7dxksh7) |
| **Amandeep Singh** | CLI UX, Reporting, Benchmarking & CI/CD | [@Amandeep-bajwa](https://github.com/Amandeep-bajwa) |
| **Mukul Chauhan** | Project Presentation & Documentation Support | [@Mukul09800](https://github.com/Mukul09800) |

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
