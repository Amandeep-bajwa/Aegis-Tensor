# Security Policy

Aegis answers one question: **is this model file safe to load and run?** This document covers
two things. First, how to report a vulnerability in Aegis itself. Second, what to do when
Aegis flags a model you downloaded.

## Reporting a vulnerability in Aegis

Please report privately through GitHub's private vulnerability reporting:
**Security → Report a vulnerability** on
[KrishG7/Aegis-Tensor](https://github.com/KrishG7/Aegis-Tensor/security/advisories/new).
Do not open a public issue for a bypass until a fix is released.

Useful reports include:
- a model file that Aegis passes but that runs code or hides data when loaded;
- a crash or hang caused by a crafted file;
- a way for `aegis fuzz --trust-pickle` to escape its sandbox (see [docs/FUZZ_SAFETY.md](docs/FUZZ_SAFETY.md)).

Reproductions must be benign. For example, a payload should only create a marker file. We
aim to acknowledge reports within 7 days and to agree on a disclosure date with you. Only the
latest release on `main` receives fixes.

The rules are public, so we assume attackers know every threshold. A bypass that keeps a
payload just under a threshold is a valid report. It is also the reason `aegis disarm` is the
recommended treatment even when a scan finds nothing (see below).

## What a scan result means

| Verdict | Meaning | What to do |
|---|---|---|
| **CRITICAL: do not load** | A code-execution payload, or a file that loads code (for example a malicious pickle global, a Jinja injection in a GGUF chat template, or a Keras `Lambda` layer) | Do not load the file. Quarantine it. Report it to the platform (below). |
| **SUSPICIOUS** | A statistical anomaly or a structural oddity, such as overwritten tensors, structured low-order bits, an oversized header or a tampered archive | Run `aegis disarm` where the format supports it, or get the file from another source. Report it if it still looks deliberate. |
| **NO KNOWN FINDINGS** | Nothing Aegis knows how to detect was found | This does not mean the file is safe. Disarm is still recommended, because it removes low-order-bit payloads that no scanner can detect. |

## Reporting a model you found

Aegis never sends reports. False positives happen, and whether to report is your decision.
To prepare a report, run `aegis report-bundle`. It packages the findings, file hashes, the
repository URL and the revision into a folder you can review before submitting anything.

### 1. Clearly malicious: report to the platform first

Examples are a code-execution payload (verdict CRITICAL), or hidden data in weights or
metadata that is plainly deliberate.

- **Do not contact the uploader.** The owner may be the attacker, and a warning lets them
  re-upload elsewhere.
- **Hugging Face:** use the **Report** option in the repository's menu, and email
  `security@huggingface.co` with the report bundle.
- **Other hubs or mirrors:** use the platform's abuse or security contact.
- **Do not name the repository publicly until the platform has acted.** This includes issues,
  posts and discussion threads.

### 2. Probably accidental: contact the owner privately

Examples are an insecure but benign pickle checkpoint, or a secret (an API token or key)
left in the metadata.

- Contact the owner privately and suggest a fix, for example "publish a `.safetensors`
  version". Public pull requests and discussions on the Hub are visible to everyone, so do not
  put a leaked secret or exploit details there.
- For a leaked credential, also tell its issuer so it can be revoked. Hugging Face tokens can
  be reported to `security@huggingface.co`.

### 3. Unsure

Treat it as case 1. A platform security team can judge intent without warning a possible
attacker.
