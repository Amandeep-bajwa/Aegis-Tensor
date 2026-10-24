"""Credentials left in model metadata (an accidental leak, not an attack).

Reported so the owner can be told privately and the credential revoked (SECURITY.md).
Only a masked form of the secret ever appears in output.
"""

import re

PATTERNS = [
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b")),
    ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
]


def mask(secret: str) -> str:
    return secret[:6] + "…" + secret[-4:] if len(secret) > 14 else secret[:4] + "…"


def find_secrets(text: str) -> list[tuple[str, str]]:
    """[(kind, masked value)] for each credential-like string in ``text``."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            if match.group(0) not in seen and not any(match.group(0) in s for s in seen):
                seen.add(match.group(0))
                found.append((kind, mask(match.group(0))))
    return found
