"""Fingerprint of the code and defaults that decide a scan's findings.

Benchmark results record the fingerprint they were produced under. tests/test_claims.py
compares it with the current one, so a claim that rests on a benchmark fails once the rules
change, until the benchmark is re-run. Comments, docstrings and formatting are ignored.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What decides findings on safetensors and GGUF files: tensor rules (Rust), header rules,
# the GGUF parser, template analysis and the default thresholds.
SCAN_SOURCES = [
    "src/lib.rs",
    "aegis/formatscan.py",
    "aegis/modelformats.py",
    "aegis/templatescan.py",
    "aegis/thresholds.py",
]
DISARM_SOURCES = ["aegis/disarm.py"]


def _normalise_python(text: str) -> str:
    """Source without comments, docstrings or whitespace.

    Comment and docstring spans come from tokenize and are cut out of the text, which is
    the same on every Python version. (ast.dump and the token stream are not: f-strings
    tokenize differently from 3.12 on, and the fingerprint must match between CI and a
    developer's machine.)
    """
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    position = lambda row_col: offsets[row_col[0] - 1] + row_col[1]
    cuts, previous = [], tokenize.NEWLINE
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type != tokenize.NL]
    for index, token in enumerate(tokens):
        following = tokens[index + 1].type if index + 1 < len(tokens) else tokenize.ENDMARKER
        if token.type == tokenize.COMMENT:
            cuts.append((position(token.start), position(token.end)))
            continue
        if (
            token.type == tokenize.STRING
            and previous in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            and following in (tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENDMARKER)
        ):
            cuts.append((position(token.start), position(token.end)))
        previous = token.type
    kept, cursor = [], 0
    for start, end in cuts:
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    return re.sub(r"\s+", "", "".join(kept))


def _normalise_rust(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"\s+", "", text)


def fingerprint(sources: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sources:
        text = (ROOT / name).read_text()
        normalised = _normalise_python(text) if name.endswith(".py") else _normalise_rust(text)
        digest.update(name.encode() + b"\0" + normalised.encode() + b"\0")
    return digest.hexdigest()[:16]


def scan_fingerprint() -> str:
    return fingerprint(SCAN_SOURCES)


def disarm_fingerprint() -> str:
    return fingerprint(DISARM_SOURCES)


if __name__ == "__main__":
    print(json.dumps({"scan": scan_fingerprint(), "disarm": disarm_fingerprint()}))
