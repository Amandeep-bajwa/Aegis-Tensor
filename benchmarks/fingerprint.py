"""Fingerprint of the code and defaults that decide a scan's findings.

Benchmark results record the fingerprint they were produced under. tests/test_claims.py
compares it with the current one, so a claim that rests on a benchmark fails once the rules
change, until the benchmark is re-run. Comments, docstrings and formatting are ignored.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What decides findings on safetensors and GGUF files: tensor rules (Rust), header rules,
# the GGUF parser, template analysis and the default thresholds.
SCAN_SOURCES = ["src/lib.rs", "aegis/formatscan.py", "aegis/modelformats.py",
                "aegis/templatescan.py", "aegis/thresholds.py"]
DISARM_SOURCES = ["aegis/disarm.py"]


def _normalise_python(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False)


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
