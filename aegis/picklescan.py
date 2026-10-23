"""Static scanner for pickle-based model files (.pt, .pth, .bin, .pkl).

Loading a pickle runs whatever globals it imports, which is how malicious PyTorch
checkpoints execute code. This scanner reads the pickle opcodes with
``pickletools`` and never executes them. Every global the file imports is checked
against an allowlist. A malformed pickle is reported as a finding, not skipped,
because broken pickles can evade loaders that stop at the first error.
"""

import io
import pickletools
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

# Modules whose import alone means arbitrary code execution or system access.
DANGEROUS_MODULES = frozenset({
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "pty", "runpy",
    "importlib", "ctypes", "multiprocessing", "pickle", "_pickle", "code", "codeop",
    "commands", "webbrowser", "tempfile", "http", "urllib", "requests", "asyncio",
})
# Builtins that run code or reach into the interpreter.
DANGEROUS_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "open", "breakpoint", "input",
})
# Modules that legitimate tensor checkpoints reference. Older pickle protocols
# name the builtins module `__builtin__`, so both spellings are treated alike.
BUILTIN_MODULES = frozenset({"builtins", "__builtin__"})
SAFE_MODULE_PREFIXES = ("torch", "collections", "numpy", "_codecs", "copyreg")


@dataclass
class PickleFinding:
    severity: str  # CRITICAL | SUSPICIOUS
    message: str


@dataclass
class PickleScanResult:
    path: str
    findings: List[PickleFinding] = field(default_factory=list)
    globals_seen: List[str] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.findings)

    @property
    def is_critical(self) -> bool:
        return any(f.severity == "CRITICAL" for f in self.findings)


def _classify(module: str, name: str) -> Tuple[str, str]:
    """Return (severity, message) for an imported global, or ("", "") if it is allowed."""
    if module in DANGEROUS_MODULES or module.split(".")[0] in DANGEROUS_MODULES:
        return "CRITICAL", f"imports {module}.{name} (code execution or system access)"
    if module in BUILTIN_MODULES and name in DANGEROUS_BUILTINS:
        return "CRITICAL", f"calls builtins.{name} (runs code)"
    if module in BUILTIN_MODULES or module.startswith(SAFE_MODULE_PREFIXES):
        return "", ""
    return "SUSPICIOUS", f"imports unrecognised global {module}.{name}"


def _scan_pickle_bytes(data: bytes, result: PickleScanResult) -> None:
    recent_strings: List[str] = []
    try:
        for opcode, arg, _ in pickletools.genops(io.BytesIO(data)):
            name = opcode.name
            if name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINSTRING", "SHORT_BINSTRING"):
                recent_strings = (recent_strings + [str(arg)])[-2:]
            elif name == "GLOBAL":
                module, _, attr = str(arg).partition(" ")
                _record(module, attr, result)
            elif name == "STACK_GLOBAL" and len(recent_strings) == 2:
                _record(recent_strings[0], recent_strings[1], result)
            elif name == "STACK_GLOBAL":
                result.findings.append(PickleFinding(
                    "SUSPICIOUS", "STACK_GLOBAL without two preceding string arguments (obfuscated import)"
                ))
    except Exception as error:  # malformed opcode stream
        result.findings.append(PickleFinding(
            "SUSPICIOUS", f"malformed pickle stream ({error}); a broken pickle can evade loaders"
        ))


def _record(module: str, name: str, result: PickleScanResult) -> None:
    result.globals_seen.append(f"{module}.{name}")
    severity, message = _classify(module, name)
    if severity:
        result.findings.append(PickleFinding(severity, message))


def scan_pickle_file(path: Union[str, Path]) -> PickleScanResult:
    """Scan a pickle-based model file without loading it.

    PyTorch checkpoints are zip archives with the pickle in ``*data.pkl``. Older
    checkpoints are a bare pickle. Other archive layouts are reported as findings.
    """
    path = Path(path)
    result = PickleScanResult(path=str(path))
    raw = path.read_bytes()

    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [n for n in archive.namelist() if n.endswith("data.pkl") or n.endswith(".pkl")]
            if not members:
                result.findings.append(PickleFinding("SUSPICIOUS", "zip archive contains no pickle payload"))
            for member in members:
                _scan_pickle_bytes(archive.read(member), result)
    else:
        _scan_pickle_bytes(raw, result)
    return result
