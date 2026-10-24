"""Optional engines: other scanners folded into one Aegis verdict.

If Fickling or ModelScan is installed, ``aegis scan`` runs it on the same file and turns
what it reports into Aegis findings (AEGIS-E001). When an engine and Aegis disagree about
how dangerous a file is, that is a finding too (AEGIS-E002), so a miss by either side is
visible. The user never has to run, or read, another tool.

Both engines are static: neither unpickles or imports the model. They still run as
separate processes with a timeout, from an empty working directory (ModelScan reads a
settings file from the working directory, which must not be one a download put there).

  * Fickling reads raw pickle streams. Aegis extracts every stream it found (including
    those inside zip, tar, 7z and numpy containers, and zips with tampered headers) and
    passes each one on stdin, so Fickling sees what PyTorch would unpickle. Imports are
    judged against Fickling's ML allowlist, not its standard-library-only default.
  * ModelScan reads files and chooses its scanner by file extension. It is given a link to
    the file named with the extension that matches the format Aegis detected, so a
    renamed file is not skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ENGINES = ("fickling", "modelscan")
TIMEOUT_SECONDS = 300
MAX_STREAMS = 64

_FICKLING_SEVERITY = {
    "OVERTLY_MALICIOUS": "critical",
    "LIKELY_OVERTLY_MALICIOUS": "critical",
    # Fickling gives every ordinary PyTorch checkpoint this level (calls to torch's tensor
    # rebuild functions), so it is folded in as low: visible, but it does not change the verdict.
    "LIKELY_UNSAFE": "low",
    "SUSPICIOUS": "medium",
    "POSSIBLY_UNSAFE": "low",
    "LIKELY_SAFE": None,
}
_MODELSCAN_SEVERITY = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
# Extension ModelScan needs for each format Aegis detects.
_MODELSCAN_EXTENSION = {
    "pickle": ".pkl",
    "pytorch-legacy": ".pt",
    "zip": ".pt",
    "tar": ".pt",
    "npy": ".npy",
    "hdf5": ".h5",
    "keras": ".keras",
}
_LEVELS = ("low", "medium", "high", "critical")


@dataclass
class EngineIssue:
    severity: str
    message: str
    location: str = ""


@dataclass
class EngineResult:
    engine: str
    version: str | None = None
    ran: bool = False
    issues: list[EngineIssue] = field(default_factory=list)
    error: str | None = None

    @property
    def worst(self) -> str | None:
        return max((i.severity for i in self.issues), key=_LEVELS.index, default=None)


def available() -> dict[str, str | None]:
    """Installed engines and their versions."""
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str | None] = {}
    for name in ENGINES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue  # not installed
    return found


def select(requested: str) -> list[str]:
    """``auto`` = every installed engine; ``none``; or a comma-separated list (must be installed)."""
    requested = requested.strip().lower()
    if requested == "none":
        return []
    installed = available()
    if requested == "auto":
        return [name for name in ENGINES if name in installed]
    names = [n.strip() for n in requested.split(",") if n.strip()]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise ValueError(f"unknown engine(s): {', '.join(unknown)} (choose from {', '.join(ENGINES)})")
    missing = [n for n in names if n not in installed]
    if missing:
        raise ValueError(f"engine(s) not installed: {', '.join(missing)} (pip install {' '.join(missing)})")
    return names


def _console_script(name: str) -> list[str]:
    """The engine's command, preferring the one installed next to this interpreter."""
    local = Path(sys.executable).parent / name
    if local.exists():
        return [str(local)]
    found = shutil.which(name)
    if found:
        return [found]
    raise FileNotFoundError(name)


def _run(command: Sequence[str], stdin: bytes | None = None, capture: bool = False):
    with tempfile.TemporaryDirectory(prefix="aegis-engine-") as workdir:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "MODELSCAN"))}
        # argv list without a shell; the command is an engine executable resolved by _console_script,
        # and the scanned file is passed as a path argument, never interpreted here.
        completed = subprocess.run(  # noqa: S603
            list(command), input=stdin, capture_output=True, cwd=workdir, env=env, timeout=TIMEOUT_SECONDS
        )
    stderr = completed.stderr.decode("utf-8", "replace")[-400:]
    if capture:
        return completed.returncode, completed.stdout.decode("utf-8", "replace"), stderr
    return completed.returncode, stderr


# --- Fickling ------------------------------------------------------------------------

# Fickling's default analysis treats every import outside the standard library as unsafe,
# so an ordinary PyTorch checkpoint (torch._utils._rebuild_tensor_v2) rates LIKELY_UNSAFE.
# This driver keeps every other analysis and judges imports against Fickling's own ML
# allowlist instead. It only parses: Pickled.load builds an AST, it does not unpickle.
_FICKLING_DRIVER = """
import json, sys
from fickling.analysis import Analysis, Analyzer, NonStandardImports, check_safety
from fickling.fickle import Pickled
from fickling.ml import MLAllowlist
pickled = Pickled.load(sys.stdin.buffer.read(), fail_on_decode_error=False)
passes = [a for a in Analysis.ALL if not isinstance(a, NonStandardImports)] + [MLAllowlist()]
results = check_safety(pickled, analyzer=Analyzer(passes))
json.dump({"severity": results.severity.name, "analysis": str(results)}, sys.stdout)
"""


def run_fickling(streams: list[tuple[str, bytes]]) -> EngineResult:
    result = EngineResult("fickling", available().get("fickling"))
    if not streams:
        return result  # nothing pickled: Fickling has no opinion on this file
    for location, data in streams[:MAX_STREAMS]:
        try:
            code, stdout, stderr = _run([sys.executable, "-I", "-c", _FICKLING_DRIVER], stdin=data, capture=True)
            report = json.loads(stdout)
        except subprocess.TimeoutExpired:
            result.error = f"timed out on {location}"
            continue
        except ValueError:
            result.error = f"no result for {location} (exit {code}): {stderr.strip()[:160]}"
            continue
        result.ran = True
        severity = _FICKLING_SEVERITY.get(str(report.get("severity")), "medium")
        if severity:
            analysis = " ".join(str(report.get("analysis", "")).split())[:300]
            result.issues.append(EngineIssue(severity, f"Fickling: {report.get('severity')}: {analysis}", location))
    if len(streams) > MAX_STREAMS:
        result.error = f"only the first {MAX_STREAMS} of {len(streams)} pickle streams were checked"
    return result


# --- ModelScan -----------------------------------------------------------------------


def run_modelscan(path: Path, detected_format: str) -> EngineResult:
    result = EngineResult("modelscan", available().get("modelscan"))
    extension = _MODELSCAN_EXTENSION.get(detected_format)
    if extension is None:
        return result  # a format ModelScan does not read
    try:
        command = _console_script("modelscan")
    except FileNotFoundError:
        result.error = "modelscan command not found"
        return result
    with tempfile.TemporaryDirectory(prefix="aegis-modelscan-") as work:
        link = Path(work) / f"model{extension}"
        os.symlink(Path(path).resolve(), link)
        out = Path(work) / "report.json"
        try:
            code, stderr = _run([*command, "scan", "-p", str(link), "-r", "json", "-o", str(out)])
        except subprocess.TimeoutExpired:
            result.error = "timed out"
            return result
        if not out.exists():
            result.error = f"no report (exit {code}): {stderr.strip()[:160]}"
            return result
        report = json.loads(out.read_text())
    scanned = report.get("summary", {}).get("scanned", {}).get("total_scanned", 0)
    result.ran = bool(scanned)
    for issue in report.get("issues", []):
        severity = _MODELSCAN_SEVERITY.get(str(issue.get("severity", "")).upper(), "medium")
        where = str(issue.get("source", "")).replace(link.name, Path(path).name, 1)
        result.issues.append(EngineIssue(severity, f"ModelScan: {issue.get('description', 'issue')}", where))
    errors = report.get("errors") or []
    if errors:
        result.error = "; ".join(str(e.get("description", e)) if isinstance(e, dict) else str(e) for e in errors)[:300]
    return result


def run_engines(
    names: list[str], path: Path, detected_format: str, streams: list[tuple[str, bytes]]
) -> list[EngineResult]:
    results = []
    for name in names:
        try:
            results.append(run_fickling(streams) if name == "fickling" else run_modelscan(path, detected_format))
        except Exception as error:  # an engine failure must never break the Aegis scan
            results.append(EngineResult(name, available().get(name), error=f"{type(error).__name__}: {error}"))
    return results
