"""Run `aegis fuzz --trust-pickle` in an isolated child process.

Unpickling a model runs code from the file. The child process that does it gets:

* no network: ``sandbox-exec`` with ``(deny network*)`` on macOS, a fresh network
  namespace via ``unshare -rn`` on Linux. Without either, the run is refused.
* on macOS only: writes confined to a throwaway directory, and exec limited to the
  Python installation, so spawning ``/bin/sh`` fails.
* a minimal environment (no tokens or cloud credentials from the parent), with
  HOME and TMPDIR pointing at the throwaway directory.

Residual risk, documented in docs/FUZZ_SAFETY.md: the payload can still read files the
user can read, use CPU and memory until the timeout, write inside the throwaway
directory (Linux: anywhere the user can write), and, because it runs inside the
process that measures it, forge its own fuzz report.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from aegis.fuzzer import TrojanScanReport


class SandboxUnavailable(RuntimeError):
    """No supported isolation mechanism on this system."""


class SandboxError(RuntimeError):
    """The sandboxed fuzz run failed or produced no report."""


def _macos_profile(workdir: str) -> str:
    python_dirs = sorted({sys.prefix, sys.base_prefix, os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})
    exec_allowed = " ".join(f'(subpath "{d}")' for d in python_dirs)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/zero") (subpath "/dev/fd"))
(deny process-exec*)
(allow process-exec* {exec_allowed})
"""


def sandbox_prefix(workdir: str) -> list[str]:
    """Command prefix that isolates the child, or SandboxUnavailable."""
    # Tools are called by the absolute path that was found, never re-resolved through PATH.
    sandbox_exec = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if sandbox_exec:
        return [sandbox_exec, "-p", _macos_profile(workdir)]
    unshare = shutil.which("unshare") if sys.platform.startswith("linux") else None
    if unshare:
        # Fixed arguments; the probe only checks that unprivileged user namespaces work.
        probe = subprocess.run([unshare, "-rn", "true"], capture_output=True)  # noqa: S603 - fixed argv, no shell
        if probe.returncode == 0:
            return [unshare, "-rn"]
    raise SandboxUnavailable(
        "no sandbox available: need sandbox-exec (macOS) or unprivileged `unshare -rn` (Linux). "
        "Refusing to unpickle without network isolation."
    )


def _child_env(workdir: str) -> dict[str, str]:
    import aegis

    package_root = str(Path(aegis.__file__).resolve().parent.parent)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": package_root,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }


def run_sandboxed(request: dict[str, Any], timeout: float = 1800.0) -> dict[str, Any]:
    """Run the worker on ``request`` inside the sandbox and return its JSON response."""
    with tempfile.TemporaryDirectory(prefix="aegis-sandbox-") as tmp:
        workdir = os.path.realpath(tmp)
        request_path = Path(workdir) / "request.json"
        response_path = Path(workdir) / "response.json"
        request_path.write_text(json.dumps(request))
        command = [
            *sandbox_prefix(workdir),
            sys.executable,
            "-m",
            "aegis.sandbox_worker",
            str(request_path),
            str(response_path),
        ]
        try:
            # argv list without a shell: the only untrusted input is the model path, passed inside
            # request.json, and the child runs under the sandbox prefix above.
            proc = subprocess.run(  # noqa: S603
                command,
                env=_child_env(workdir),
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(f"sandboxed process exceeded {timeout:.0f}s and was killed") from None
        if not response_path.exists():
            raise SandboxError(f"sandboxed process exited {proc.returncode} without a report: {proc.stderr[-1500:]}")
        return json.loads(response_path.read_text())


def fuzz_pickled_model(model_path: Path, timeout: float = 1800.0, **params: Any) -> TrojanScanReport:
    """Unpickle and fuzz ``model_path`` in the sandbox. ``params`` go to modelload.fuzz_model."""
    response = run_sandboxed({"action": "fuzz", "model_path": str(Path(model_path).resolve()), **params}, timeout)
    if "error" in response:
        raise SandboxError(response["error"])
    return TrojanScanReport.from_dict(response["report"])
