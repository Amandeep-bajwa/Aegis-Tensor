"""Static scanner for pickle-based and other non-safetensors model files.

Loading a pickle runs whatever globals it imports, which is how malicious PyTorch
checkpoints execute code. This scanner never executes or unpickles anything. It:

* routes by magic bytes, not by file extension: raw pickle (protocols 0-5),
  PyTorch zip (``data.pkl``), legacy PyTorch tar and legacy concatenated
  pickles, 7z, numpy ``.npy``/``.npz``. Unknown formats are a finding.
* simulates the pickle stack and memo statically, so a callable assembled from
  memoized strings (MEMOIZE/PUT then GET then STACK_GLOBAL) is resolved and
  checked against the allowlist.
* continues past malformed opcodes and reports every global it sees, because
  broken pickles have been used to stop scanners before the payload (nullifAI).
* reads zip entries from their local headers, so tampered flag bits or bad CRCs
  are reported instead of crashing the scan.
"""

import ast
import io
import mmap
import pickletools
import re
import struct
import tarfile
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Modules whose import alone means arbitrary code execution or system access.
DANGEROUS_MODULES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "socket",
        "shutil",
        "pty",
        "runpy",
        "importlib",
        "ctypes",
        "multiprocessing",
        "pickle",
        "_pickle",
        "code",
        "codeop",
        "commands",
        "webbrowser",
        "tempfile",
        "http",
        "urllib",
        "requests",
        "asyncio",
    }
)
# Builtins that run code or reach into the interpreter.
DANGEROUS_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "open",
        "breakpoint",
        "input",
    }
)
# Modules that legitimate tensor checkpoints reference. Older pickle protocols
# name the builtins module `__builtin__`, so both spellings are treated alike.
BUILTIN_MODULES = frozenset({"builtins", "__builtin__"})
SAFE_MODULE_PREFIXES = ("torch", "collections", "numpy", "_codecs", "copyreg")

MAX_KEPT_STREAMS = 64
MAX_KEPT_STREAM_BYTES = 64 << 20
# First pickle of a legacy (pre-zip) torch.save file is this magic number.
LEGACY_TORCH_MAGIC = 0x1950A86A20F9469CFC6C
# A legacy file holds five pickles (magic, protocol, sys_info, object, storage keys)
# followed by raw storage bytes, which must not be parsed as opcodes.
LEGACY_TORCH_PICKLES = 5
MAX_NESTING = 3
MAX_RESYNCS = 10_000
MAX_7Z_UNPACKED_BYTES = 2 << 30
# Unknown files get a best-effort opcode scan of this many leading bytes.
BEST_EFFORT_BYTES = 1 << 20

STRING_OPS = frozenset(
    {
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "UNICODE",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
    }
)
# Only names that are valid dotted identifiers can be imported. Bytes read while
# resynchronising after a malformed region often look like GLOBAL opcodes with
# garbage operands; those cannot execute anything and are not reported.
_IMPORTABLE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$")
# Protocol 0 and 1 pickles have no PROTO header; they start with one of these opcodes.
_PROTO01_FIRST_BYTES = frozenset(op.code.encode("latin-1") for op in pickletools.opcodes if op.proto <= 1)


@dataclass
class PickleFinding:
    severity: str  # CRITICAL | SUSPICIOUS
    message: str
    location: str = ""
    # Stable machine code: dangerous_import, unknown_import, unresolved_import,
    # extension_registry, malformed_pickle, zip_tampered, no_payload, container_unreadable,
    # archive_7z, malformed_npy, npy_object, unknown_format. See aegis/findings.py.
    code: str = "unknown_format"


@dataclass
class PickleScanResult:
    path: str
    format: str = "unknown"
    findings: list[PickleFinding] = field(default_factory=list)
    globals_seen: list[str] = field(default_factory=list)
    # When a list, every pickle stream found is copied here as (location, bytes), for the
    # optional engines in aegis/engines.py. None (the default) keeps nothing.
    streams: list[tuple[str, bytes]] | None = field(default=None, repr=False, compare=False)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.findings)

    @property
    def is_critical(self) -> bool:
        return any(f.severity == "CRITICAL" for f in self.findings)

    def add(self, severity: str, message: str, location: str = "", code: str = "unknown_format") -> None:
        finding = PickleFinding(severity, message, location, code)
        if finding not in self.findings:
            self.findings.append(finding)


# --- Format detection ----------------------------------------------------------


def detect_format(data: bytes | mmap.mmap, total_size: int | None = None) -> str:
    """Identify a model container by its leading bytes. Never trusts the extension.

    ``total_size`` is the full file size when ``data`` is only the file's prefix.
    """
    head = bytes(data[:512])
    size = total_size if total_size is not None else len(data)
    if len(head) >= 9:
        (header_len,) = struct.unpack("<Q", head[:8])
        if 0 < header_len <= size - 8 and head[8:9] == b"{":
            return "safetensors"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if head[:6] == b"\x93NUMPY":
        return "npy"
    if len(head) > 262 and head[257:262] == b"ustar":
        return "tar"
    if head[:1] == b"\x80" and len(head) > 1 and 2 <= head[1] <= 5:  # PROTO is only emitted for 2+
        return "pickle"
    if head[:1] in _PROTO01_FIRST_BYTES and _parses_as_pickle(bytes(data[:65536])):
        return "pickle"
    return "unknown"


def _parses_as_pickle(data: bytes) -> bool:
    try:
        for opcode, _, _ in pickletools.genops(io.BytesIO(data)):
            if opcode.name == "STOP":
                return True
    except Exception:
        return False
    return False


# --- Global classification -------------------------------------------------------


def _classify(module: str, name: str) -> tuple[str, str]:
    """Return (severity, message) for an imported global, or ("", "") if it is allowed."""
    if module in DANGEROUS_MODULES or module.split(".")[0] in DANGEROUS_MODULES:
        return "CRITICAL", f"imports {module}.{name} (code execution or system access)"
    if module in BUILTIN_MODULES and name in DANGEROUS_BUILTINS:
        return "CRITICAL", f"calls builtins.{name} (runs code)"
    if module in BUILTIN_MODULES or module.startswith(SAFE_MODULE_PREFIXES):
        return "", ""
    return "SUSPICIOUS", f"imports unrecognised global {module}.{name}"


def _record(module: str, name: str, result: PickleScanResult, location: str) -> None:
    if not (_IMPORTABLE.match(module) and _IMPORTABLE.match(name)):
        return
    result.globals_seen.append(f"{module}.{name}")
    severity, message = _classify(module, name)
    if severity:
        result.add(severity, message, location, code="dangerous_import" if severity == "CRITICAL" else "unknown_import")


# --- Static pickle machine -------------------------------------------------------

_MARK = object()
_UNKNOWN = object()


def _pop(stack: list) -> object:
    return stack.pop() if stack else _UNKNOWN


def _pop_to_mark(stack: list) -> None:
    while stack:
        if stack.pop() is _MARK:
            return


def _step(op, arg, offset: int, stack: list, memo: dict, result: PickleScanResult, location: str) -> None:
    """Apply one opcode to the symbolic stack. Only strings and imports are tracked."""
    name = op.name
    where = f"{location}@{offset}"
    if name in STRING_OPS:
        stack.append(arg if isinstance(arg, str) else _UNKNOWN)
    elif name == "MARK":
        stack.append(_MARK)
    elif name == "MEMOIZE":
        memo[len(memo)] = stack[-1] if stack else _UNKNOWN
    elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
        memo[arg] = stack[-1] if stack else _UNKNOWN
    elif name in ("GET", "BINGET", "LONG_BINGET"):
        stack.append(memo.get(arg, _UNKNOWN))
    elif name == "DUP":
        stack.append(stack[-1] if stack else _UNKNOWN)
    elif name in ("GLOBAL", "INST"):
        module, _, attr = str(arg).partition(" ")
        _record(module, attr, result, where)
        if name == "INST":
            _pop_to_mark(stack)
        stack.append(_UNKNOWN)
    elif name == "STACK_GLOBAL":
        attr, module = _pop(stack), _pop(stack)
        if isinstance(module, str) and isinstance(attr, str):
            _record(module, attr, result, where)
        else:
            result.add(
                "SUSPICIOUS",
                "STACK_GLOBAL operands could not be resolved statically (obfuscated import)",
                where,
                code="unresolved_import",
            )
        stack.append(_UNKNOWN)
    elif name.startswith("EXT"):
        result.add(
            "SUSPICIOUS",
            f"{name} resolves a global through the copyreg extension registry",
            where,
            code="extension_registry",
        )
        stack.append(_UNKNOWN)
    else:
        before = op.stack_before
        if pickletools.markobject in before:
            _pop_to_mark(stack)
            for _ in range(before.index(pickletools.markobject)):
                _pop(stack)
        else:
            for _ in before:
                _pop(stack)
        for item in op.stack_after:
            stack.append(_MARK if item is pickletools.markobject else _UNKNOWN)


def _scan_pickle_stream(
    data: bytes | mmap.mmap,
    result: PickleScanResult,
    location: str,
    max_pickles: int | None = None,
    report_malformed: bool = True,
    best_effort: bool = False,
) -> int:
    """Scan concatenated pickles in ``data``. Returns the number of complete pickles.

    A malformed opcode is reported, then scanning resumes at the next byte, so a
    payload placed after (or before) a broken region is still seen. A memory map is
    read in place, so large legacy checkpoints are not copied into memory.

    ``best_effort`` scans bytes that are not known to be a pickle: it reports only
    importable globals, never malformed regions.
    """
    if best_effort:
        report_malformed = False
    stream = data if isinstance(data, mmap.mmap) else io.BytesIO(data)
    pos, complete, resyncs = 0, 0, 0
    while pos < len(data):
        if max_pickles is not None and complete >= max_pickles:
            break
        stack: list = []
        memo: dict = {}
        stream.seek(pos)
        last_offset = pos - 1
        try:
            finished = False
            for op, arg, offset in pickletools.genops(stream):
                last_offset = offset
                _step(op, arg, offset, stack, memo, result, location)
                if op.name == "STOP":
                    finished = True
                    break
            if not finished:
                break
            end = stream.tell()
            _keep_stream(result, f"{location}@{pos}", data, pos, end, best_effort)
            pos = end
            complete += 1
        except Exception as error:
            if resyncs == 0:
                _keep_stream(result, f"{location}@{pos}", data, pos, len(data), best_effort)
            if report_malformed:
                result.add(
                    "SUSPICIOUS",
                    f"malformed pickle stream ({str(error)[:80]}); scanning continued past it",
                    f"{location}@{last_offset + 1}",
                    code="malformed_pickle",
                )
            resyncs += 1
            if resyncs > MAX_RESYNCS:
                if not best_effort:
                    result.add(
                        "SUSPICIOUS",
                        "too many malformed regions; scan stopped early",
                        location,
                        code="malformed_pickle",
                    )
                break
            pos = max(pos + 1, last_offset + 1)
            report_malformed = False  # one report per stream; later errors are resync noise
    return complete


def _keep_stream(result: PickleScanResult, location: str, data, start: int, end: int, best_effort: bool) -> None:
    if result.streams is not None and not best_effort and len(result.streams) < MAX_KEPT_STREAMS:
        result.streams.append((location, bytes(data[start : min(end, start + MAX_KEPT_STREAM_BYTES)])))


def _is_legacy_torch(data: bytes) -> bool:
    try:
        for op, arg, _ in pickletools.genops(io.BytesIO(data[:64])):
            if op.name in ("LONG1", "LONG4", "LONG", "INT") and arg == LEGACY_TORCH_MAGIC:
                return True
            if op.name == "STOP":
                return False
    except Exception:
        return False
    return False


# --- Containers ----------------------------------------------------------------


def _read_local_entry(data, header_offset: int, method: int, compressed_size: int) -> bytes:
    """Read a zip entry from its local header, ignoring flag bits and CRC like miniz does."""
    header = bytes(data[header_offset : header_offset + 30])
    if header[:4] != b"PK\x03\x04":
        raise ValueError("local file header signature missing")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = header_offset + 30 + name_len + extra_len
    raw = bytes(data[start : start + compressed_size])
    if method == zipfile.ZIP_STORED:
        return raw
    if method == zipfile.ZIP_DEFLATED:
        return zlib.decompressobj(-15).decompress(raw)
    raise ValueError(f"unsupported compression method {method}")


def _zip_entries(data, result: PickleScanResult, label: str) -> Iterator[tuple[str, bytes]]:
    try:
        infos = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data).infolist()
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        result.add(
            "SUSPICIOUS",
            f"zip central directory unreadable ({error}); read local headers instead",
            label,
            code="zip_tampered",
        )
        yield from _zip_local_headers(data, result, label)
        return
    for info in infos:
        where = f"{label}:{info.filename}"
        try:
            payload = _read_local_entry(data, info.header_offset, info.compress_type, info.compress_size)
        except Exception as error:
            result.add("SUSPICIOUS", f"zip entry could not be read ({error})", where, code="zip_tampered")
            continue
        if info.flag_bits & 0x1:
            result.add(
                "SUSPICIOUS",
                "zip entry has the encryption flag set; PyTorch ignores it but Python scanners stop (evasion)",
                where,
                code="zip_tampered",
            )
        if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
            result.add(
                "SUSPICIOUS", "zip entry CRC does not match its contents (tampered archive)", where, code="zip_tampered"
            )
        yield info.filename, payload


def _zip_local_headers(data, result: PickleScanResult, label: str) -> Iterator[tuple[str, bytes]]:
    """Fallback when the central directory is damaged: walk local file headers."""
    blob = bytes(data)
    for match in re.finditer(rb"PK\x03\x04", blob):
        start = match.start()
        if start + 30 > len(blob):
            continue
        flags, method = struct.unpack("<HH", blob[start + 6 : start + 10])
        csize, _ = struct.unpack("<II", blob[start + 18 : start + 26])
        name_len, extra_len = struct.unpack("<HH", blob[start + 26 : start + 30])
        name = blob[start + 30 : start + 30 + name_len].decode("utf-8", "replace")
        body = start + 30 + name_len + extra_len
        if csize == 0 and flags & 0x8:
            nxt = re.search(rb"PK\x03\x04|PK\x01\x02", blob[body:])
            csize = nxt.start() if nxt else len(blob) - body
        try:
            yield name, _read_local_entry(blob, start, method, csize)
        except Exception as error:
            result.add("SUSPICIOUS", f"zip entry could not be read ({error})", f"{label}:{name}", code="zip_tampered")


def _scan_zip(data, result: PickleScanResult, label: str, depth: int) -> None:
    has_payload = False
    for name, payload in _zip_entries(data, result, label):
        where = f"{label}:{name}"
        lower = name.lower()
        if "/data/" in f"/{lower}" and not lower.endswith((".pkl", ".npy")):
            continue  # PyTorch tensor storage: raw bytes, never unpickled
        if lower.endswith(".pkl"):
            has_payload = True
            _scan_pickle_stream(payload, result, where)
        elif lower.endswith(".npy"):
            has_payload = True
            _scan_npy(payload, result, where)
        elif depth < MAX_NESTING:
            if _scan_bytes(payload, result, where, depth + 1, report_unknown=False):
                has_payload = True
    if not has_payload:
        result.add("SUSPICIOUS", "zip archive contains no pickle or numpy payload", label, code="no_payload")


def _scan_tar(data, result: PickleScanResult, label: str, depth: int) -> None:
    """Legacy torch.save tar: 'pickle' and 'sys_info' are pickles; 'storages' and
    'tensors' start with pickles followed by raw bytes."""
    try:
        archive = tarfile.open(fileobj=io.BytesIO(bytes(data)), mode="r:")
        members = [m for m in archive.getmembers() if m.isfile()]
    except (tarfile.TarError, OSError) as error:
        result.add("SUSPICIOUS", f"tar archive unreadable ({error})", label, code="container_unreadable")
        return
    for member in members:
        handle = archive.extractfile(member)
        if handle is None:
            continue
        payload = handle.read()  # read into memory; nothing is written to disk
        where = f"{label}:{member.name}"
        if member.name in ("pickle", "sys_info"):
            _scan_pickle_stream(payload, result, where)
        elif member.name in ("storages", "tensors"):
            _scan_pickle_stream(payload, result, where, max_pickles=1, report_malformed=False)
        elif depth < MAX_NESTING:
            _scan_bytes(payload, result, where, depth + 1, report_unknown=False)


def _scan_7z(data, result: PickleScanResult, label: str, depth: int) -> None:
    result.add(
        "SUSPICIOUS",
        "7z archive: PyTorch does not load 7z, so the file was repackaged (seen in nullifAI)",
        label,
        code="archive_7z",
    )
    try:
        import py7zr
    except ImportError:
        result.add("SUSPICIOUS", "7z contents not inspected: install py7zr to scan inside", label, code="archive_7z")
        return
    try:
        with py7zr.SevenZipFile(io.BytesIO(bytes(data)), mode="r") as archive:
            unpacked = sum(f.uncompressed or 0 for f in archive.list())
            if unpacked > MAX_7Z_UNPACKED_BYTES:
                result.add(
                    "SUSPICIOUS", f"7z archive unpacks to {unpacked} bytes; not inspected", label, code="archive_7z"
                )
                return
            entries = _read_7z(archive)
    except Exception as error:
        result.add("SUSPICIOUS", f"7z archive unreadable ({error})", label, code="container_unreadable")
        return
    for name, payload in entries.items():
        where = f"{label}:{name}"
        if name.lower().endswith(".pkl"):
            _scan_pickle_stream(payload, result, where)
        elif depth < MAX_NESTING:
            _scan_bytes(payload, result, where, depth + 1, report_unknown=False)


def _read_7z(archive) -> dict:
    """Read every 7z member into memory (py7zr's API differs across versions)."""
    if hasattr(archive, "readall"):
        return {name: handle.read() for name, handle in archive.readall().items()}
    import py7zr

    factory = py7zr.io.BytesIOFactory(limit=MAX_7Z_UNPACKED_BYTES)
    archive.extractall(factory=factory)
    return {name: buffer.read() for name, buffer in factory.products.items()}


def _scan_npy(data, result: PickleScanResult, label: str) -> None:
    """A .npy with object dtype carries a pickle after its header."""
    blob = bytes(data)
    try:
        major = blob[6]
        if major == 1:
            (header_len,) = struct.unpack("<H", blob[8:10])
            start = 10
        else:
            (header_len,) = struct.unpack("<I", blob[8:12])
            start = 12
        header = ast.literal_eval(blob[start : start + header_len].decode("latin-1"))
        descr = str(header.get("descr", ""))
    except Exception as error:
        result.add("SUSPICIOUS", f"malformed .npy header ({error})", label, code="malformed_npy")
        return
    if "O" in descr:
        result.add("SUSPICIOUS", "object-dtype numpy array: loading it requires unpickling", label, code="npy_object")
        _scan_pickle_stream(blob[start + header_len :], result, f"{label}[object data]")


def _scan_bytes(data, result: PickleScanResult, label: str, depth: int, report_unknown: bool = True) -> bool:
    """Route ``data`` by magic bytes. Returns True when a payload container was recognised."""
    fmt = detect_format(data)
    if depth == 0:
        result.format = fmt
    if fmt == "pickle":
        if _is_legacy_torch(bytes(data[:64])):
            if depth == 0:
                result.format = "pytorch-legacy"
            _scan_pickle_stream(data, result, label, max_pickles=LEGACY_TORCH_PICKLES)
        else:
            _scan_pickle_stream(data, result, label)
        return True
    if fmt == "zip":
        _scan_zip(data, result, label, depth)
        return True
    if fmt == "tar":
        _scan_tar(data, result, label, depth)
        return True
    if fmt == "7z":
        _scan_7z(data, result, label, depth)
        return True
    if fmt == "npy":
        _scan_npy(data, result, label)
        return True
    if fmt == "safetensors":
        return False  # handled by the safetensors scanner; contains no pickle
    if report_unknown:
        result.add(
            "SUSPICIOUS",
            "unrecognised file format: not a pickle, PyTorch zip or tar, numpy, 7z or safetensors file",
            label,
            code="unknown_format",
        )
        # The classifier is not the only gate: a malformed protocol-0 payload may not be
        # recognised as a pickle, so unknown bytes still get an opcode scan for imports.
        _scan_pickle_stream(bytes(data[:BEST_EFFORT_BYTES]), result, label, best_effort=True)
    return False


def detect_file_format(path: str | Path) -> str:
    """Format of the file at ``path``, read from its first 64 KiB and its size."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as handle:
        prefix = handle.read(65536)
    return detect_format(prefix, total_size=size) if prefix else "unknown"


def scan_pickle_file(path: str | Path, keep_streams: bool = False) -> PickleScanResult:
    """Scan a model file without loading it. The format is taken from its bytes.

    ``keep_streams`` copies each pickle stream into ``result.streams`` for the optional engines.
    """
    path = Path(path)
    result = PickleScanResult(path=str(path), streams=[] if keep_streams else None)
    with open(path, "rb") as handle:
        if path.stat().st_size == 0:
            result.add("SUSPICIOUS", "empty file", path.name, code="unknown_format")
            return result
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            _scan_bytes(data, result, path.name, depth=0)
    return result
