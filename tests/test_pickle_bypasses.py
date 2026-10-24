"""Pickle-scanner bypass classes. Each payload is a real, loadable pickle whose only
effect would be `touch <tmp>/MARKER`. Every test asserts the marker never appears,
proving the scanner never executes or unpickles anything."""

import io
import os
import pickle
import struct
import zipfile

import numpy as np
import pytest
from typer.testing import CliRunner

from aegis import TORCH_AVAILABLE
from aegis.cli import app
from aegis.picklescan import detect_file_format, scan_pickle_file

if TORCH_AVAILABLE:
    import torch

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
runner = CliRunner()


def _u(text: str) -> bytes:
    """SHORT_BINUNICODE opcode for ``text``."""
    raw = text.encode()
    return b"\x8c" + bytes([len(raw)]) + raw


def memo_assembled_system(marker) -> bytes:
    """os.system assembled from memoized strings, with decoy strings pushed in between.

    A scanner that only looks at the last two string literals before STACK_GLOBAL
    sees 'torch' and 'nn' and passes the file.
    """
    return (
        b"\x80\x04"
        + _u("os")
        + b"\x94"
        + _u("system")
        + b"\x94"  # MEMOIZE -> memo 0, 1
        + _u("torch")
        + b"\x94"
        + _u("nn")
        + b"\x94"  # decoys
        + b"00"  # POP, POP
        + b"h\x00"
        + b"h\x01"
        + b"\x93"  # BINGET 0, BINGET 1, STACK_GLOBAL
        + _u(f"touch {marker}")
        + b"\x85"
        + b"R"
        + b"."  # TUPLE1, REDUCE, STOP
    )


def eval_global(code: str = "1") -> bytes:
    return b"\x80\x02cbuiltins\neval\n" + _u(code) + b"\x85R."


@pytest.fixture
def marker(tmp_path):
    path = tmp_path / "MARKER"
    yield path
    assert not path.exists(), "a payload executed during scanning"


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


# --- (b) stack and memo simulation ---------------------------------------------


@pytest.mark.claim("C01", "C03")
def test_memo_assembled_os_system_is_resolved(tmp_path, marker):
    result = scan_pickle_file(_write(tmp_path / "m.pkl", memo_assembled_system(marker)))
    assert result.is_critical
    assert "os.system" in result.globals_seen
    assert "torch.nn" not in result.globals_seen  # decoys were not mistaken for the import
    critical = next(f for f in result.findings if f.severity == "CRITICAL")
    assert critical.location.startswith("m.pkl@")


@pytest.mark.claim("C03")
def test_long_binput_and_get_variants_are_resolved(tmp_path, marker):
    # PUT/GET (protocol 0 text), BINPUT/BINGET and LONG_BINPUT/LONG_BINGET memo opcodes.
    payload = (
        b"\x80\x04"
        + _u("subprocess")
        + b"p7\n"  # PUT 7
        + _u("check_call")
        + b"r\x00\x01\x00\x00"  # LONG_BINPUT 256
        + b"0"
        + b"0"
        + b"g7\n"
        + b"j\x00\x01\x00\x00"
        + b"\x93"  # GET 7, LONG_BINGET 256, STACK_GLOBAL
        + b"N"
        + b"."
    )
    result = scan_pickle_file(_write(tmp_path / "m.pkl", payload))
    assert "subprocess.check_call" in result.globals_seen
    assert result.is_critical


@pytest.mark.claim("C03")
def test_unresolvable_stack_global_is_reported(tmp_path, marker):
    payload = b"\x80\x04" + b"K\x01" + b"K\x02" + b"\x93" + b"N."  # ints, not strings
    result = scan_pickle_file(_write(tmp_path / "m.pkl", payload))
    assert any("could not be resolved" in f.message for f in result.findings)


@pytest.mark.claim("C02")
def test_protocol_0_global_is_detected(tmp_path, marker):
    class _Exec:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    path = _write(tmp_path / "p0.pkl", pickle.dumps(_Exec(), protocol=0))
    assert detect_file_format(path) == "pickle"
    assert scan_pickle_file(path).is_critical


# --- (a) routing by magic bytes -----------------------------------------------


@pytest.mark.claim("C01", "C02")
def test_renamed_extension_is_routed_by_magic_bytes(tmp_path, marker):
    path = _write(tmp_path / "model.safetensors", memo_assembled_system(marker))
    assert detect_file_format(path) == "pickle"
    result = runner.invoke(app, ["scan", str(path)])
    assert result.exit_code == 1, result.output
    assert "os.system" in result.output


@pytest.mark.claim("C01", "C02")
def test_7z_wrapped_payload_is_inspected(tmp_path, marker):
    py7zr = pytest.importorskip("py7zr")
    path = tmp_path / "model.pt"
    with py7zr.SevenZipFile(path, "w") as archive:
        archive.writestr(memo_assembled_system(marker), "archive/data.pkl")
    result = scan_pickle_file(path)
    assert result.format == "7z"
    assert any("7z archive" in f.message for f in result.findings)
    assert result.is_critical
    assert any(f.location.endswith("archive/data.pkl@35") for f in result.findings if f.severity == "CRITICAL")


@pytest.mark.claim("C02")
def test_npy_object_array_payload_is_detected(tmp_path, marker):
    class _Exec:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    path = tmp_path / "weights.npy"
    np.save(path, np.array([_Exec()], dtype=object), allow_pickle=True)
    result = scan_pickle_file(path)
    assert result.format == "npy"
    assert result.is_critical
    assert any("object-dtype" in f.message for f in result.findings)


def test_clean_npz_has_no_findings(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, a=np.ones(4), b=np.zeros((2, 2)))
    assert not scan_pickle_file(path).is_suspicious


def test_unknown_format_is_a_finding(tmp_path):
    # 0x00 is not a pickle opcode, so this is deterministically not a pickle.
    path = _write(tmp_path / "model.bin", b"\x00" * 16 + bytes(range(256)) * 16)
    result = scan_pickle_file(path)
    assert result.format == "unknown"
    assert any("unrecognised file format" in f.message for f in result.findings)
    assert not result.is_critical


@pytest.mark.claim("C03")
def test_payload_behind_junk_prefix_is_found_by_best_effort_scan(tmp_path, marker):
    path = _write(tmp_path / "model.bin", b"\x00\x00\x00" + eval_global())
    result = scan_pickle_file(path)
    assert result.format == "unknown"
    assert result.is_critical
    assert "builtins.eval" in result.globals_seen


# --- (c) continue past malformed opcodes --------------------------------------


@pytest.mark.claim("C01", "C03")
def test_broken_stream_after_payload(tmp_path, marker):
    payload = memo_assembled_system(marker)[:-1] + b"\xff\xfe broken tail"
    result = scan_pickle_file(_write(tmp_path / "broken.pkl", payload))
    assert result.is_critical
    assert any("malformed" in f.message for f in result.findings)


@pytest.mark.claim("C03")
def test_every_global_is_reported_across_a_malformed_region(tmp_path, marker):
    payload = memo_assembled_system(marker) + b"\xff\xfe\xfd" + eval_global()
    result = scan_pickle_file(_write(tmp_path / "two.pkl", payload))
    assert "os.system" in result.globals_seen
    assert "builtins.eval" in result.globals_seen


@needs_torch
def test_legacy_torch_format_is_clean_and_its_storage_tail_is_not_parsed(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"w": torch.randn(64, 64)}, path, _use_new_zipfile_serialization=False)
    result = scan_pickle_file(path)
    assert result.format == "pytorch-legacy"
    assert not result.is_suspicious, result.findings


@pytest.mark.claim("C02")
def test_payload_in_legacy_torch_object_pickle_is_found(tmp_path, marker):
    magic = pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2)
    path = _write(
        tmp_path / "legacy.pt",
        magic
        + pickle.dumps(1001, protocol=2)
        + pickle.dumps({}, protocol=2)
        + memo_assembled_system(marker)
        + pickle.dumps([], protocol=2),
    )
    result = scan_pickle_file(path)
    assert result.format == "pytorch-legacy"
    assert any(f.code == "dangerous_import" and "os.system" in f.message for f in result.findings), result.findings


@pytest.mark.claim("C02")
def test_payload_in_legacy_tar_checkpoint_is_found(tmp_path, marker):
    import tarfile

    path = tmp_path / "legacy_tar.pt"
    with tarfile.open(path, "w") as archive:
        for name, data in (("sys_info", pickle.dumps({})), ("pickle", memo_assembled_system(marker))):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    result = scan_pickle_file(path)
    assert result.format == "tar"
    assert any(f.code == "dangerous_import" for f in result.findings), result.findings


# --- (d) tampered zip entries -------------------------------------------------


def _torch_style_zip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", "3\n")
    return buffer.getvalue()


def _patch_all(blob: bytes, signature: bytes, offset: int, fmt: str, transform) -> bytes:
    data = bytearray(blob)
    start = 0
    while (index := data.find(signature, start)) != -1:
        (value,) = struct.unpack_from(fmt, data, index + offset)
        struct.pack_into(fmt, data, index + offset, transform(value))
        start = index + 4
    return bytes(data)


@pytest.mark.claim("C03")
def test_zip_with_encryption_flag_set_is_still_scanned(tmp_path, marker):
    blob = _torch_style_zip(memo_assembled_system(marker))
    blob = _patch_all(blob, b"PK\x03\x04", 6, "<H", lambda v: v | 0x1)  # local headers
    blob = _patch_all(blob, b"PK\x01\x02", 8, "<H", lambda v: v | 0x1)  # central directory
    with pytest.raises(RuntimeError):  # what stops a naive zipfile-based scanner
        zipfile.ZipFile(io.BytesIO(blob)).read("archive/data.pkl")
    result = scan_pickle_file(_write(tmp_path / "flagged.pt", blob))
    assert result.is_critical
    assert any("encryption flag" in f.message for f in result.findings)


@pytest.mark.claim("C03")
def test_zip_with_bad_crc_is_still_scanned(tmp_path, marker):
    blob = _torch_style_zip(memo_assembled_system(marker))
    blob = _patch_all(blob, b"PK\x01\x02", 16, "<I", lambda v: v ^ 0xDEADBEEF)
    blob = _patch_all(blob, b"PK\x03\x04", 14, "<I", lambda v: v ^ 0xDEADBEEF)
    result = scan_pickle_file(_write(tmp_path / "crc.pt", blob))
    assert result.is_critical
    assert any("CRC" in f.message for f in result.findings)


@pytest.mark.claim("C03")
def test_zip_with_damaged_central_directory_falls_back_to_local_headers(tmp_path, marker):
    blob = _torch_style_zip(memo_assembled_system(marker))
    cut = blob.rfind(b"PK\x05\x06")
    blob = blob[:cut] + b"\x00" * (len(blob) - cut)  # destroy the end-of-central-directory record
    result = scan_pickle_file(_write(tmp_path / "nocd.pt", blob))
    assert result.is_critical
    assert any("central directory unreadable" in f.message for f in result.findings)


@needs_torch
def test_real_torch_checkpoint_is_clean(tmp_path):
    path = tmp_path / "clean.pt"
    torch.save(torch.nn.Linear(8, 4).state_dict(), path)
    result = scan_pickle_file(path)
    assert result.format == "zip"
    assert not result.is_suspicious, result.findings
