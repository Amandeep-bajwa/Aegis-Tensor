"""Tests for the static pickle scanner. Malicious payloads are only ever scanned, never loaded."""

import os
import pickle

import pytest

from aegis import TORCH_AVAILABLE
from aegis.picklescan import scan_pickle_file

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class _SystemCall:
    """Pickles as a call to os.system. Harmless here: the bytes are scanned, never loaded."""

    def __reduce__(self):
        return (os.system, ("echo aegis-test-payload",))


@needs_torch
def test_clean_state_dict_has_no_findings(tmp_path):
    path = tmp_path / "clean.pt"
    torch.save(nn.Linear(4, 2).state_dict(), path)
    result = scan_pickle_file(path)
    assert result.globals_seen  # torch rebuild helpers were seen
    assert not result.is_suspicious


def test_os_system_import_is_critical(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"w": _SystemCall()}))
    result = scan_pickle_file(path)
    assert result.is_critical
    assert any("system" in f.message for f in result.findings)  # pickles as posix.system on macOS


def test_builtins_eval_is_critical(tmp_path):
    path = tmp_path / "eval.pkl"
    path.write_bytes(b"\x80\x02cbuiltins\neval\nq\x00(X\x0c\x00\x00\x00print(1)q\x01tq\x02Rq\x03.")
    result = scan_pickle_file(path)
    assert result.is_critical
    assert any("builtins.eval" in f.message for f in result.findings)


def test_unknown_global_is_suspicious(tmp_path):
    path = tmp_path / "unknown.pkl"
    path.write_bytes(b"\x80\x02cmystery_pkg.thing\nWidget\nq\x00.")
    result = scan_pickle_file(path)
    assert result.is_suspicious
    assert not result.is_critical


def test_truncated_pickle_is_reported_not_skipped(tmp_path):
    path = tmp_path / "broken.pkl"
    clean = pickle.dumps({"w": [1.0, 2.0, 3.0]})
    path.write_bytes(clean[: len(clean) // 2])
    result = scan_pickle_file(path)
    assert result.is_suspicious
    assert any("malformed" in f.message for f in result.findings)


def test_python2_builtins_spelling_is_treated_as_builtins(tmp_path):
    safe = tmp_path / "set.pkl"
    safe.write_bytes(b"\x80\x02c__builtin__\nset\nq\x00]q\x01\x85q\x02Rq\x03.")
    assert not scan_pickle_file(safe).is_suspicious

    evil = tmp_path / "eval2.pkl"
    evil.write_bytes(b"\x80\x02c__builtin__\neval\nq\x00(X\x0c\x00\x00\x00print(1)q\x01tq\x02Rq\x03.")
    assert scan_pickle_file(evil).is_critical
