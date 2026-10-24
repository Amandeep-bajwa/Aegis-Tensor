"""Which globals a checkpoint may import.

Before this, every global under `torch` and `numpy` was allowed by prefix, so a pickle that
called `torch.hub.load` or `numpy.testing` helpers that exec code got no finding at all.
Payloads here are built as opcodes and only scanned; nothing is unpickled.
"""

import collections
import warnings

import numpy as np
import pytest

from aegis import TORCH_AVAILABLE
from aegis.picklescan import scan_pickle_file

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _call(module: str, name: str) -> bytes:
    """Protocol-0 pickle that would call module.name('x') if it were ever loaded."""
    return f"c{module}\n{name}\n(S'x'\ntR.".encode()


def _scan(tmp_path, data: bytes):
    path = tmp_path / "m.pkl"
    path.write_bytes(data)
    return scan_pickle_file(path)


# Entry points that run code, load code or load another unrestricted pickle.
CODE_EXECUTION = [
    ("torch.hub", "load"),
    ("torch.hub", "load_state_dict_from_url"),
    ("torch.utils.cpp_extension", "load_inline"),
    ("torch.jit", "load"),
    ("torch._C", "_jit_pass_inline"),
    ("torch.ops", "load_library"),
    ("torch._ops", "load_library"),
    ("torch.package", "PackageImporter"),
    ("torch.distributed.rpc", "rpc_sync"),
    ("torch.multiprocessing", "spawn"),
    ("torch", "load"),
    ("torch.serialization", "load"),
    ("torch._dynamo", "optimize"),
    ("numpy.testing._private.utils", "runstring"),
    ("numpy.f2py", "compile"),
    ("numpy.distutils.exec_command", "exec_command"),
    ("numpy.ctypeslib", "load_library"),
    ("numpy", "load"),
    ("numpy.lib.npyio", "load"),
]


@pytest.mark.claim("C02")
@pytest.mark.parametrize("module,name", CODE_EXECUTION)
def test_torch_and_numpy_entry_points_that_run_code_are_critical(tmp_path, module, name):
    result = _scan(tmp_path, _call(module, name))
    assert result.is_critical, result.findings
    assert any(f.code == "dangerous_import" and f"{module}.{name}" in f.message for f in result.findings)


@pytest.mark.parametrize(
    "module,name", [("torch", "some_future_function"), ("torch.nn.init", "trunc_normal_"), ("numpy.lib", "something")]
)
def test_unlisted_torch_and_numpy_globals_are_reported_not_silently_allowed(tmp_path, module, name):
    result = _scan(tmp_path, _call(module, name))
    assert [f.code for f in result.findings] == ["unknown_import"]
    assert not result.is_critical


# --- Legitimate checkpoints must stay clean --------------------------------------------


def _checkpoints():
    linear = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(linear.parameters())
    linear(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int8, torch.uint8, torch.int16]
    dtypes += [torch.int32, torch.int64, torch.bool, torch.complex64, torch.complex128]
    dtypes += [getattr(torch, d) for d in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, d)]
    return {
        "state_dict": linear.state_dict(),
        "every_dtype": {str(d): torch.zeros(3).to(d) for d in dtypes},
        "mlp": nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.LayerNorm(8), nn.Dropout(0.1), nn.Linear(8, 2)),
        "cnn": nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.MaxPool2d(2), nn.Flatten(), nn.GELU()),
        "transformer": nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), 1),
        "rnn": nn.ModuleDict({"e": nn.Embedding(10, 4), "l": nn.LSTM(4, 4), "g": nn.GRU(4, 4)}),
        "training": {"model": linear.state_dict(), "optim": optimizer.state_dict(), "rng": torch.get_rng_state()},
        "sparse": {"coo": torch.eye(3).to_sparse(), "csr": torch.eye(3).to_sparse_csr()},
        "quantized": {"q": torch.quantize_per_tensor(torch.ones(4), 0.1, 0, torch.qint8)},
        "nested": {"n": torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])},
        "misc": {
            "param": nn.Parameter(torch.ones(2)),
            "meta": torch.empty(3, device="meta"),
            "size": torch.Size([2, 3]),
            "device": torch.device("cpu"),
            "dtype": torch.float16,
            "ordered": collections.OrderedDict(a=1),
            "numpy": np.ones(3, dtype=np.float32),
            "np_scalar": np.float32(1.5),
            "np_dtype": np.dtype("float64"),
        },
    }


@needs_torch
@pytest.mark.parametrize("serialization", ["zip", "legacy"])
def test_legitimate_torch_checkpoints_have_no_findings(tmp_path, serialization):
    warnings.filterwarnings("ignore")
    for name, obj in _checkpoints().items():
        path = tmp_path / f"{name}.pt"
        torch.save(obj, path, _use_new_zipfile_serialization=serialization == "zip")
        result = scan_pickle_file(path)
        assert result.findings == [], (name, result.findings)


def test_numpy_archives_have_no_findings(tmp_path):
    np.savez(tmp_path / "arrays.npz", a=np.ones(3), b=np.arange(4))
    assert scan_pickle_file(tmp_path / "arrays.npz").findings == []
