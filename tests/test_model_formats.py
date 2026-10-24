"""GGUF, Keras (.keras and HDF5) and ONNX scanners. Fixtures are built byte by byte or as
plain JSON; no framework loads any model."""

import json
import struct
import zipfile

import numpy as np
import pytest
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE
from aegis.cli import app
from aegis.modelformats import sniff_format

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
runner = CliRunner()

CLEAN_TEMPLATE = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
SSTI_TEMPLATE = "{{ ''.__class__.__mro__[1].__subclasses__()[133].__init__.__globals__['popen']('id').read() }}"


def _weights(seed=0, n=1 << 16):
    return np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)


def _blob(seed=1, n=1 << 16):
    return np.random.default_rng(seed).integers(0, 256, 4 * n, dtype=np.uint8).tobytes()


def _scan_json(path):
    result = runner.invoke(app, ["scan", str(path), "--format", "json"])
    return result.exit_code, json.loads(result.stdout)["files"][0]


# --- GGUF ----------------------------------------------------------------------


def gguf_bytes(metadata, tensors, alignment=32) -> bytes:
    """GGUF v3. metadata: [(key, value)] with str or int values; tensors: [(name, dims, ggml_type, raw)]."""

    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata:
        out += s(key)
        out += struct.pack("<I", 8) + s(value) if isinstance(value, str) else struct.pack("<II", 4, value)
    data = bytearray()
    for name, dims, ggml_type, raw in tensors:
        data += b"\0" * ((-len(data)) % alignment)
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ggml_type, len(data))
        data += raw
    out += b"\0" * ((-len(out)) % alignment)
    return bytes(out + data)


@needs_core
def test_gguf_clean_model_has_no_findings(tmp_path):
    path = tmp_path / "clean.gguf"
    path.write_bytes(
        gguf_bytes(
            [("general.alignment", 32), ("tokenizer.chat_template", CLEAN_TEMPLATE)],
            [("blk.0.attn_q.weight", [1 << 16], 0, _weights().tobytes())],
        )
    )
    assert sniff_format(path) == "gguf"
    code, entry = _scan_json(path)
    assert code == 0 and entry["findings"] == []
    assert entry["tensors"][0]["name"] == "blk.0.attn_q.weight"


@pytest.mark.claim("C04")
@needs_core
def test_gguf_template_injection_is_critical(tmp_path):
    path = tmp_path / "evil.gguf"
    path.write_bytes(gguf_bytes([("tokenizer.chat_template", SSTI_TEMPLATE)], []))
    code, entry = _scan_json(path)
    assert code == 1
    finding = entry["findings"][0]
    assert finding["rule"] == "AEGIS-G001" and finding["severity"] == "critical"
    assert finding["location"]["entry"] == "tokenizer.chat_template"


@needs_core
def test_gguf_named_template_variant_is_checked(tmp_path):
    path = tmp_path / "evil.gguf"
    path.write_bytes(gguf_bytes([("tokenizer.chat_template.tool_use", "{{ x | attr('__globals__') }}")], []))
    _, entry = _scan_json(path)
    assert entry["findings"][0]["rule"] == "AEGIS-G001"


@pytest.mark.claim("C05")
@needs_core
def test_gguf_overwritten_float_tensor_gets_tensor_rules(tmp_path):
    path = tmp_path / "blob.gguf"
    path.write_bytes(gguf_bytes([], [("ok", [1 << 16], 0, _weights().tobytes()), ("bad", [1 << 16], 0, _blob())]))
    code, entry = _scan_json(path)
    assert code == 1
    assert {"rule": "AEGIS-T002", "tensor": "bad"} in [
        {"rule": f["rule"], "tensor": f["location"].get("tensor")} for f in entry["findings"]
    ]


@needs_core
def test_gguf_quantized_tensors_are_not_given_float_statistics(tmp_path):
    q4_0 = _blob(n=1 << 12)  # random bytes would trip every float rule if analysed
    path = tmp_path / "q.gguf"
    path.write_bytes(gguf_bytes([], [("blk.0.ffn.weight", [1 << 14], 2, q4_0)]))
    code, entry = _scan_json(path)
    assert code == 0 and entry["tensors"] == []


@needs_core
def test_truncated_gguf_is_malformed_not_a_crash(tmp_path):
    path = tmp_path / "cut.gguf"
    path.write_bytes(gguf_bytes([("tokenizer.chat_template", CLEAN_TEMPLATE)], [])[:40])
    code, entry = _scan_json(path)
    assert code == 1 and entry["findings"][0]["rule"] == "AEGIS-G002"


# --- Keras ---------------------------------------------------------------------


def keras_file(path, layers):
    config = {"class_name": "Sequential", "config": {"name": "seq", "layers": layers}}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))
        archive.writestr("metadata.json", json.dumps({"keras_version": "3.5.0"}))
    return path


DENSE = {
    "module": "keras.layers",
    "class_name": "Dense",
    "config": {"name": "dense", "units": 4},
    "registered_name": None,
}
LAMBDA = {
    "module": "keras.layers",
    "class_name": "Lambda",
    "config": {"name": "sneaky", "function": {"class_name": "__lambda__", "config": {"code": "4wEAAAAA"}}},
    "registered_name": None,
}
CUSTOM = {
    "module": "evil_pkg.layers",
    "class_name": "Backdoor",
    "config": {"name": "bd"},
    "registered_name": "evil_pkg>Backdoor",
}


def test_clean_keras_model_has_no_findings(tmp_path):
    path = keras_file(tmp_path / "clean.keras", [DENSE])
    assert sniff_format(path) == "keras"
    code, entry = _scan_json(path)
    assert code == 0 and entry["findings"] == []


@pytest.mark.claim("C08")
def test_keras_lambda_and_custom_objects_are_flagged(tmp_path):
    path = keras_file(tmp_path / "evil.keras", [DENSE, LAMBDA, CUSTOM])
    code, entry = _scan_json(path)
    assert code == 1
    rules = {(f["rule"], f["location"]["entry"]) for f in entry["findings"]}
    assert ("AEGIS-K001", "sneaky") in rules
    assert ("AEGIS-K002", "bd") in rules


@pytest.mark.claim("C05", "C08")
@needs_core
def test_legacy_h5_model_config_and_weights_are_checked(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["model_config"] = json.dumps({"class_name": "Sequential", "config": {"layers": [LAMBDA]}})
        handle.create_dataset("model_weights/dense/kernel", data=np.frombuffer(_blob(), dtype=np.float32))
    assert sniff_format(path) == "hdf5"
    code, entry = _scan_json(path)
    rules = {f["rule"] for f in entry["findings"]}
    assert code == 1 and {"AEGIS-K001", "AEGIS-T002"} <= rules


# --- ONNX ----------------------------------------------------------------------


def onnx_model(path, initializers, nodes):
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path.write_bytes(model.SerializeToString())
    return path


def _float_init(name, array):
    onnx = pytest.importorskip("onnx")
    return onnx.numpy_helper.from_array(array, name)


def _external_init(name, location):
    onnx = pytest.importorskip("onnx")
    tensor = onnx.TensorProto(name=name, data_type=onnx.TensorProto.FLOAT, dims=[4])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value=location)
    return tensor


@needs_core
def test_clean_onnx_model_has_no_findings(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = onnx_model(
        tmp_path / "clean.onnx", [_float_init("w", _weights())], [onnx.helper.make_node("Identity", ["x"], ["y"])]
    )
    assert sniff_format(path) == "onnx"
    code, entry = _scan_json(path)
    assert code == 0 and entry["findings"] == []
    assert entry["tensors"][0]["name"] == "w"


@pytest.mark.claim("C05", "C08")
@needs_core
def test_onnx_external_paths_custom_ops_and_tensor_rules(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = onnx_model(
        tmp_path / "evil.onnx",
        [
            _external_init("leak", "../../../etc/passwd"),
            _external_init("ok_ext", "weights.bin"),
            _float_init("bad", np.frombuffer(_blob(), dtype=np.float32)),
        ],
        [onnx.helper.make_node("Steal", ["x"], ["y"], domain="evil.ops", name="steal")],
    )
    code, entry = _scan_json(path)
    assert code == 1
    found = {(f["rule"], f["location"].get("entry") or f["location"].get("tensor")) for f in entry["findings"]}
    assert ("AEGIS-X001", "leak") in found
    assert ("AEGIS-X002", "ok_ext") in found
    assert ("AEGIS-X003", "steal") in found
    assert ("AEGIS-T002", "bad") in found
    severities = {f["rule"]: f["severity"] for f in entry["findings"]}
    assert severities["AEGIS-X001"] == "high" and severities["AEGIS-X002"] == "low"


def test_gguf_tensor_pointing_past_the_end_is_flagged(tmp_path):
    full = gguf_bytes([("general.alignment", 32)], [("blk.0.attn_q.weight", [1 << 16], 0, _weights().tobytes())])
    path = tmp_path / "truncated.gguf"
    path.write_bytes(full[:-4096])
    code, entry = _scan_json(path)
    assert code == 1
    assert any("points past the end of the file" in f["message"] for f in entry["findings"]), entry["findings"]


def test_gguf_text_report_lists_findings(tmp_path):
    path = tmp_path / "evil.gguf"
    template = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    path.write_bytes(gguf_bytes([("general.alignment", 32), ("tokenizer.chat_template", template)], []))
    result = runner.invoke(app, ["scan", str(path)])
    assert result.exit_code == 1
    assert "format: gguf" in result.output
    assert "tokenizer.chat_template" in result.output
