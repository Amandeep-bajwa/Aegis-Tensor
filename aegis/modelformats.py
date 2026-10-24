"""Scanners for GGUF, Keras (.keras and legacy HDF5) and ONNX model files.

Nothing is executed and no model is loaded through its framework:
* GGUF is parsed from its binary layout. Chat templates are checked for Jinja
  template-injection patterns (the class behind CVE-2024-34359 in llama-cpp-python),
  and unquantized tensors get the same statistics as safetensors tensors.
* Keras configs are read as JSON. Lambda layers (which carry marshalled Python
  bytecode) and custom objects outside Keras are flagged. Weights get tensor statistics
  when h5py is installed.
* ONNX graphs are read with `onnx.load(load_external_data=False)`, so external data
  files are never opened. External data paths that leave the model directory and
  custom operator domains are flagged; float initializers get tensor statistics.
"""

import io
import json
import struct
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from aegis.picklescan import PickleScanResult

GGUF_MAGIC = b"GGUF"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# GGUF metadata value types -> struct format (strings and arrays are handled separately).
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# ggml tensor types that store plain floats, with their safetensors dtype names.
_GGML_FLOAT_TYPES = {0: ("F32", 4), 1: ("F16", 2), 28: ("F64", 8), 30: ("BF16", 2)}
# Jinja patterns that reach Python internals or the OS. Legitimate chat templates do not
# need dunder attributes, the `attr` filter or process/eval calls.
_ONNX_STANDARD_DOMAINS = {"", "ai.onnx", "ai.onnx.ml", "ai.onnx.preview.training", "ai.onnx.training"}


@dataclass
class ModelFormatResult(PickleScanResult):
    """Container-style findings plus per-tensor statistics."""

    tensor_results: list[Any] = field(default_factory=list)


def sniff_format(path: str | Path) -> str | None:
    """'gguf', 'hdf5', 'keras' or 'onnx' when the file is one of these formats, else None."""
    path = Path(path)
    with open(path, "rb") as handle:
        head = handle.read(16)
    if head[:4] == GGUF_MAGIC:
        return "gguf"
    if head[:8] == HDF5_MAGIC:
        return "hdf5"
    if head[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(path).namelist())
        except (zipfile.BadZipFile, OSError):
            return None
        if "config.json" in names and ("metadata.json" in names or any(n.endswith(".h5") for n in names)):
            return "keras"
        return None
    # ONNX has no magic number; a ModelProto starts with field 1 (ir_version), tag 0x08.
    if head[:1] == b"\x08" and path.suffix.lower() == ".onnx":
        return "onnx"
    return None


def scan_model_format(path: str | Path, fmt: str, thresholds: dict[str, Any]) -> ModelFormatResult:
    path = Path(path)
    result = ModelFormatResult(path=str(path), format=fmt)
    {"gguf": _scan_gguf, "hdf5": _scan_hdf5, "keras": _scan_keras, "onnx": _scan_onnx}[fmt](path, result, thresholds)
    return result


def _tensor_stats(
    result: ModelFormatResult, name: str, dtype: str, shape: list[int], data: bytes, thresholds: dict[str, Any]
) -> None:
    from aegis import scan_tensor_bytes

    if scan_tensor_bytes is None:
        return
    result.tensor_results.append(
        scan_tensor_bytes(
            name,
            dtype,
            list(shape),
            data,
            thresholds["entropy"],
            thresholds["benford"],
            thresholds["exponent"],
            thresholds["plane"],
        )
    )


def _check_template(text: str, result: ModelFormatResult, where: str) -> None:
    from aegis.templatescan import check_template

    for code, message in check_template(text):
        if code == "injection":
            result.add(
                "CRITICAL",
                f"chat template: {message}. An unsandboxed Jinja loader would run code (CVE-2024-34359)",
                where,
                code="gguf_template",
            )
        else:
            result.add("SUSPICIOUS", f"chat template: {message}", where, code="gguf_template_unverifiable")


def _check_secrets(metadata: dict[str, Any], result: ModelFormatResult, label: str) -> None:
    from aegis.secrets import find_secrets

    for key, value in metadata.items():
        if isinstance(value, str):
            for kind, masked in find_secrets(value):
                result.add(
                    "SUSPICIOUS",
                    f"{kind} in metadata ({masked}) - leaked credential; tell the owner privately and have it revoked",
                    f"{label}:{key}",
                    code="secret",
                )


# --- GGUF -------------------------------------------------------------------------


class _Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError(f"truncated at byte {self.pos}")
        (value,) = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value

    def string(self) -> str:
        length = self.take("<Q")
        if self.pos + length > len(self.data):
            raise ValueError(f"string of {length} bytes runs past the end of the file")
        raw = bytes(self.data[self.pos : self.pos + length])
        self.pos += length
        return raw.decode("utf-8", "replace")

    def value(self, kind: int):
        if kind in _GGUF_SCALARS:
            return self.take(_GGUF_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind, count = self.take("<I"), self.take("<Q")
            if item_kind in _GGUF_SCALARS:  # skip fixed-size arrays without materialising them
                size = struct.calcsize(_GGUF_SCALARS[item_kind]) * count
                if self.pos + size > len(self.data):
                    raise ValueError("array runs past the end of the file")
                self.pos += size
                return None
            return [self.value(item_kind) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")


def parse_gguf_header(data) -> tuple[dict[str, Any], list[tuple[str, list[int], int, int]], int]:
    """(metadata, tensor infos, end of header). ``data`` may be a prefix of the file;
    ValueError("truncated ...") means more bytes are needed. Fixed-size arrays are skipped."""
    reader = _Reader(data)
    if reader.take("<I") != struct.unpack("<I", GGUF_MAGIC)[0]:
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = reader.take("<I"), reader.take("<Q"), reader.take("<Q")
    metadata = {}
    for _ in range(n_kv):
        key = reader.string()
        metadata[key] = reader.value(reader.take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = reader.string()
        dims = [reader.take("<Q") for _ in range(reader.take("<I"))]
        infos.append((name, dims, reader.take("<I"), reader.take("<Q")))
    return metadata, infos, reader.pos


def chat_templates(metadata: dict[str, Any]) -> dict[str, str]:
    return {k: v for k, v in metadata.items() if k.startswith("tokenizer.chat_template") and isinstance(v, str)}


def _scan_gguf(path: Path, result: ModelFormatResult, thresholds: dict[str, Any]) -> None:
    import mmap

    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        try:
            metadata, infos, header_end = parse_gguf_header(data)
        except (ValueError, struct.error) as error:
            result.add("SUSPICIOUS", f"malformed GGUF ({error})", path.name, code="gguf_malformed")
            return
        for key, value in chat_templates(metadata).items():
            _check_template(value, result, f"{path.name}:{key}")
        _check_secrets(metadata, result, path.name)
        alignment = metadata.get("general.alignment") or 32
        data_start = -(-header_end // alignment) * alignment
        for name, dims, ggml_type, offset in infos:
            if ggml_type not in _GGML_FLOAT_TYPES:
                continue  # quantized blocks: statistics for floats do not apply
            dtype, width = _GGML_FLOAT_TYPES[ggml_type]
            count = 1
            for d in dims:
                count *= d
            start, end = data_start + offset, data_start + offset + count * width
            if end > len(data):
                result.add(
                    "SUSPICIOUS",
                    f"tensor {name} points past the end of the file",
                    f"{path.name}:{name}",
                    code="gguf_malformed",
                )
                continue
            _tensor_stats(result, name, dtype, list(reversed(dims)), bytes(data[start:end]), thresholds)


# --- Keras -------------------------------------------------------------------------


def _walk_layers(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "class_name" in node:
            yield node
        for value in node.values():
            yield from _walk_layers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_layers(value)


def _check_keras_config(config: Any, result: ModelFormatResult, where: str) -> None:
    for layer in _walk_layers(config):
        cls = layer.get("class_name")
        layer_cfg = layer.get("config") if isinstance(layer.get("config"), dict) else {}
        name = layer_cfg.get("name", cls)
        if cls == "Lambda":
            result.add(
                "SUSPICIOUS",
                f"Lambda layer '{name}' stores Python bytecode that runs when the model is loaded without safe_mode",
                f"{where}:{name}",
                code="keras_lambda",
            )
        module, registered = layer.get("module"), layer.get("registered_name")
        custom_module = isinstance(module, str) and not module.startswith(("keras", "tf_keras", "tensorflow"))
        custom_registration = isinstance(registered, str) and ">" in registered
        if custom_module or custom_registration:
            result.add(
                "SUSPICIOUS",
                f"custom object '{cls}' from {module or registered} must be imported to load the model",
                f"{where}:{name}",
                code="keras_custom_object",
            )


def _scan_h5_weights(blob: bytes | Path, result: ModelFormatResult, label: str, thresholds) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError:
        result.add("SUSPICIOUS", "HDF5 weights not inspected: install h5py", label, code="format_unavailable")
        return
    dtypes = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("float64"): "F64"}
    handle = h5py.File(io.BytesIO(blob) if isinstance(blob, bytes) else blob, "r")
    with handle:
        model_config = handle.attrs.get("model_config")
        if model_config is not None:
            text = model_config.decode() if isinstance(model_config, bytes) else str(model_config)
            _check_keras_config(json.loads(text), result, label)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype in dtypes and obj.size:
                array = np.ascontiguousarray(obj[()], dtype=obj.dtype.newbyteorder("<"))
                _tensor_stats(result, name, dtypes[obj.dtype], list(array.shape), array.tobytes(), thresholds)

        handle.visititems(visit)


def _scan_keras(path: Path, result: ModelFormatResult, thresholds: dict[str, Any]) -> None:
    with zipfile.ZipFile(path) as archive:
        try:
            config = json.loads(archive.read("config.json"))
        except (KeyError, ValueError) as error:
            result.add("SUSPICIOUS", f"unreadable Keras config ({error})", path.name, code="container_unreadable")
            return
        _check_keras_config(config, result, path.name)
        for name in archive.namelist():
            if name.endswith(".h5"):
                _scan_h5_weights(archive.read(name), result, f"{path.name}:{name}", thresholds)


def _scan_hdf5(path: Path, result: ModelFormatResult, thresholds: dict[str, Any]) -> None:
    _scan_h5_weights(path, result, path.name, thresholds)


# --- ONNX --------------------------------------------------------------------------


def _graphs(graph) -> Iterator[Any]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _graphs(attribute.g)
            for sub in attribute.graphs:
                yield from _graphs(sub)


def _escapes_directory(location: str) -> bool:
    for flavour in (PurePosixPath(location), PureWindowsPath(location)):
        if flavour.is_absolute() or ".." in flavour.parts:
            return True
    return location.startswith("~")


def _scan_onnx(path: Path, result: ModelFormatResult, thresholds: dict[str, Any]) -> None:
    try:
        import onnx
    except ImportError:
        result.add("SUSPICIOUS", "ONNX model not inspected: install onnx", path.name, code="format_unavailable")
        return
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as error:
        result.add("SUSPICIOUS", f"unreadable ONNX model ({error})", path.name, code="container_unreadable")
        return
    float_types = {
        onnx.TensorProto.FLOAT: "F32",
        onnx.TensorProto.FLOAT16: "F16",
        onnx.TensorProto.BFLOAT16: "BF16",
        onnx.TensorProto.DOUBLE: "F64",
    }
    for graph in _graphs(model.graph):
        for tensor in graph.initializer:
            where = f"{path.name}:{tensor.name}"
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                location = {e.key: e.value for e in tensor.external_data}.get("location", "")
                if _escapes_directory(location):
                    result.add(
                        "SUSPICIOUS",
                        f"external data path '{location}' leaves the model directory "
                        "(a loader would read an arbitrary file)",
                        where,
                        code="onnx_external_path",
                    )
                else:
                    result.add(
                        "SUSPICIOUS",
                        f"tensor stored in external file '{location}' (not scanned)",
                        where,
                        code="onnx_external_data",
                    )
            elif tensor.data_type in float_types and tensor.raw_data:
                _tensor_stats(
                    result, tensor.name, float_types[tensor.data_type], list(tensor.dims), tensor.raw_data, thresholds
                )
        for node in graph.node:
            if node.domain not in _ONNX_STANDARD_DOMAINS:
                result.add(
                    "SUSPICIOUS",
                    f"node '{node.name or node.op_type}' uses custom operator "
                    f"{node.domain}::{node.op_type}, which needs a native library to run",
                    f"{path.name}:{node.name or node.op_type}",
                    code="onnx_custom_op",
                )
