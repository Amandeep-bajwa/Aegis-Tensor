"""Build a model from a trusted architecture and load untrusted weights without pickle.

The architecture comes from the user: a Python import path to a class or factory, or
a Hugging Face config (plain JSON, loaded with trust_remote_code=False). The weights
come from a .safetensors file, which holds only tensors. Nothing from the model
file is executed.
"""

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def build_from_import_path(spec: str, kwargs: dict[str, Any] | None = None) -> torch.nn.Module:
    """Instantiate ``pkg.module:Name`` (or ``pkg.module.Name``) with ``kwargs``.

    The import runs the user's own code, so the path must point at code the user trusts.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep:
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"architecture must look like 'package.module:ClassName', got '{spec}'")
    factory = getattr(importlib.import_module(module_name), attr)
    model = factory(**(kwargs or {}))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{spec} returned {type(model).__name__}, not a torch.nn.Module")
    return model


def build_from_hf_config(source: str, revision: str | None = None) -> tuple[torch.nn.Module, Any]:
    """Randomly initialised model from a Hugging Face config. No weights or code are downloaded."""
    import transformers
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(source, revision=revision, trust_remote_code=False)
    architectures = getattr(config, "architectures", None) or []
    model_class = getattr(transformers, architectures[0], None) if architectures else None
    if model_class is None:
        model_class = AutoModel
    return model_class.from_config(config), config


def load_safetensors_weights(model: torch.nn.Module, path: Path, strict: bool = False) -> dict[str, list]:
    """Load ``path`` into ``model``. Returns missing and unexpected keys.

    Raises ValueError when no key matches (wrong architecture) or, with ``strict``,
    when any key is missing or unexpected.
    """
    from safetensors.torch import load_file

    state = load_file(str(path))
    expected = set(model.state_dict())
    if not expected & set(state):
        raise ValueError(
            f"none of the {len(state)} tensors in {path.name} match the architecture's "
            f"{len(expected)} parameters; check --arch or --hf-config"
        )
    result = model.load_state_dict(state, strict=False)
    keys = {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}
    if strict and (keys["missing"] or keys["unexpected"]):
        raise ValueError(f"weights do not match the architecture: {keys}")
    return keys


def fuzz_model(
    model: torch.nn.Module,
    spike_threshold: float,
    iterations: int,
    seed: int,
    input_shape: tuple[int, ...] | None = None,
    vocab_size: int | None = None,
    seq_len: int | None = None,
    progress_callback: Callable | None = None,
):
    """Run the automated fuzz suite on ``model``. Shared by the in-process and sandboxed paths."""
    from aegis.fuzzer import DynamicTrojanFuzzer

    model.eval()
    if isinstance(model, torch.jit.ScriptModule):
        raise TypeError("TorchScript models cannot be fuzzed: PyTorch disallows forward hooks on them")
    device, dtype = torch.device("cpu"), torch.float32
    try:
        first = next(model.parameters())
        device = first.device
        if first.is_floating_point():
            dtype = first.dtype
    except StopIteration:
        pass

    fuzzer = DynamicTrojanFuzzer(model, spike_threshold=spike_threshold)
    if vocab_size is not None:
        return fuzzer.run_fuzzing(
            vocab_size=vocab_size,
            seq_len=seq_len or 32,
            seed=seed,
            num_iterations=iterations,
            progress_callback=progress_callback,
        )
    baseline = torch.zeros(input_shape, dtype=dtype, device=device)
    return fuzzer.run_fuzzing(
        baseline_input=baseline,
        input_shape=input_shape,
        seed=seed,
        input_dtype=dtype,
        input_device=device,
        num_iterations=iterations,
        progress_callback=progress_callback,
    )


def parse_kwargs(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--arch-kwargs must be a JSON object")
    return value
