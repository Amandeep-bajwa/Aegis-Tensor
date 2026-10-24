"""Dynamic Fuzzing Engine for Aegis-Tensor.

Hooks into PyTorch models using forward hooks, measuring intermediate layer
activations under fuzzing to detect Sleeper Agent Trojan triggers via L_infinity norm spikes.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LayerActivationStat:
    layer_name: str
    l_inf_norm: float
    mean_activation: float
    std_activation: float


@dataclass
class FuzzIterationReport:
    iteration: int
    input_tag: str
    layer_stats: dict[str, LayerActivationStat]
    max_l_inf: float
    highest_layer: str


@dataclass
class LayerBaselineStat:
    """Clean-calibration statistics of a layer's L_inf norm."""

    layer_name: str
    max_l_inf: float
    mean_l_inf: float
    std_l_inf: float
    num_samples: int


@dataclass
class LayerAnomaly:
    """A layer whose fuzzed L_inf norm spiked past the threshold."""

    layer_name: str
    module_type: str
    region: str  # attention | mlp | head | embedding | norm | conv | other
    depth: str  # early | middle | late
    parent_module: str
    spike_ratio: float  # peak fuzzed L_inf / clean max L_inf
    z_score: float  # (peak - clean mean) / clean std
    peak_l_inf: float
    baseline_max_l_inf: float
    trigger_tag: str
    trigger_iteration: int


@dataclass
class TrojanScanReport:
    model_name: str
    num_fuzz_samples: int
    baseline_l_inf: float
    peak_fuzzed_l_inf: float
    spike_ratio: float
    suspected_trojan: bool
    suspicious_layers: list[str]
    iteration_reports: list[FuzzIterationReport] = field(default_factory=list)
    layer_anomalies: list[LayerAnomaly] = field(default_factory=list)
    anomaly_regions: dict[str, list[str]] = field(default_factory=dict)
    baseline_stats: dict[str, LayerBaselineStat] = field(default_factory=dict)
    # Peak ratio of fuzzed L_inf to the magnitude-matched clean L_inf, per layer.
    layer_spike_ratios: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form, used to return a report from the sandboxed fuzz process."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrojanScanReport":
        data = dict(data)
        data["iteration_reports"] = [
            FuzzIterationReport(
                **{
                    **it,
                    "layer_stats": {k: LayerActivationStat(**v) for k, v in it["layer_stats"].items()},
                }
            )
            for it in data.get("iteration_reports", [])
        ]
        data["layer_anomalies"] = [LayerAnomaly(**a) for a in data.get("layer_anomalies", [])]
        data["baseline_stats"] = {k: LayerBaselineStat(**v) for k, v in data.get("baseline_stats", {}).items()}
        return cls(**data)


# ---------------------------------------------------------------------------
# Input perturbation generators (Issue #6)
# ---------------------------------------------------------------------------

FuzzCase = tuple[str, Any]  # (input_tag, input_tensor)

# L_inf magnitudes of the clean calibration inputs, cycled over the calibration set.
# Kept to the benign input range. Calibrating at larger magnitudes would let clean
# inputs fire magnitude-keyed triggers, and the reference would then hide them.
# Fuzzed inputs above this range are compared against the largest clean magnitude.
CALIBRATION_MAGNITUDES = (0.25, 0.5, 1.0, 2.0, 4.0)
# A fuzzed input is compared with clean inputs whose magnitude is within this factor.
MAGNITUDE_MATCH_FACTOR = 1.5


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for dynamic fuzzing. Install with `pip install torch`.")


def generate_gaussian_noise(
    shape: tuple[int, ...],
    sigma: float = 1.0,
    generator: Optional["torch.Generator"] = None,
) -> "torch.Tensor":
    """Zero-mean Gaussian noise with standard deviation ``sigma``.

    Sweeping ``sigma`` across a wide range probes activation behaviour far
    outside the benign input distribution.
    """
    _require_torch()
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return torch.randn(tuple(shape), generator=generator) * sigma


def generate_boundary_inputs(
    shape: tuple[int, ...],
    magnitude: float = 10.0,
    num_impulses: int = 3,
) -> list[FuzzCase]:
    """Extreme-value inputs: all zeros, saturated +/-magnitude and sparse delta impulses.

    A delta impulse sets a single feature to ``magnitude`` and everything else
    to zero, isolating triggers keyed on one input position.
    """
    _require_torch()
    shape = tuple(shape)
    cases: list[FuzzCase] = [
        ("boundary_zeros", torch.zeros(shape)),
        (f"boundary_pos_{magnitude:g}", torch.full(shape, float(magnitude))),
        (f"boundary_neg_{magnitude:g}", torch.full(shape, -float(magnitude))),
    ]

    per_sample = int(np.prod(shape[1:])) if len(shape) > 1 else int(np.prod(shape))
    if per_sample > 0 and num_impulses > 0:
        # Spread impulses evenly: first, middle and last features, etc.
        positions = sorted({int(p) for p in np.linspace(0, per_sample - 1, num_impulses)})
        for pos in positions:
            x = torch.zeros(shape)
            if len(shape) > 1:
                x.view(shape[0], -1)[:, pos] = float(magnitude)
            else:
                x.view(-1)[pos] = float(magnitude)
            cases.append((f"boundary_delta_{pos}", x))
    return cases


def generate_patch_triggers(
    shape: tuple[int, ...],
    patch_size: tuple[int, int] = (8, 8),
    intensity: float = 1.0,
    positions: tuple[str, ...] = ("bottom_right", "top_left", "center"),
    base_input: Optional["torch.Tensor"] = None,
) -> list[FuzzCase]:
    """BadNets-style triggers: a high-contrast checkerboard patch stamped on an image.

    The patch is written to the last two (spatial) dimensions, so ``shape`` is
    expected to look like (N, C, H, W), (C, H, W) or (H, W).
    """
    _require_torch()
    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError("Patch triggers need at least 2 spatial dimensions (H, W)")

    height, width = shape[-2], shape[-1]
    ph, pw = min(patch_size[0], height), min(patch_size[1], width)
    # Checkerboard of +intensity / -intensity (maximum local contrast)
    rows = torch.arange(ph).unsqueeze(1)
    cols = torch.arange(pw).unsqueeze(0)
    patch = torch.where((rows + cols) % 2 == 0, float(intensity), -float(intensity))

    anchors = {
        "top_left": (0, 0),
        "top_right": (0, width - pw),
        "bottom_left": (height - ph, 0),
        "bottom_right": (height - ph, width - pw),
        "center": ((height - ph) // 2, (width - pw) // 2),
    }
    cases: list[FuzzCase] = []
    for pos in positions:
        if pos not in anchors:
            raise ValueError(f"Unknown patch position '{pos}'. Choose from {sorted(anchors)}")
        top, left = anchors[pos]
        x = base_input.clone().float() if base_input is not None else torch.zeros(shape)
        x[..., top : top + ph, left : left + pw] = patch
        cases.append((f"patch_{pos}_{ph}x{pw}", x))
    return cases


def generate_token_perturbations(
    vocab_size: int,
    seq_len: int,
    batch_size: int = 1,
    generator: Optional["torch.Generator"] = None,
) -> list[FuzzCase]:
    """Token-ID perturbations for language models.

    Covers extreme repetition of a single token, alternating token pairs,
    vocabulary boundary IDs and rarely used tail-of-vocabulary tokens.
    """
    _require_torch()
    if vocab_size < 2 or seq_len < 1:
        raise ValueError("vocab_size must be >= 2 and seq_len >= 1")
    shape = (batch_size, seq_len)
    rand_tok = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
    tail_start = max(0, int(vocab_size * 0.95))  # last 5% of the vocabulary

    alternating = torch.tensor([rand_tok, vocab_size - 1]).repeat(seq_len // 2 + 1)[:seq_len]
    return [
        (f"token_repeat_{rand_tok}", torch.full(shape, rand_tok, dtype=torch.long)),
        ("token_repeat_min_id", torch.zeros(shape, dtype=torch.long)),
        ("token_repeat_max_id", torch.full(shape, vocab_size - 1, dtype=torch.long)),
        ("token_alternating", alternating.unsqueeze(0).repeat(batch_size, 1)),
        ("token_vocab_tail", torch.randint(tail_start, vocab_size, shape, generator=generator)),
    ]


def build_auto_fuzz_suite(
    input_shape: tuple[int, ...] | None = None,
    vocab_size: int | None = None,
    seq_len: int | None = None,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
    patch_size: tuple[int, int] = (8, 8),
    seed: int | None = None,
) -> list[FuzzCase]:
    """Build the default fuzzing suite and interleave strategies round-robin.

    Interleaving means even a short run touches every strategy
    (noise, boundary, patch, token) instead of exhausting one first.
    """
    _require_torch()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    groups: list[list[FuzzCase]] = []

    if vocab_size is not None:
        groups.append(generate_token_perturbations(vocab_size, seq_len or 32, generator=gen))
    elif input_shape is not None:
        shape = tuple(input_shape)
        groups.append([(f"gaussian_sigma_{s:g}", generate_gaussian_noise(shape, s, gen)) for s in sigmas])
        groups.append(generate_boundary_inputs(shape))
        if len(shape) >= 3:  # image-like: (N, C, H, W) or (N, H, W)
            groups.append(generate_patch_triggers(shape, patch_size))
    else:
        raise ValueError("Automated fuzzing needs either input_shape or vocab_size")

    suite: list[FuzzCase] = []
    for i in range(max(len(g) for g in groups)):
        for g in groups:
            if i < len(g):
                suite.append(g[i])
    return suite


class ActivationHookManager:
    """Manages PyTorch forward hooks to record activation statistics."""

    def __init__(
        self,
        model: Any,
        module_types: type["nn.Module"] | tuple[type["nn.Module"], ...] | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dynamic fuzzing. Install with `pip install torch`.")
        self.model = model
        if isinstance(module_types, type):
            module_types = (module_types,)
        self.module_types = module_types
        self.hooks: list[Any] = []
        self.hooked_modules: dict[str, Any] = {}
        self.current_activations: dict[str, LayerActivationStat] = {}
        self._register_hooks()

    @staticmethod
    def _find_tensor(output: Any) -> Any | None:
        """Find the first tensor in common PyTorch output containers."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, Mapping):
            for value in output.values():
                tensor = ActivationHookManager._find_tensor(value)
                if tensor is not None:
                    return tensor
        elif isinstance(output, (tuple, list)):
            for value in output:
                tensor = ActivationHookManager._find_tensor(value)
                if tensor is not None:
                    return tensor
        elif is_dataclass(output) and not isinstance(output, type):
            for output_field in fields(output):
                tensor = ActivationHookManager._find_tensor(getattr(output, output_field.name))
                if tensor is not None:
                    return tensor
        elif hasattr(output, "last_hidden_state"):
            return ActivationHookManager._find_tensor(output.last_hidden_state)
        return None

    def _hook_fn(self, name: str) -> Callable:
        def hook(module: Any, input_tensors: Any, output_tensor: Any):
            tensor = self._find_tensor(output_tensor)
            if tensor is not None:
                with torch.no_grad():
                    detached = tensor.detach().float().cpu()
                    if detached.numel() == 0:
                        return
                    # L_infinity norm: max absolute activation
                    l_inf = float(torch.max(torch.abs(detached)).item())
                    mean_val = float(torch.mean(detached).item())
                    std_val = float(torch.std(detached).item()) if detached.numel() > 1 else 0.0

                    self.current_activations[name] = LayerActivationStat(
                        layer_name=name,
                        l_inf_norm=l_inf,
                        mean_activation=mean_val,
                        std_activation=std_val,
                    )

        return hook

    def _register_hooks(self):
        """Recursively registers hooks on named modules."""
        for name, module in self.model.named_modules():
            if self.module_types is not None:
                should_hook = isinstance(module, self.module_types)
            else:
                # Hook non-container modules (layers that perform operations)
                children = list(module.children())
                should_hook = not children and len(list(module.parameters())) > 0
            if should_hook:
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)
                self.hooked_modules[name] = module

    def clear(self):
        self.current_activations.clear()

    def remove(self):
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.current_activations.clear()


# ---------------------------------------------------------------------------
# Sub-network isolation helpers (Issue #5)
# ---------------------------------------------------------------------------

_ATTENTION_KEYS = ("attn", "attention", "q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value")
_HEAD_KEYS = ("head", "classifier", "lm_head", "score", "logits")
_EMBED_KEYS = ("embed", "emb", "wte", "wpe")
_NORM_KEYS = ("norm", "ln_", "layernorm", "batchnorm")
_MLP_KEYS = ("mlp", "ffn", "feed_forward", "intermediate", "dense", "fc", "linear")


def classify_module_region(name: str, module: Any, is_last: bool = False) -> str:
    """Label a module as attention, mlp, head, embedding, norm, conv or other.

    Uses the module type first, then common naming conventions
    (HuggingFace, timm, nanoGPT). The final hooked layer is treated as the head.
    """
    lname = name.lower()
    tname = type(module).__name__.lower()
    parts = lname.replace("_", ".").split(".")

    if "attention" in tname or any(k in lname for k in _ATTENTION_KEYS):
        return "attention"
    if is_last or any(k in parts or lname.endswith(k) for k in _HEAD_KEYS):
        return "head"
    if "embedding" in tname or any(k in lname for k in _EMBED_KEYS):
        return "embedding"
    if "norm" in tname or any(k in lname for k in _NORM_KEYS):
        return "norm"
    if "conv" in tname:
        return "conv"
    if "linear" in tname or any(k in lname for k in _MLP_KEYS):
        return "mlp"
    return "other"


def _depth_bucket(index: int, total: int) -> str:
    position = index / max(total - 1, 1)
    if position < 1 / 3:
        return "early"
    if position < 2 / 3:
        return "middle"
    return "late"


def _input_magnitude(x: Any) -> float | None:
    """L_inf magnitude of a floating-point input, or None for token IDs and other inputs."""
    if torch.is_tensor(x) and x.is_floating_point() and x.numel() > 0:
        return float(torch.max(torch.abs(x.detach())).item())
    return None


def _matched_baseline(
    records: list[tuple[float | None, dict[str, float]]],
    magnitude: float | None,
    layer: str,
) -> float:
    """Clean L_inf of ``layer`` for inputs whose magnitude matches ``magnitude``.

    Comparing a fuzzed input against clean inputs of the same magnitude keeps
    ordinary input-scale effects, including bias-dominated responses at small
    magnitudes, from being read as a backdoor. A clean network stays near ratio 1
    at every scale. Falls back to the three nearest magnitudes when none match.
    """
    candidates = [(m, layers[layer]) for m, layers in records if layer in layers]
    if magnitude is not None and magnitude > 0:
        matched = [
            v
            for m, v in candidates
            if m is not None and magnitude / MAGNITUDE_MATCH_FACTOR <= m <= magnitude * MAGNITUDE_MATCH_FACTOR
        ]
        if not matched:
            # Outside the calibrated range: scale the nearest clean responses linearly
            # to the fuzzed magnitude. A homogeneous network responds linearly, so this
            # keeps clean networks near ratio 1 at any scale.
            valid = [(m, v) for m, v in candidates if m is not None and m > 0]
            if valid:
                nearest = min(valid, key=lambda mv: abs(math.log(mv[0]) - math.log(magnitude)))[0]
                group = [
                    (m, v) for m, v in valid if abs(math.log(m) - math.log(nearest)) <= math.log(MAGNITUDE_MATCH_FACTOR)
                ]
                matched = [v * magnitude / m for m, v in group]
    else:
        matched = [v for _, v in candidates]
    return max(max(matched, default=0.0), 1e-6)


class _RunningStat:
    """Welford running mean / std / max for one layer's L_inf norms."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.max = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        self.max = max(self.max, x)

    @property
    def std(self) -> float:
        return (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class DynamicTrojanFuzzer:
    """Fuzzes AI models to detect latent Sleeper Agent Trojans via activation spikes."""

    def __init__(self, model: Any, spike_threshold: float = 4.0):
        """
        Args:
            model: The PyTorch model to inspect.
            spike_threshold: Ratio of fuzzed L_infinity to baseline L_infinity to trigger suspicion.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dynamic fuzzing.")
        self.model = model
        self.model.eval()
        self.spike_threshold = spike_threshold

    def run_fuzzing(
        self,
        sample_inputs_generator: Callable[[int], Any] | None = None,
        baseline_input: Any = None,
        num_iterations: int = 50,
        progress_callback: Callable[[int, int, str, float], None] | None = None,
        input_shape: tuple[int, ...] | None = None,
        vocab_size: int | None = None,
        seq_len: int | None = None,
        seed: int | None = None,
        calibration_inputs: list[Any] | None = None,
        num_calibration: int = 64,
        calibration_noise_std: float = 1.0,
        input_dtype: Any = None,
        input_device: Any = None,
    ) -> TrojanScanReport:
        """Fuzz the model and flag layers whose activations spike above clean behaviour.

        Manual mode: pass ``sample_inputs_generator`` and ``baseline_input``.
        Automated mode: leave ``sample_inputs_generator`` as None and pass
        ``input_shape`` (float models) or ``vocab_size`` (token models).

        Clean calibration: layer statistics are measured over several benign
        inputs, not a single tensor. Pass ``calibration_inputs`` explicitly,
        or they are generated as ``baseline_input`` plus Gaussian noise
        (``num_calibration`` samples, std ``calibration_noise_std``).

        Each fuzzed input is compared with clean inputs of the same L_inf magnitude
        (within a factor of ``MAGNITUDE_MATCH_FACTOR``). A layer is suspicious when
        its peak fuzzed activation is at least ``spike_threshold`` times the matched
        clean activation. Comparing like with like keeps ordinary input-scale effects
        from being read as a backdoor.

        ``input_dtype`` and ``input_device`` cast the generated inputs, so automated
        mode works for half-precision or GPU models.
        """

        def _cast(x: Any) -> Any:
            if not torch.is_tensor(x) or not x.is_floating_point():
                return x
            if input_dtype is None and input_device is None:
                return x
            return x.to(dtype=input_dtype or x.dtype, device=input_device or x.device)

        auto_suite: list[FuzzCase] | None = None
        if sample_inputs_generator is None:
            auto_suite = build_auto_fuzz_suite(
                input_shape=input_shape, vocab_size=vocab_size, seq_len=seq_len, seed=seed
            )
            if baseline_input is None:
                gen = torch.Generator().manual_seed(seed) if seed is not None else None
                # build_auto_fuzz_suite has already refused a call with neither vocab_size nor input_shape.
                if vocab_size is not None:
                    baseline_input = torch.randint(0, vocab_size, (1, seq_len or 32), generator=gen)
                elif input_shape is not None:
                    baseline_input = torch.randn(tuple(input_shape), generator=gen)
        elif baseline_input is None:
            raise ValueError("baseline_input is required when a custom sample_inputs_generator is given")

        baseline_input = _cast(baseline_input)
        calibration = [
            _cast(x)
            for x in self._build_calibration(
                baseline_input, calibration_inputs, num_calibration, calibration_noise_std, vocab_size, seed
            )
        ]

        hook_mgr = ActivationHookManager(self.model)
        try:
            # 1. Clean calibration: global statistics per layer, plus one record per clean
            #    input so each fuzzed input can be matched to clean inputs of its magnitude.
            running: dict[str, _RunningStat] = {}
            records: list[tuple[float | None, dict[str, float]]] = []
            for clean_input in calibration:
                hook_mgr.clear()
                with torch.no_grad():
                    _ = self.model(clean_input)
                layer_max = {name: stat.l_inf_norm for name, stat in hook_mgr.current_activations.items()}
                records.append((_input_magnitude(clean_input), layer_max))
                for name, value in layer_max.items():
                    running.setdefault(name, _RunningStat()).update(value)

            baseline_stats = {
                name: LayerBaselineStat(name, rs.max, rs.mean, rs.std, rs.n) for name, rs in running.items()
            }
            baseline_max_l_inf = max((b.max_l_inf for b in baseline_stats.values()), default=1.0)
            baseline_max_l_inf = max(baseline_max_l_inf, 1e-6)

            # 2. Fuzzing loop: each layer's spike ratio is its fuzzed L_inf divided by the
            #    clean L_inf at the same input magnitude. Track the strongest trigger per layer.
            peak_by_layer: dict[str, tuple[float, str, int, float]] = {}  # ratio, tag, iteration, raw L_inf
            peak_fuzzed_l_inf = 0.0
            iteration_reports: list[FuzzIterationReport] = []

            for i in range(num_iterations):
                if sample_inputs_generator is not None:
                    input_tag, test_input = f"fuzz_step_{i + 1}", sample_inputs_generator(i)
                elif auto_suite is not None:
                    input_tag, test_input = auto_suite[i % len(auto_suite)]
                    test_input = _cast(test_input)
                hook_mgr.clear()
                with torch.no_grad():
                    _ = self.model(test_input)

                layer_stats = dict(hook_mgr.current_activations)
                if not layer_stats:
                    continue

                magnitude = _input_magnitude(test_input)
                iter_top_layer, iter_ratio, iter_raw = "", 0.0, 0.0
                for name, stat in layer_stats.items():
                    ratio = stat.l_inf_norm / _matched_baseline(records, magnitude, name)
                    if ratio > iter_ratio:
                        iter_ratio, iter_top_layer = ratio, name
                    iter_raw = max(iter_raw, stat.l_inf_norm)
                    if name not in peak_by_layer or ratio > peak_by_layer[name][0]:
                        peak_by_layer[name] = (ratio, input_tag, i + 1, stat.l_inf_norm)

                peak_fuzzed_l_inf = max(peak_fuzzed_l_inf, iter_raw)
                iteration_reports.append(
                    FuzzIterationReport(
                        iteration=i + 1,
                        input_tag=input_tag,
                        layer_stats=layer_stats,
                        max_l_inf=iter_raw,
                        highest_layer=iter_top_layer,
                    )
                )
                if progress_callback is not None:
                    progress_callback(i + 1, num_iterations, iter_top_layer, iter_raw)

            # 3. Sub-network isolation
            anomalies = self._isolate_anomalies(hook_mgr.hooked_modules, baseline_stats, peak_by_layer)
            layer_spike_ratios = {name: peak[0] for name, peak in peak_by_layer.items()}
            spike_ratio = max(layer_spike_ratios.values(), default=0.0)
            regions: dict[str, list[str]] = {}
            for a in anomalies:
                regions.setdefault(a.region, []).append(a.layer_name)

            return TrojanScanReport(
                model_name=getattr(self.model, "__class__", type(self.model)).__name__,
                num_fuzz_samples=num_iterations,
                baseline_l_inf=baseline_max_l_inf,
                peak_fuzzed_l_inf=peak_fuzzed_l_inf,
                spike_ratio=spike_ratio,
                suspected_trojan=len(anomalies) > 0,
                suspicious_layers=sorted(a.layer_name for a in anomalies),
                iteration_reports=iteration_reports,
                layer_anomalies=anomalies,
                anomaly_regions=regions,
                baseline_stats=baseline_stats,
                layer_spike_ratios=layer_spike_ratios,
            )
        finally:
            hook_mgr.remove()

    @staticmethod
    def _build_calibration(
        baseline_input: Any,
        calibration_inputs: list[Any] | None,
        num_calibration: int,
        noise_std: float,
        vocab_size: int | None,
        seed: int | None,
    ) -> list[Any]:
        """Assemble the clean calibration set (always includes baseline_input)."""
        if calibration_inputs is not None:
            return [baseline_input, *calibration_inputs]

        calibration = [baseline_input]
        if not torch.is_tensor(baseline_input) or num_calibration <= 0:
            return calibration
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        for k in range(num_calibration):
            if baseline_input.is_floating_point():
                # Each clean input has a known L_inf magnitude, so fuzzed inputs can be
                # compared against clean inputs of the same size (see _matched_baseline).
                target = CALIBRATION_MAGNITUDES[k % len(CALIBRATION_MAGNITUDES)] * noise_std
                noise = torch.randn(baseline_input.shape, generator=gen)
                noise = noise / noise.abs().max().clamp_min(1e-12) * target
                calibration.append(baseline_input + noise.to(baseline_input.dtype))
            elif vocab_size is not None:
                calibration.append(torch.randint(0, vocab_size, baseline_input.shape, generator=gen))
        return calibration

    def _isolate_anomalies(
        self,
        hooked_modules: dict[str, Any],
        baseline_stats: dict[str, LayerBaselineStat],
        peak_by_layer: dict[str, tuple[float, str, int, float]],
    ) -> list[LayerAnomaly]:
        """Return every layer whose spike ratio >= threshold, strongest first.

        ``peak_by_layer`` holds (spike ratio, trigger tag, iteration, raw peak L_inf).
        """
        names = list(hooked_modules)
        anomalies: list[LayerAnomaly] = []
        for idx, name in enumerate(names):
            if name not in peak_by_layer or name not in baseline_stats:
                continue
            ratio, tag, iteration, peak = peak_by_layer[name]
            base = baseline_stats[name]
            if ratio < self.spike_threshold:
                continue
            module = hooked_modules[name]
            anomalies.append(
                LayerAnomaly(
                    layer_name=name,
                    module_type=type(module).__name__,
                    region=classify_module_region(name, module, is_last=(idx == len(names) - 1)),
                    depth=_depth_bucket(idx, len(names)),
                    parent_module=name.rsplit(".", 1)[0] if "." in name else "<root>",
                    spike_ratio=ratio,
                    z_score=(peak - base.mean_l_inf) / max(base.std_l_inf, 1e-6),
                    peak_l_inf=peak,
                    baseline_max_l_inf=base.max_l_inf,
                    trigger_tag=tag,
                    trigger_iteration=iteration,
                )
            )
        anomalies.sort(key=lambda a: a.spike_ratio, reverse=True)
        return anomalies
