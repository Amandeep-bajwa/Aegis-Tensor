"""Aegis-Tensor: Open-source security tool for AI models.

Detects Steganography (hidden malware) and Sleeper Agent Trojans (backdoors)
inside AI models like .safetensors using static Rust analysis and dynamic PyTorch fuzzing.
"""

__version__ = "0.1.0"

# Import compiled Rust core extension if available
try:
    from aegis.aegis_core import (  # type: ignore
        TensorScanResult,
        benford_law_mad,
        benford_law_stats,
        mantissa_bit_plane_entropy,
        scan_safetensors,
        scan_tensor_bytes,
        shannon_entropy_chunked,
        shannon_entropy_bytes,
    )

    CORE_AVAILABLE = True
except ImportError:
    # Graceful fallback when aegis_core hasn't been compiled yet with maturin
    CORE_AVAILABLE = False
    TensorScanResult = None  # type: ignore
    scan_safetensors = None  # type: ignore
    scan_tensor_bytes = None  # type: ignore
    shannon_entropy_chunked = None  # type: ignore
    shannon_entropy_bytes = None  # type: ignore
    mantissa_bit_plane_entropy = None  # type: ignore
    benford_law_mad = None  # type: ignore
    benford_law_stats = None  # type: ignore

# Dynamic Fuzzer imports
from aegis.fuzzer import (
    DynamicTrojanFuzzer,
    ActivationHookManager,
    TrojanScanReport,
    LayerActivationStat,
    LayerAnomaly,
    LayerBaselineStat,
    classify_module_region,
    TORCH_AVAILABLE,
    generate_gaussian_noise,
    generate_boundary_inputs,
    generate_patch_triggers,
    generate_token_perturbations,
    build_auto_fuzz_suite,
)

__all__ = [
    "__version__",
    "CORE_AVAILABLE",
    "TORCH_AVAILABLE",
    "TensorScanResult",
    "scan_safetensors",
    "scan_tensor_bytes",
    "shannon_entropy_bytes",
    "shannon_entropy_chunked",
    "mantissa_bit_plane_entropy",
    "benford_law_mad",
    "benford_law_stats",
    "DynamicTrojanFuzzer",
    "ActivationHookManager",
    "TrojanScanReport",
    "LayerActivationStat",
    "LayerAnomaly",
    "LayerBaselineStat",
    "classify_module_region",
    "generate_gaussian_noise",
    "generate_boundary_inputs",
    "generate_patch_triggers",
    "generate_token_perturbations",
    "build_auto_fuzz_suite",
]
