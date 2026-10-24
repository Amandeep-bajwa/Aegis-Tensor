"""Aegis-Tensor: Open-source security tool for AI models.

Detects Steganography (hidden malware) and Sleeper Agent Trojans (backdoors)
inside AI models like .safetensors using static Rust analysis and dynamic PyTorch fuzzing.
"""

__version__ = "0.1.0"

# Import compiled Rust core extension if available
try:
    from aegis.aegis_core import (
        TensorScanResult,
        benford_law_mad,
        benford_law_stats,
        mantissa_bit_plane_entropy,
        scan_safetensors,
        scan_tensor_bytes,
        shannon_entropy_bytes,
        shannon_entropy_chunked,
    )

    CORE_AVAILABLE = True
except ImportError:
    # Graceful fallback when aegis_core hasn't been compiled yet with maturin
    CORE_AVAILABLE = False
    TensorScanResult = None
    scan_safetensors = None
    scan_tensor_bytes = None
    shannon_entropy_chunked = None
    shannon_entropy_bytes = None
    mantissa_bit_plane_entropy = None
    benford_law_mad = None
    benford_law_stats = None

# Dynamic Fuzzer imports
from aegis.fuzzer import (
    TORCH_AVAILABLE,
    ActivationHookManager,
    DynamicTrojanFuzzer,
    LayerActivationStat,
    LayerAnomaly,
    LayerBaselineStat,
    TrojanScanReport,
    build_auto_fuzz_suite,
    classify_module_region,
    generate_boundary_inputs,
    generate_gaussian_noise,
    generate_patch_triggers,
    generate_token_perturbations,
)

__all__ = [
    "CORE_AVAILABLE",
    "TORCH_AVAILABLE",
    "ActivationHookManager",
    "DynamicTrojanFuzzer",
    "LayerActivationStat",
    "LayerAnomaly",
    "LayerBaselineStat",
    "TensorScanResult",
    "TrojanScanReport",
    "__version__",
    "benford_law_mad",
    "benford_law_stats",
    "build_auto_fuzz_suite",
    "classify_module_region",
    "generate_boundary_inputs",
    "generate_gaussian_noise",
    "generate_patch_triggers",
    "generate_token_perturbations",
    "mantissa_bit_plane_entropy",
    "scan_safetensors",
    "scan_tensor_bytes",
    "shannon_entropy_bytes",
    "shannon_entropy_chunked",
]
