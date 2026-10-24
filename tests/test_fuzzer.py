"""Tests for Dynamic Fuzzing Engine in aegis/fuzzer.py."""

from dataclasses import dataclass

import pytest

from aegis import TORCH_AVAILABLE, ActivationHookManager, DynamicTrojanFuzzer

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class SimpleLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")
def test_activation_hook_manager():
    """Verify that forward hooks register, capture activations, and compute L_inf."""
    model = SimpleLinearModel()
    hook_mgr = ActivationHookManager(model)

    test_input = torch.randn(2, 10)
    _ = model(test_input)

    assert len(hook_mgr.current_activations) > 0
    assert "fc1" in hook_mgr.current_activations
    assert "fc2" in hook_mgr.current_activations

    fc1_stat = hook_mgr.current_activations["fc1"]
    assert fc1_stat.l_inf_norm >= 0.0

    # Cleanup
    hook_mgr.remove()
    assert len(hook_mgr.hooks) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")
def test_dynamic_trojan_fuzzer_clean_model():
    """Verify that a clean model does not trigger high spike ratio."""
    model = SimpleLinearModel()
    fuzzer = DynamicTrojanFuzzer(model, spike_threshold=10.0)

    report = fuzzer.run_fuzzing(
        sample_inputs_generator=lambda i: torch.randn(2, 10),
        baseline_input=torch.zeros(2, 10),
        num_iterations=10,
    )

    assert report.model_name == "SimpleLinearModel"
    assert report.num_fuzz_samples == 10
    assert not report.suspected_trojan


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")
def test_hook_manager_extracts_nested_outputs_and_filters_modules():
    @dataclass
    class ModelOutput:
        last_hidden_state: torch.Tensor

    class StructuredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            return {"nested": (ModelOutput(self.linear(x)),)}

    model = StructuredModel()
    hook_mgr = ActivationHookManager(model, module_types=nn.Linear)

    try:
        _ = model(torch.ones(2, 3, requires_grad=True))

        assert list(hook_mgr.current_activations) == ["linear"]
        assert hook_mgr.current_activations["linear"].l_inf_norm > 0
    finally:
        hook_mgr.remove()

    assert hook_mgr.hooks == []
    assert hook_mgr.current_activations == {}
    hook_mgr.remove()


# ---------------------------------------------------------------------------
# Issue #6: input perturbation generators and automated fuzzing mode
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    from aegis import (
        build_auto_fuzz_suite,
        generate_boundary_inputs,
        generate_gaussian_noise,
        generate_patch_triggers,
        generate_token_perturbations,
    )

    class TrojanModel(nn.Module):
        """Linear model with an artificial sleeper backdoor on feature 0."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            h = self.fc1(x)
            if (x[:, 0] > 8.0).any():
                h = h * 50.0
            return self.fc2(h)

    class TinyTokenModel(nn.Module):
        def __init__(self, vocab_size=100):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 16)
            self.fc = nn.Linear(16, 4)

        def forward(self, ids):
            return self.fc(self.emb(ids))


needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")


@needs_torch
def test_gaussian_noise_shape_and_scale():
    gen = torch.Generator().manual_seed(0)
    x = generate_gaussian_noise((4, 1000), sigma=5.0, generator=gen)
    assert x.shape == (4, 1000)
    assert 4.0 < x.std().item() < 6.0
    with pytest.raises(ValueError):
        generate_gaussian_noise((2, 2), sigma=-1.0)


@needs_torch
def test_boundary_inputs_cover_extremes_and_impulses():
    cases = dict(generate_boundary_inputs((2, 10), magnitude=10.0, num_impulses=3))
    assert torch.all(cases["boundary_zeros"] == 0)
    assert torch.all(cases["boundary_pos_10"] == 10.0)
    assert torch.all(cases["boundary_neg_10"] == -10.0)
    delta = cases["boundary_delta_0"]
    assert torch.all(delta[:, 0] == 10.0) and delta[:, 1:].abs().sum() == 0


@needs_torch
def test_patch_triggers_stamp_checkerboard():
    cases = generate_patch_triggers((1, 3, 32, 32), patch_size=(8, 8), intensity=1.0)
    tags = [t for t, _ in cases]
    assert tags == ["patch_bottom_right_8x8", "patch_top_left_8x8", "patch_center_8x8"]
    x = dict(cases)["patch_bottom_right_8x8"]
    patch = x[0, 0, 24:, 24:]
    assert patch.abs().min() == 1.0  # every pixel in the patch is set
    assert patch[0, 0] == -patch[0, 1]  # alternating contrast
    assert x[0, 0, :24, :].abs().sum() == 0  # rest of the image untouched
    with pytest.raises(ValueError):
        generate_patch_triggers((10,))


@needs_torch
def test_token_perturbations_valid_ids():
    cases = generate_token_perturbations(vocab_size=50, seq_len=12, batch_size=2)
    for tag, ids in cases:
        assert ids.dtype == torch.long, tag
        assert ids.shape == (2, 12), tag
        assert ids.min() >= 0 and ids.max() < 50, tag
    repeat = dict(cases)["token_repeat_max_id"]
    assert torch.all(repeat == 49)


@needs_torch
def test_auto_suite_interleaves_strategies():
    suite = build_auto_fuzz_suite(input_shape=(1, 3, 16, 16), seed=0)
    first_three = [tag.split("_")[0] for tag, _ in suite[:3]]
    assert first_three == ["gaussian", "boundary", "patch"]
    with pytest.raises(ValueError):
        build_auto_fuzz_suite()


@needs_torch
def test_auto_mode_uncovers_trojan_without_custom_generator():
    torch.manual_seed(0)
    fuzzer = DynamicTrojanFuzzer(TrojanModel(), spike_threshold=4.0)
    report = fuzzer.run_fuzzing(input_shape=(2, 10), num_iterations=12, seed=0)

    assert report.suspected_trojan
    # The gate sits between fc1 and fc2 and multiplies fc1's output, so the spike
    # appears in fc2. fc1's own output is unchanged by the trigger.
    assert "fc2" in report.suspicious_layers
    assert "fc1" not in report.suspicious_layers
    tags = {r.input_tag for r in report.iteration_reports}
    assert any(t.startswith("gaussian_") for t in tags)
    assert any(t.startswith("boundary_") for t in tags)


@needs_torch
def test_auto_mode_token_model_runs():
    fuzzer = DynamicTrojanFuzzer(TinyTokenModel(vocab_size=100))
    report = fuzzer.run_fuzzing(vocab_size=100, seq_len=8, num_iterations=5, seed=1)
    assert report.num_fuzz_samples == 5
    assert all(r.input_tag.startswith("token_") for r in report.iteration_reports)


@needs_torch
def test_custom_generator_still_requires_baseline():
    fuzzer = DynamicTrojanFuzzer(SimpleLinearModel())
    with pytest.raises(ValueError):
        fuzzer.run_fuzzing(sample_inputs_generator=lambda i: torch.randn(2, 10))


# ---------------------------------------------------------------------------
# Issue #5: multi-sample clean calibration and sub-network isolation
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    from aegis import classify_module_region

    class BackdoorNet(nn.Module):
        """Surge is injected after layer1, so it first shows up at layer2."""

        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(8, 16)
            self.layer2 = nn.Linear(16, 4)

        def forward(self, x):
            h = torch.relu(self.layer1(x))
            if (x > 5.0).any():
                h = h * 25.0
            return self.layer2(h)

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.norm = nn.LayerNorm(8)
            self.mlp = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(20, 8)
            self.block = TinyBlock()
            self.classifier = nn.Linear(8, 3)


@needs_torch
def test_calibration_prevents_false_alarm_on_benign_noise():
    """Old behaviour: zeros baseline made ordinary randn inputs look like spikes."""
    torch.manual_seed(0)
    fuzzer = DynamicTrojanFuzzer(BackdoorNet(), spike_threshold=4.0)
    report = fuzzer.run_fuzzing(
        sample_inputs_generator=lambda i: torch.randn(2, 8),
        baseline_input=torch.zeros(2, 8),
        num_iterations=15,
        seed=0,
    )
    assert not report.suspected_trojan
    assert report.layer_anomalies == []
    # baseline_input + 64 noisy calibration samples (the default num_calibration)
    assert report.baseline_stats["layer1"].num_samples == 65
    assert report.baseline_stats["layer1"].std_l_inf > 0


@needs_torch
def test_isolation_localizes_trigger_layer_and_input():
    torch.manual_seed(0)
    fuzzer = DynamicTrojanFuzzer(BackdoorNet(), spike_threshold=4.0)
    report = fuzzer.run_fuzzing(
        sample_inputs_generator=lambda i: torch.full((2, 8), 10.0 if i == 5 else 0.5),
        baseline_input=torch.zeros(2, 8),
        num_iterations=10,
        seed=0,
    )
    assert report.suspected_trojan
    top = report.layer_anomalies[0]  # sorted strongest first
    assert top.layer_name == "layer2"
    assert top.region == "head" and top.depth == "late"
    assert top.trigger_iteration == 6  # i == 5 -> iteration 6
    assert top.spike_ratio >= 4.0 and top.z_score > 3.0
    assert "layer2" in report.anomaly_regions["head"]
    assert report.spike_ratio == top.spike_ratio


@needs_torch
def test_explicit_calibration_inputs_are_used():
    fuzzer = DynamicTrojanFuzzer(SimpleLinearModel())
    cal = [torch.randn(2, 10) for _ in range(5)]
    report = fuzzer.run_fuzzing(
        sample_inputs_generator=lambda i: torch.randn(2, 10),
        baseline_input=torch.zeros(2, 10),
        num_iterations=3,
        calibration_inputs=cal,
    )
    assert report.baseline_stats["fc1"].num_samples == 6


@needs_torch
def test_classify_module_region():
    model = TinyTransformer()
    modules = dict(model.named_modules())
    assert classify_module_region("block.attn", modules["block.attn"]) == "attention"
    assert classify_module_region("block.mlp.0", modules["block.mlp.0"]) == "mlp"
    assert classify_module_region("block.norm", modules["block.norm"]) == "norm"
    assert classify_module_region("embed", modules["embed"]) == "embedding"
    assert classify_module_region("classifier", modules["classifier"]) == "head"
    assert classify_module_region("block.mlp.1", modules["block.mlp.1"], is_last=True) == "head"


@needs_torch
def test_fuzz_model_without_input_shape_or_vocab_size_is_refused():
    """Neither a float input shape nor a token vocabulary: a clear error, not torch.zeros(None)."""
    from aegis.modelload import fuzz_model

    with pytest.raises(ValueError, match=r"input_shape .* or vocab_size"):
        fuzz_model(SimpleLinearModel(), spike_threshold=3.0, iterations=1, seed=0)


@needs_torch
def test_hook_manager_finds_tensors_inside_dataclass_outputs():
    @dataclass
    class Output:
        label: str
        logits: "torch.Tensor"

    tensor = torch.ones(2)
    assert ActivationHookManager._find_tensor(Output("x", tensor)) is tensor
    assert ActivationHookManager._find_tensor(Output("x", None)) is None
