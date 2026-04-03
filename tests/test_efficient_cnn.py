"""Tests for EfficientAudioCNN model."""

import pytest
import torch

from src.core.config import ModelConfig
from src.models.components import WeightNormalizedConv2d
from src.models.efficient_cnn import EfficientAudioCNN


@pytest.fixture
def model_config() -> ModelConfig:
    """Create standard model configuration for testing."""
    return ModelConfig(
        encoder_channels=[24, 32, 48, 96],
        encoder_repeats=[2, 2, 3, 3],
        expansions=[3, 3, 4, 4],  # Reduced from [4, 4, 6, 6] to fit 300-500k params
        activation="Mish",
        use_se=True,
        se_reduction=4,
        drop_connect_rate=0.2,
        dropout=0.4,
        num_classes=80,
    )


@pytest.fixture
def model(model_config: ModelConfig) -> EfficientAudioCNN:
    """Create model instance for testing."""
    return EfficientAudioCNN(model_config)


def test_forward_pass_shape(model: EfficientAudioCNN) -> None:
    """Test that forward pass produces correct output shape."""
    batch_size = 4
    height = 128  # Frequency bins
    width = 431  # Time frames

    x = torch.randn(batch_size, 1, height, width)
    output = model(x)

    assert output.shape == (batch_size, 80), f"Expected (4, 80), got {output.shape}"


def test_no_nan_inf_in_outputs(model: EfficientAudioCNN) -> None:
    """Test that outputs contain no NaN or Inf values."""
    x = torch.randn(4, 1, 128, 431)
    output = model(x)

    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_parameter_count(model: EfficientAudioCNN) -> None:
    """Test that parameter count is in target range (300-500k)."""
    num_params = model.get_num_parameters()

    print(f"\nTotal parameters: {num_params:,}")
    print("Target range: 300,000 - 500,000")

    assert 300_000 <= num_params <= 500_000, (
        f"Parameter count {num_params:,} outside target range [300k, 500k]"
    )


def test_variable_time_dimension(model: EfficientAudioCNN) -> None:
    """Test that model handles variable time dimensions correctly."""
    batch_size = 2

    # Test different time dimensions
    time_dims = [100, 215, 431, 500]

    for time_dim in time_dims:
        x = torch.randn(batch_size, 1, 128, time_dim)
        output = model(x)

        assert output.shape == (batch_size, 80), (
            f"Failed for time_dim={time_dim}: expected (2, 80), got {output.shape}"
        )
        assert not torch.isnan(output).any(), f"NaN for time_dim={time_dim}"


def test_weight_normalization_applied(model: EfficientAudioCNN) -> None:
    """Test that weight normalization is applied correctly."""
    # Get initial weight norms
    initial_norms = []
    for module in model.modules():
        if isinstance(module, WeightNormalizedConv2d):
            # Compute L2 norm per filter (over dims 1,2,3)
            norm = torch.sqrt((module.weight**2).sum(dim=(1, 2, 3)))
            initial_norms.append(norm.clone())

    # Apply weight scaling
    model.apply_weight_scaling()

    # Check that norms are now close to 1.0
    idx = 0
    for module in model.modules():
        if isinstance(module, WeightNormalizedConv2d):
            norm = torch.sqrt((module.weight**2).sum(dim=(1, 2, 3)))

            # All norms should be ~1.0 after normalization
            # Use rtol to handle numerical precision issues
            assert torch.allclose(norm, torch.ones_like(norm), rtol=0.01, atol=0.01), (
                f"Weight norms not normalized: {norm.mean():.4f} ± {norm.std():.4f}"
            )

            # Norms should have changed from initial values (unless already normalized)
            # We allow small tolerance in case weights were already close to normalized
            assert not torch.allclose(norm, initial_norms[idx], atol=1e-6) or torch.allclose(
                initial_norms[idx], torch.ones_like(norm), atol=1e-4
            ), "Weight scaling did not modify weights"

            idx += 1


def test_se_blocks_output_range(model: EfficientAudioCNN) -> None:
    """Test that SE blocks produce outputs in valid range."""
    x = torch.randn(4, 1, 128, 431)

    # Register hooks to capture SE outputs
    se_outputs = []

    def hook_fn(module, input, output):
        se_outputs.append(output.clone())

    # Register hooks on all SE blocks
    from src.models.components import SqueezeAndExcitation

    hooks = []
    for module in model.modules():
        if isinstance(module, SqueezeAndExcitation):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Forward pass
    _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Check SE outputs
    assert len(se_outputs) > 0, "No SE blocks found in model"

    for i, se_out in enumerate(se_outputs):
        # SE outputs should not contain NaN/Inf
        assert not torch.isnan(se_out).any(), f"SE block {i} output contains NaN"
        assert not torch.isinf(se_out).any(), f"SE block {i} output contains Inf"

        # SE outputs should be reasonable (not exploding)
        assert se_out.abs().max() < 1e6, f"SE block {i} output too large: {se_out.abs().max()}"


def test_dropconnect_train_vs_eval(model: EfficientAudioCNN) -> None:
    """Test that DropConnect behaves differently in train vs eval mode."""
    torch.manual_seed(42)
    x = torch.randn(4, 1, 128, 431)

    # Training mode: outputs should differ across runs (stochastic)
    model.train()
    out1_train = model(x)
    out2_train = model(x)

    # Outputs should differ due to DropConnect randomness
    assert not torch.allclose(out1_train, out2_train, atol=1e-5), (
        "DropConnect outputs identical in train mode (should be stochastic)"
    )

    # Eval mode: outputs should be deterministic
    model.eval()
    out1_eval = model(x)
    out2_eval = model(x)

    # Outputs should be identical in eval mode
    assert torch.allclose(out1_eval, out2_eval, atol=1e-7), (
        "DropConnect outputs differ in eval mode (should be deterministic)"
    )


def test_gradient_flow(model: EfficientAudioCNN) -> None:
    """Test that gradients flow through all layers."""
    x = torch.randn(4, 1, 128, 431)
    target = torch.randint(0, 2, (4, 80)).float()  # Multi-label targets

    # Forward pass
    output = model(x)

    # Compute loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

    # Backward pass
    loss.backward()

    # Check that all parameters have gradients
    params_without_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)

    assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"

    # Check that gradients are not zero everywhere
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()

    assert total_grad_norm > 0, "All gradients are zero"


def test_model_device_handling(model: EfficientAudioCNN) -> None:
    """Test that model can be moved to different devices."""
    # Test CPU
    model_cpu = model.cpu()
    x_cpu = torch.randn(2, 1, 128, 431)
    out_cpu = model_cpu(x_cpu)
    assert out_cpu.device.type == "cpu"

    # Test MPS if available
    if torch.backends.mps.is_available():
        model_mps = model.to("mps")
        x_mps = torch.randn(2, 1, 128, 431, device="mps")
        out_mps = model_mps(x_mps)
        assert out_mps.device.type == "mps"


def test_batch_size_flexibility(model: EfficientAudioCNN) -> None:
    """Test that model handles different batch sizes."""
    for batch_size in [1, 4, 16, 32]:
        x = torch.randn(batch_size, 1, 128, 431)
        output = model(x)
        assert output.shape == (batch_size, 80), f"Failed for batch_size={batch_size}"
