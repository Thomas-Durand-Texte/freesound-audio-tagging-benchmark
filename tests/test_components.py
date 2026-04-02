"""Tests for neural network components."""

import pytest
import torch
import torch.nn as nn

from src.models.components import WeightNormalizedConv2d


class TestWeightNormalizedConv2d:
    """Test suite for WeightNormalizedConv2d component."""

    @pytest.fixture
    def conv_layer(self) -> WeightNormalizedConv2d:
        """Create a standard WeightNormalizedConv2d layer for testing.

        Returns:
            WeightNormalizedConv2d layer with typical configuration
        """
        return WeightNormalizedConv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
        )

    def test_inherits_from_conv2d(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test that WeightNormalizedConv2d properly inherits from nn.Conv2d.

        Verifies:
        - Is an instance of nn.Conv2d
        - Has all standard Conv2d attributes
        """
        assert isinstance(conv_layer, nn.Conv2d), "Should inherit from nn.Conv2d"
        assert hasattr(conv_layer, "weight"), "Should have weight parameter"
        assert hasattr(conv_layer, "forward"), "Should have forward method"

    def test_drop_in_replacement(self) -> None:
        """Test that WeightNormalizedConv2d can replace nn.Conv2d without changes.

        Verifies that it works as a drop-in replacement with identical API.
        """
        batch_size = 4
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        height = 64
        width = 128

        # Create both types of layers with same config
        standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        weight_norm_conv = WeightNormalizedConv2d(in_channels, out_channels, kernel_size, padding=1)

        # Copy weights to ensure identical behavior (before normalization)
        with torch.no_grad():
            weight_norm_conv.weight.copy_(standard_conv.weight)
            if standard_conv.bias is not None and weight_norm_conv.bias is not None:
                weight_norm_conv.bias.copy_(standard_conv.bias)

        # Test forward pass
        x = torch.randn(batch_size, in_channels, height, width)
        output_standard = standard_conv(x)
        output_weight_norm = weight_norm_conv(x)

        # Should produce identical outputs (before normalization)
        assert torch.allclose(output_standard, output_weight_norm, rtol=1e-5), (
            "Output should match standard Conv2d before weight normalization"
        )

        # Check output shape
        expected_shape = (batch_size, out_channels, height, width)
        assert output_weight_norm.shape == expected_shape, f"Expected shape {expected_shape}"

    def test_forward_pass(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test forward pass with standard input shape.

        Verifies:
        - Output has correct shape
        - No NaN or Inf values in output
        - Forward pass works after weight normalization
        """
        batch_size = 4
        in_channels = 64
        height = 32
        width = 64

        x = torch.randn(batch_size, in_channels, height, width)

        # Forward pass before normalization
        output_before = conv_layer(x)
        assert output_before.shape == (batch_size, 128, height, width)
        assert not torch.isnan(output_before).any(), "Output contains NaN before normalization"
        assert not torch.isinf(output_before).any(), "Output contains Inf before normalization"

        # Apply weight normalization
        conv_layer.weight_scaling()

        # Forward pass after normalization
        output_after = conv_layer(x)
        assert output_after.shape == (batch_size, 128, height, width)
        assert not torch.isnan(output_after).any(), "Output contains NaN after normalization"
        assert not torch.isinf(output_after).any(), "Output contains Inf after normalization"

    def test_weights_unit_norm_after_scaling(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test that weights have unit L2 norm after weight_scaling().

        Verifies:
        - Each filter has L2 norm = 1.0 (within numerical tolerance)
        - Normalization is applied per output channel
        """
        # Apply weight normalization
        conv_layer.weight_scaling()

        # Compute L2 norm per filter (over dims 1, 2, 3)
        # Shape of weight: (out_channels, in_channels, kernel_h, kernel_w)
        norms = torch.sqrt((conv_layer.weight**2).sum(dim=(1, 2, 3)))

        # All norms should be very close to 1.0 (within epsilon tolerance)
        expected_norms = torch.ones_like(norms)
        assert torch.allclose(norms, expected_norms, rtol=1e-4, atol=1e-4), (
            f"Filters should have unit L2 norm. Got min={norms.min():.6f}, "
            f"max={norms.max():.6f}, mean={norms.mean():.6f}"
        )

        print(
            f"\nFilter norms after weight_scaling(): min={norms.min():.6f}, "
            f"max={norms.max():.6f}, mean={norms.mean():.6f}"
        )

    def test_weight_scaling_idempotent(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test that calling weight_scaling() multiple times is stable.

        Verifies that applying normalization multiple times produces the same result.
        """
        # Apply normalization once
        conv_layer.weight_scaling()
        weights_after_first = conv_layer.weight.clone()

        # Apply normalization again
        conv_layer.weight_scaling()
        weights_after_second = conv_layer.weight.clone()

        # Weights should remain unchanged (idempotent operation)
        assert torch.allclose(weights_after_first, weights_after_second, rtol=1e-4, atol=1e-6), (
            "Weight normalization should be idempotent"
        )

    def test_gradient_flow(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test that gradients propagate correctly through normalized weights.

        Verifies:
        - Gradients are computed for weight parameter
        - Gradient flow is not blocked by weight normalization
        - Gradients are not extreme
        """
        conv_layer.train()

        # Create dummy input and target
        x = torch.randn(2, 64, 32, 32)
        target = torch.randn(2, 128, 32, 32)

        # Apply weight normalization
        conv_layer.weight_scaling()

        # Forward pass
        output = conv_layer(x)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradient exists
        assert conv_layer.weight.grad is not None, "Weight should have gradient"

        # Check gradient is not extreme
        grad_norm = conv_layer.weight.grad.norm().item()
        assert grad_norm < 1000, f"Gradient norm {grad_norm} is too large"
        assert grad_norm > 1e-7, f"Gradient norm {grad_norm} is too small (likely dead)"

        print(f"\nWeight gradient norm: {grad_norm:.6f}")

    def test_weight_scaling_numerical_stability(self) -> None:
        """Test weight_scaling() handles near-zero weights gracefully.

        Verifies that the epsilon term prevents division by zero when weights
        are very small.
        """
        conv = WeightNormalizedConv2d(in_channels=3, out_channels=16, kernel_size=3)

        # Set weights to very small values
        with torch.no_grad():
            conv.weight.fill_(1e-10)

        # Should not raise error or produce NaN/Inf
        conv.weight_scaling()

        assert not torch.isnan(conv.weight).any(), "Should not produce NaN with small weights"
        assert not torch.isinf(conv.weight).any(), "Should not produce Inf with small weights"

        # Weights should be non-zero and finite after normalization
        assert (conv.weight != 0).any(), "Weights should be non-zero after normalization"
        assert torch.all(torch.isfinite(conv.weight)), "All weights should be finite"

    def test_weight_scaling_preserves_device(self) -> None:
        """Test that weight_scaling() preserves tensor device.

        Verifies operations stay on the same device (important for GPU training).
        """
        conv = WeightNormalizedConv2d(in_channels=8, out_channels=16, kernel_size=3)

        # Get initial device
        initial_device = conv.weight.device

        # Apply normalization
        conv.weight_scaling()

        # Device should not change
        assert conv.weight.device == initial_device, "Weight device should not change"

    def test_no_grad_context(self, conv_layer: WeightNormalizedConv2d) -> None:
        """Test that weight_scaling() operates in no_grad context.

        Verifies that weight normalization doesn't create computation graph,
        which would waste memory during training.
        """
        conv_layer.train()

        # Enable gradient tracking
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = conv_layer(x)
        loss = output.sum()
        loss.backward()

        # Clear gradients
        conv_layer.zero_grad()

        # Apply weight scaling (should not track gradients)
        conv_layer.weight_scaling()

        # Weight should not have gradients from the normalization operation
        assert conv_layer.weight.grad is None or torch.all(conv_layer.weight.grad == 0), (
            "weight_scaling() should operate in no_grad context"
        )

    def test_different_kernel_sizes(self) -> None:
        """Test weight normalization with various kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for kernel_size in kernel_sizes:
            conv = WeightNormalizedConv2d(
                in_channels=32, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2
            )

            # Apply normalization
            conv.weight_scaling()

            # Check norms
            norms = torch.sqrt((conv.weight**2).sum(dim=(1, 2, 3)))
            assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4), (
                f"Kernel size {kernel_size}: weights should have unit norm"
            )

    def test_with_bias(self) -> None:
        """Test that weight normalization works correctly with bias enabled.

        Verifies that bias parameter is not affected by weight normalization.
        """
        conv = WeightNormalizedConv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=True
        )

        # Store initial bias
        initial_bias = conv.bias.clone()

        # Apply weight normalization
        conv.weight_scaling()

        # Bias should be unchanged
        assert torch.allclose(conv.bias, initial_bias), (
            "Bias should not be affected by weight normalization"
        )

        # Weights should still be normalized
        norms = torch.sqrt((conv.weight**2).sum(dim=(1, 2, 3)))
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4), (
            "Weights should be normalized even with bias enabled"
        )
