"""Tests for neural network components."""

import pytest
import torch
import torch.nn as nn

from src.models.components import (
    DropConnect,
    SepConvLayer,
    SqueezeAndExcitation,
    WeightNormalizedConv2d,
)


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


class TestSqueezeAndExcitation:
    """Test suite for Squeeze-and-Excitation block."""

    @pytest.fixture
    def se_block(self) -> SqueezeAndExcitation:
        """Create a standard SqueezeAndExcitation block for testing.

        Returns:
            SqueezeAndExcitation block with typical configuration
        """
        return SqueezeAndExcitation(channels=64, reduction=4)

    def test_output_shape_matches_input(self, se_block: SqueezeAndExcitation) -> None:
        """Test that output shape matches input shape.

        SE block should preserve spatial dimensions and channel count.
        """
        batch_size = 8
        channels = 64
        height = 32
        width = 32

        x = torch.randn(batch_size, channels, height, width)
        output = se_block(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        assert output.shape == (batch_size, channels, height, width)

    def test_scaling_factors_in_range(self, se_block: SqueezeAndExcitation) -> None:
        """Test that scaling factors are in [0, 1] range.

        The sigmoid activation should produce values between 0 and 1.
        """
        x = torch.randn(4, 64, 16, 16)

        # Extract intermediate scale values by modifying forward pass
        z = x.mean(dim=(-2, -1), keepdim=True)
        z = se_block.fc1(z)
        z = se_block.mish(z)
        z = se_block.fc2(z)
        scale = se_block.sigmoid(z)

        # Verify scale is in [0, 1]
        assert torch.all(scale >= 0.0), f"Scale min={scale.min():.6f} should be >= 0"
        assert torch.all(scale <= 1.0), f"Scale max={scale.max():.6f} should be <= 1"

        print(
            f"\nScale range: min={scale.min():.6f}, max={scale.max():.6f}, mean={scale.mean():.6f}"
        )

    def test_forward_pass_various_sizes(self) -> None:
        """Test forward pass with various input sizes.

        SE block should work with different spatial dimensions and batch sizes.
        """
        test_configs = [
            (1, 64, 16, 16),  # Single sample
            (8, 64, 32, 32),  # Standard batch
            (16, 64, 64, 128),  # Large spatial dims
            (4, 64, 8, 8),  # Small spatial dims
        ]

        for batch_size, channels, height, width in test_configs:
            se = SqueezeAndExcitation(channels=channels, reduction=4)
            x = torch.randn(batch_size, channels, height, width)
            output = se(x)

            assert output.shape == (batch_size, channels, height, width), (
                f"Failed for shape {(batch_size, channels, height, width)}"
            )
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

    def test_parameter_count(self) -> None:
        """Test parameter count is approximately 2*C²/r + C + C/r (including bias).

        For C=64, r=4: expected params ≈ 2 * 64 * 64/4 + 64 + 16 = 2128
        Actual: FC1(64→16): 64*16 + 16 = 1040, FC2(16→64): 16*64 + 64 = 1088, Total = 2128
        """
        channels = 64
        reduction = 4
        se = SqueezeAndExcitation(channels=channels, reduction=reduction)

        # Count parameters
        total_params = sum(p.numel() for p in se.parameters())

        # Expected: FC1(C→C/r + bias) + FC2(C/r→C + bias) = C*(C/r) + (C/r) + (C/r)*C + C
        expected_params = (
            2 * channels * (channels // reduction) + channels + (channels // reduction)
        )

        assert total_params == expected_params, (
            f"Expected {expected_params} params, got {total_params}"
        )

        print(f"\nSE block parameters (C={channels}, r={reduction}): {total_params}")

    def test_gradient_flow(self, se_block: SqueezeAndExcitation) -> None:
        """Test that gradients propagate correctly through SE block.

        Verifies:
        - Gradients are computed for all parameters
        - Gradient flow is not blocked
        - Gradients are not extreme
        """
        se_block.train()

        # Create dummy input and target
        x = torch.randn(4, 64, 32, 32, requires_grad=True)
        target = torch.randn(4, 64, 32, 32)

        # Forward pass
        output = se_block(x)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in se_block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

            grad_norm = param.grad.norm().item()
            assert grad_norm < 1000, f"Parameter {name} gradient norm {grad_norm} is too large"
            assert grad_norm > 1e-7, f"Parameter {name} gradient norm {grad_norm} is too small"

        # Check input gradient
        assert x.grad is not None, "Input should have gradient"
        input_grad_norm = x.grad.norm().item()
        print(f"\nInput gradient norm: {input_grad_norm:.6f}")

    def test_different_reduction_ratios(self) -> None:
        """Test SE block with various reduction ratios."""
        channels = 64
        reduction_ratios = [2, 4, 8, 16]

        for reduction in reduction_ratios:
            se = SqueezeAndExcitation(channels=channels, reduction=reduction)
            x = torch.randn(4, channels, 16, 16)
            output = se(x)

            # Verify output shape
            assert output.shape == x.shape, f"Reduction {reduction}: shape mismatch"

            # Verify parameter count (including bias)
            expected_params = (
                2 * channels * (channels // reduction) + channels + (channels // reduction)
            )
            actual_params = sum(p.numel() for p in se.parameters())
            assert actual_params == expected_params, (
                f"Reduction {reduction}: expected {expected_params} params, got {actual_params}"
            )

    def test_channel_recalibration(self) -> None:
        """Test that SE block actually recalibrates channels differently.

        SE should apply different scaling factors to different channels.
        """
        se = SqueezeAndExcitation(channels=64, reduction=4)

        # Create input with different channel statistics
        x = torch.randn(4, 64, 16, 16)

        # Get the scaling factors
        z = x.mean(dim=(-2, -1), keepdim=True)
        z = se.fc1(z)
        z = se.mish(z)
        z = se.fc2(z)
        scale = se.sigmoid(z)

        # Check that not all channels are scaled equally
        scale_per_channel = scale.squeeze()  # (B, C)
        scale_std = scale_per_channel.std(dim=1).mean().item()

        # There should be some variance in scaling factors across channels
        assert scale_std > 0.01, (
            f"Scale std={scale_std:.6f} is too small - SE might not be recalibrating"
        )

        print(f"\nChannel-wise scale std: {scale_std:.6f}")

    def test_numerical_stability(self) -> None:
        """Test SE block handles extreme inputs gracefully."""
        se = SqueezeAndExcitation(channels=32, reduction=4)

        # Test with very small values
        x_small = torch.randn(2, 32, 16, 16) * 1e-6
        output_small = se(x_small)
        assert not torch.isnan(output_small).any(), "Should handle small inputs"
        assert not torch.isinf(output_small).any(), "Should handle small inputs"

        # Test with very large values
        x_large = torch.randn(2, 32, 16, 16) * 1e3
        output_large = se(x_large)
        assert not torch.isnan(output_large).any(), "Should handle large inputs"
        assert not torch.isinf(output_large).any(), "Should handle large inputs"

    def test_preserves_device(self) -> None:
        """Test that SE block preserves tensor device.

        Verifies operations stay on the same device (important for GPU training).
        """
        se = SqueezeAndExcitation(channels=32, reduction=4)
        x = torch.randn(2, 32, 16, 16)

        initial_device = x.device
        output = se(x)

        assert output.device == initial_device, "Output device should match input device"

    def test_eval_mode_consistency(self, se_block: SqueezeAndExcitation) -> None:
        """Test that SE block produces consistent outputs in eval mode.

        Verifies deterministic behavior in evaluation mode.
        """
        se_block.eval()

        x = torch.randn(4, 64, 16, 16)

        # Multiple forward passes should produce identical results
        with torch.no_grad():
            output1 = se_block(x)
            output2 = se_block(x)

        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8), (
            "SE block should be deterministic in eval mode"
        )


class TestDropConnect:
    """Test suite for DropConnect regularization layer."""

    @pytest.fixture
    def drop_layer(self) -> DropConnect:
        """Create a standard DropConnect layer for testing.

        Returns:
            DropConnect layer with typical drop rate
        """
        return DropConnect(drop_rate=0.2)

    def test_training_mode_stochastic(self, drop_layer: DropConnect) -> None:
        """Test that outputs differ across forward passes in training mode.

        Verifies:
        - Stochastic behavior during training
        - Different forward passes produce different outputs
        """
        drop_layer.train()

        x = torch.randn(8, 64, 32, 32)

        # Multiple forward passes should produce different results
        output1 = drop_layer(x)
        output2 = drop_layer(x)

        # Outputs should differ due to randomness
        assert not torch.allclose(output1, output2, rtol=1e-5, atol=1e-6), (
            "DropConnect should be stochastic in training mode"
        )

        # Verify shapes match
        assert output1.shape == x.shape
        assert output2.shape == x.shape

        print(
            f"\nTraining mode: output1 mean={output1.mean():.6f}, output2 mean={output2.mean():.6f}"
        )

    def test_eval_mode_deterministic(self, drop_layer: DropConnect) -> None:
        """Test that outputs are identical across forward passes in eval mode.

        Verifies:
        - Deterministic behavior during evaluation
        - DropConnect acts as identity in eval mode
        """
        drop_layer.eval()

        x = torch.randn(8, 64, 32, 32)

        # Multiple forward passes should produce identical results
        with torch.no_grad():
            output1 = drop_layer(x)
            output2 = drop_layer(x)

        assert torch.allclose(output1, output2, rtol=1e-6, atol=1e-8), (
            "DropConnect should be deterministic in eval mode"
        )

        # In eval mode, output should be identical to input
        assert torch.allclose(output1, x, rtol=1e-6, atol=1e-8), (
            "DropConnect should be identity in eval mode"
        )

    def test_expected_value_preserved(self) -> None:
        """Test that expected mean is preserved due to scaling by 1/keep_prob.

        Verifies:
        - E[output] ≈ input due to proper scaling
        - Statistical properties maintained over multiple runs
        """
        drop_rate = 0.3
        drop_layer = DropConnect(drop_rate=drop_rate)
        drop_layer.train()

        # Use input with non-zero mean for meaningful relative error calculation
        x = torch.randn(16, 64, 32, 32) + 1.0
        input_mean = x.mean().item()

        # Average over multiple forward passes to estimate expected value
        num_runs = 100
        outputs = [drop_layer(x) for _ in range(num_runs)]

        # Compute mean across all runs
        mean_output = torch.stack(outputs).mean(dim=0)
        output_mean = mean_output.mean().item()

        # Expected value should be close to input mean
        # Allow 5% tolerance due to finite sampling
        relative_error = abs(output_mean - input_mean) / abs(input_mean)
        assert relative_error < 0.05, (
            f"Expected value not preserved: input_mean={input_mean:.6f}, "
            f"output_mean={output_mean:.6f}, relative_error={relative_error:.4f}"
        )

        print(
            f"\nExpected value test (drop_rate={drop_rate}): "
            f"input_mean={input_mean:.6f}, output_mean={output_mean:.6f}, "
            f"relative_error={relative_error:.4f}"
        )

    def test_drop_rate_zero_identity(self) -> None:
        """Test that drop_rate=0.0 acts as identity (no dropping).

        Verifies:
        - No channels dropped when drop_rate=0.0
        - Output identical to input
        """
        drop_layer = DropConnect(drop_rate=0.0)
        drop_layer.train()

        x = torch.randn(8, 64, 32, 32)
        output = drop_layer(x)

        # Should be identical to input
        assert torch.allclose(output, x, rtol=1e-6, atol=1e-8), (
            "drop_rate=0.0 should produce identity output"
        )

    def test_drop_rate_one_all_zeros(self) -> None:
        """Test that drop_rate=1.0 produces all zeros.

        Verifies:
        - All channels dropped when drop_rate=1.0
        - Output is all zeros
        """
        drop_layer = DropConnect(drop_rate=1.0)
        drop_layer.train()

        x = torch.randn(8, 64, 32, 32)
        output = drop_layer(x)

        # Should be all zeros
        assert torch.allclose(output, torch.zeros_like(output), rtol=1e-6, atol=1e-8), (
            "drop_rate=1.0 should produce all-zero output"
        )

    def test_output_shape_preserved(self, drop_layer: DropConnect) -> None:
        """Test that output shape matches input shape.

        Verifies DropConnect preserves tensor dimensions.
        """
        test_configs = [
            (1, 64, 16, 16),  # Single sample
            (8, 32, 32, 32),  # Standard batch
            (16, 128, 64, 128),  # Large spatial dims
            (4, 256, 8, 8),  # Many channels, small spatial
        ]

        drop_layer.train()

        for batch_size, channels, height, width in test_configs:
            x = torch.randn(batch_size, channels, height, width)
            output = drop_layer(x)

            assert output.shape == x.shape, (
                f"Shape mismatch for input {x.shape}: got {output.shape}"
            )

    def test_channel_wise_dropout(self) -> None:
        """Test that entire channels are dropped (not individual activations).

        Verifies:
        - Dropout is applied per channel (across all spatial locations)
        - When a channel is dropped, all spatial locations are zero
        """
        drop_layer = DropConnect(drop_rate=0.5)
        drop_layer.train()

        # Use large batch and spatial dims to make this clearer
        x = torch.ones(4, 16, 32, 32)
        output = drop_layer(x)

        # For each sample and channel, check if entire spatial map is dropped
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                channel_slice = output[b, c, :, :]
                # Channel should be either all scaled (non-zero) or all zero
                if channel_slice.abs().max() > 1e-6:
                    # Channel is kept - should have uniform non-zero values
                    assert channel_slice.abs().min() > 1e-6, (
                        f"Channel ({b}, {c}) should be uniformly dropped or kept"
                    )
                else:
                    # Channel is dropped - should be all zeros
                    assert torch.allclose(channel_slice, torch.zeros_like(channel_slice)), (
                        f"Channel ({b}, {c}) should be all zeros if dropped"
                    )

    def test_gradient_flow(self) -> None:
        """Test that gradients propagate correctly through DropConnect.

        Verifies:
        - Gradients are computed for input
        - Gradient flow is not blocked
        - Gradients are reasonable magnitude
        """
        drop_layer = DropConnect(drop_rate=0.2)
        drop_layer.train()

        x = torch.randn(4, 32, 16, 16, requires_grad=True)
        target = torch.randn(4, 32, 16, 16)

        # Forward pass
        output = drop_layer(x)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check input gradient exists
        assert x.grad is not None, "Input should have gradient"

        # Check gradient is reasonable
        grad_norm = x.grad.norm().item()
        assert grad_norm < 1000, f"Gradient norm {grad_norm} is too large"
        assert grad_norm > 1e-7, f"Gradient norm {grad_norm} is too small (likely dead)"

        print(f"\nInput gradient norm: {grad_norm:.6f}")

    def test_different_drop_rates(self) -> None:
        """Test DropConnect with various drop rates."""
        drop_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

        x = torch.randn(8, 64, 16, 16)

        for drop_rate in drop_rates:
            drop_layer = DropConnect(drop_rate=drop_rate)
            drop_layer.train()

            output = drop_layer(x)

            # Verify shape preserved
            assert output.shape == x.shape, f"drop_rate={drop_rate}: shape mismatch"

            # Verify no NaN or Inf
            assert not torch.isnan(output).any(), f"drop_rate={drop_rate}: contains NaN"
            assert not torch.isinf(output).any(), f"drop_rate={drop_rate}: contains Inf"

    def test_numerical_stability(self) -> None:
        """Test DropConnect handles extreme inputs gracefully."""
        drop_layer = DropConnect(drop_rate=0.3)
        drop_layer.train()

        # Test with very small values
        x_small = torch.randn(2, 32, 16, 16) * 1e-6
        output_small = drop_layer(x_small)
        assert not torch.isnan(output_small).any(), "Should handle small inputs"
        assert not torch.isinf(output_small).any(), "Should handle small inputs"

        # Test with very large values
        x_large = torch.randn(2, 32, 16, 16) * 1e3
        output_large = drop_layer(x_large)
        assert not torch.isnan(output_large).any(), "Should handle large inputs"
        assert not torch.isinf(output_large).any(), "Should handle large inputs"

    def test_preserves_device(self) -> None:
        """Test that DropConnect preserves tensor device.

        Verifies operations stay on the same device (important for GPU training).
        """
        drop_layer = DropConnect(drop_rate=0.2)
        drop_layer.train()

        x = torch.randn(4, 32, 16, 16)
        initial_device = x.device

        output = drop_layer(x)

        assert output.device == initial_device, "Output device should match input device"

    def test_no_parameters(self) -> None:
        """Test that DropConnect has no learnable parameters.

        DropConnect is a regularization technique with no trainable weights.
        """
        drop_layer = DropConnect(drop_rate=0.2)

        # Should have no parameters
        total_params = sum(p.numel() for p in drop_layer.parameters())
        assert total_params == 0, f"DropConnect should have 0 parameters, got {total_params}"


class TestSepConvLayer:
    """Test suite for SepConvLayer component."""

    @pytest.fixture
    def sepconv_layer(self) -> SepConvLayer:
        """Create a standard SepConvLayer for testing.

        Returns:
            SepConvLayer with typical configuration
        """
        return SepConvLayer(
            in_channels=64, out_channels=128, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

    def test_forward_pass_output_shape(self, sepconv_layer: SepConvLayer) -> None:
        """Test that forward pass produces correct output shape.

        Verifies:
        - Output has expected shape
        - Spatial dimensions preserved (same padding)
        - Channel dimension matches out_channels
        """
        batch_size = 8
        in_channels = 64
        out_channels = 128
        height = 32
        width = 32

        x = torch.randn(batch_size, in_channels, height, width)
        output = sepconv_layer(x)

        expected_shape = (batch_size, out_channels, height, width)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

        # Verify no NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_parameter_count_reduction(self) -> None:
        """Test parameter count breakdown of SepConvLayer vs standard Conv2d.

        Note: While depthwise separable convolution alone achieves ~8x reduction,
        the SE block adds significant parameters. The tradeoff is:
        - More parameters than plain Conv2d (due to SE block)
        - But better feature quality and model expressiveness
        - Still efficient compared to multiple stacked Conv2d layers

        Breakdown for 64->128 channels, expansion=4:
        - Standard Conv2d (3x3): 64 x 128 x 3 x 3 = 73,728 params
        - SepConv components:
          - Expansion (1x1): 64 x 256 = 16,384
          - Depthwise (3x3): 256 x 3 x 3 = 2,304
          - SE block: ~33k params (main overhead)
          - Pointwise (1x1): 256 x 128 = 32,768
          - GroupNorm: 512
          - Total: ~85k params
        """
        in_channels = 64
        out_channels = 128
        expansion = 4

        # Standard Conv2d
        standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        standard_params = sum(p.numel() for p in standard_conv.parameters())

        # SepConvLayer with SE block
        sepconv = SepConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=expansion,
            se_reduction=4,
            drop_connect_rate=0.2,
        )
        sepconv_params = sum(p.numel() for p in sepconv.parameters())

        # Break down SepConv parameters
        expand_params = sum(p.numel() for p in sepconv.expand.parameters())
        depthwise_params = sum(p.numel() for p in sepconv.depthwise.parameters())
        se_params = sum(p.numel() for p in sepconv.se.parameters())
        project_params = sum(p.numel() for p in sepconv.project.parameters())
        norm_params = sum(p.numel() for p in sepconv.norm.parameters())

        print(
            f"\nParameter breakdown:"
            f"\n  Standard Conv2d (3x3): {standard_params:,}"
            f"\n  SepConv total: {sepconv_params:,}"
            f"\n    - Expansion (1x1): {expand_params:,}"
            f"\n    - Depthwise (3x3): {depthwise_params:,}"
            f"\n    - SE block: {se_params:,}"
            f"\n    - Projection (1x1): {project_params:,}"
            f"\n    - GroupNorm: {norm_params:,}"
        )

        # Verify depthwise conv is much smaller than standard conv
        # (depthwise alone should be ~32x smaller)
        assert depthwise_params < standard_params / 20, (
            f"Depthwise ({depthwise_params:,}) should be much smaller than "
            f"standard conv ({standard_params:,})"
        )

        # Verify SE block has expected size (should be largest component)
        expected_channels = in_channels * expansion
        se_reduction = 4
        expected_se_params = (
            expected_channels * (expected_channels // se_reduction)  # FC1 weights
            + (expected_channels // se_reduction)  # FC1 bias
            + (expected_channels // se_reduction) * expected_channels  # FC2 weights
            + expected_channels  # FC2 bias
        )
        assert se_params == expected_se_params, (
            f"SE params ({se_params:,}) should match expected ({expected_se_params:,})"
        )

        # Verify total parameter count is reasonable (between 70k-90k for this config)
        assert 70000 < sepconv_params < 90000, (
            f"Total params ({sepconv_params:,}) outside expected range"
        )

    def test_weight_normalization_integration(self) -> None:
        """Test that weight normalization applies to expansion and depthwise convs.

        Verifies:
        - Expansion conv is WeightNormalizedConv2d
        - Depthwise conv is WeightNormalizedConv2d
        - weight_scaling() can be called successfully
        """
        layer = SepConvLayer(
            in_channels=64, out_channels=128, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

        # Check that expansion and depthwise are WeightNormalizedConv2d
        assert isinstance(layer.expand, WeightNormalizedConv2d), (
            "Expansion should be WeightNormalizedConv2d"
        )
        assert isinstance(layer.depthwise, WeightNormalizedConv2d), (
            "Depthwise should be WeightNormalizedConv2d"
        )
        assert isinstance(layer.project, WeightNormalizedConv2d), (
            "Projection should be WeightNormalizedConv2d"
        )

        # Test that weight_scaling() can be called
        layer.expand.weight_scaling()
        layer.depthwise.weight_scaling()
        layer.project.weight_scaling()

        # Verify weights are normalized (unit L2 norm)
        expand_norms = torch.sqrt((layer.expand.weight**2).sum(dim=(1, 2, 3)))
        depthwise_norms = torch.sqrt((layer.depthwise.weight**2).sum(dim=(1, 2, 3)))
        project_norms = torch.sqrt((layer.project.weight**2).sum(dim=(1, 2, 3)))

        assert torch.allclose(expand_norms, torch.ones_like(expand_norms), rtol=1e-4, atol=1e-4), (
            "Expansion weights should have unit norm after scaling"
        )
        assert torch.allclose(
            depthwise_norms, torch.ones_like(depthwise_norms), rtol=1e-4, atol=1e-4
        ), "Depthwise weights should have unit norm after scaling"
        assert torch.allclose(
            project_norms, torch.ones_like(project_norms), rtol=1e-4, atol=1e-4
        ), "Project weights should have unit norm after scaling"

    def test_se_block_integration(self, sepconv_layer: SepConvLayer) -> None:
        """Test that SE block is integrated correctly.

        Verifies:
        - SE block is present
        - SE block operates on expanded channels
        - SE output affects final output
        """
        # Check SE block exists
        assert isinstance(sepconv_layer.se, SqueezeAndExcitation), "Layer should contain SE block"

        # Verify SE operates on expanded channels (64 * 4 = 256)
        expected_channels = 64 * 4
        assert sepconv_layer.se.fc1.in_channels == expected_channels, (
            f"SE should operate on {expected_channels} channels"
        )

        # Test that SE block affects output
        x = torch.randn(4, 64, 16, 16)

        # Forward pass with SE
        sepconv_layer.train()
        output_with_se = sepconv_layer(x)

        # Manually bypass SE (replace with identity temporarily)
        original_se = sepconv_layer.se
        sepconv_layer.se = nn.Identity()
        output_without_se = sepconv_layer(x)
        sepconv_layer.se = original_se

        # Outputs should differ (SE modifies features)
        assert not torch.allclose(output_with_se, output_without_se, rtol=1e-3, atol=1e-5), (
            "SE block should affect output"
        )

    def test_drop_connect_train_eval_modes(self) -> None:
        """Test that DropConnect works correctly in train/eval modes.

        Verifies:
        - Training mode: stochastic behavior
        - Eval mode: deterministic (identity)
        """
        layer = SepConvLayer(
            in_channels=32, out_channels=64, expansion=4, se_reduction=4, drop_connect_rate=0.3
        )

        x = torch.randn(8, 32, 16, 16)

        # Training mode: should be stochastic
        layer.train()
        output1 = layer(x)
        output2 = layer(x)

        # Outputs should differ due to DropConnect randomness
        assert not torch.allclose(output1, output2, rtol=1e-5, atol=1e-6), (
            "Training mode should produce stochastic outputs"
        )

        # Eval mode: should be deterministic
        layer.eval()
        with torch.no_grad():
            output3 = layer(x)
            output4 = layer(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output3, output4, rtol=1e-6, atol=1e-8), (
            "Eval mode should produce deterministic outputs"
        )

    def test_gradient_flow(self) -> None:
        """Test that gradients propagate through all components.

        Verifies:
        - Gradients reach all learnable parameters
        - No gradient vanishing or explosion
        - Backward pass succeeds
        """
        layer = SepConvLayer(
            in_channels=32, out_channels=64, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )
        layer.train()

        x = torch.randn(4, 32, 16, 16, requires_grad=True)
        target = torch.randn(4, 64, 16, 16)

        # Forward pass
        output = layer(x)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check that all parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

            grad_norm = param.grad.norm().item()
            assert grad_norm < 1000, f"Parameter {name} gradient norm {grad_norm} is too large"
            assert grad_norm > 1e-7, f"Parameter {name} gradient norm {grad_norm} is too small"

        # Check input gradient
        assert x.grad is not None, "Input should have gradient"
        input_grad_norm = x.grad.norm().item()
        print(f"\nInput gradient norm: {input_grad_norm:.6f}")

    def test_expansion_factor_one(self) -> None:
        """Test SepConvLayer with expansion=1 (no expansion).

        Verifies:
        - Layer works without expansion
        - Expansion layer becomes Identity
        """
        layer = SepConvLayer(
            in_channels=64, out_channels=128, expansion=1, se_reduction=4, drop_connect_rate=0.2
        )

        # Check that expansion is Identity
        assert isinstance(layer.expand, nn.Identity), "expansion=1 should use Identity"

        # Forward pass should work
        x = torch.randn(4, 64, 16, 16)
        output = layer(x)

        assert output.shape == (4, 128, 16, 16), "Output shape should be correct with expansion=1"

    def test_various_configurations(self) -> None:
        """Test SepConvLayer with various hyperparameter configurations.

        Verifies flexibility and robustness across different settings.
        """
        test_configs = [
            # (in_channels, out_channels, expansion, se_reduction, drop_rate)
            (32, 64, 2, 4, 0.1),
            (64, 128, 4, 4, 0.2),
            (128, 256, 6, 8, 0.3),
            (64, 64, 4, 4, 0.0),  # Same in/out channels, no dropout
            (32, 128, 1, 2, 0.5),  # No expansion, high dropout
        ]

        for in_ch, out_ch, exp, se_red, drop_rate in test_configs:
            layer = SepConvLayer(
                in_channels=in_ch,
                out_channels=out_ch,
                expansion=exp,
                se_reduction=se_red,
                drop_connect_rate=drop_rate,
            )

            x = torch.randn(4, in_ch, 16, 16)
            output = layer(x)

            assert output.shape == (4, out_ch, 16, 16), (
                f"Config ({in_ch}, {out_ch}, {exp}, {se_red}, {drop_rate}) failed"
            )
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

    def test_depthwise_convolution(self) -> None:
        """Test that depthwise convolution is configured correctly.

        Verifies:
        - Depthwise conv has groups = expanded_channels
        - Each channel has its own filter
        """
        layer = SepConvLayer(
            in_channels=64, out_channels=128, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

        expanded_channels = 64 * 4  # 256

        # Check depthwise conv configuration
        assert layer.depthwise.groups == expanded_channels, (
            f"Depthwise should have groups={expanded_channels}"
        )
        assert layer.depthwise.in_channels == expanded_channels, (
            f"Depthwise should have in_channels={expanded_channels}"
        )
        assert layer.depthwise.out_channels == expanded_channels, (
            f"Depthwise should have out_channels={expanded_channels}"
        )
        assert layer.depthwise.kernel_size == (3, 3), "Depthwise should use 3x3 kernel"
        assert layer.depthwise.padding == (1, 1), "Depthwise should use padding=1"

    def test_normalization_layer(self) -> None:
        """Test that normalization layer is configured correctly.

        Verifies:
        - GroupNorm is used
        - num_groups=1 creates LayerNorm-like behavior
        """
        layer = SepConvLayer(
            in_channels=64,
            out_channels=128,
            expansion=4,
            se_reduction=4,
            drop_connect_rate=0.2,
            norm_groups=1,
        )

        expanded_channels = 64 * 4  # 256

        # Check normalization
        assert isinstance(layer.norm, nn.GroupNorm), "Should use GroupNorm"
        assert layer.norm.num_groups == 1, "Should use num_groups=1 (LayerNorm-like)"
        assert layer.norm.num_channels == expanded_channels, (
            f"Should normalize {expanded_channels} channels"
        )

    def test_activation_function(self) -> None:
        """Test that activation function is configured correctly.

        Verifies:
        - Default is Mish
        - Can use ReLU as alternative
        """
        # Test Mish activation (default)
        layer_mish = SepConvLayer(in_channels=32, out_channels=64, expansion=4, activation="Mish")
        assert isinstance(layer_mish.activation, nn.Mish), "Should use Mish activation"

        # Test ReLU activation
        layer_relu = SepConvLayer(in_channels=32, out_channels=64, expansion=4, activation="ReLU")
        assert isinstance(layer_relu.activation, nn.ReLU), "Should use ReLU activation"

    def test_numerical_stability(self) -> None:
        """Test SepConvLayer handles extreme inputs gracefully."""
        layer = SepConvLayer(
            in_channels=32, out_channels=64, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

        # Test with very small values
        x_small = torch.randn(2, 32, 16, 16) * 1e-6
        output_small = layer(x_small)
        assert not torch.isnan(output_small).any(), "Should handle small inputs"
        assert not torch.isinf(output_small).any(), "Should handle small inputs"

        # Test with very large values
        x_large = torch.randn(2, 32, 16, 16) * 1e3
        output_large = layer(x_large)
        assert not torch.isnan(output_large).any(), "Should handle large inputs"
        assert not torch.isinf(output_large).any(), "Should handle large inputs"

    def test_preserves_device(self) -> None:
        """Test that SepConvLayer preserves tensor device.

        Verifies operations stay on the same device (important for GPU training).
        """
        layer = SepConvLayer(
            in_channels=32, out_channels=64, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

        x = torch.randn(2, 32, 16, 16)
        initial_device = x.device

        output = layer(x)

        assert output.device == initial_device, "Output device should match input device"

    def test_various_spatial_dimensions(self) -> None:
        """Test SepConvLayer with various spatial dimensions.

        Verifies layer works with different input sizes.
        """
        layer = SepConvLayer(
            in_channels=64, out_channels=128, expansion=4, se_reduction=4, drop_connect_rate=0.2
        )

        test_shapes = [
            (4, 64, 8, 8),  # Small spatial dims
            (4, 64, 16, 16),  # Medium spatial dims
            (4, 64, 32, 32),  # Standard spatial dims
            (4, 64, 64, 128),  # Large, non-square spatial dims
        ]

        for batch, channels, height, width in test_shapes:
            x = torch.randn(batch, channels, height, width)
            output = layer(x)

            expected_shape = (batch, 128, height, width)
            assert output.shape == expected_shape, (
                f"Failed for input shape {x.shape}, expected {expected_shape}, got {output.shape}"
            )
