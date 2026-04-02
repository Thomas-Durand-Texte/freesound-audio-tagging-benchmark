"""Tests for audio tagging models."""

import pytest
import torch

from src.models.baseline_cnn import AudioCNN


class TestAudioCNN:
    """Test suite for AudioCNN baseline model."""

    @pytest.fixture
    def model(self) -> AudioCNN:
        """Create a standard AudioCNN model for testing.

        Returns:
            AudioCNN model with default test configuration
        """
        return AudioCNN(num_classes=80, channels=[16, 32, 64], dropout=0.4)

    def test_forward_pass(self, model: AudioCNN) -> None:
        """Test forward pass with standard input shape.

        Verifies:
        - Output has correct shape (batch_size, num_classes)
        - No NaN or Inf values in output
        """
        batch_size = 4
        freq_bins = 128
        time_steps = 431  # ~5 seconds at typical hop length

        x = torch.randn(batch_size, 1, freq_bins, time_steps)
        output = model(x)

        assert output.shape == (batch_size, 80), f"Expected shape (4, 80), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_variable_time_dimension(self, model: AudioCNN) -> None:
        """Test model handles variable time lengths via AdaptiveAvgPool.

        The global average pooling layer should handle different time dimensions
        while producing consistent output shape.
        """
        batch_size = 2
        freq_bins = 128
        num_classes = 80

        # Test with different time dimensions
        time_lengths = [100, 215, 431, 862]  # Various audio lengths

        for time_steps in time_lengths:
            x = torch.randn(batch_size, 1, freq_bins, time_steps)
            output = model(x)

            expected_shape = (batch_size, num_classes)
            assert output.shape == expected_shape, (
                f"Time length {time_steps}: expected shape {expected_shape}, got {output.shape}"
            )
            assert not torch.isnan(output).any(), f"Time length {time_steps}: output contains NaN"

    def test_parameter_count(self, model: AudioCNN) -> None:
        """Test parameter count is within expected range.

        Expected range: ~20k-50k parameters for baseline model.
        Prints actual count for documentation.
        """
        param_count = model.get_num_parameters()
        print(f"\nAudioCNN parameter count: {param_count:,}")

        # Verify parameter count is in reasonable range
        assert 10_000 < param_count < 100_000, (
            f"Parameter count {param_count:,} outside expected range (10k-100k). "
            "This may indicate an architectural issue."
        )

        # Also verify using PyTorch's parameter counting
        pytorch_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == pytorch_count, "Parameter count mismatch between methods"

    def test_output_range(self, model: AudioCNN) -> None:
        """Test output logits are in reasonable range.

        Logits should not be extreme (e.g., > 100 or < -100) as this indicates
        training instability or gradient issues.
        """
        x = torch.randn(4, 1, 128, 431)
        output = model(x)

        # Check logits are in reasonable range
        max_logit = output.max().item()
        min_logit = output.min().item()

        print(f"\nLogit range: [{min_logit:.2f}, {max_logit:.2f}]")

        assert -100 < min_logit < 100, f"Minimum logit {min_logit} is extreme"
        assert -100 < max_logit < 100, f"Maximum logit {max_logit} is extreme"

    def test_training_mode(self, model: AudioCNN) -> None:
        """Test dropout behaves differently in train vs eval mode.

        Dropout should be active during training and disabled during evaluation.
        This test verifies stochastic behavior in train mode and deterministic
        behavior in eval mode.
        """
        x = torch.randn(1, 1, 128, 431)

        # Test training mode - should produce different outputs due to dropout
        model.train()
        outputs_train = []
        for _ in range(5):
            output = model(x)
            outputs_train.append(output)

        # At least some outputs should differ (dropout is stochastic)
        differences = 0
        for i in range(1, len(outputs_train)):
            if not torch.allclose(outputs_train[0], outputs_train[i], rtol=1e-5):
                differences += 1

        assert differences > 0, "Training mode should produce varying outputs due to dropout"

        # Test eval mode - should produce identical outputs
        model.eval()
        outputs_eval = []
        for _ in range(3):
            with torch.no_grad():
                output = model(x)
                outputs_eval.append(output)

        # All eval outputs should be identical
        for i in range(1, len(outputs_eval)):
            assert torch.allclose(outputs_eval[0], outputs_eval[i]), (
                "Eval mode should produce identical outputs"
            )

    def test_gradient_flow(self, model: AudioCNN) -> None:
        """Test gradients propagate correctly through the network.

        Verifies:
        - Gradients are computed for all parameters
        - No dead neurons (all parameters receive gradients)
        - Gradients are not extreme (which could indicate instability)
        """
        model.train()

        # Create dummy input and target
        x = torch.randn(2, 1, 128, 431)
        target = torch.randint(0, 2, (2, 80)).float()  # Multi-label targets

        # Forward pass
        output = model(x)

        # Compute loss (BCEWithLogitsLoss for multi-label)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)

        # Backward pass
        loss.backward()

        # Check gradients exist for all parameters
        params_with_grad = 0
        params_without_grad = 0
        max_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)

                if grad_norm > 1e-7:  # Non-zero gradient
                    params_with_grad += 1
                else:
                    params_without_grad += 1

        print("\nGradient statistics:")
        print(f"  Parameters with gradients: {params_with_grad}")
        print(f"  Parameters with zero gradients: {params_without_grad}")
        print(f"  Maximum gradient norm: {max_grad_norm:.6f}")

        # All parameters should have non-zero gradients
        assert params_with_grad > 0, "No parameters received gradients"
        assert params_without_grad == 0, f"{params_without_grad} parameters have zero gradients"

        # Gradients should not be extreme
        assert max_grad_norm < 1000, f"Gradient norm {max_grad_norm} is too large"


class TestAudioCNNConfigurations:
    """Test AudioCNN with different architectural configurations."""

    def test_different_channel_configs(self) -> None:
        """Test model with various channel configurations."""
        channel_configs = [
            [8, 16],  # Smaller model
            [16, 32, 64],  # Default
            [32, 64, 128, 256],  # Larger model
        ]

        for channels in channel_configs:
            model = AudioCNN(num_classes=80, channels=channels)
            x = torch.randn(2, 1, 128, 431)
            output = model(x)

            assert output.shape == (2, 80), f"Channels {channels}: wrong output shape"
            assert not torch.isnan(output).any(), f"Channels {channels}: NaN in output"

    def test_different_num_classes(self) -> None:
        """Test model with different number of output classes."""
        num_classes_options = [10, 41, 80, 527]  # Various dataset sizes

        for num_classes in num_classes_options:
            model = AudioCNN(num_classes=num_classes, channels=[16, 32])
            x = torch.randn(2, 1, 128, 431)
            output = model(x)

            assert output.shape == (
                2,
                num_classes,
            ), f"Classes {num_classes}: wrong output shape"

    def test_different_dropout_rates(self) -> None:
        """Test model with different dropout rates."""
        dropout_rates = [0.0, 0.2, 0.4, 0.6]

        for dropout in dropout_rates:
            model = AudioCNN(num_classes=80, channels=[16, 32], dropout=dropout)
            x = torch.randn(2, 1, 128, 431)

            model.eval()
            with torch.no_grad():
                output = model(x)

            assert output.shape == (2, 80), f"Dropout {dropout}: wrong output shape"
            assert not torch.isnan(output).any(), f"Dropout {dropout}: NaN in output"

    def test_kernel_sizes(self) -> None:
        """Test model with different convolution kernel sizes."""
        kernel_sizes = [3, 5, 7]

        for kernel_size in kernel_sizes:
            model = AudioCNN(num_classes=80, channels=[16, 32], kernel_size=kernel_size)
            x = torch.randn(2, 1, 128, 431)
            output = model(x)

            assert output.shape == (2, 80), f"Kernel {kernel_size}: wrong output shape"
            assert not torch.isnan(output).any(), f"Kernel {kernel_size}: NaN in output"
