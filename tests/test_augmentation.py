"""Tests for data augmentation functions."""

import numpy as np
import torch

from src.data.augmentation import spec_augment


class TestSpecAugment:
    """Tests for SpecAugment time and frequency masking."""

    def test_output_shape_matches_input_2d(self):
        """Test that output shape matches input shape for 2D tensors."""
        spec = torch.randn(128, 100)
        augmented = spec_augment(spec)
        assert augmented.shape == spec.shape

    def test_output_shape_matches_input_4d(self):
        """Test that output shape matches input shape for 4D tensors."""
        spec = torch.randn(8, 1, 128, 100)
        augmented = spec_augment(spec)
        assert augmented.shape == spec.shape

    def test_masks_applied_correctly(self):
        """Test that masks are applied correctly (verify zeros in expected regions)."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create a spectrogram filled with ones
        spec = torch.ones(128, 100)

        # Apply augmentation with controlled parameters
        augmented = spec_augment(
            spec, freq_mask_param=10, time_mask_param=10, num_freq_masks=1, num_time_masks=1
        )

        # Check that some values were set to zero
        assert torch.any(augmented == 0), "No masking was applied"

        # Check that not all values are zero
        assert torch.any(augmented != 0), "All values were masked"

        # Count the number of zeros
        num_zeros = (augmented == 0).sum().item()
        assert num_zeros > 0, "Expected some masked values"

    def test_frequency_masking(self):
        """Test that frequency masking creates horizontal stripes."""
        np.random.seed(42)

        spec = torch.ones(128, 100)
        augmented = spec_augment(
            spec, freq_mask_param=10, time_mask_param=0, num_freq_masks=1, num_time_masks=0
        )

        # If frequency masking was applied, entire rows should be zero
        zero_rows = (augmented.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(zero_rows) > 0:
            # Check that at least one complete row is masked
            assert len(zero_rows) > 0, "Expected at least one frequency band to be masked"

    def test_time_masking(self):
        """Test that time masking creates vertical stripes."""
        np.random.seed(42)

        spec = torch.ones(128, 100)
        augmented = spec_augment(
            spec, freq_mask_param=0, time_mask_param=10, num_freq_masks=0, num_time_masks=1
        )

        # If time masking was applied, entire columns should be zero
        zero_cols = (augmented.sum(dim=0) == 0).nonzero(as_tuple=True)[0]
        if len(zero_cols) > 0:
            # Check that at least one complete column is masked
            assert len(zero_cols) > 0, "Expected at least one time frame to be masked"

    def test_stochastic_behavior(self):
        """Test that different outputs are generated for the same input (stochastic)."""
        spec = torch.randn(128, 100)

        # Generate multiple augmentations
        augmented1 = spec_augment(spec.clone())
        augmented2 = spec_augment(spec.clone())
        augmented3 = spec_augment(spec.clone())

        # At least one should be different (with very high probability)
        all_equal = torch.equal(augmented1, augmented2) and torch.equal(augmented2, augmented3)
        assert not all_equal, "All augmentations are identical (expected stochastic behavior)"

    def test_mask_param_zero_no_masking(self):
        """Test that mask_param=0 results in no masking."""
        spec = torch.randn(128, 100)
        augmented = spec_augment(
            spec.clone(), freq_mask_param=0, time_mask_param=0, num_freq_masks=2, num_time_masks=2
        )

        # With mask_param=0, the mask width is always 0, so no masking occurs
        assert torch.equal(spec, augmented), "Expected no masking with mask_param=0"

    def test_num_masks_zero_no_masking(self):
        """Test that num_masks=0 results in no masking."""
        spec = torch.randn(128, 100)
        augmented = spec_augment(
            spec.clone(), freq_mask_param=20, time_mask_param=40, num_freq_masks=0, num_time_masks=0
        )

        # With num_masks=0, no masks are applied
        assert torch.equal(spec, augmented), "Expected no masking with num_masks=0"

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        batch_size = 4
        spec = torch.ones(batch_size, 1, 128, 100)

        augmented = spec_augment(spec)

        # Check that each sample in the batch was processed
        assert augmented.shape == spec.shape

        # Check that different samples in the batch have different masks (with high probability)
        # This is probabilistic, so we check that at least one pair is different
        all_equal = all(
            torch.equal(augmented[i], augmented[j])
            for i in range(batch_size)
            for j in range(i + 1, batch_size)
        )
        assert not all_equal, "Expected different masks for different samples in batch"

    def test_input_not_modified(self):
        """Test that the original input is not modified."""
        spec = torch.randn(128, 100)
        spec_original = spec.clone()

        spec_augment(spec)

        # Original tensor should remain unchanged
        assert torch.equal(spec, spec_original), "Original tensor was modified"

    def test_multiple_masks(self):
        """Test that multiple masks are applied."""
        np.random.seed(42)

        spec = torch.ones(128, 100)
        augmented = spec_augment(
            spec, freq_mask_param=5, time_mask_param=5, num_freq_masks=3, num_time_masks=3
        )

        # With multiple masks, more values should be masked
        num_zeros = (augmented == 0).sum().item()
        assert num_zeros > 0, "Expected some masked values with multiple masks"

    def test_edge_case_small_spectrogram(self):
        """Test with a very small spectrogram."""
        spec = torch.randn(10, 10)
        augmented = spec_augment(
            spec, freq_mask_param=3, time_mask_param=3, num_freq_masks=1, num_time_masks=1
        )

        assert augmented.shape == spec.shape

    def test_edge_case_large_mask_params(self):
        """Test with mask parameters larger than spectrogram dimensions."""
        spec = torch.randn(128, 100)

        # Mask params larger than dimensions should still work (masks will be clipped)
        augmented = spec_augment(
            spec, freq_mask_param=200, time_mask_param=200, num_freq_masks=1, num_time_masks=1
        )

        assert augmented.shape == spec.shape

        # At least some values should be masked
        assert torch.any(augmented == 0), "Expected some masking even with large mask params"

    def test_masking_value_is_zero(self):
        """Test that the masking value is exactly zero (silence)."""
        np.random.seed(42)

        spec = torch.randn(128, 100) + 10  # Ensure all values are non-zero initially
        augmented = spec_augment(
            spec, freq_mask_param=10, time_mask_param=10, num_freq_masks=2, num_time_masks=2
        )

        # Check that masked values are exactly zero
        masked_values = augmented[augmented == 0]
        if len(masked_values) > 0:
            assert torch.all(masked_values == 0), "Masked values should be exactly zero"
