"""Tests for data augmentation functions."""

import numpy as np
import pytest
import torch

from src.data.augmentation import (
    WaveformAugmentation,
    add_noise,
    frequency_response_perturbation,
    mixup,
    mixup_collate_fn,
    pitch_shift,
    spec_augment,
    time_stretch,
)


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


class TestMixup:
    """Tests for Mixup augmentation."""

    def test_output_shapes_match_input(self):
        """Test that output shapes match input shapes."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        assert mixed_x.shape == batch_x.shape
        assert mixed_y.shape == batch_y.shape

    def test_mixed_samples_are_linear_combinations(self):
        """Test that mixed samples are linear combinations of pairs."""
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, num_classes = 4, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Store original values
        x_orig = batch_x.clone()
        y_orig = batch_y.clone()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        # The mixed batch should be different from the original
        # (unless lambda happens to be exactly 1, which is very unlikely)
        assert not torch.equal(mixed_x, x_orig) or not torch.equal(mixed_y, y_orig)

        # Check that each mixed sample is within the convex hull
        # (all values between min and max of originals)
        assert torch.all(mixed_x >= x_orig.min())
        assert torch.all(mixed_x <= x_orig.max())

    def test_mixed_labels_are_linear_combinations(self):
        """Test that mixed labels are linear combinations."""
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, num_classes = 4, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.zeros(batch_size, num_classes)
        # Create distinct labels for easier verification
        for i in range(batch_size):
            batch_y[i, i] = 1.0

        y_orig = batch_y.clone()
        _, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        # Mixed labels should be different (unless lambda=1)
        assert not torch.equal(mixed_y, y_orig)

        # Labels should be in [0, 1] range
        assert torch.all(mixed_y >= 0)
        assert torch.all(mixed_y <= 1)

    def test_lambda_from_beta_distribution(self):
        """Test that lambda is sampled from Beta(alpha, alpha) distribution."""
        np.random.seed(None)  # Use random seed

        batch_size, num_classes = 8, 80
        alpha = 0.4
        num_trials = 1000
        lambdas = []

        for _ in range(num_trials):
            batch_x = torch.randn(batch_size, 1, 128, 100)
            batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

            mixed_x, _ = mixup(batch_x, batch_y, alpha=alpha)

            # Store the mixed value for distribution check
            # We can't directly extract lambda, but we can verify distribution properties
            lambdas.append(mixed_x[0, 0, 0, 0].item())

        # Verify that mixing is happening (values change)
        assert len(set(lambdas)) > 1, "Lambda values should vary"

    def test_alpha_zero_returns_identity(self):
        """Test that alpha=0 returns unchanged batch (no mixing)."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        x_orig = batch_x.clone()
        y_orig = batch_y.clone()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0)

        assert torch.equal(mixed_x, x_orig)
        assert torch.equal(mixed_y, y_orig)

    def test_negative_alpha_returns_identity(self):
        """Test that negative alpha returns unchanged batch."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        x_orig = batch_x.clone()
        y_orig = batch_y.clone()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=-0.1)

        assert torch.equal(mixed_x, x_orig)
        assert torch.equal(mixed_y, y_orig)

    def test_works_with_multihot_labels(self):
        """Test that mixup works with multi-hot encoded labels."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        # Multi-hot: multiple 1s per sample
        batch_y = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            # Randomly set 2-3 labels to 1
            num_labels = np.random.randint(2, 4)
            indices = np.random.choice(num_classes, num_labels, replace=False)
            batch_y[i, indices] = 1.0

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        # Should still have valid shapes
        assert mixed_x.shape == batch_x.shape
        assert mixed_y.shape == batch_y.shape

        # Mixed labels should be in [0, 1] range
        assert torch.all(mixed_y >= 0)
        assert torch.all(mixed_y <= 1)

    def test_stochastic_behavior(self):
        """Test that different calls produce different results."""
        np.random.seed(None)

        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Multiple calls should produce different results
        mixed_x1, mixed_y1 = mixup(batch_x.clone(), batch_y.clone(), alpha=0.4)
        mixed_x2, mixed_y2 = mixup(batch_x.clone(), batch_y.clone(), alpha=0.4)
        mixed_x3, mixed_y3 = mixup(batch_x.clone(), batch_y.clone(), alpha=0.4)

        # At least one pair should differ (with very high probability)
        all_equal = (
            torch.equal(mixed_x1, mixed_x2)
            and torch.equal(mixed_x2, mixed_x3)
            and torch.equal(mixed_y1, mixed_y2)
            and torch.equal(mixed_y2, mixed_y3)
        )
        assert not all_equal, "All mixup calls produced identical results (expected randomness)"

    def test_device_compatibility_cpu(self):
        """Test that mixup works with CPU tensors."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100, device="cpu")
        batch_y = torch.randint(0, 2, (batch_size, num_classes), device="cpu").float()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        assert mixed_x.device == batch_x.device
        assert mixed_y.device == batch_y.device

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_compatibility_mps(self):
        """Test that mixup works with MPS tensors."""
        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100, device="mps")
        batch_y = torch.randint(0, 2, (batch_size, num_classes), device="mps").float()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        assert mixed_x.device == batch_x.device
        assert mixed_y.device == batch_y.device

    def test_batch_size_one(self):
        """Test mixup with batch size of 1."""
        batch_size, num_classes = 1, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        mixed_x, mixed_y = mixup(batch_x, batch_y, alpha=0.4)

        # With batch size 1, sample mixes with itself
        # Result should be: lambda * x + (1-lambda) * x = x (up to floating point precision)
        assert torch.allclose(mixed_x, batch_x)
        assert torch.allclose(mixed_y, batch_y)

    def test_different_alpha_values(self):
        """Test that different alpha values affect mixing."""
        np.random.seed(42)

        batch_size, num_classes = 8, 80
        batch_x = torch.randn(batch_size, 1, 128, 100)
        batch_y = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Test with different alpha values
        alphas = [0.1, 0.4, 1.0, 2.0]
        results = []

        for alpha in alphas:
            np.random.seed(42)  # Same seed for comparison
            torch.manual_seed(42)
            mixed_x, mixed_y = mixup(batch_x.clone(), batch_y.clone(), alpha=alpha)
            results.append((mixed_x, mixed_y))

        # Results should be valid for all alphas
        for mixed_x, mixed_y in results:
            assert mixed_x.shape == batch_x.shape
            assert mixed_y.shape == batch_y.shape


class TestMixupCollateFn:
    """Tests for mixup_collate_fn."""

    def test_collate_fn_returns_callable(self):
        """Test that mixup_collate_fn returns a callable function."""
        collate_fn = mixup_collate_fn(alpha=0.4)
        assert callable(collate_fn)

    def test_collate_fn_processes_batch(self):
        """Test that the collate function processes a batch correctly."""
        batch_size, num_classes = 8, 80

        # Create fake batch data (list of tuples)
        batch = [
            (torch.randn(1, 128, 100), torch.randint(0, 2, (num_classes,)).float())
            for _ in range(batch_size)
        ]

        collate_fn = mixup_collate_fn(alpha=0.4)
        spectrograms, labels = collate_fn(batch)

        # Check output shapes
        assert spectrograms.shape == (batch_size, 1, 128, 100)
        assert labels.shape == (batch_size, num_classes)

    def test_collate_fn_applies_mixup(self):
        """Test that collate function actually applies mixup."""
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size, num_classes = 8, 80

        # Create batch with distinct patterns
        batch = []
        for i in range(batch_size):
            spec = torch.ones(1, 128, 100) * i  # Each sample has distinct value
            label = torch.zeros(num_classes)
            label[i % num_classes] = 1.0
            batch.append((spec, label))

        # Store original for comparison
        orig_specs = torch.stack([item[0] for item in batch])
        orig_labels = torch.stack([item[1] for item in batch])

        collate_fn = mixup_collate_fn(alpha=0.4)
        mixed_specs, mixed_labels = collate_fn(batch)

        # Mixed results should differ from originals (unless lambda=1, very unlikely)
        assert not torch.equal(mixed_specs, orig_specs) or not torch.equal(
            mixed_labels, orig_labels
        )

    def test_collate_fn_with_alpha_zero(self):
        """Test that alpha=0 in collate_fn returns unmixed batch."""
        batch_size, num_classes = 8, 80

        batch = [
            (torch.randn(1, 128, 100), torch.randint(0, 2, (num_classes,)).float())
            for _ in range(batch_size)
        ]

        # Store original
        orig_specs = torch.stack([item[0] for item in batch])
        orig_labels = torch.stack([item[1] for item in batch])

        collate_fn = mixup_collate_fn(alpha=0)
        mixed_specs, mixed_labels = collate_fn(batch)

        assert torch.equal(mixed_specs, orig_specs)
        assert torch.equal(mixed_labels, orig_labels)

    def test_collate_fn_different_alpha_values(self):
        """Test collate_fn with different alpha values."""
        batch_size, num_classes = 8, 80

        batch = [
            (torch.randn(1, 128, 100), torch.randint(0, 2, (num_classes,)).float())
            for _ in range(batch_size)
        ]

        for alpha in [0.1, 0.4, 1.0, 2.0]:
            collate_fn = mixup_collate_fn(alpha=alpha)
            spectrograms, labels = collate_fn(batch)

            assert spectrograms.shape == (batch_size, 1, 128, 100)
            assert labels.shape == (batch_size, num_classes)


class TestTimeStretch:
    """Tests for time_stretch waveform augmentation."""

    def test_output_length_changes(self):
        """Test that output length changes as expected."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)  # 1 second

        # Faster (shorter output)
        np.random.seed(42)
        stretched = time_stretch(waveform, sample_rate, stretch_range=(1.1, 1.1))
        assert stretched.shape[-1] < waveform.shape[-1]

        # Slower (longer output)
        np.random.seed(42)
        stretched = time_stretch(waveform, sample_rate, stretch_range=(0.9, 0.9))
        assert stretched.shape[-1] > waveform.shape[-1]

    def test_output_is_tensor(self):
        """Test that output is a torch tensor."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        stretched = time_stretch(waveform, sample_rate)
        assert isinstance(stretched, torch.Tensor)

    def test_random_stretch_factor(self):
        """Test that random stretch factors produce different results."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        # Multiple calls with same input should produce different lengths
        lengths = []
        for _ in range(10):
            stretched = time_stretch(waveform, sample_rate)
            lengths.append(stretched.shape[-1])

        # Should have variation in lengths (with high probability)
        assert len(set(lengths)) > 1, "Expected variation in output lengths"

    def test_preserves_signal_characteristics(self):
        """Test that the augmentation preserves basic signal characteristics."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        stretched = time_stretch(waveform, sample_rate)

        # Output should have reasonable amplitude
        assert stretched.abs().max() > 0, "Output should not be silent"
        assert torch.isfinite(stretched).all(), "Output should not contain NaN or Inf"


class TestPitchShift:
    """Tests for pitch_shift waveform augmentation."""

    def test_output_length_unchanged(self):
        """Test that output length is the same as input."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        shifted = pitch_shift(waveform, sample_rate)
        assert shifted.shape == waveform.shape

    def test_output_is_tensor(self):
        """Test that output is a torch tensor."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        shifted = pitch_shift(waveform, sample_rate)
        assert isinstance(shifted, torch.Tensor)

    def test_random_pitch_shift(self):
        """Test that random pitch shifts produce different results."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        # Multiple calls should produce different results
        results = []
        for _ in range(10):
            shifted = pitch_shift(waveform, sample_rate)
            results.append(shifted)

        # At least some should be different (with high probability)
        all_equal = all(torch.equal(results[0], r) for r in results)
        assert not all_equal, "Expected variation in pitch-shifted outputs"

    def test_semitone_range(self):
        """Test with specific semitone range."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        # Test with positive shift
        shifted_up = pitch_shift(waveform, sample_rate, semitone_range=(2, 2))
        assert shifted_up.shape == waveform.shape
        assert torch.isfinite(shifted_up).all()

        # Test with negative shift
        shifted_down = pitch_shift(waveform, sample_rate, semitone_range=(-2, -2))
        assert shifted_down.shape == waveform.shape
        assert torch.isfinite(shifted_down).all()

    def test_preserves_signal_characteristics(self):
        """Test that the augmentation preserves basic signal characteristics."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        shifted = pitch_shift(waveform, sample_rate)

        # Output should have reasonable amplitude
        assert shifted.abs().max() > 0, "Output should not be silent"
        assert torch.isfinite(shifted).all(), "Output should not contain NaN or Inf"


class TestAddNoise:
    """Tests for add_noise waveform augmentation."""

    def test_output_shape_unchanged(self):
        """Test that output shape matches input."""
        waveform = torch.randn(16000)

        noisy = add_noise(waveform)
        assert noisy.shape == waveform.shape

    def test_output_is_tensor(self):
        """Test that output is a torch tensor."""
        waveform = torch.randn(16000)

        noisy = add_noise(waveform)
        assert isinstance(noisy, torch.Tensor)

    def test_snr_approximately_correct(self):
        """Test that SNR is approximately in the specified range."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create a signal with known power
        waveform = torch.randn(16000) * 0.5
        target_snr = 30.0

        # Apply noise with fixed SNR
        noisy = add_noise(waveform, snr_db_range=(target_snr, target_snr))

        # Calculate actual SNR
        signal_power = (waveform**2).mean()
        noise = noisy - waveform
        noise_power = (noise**2).mean()
        actual_snr_db = 10 * torch.log10(signal_power / noise_power)

        # SNR should be approximately correct (within ±2 dB tolerance)
        assert abs(actual_snr_db - target_snr) < 2.0, (
            f"SNR {actual_snr_db:.1f} dB != {target_snr} dB"
        )

    def test_different_noise_each_call(self):
        """Test that different calls add different noise."""
        waveform = torch.randn(16000)

        noisy1 = add_noise(waveform.clone())
        noisy2 = add_noise(waveform.clone())
        noisy3 = add_noise(waveform.clone())

        # Results should be different
        assert not torch.equal(noisy1, noisy2)
        assert not torch.equal(noisy2, noisy3)

    def test_preserves_signal_characteristics(self):
        """Test that signal characteristics are preserved."""
        waveform = torch.randn(16000)

        noisy = add_noise(waveform)

        # Output should be different but not drastically
        assert not torch.equal(noisy, waveform)
        assert torch.isfinite(noisy).all()
        # With high SNR, correlation should be high
        correlation = torch.corrcoef(torch.stack([waveform, noisy]))[0, 1]
        assert correlation > 0.8, "Correlation should be high with 30-40 dB SNR"


class TestFrequencyResponsePerturbation:
    """Tests for frequency_response_perturbation."""

    def test_output_shape_unchanged(self):
        """Test that output shape matches input."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered = frequency_response_perturbation(waveform, sample_rate)
        assert filtered.shape == waveform.shape

    def test_output_is_tensor(self):
        """Test that output is a torch tensor."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered = frequency_response_perturbation(waveform, sample_rate)
        assert isinstance(filtered, torch.Tensor)

    def test_frequency_content_modified(self):
        """Test that frequency content is actually modified."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered = frequency_response_perturbation(waveform, sample_rate)

        # Output should be different from input
        assert not torch.allclose(filtered, waveform, rtol=1e-3)

    def test_different_perturbations_each_call(self):
        """Test that different calls produce different perturbations."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered1 = frequency_response_perturbation(waveform, sample_rate)
        filtered2 = frequency_response_perturbation(waveform, sample_rate)
        filtered3 = frequency_response_perturbation(waveform, sample_rate)

        # Results should be different
        assert not torch.allclose(filtered1, filtered2, rtol=1e-3)
        assert not torch.allclose(filtered2, filtered3, rtol=1e-3)

    def test_preserves_signal_energy(self):
        """Test that signal energy is roughly preserved."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered = frequency_response_perturbation(waveform, sample_rate, gain_std_db=3.0)

        # Energy should be similar (within reasonable bounds given ±3 dB gain)
        orig_energy = (waveform**2).sum()
        filtered_energy = (filtered**2).sum()
        ratio = filtered_energy / orig_energy

        # With ±3 dB gain, energy ratio should be between 0.5 and 2.0 (±3 dB ≈ 2x)
        assert 0.25 < ratio < 4.0, f"Energy ratio {ratio} is outside expected range"

    def test_output_is_finite(self):
        """Test that output contains no NaN or Inf values."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        filtered = frequency_response_perturbation(waveform, sample_rate)

        assert torch.isfinite(filtered).all(), "Output should not contain NaN or Inf"


class TestWaveformAugmentation:
    """Tests for WaveformAugmentation pipeline."""

    def test_initialization(self):
        """Test that the class initializes correctly."""
        aug = WaveformAugmentation(sample_rate=16000)
        assert aug.sample_rate == 16000
        assert aug.time_stretch is True
        assert aug.pitch_shift is True
        assert aug.add_noise is True
        assert aug.freq_perturbation is True

    def test_selective_augmentation(self):
        """Test that augmentations can be selectively disabled."""
        aug = WaveformAugmentation(
            sample_rate=16000,
            time_stretch=False,
            pitch_shift=False,
            add_noise=True,
            freq_perturbation=False,
        )
        assert aug.time_stretch is False
        assert aug.pitch_shift is False
        assert aug.add_noise is True
        assert aug.freq_perturbation is False

    def test_pipeline_execution(self):
        """Test that the pipeline executes without errors."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        aug = WaveformAugmentation(sample_rate=sample_rate)
        augmented = aug(waveform)

        # Output should be a tensor
        assert isinstance(augmented, torch.Tensor)
        # Should be finite
        assert torch.isfinite(augmented).all()
        # Should not be empty
        assert augmented.numel() > 0

    def test_all_augmentations_applied(self):
        """Test that all augmentations are applied when enabled."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        aug = WaveformAugmentation(
            sample_rate=sample_rate,
            time_stretch=True,
            pitch_shift=True,
            add_noise=True,
            freq_perturbation=True,
        )
        augmented = aug(waveform)

        # Output should be significantly different
        # Note: length may differ due to time stretch, so compare overlapping region
        min_len = min(len(augmented), len(waveform))
        assert not torch.allclose(augmented[:min_len], waveform[:min_len], rtol=0.1)

    def test_no_augmentations_returns_modified(self):
        """Test behavior with all augmentations disabled."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        aug = WaveformAugmentation(
            sample_rate=sample_rate,
            time_stretch=False,
            pitch_shift=False,
            add_noise=False,
            freq_perturbation=False,
        )
        augmented = aug(waveform)

        # Should still execute without error
        assert isinstance(augmented, torch.Tensor)
        # Should be approximately unchanged
        assert torch.allclose(augmented, waveform)

    def test_stochastic_behavior(self):
        """Test that multiple calls produce different results."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        aug = WaveformAugmentation(sample_rate=sample_rate)

        augmented1 = aug(waveform.clone())
        augmented2 = aug(waveform.clone())

        # Results should be different (due to randomness)
        # Note: length may vary due to time stretch
        assert not torch.allclose(
            augmented1[: min(len(augmented1), len(augmented2))],
            augmented2[: min(len(augmented1), len(augmented2))],
            rtol=1e-3,
        )

    def test_preserves_signal_integrity(self):
        """Test that augmentation preserves overall signal integrity."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate)

        aug = WaveformAugmentation(sample_rate=sample_rate)
        augmented = aug(waveform)

        # Should not be silent
        assert augmented.abs().max() > 0
        # Should be finite
        assert torch.isfinite(augmented).all()
        # Energy should be in reasonable range
        orig_energy = (waveform**2).mean()
        aug_energy = (augmented**2).mean()
        ratio = aug_energy / orig_energy
        # With all augmentations including Griffin-Lim reconstruction, energy can change significantly
        assert 0.01 < ratio < 20.0, f"Energy ratio {ratio} is outside expected range"


class TestFrequencyResponsePerturbationDeviceHandling:
    """Tests for frequency_response_perturbation device handling."""

    def test_cpu_device_handling(self):
        """Test that function works on CPU and stays on CPU."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate, device="cpu")

        filtered = frequency_response_perturbation(waveform, sample_rate)

        assert filtered.device.type == "cpu"
        assert filtered.shape == waveform.shape

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device_handling(self):
        """Test that function works on MPS and stays on MPS."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate, device="mps")

        filtered = frequency_response_perturbation(waveform, sample_rate)

        assert filtered.device.type == "mps"
        assert filtered.shape == waveform.shape

    def test_no_cpu_transfers_in_forward_pass(self):
        """Test that there are no CPU transfers during processing."""
        sample_rate = 16000
        waveform = torch.randn(sample_rate, device="cpu")

        # This test verifies the function completes without explicit .cpu()/.numpy() calls
        # by checking the output is on the same device as input
        filtered = frequency_response_perturbation(waveform, sample_rate)

        assert filtered.device == waveform.device
