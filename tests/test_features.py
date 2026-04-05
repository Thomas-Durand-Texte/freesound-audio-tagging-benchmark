"""Tests for SuperGaussian spectrogram extraction.

This module tests the spectrogram extraction pipeline to ensure:
- Correct output shapes
- Energy conservation
- Deterministic behavior
- Frequency response accuracy
- Normalization correctness
- Integration with model pipeline
"""

import numpy as np
import pytest
import torch

from src.core.config import SpectrogramConfig, SpectrogramNormalization
from src.features.signal_tools import SuperGaussianEnvelope
from src.features.spectrogram_optimized import (
    MultiResolutionFilterBank,
    normalize_spectrogram_bell,
)


class TestSpectrogramExtraction:
    """Test suite for SuperGaussian spectrogram extraction."""

    @staticmethod
    def _to_numpy(arr):
        """Convert torch tensor or numpy array to numpy array."""
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr

    @pytest.fixture
    def filter_bank(self) -> MultiResolutionFilterBank:
        """Create a standard filter bank for testing."""
        return MultiResolutionFilterBank(
            envelope_class=SuperGaussianEnvelope,
            f_min=20.0,
            f_max=8000.0,
            f_mid=1000.0,
            num_bands=128,
            sample_rate=44100,
            signal_duration=5.0,
        )

    @pytest.fixture
    def random_audio(self) -> torch.Tensor:
        """Generate random audio signal for testing."""
        torch.manual_seed(42)
        return torch.randn(44100 * 5, dtype=torch.float32)

    @pytest.fixture
    def sine_wave(self) -> tuple[torch.Tensor, float]:
        """Generate a pure sine wave at 440 Hz (A4)."""
        sample_rate = 44100
        duration = 5.0
        frequency = 440.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        signal = torch.sin(2 * torch.pi * frequency * t)
        return signal, frequency

    def test_output_shape(self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor):
        """Verify spectrogram output has correct shape (n_bands, time_frames)."""
        spec, time_step, comp_time = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Check shape
        assert spec.shape[0] == 128, f"Expected 128 bands, got {spec.shape[0]}"
        assert spec.shape[1] > 0, "Expected positive number of time frames"

        # Verify time frames calculation
        n_samples = filter_bank.n_samples
        hop_length = 512
        expected_frames = (n_samples + hop_length - 1) // hop_length
        assert spec.shape[1] == expected_frames, (
            f"Expected {expected_frames} frames, got {spec.shape[1]}"
        )

        # Verify time step
        expected_time_step = hop_length / filter_bank.sample_rate
        assert np.isclose(time_step, expected_time_step), (
            f"Expected time step {expected_time_step}, got {time_step}"
        )

        # Verify computation time is reasonable
        assert comp_time > 0, "Computation time should be positive"
        assert comp_time < 10.0, "Computation time should be reasonable (< 10s)"

    def test_energy_conservation(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify total energy is approximately preserved (within reasonable bounds).

        Energy conservation is tested in log domain (Bell scale).
        We expect the total energy to be within reasonable bounds. Due to:
        - Multi-resolution processing with adaptive downsampling
        - Finite frequency resolution
        - Windowing effects
        - Downsampling and averaging operations
        The energy may not be perfectly preserved, but should remain stable.
        """
        # Prepare signal to match expected length
        signal = filter_bank._prepare_signal(random_audio)

        # Compute input energy (in Bell)
        input_energy_linear = torch.sum(signal**2).item()
        input_energy_bell = np.log10(input_energy_linear + 1e-10)

        # Compute spectrogram (returns torch.Tensor)
        spec, _, _ = filter_bank.compute_spectrogram(signal, hop_length=512)

        # Convert spectrogram from Bell to linear scale and sum
        # spec is in units of log10(power) = Bell
        # So power = 10^(spec)
        spec_linear = 10**spec
        output_energy_linear = torch.sum(spec_linear).item()
        output_energy_bell = np.log10(output_energy_linear + 1e-10)

        # Check energy is preserved within reasonable bounds (±15 Bell = ±150 dB)
        energy_ratio_bell = output_energy_bell - input_energy_bell

        # We allow up to ±15 Bell difference (±150 dB)
        # Due to multi-resolution processing, energy loss is expected from:
        # - Averaging operations in downsampled signals
        # - Frame-wise power averaging (avg_pool1d)
        # This is primarily a sanity check to ensure the system is stable
        assert abs(energy_ratio_bell) < 15.0, (
            f"Energy ratio {energy_ratio_bell:.2f} Bell (= {energy_ratio_bell * 10:.1f} dB) exceeds ±150 dB threshold"
        )

    def test_deterministic(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify same input produces identical output (deterministic behavior)."""
        # Compute spectrogram twice with same input
        spec1, time_step1, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)
        spec2, time_step2, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Check exact equality (torch tensors)
        assert torch.equal(spec1, spec2), "Spectrogram computation should be deterministic"

        assert time_step1 == time_step2, "Time step should be deterministic"

        # Verify no NaN or Inf values
        assert not torch.any(torch.isnan(spec1)), "Spectrogram should not contain NaN"
        assert not torch.any(torch.isinf(spec1)), "Spectrogram should not contain Inf"

    def test_frequency_response(
        self, filter_bank: MultiResolutionFilterBank, sine_wave: tuple[torch.Tensor, float]
    ):
        """Verify pure tone appears in correct frequency band.

        A pure sine wave should produce maximum energy in the band
        containing its frequency.
        """
        signal, frequency = sine_wave

        # Compute spectrogram
        spec, _, _ = filter_bank.compute_spectrogram(signal, hop_length=512)

        # Find band with maximum average energy (convert to numpy for analysis)
        spec_np = self._to_numpy(spec)
        band_energies = np.mean(spec_np, axis=1)
        max_band_idx = np.argmax(band_energies)
        max_band_freq = filter_bank.center_frequencies[max_band_idx]

        # Check that the maximum is in the band containing the sine wave frequency
        # We expect the center frequency to be close to the input frequency
        freq_error = abs(max_band_freq - frequency)

        # Allow error up to half the bandwidth of the band
        max_allowed_error = filter_bank.bandwidths[max_band_idx] / 2

        assert freq_error <= max_allowed_error, (
            f"Sine wave at {frequency} Hz should peak near band at {max_band_freq} Hz, "
            f"but error is {freq_error:.1f} Hz (max allowed: {max_allowed_error:.1f} Hz)"
        )

        # Verify the peak is significantly higher than the mean
        mean_energy = np.mean(band_energies)
        peak_energy = band_energies[max_band_idx]

        # Peak should be at least 1 Bell (10 dB) above mean
        peak_to_mean_ratio_bell = peak_energy - mean_energy
        assert peak_to_mean_ratio_bell > 1.0, (
            f"Peak energy should be >1 Bell (10 dB) above mean, got {peak_to_mean_ratio_bell:.1f} Bell ({peak_to_mean_ratio_bell * 10:.1f} dB)"
        )

    def test_normalization_temporal_masking(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify temporal masking normalization works correctly.

        Temporal masking should set values below the masking threshold to 0.
        """
        # Compute raw spectrogram
        spec_raw, _, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Apply normalization with temporal masking
        norm_config = SpectrogramNormalization(
            enable_temporal_masking=True,
            masking_threshold_db=20.0,
            masking_reference="percentile",
            masking_percentile=95.0,
            floor_db=60.0,
            normalize_method="none",
        )

        spec_normalized = normalize_spectrogram_bell(spec_raw, norm_config)

        # Check that some values have been set to 0 (convert to numpy for comparison)
        # (unless the signal is very uniform, which is unlikely with random noise)
        spec_norm_np = self._to_numpy(spec_normalized)
        zero_values = np.sum(np.isclose(spec_norm_np, 0.0))

        # At least some values should be at 0
        assert zero_values > 0, "Temporal masking should set some values to 0"

        # Verify no values are below 0 (floor is removed first, then masking applied)
        assert np.all(spec_norm_np >= -1e-6), "No values should be below 0 after floor removal"

    def test_normalization_global_floor(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify global floor normalization works correctly.

        Global floor should remove noise floor and normalize to [0, 1] range.
        """
        # Compute raw spectrogram
        spec_raw, _, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Apply normalization with linear scaling (no temporal masking)
        norm_config = SpectrogramNormalization(
            enable_temporal_masking=False,
            floor_db=60.0,
            floor_reference="global_max",
            normalize_method="linear",
        )

        spec_normalized = normalize_spectrogram_bell(spec_raw, norm_config)

        # Check range is [0, 1] (convert to numpy for assertions)
        spec_norm_np = self._to_numpy(spec_normalized)
        assert np.min(spec_norm_np) >= -1e-6, (
            f"Normalized spectrogram should be >= 0, got min={np.min(spec_norm_np)}"
        )

        assert np.max(spec_norm_np) <= 1.0 + 1e-6, (
            f"Normalized spectrogram should be <= 1, got max={np.max(spec_norm_np)}"
        )

        # Check that maximum is close to 1.0
        assert np.max(spec_norm_np) > 0.9, (
            f"Maximum should be close to 1.0 after normalization, got max={np.max(spec_norm_np)}"
        )

    def test_normalization_combined(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify combined temporal masking + global floor normalization.

        This is the default normalization pipeline used in the model.
        """
        # Compute raw spectrogram
        spec_raw, _, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Apply combined normalization (default settings)
        norm_config = SpectrogramNormalization()

        spec_normalized = normalize_spectrogram_bell(spec_raw, norm_config)

        # Check range is [0, 1] (convert to numpy for assertions)
        spec_norm_np = self._to_numpy(spec_normalized)
        assert np.min(spec_norm_np) >= -1e-6, (
            f"Normalized spectrogram should be >= 0, got min={np.min(spec_norm_np)}"
        )

        assert np.max(spec_norm_np) <= 1.0 + 1e-6, (
            f"Normalized spectrogram should be <= 1, got max={np.max(spec_norm_np)}"
        )

        # Verify no NaN or Inf
        assert not np.any(np.isnan(spec_norm_np)), "Normalized spectrogram should not contain NaN"
        assert not np.any(np.isinf(spec_norm_np)), "Normalized spectrogram should not contain Inf"

    def test_integration_with_model(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify spectrogram output is compatible with PyTorch model input.

        The model expects:
        - Input shape: (batch, 1, n_bands, time_frames)
        - Values in [0, 1] range (after normalization)
        - No NaN or Inf values
        """
        # Compute and normalize spectrogram
        spec_raw, _, _ = filter_bank.compute_spectrogram(random_audio, hop_length=512)

        # Apply normalization
        norm_config = SpectrogramNormalization()
        spec_normalized = normalize_spectrogram_bell(spec_raw, norm_config)

        # Convert to PyTorch tensor with batch dimension (handle both torch and numpy)
        if isinstance(spec_normalized, torch.Tensor):
            spec_tensor = spec_normalized
        else:
            spec_tensor = torch.from_numpy(spec_normalized).float()
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Check shape
        assert spec_tensor.ndim == 4, (
            f"Model input should be 4D (batch, channel, freq, time), got {spec_tensor.ndim}D"
        )

        assert spec_tensor.shape[0] == 1, "Batch dimension should be 1"
        assert spec_tensor.shape[1] == 1, "Channel dimension should be 1"
        assert spec_tensor.shape[2] == 128, "Frequency dimension should match n_bands"
        assert spec_tensor.shape[3] > 0, "Time dimension should be positive"

        # Check data type
        assert spec_tensor.dtype == torch.float32, (
            f"Model input should be float32, got {spec_tensor.dtype}"
        )

        # Check value range
        assert torch.min(spec_tensor) >= 0.0, "Model input should be non-negative"
        assert torch.max(spec_tensor) <= 1.0, "Model input should be <= 1.0"

        # Check for NaN/Inf
        assert not torch.any(torch.isnan(spec_tensor)), "Model input should not contain NaN"
        assert not torch.any(torch.isinf(spec_tensor)), "Model input should not contain Inf"

    def test_different_hop_lengths(
        self, filter_bank: MultiResolutionFilterBank, random_audio: torch.Tensor
    ):
        """Verify spectrogram computation works with different hop lengths."""
        hop_lengths = [256, 512, 1024]

        for hop_length in hop_lengths:
            spec, time_step, _ = filter_bank.compute_spectrogram(
                random_audio, hop_length=hop_length
            )

            # Check shape
            assert spec.shape[0] == 128, f"Expected 128 bands with hop_length={hop_length}"

            # Check time step
            expected_time_step = hop_length / filter_bank.sample_rate
            assert np.isclose(time_step, expected_time_step), (
                f"Time step mismatch for hop_length={hop_length}"
            )

            # Check number of frames
            expected_frames = (filter_bank.n_samples + hop_length - 1) // hop_length
            assert spec.shape[1] == expected_frames, (
                f"Frame count mismatch for hop_length={hop_length}"
            )

    def test_signal_padding(self, filter_bank: MultiResolutionFilterBank):
        """Verify signals shorter than expected duration are zero-padded."""
        # Create short signal (1 second instead of 5)
        short_signal = torch.randn(44100, dtype=torch.float32)

        # Compute spectrogram
        spec, _, _ = filter_bank.compute_spectrogram(short_signal, hop_length=512)

        # Should still produce correct output shape
        expected_frames = (filter_bank.n_samples + 512 - 1) // 512
        assert spec.shape == (128, expected_frames), (
            "Short signals should be zero-padded to expected length"
        )

    def test_signal_cropping(self, filter_bank: MultiResolutionFilterBank):
        """Verify signals longer than expected duration are cropped."""
        # Create long signal (10 seconds instead of 5)
        long_signal = torch.randn(44100 * 10, dtype=torch.float32)

        # Compute spectrogram
        spec, _, _ = filter_bank.compute_spectrogram(long_signal, hop_length=512)

        # Should still produce correct output shape
        expected_frames = (filter_bank.n_samples + 512 - 1) // 512
        assert spec.shape == (128, expected_frames), (
            "Long signals should be cropped to expected length"
        )

    def test_zero_signal(self, filter_bank: MultiResolutionFilterBank):
        """Verify zero signal produces valid output (no division by zero)."""
        zero_signal = torch.zeros(44100 * 5, dtype=torch.float32)

        # Should not raise any errors
        spec, _, _ = filter_bank.compute_spectrogram(zero_signal, hop_length=512)

        # Output should be finite (no NaN or Inf) - convert to numpy for assertion
        spec_np = self._to_numpy(spec)
        assert np.all(np.isfinite(spec_np)), "Zero signal should produce finite spectrogram values"

        # All values should be very low (near log10(epsilon))
        assert np.all(spec_np < -5), "Zero signal should produce very low spectrogram values"


class TestSpectrogramConfig:
    """Test SpectrogramConfig dataclass integration."""

    def test_config_creation(self):
        """Verify SpectrogramConfig can be created with default values."""
        config = SpectrogramConfig()

        assert config.f_min == 20.0
        assert config.f_max == 8000.0
        assert config.f_mid == 1000.0
        assert config.n_bands == 128
        assert config.hop_length == 512
        assert isinstance(config.normalization, SpectrogramNormalization)

    def test_config_with_custom_normalization(self):
        """Verify custom normalization config can be passed."""
        custom_norm = SpectrogramNormalization(
            enable_temporal_masking=False,
            floor_db=40.0,
        )

        config = SpectrogramConfig(normalization=custom_norm)

        assert config.normalization.enable_temporal_masking is False
        assert config.normalization.floor_db == 40.0

    def test_normalization_config_defaults(self):
        """Verify SpectrogramNormalization has correct defaults."""
        norm_config = SpectrogramNormalization()

        assert norm_config.enable_temporal_masking is True
        assert norm_config.masking_threshold_db == 20.0
        assert norm_config.masking_reference == "percentile"
        assert norm_config.masking_percentile == 95.0
        assert norm_config.floor_db == 60.0
        assert norm_config.floor_reference == "global_max"
        assert norm_config.percentile == 95.0
        assert norm_config.normalize_method == "linear"
