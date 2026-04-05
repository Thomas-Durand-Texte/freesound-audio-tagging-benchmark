"""Tests for signal processing tools including device handling."""

import pytest
import torch

from src.features.signal_tools import SuperGaussianEnvelope
from src.features.spectrogram_optimized import MultiResolutionFilterBank


class TestMultiResolutionFilterBankDeviceHandling:
    """Tests for MultiResolutionFilterBank device handling."""

    def test_init_with_cpu_device(self):
        """Test that filter bank initializes correctly on CPU."""
        fb = MultiResolutionFilterBank(
            envelope_class=SuperGaussianEnvelope,
            f_min=20,
            f_max=8000,
            num_bands=8,
            sample_rate=16000,
            signal_duration=5.0,
            device="cpu",
        )
        assert fb.device.type == "cpu"
        # Check that spectrum weights are on CPU
        assert fb.band_infos[0].spectrum_weights_pos.device.type == "cpu"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_init_with_mps_device(self):
        """Test that filter bank initializes correctly on MPS."""
        fb = MultiResolutionFilterBank(
            envelope_class=SuperGaussianEnvelope,
            f_min=20,
            f_max=8000,
            num_bands=8,
            sample_rate=16000,
            signal_duration=5.0,
            device="mps",
        )
        assert fb.device.type == "mps"
        # Check that spectrum weights are on MPS
        assert fb.band_infos[0].spectrum_weights_pos.device.type == "mps"

    def test_device_parameter_as_torch_device(self):
        """Test that device parameter accepts torch.device objects."""
        device = torch.device("cpu")
        fb = MultiResolutionFilterBank(
            envelope_class=SuperGaussianEnvelope,
            f_min=20,
            f_max=8000,
            num_bands=8,
            sample_rate=16000,
            signal_duration=5.0,
            device=device,
        )
        assert fb.device == device
