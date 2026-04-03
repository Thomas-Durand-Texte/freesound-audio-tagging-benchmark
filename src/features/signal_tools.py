from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import fft

from src.visualization.plots import configure_axes

# ── Utility Functions ────────────────────────────────────────────────────────


def centered_time_array(n_samples: int, sample_rate: float) -> np.ndarray:
    """Create centered time array symmetric around zero.

    Args:
        n_samples: Number of samples (should be odd for perfect centering)
        sample_rate: Sample rate in Hz

    Returns:
        Time array from -t_center to +t_center
    """
    t_center = (n_samples - 1) / (2.0 * sample_rate)
    return np.linspace(-t_center, t_center, n_samples)


# ── Abstract Base Class ──────────────────────────────────────────────────────


class EnvelopePattern(ABC):
    """Abstract base class for envelope patterns (Gaussian, Super-Gaussian, etc.)."""

    def __init__(self, bandwidth: float) -> None:
        """Initialize envelope with target bandwidth.

        Args:
            bandwidth: Target -3dB bandwidth in Hz
        """
        self.bandwidth = bandwidth

    @abstractmethod
    def spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute analytical frequency spectrum.

        Args:
            frequencies: Array of frequencies (Hz), centered at 0

        Returns:
            Magnitude spectrum values
        """
        pass

    @abstractmethod
    def time_domain(
        self, sample_rate: float, edge_threshold: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate time-domain envelope with optimized length.

        Window length is chosen so envelope decays to edge_threshold at edges.

        Args:
            sample_rate: Sample rate in Hz
            edge_threshold: Target amplitude at window edges (0 < threshold < 1)

        Returns:
            Tuple of (time_array, envelope_values)
        """
        pass

    @abstractmethod
    def get_parameter_info(self) -> dict:
        """Get parameter information for debugging.

        Returns:
            Dictionary with envelope parameters
        """
        pass


def gaussian_pulse_bandwidth(sigma: float) -> float:
    # Compute analytical bandwidth (-3dB from max)
    # For a Gaussian spectrum G(f) = exp(-2*pi^2*sigma^2*f^2)
    # -3dB point occurs when G(f) = G(0) / sqrt(2) = 0.7071
    # exp(-2*pi^2*sigma^2*f_3dB^2) = 0.7071
    # -2*pi^2*sigma^2*f_3dB^2 = ln(0.7071)
    # f_3dB = sqrt(-ln(0.7071) / (2*pi^2*sigma^2))
    # -np.log(0.7071)) = 0.173291590185971
    # return 2 * np.sqrt(0.173291590185971 / (pi * sigma) ** 2)
    return np.sqrt(0.07023243613162203 / sigma**2)


def gaussian_sigma_from_bandwidth(bandwidth: float) -> float:
    return np.sqrt(0.07023243613162203 / bandwidth**2)


def compute_analytical_fft_gaussian_pulse(frequencies: np.ndarray, sigma: float) -> Any:
    return (
        # np.exp(-2 * (pi * sigma * frequencies) ** 2)
        np.exp((-2 * (pi * sigma) ** 2) * frequencies**2)
    )


class GaussianEnvelope(EnvelopePattern):
    """Gaussian envelope pattern with f^2 frequency exponent.

    Time domain: exp(-t^2 / (2*sigma^2))
    Frequency domain: exp(-2*pi^2*sigma^2*f^2)
    """

    def __init__(self, bandwidth: float) -> None:
        """Initialize Gaussian envelope.

        Args:
            bandwidth: Target -3dB bandwidth in Hz
        """
        super().__init__(bandwidth)
        self.sigma = gaussian_sigma_from_bandwidth(bandwidth)

    def spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Analytical Gaussian frequency spectrum."""
        return compute_analytical_fft_gaussian_pulse(frequencies, self.sigma)

    def time_domain(
        self, sample_rate: float, edge_threshold: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Gaussian time-domain envelope.

        Window length chosen so envelope = edge_threshold at edges.
        """
        # For Gaussian exp(-t^2 / (2*sigma^2)) = edge_threshold at edges
        # Solve: t_edge = sigma * sqrt(-2 * ln(edge_threshold))
        t_edge = self.sigma * np.sqrt(-2.0 * np.log(edge_threshold))

        # Number of samples from center to edge
        n_half = int(np.ceil(t_edge * sample_rate))
        n_total = 2 * n_half + 1  # Ensure odd length for symmetry

        # Time array using optimized centering
        t_array = centered_time_array(n_total, sample_rate)

        # Gaussian envelope
        envelope = np.exp(-(t_array**2) / (2.0 * self.sigma**2))

        return t_array, envelope

    def get_parameter_info(self) -> dict:
        """Get Gaussian parameters."""
        return {
            "type": "Gaussian",
            "bandwidth": self.bandwidth,
            "sigma": self.sigma,
            "formula_time": "exp(-t^2 / (2*sigma^2))",
            "formula_freq": "exp(-2*pi^2*sigma^2*f^2)",
        }


class SuperGaussianEnvelope(EnvelopePattern):
    """Super-Gaussian envelope pattern with f^4 frequency exponent.

    Frequency domain: exp(-alpha * f^4)
    Time domain: computed via IFFT (no simple analytical formula)
    """

    def __init__(self, bandwidth: float) -> None:
        """Initialize Super-Gaussian envelope.

        Args:
            bandwidth: Target -3dB bandwidth in Hz
        """
        super().__init__(bandwidth)
        self.alpha = super_gaussian_alpha_from_bandwidth(bandwidth)

    def spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Analytical Super-Gaussian frequency spectrum."""
        return compute_analytical_fft_super_gaussian_pulse(frequencies, self.alpha)

    def time_domain(
        self, sample_rate: float, edge_threshold: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Super-Gaussian time-domain envelope via IFFT.

        Window length chosen so envelope decays to edge_threshold at edges.
        """
        # Strategy: compute frequency spectrum, then IFFT
        # Need to determine appropriate FFT size and frequency range

        # Nyquist frequency must be above f_cutoff
        # Choose FFT size to get good frequency resolution
        # Frequency resolution: df = sample_rate / n_fft
        # Want at least 10 points across bandwidth
        df_target = self.bandwidth / 10.0
        n_fft = int(2 ** np.ceil(np.log2(sample_rate / df_target)))

        # Create frequency array (centered at 0 for fftshift)
        freqs = fft.fftfreq(n_fft, 1.0 / sample_rate)
        freqs_shifted = fft.fftshift(freqs)

        # Compute spectrum
        spectrum = self.spectrum(freqs_shifted)

        # IFFT to get time domain (unshift first)
        spectrum_unshifted = fft.ifftshift(spectrum)
        time_envelope = fft.ifft(spectrum_unshifted)
        time_envelope = np.real(time_envelope)  # Should be real due to symmetry

        # Shift time domain: IFFT puts negative times at end, move them to beginning
        time_envelope = fft.fftshift(time_envelope)

        # Normalize
        time_envelope = time_envelope / np.max(np.abs(time_envelope))

        # Find where envelope drops to edge_threshold
        above_threshold = np.where(np.abs(time_envelope) >= edge_threshold)[0]
        if len(above_threshold) == 0:
            # Entire envelope below threshold - use full length
            i_start, i_end = 0, n_fft
        else:
            i_start = above_threshold[0]
            i_end = above_threshold[-1] + 1

        # Extract window and make odd length
        n_window = i_end - i_start
        if n_window % 2 == 0:
            n_window += 1
            i_end = i_start + n_window

        # Ensure we don't exceed array bounds
        if i_end > n_fft:
            i_end = n_fft
            i_start = i_end - n_window

        time_envelope = time_envelope[i_start:i_end] / (i_end - i_start)

        # Create time array
        t_array = centered_time_array(len(time_envelope), sample_rate)

        return t_array, time_envelope

    def get_parameter_info(self) -> dict:
        """Get Super-Gaussian parameters."""
        return {
            "type": "SuperGaussian",
            "bandwidth": self.bandwidth,
            "alpha": self.alpha,
            "formula_freq": "exp(-alpha * f^4)",
            "formula_time": "computed via IFFT",
        }


# ── Log-Spaced Filter Bank ──────────────────────────────────────────────────


class LogSpacedFilterBank:
    """Log-spaced filter bank using any EnvelopePattern.

    Pre-computes all kernels at initialization for efficient spectrogram computation.
    """

    def __init__(
        self,
        envelope_class: type[EnvelopePattern],
        f_min: float,
        f_max: float,
        num_bands: int,
        sample_rate: float,
        edge_threshold: float = 0.01,
    ) -> None:
        """Initialize filter bank with pre-computed kernels.

        Args:
            envelope_class: EnvelopePattern subclass (GaussianEnvelope, SuperGaussianEnvelope)
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
            num_bands: Number of frequency bands
            sample_rate: Sample rate in Hz
            edge_threshold: Threshold for kernel edge decay
        """
        self.envelope_class = envelope_class
        self.f_min = f_min
        self.f_max = f_max
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.edge_threshold = edge_threshold

        # Compute log-spaced frequency bands
        cutoff_frequencies = np.geomspace(f_min, f_max, num_bands + 1)
        self.bandwidths = cutoff_frequencies[1:] - cutoff_frequencies[:-1]
        self.center_frequencies = (cutoff_frequencies[1:] + cutoff_frequencies[:-1]) / 2.0

        # Pre-compute all kernels (cosine and sine modulated by envelope)
        self.envelopes = []
        self.kernels = []  # List of (kernel_cos, kernel_sin) tuples
        self.kernel_time_arrays = []

        for fc, bandwidth in zip(self.center_frequencies, self.bandwidths, strict=False):
            envelope_obj = envelope_class(bandwidth)
            t_array, envelope = envelope_obj.time_domain(sample_rate, edge_threshold)

            # Create modulated kernels: cos/sin carrier * envelope
            two_pi_fc = 2.0 * pi * fc
            kernel_cos = np.cos(two_pi_fc * t_array) * envelope
            kernel_sin = np.sin(two_pi_fc * t_array) * envelope

            self.envelopes.append(envelope_obj)
            self.kernels.append((kernel_cos, kernel_sin))
            self.kernel_time_arrays.append(t_array)

    def get_info(self) -> dict:
        """Get filter bank information."""
        test_envelope = self.envelopes[0]
        info = test_envelope.get_parameter_info()
        return {
            "envelope_type": info["type"],
            "num_bands": self.num_bands,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "sample_rate": self.sample_rate,
            "center_frequencies": self.center_frequencies,
            "bandwidths": self.bandwidths,
        }


# ── Super-Gaussian (order 4) helpers ─────────────────────────────────────────
# Spectrum: G4(f) = exp(-alpha * f^4)
# -3dB condition: alpha = (ln2 / 2) * (2 / bw)^4 = 8*ln2 / bw^4
#
# Steeper roll-off than Gaussian (f^4 vs f^2 in exponent) while remaining
# smooth and entirely analytical.

_LN2 = np.log(2.0)
_SUPER_GAUSSIAN_ALPHA_COEFF = 8.0 * _LN2  # alpha = coeff / bw^4


def super_gaussian_alpha_from_bandwidth(bandwidth: float) -> float:
    """Return alpha such that exp(-alpha * f^4) = 1/sqrt(2) at f = bandwidth/2."""
    return _SUPER_GAUSSIAN_ALPHA_COEFF / bandwidth**4


def super_gaussian_bandwidth_from_alpha(alpha: float) -> float:
    """Inverse: bandwidth from alpha."""
    return (_SUPER_GAUSSIAN_ALPHA_COEFF / alpha) ** 0.25


def super_gaussian_bandwidth(alpha: float) -> float:
    """Full -3dB bandwidth for a given alpha."""
    return super_gaussian_bandwidth_from_alpha(alpha)


def compute_analytical_fft_super_gaussian_pulse(
    frequencies: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Analytical frequency spectrum of a super-Gaussian (order 4) pulse.

        G4(f) = exp(-alpha * f^4)

    Parameters
    ----------
    frequencies : array of frequencies (Hz), centred at 0
    alpha       : shape parameter — use super_gaussian_alpha_from_bandwidth()
    """
    return np.exp((-alpha) * frequencies**4)


# ── Dev / demo ────────────────────────────────────────────────────────────────


def test_envelope_pattern(
    envelope: EnvelopePattern,
    sample_rate: float = 1000.0,
    edge_threshold: float = 0.01,
) -> None:
    """Universal test function for any EnvelopePattern.

    Validates the envelope by comparing numerical FFT against analytical spectrum.

    Args:
        envelope: EnvelopePattern instance to test
        sample_rate: Sample rate for time domain generation
        edge_threshold: Threshold for time domain window edges
    """
    info = envelope.get_parameter_info()
    print(f"\nTesting {info['type']} envelope:")
    print(f"  Bandwidth: {envelope.bandwidth:.2f} Hz")
    for key, value in info.items():
        if key not in ["type", "bandwidth"]:
            print(f"  {key}: {value}")

    # Generate time domain
    t_array, time_envelope = envelope.time_domain(sample_rate, edge_threshold)
    n_samples = len(time_envelope)
    print(f"  Time samples: {n_samples}")
    print(f"  Duration: {t_array[-1] - t_array[0]:.4f} s")

    # Compute numerical FFT
    nfft = max(2 ** np.ceil(np.log2(n_samples)), 8192)
    fft_numerical = fft.fft(time_envelope, nfft)
    fft_numerical = fft.fftshift(fft_numerical)
    fft_numerical = fft_numerical / np.max(np.abs(fft_numerical))  # Normalize

    # Frequency array
    freqs = fft.fftfreq(
        nfft,
        1.0 / sample_rate,
    )
    freqs = fft.fftshift(freqs)

    # Compute analytical spectrum
    spectrum_analytical = envelope.spectrum(freqs)
    spectrum_analytical = spectrum_analytical / np.max(spectrum_analytical)  # Normalize

    # Plot: time domain and frequency comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Time domain plot
    ax1.plot(t_array, time_envelope, "b-", linewidth=2)
    ax1.axhline(
        y=edge_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold={edge_threshold}",
    )
    configure_axes(
        ax1,
        xlabel="Time (s)",
        ylabel="Amplitude",
        title=f"{info['type']} Envelope - Time Domain",
        grid=True,
        legend=True,
    )

    # Frequency domain comparison
    ax2.loglog(
        np.abs(freqs),
        np.abs(fft_numerical),
        "b-",
        label="Numerical FFT",
        linewidth=3,
        alpha=0.7,
    )
    ax2.loglog(
        np.abs(freqs),
        spectrum_analytical,
        "r--",
        label="Analytical spectrum",
        linewidth=2,
    )
    ax2.axvline(
        x=envelope.bandwidth / 2.0,
        color="k",
        linestyle="--",
        label=f"BW/2 = {envelope.bandwidth / 2.0:.2f} Hz",
    )
    ax2.axhline(y=1.0 / np.sqrt(2.0), color="k", linestyle=":", label="-3 dB")
    configure_axes(
        ax2,
        xlabel="Frequency (Hz)",
        ylabel="Normalized Magnitude",
        title=f"{info['type']} - Numerical FFT vs Analytical",
        ylim=(1e-3, 2),
        grid=True,
        legend=True,
    )

    fig.tight_layout()
    plt.show()


def test_filter_bank(
    envelope_class: type[EnvelopePattern],
    f_min: float = 20.0,
    f_max: float = 10000.0,
    num_bands: int = 100,
    sample_rate: float = 44100.0,
) -> None:
    """Test log-spaced filter bank with given envelope pattern.

    Args:
        envelope_class: EnvelopePattern subclass to use
        f_min: Minimum frequency in Hz
        f_max: Maximum frequency in Hz
        num_bands: Number of frequency bands
        sample_rate: Sample rate in Hz
    """
    print(f"\nCreating filter bank with {envelope_class.__name__}...")
    print(f"  Frequency range: {f_min} - {f_max} Hz")
    print(f"  Number of bands: {num_bands}")
    print(f"  Sample rate: {sample_rate} Hz")

    # Create filter bank
    filter_bank = LogSpacedFilterBank(
        envelope_class=envelope_class,
        f_min=f_min,
        f_max=f_max,
        num_bands=num_bands,
        sample_rate=sample_rate,
    )

    # Get info
    info = filter_bank.get_info()
    print("\nFilter bank created:")
    print(f"  Envelope type: {info['envelope_type']}")
    print(f"  Kernel pairs count: {len(filter_bank.kernels)}")

    # Plot 1: Frequency responses
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each band's frequency response
    for i, envelope in enumerate(filter_bank.envelopes):
        fc = filter_bank.center_frequencies[i]
        bw = filter_bank.bandwidths[i]

        # Frequency range for this band
        freq_range = np.linspace(-3 * bw, 3 * bw, 200)
        spectrum = envelope.spectrum(freq_range)

        ax.loglog(freq_range + fc, spectrum, alpha=0.6, linewidth=0.8)

    # Add -3dB line
    ax.axhline(
        y=1.0 / np.sqrt(2.0),
        color="k",
        linestyle="--",
        label="-3 dB",
        linewidth=2,
    )

    configure_axes(
        ax,
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        title=f"{info['envelope_type']} Filter Bank: {num_bands} bands from {f_min}-{f_max} Hz",
        xlim=(f_min * 0.5, f_max * 2.0),
        ylim=(1e-2, 2),
        grid=True,
        legend=True,
    )

    fig.tight_layout()
    plt.show()

    # Plot 2: Example kernel pairs (low, mid, high frequency) - Time domain
    indices = [0, num_bands // 2, num_bands - 1]  # Low, mid, high
    labels = ["Low", "Mid", "High"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for idx, (band_idx, label) in enumerate(zip(indices, labels, strict=False)):
        ax = axes[idx]
        fc = filter_bank.center_frequencies[band_idx]
        t_array = filter_bank.kernel_time_arrays[band_idx]
        kernel_cos, kernel_sin = filter_bank.kernels[band_idx]

        ax.plot(t_array * 1000, kernel_cos, "b-", label="Cosine kernel", linewidth=1.5)
        ax.plot(t_array * 1000, kernel_sin, "r-", label="Sine kernel", linewidth=1.5)

        configure_axes(
            ax,
            xlabel="Time (ms)" if idx == 2 else None,
            ylabel="Amplitude",
            title=f"{label} Frequency: fc = {fc:.1f} Hz",
            grid=True,
            legend=True,
        )

    fig.suptitle(f"{info['envelope_type']} Kernel Pairs - Time Domain", fontsize=14)
    fig.tight_layout()
    plt.show()

    # Plot 3: FFT of kernel pairs - Frequency domain
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for idx, (band_idx, label) in enumerate(zip(indices, labels, strict=False)):
        ax = axes[idx]
        fc = filter_bank.center_frequencies[band_idx]
        kernel_cos, kernel_sin = filter_bank.kernels[band_idx]

        # Compute FFT of kernels
        n_samples = len(kernel_cos)
        n_fft = max(2 ** int(np.ceil(np.log2(n_samples))), 4096)

        fft_cos = fft.fft(kernel_cos, n_fft)
        fft_cos = fft.fftshift(fft_cos)
        fft_cos = fft_cos / np.max(np.abs(fft_cos))

        fft_sin = fft.fft(kernel_sin, n_fft)
        fft_sin = fft.fftshift(fft_sin)
        fft_sin = fft_sin / np.max(np.abs(fft_sin))

        # Frequency array
        freqs = fft.fftfreq(n_fft, 1.0 / sample_rate)
        freqs = fft.fftshift(freqs)

        # Plot magnitude
        ax.semilogy(freqs, np.abs(fft_cos), "b-", label="FFT(Cosine)", linewidth=1.5, alpha=0.8)
        ax.semilogy(freqs, np.abs(fft_sin), "r-", label="FFT(Sine)", linewidth=1.5, alpha=0.8)
        ax.axvline(x=fc, color="k", linestyle="--", label=f"fc = {fc:.1f} Hz", linewidth=1)

        configure_axes(
            ax,
            xlabel="Frequency (Hz)" if idx == 2 else None,
            ylabel="Magnitude",
            title=f"{label} Frequency Kernel FFTs",
            xlim=(
                max(0, fc - 3 * filter_bank.bandwidths[band_idx]),
                fc + 3 * filter_bank.bandwidths[band_idx],
            ),
            ylim=(1e-2, 2),
            grid=True,
            legend=True,
        )

    fig.suptitle(f"{info['envelope_type']} Kernel FFTs - Frequency Domain", fontsize=14)
    fig.tight_layout()
    plt.show()


def dev_envelope_pattern() -> None:
    """Test both envelope patterns and filter banks."""
    # Test individual envelopes
    gaussian_env = GaussianEnvelope(bandwidth=100.0)
    test_envelope_pattern(gaussian_env, sample_rate=10000.0)

    super_gaussian_env = SuperGaussianEnvelope(bandwidth=100.0)
    test_envelope_pattern(super_gaussian_env, sample_rate=10000.0)

    # Test filter banks
    test_filter_bank(GaussianEnvelope, f_min=20.0, f_max=2000.0, num_bands=20)
    test_filter_bank(SuperGaussianEnvelope, f_min=20.0, f_max=2000.0, num_bands=20)
