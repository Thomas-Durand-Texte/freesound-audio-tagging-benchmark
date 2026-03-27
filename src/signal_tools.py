from typing import Any

import numpy as np
from numpy import pi
from scipy import fft
import matplotlib.pyplot as plt
from scipy import fft, signal as sp_signal

from .plot_utils import init_figure


def gaussian_pulse_bandwidth(sigma: float):
    # Compute analytical bandwidth (-3dB from max)
    # For a Gaussian spectrum G(f) = exp(-2*pi^2*sigma^2*f^2)
    # -3dB point occurs when G(f) = G(0) / sqrt(2) = 0.7071
    # exp(-2*pi^2*sigma^2*f_3dB^2) = 0.7071
    # -2*pi^2*sigma^2*f_3dB^2 = ln(0.7071)
    # f_3dB = sqrt(-ln(0.7071) / (2*pi^2*sigma^2))
    # -np.log(0.7071)) = 0.173291590185971
    # return 2 * np.sqrt(0.173291590185971 / (pi * sigma) ** 2)
    return np.sqrt(0.07023243613162203 / sigma ** 2)


def gaussian_sigma_from_bandwidth(bandwidth: float):
    return np.sqrt(0.07023243613162203 / bandwidth ** 2)


def compute_analytical_fft_gaussian_pulse(frequencies: np.ndarray, sigma: float) -> Any:
    return (
        # np.exp(-2 * (pi * sigma * frequencies) ** 2)
        np.exp((-2 * (pi * sigma) ** 2) * frequencies ** 2)
    )


class GaussianPulsesLogSpaced:
    def __init__(self, fmin: float, fmax: float, num_pulses: int):
        self.fmin = fmin
        self.fmax = fmax
        self.num_pulses = num_pulses
        cutoff_frequencies = np.geomspace(fmin, fmax, num_pulses + 1)
        self.bandwidths = cutoff_frequencies[1:] - cutoff_frequencies[:-1]
        self.pulse_frequencies = (cutoff_frequencies[1:] + cutoff_frequencies[:-1]) / 2



# ... existing code ...

def gaussian_sigma_from_bandwidth(bandwidth: float):
    return np.sqrt(0.07023243613162203 / bandwidth ** 2)


def compute_analytical_fft_gaussian_pulse(frequencies: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp((-2 * (pi * sigma) ** 2) * frequencies ** 2)


# ── Super-Gaussian (order 4) helpers ─────────────────────────────────────────
# Spectrum: G4(f) = exp(-alpha * f^4)
# -3dB condition: alpha = (ln2 / 2) * (2 / bw)^4 = 8*ln2 / bw^4
#
# Steeper roll-off than Gaussian (f^4 vs f^2 in exponent) while remaining
# smooth and entirely analytical.

_LN2 = np.log(2.0)
_SUPER_GAUSSIAN_ALPHA_COEFF = 8.0 * _LN2   # alpha = coeff / bw^4


def super_gaussian_alpha_from_bandwidth(bandwidth: float) -> float:
    """Return alpha such that exp(-alpha * f^4) = 1/sqrt(2) at f = bandwidth/2."""
    return _SUPER_GAUSSIAN_ALPHA_COEFF / bandwidth ** 4


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
    return np.exp((-alpha) * frequencies ** 4)


class SuperGaussianPulsesLogSpaced:
    """
    Log-spaced filter bank of super-Gaussian (f^4 exponent) pulses.

    Mirrors GaussianPulsesLogSpaced exactly; drop-in replacement with
    steeper frequency roll-off.
    """

    def __init__(self, fmin: float, fmax: float, num_pulses: int):
        self.fmin = fmin
        self.fmax = fmax
        self.num_pulses = num_pulses
        cutoff_frequencies = np.geomspace(fmin, fmax, num_pulses + 1)
        self.bandwidths = cutoff_frequencies[1:] - cutoff_frequencies[:-1]
        self.pulse_frequencies = (cutoff_frequencies[1:] + cutoff_frequencies[:-1]) / 2






# ── Dev / demo ────────────────────────────────────────────────────────────────


def dev_gaussian():
    # Parameters for Gaussian pulse
    N = 2048  # Number of samples
    fs = 1000  # Sampling frequency (Hz)
    sigma = 0.05  # Standard deviation of Gaussian

    # Generate time array
    t = np.arange(N) / fs
    t = t - t[N // 2]  # Center at zero

    # 1. Generate Gaussian pulse
    gaussian_pulse = np.exp(-t ** 2 / (2 * sigma ** 2))
    gaussian_pulse = gaussian_pulse / np.sum(gaussian_pulse)

    # 2. Compute FFT numerically
    fft_numerical = fft.fft(gaussian_pulse)
    fft_numerical = fft.fftshift(fft_numerical)

    # Frequency array
    freqs = fft.fftfreq(N, 1 / fs)
    freqs = fft.fftshift(freqs)

    # 2. Compute FFT from formula (analytical)
    # FFT of Gaussian is: sigma * sqrt(2*pi) * exp(-2*pi^2*sigma^2*f^2)
    fft_analytical = compute_analytical_fft_gaussian_pulse(freqs, sigma)

    bandwidth_analytical = gaussian_pulse_bandwidth(sigma)
    # print(f"Analytical bandwidth: {bandwidth_analytical:.2f} Hz")
    # print(f"check sigma from bandwidth: {gaussian_sigma_from_bandwidth(bandwidth_analytical):.2f}")

    # 3. Plot both curves in log-log space
    fig, ax = init_figure(
        xlabel='Frequency (Hz)',
        ylabel='Magnitude',
        title='Gaussian Pulse FFT: Numerical vs Analytical',
    )
    ax.loglog(np.abs(freqs), np.abs(fft_numerical), 'b-', label='Numerical FFT', linewidth=3)
    ax.loglog(np.abs(freqs), np.abs(fft_analytical), 'r--', label='Analytical FFT', linewidth=2)
    ax.axvline(x=bandwidth_analytical / 2, color='k', linestyle='--', label=f'Bandwidth: {bandwidth_analytical:.2f} Hz')
    ax.axhline(y=0.707, color='k', linestyle='--', label='1e-3 dB')
    ax.set_ylim(1e-3, 2)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()

    gaussian_pulses = GaussianPulsesLogSpaced(fmin=20, fmax=10000, num_pulses=100)
    fig, ax = init_figure(
        xlabel='Frequency (Hz)',
        ylabel='Magnitude',
    )
    ax.hlines(0.707, gaussian_pulses.fmin, gaussian_pulses.fmax, color='k', linestyle='--', label='1e-3 dB')
    for i, (fc, bw) in enumerate(zip(gaussian_pulses.pulse_frequencies, gaussian_pulses.bandwidths)):
        frequencies = np.linspace(-3*bw, 3*bw, 200)
        ax.loglog(
            frequencies + fc,
            compute_analytical_fft_gaussian_pulse(
                frequencies,
                gaussian_sigma_from_bandwidth(bw)
            ),
            label=f"Pulse {fc - bw / 2:.2f} - {fc + bw / 2:.2f} Hz"
        )
        if len(gaussian_pulses.pulse_frequencies) < 10:
            ax.legend(ncols=2)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_ylim(1e-2, 2)
        ax.set_xlim(10, 10000)

    plt.show()



def dev_super_gaussian():
    pulses = SuperGaussianPulsesLogSpaced(fmin=20, fmax=10_000, num_pulses=100)

    fig, ax = init_figure(xlabel='Frequency (Hz)', ylabel='Magnitude')
    ax.hlines(0.707, pulses.fmin, pulses.fmax, color='k', linestyle='--', label='-3 dB')

    for fc, bw in zip(pulses.pulse_frequencies, pulses.bandwidths):
        alpha = super_gaussian_alpha_from_bandwidth(bw)
        frequencies = np.linspace(-3 * bw, 3 * bw, 200)
        ax.loglog(
            frequencies + fc,
            compute_analytical_fft_super_gaussian_pulse(frequencies, alpha),
        )

    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_ylim(1e-2, 2)
    ax.set_xlim(10, 10_000)
    plt.show()


