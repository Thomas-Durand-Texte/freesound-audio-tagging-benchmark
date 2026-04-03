"""Spectrogram computation using custom filter banks and librosa comparison."""

import importlib.util
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from src.features.signal_tools import LogSpacedFilterBank, SuperGaussianEnvelope
from src.features.spectrogram_optimized import MultiResolutionFilterBank
from src.visualization.plots import configure_axes

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def compute_super_gaussian_spectrogram(
    waveform: np.ndarray,
    filter_bank: LogSpacedFilterBank,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute spectrogram using pre-initialized SuperGaussian filter bank.

    Uses log10(|H(f)|^2) where H(f) is the complex response computed via
    fftconvolve with pre-computed cos and sin kernels: H = coef_cos + j*coef_sin.

    Args:
        waveform: Audio waveform
        filter_bank: Pre-initialized LogSpacedFilterBank (initialization not timed)

    Returns:
        Tuple of (spectrogram, time_axis, computation_time)
        - spectrogram: 2D array (n_bands, n_time_steps) in cB (centibels)
        - time_axis: Time axis in seconds
        - computation_time: Computation time in seconds (excludes filter bank init)
    """
    start_time = time.perf_counter()

    n_samples = len(waveform)
    n_bands = filter_bank.num_bands
    time_axis = np.arange(n_samples) / filter_bank.sample_rate

    # Initialize spectrogram array
    spectrogram = np.zeros((n_bands, n_samples))

    # Compute response for each frequency band using pre-computed kernels
    for i, (_kernel_cos, kernel_sin) in enumerate(filter_bank.kernels):
        # Fast convolution using FFT with pre-computed kernels
        # coef_cos = fftconvolve(waveform, kernel_cos, mode="same")
        # coef_sin = fftconvolve(waveform, kernel_sin, mode="same")
        coef_complex = fftconvolve(waveform, kernel_sin, mode="same")

        # Magnitude squared: |H(f)|^2 = coef_cos^2 + coef_sin^2
        # magnitude_squared = coef_cos**2 + coef_sin**2
        magnitude_squared = coef_complex.real**2 + coef_complex.imag**2

        # Convert to dB: 10*log10(magnitude_squared)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        spectrogram[i, :] = np.log10(magnitude_squared + epsilon)

    computation_time = time.perf_counter() - start_time

    return spectrogram, time_axis, computation_time


def compute_librosa_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute mel spectrogram using librosa.

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Tuple of (spectrogram, time_axis, computation_time)
        - spectrogram: 2D array (n_mels, n_time_steps) in dB
        - time_axis: Time axis in seconds
        - computation_time: Computation time in seconds
    """
    start_time = time.perf_counter()

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,  # Power spectrogram (magnitude squared)
    )

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Time axis
    time_axis = librosa.frames_to_time(
        np.arange(mel_spec_db.shape[1]), sr=sample_rate, hop_length=hop_length
    )

    computation_time = time.perf_counter() - start_time

    return mel_spec_db, time_axis, computation_time


def compare_spectrograms(
    waveform: np.ndarray,
    sample_rate: int,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    f_mid: float = 1000.0,
    n_bands: int = 128,
    audio_label: str = "Audio sample",
    signal_duration: float = 3.0,
    hop_length: int = 512,
) -> None:
    """Compare librosa mel spectrogram vs custom SuperGaussian filter bank vs multi-resolution.

    Creates comprehensive visualization showing:
    - Side-by-side spectrograms (all three methods)
    - Timing comparison
    - Frequency response comparison showing benefits

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        f_min: Minimum frequency
        f_max: Maximum frequency
        f_mid: Middle frequency for dual-range filter bank
        n_bands: Number of frequency bands
        audio_label: Label for the audio sample (for plot titles)
        signal_duration: Fixed duration for multi-resolution method
        hop_length: Hop length for multi-resolution downsampling
    """
    print(f"\n{'=' * 70}")
    print(f"Spectrogram Comparison: {audio_label}")
    print(f"{'=' * 70}")
    print(f"Audio duration: {len(waveform) / sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frequency range: {f_min} - {f_max} Hz")
    print(f"Number of bands: {n_bands}")

    # ── 1. Initialize SuperGaussian filter bank (not included in timing) ────────
    print(f"\n{'-' * 70}")
    print("Initializing SuperGaussian filter bank...")
    init_start = time.perf_counter()
    filter_bank = LogSpacedFilterBank(
        envelope_class=SuperGaussianEnvelope,
        # envelope_class=GaussianEnvelope,
        f_min=f_min,
        f_max=f_max,
        num_bands=n_bands,
        sample_rate=sample_rate,
    )
    init_time = time.perf_counter() - init_start
    print(f"  Filter bank initialization: {init_time * 1000:.2f} ms")
    print("  (Note: This is done once, not included in spectrogram computation)")

    # ── 2. Compute custom SuperGaussian spectrogram ─────────────────────────────
    print(f"\n{'-' * 70}")
    print("Computing SuperGaussian spectrogram...")
    sg_spec, sg_time, sg_comp_time = compute_super_gaussian_spectrogram(waveform, filter_bank)
    print(f"  Computation time: {sg_comp_time * 1000:.2f} ms")
    print(f"  Spectrogram shape: {sg_spec.shape}")

    # ── 3. Initialize and compute multi-resolution spectrogram ──────────────────
    print(f"\n{'-' * 70}")
    print("Initializing multi-resolution filter bank...")
    mr_init_start = time.perf_counter()
    mr_filter_bank = MultiResolutionFilterBank(
        envelope_class=SuperGaussianEnvelope,
        f_min=f_min,
        f_max=f_max,
        f_mid=f_mid,
        num_bands=n_bands,
        sample_rate=sample_rate,
        signal_duration=signal_duration,
        spectrum_threshold=0.001,
    )
    mr_init_time = time.perf_counter() - mr_init_start
    print(f"  Multi-resolution initialization: {mr_init_time * 1000:.2f} ms")
    print(f"  Downsample levels used: {mr_filter_bank.downsample_levels}")

    print("Computing multi-resolution spectrogram...")
    mr_spec, mr_time_step, mr_comp_time = mr_filter_bank.compute_spectrogram(
        waveform, hop_length=hop_length
    )
    # Create time axis for multi-resolution
    mr_time = np.arange(mr_spec.shape[1]) * mr_time_step
    print(f"  Computation time: {mr_comp_time * 1000:.2f} ms")
    print(f"  Spectrogram shape: {mr_spec.shape}")

    # ── 4. Compute librosa mel spectrogram ──────────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Computing librosa mel spectrogram...")
    mel_spec, mel_time, mel_comp_time = compute_librosa_spectrogram(
        waveform,
        sample_rate,
        n_mels=n_bands,
        f_min=f_min,
        f_max=f_max,
    )
    print(f"  Computation time: {mel_comp_time * 1000:.2f} ms")
    print(f"  Spectrogram shape: {mel_spec.shape}")

    # ── 5. Performance summary ───────────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Performance Summary:")
    print(f"  SuperGaussian (computation only):  {sg_comp_time * 1000:.2f} ms")
    print(f"  Multi-Resolution (computation only): {mr_comp_time * 1000:.2f} ms")
    print(f"  Librosa mel spectrogram:            {mel_comp_time * 1000:.2f} ms")
    speedup_sg = mel_comp_time / sg_comp_time
    speedup_mr = mel_comp_time / mr_comp_time
    print(f"  Speedup (SG vs Librosa):  {speedup_sg:.2f}x")
    print(f"  Speedup (MR vs Librosa):  {speedup_mr:.2f}x")

    # ── 6. Visualization ─────────────────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Creating visualizations...")

    # Figure 1: Three spectrograms comparison
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    # SuperGaussian spectrogram (log-spaced bands displayed linearly)
    # Clip to max-40 dB (= max-4 cB since we use log10 without *10)
    sg_max = np.max(sg_spec)
    sg_vmin = sg_max - 4.0  # 40 dB = 4 cB
    sg_vmin = sg_max - 6.0  # 60 dB = 6 cB
    print(f"  SuperGaussian max: {sg_max:.2f} cB")
    print(f"  SuperGaussian vmin: {np.min(sg_spec):.2f} cB")

    img1 = ax1.imshow(
        sg_spec,
        aspect="auto",
        origin="lower",
        extent=[sg_time[0], sg_time[-1], 0, n_bands - 1],
        cmap="viridis",
        interpolation="nearest",
        vmin=sg_vmin,
        vmax=sg_max,
    )

    # Add custom y-ticks showing actual frequencies
    # Choose nice frequency values to display
    freq_ticks_hz = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
    freq_ticks_hz = freq_ticks_hz[(freq_ticks_hz >= f_min) & (freq_ticks_hz <= f_max)]

    # Find corresponding band indices for SuperGaussian
    band_indices = []
    for freq in freq_ticks_hz:
        # Find closest center frequency
        idx = np.argmin(np.abs(filter_bank.center_frequencies - freq))
        band_indices.append(idx)

    ax1.set_yticks(band_indices)
    ax1.set_yticklabels([f"{int(f)}" for f in freq_ticks_hz])

    configure_axes(
        ax1,
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title=f"SuperGaussian Spectrogram ({sg_comp_time * 1000:.1f} ms)",
    )
    plt.colorbar(img1, ax=ax1, label="Magnitude (cB)")

    # Multi-Resolution spectrogram (log-spaced bands, downsampled output)
    # Clip to max-40 dB (= max-4 cB since we use log10 without *10)
    mr_max = np.max(mr_spec)
    mr_vmin = mr_max - 4.0  # 40 dB = 4 cB
    mr_vmin = mr_max - 6.0  # 60 dB = 6 cB

    img2 = ax2.imshow(
        mr_spec,
        aspect="auto",
        origin="lower",
        extent=[mr_time[0], mr_time[-1], 0, n_bands - 1],
        cmap="viridis",
        interpolation="nearest",
        vmin=mr_vmin,
        vmax=mr_max,
    )

    # Use same band indices as SuperGaussian (same filter bank structure)
    ax2.set_yticks(band_indices)
    ax2.set_yticklabels([f"{int(f)}" for f in freq_ticks_hz])

    configure_axes(
        ax2,
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title=f"Multi-Resolution Spectrogram ({mr_comp_time * 1000:.1f} ms)",
    )
    plt.colorbar(img2, ax=ax2, label="Magnitude (cB)")

    # Librosa mel spectrogram (mel-spaced bands displayed linearly)
    # Clip to max-40 dB (librosa uses actual dB scale)
    mel_max = np.max(mel_spec)
    mel_vmin = mel_max - 40.0  # 40 dB dynamic range

    img3 = ax3.imshow(
        mel_spec,
        aspect="auto",
        origin="lower",
        extent=[mel_time[0], mel_time[-1], 0, n_bands - 1],
        cmap="viridis",
        interpolation="nearest",
        vmin=mel_vmin,
        vmax=mel_max,
    )

    # Get mel band center frequencies using librosa
    mel_freqs = librosa.mel_frequencies(n_mels=n_bands, fmin=f_min, fmax=f_max)
    print("sg_frequencies:", filter_bank.center_frequencies)
    print("mel_frequencies:", mel_freqs)

    # Find corresponding band indices for the same frequency ticks
    mel_band_indices = []
    for freq in freq_ticks_hz:
        # Find closest mel band center frequency
        idx = np.argmin(np.abs(mel_freqs - freq))
        mel_band_indices.append(idx)

    ax3.set_yticks(mel_band_indices)
    ax3.set_yticklabels([f"{int(f)}" for f in freq_ticks_hz])

    configure_axes(
        ax3,
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title=f"Librosa Mel Spectrogram ({mel_comp_time * 1000:.1f} ms)",
    )
    plt.colorbar(img3, ax=ax3, label="Magnitude (dB)")

    fig1.suptitle(f"Spectrogram Comparison: {audio_label}", fontsize=14, y=0.998)
    fig1.tight_layout()
    plt.show()

    # Figure 2: Frequency response comparison - SuperGaussian vs Mel
    fig2, ax = plt.subplots(figsize=(12, 7))

    # Plot SuperGaussian filter bank frequency responses
    # for i in range(0, n_bands, max(1, n_bands // 20)):  # Plot every ~20th band
    for i in range(0, n_bands):
        envelope = filter_bank.envelopes[i]
        fc = filter_bank.center_frequencies[i]
        bw = filter_bank.bandwidths[i]

        # Frequency range for this band (centered at 0, then shifted to fc for plotting)
        freq_range = np.linspace(-3 * bw, 3 * bw, 200)
        spectrum = envelope.spectrum(freq_range)

        ax.loglog(
            freq_range + fc,
            spectrum,
            "b-",
            alpha=0.4,
            linewidth=1.5,
            label="SuperGaussian" if i == 0 else "",
        )

    # Plot librosa mel filter bank for comparison
    mel_filters = librosa.filters.mel(
        sr=sample_rate,
        n_fft=2048,
        n_mels=n_bands,
        f_min=f_min,
        f_max=f_max,
    )
    freqs_mel = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    for i in range(0, n_bands, max(1, n_bands // 20)):
        ax.semilogy(
            freqs_mel,
            mel_filters[i, :],
            "r-",
            alpha=0.3,
            linewidth=1.2,
            label="Mel filter" if i == 0 else "",
        )

    # Add -3dB reference line
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
        title="Frequency Response Comparison: SuperGaussian vs Mel Filters",
        xlim=(f_min, f_max),
        ylim=(1e-3, 2),
        grid=True,
        legend=True,
        legend_kwargs={"loc": "upper right", "fontsize": 10},
    )

    # Add text box with benefits
    textstr = (
        "SuperGaussian Benefits:\n"
        "• Steeper roll-off (f⁴ vs triangular)\n"
        "• Smoother frequency response\n"
        "• Adjustable time-frequency trade-off\n"
        "• Better frequency selectivity"
    )
    ax.text(
        0.02,
        0.40,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    fig2.tight_layout()
    plt.show()

    print("Visualizations complete!")
    print(f"{'=' * 70}\n")
