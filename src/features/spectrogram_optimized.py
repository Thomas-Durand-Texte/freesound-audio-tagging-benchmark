"""Optimized spectrogram computation methods."""

import time
from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.signal import oaconvolve

from src.features.signal_tools import LogSpacedFilterBank

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def compute_sg_spectrogram_fft_optimized(
    waveform: np.ndarray,
    filter_bank: LogSpacedFilterBank,
    hop_length: int = 512,
    spectrum_threshold: float = 0.01,
) -> tuple[np.ndarray, float, float]:
    """FFT-based filtering with vectorized operations.

    Args:
        waveform: Audio waveform
        filter_bank: Pre-initialized LogSpacedFilterBank
        hop_length: Downsampling factor
        spectrum_threshold: Threshold for sparse filtering

    Returns:
        Tuple of (spectrogram, time_step, computation_time)
        - spectrogram: 2D array (n_bands, n_frames) in Bell
        - time_step: Time between frames in seconds
        - computation_time: Computation time in seconds
    """
    start_time = time.perf_counter()

    n_samples = len(waveform)
    n_bands = filter_bank.num_bands
    sample_rate = filter_bank.sample_rate

    # Compute FFT of waveform once
    n_fft = n_samples
    waveform_fft = fft.fft(waveform, n=n_fft)

    # Frequency step
    df = sample_rate / n_fft

    # Number of output frames
    n_frames = (n_samples + hop_length - 1) // hop_length
    time_step = hop_length / sample_rate

    # Initialize spectrogram
    spectrogram = np.zeros((n_bands, n_frames))

    # Process each frequency band
    for i in range(n_bands):
        envelope = filter_bank.envelopes[i]
        fc = filter_bank.center_frequencies[i]
        alpha = envelope.alpha

        # Compute frequency extent analytically (where spectrum > threshold)
        # SuperGaussian: exp(-alpha * f^4) > threshold
        # Solve: f_max = (ln(1/threshold) / alpha)^(1/4)
        f_max_band = (np.log(1.0 / spectrum_threshold) / alpha) ** 0.25

        # Number of frequency bins needed (rounded)
        n_bins = int(f_max_band / df + 0.5)

        # Create frequency array directly (centered at 0, symmetric)
        freq_rel = np.linspace(-n_bins * df, n_bins * df, 2 * n_bins + 1)

        # Compute spectrum values (vectorized)
        spectrum_vals = envelope.spectrum(freq_rel)

        # Initialize filtered spectrum
        filtered_spectrum = np.zeros(n_fft, dtype=complex)

        # Positive frequency component (around +fc)
        bin_center_pos = int(fc / df + 0.5)

        # Compute valid index range analytically (bins are linearly spaced)
        bin_start_pos = bin_center_pos - n_bins
        bin_end_pos = bin_center_pos + n_bins + 1

        # Clip to valid FFT range
        i_start_pos = max(0, -bin_start_pos) if bin_start_pos < 0 else 0
        i_end_pos = (
            len(spectrum_vals) - max(0, bin_end_pos - n_fft // 2)
            if bin_end_pos > n_fft // 2
            else len(spectrum_vals)
        )

        bin_start_clipped = max(0, bin_start_pos)
        bin_end_clipped = min(n_fft // 2, bin_end_pos)

        # Apply filter
        filtered_spectrum[bin_start_clipped:bin_end_clipped] = (
            waveform_fft[bin_start_clipped:bin_end_clipped] * spectrum_vals[i_start_pos:i_end_pos]
        )

        # Negative frequency component (around -fc, in upper half of FFT)
        bin_center_neg = n_fft - bin_center_pos
        bin_start_neg = bin_center_neg - n_bins
        bin_end_neg = bin_center_neg + n_bins + 1

        # Clip to valid FFT range (upper half)
        i_start_neg = max(0, (n_fft // 2) - bin_start_neg) if bin_start_neg < n_fft // 2 else 0
        i_end_neg = (
            len(spectrum_vals) - max(0, bin_end_neg - n_fft)
            if bin_end_neg > n_fft
            else len(spectrum_vals)
        )

        bin_start_clipped_neg = max(n_fft // 2, bin_start_neg)
        bin_end_clipped_neg = min(n_fft, bin_end_neg)

        # Apply filter
        filtered_spectrum[bin_start_clipped_neg:bin_end_clipped_neg] = (
            waveform_fft[bin_start_clipped_neg:bin_end_clipped_neg]
            * spectrum_vals[i_start_neg:i_end_neg]
        )

        # IFFT to time domain
        filtered_signal = np.real(fft.ifft(filtered_spectrum))

        # Magnitude squared and downsample (vectorized slice)
        magnitude_squared = filtered_signal**2
        magnitude_squared_downsampled = magnitude_squared[::hop_length][:n_frames]

        # Convert to cB
        epsilon = 1e-10
        spectrogram[i, :] = np.log10(magnitude_squared_downsampled + epsilon)

    computation_time = time.perf_counter() - start_time

    return spectrogram, time_step, computation_time


def compute_sg_spectrogram_oaconvolve_optimized(
    waveform: np.ndarray,
    filter_bank: LogSpacedFilterBank,
    hop_length: int = 512,
) -> tuple[np.ndarray, float, float]:
    """Overlap-add convolution with optimized downsampling.

    Args:
        waveform: Audio waveform
        filter_bank: Pre-initialized LogSpacedFilterBank
        hop_length: Downsampling factor

    Returns:
        Tuple of (spectrogram, time_step, computation_time)
    """
    start_time = time.perf_counter()

    n_samples = len(waveform)
    n_bands = filter_bank.num_bands
    n_frames = (n_samples + hop_length - 1) // hop_length
    time_step = hop_length / filter_bank.sample_rate

    # Initialize spectrogram
    spectrogram = np.zeros((n_bands, n_frames))

    # Compute response for each frequency band
    for i, (kernel_cos, kernel_sin) in enumerate(filter_bank.kernels):
        # Overlap-add convolution
        coef_cos = oaconvolve(waveform, kernel_cos, mode="same")
        coef_sin = oaconvolve(waveform, kernel_sin, mode="same")

        # Magnitude squared and downsample with slice (faster than indexing)
        magnitude_squared = coef_cos**2 + coef_sin**2
        magnitude_squared_downsampled = magnitude_squared[::hop_length][:n_frames]

        # Convert to cB
        epsilon = 1e-10
        spectrogram[i, :] = np.log10(magnitude_squared_downsampled + epsilon)

    computation_time = time.perf_counter() - start_time

    return spectrogram, time_step, computation_time


class STFTFilterBank:
    """Pre-computed STFT filter bank weights."""

    def __init__(
        self, filter_bank: LogSpacedFilterBank, n_fft: int = 2048, spectrum_threshold: float = 0.01
    ) -> None:
        """Initialize and pre-compute filter weights.

        Args:
            filter_bank: LogSpacedFilterBank instance
            n_fft: FFT size for STFT
            spectrum_threshold: Threshold for sparse weights
        """
        self.filter_bank = filter_bank
        self.n_fft = n_fft
        self.sample_rate = filter_bank.sample_rate
        self.n_bands = filter_bank.num_bands

        # Pre-compute frequency bins
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa not available")

        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        n_freqs = len(freqs)

        # Pre-compute filter weights (n_bands, n_freqs)
        self.filter_weights = np.zeros((self.n_bands, n_freqs))

        for i in range(self.n_bands):
            envelope = filter_bank.envelopes[i]
            fc = filter_bank.center_frequencies[i]

            # Spectrum centered at fc (vectorized)
            freq_relative = freqs - fc
            spectrum = envelope.spectrum(freq_relative)

            # Apply threshold for sparsity
            spectrum[spectrum < spectrum_threshold] = 0.0
            self.filter_weights[i, :] = spectrum

    def compute_spectrogram(
        self, waveform: np.ndarray, hop_length: int = 512
    ) -> tuple[np.ndarray, float, float]:
        """Compute spectrogram (only STFT + matrix mult are timed).

        Args:
            waveform: Audio waveform
            hop_length: Hop length for STFT

        Returns:
            Tuple of (spectrogram, time_step, computation_time)
        """
        start_time = time.perf_counter()

        # Compute STFT
        stft_matrix = librosa.stft(waveform, n_fft=self.n_fft, hop_length=hop_length)
        power_spectrogram = np.abs(stft_matrix) ** 2

        # Apply filter bank: matrix multiplication
        spectrogram = self.filter_weights @ power_spectrogram

        # Convert to cB
        epsilon = 1e-10
        spectrogram = np.log10(spectrogram + epsilon)

        computation_time = time.perf_counter() - start_time

        # Time step
        time_step = hop_length / self.sample_rate

        return spectrogram, time_step, computation_time


class GPUFilterBank:
    """Pre-computed GPU filter bank for PyTorch."""

    def __init__(self, filter_bank: LogSpacedFilterBank, device: str = "mps") -> None:
        """Initialize and move kernels to GPU.

        Args:
            filter_bank: LogSpacedFilterBank instance
            device: PyTorch device ('mps', 'cuda', or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.filter_bank = filter_bank
        self.device = device
        self.n_bands = filter_bank.num_bands
        self.sample_rate = filter_bank.sample_rate

        # Find maximum kernel length (kernels have different lengths)
        max_kernel_length = max(len(k[0]) for k in filter_bank.kernels)

        # Pad all kernels to same length and stack
        kernels_cos = []
        kernels_sin = []
        for kernel_cos, kernel_sin in filter_bank.kernels:
            # Pad with zeros to max length
            pad_length = max_kernel_length - len(kernel_cos)
            if pad_length > 0:
                # Symmetric padding (half on each side)
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                kernel_cos_padded = np.pad(kernel_cos, (pad_left, pad_right), mode="constant")
                kernel_sin_padded = np.pad(kernel_sin, (pad_left, pad_right), mode="constant")
            else:
                kernel_cos_padded = kernel_cos
                kernel_sin_padded = kernel_sin

            kernels_cos.append(kernel_cos_padded)
            kernels_sin.append(kernel_sin_padded)

        # Shape: (out_channels=n_bands, in_channels=1, kernel_length)
        self.kernels_cos_tensor = (
            torch.from_numpy(np.stack(kernels_cos)[:, np.newaxis, :]).float().to(device)
        )
        self.kernels_sin_tensor = (
            torch.from_numpy(np.stack(kernels_sin)[:, np.newaxis, :]).float().to(device)
        )

    def compute_spectrogram(
        self, waveform: np.ndarray, hop_length: int = 512
    ) -> tuple[np.ndarray, float, float]:
        """Compute spectrogram on GPU (only convolution is timed).

        Args:
            waveform: Audio waveform (numpy array)
            hop_length: Downsampling factor

        Returns:
            Tuple of (spectrogram, time_step, computation_time)
        """
        n_samples = len(waveform)
        n_frames = (n_samples + hop_length - 1) // hop_length
        time_step = hop_length / self.sample_rate

        # Convert waveform to tensor and move to GPU
        waveform_tensor = (
            torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Start timing AFTER data transfer
        start_time = time.perf_counter()

        # Convolve with all kernels in parallel
        coef_cos = F.conv1d(waveform_tensor, self.kernels_cos_tensor, padding="same")
        coef_sin = F.conv1d(waveform_tensor, self.kernels_sin_tensor, padding="same")

        # Magnitude squared: shape (1, n_bands, n_samples)
        magnitude_squared = coef_cos**2 + coef_sin**2

        # Downsample
        magnitude_squared_downsampled = magnitude_squared[0, :, ::hop_length][:, :n_frames]

        # Transfer back to CPU
        magnitude_squared_np = magnitude_squared_downsampled.cpu().numpy()

        computation_time = time.perf_counter() - start_time

        # Convert to cB
        epsilon = 1e-10
        spectrogram = np.log10(magnitude_squared_np + epsilon)

        return spectrogram, time_step, computation_time


def benchmark_spectrogram_methods(
    waveform: np.ndarray,
    filter_bank: LogSpacedFilterBank,
    hop_length: int = 512,
    n_fft: int = 2048,
    spectrum_threshold: float = 0.01,
    test_gpu: bool = False,
    gpu_device: str = "mps",
) -> dict:
    """Benchmark all spectrogram computation methods.

    Args:
        waveform: Audio waveform
        filter_bank: Pre-initialized LogSpacedFilterBank
        hop_length: Downsampling factor
        n_fft: FFT size for STFT method
        spectrum_threshold: Threshold for FFT/STFT methods
        test_gpu: Whether to test GPU method
        gpu_device: PyTorch device for GPU method

    Returns:
        Dictionary with results for each method
    """
    sample_rate = filter_bank.sample_rate
    results = {}

    print(f"\n{'=' * 70}")
    print("Benchmarking Spectrogram Methods")
    print(f"{'=' * 70}")
    print(f"Waveform length: {len(waveform)} samples ({len(waveform) / sample_rate:.2f}s)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Hop length: {hop_length}")
    print(f"Number of bands: {filter_bank.num_bands}")

    # Method 1: FFT-based optimized
    print(f"\n{'-' * 70}")
    print("Method 1: FFT-based (optimized, vectorized)")
    spec, time_step, comp_time = compute_sg_spectrogram_fft_optimized(
        waveform, filter_bank, hop_length, spectrum_threshold
    )
    print(f"  Computation time: {comp_time * 1000:.2f} ms")
    print(f"  Output shape: {spec.shape}")
    print(f"  Time step: {time_step:.6f} s")
    results["fft_optimized"] = {
        "time_ms": comp_time * 1000,
        "shape": spec.shape,
        "spectrogram": spec,
    }

    # Method 2: OAConvolve optimized
    print(f"\n{'-' * 70}")
    print("Method 2: Overlap-Add Convolution (oaconvolve)")
    spec, time_step, comp_time = compute_sg_spectrogram_oaconvolve_optimized(
        waveform, filter_bank, hop_length
    )
    print(f"  Computation time: {comp_time * 1000:.2f} ms")
    print(f"  Output shape: {spec.shape}")
    results["oaconvolve"] = {
        "time_ms": comp_time * 1000,
        "shape": spec.shape,
        "spectrogram": spec,
    }

    # Method 3: STFT-based
    if LIBROSA_AVAILABLE:
        print(f"\n{'-' * 70}")
        print("Method 3: STFT-based (librosa)")
        print("  Initializing STFT filter bank...")
        init_start = time.perf_counter()
        stft_bank = STFTFilterBank(filter_bank, n_fft, spectrum_threshold)
        init_time = time.perf_counter() - init_start
        print(f"    Initialization time: {init_time * 1000:.2f} ms (done once)")

        spec, time_step, comp_time = stft_bank.compute_spectrogram(waveform, hop_length)
        print(f"  Computation time: {comp_time * 1000:.2f} ms")
        print(f"  Output shape: {spec.shape}")
        results["stft"] = {
            "init_time_ms": init_time * 1000,
            "time_ms": comp_time * 1000,
            "shape": spec.shape,
            "spectrogram": spec,
        }

    # Method 4: GPU-based
    if test_gpu and TORCH_AVAILABLE:
        print(f"\n{'-' * 70}")
        print(f"Method 4: GPU-based (PyTorch on {gpu_device})")
        print("  Initializing GPU filter bank...")
        init_start = time.perf_counter()
        gpu_bank = GPUFilterBank(filter_bank, gpu_device)
        init_time = time.perf_counter() - init_start
        print(f"    Initialization time: {init_time * 1000:.2f} ms (done once)")

        spec, time_step, comp_time = gpu_bank.compute_spectrogram(waveform, hop_length)
        print(f"  Computation time: {comp_time * 1000:.2f} ms")
        print(f"  Output shape: {spec.shape}")
        results["gpu"] = {
            "init_time_ms": init_time * 1000,
            "time_ms": comp_time * 1000,
            "shape": spec.shape,
            "spectrogram": spec,
        }

    # Comparison with librosa (if available)
    if LIBROSA_AVAILABLE:
        print(f"\n{'-' * 70}")
        print("Reference: Librosa Mel Spectrogram")
        start_time = time.perf_counter()
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=filter_bank.num_bands,
            power=2.0,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        comp_time = time.perf_counter() - start_time
        print(f"  Computation time: {comp_time * 1000:.2f} ms")
        print(f"  Output shape: {mel_spec_db.shape}")
        results["librosa"] = {
            "time_ms": comp_time * 1000,
            "shape": mel_spec_db.shape,
            "spectrogram": mel_spec_db,
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("Performance Summary")
    print(f"{'=' * 70}")
    print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup vs Librosa':<20}")
    print(f"{'-' * 70}")

    librosa_time = results.get("librosa", {}).get("time_ms", None)
    for method, data in results.items():
        time_ms = data["time_ms"]
        if librosa_time:
            speedup = librosa_time / time_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        print(f"{method:<30} {time_ms:<15.2f} {speedup_str:<20}")

    return results


@dataclass
class BandProcessingInfo:
    """Pre-computed information for processing one frequency band."""

    downsample_level: int  # Number of times to downsample (0 = no downsampling)
    sample_rate_effective: float  # Effective sample rate after downsampling
    n_samples_effective: int  # Number of samples after downsampling

    # Frequency domain slicing for positive and negative frequencies
    bin_start_pos: int
    bin_end_pos: int
    weight_start_pos: int
    weight_end_pos: int

    bin_start_neg: int
    bin_end_neg: int
    weight_start_neg: int
    weight_end_neg: int

    # Pre-computed spectrum weights (centered at 0)
    spectrum_weights: np.ndarray
    # Pre-computed spectrum weights (centered at 0) for positive frequencies only
    spectrum_weights_pos: np.ndarray


class MultiResolutionFilterBank:
    """Multi-resolution filter bank with pre-computed downsampling and frequency weights.

    Optimizations:
    - All weights pre-computed for fixed signal duration
    - Adaptive downsampling: high frequencies on downsampled signal
    - Downsampled signals computed once at inference start
    - Frequency domain filtering with pre-computed slices (no index arrays)
    - All indices computed analytically (no boolean masking)
    """

    def __init__(
        self,
        envelope_class: type,
        f_min: float,
        f_max: float,
        num_bands: int,
        sample_rate: int,
        signal_duration: float,
        f_mid: float | None = None,
        spectrum_threshold: float = 0.001,
    ) -> None:
        """Initialize multi-resolution filter bank.

        Args:
            envelope_class: Envelope pattern class (GaussianEnvelope or SuperGaussianEnvelope)
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
            num_bands: Number of frequency bands
            sample_rate: Sample rate in Hz
            signal_duration: Fixed signal duration in seconds (signals will be cropped/padded)
            f_mid: Middle frequency for dual-range splitting (optional)
            spectrum_threshold: Threshold for sparse filtering
        """
        self.envelope_class = envelope_class
        self.f_min = f_min
        self.f_max = f_max
        self.f_mid = f_mid
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.signal_duration = signal_duration
        self.spectrum_threshold = spectrum_threshold
        self.n_samples = int(sample_rate * signal_duration)

        # Compute frequency limits with optional dual-range splitting
        if f_mid is not None:
            # Split bands into two ranges: f_min to f_mid and f_mid to f_max
            n_bands_low = num_bands // 2
            n_bands_high = num_bands - n_bands_low

            frequency_limits_low = np.geomspace(f_min, f_mid, n_bands_low + 1)
            frequency_limits_high = np.geomspace(f_mid, f_max, n_bands_high + 1)

            # Concatenate, avoiding duplicate f_mid
            frequency_limits = np.concatenate([frequency_limits_low[:-1], frequency_limits_high])
        else:
            # Single range from f_min to f_max
            frequency_limits = np.geomspace(f_min, f_max, num_bands + 1)

        # Compute log-spaced center frequencies and bandwidths
        self.center_frequencies = (frequency_limits[:-1] + frequency_limits[1:]) / 2

        # Bandwidth: geometric mean of adjacent center frequencies
        self.bandwidths = frequency_limits[1:] - frequency_limits[:-1]

        # Pre-compute processing info for each band
        self.band_infos = [self._precompute_band_processing(i) for i in range(num_bands)]

        # Determine unique downsample levels needed
        self.downsample_levels = sorted({info.downsample_level for info in self.band_infos})

    def _determine_downsample_level(self, fc: float, bw: float) -> int:
        """Determine optimal downsample level for a frequency band.

        Rules:
        - fmin = fc - bw/2 must be > sample_rate_downsampled / 10
        - Preferably fmax = fc + bw/2 < sample_rate_downsampled / 6

        Returns:
            Number of downsampling steps (0 = no downsampling)
        """
        fc - bw / 2
        fmax_band = fc + bw / 2

        level = 10  # Safety limit
        while level:
            sr_ds = self.sample_rate / (2**level)
            if fmax_band <= sr_ds / 6:
                break
            level -= 1
        level = max(level, 0)

        # level = 0
        # while level < 10:  # Safety limit
        #     sr_ds = self.sample_rate / (2**level)
        #
        #     # Check minimum constraint (must satisfy)
        #     # if fmin_band <= sr_ds / 10:
        #     if fmax_band > sr_ds / 6:
        #         break
        #
        #     # Check preferred constraint
        #     # if fmax_band >= sr_ds / 6:
        #     #     break
        #
        #     # Try next level
        #     level += 1
        #
        # # Go back one level if we exceeded constraints
        # if level > 0:
        #     level -= 1

        return level

    def _precompute_band_processing(self, band_idx: int) -> BandProcessingInfo:
        """Pre-compute all processing info for one frequency band.

        Args:
            band_idx: Band index

        Returns:
            BandProcessingInfo with all pre-computed data
        """
        fc = self.center_frequencies[band_idx]
        bw = self.bandwidths[band_idx]

        # Determine downsample level
        level = self._determine_downsample_level(fc, bw)
        sr_effective = self.sample_rate / (2**level)
        n_effective = self.n_samples // (2**level)

        # Create envelope for this band
        envelope = self.envelope_class(bandwidth=bw)

        # Frequency step for FFT
        df = sr_effective / n_effective
        n_fft = n_effective

        # Compute frequency extent analytically (where spectrum > threshold)
        if hasattr(envelope, "alpha"):  # SuperGaussian
            alpha = envelope.alpha
            f_max_band = (np.log(1.0 / self.spectrum_threshold) / alpha) ** 0.25
        else:  # Gaussian or other
            # Use 3*bandwidth as conservative estimate
            f_max_band = 3.0 * bw

        # Number of frequency bins needed (rounded)
        n_bins = int(f_max_band / df + 0.5)

        # Create frequency array directly (centered at 0, symmetric)
        freq_rel = np.linspace(-n_bins * df, n_bins * df, 2 * n_bins + 1)

        # Compute spectrum values (vectorized)
        spectrum_vals = envelope.spectrum(freq_rel)

        # ── Positive frequency component (around +fc) ────────────────────────
        bin_center_pos = int(fc / df + 0.5)

        # Compute valid index range analytically
        bin_start_pos = bin_center_pos - n_bins
        bin_end_pos = bin_center_pos + n_bins + 1

        # Clip to valid FFT range [0, n_fft//2]
        weight_start_pos = max(0, -bin_start_pos) if bin_start_pos < 0 else 0
        weight_end_pos = (
            len(spectrum_vals) - max(0, bin_end_pos - n_fft // 2)
            if bin_end_pos > n_fft // 2
            else len(spectrum_vals)
        )

        bin_start_pos_clipped = max(0, bin_start_pos)
        bin_end_pos_clipped = min(n_fft // 2, bin_end_pos)

        # ── Negative frequency component (around -fc, in upper half of FFT) ──
        bin_center_neg = n_fft - bin_center_pos
        bin_start_neg = bin_center_neg - n_bins
        bin_end_neg = bin_center_neg + n_bins + 1

        # Clip to valid FFT range [n_fft//2, n_fft]
        weight_start_neg = max(0, (n_fft // 2) - bin_start_neg) if bin_start_neg < n_fft // 2 else 0
        weight_end_neg = (
            len(spectrum_vals) - max(0, bin_end_neg - n_fft)
            if bin_end_neg > n_fft
            else len(spectrum_vals)
        )

        bin_start_neg_clipped = max(n_fft // 2, bin_start_neg)
        bin_end_neg_clipped = min(n_fft, bin_end_neg)

        return BandProcessingInfo(
            downsample_level=level,
            sample_rate_effective=sr_effective,
            n_samples_effective=n_effective,
            bin_start_pos=bin_start_pos_clipped,
            bin_end_pos=bin_end_pos_clipped,
            weight_start_pos=weight_start_pos,
            weight_end_pos=weight_end_pos,
            bin_start_neg=bin_start_neg_clipped,
            bin_end_neg=bin_end_neg_clipped,
            weight_start_neg=weight_start_neg,
            weight_end_neg=weight_end_neg,
            spectrum_weights=spectrum_vals,
            spectrum_weights_pos=spectrum_vals[weight_start_pos:weight_end_pos],
        )

    def _prepare_signal(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or zero-pad signal to fixed length.

        Args:
            waveform: Input waveform

        Returns:
            Waveform with exactly self.n_samples length
        """
        if len(waveform) >= self.n_samples:
            return waveform[: self.n_samples]
        else:
            padded = np.zeros(self.n_samples)
            padded[: len(waveform)] = waveform
            return padded

    def compute_spectrogram(
        self,
        waveform: np.ndarray,
        hop_length: int = 512,
    ) -> tuple[np.ndarray, float, float]:
        """Compute multi-resolution spectrogram.

        Args:
            waveform: Audio waveform (will be cropped/padded to fixed length)
            hop_length: Downsampling factor for output

        Returns:
            Tuple of (spectrogram, time_step, computation_time)
            - spectrogram: 2D array (n_bands, n_frames) in cB
            - time_step: Time between frames in seconds
            - computation_time: Computation time in seconds
        """
        start_time = time.perf_counter()

        # Prepare signal (crop or pad to fixed length)
        signal = self._prepare_signal(waveform)

        # ── Pre-compute ALL downsampled versions and their FFTs ONCE ─────────
        downsampled_signals = {0: signal}
        downsampled_ffts = {0: fft.fft(signal)}

        for level in self.downsample_levels:
            if level == 0:
                continue

            # Start from previous level
            signal_ds = downsampled_signals[level - 1]

            # Downsample once more: signal[:int(len(signal)/2)*2].reshape(-1, 2).mean(axis=1)
            n_even = (len(signal_ds) // 2) * 2
            signal_ds = signal_ds[:n_even].reshape(-1, 2).mean(axis=1)

            downsampled_signals[level] = signal_ds
            downsampled_ffts[level] = fft.fft(signal_ds)

        # Initialize output
        n_frames = (self.n_samples + hop_length - 1) // hop_length
        time_step = hop_length / self.sample_rate
        spectrogram = np.zeros((self.num_bands, n_frames))

        # ── Process each band using pre-computed info and stored FFTs ─────────
        for i, info in enumerate(self.band_infos):
            # Get appropriate downsampled FFT (pre-computed above)
            waveform_fft = downsampled_ffts[info.downsample_level]
            n_fft = len(waveform_fft)

            # Create filtered spectrum (initialize to zero)
            filtered_spectrum = np.zeros(n_fft, dtype=complex)

            # Apply weights to positive frequencies using SLICES
            filtered_spectrum[info.bin_start_pos : info.bin_end_pos] = (
                waveform_fft[info.bin_start_pos : info.bin_end_pos]
                # * info.spectrum_weights[info.weight_start_pos : info.weight_end_pos]
                * info.spectrum_weights_pos
            )

            # Apply weights to negative frequencies using SLICES
            # filtered_spectrum[info.bin_start_neg : info.bin_end_neg] = (
            #     waveform_fft[info.bin_start_neg : info.bin_end_neg]
            #     * info.spectrum_weights[info.weight_start_neg : info.weight_end_neg]
            # )

            # IFFT to get filtered signal at downsampled rate
            filtered_signal = fft.ifft(filtered_spectrum)

            # Compute magnitude squared at downsampled rate
            magnitude_squared = filtered_signal.real**2 + filtered_signal.imag**2

            # Determine effective hop length at this downsample level
            # At level L, signal is at rate sr/(2^L)
            # We want output at rate sr/hop_length
            # Effective hop at level L: hop_length/(2^L)
            hop_effective = max(1, hop_length // (2**info.downsample_level))

            # Downsample to final output rate using effective hop
            # magnitude_squared_downsampled = magnitude_squared[::hop_effective]
            magnitude_squared_downsampled = (
                magnitude_squared[: int(len(magnitude_squared) // hop_effective) * hop_effective]
                .reshape(-1, hop_effective)
                .mean(axis=1)
            )

            # Ensure correct length (may differ slightly due to rounding)
            if len(magnitude_squared_downsampled) < n_frames:
                # Pad with zeros
                temp = np.zeros(n_frames)
                temp[: len(magnitude_squared_downsampled)] = magnitude_squared_downsampled
                magnitude_squared_downsampled = temp
            elif len(magnitude_squared_downsampled) > n_frames:
                # Crop to exact length
                magnitude_squared_downsampled = magnitude_squared_downsampled[:n_frames]

            # Convert to cB
            epsilon = 1e-10
            spectrogram[i, :] = np.log10(magnitude_squared_downsampled + epsilon)

        computation_time = time.perf_counter() - start_time

        return spectrogram, time_step, computation_time


@dataclass
class SpectrogramNormalization:
    """Spectrogram dynamic range normalization configuration.

    Combines two complementary approaches:
    1. Psychoacoustic temporal masking (frame-wise)
    2. Global noise floor normalization
    """

    # Frame-wise temporal masking (psychoacoustic)
    # Based on human auditory masking: sounds >20dB below loudest are imperceptible
    enable_temporal_masking: bool = True
    masking_threshold_db: float = 20.0  # 2 Bell (20 dB below frame max/percentile)
    masking_reference: str = "percentile"  # "max" or "percentile" for threshold reference
    masking_percentile: float = 95.0  # Percentile for masking reference (more robust)

    # Global noise floor removal
    floor_db: float = 60.0  # Total dynamic range (6 Bell = 60 dB)
    floor_reference: str = "global_max"  # "global_max", "percentile", "rms"
    percentile: float = 95.0  # If using percentile reference

    # Normalization method
    normalize_method: str = "linear"  # "linear" → [0,1], "standardize" → zero mean/unit std, "none"


def normalize_spectrogram_bell(
    spec_bell: np.ndarray,  # Shape: (freq, time)
    config: SpectrogramNormalization,
) -> np.ndarray:
    """
    Normalize spectrogram using psychoacoustic masking + noise floor removal.

    This combines two complementary approaches:
    1. Frame-wise temporal masking: Removes imperceptible details based on
       human auditory masking (sounds >20dB below loudest are inaudible)
    2. Global floor normalization: Removes absolute noise floor and
       normalizes to consistent dynamic range

    Args:
        spec_bell: Spectrogram in B (log magnitude), shape (freq, time)
        config: SpectrogramNormalization configuration

    Returns:
        Normalized spectrogram in [0, 1] range (or standardized)
    """
    spec_processed = spec_bell.copy()

    # STAGE 1: Global noise floor removal
    # Determine floor reference level
    if config.floor_reference == "global_max":
        signal_level = spec_processed.max()
    elif config.floor_reference == "percentile":
        signal_level = np.percentile(spec_processed, config.percentile)
    elif config.floor_reference == "rms":
        # Use the 95th percentile as a robust signal level estimate
        signal_level = np.percentile(spec_processed, 95)
    else:
        raise ValueError(f"Unknown floor_reference: {config.floor_reference}")

    floor_level = signal_level - config.floor_db / 10

    # Remove noise floor (set floor to 0)
    spec_processed = np.maximum(0, spec_processed - floor_level)

    # STAGE 2: Frame-wise temporal masking (psychoacoustic)
    if config.enable_temporal_masking:
        # Find reference level per time frame (95th percentile for robustness)
        if config.masking_reference == "percentile":
            ref_per_frame = np.percentile(
                spec_processed, config.masking_percentile, axis=0, keepdims=True
            )  # (1, time)
        else:  # "max"
            ref_per_frame = spec_processed.max(axis=0, keepdims=True)  # (1, time)

        # Set masking threshold (20 dB below frame reference)
        masking_threshold = ref_per_frame - config.masking_threshold_db / 10

        # Set values below the threshold to 0
        # This correctly implements psychoacoustic masking
        spec_processed = np.where(
            spec_processed < masking_threshold,
            0,  # Set to floor level
            spec_processed,  # Keep original value
        )

    # STAGE 3: Normalization
    if config.normalize_method == "linear":
        # Map to [0, 1]: 0 dB → 0, floor_db → 1
        spec_norm = spec_processed / (config.floor_db / 10)

    elif config.normalize_method == "standardize":
        # Z-score normalization (mean=0, std=1)
        mean = spec_processed.mean()
        std = spec_processed.std() + 1e-8
        spec_norm = (spec_processed - mean) / std

    elif config.normalize_method == "none":
        spec_norm = spec_processed
    else:
        raise ValueError(f"Unknown normalize_method: {config.normalize_method}")

    return spec_norm
