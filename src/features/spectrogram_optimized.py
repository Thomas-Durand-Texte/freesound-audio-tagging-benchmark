"""Optimized spectrogram computation methods."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.signal import oaconvolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.features.signal_tools import LogSpacedFilterBank

try:
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

    # Compute FFT of waveform once using torch
    n_fft = n_samples
    with torch.no_grad():
        waveform_torch = torch.from_numpy(waveform)
        waveform_fft_torch = torch.fft.fft(waveform_torch, n=n_fft)
        waveform_fft = waveform_fft_torch.numpy()

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

        # IFFT to time domain using torch
        with torch.no_grad():
            filtered_spectrum_torch = torch.from_numpy(filtered_spectrum)
            filtered_signal_torch = torch.fft.ifft(filtered_spectrum_torch)
            filtered_signal = filtered_signal_torch.real.numpy()

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
        # warmup
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=filter_bank.num_bands,
            power=2.0,
        )
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

    # Pre-computed spectrum weights (centered at 0) for positive frequencies only
    spectrum_weights_pos: torch.Tensor


@dataclass
class LevelBatchInfo:
    """Everything needed to process all bands at one downsample level."""

    level: int
    n_fft: int  # FFT length at this level
    band_indices: list[int]  # global band indices
    band_index_tensor: torch.Tensor  # LongTensor for scatter-write, on device
    hop_effective: int  # hop_length // 2**level, clamped >= 1
    # compact W storage
    bin_starts: torch.Tensor  # (B,) int64
    bin_lengths: torch.Tensor  # (B,) int64
    bin_start: int  # min(bin_starts
    weights_compact: torch.Tensor  # (B, Lmax) complex64


# ---------------------------------------------------------------------------
# Compiled inner kernel
# ---------------------------------------------------------------------------


def _make_batch_kernel(
    device_type: str,
) -> Callable[[torch.Tensor, torch.Tensor, int, int, int, int], torch.Tensor]:
    """Return (optionally compiled) per-level batch processing function.

    Kernel contract
    ---------------
    Inputs
        sig_fft       (n_fft,)                  complex64  — FFT of the downsampled signal
        weight_matrix (n_bands, n_fft)           complex64  — pre-built sparse weight rows
        n_samples     int                        — original length at this level (for trim)
        hop_effective int
        n_frames      int                        — target output length

    Output
        (n_bands, n_frames)  float32             — log10(real² + imag²)
    """

    def _kernel(
        sig_fft: torch.Tensor,
        weight_matrix: torch.Tensor,
        bin_start: int,
        n_samples: int,
        hop_effective: int,
        n_frames: int,
    ) -> torch.Tensor:
        # Batched multiply — sig_fft broadcast over band dimension
        # filtered_spectra = sig_fft.unsqueeze(0) * weight_matrix  # (B, n_fft) complex64
        n_bands, n_bins = weight_matrix.shape
        n_fft = sig_fft.shape[0]
        filtered_spectra = torch.zeros(
            (n_bands, n_fft), dtype=torch.complex64, device=sig_fft.device
        )
        bin_end = min(bin_start + n_bins, n_fft)
        filtered_spectra[:, bin_start:bin_end] = (
            sig_fft[bin_start:bin_end].unsqueeze(0) * weight_matrix
        )

        # Batched IFFT → complex analytic signals
        analytic = torch.fft.ifft(filtered_spectra, dim=-1)  # (B, n_fft) complex64

        # Instantaneous power via squared magnitude
        mag2 = analytic.real**2 + analytic.imag**2  # (B, n_fft) float32

        # Trim to exact multiple of hop_effective before pooling
        n_valid = (n_samples // hop_effective) * hop_effective
        mag2_trim = mag2[:, :n_valid]  # (B, n_valid)

        # avg_pool1d expects (N, C, L) — treat bands as channels
        pooled = F.avg_pool1d(
            mag2_trim.unsqueeze(0),  # (1, B, n_valid)
            kernel_size=hop_effective,
            stride=hop_effective,
        ).squeeze(0)  # (B, n_frames_actual)

        # Align to exact n_frames (rounding edge case)
        actual = pooled.shape[1]
        if actual < n_frames:
            pad = torch.zeros(
                pooled.shape[0],
                n_frames,
                dtype=torch.float32,
                device=pooled.device,
            )
            pad[:, :actual] = pooled
            pooled = pad
        elif actual > n_frames:
            pooled = pooled[:, :n_frames]

        return torch.log10(pooled + 1e-10)  # (B, n_frames)

    if device_type == "cuda":
        # dynamic=True: kernel is reusable when n_frames varies slightly
        return torch.compile(_kernel, fullgraph=False, dynamic=True)

    # MPS / CPU — eager is faster than compile for these backends
    return _kernel


class MultiResolutionFilterBank:
    """Multi-resolution filter bank with fully-batched GPU inference.

    All public parameters and BandProcessingInfo fields are identical to the
    original implementation.  The only new init parameter is hop_length, which
    must be fixed at construction time to enable output pre-allocation.
    """

    def __init__(
        self,
        envelope_class: type,
        f_min: float,
        f_max: float,
        num_bands: int,
        sample_rate: int,
        signal_duration: float,
        hop_length: int = 512,
        f_mid: float | None = None,
        spectrum_threshold: float = 0.001,
        device: str | torch.device = "mps",
    ) -> None:
        self.envelope_class = envelope_class
        self.f_min = f_min
        self.f_max = f_max
        self.f_mid = f_mid
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.signal_duration = signal_duration
        self.spectrum_threshold = spectrum_threshold
        self.hop_length = hop_length
        self.n_samples = int(sample_rate * signal_duration)
        self.n_samples = (self.n_samples // 2) * 2
        self.device = torch.device(device) if isinstance(device, str) else device

        # ── Frequency grid ────────────────────────────────────────────────
        if f_mid is not None:
            n_bands_low = num_bands // 2
            n_bands_high = num_bands - n_bands_low
            lim_low = np.geomspace(f_min, f_mid, n_bands_low + 1)
            lim_high = np.geomspace(f_mid, f_max, n_bands_high + 1)
            frequency_limits = np.concatenate([lim_low[:-1], lim_high])
        else:
            frequency_limits = np.geomspace(f_min, f_max, num_bands + 1)

        self.center_frequencies = (frequency_limits[:-1] + frequency_limits[1:]) / 2
        self.bandwidths = frequency_limits[1:] - frequency_limits[:-1]

        # ── Per-band info ───────────────
        self.band_infos: list[BandProcessingInfo] = [
            self._precompute_band_processing(i) for i in range(num_bands)
        ]
        self.downsample_levels = sorted({info.downsample_level for info in self.band_infos})

        # ── Batch structures — one per downsample level ───────────────────
        self._n_frames: int = (self.n_samples + hop_length - 1) // hop_length
        self._level_batches: list[LevelBatchInfo] = self._build_level_batches()

        # ── Pre-allocated output buffer ───────────────────────────────────
        # Reused across every call that uses the init hop_length.
        self._output: torch.Tensor = torch.zeros(
            (num_bands, self._n_frames),
            dtype=torch.float32,
            device=self.device,
        )

        # ── Compiled inner kernel ─────────────────────────────────────────
        self._kernel = _make_batch_kernel(self.device.type)

        # ── pre-allocate tensors ────────────────────────
        self._signal_buf = torch.zeros(self.n_samples, dtype=torch.float32, device=self.device)

        self.n_ffts_downsampled: list[int] = [self.n_samples]
        for level in self.downsample_levels[1:]:
            self.n_ffts_downsampled.append(self.n_ffts_downsampled[level - 1] // 2)

        self.fft_downsampled_outputs: list[torch.Tensor] = [
            torch.empty(n, dtype=torch.complex64, device=self.device)
            for n in self.n_ffts_downsampled
        ]

    # ── Init helpers ──────────────────────────────────────────────────────

    def _determine_downsample_level(self, fc: float, bw: float) -> int:
        """Identical to original."""
        fmax_band = fc + bw / 2
        level = 10
        while level:
            sr_ds = self.sample_rate / (2**level)
            if fmax_band <= sr_ds / 6:
                break
            level -= 1
        return max(level, 0)

    def _precompute_band_processing(self, band_idx: int) -> BandProcessingInfo:
        """Identical to original — produces per-band weight slices on device."""
        fc = self.center_frequencies[band_idx]
        bw = self.bandwidths[band_idx]

        level = self._determine_downsample_level(fc, bw)
        sr_effective = self.sample_rate / (2**level)
        n_effective = self.n_samples // (2**level)

        envelope = self.envelope_class(bandwidth=bw)
        df = sr_effective / n_effective
        n_fft = n_effective

        if hasattr(envelope, "alpha"):  # SuperGaussian
            alpha = envelope.alpha
            f_max_band = (np.log(1.0 / self.spectrum_threshold) / alpha) ** 0.25
        else:  # Gaussian or other
            f_max_band = 3.0 * bw

        n_bins = int(f_max_band / df + 0.5)
        freq_rel = np.linspace(-n_bins * df, n_bins * df, 2 * n_bins + 1)
        spectrum_vals = envelope.spectrum(freq_rel)

        bin_center_pos = int(fc / df + 0.5)
        bin_start_pos = bin_center_pos - n_bins
        bin_end_pos = bin_center_pos + n_bins + 1

        weight_start_pos = max(0, -bin_start_pos) if bin_start_pos < 0 else 0
        weight_end_pos = (
            len(spectrum_vals) - max(0, bin_end_pos - n_fft // 2)
            if bin_end_pos > n_fft // 2
            else len(spectrum_vals)
        )

        bin_start_clipped = max(0, bin_start_pos)
        bin_end_clipped = min(n_fft // 2, bin_end_pos)

        weights_torch = torch.from_numpy(spectrum_vals[weight_start_pos:weight_end_pos]).to(
            dtype=torch.complex64, device=self.device
        )

        return BandProcessingInfo(
            downsample_level=level,
            sample_rate_effective=sr_effective,
            n_samples_effective=n_effective,
            bin_start_pos=bin_start_clipped,
            bin_end_pos=bin_end_clipped,
            weight_start_pos=weight_start_pos,
            weight_end_pos=weight_end_pos,
            spectrum_weights_pos=weights_torch,
        )

    def _build_level_batches(self) -> list[LevelBatchInfo]:
        """Build one LevelBatchInfo per downsample level using compact W storage."""

        groups: dict[int, list[int]] = defaultdict(list)
        for i, info in enumerate(self.band_infos):
            groups[info.downsample_level].append(i)

        batches: list[LevelBatchInfo] = []
        for level in sorted(groups.keys()):
            band_indices = groups[level]
            n_eff = self.n_samples // (2**level)
            hop_eff = max(1, self.hop_length // (2**level))

            # --- compact conversion instead of dense W ---
            starts = []
            ends = []
            lengths = []
            slices = []
            for global_i in band_indices:
                info = self.band_infos[global_i]
                w = info.spectrum_weights_pos
                starts.append(info.bin_start_pos)
                lengths.append(w.numel())
                ends.append(int(info.bin_start_pos + w.numel()))
                slices.append(w)

            min_start = min(starts)
            max_end = max(ends)
            w_c = torch.zeros(
                (len(band_indices), max_end - min_start),
                dtype=torch.complex64,
                device=self.device,
            )
            for row, (start, w) in enumerate(zip(starts, slices, strict=True)):
                i0 = start - min_start
                w_c[row, i0 : i0 + w.numel()] = w

            # LongTensor of global indices for scatter-write into _output
            idx_tensor = torch.tensor(band_indices, dtype=torch.long, device=self.device)

            batches.append(
                LevelBatchInfo(
                    level=level,
                    n_fft=n_eff,
                    band_indices=band_indices,
                    band_index_tensor=idx_tensor,
                    bin_starts=torch.tensor(starts, dtype=torch.long, device=self.device),
                    bin_lengths=torch.tensor(lengths, dtype=torch.long, device=self.device),
                    bin_start=min_start,
                    weights_compact=w_c,
                    hop_effective=hop_eff,
                )
            )

        return batches

    # ── Signal preparation ────────────────────────────────────────────────

    def _prepare_signal(self, waveform: torch.Tensor) -> torch.Tensor:
        self._signal_buf.zero_()
        n_copy = min(len(waveform), self.n_samples)
        self._signal_buf[:n_copy].copy_(waveform[:n_copy])
        return self._signal_buf

    # ── Inference ─────────────────────────────────────────────────────────

    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        hop_length: int | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute multi-resolution spectrogram.

        Parameters
        ----------
        waveform:
            1-D audio tensor on any device.  Moved to the filter bank's device
            if necessary.
        hop_length:
            Frames stride.  Omit (or pass the same value as at init) to reuse
            the pre-allocated output buffer.  Passing a different value is
            supported but allocates a fresh output tensor for that call.

        Returns
        -------
        spectrogram:  (num_bands, n_frames) float32  — log10(instantaneous power)
        time_step:    seconds between output frames
        computation_time: wall-clock seconds
        """
        start_time = time.perf_counter()

        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        use_prealloc = (hop_length is None) or (hop_length == self.hop_length)
        effective_hop = self.hop_length if hop_length is None else hop_length

        if use_prealloc:
            out = self._output
            n_frames = self._n_frames
        else:
            n_frames = (self.n_samples + effective_hop - 1) // effective_hop
            out = torch.zeros(
                (self.num_bands, n_frames),
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            # 1. Crop / zero-pad to fixed length
            signal = self._prepare_signal(waveform)

            # 2. One FFT per level
            ffts = self.fft_downsampled_outputs
            torch.fft.fft(signal, out=ffts[0])
            downsampled = signal
            for level in self.downsample_levels[1:]:
                n_even = self.n_ffts_downsampled[level] * 2
                downsampled = downsampled[:n_even].reshape(-1, 2).mean(dim=1)
                torch.fft.fft(downsampled, out=ffts[level])

            # 3. Per-level: batched multiply → batched IFFT → pool → log10
            for batch in self._level_batches:
                hop_eff = (
                    batch.hop_effective
                    if use_prealloc
                    else max(1, effective_hop // (2**batch.level))
                )

                result = self._kernel(  # (n_bands_at_level, n_frames)
                    ffts[batch.level],
                    batch.weights_compact,
                    batch.bin_start,
                    batch.n_fft,
                    hop_eff,
                    n_frames,
                )

                # Scatter into global output — pure tensor indexing, no Python loop
                out[batch.band_index_tensor] = result

        time_step = effective_hop / self.sample_rate
        computation_time = time.perf_counter() - start_time
        return out, time_step, computation_time


class MultiResolutionFilterBank0:
    """Multi-resolution filter bank with pre-computed downsampling and frequency weights.

    Optimizations:
    - All weights pre-computed for fixed signal duration
    - Adaptive downsampling: high frequencies on downsampled signal
    - Downsampled signals computed once at inference start
    - Frequency domain filtering with pre-computed slices (no index arrays)
    - All indices computed analytically (no boolean masking)
    - Pure PyTorch forward pass (no NumPy operations)
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
        device: str | torch.device = "mps",
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
            device: PyTorch device ('mps', 'cuda', or 'cpu')
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
        self.device = torch.device(device) if isinstance(device, str) else device

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


        spectrum_weights_pos_torch = torch.from_numpy(
            spectrum_vals[weight_start_pos:weight_end_pos]
        ).to(dtype=torch.complex64, device=self.device)

        return BandProcessingInfo(
            downsample_level=level,
            sample_rate_effective=sr_effective,
            n_samples_effective=n_effective,
            bin_start_pos=bin_start_pos_clipped,
            bin_end_pos=bin_end_pos_clipped,
            weight_start_pos=weight_start_pos,
            weight_end_pos=weight_end_pos,
            spectrum_weights_pos=spectrum_weights_pos_torch,
        )

    def _prepare_signal(self, waveform: torch.Tensor) -> torch.Tensor:
        """Crop or zero-pad signal to fixed length.

        Args:
            waveform: Input waveform (torch tensor on device)

        Returns:
            Waveform with exactly self.n_samples length (torch tensor on device)
        """
        if len(waveform) >= self.n_samples:
            return waveform[: self.n_samples]
        else:
            padded = torch.zeros(self.n_samples, dtype=waveform.dtype, device=waveform.device)
            padded[: len(waveform)] = waveform
            return padded

    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        hop_length: int = 512,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute multi-resolution spectrogram using PyTorch on device.

        Args:
            waveform: Audio waveform tensor (will be moved to filter bank's device if needed)
            hop_length: Downsampling factor for output

        Returns:
            Tuple of (spectrogram, time_step, computation_time)
            - spectrogram: 2D array (n_bands, n_frames) in cB
            - time_step: Time between frames in seconds
            - computation_time: Computation time in seconds
        """
        start_time = time.perf_counter()

        # Move input to filter bank's device if needed
        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        # Prepare signal (crop or pad to fixed length) - stays on device
        signal = self._prepare_signal(waveform)

        # ── Pre-compute ALL downsampled versions and their FFTs ONCE (on device) ─
        downsampled_signals = {0: signal}

        # Downsample signals first (all levels)
        for level in self.downsample_levels:
            if level == 0:
                continue

            # Start from previous level (torch tensor)
            signal_ds = downsampled_signals[level - 1]

            # Downsample once more (all torch operations, stay on device)
            n_even = (len(signal_ds) // 2) * 2
            signal_ds = signal_ds[:n_even].reshape(-1, 2).mean(dim=1)

            downsampled_signals[level] = signal_ds

        # Compute all FFTs in one batch to avoid buffer reuse warnings
        with torch.no_grad():
            downsampled_ffts = {
                level: torch.fft.fft(
                    downsampled_signals[level],
                    out=torch.empty(
                        downsampled_signals[level].shape[0], dtype=torch.complex64, device="mps"
                    ),
                )
                for level in self.downsample_levels
            }

        # Initialize output
        n_frames = (self.n_samples + hop_length - 1) // hop_length
        time_step = hop_length / self.sample_rate
        spectrogram = torch.zeros((self.num_bands, n_frames), device=self.device)

        # ── Process each band using pre-computed info and stored FFTs (all on device) ─
        for i, info in enumerate(self.band_infos):
            with torch.no_grad():
                # Get appropriate downsampled FFT (already on device)
                waveform_fft = downsampled_ffts[info.downsample_level]
                n_fft = len(waveform_fft)

                # Create filtered spectrum (on device)
                filtered_spectrum = torch.zeros(n_fft, dtype=torch.complex64, device=self.device)

                # Apply weights to positive frequencies (all on device)
                filtered_spectrum[info.bin_start_pos : info.bin_end_pos] = (
                    waveform_fft[info.bin_start_pos : info.bin_end_pos] * info.spectrum_weights_pos
                )

                # IFFT to get filtered signal (on device)
                filtered_signal = torch.fft.ifft(filtered_spectrum)

                # Compute magnitude squared (on device)
                magnitude_squared = filtered_signal.real**2 + filtered_signal.imag**2

            # Determine effective hop length at this downsample level
            hop_effective = max(1, hop_length // (2**info.downsample_level))

            # Downsample to final output rate
            magnitude_squared_downsampled = (
                magnitude_squared[: int(len(magnitude_squared) // hop_effective) * hop_effective]
                .reshape(-1, hop_effective)
                .mean(axis=1)
            )

            # Ensure correct length (may differ slightly due to rounding)
            if len(magnitude_squared_downsampled) < n_frames:
                temp = torch.zeros(n_frames, dtype=torch.float32, device=self.device)
                temp[: len(magnitude_squared_downsampled)] = magnitude_squared_downsampled
                magnitude_squared_downsampled = temp
            elif len(magnitude_squared_downsampled) > n_frames:
                magnitude_squared_downsampled = magnitude_squared_downsampled[:n_frames]

            # Convert to cB
            epsilon = 1e-10
            spectrogram[i, :] = torch.log10(magnitude_squared_downsampled + epsilon)

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


def _normalize_numpy(
    spec_bell: np.ndarray,
    config: SpectrogramNormalization,
) -> np.ndarray:
    """Normalize NumPy spectrogram using psychoacoustic masking + noise floor removal.

    Args:
        spec_bell: Spectrogram in B (log magnitude), shape (freq, time)
        config: SpectrogramNormalization configuration

    Returns:
        Normalized spectrogram in [0, 1] range (or standardized)
    """
    spec_processed = spec_bell.copy()

    # STAGE 1: Global noise floor removal
    if config.floor_reference == "global_max":
        signal_level = spec_processed.max()
    elif config.floor_reference == "percentile":
        signal_level = np.percentile(spec_processed, config.percentile)
    elif config.floor_reference == "rms":
        signal_level = np.percentile(spec_processed, 95)
    else:
        raise ValueError(f"Unknown floor_reference: {config.floor_reference}")

    floor_level = signal_level - config.floor_db / 10
    spec_processed = np.maximum(0, spec_processed - floor_level)

    # STAGE 2: Frame-wise temporal masking (psychoacoustic)
    if config.enable_temporal_masking:
        if config.masking_reference == "percentile":
            ref_per_frame = np.percentile(
                spec_processed, config.masking_percentile, axis=0, keepdims=True
            )
        else:  # "max"
            ref_per_frame = spec_processed.max(axis=0, keepdims=True)

        masking_threshold = ref_per_frame - config.masking_threshold_db / 10
        spec_processed = np.where(
            spec_processed < masking_threshold,
            0,
            spec_processed,
        )

    # STAGE 3: Normalization
    if config.normalize_method == "linear":
        spec_norm = spec_processed / (config.floor_db / 10)
    elif config.normalize_method == "standardize":
        mean = spec_processed.mean()
        std = spec_processed.std() + 1e-8
        spec_norm = (spec_processed - mean) / std
    elif config.normalize_method == "none":
        spec_norm = spec_processed
    else:
        raise ValueError(f"Unknown normalize_method: {config.normalize_method}")

    return spec_norm


def _normalize_torch(
    spec_bell: torch.Tensor,
    config: SpectrogramNormalization,
) -> torch.Tensor:
    """Normalize torch spectrogram using psychoacoustic masking + noise floor removal.

    Args:
        spec_bell: Spectrogram in B (log magnitude), shape (freq, time)
        config: SpectrogramNormalization configuration

    Returns:
        Normalized spectrogram in [0, 1] range (or standardized)
    """
    spec_processed = spec_bell.clone()

    # STAGE 1: Global noise floor removal
    if config.floor_reference == "global_max":
        signal_level = spec_processed.max().item()
    elif config.floor_reference == "percentile":
        signal_level = torch.quantile(spec_processed.flatten(), config.percentile / 100).item()
    elif config.floor_reference == "rms":
        signal_level = torch.quantile(spec_processed.flatten(), 0.95).item()
    else:
        raise ValueError(f"Unknown floor_reference: {config.floor_reference}")

    floor_level = signal_level - config.floor_db / 10
    spec_processed = torch.maximum(torch.zeros_like(spec_processed), spec_processed - floor_level)

    # STAGE 2: Frame-wise temporal masking (psychoacoustic)
    if config.enable_temporal_masking:
        if config.masking_reference == "percentile":
            ref_per_frame = torch.quantile(
                spec_processed, config.masking_percentile / 100, dim=0, keepdim=True
            )
        else:  # "max"
            ref_per_frame = spec_processed.max(dim=0, keepdim=True)[0]

        masking_threshold = ref_per_frame - config.masking_threshold_db / 10
        spec_processed = torch.where(
            spec_processed < masking_threshold,
            torch.zeros_like(spec_processed),
            spec_processed,
        )

    # STAGE 3: Normalization
    if config.normalize_method == "linear":
        spec_norm = spec_processed / (config.floor_db / 10)
    elif config.normalize_method == "standardize":
        mean = spec_processed.mean()
        std = spec_processed.std() + 1e-8
        spec_norm = (spec_processed - mean) / std
    elif config.normalize_method == "none":
        spec_norm = spec_processed
    else:
        raise ValueError(f"Unknown normalize_method: {config.normalize_method}")

    return spec_norm


def normalize_spectrogram_bell(
    spec_bell: np.ndarray | torch.Tensor,
    config: SpectrogramNormalization,
) -> np.ndarray | torch.Tensor:
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
    if isinstance(spec_bell, torch.Tensor):
        return _normalize_torch(spec_bell, config)
    else:
        return _normalize_numpy(spec_bell, config)
