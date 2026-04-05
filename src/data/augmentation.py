"""Audio data augmentation for training.

This module provides waveform-level and spectrogram-level augmentation functions
for robust audio classification model training.
"""

from collections.abc import Callable

import numpy as np
import torch
import torchaudio.transforms as T  # noqa: N812
from scipy import signal

# =============================================================================
# Waveform Augmentation
# =============================================================================


def time_stretch(
    waveform: torch.Tensor,
    sample_rate: int,
    stretch_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Time-stretch audio without changing pitch.

    Args:
        waveform: Audio (samples,)
        sample_rate: Sampling rate
        stretch_range: (min, max) speed factors (0.9 = slower, 1.1 = faster)
        rng: Random number generator (if None, uses default)

    Returns:
        Time-stretched waveform (length may vary)
    """
    if rng is None:
        rng = np.random.default_rng()
    stretch_factor = rng.uniform(*stretch_range)

    # Need to work in spectrogram domain
    n_fft = 1024
    spec = T.Spectrogram(n_fft=n_fft, power=None)(waveform.unsqueeze(0))

    # Phase vocoder: stretches time without changing pitch
    stretch = T.TimeStretch(n_freq=n_fft // 2 + 1)
    stretched_spec = stretch(spec, stretch_factor)

    # Invert back to waveform
    griffin_lim = T.GriffinLim(n_fft=n_fft)
    return griffin_lim(stretched_spec.abs().squeeze(0))


def pitch_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    semitone_range: tuple[int, int] = (-2, 2),
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Shift pitch by random semitones.

    Args:
        waveform: Audio (samples,)
        sample_rate: Sampling rate
        semitone_range: (min, max) semitones to shift (inclusive)
        rng: Random number generator (if None, uses default)

    Returns:
        Pitch-shifted waveform (same length as input)
    """
    if rng is None:
        rng = np.random.default_rng()
    # rng.integers requires high > low, so we add 1 to make it inclusive on both ends
    n_steps = rng.integers(semitone_range[0], semitone_range[1] + 1)

    shifter = T.PitchShift(sample_rate, n_steps)
    return shifter(waveform)


def add_noise(
    waveform: torch.Tensor,
    snr_db_range: tuple[float, float] = (30.0, 40.0),
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise at random SNR.

    Args:
        waveform: Audio (samples,)
        snr_db_range: (min, max) signal-to-noise ratio in dB
        rng: Random number generator (if None, uses default)

    Returns:
        Noisy waveform
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_db = rng.uniform(*snr_db_range)

    # Signal power
    signal_power = waveform.norm(p=2) ** 2 / waveform.numel()

    # Noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

    return waveform + noise


def _torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation on GPU (PyTorch equivalent of np.interp).

    Args:
        x: Points at which to evaluate interpolated values
        xp: x-coordinates of data points (must be increasing)
        fp: y-coordinates of data points

    Returns:
        Interpolated values at x
    """
    # Find indices where x values fall in xp
    indices = torch.searchsorted(xp, x, right=False)

    # Clamp indices to valid range
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get surrounding points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    slope = (y1 - y0) / (x1 - x0)
    return y0 + (x - x0) * slope


def frequency_response_perturbation(
    waveform: torch.Tensor,
    sample_rate: int,
    n_bands: int = 8,
    gain_std_db: float = 3.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Apply random frequency-dependent gain.

    Args:
        waveform: Audio (samples,) on any device
        sample_rate: Sampling rate
        n_bands: Number of frequency bands
        gain_std_db: Standard deviation of gain in dB
        rng: Random number generator (if None, uses default)

    Returns:
        Filtered waveform (same device as input)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random gains at log-spaced frequencies (NumPy RNG only)
    freqs_np = np.logspace(np.log10(20), np.log10(sample_rate / 2), n_bands)
    gains_db_np = rng.standard_normal(n_bands) * gain_std_db

    # Convert to torch tensors on same device as waveform (convert ONCE)
    device = waveform.device
    freqs = torch.from_numpy(freqs_np).to(device=device, dtype=torch.float32)
    gains_db = torch.from_numpy(gains_db_np).to(device=device, dtype=torch.float32)

    # FFT using torch (all operations on device)
    with torch.no_grad():
        freq_bins = torch.fft.rfftfreq(waveform.shape[0], 1 / sample_rate, device=device)
        waveform_fft = torch.fft.rfft(waveform, out=torch.empty_like(freq_bins, dtype=torch.complex64))

        # Interpolate gains to all frequency bins using PyTorch
        gains_db_interp = _torch_interp(freq_bins, freqs, gains_db)

        # Convert dB to linear gain
        gains_linear = torch.pow(10.0, gains_db_interp / 20.0).to(dtype=torch.complex64)

        # Apply filter
        fft_filtered = waveform_fft * gains_linear
        waveform_filtered = torch.fft.irfft(fft_filtered, n=waveform.shape[0], out=torch.empty_like(waveform))

    return waveform_filtered.float()


class WaveformAugmentation:
    """Combined waveform augmentation pipeline."""

    def __init__(
        self,
        sample_rate: int,
        time_stretch: bool = True,
        pitch_shift: bool = True,
        add_noise: bool = True,
        freq_perturbation: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.add_noise = add_noise
        self.freq_perturbation = freq_perturbation

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentations sequentially."""
        norm = waveform.norm(p=2)
        if self.time_stretch:
            waveform = time_stretch(waveform, self.sample_rate)
        if self.pitch_shift:
            waveform = pitch_shift(waveform, self.sample_rate)
        if self.add_noise:
            waveform = add_noise(waveform)
        if self.freq_perturbation:
            waveform = frequency_response_perturbation(waveform, self.sample_rate)
        return waveform * (norm / waveform.norm(p=2))


# =============================================================================
# Reverberation Augmentation
# =============================================================================


def _generate_room_impulse_response(
    sample_rate: int,
    rt60_range: tuple[float, float] = (0.2, 1.5),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate room impulse response using exponential decay model.

    Args:
        sample_rate: Sampling rate in Hz
        rt60_range: (min, max) reverberation time in seconds
        rng: Random number generator

    Returns:
        Impulse response [ir_samples] normalized
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random RT60
    rt60 = rng.uniform(*rt60_range)

    # IR length
    ir_length = int(rt60 * sample_rate + 0.5)

    # Random reflections (use abs to avoid phase cancellation)
    noise = np.abs(rng.standard_normal(ir_length))

    # Exponential decay
    decay = np.exp(np.log(0.001) * np.linspace(0, 1, ir_length))

    # Combine and normalize
    ir = noise * decay
    ir = ir / np.sqrt(np.sum(ir**2))

    return ir


def apply_reverb(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    rt60_range: tuple[float, float] = (0.2, 1.5),
    dry_wet_mix: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random reverberation to audio waveform.

    Uses exponential decay model for fast, realistic reverb simulation.

    Args:
        waveform: Input audio [samples]
        sample_rate: Sampling rate in Hz
        rt60_range: (min, max) reverberation time in seconds
        dry_wet_mix: Mix ratio (0=dry, 1=wet)
        rng: Random number generator

    Returns:
        Reverberated waveform [samples]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate IR
    ir = _generate_room_impulse_response(
        sample_rate=sample_rate,
        rt60_range=rt60_range,
        rng=rng,
    )

    # Convolve
    wet_signal = signal.fftconvolve(waveform, ir, mode="full")[: len(waveform)]

    # Mix and normalize
    output = (1 - dry_wet_mix) * waveform + dry_wet_mix * wet_signal
    return output.astype(waveform.dtype) * (waveform.std() / output.std())


# =============================================================================
# SpecAugment (Time and Frequency Masking)
# =============================================================================


def spec_augment(
    spec: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Apply SpecAugment to spectrogram.

    SpecAugment randomly masks rectangular patches in the time-frequency plane,
    teaching the model to be robust to missing information. Simple yet highly
    effective for audio classification.

    Args:
        spec: Spectrogram (freq, time) or (batch, 1, freq, time)
        freq_mask_param: Maximum frequency bins to mask
        time_mask_param: Maximum time frames to mask
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        rng: Random number generator (if None, uses default)

    Returns:
        Augmented spectrogram (same shape as input)
    """
    if rng is None:
        rng = np.random.default_rng()

    spec = spec.clone()

    # Handle both (F, T) and (B, 1, F, T)
    if spec.dim() == 4:
        batch_size = spec.shape[0]
        # Apply to each sample in batch
        for i in range(batch_size):
            spec[i, 0] = _apply_spec_augment_single(
                spec[i, 0], freq_mask_param, time_mask_param, num_freq_masks, num_time_masks, rng
            )
    else:
        spec = _apply_spec_augment_single(
            spec, freq_mask_param, time_mask_param, num_freq_masks, num_time_masks, rng
        )

    return spec


def _apply_spec_augment_single(
    spec: torch.Tensor,
    freq_mask_param: int,
    time_mask_param: int,
    num_freq_masks: int,
    num_time_masks: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply SpecAugment to single spectrogram.

    Args:
        spec: Spectrogram tensor (freq, time)
        freq_mask_param: Maximum frequency bins to mask
        time_mask_param: Maximum time frames to mask
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        rng: Random number generator

    Returns:
        Augmented spectrogram (same shape as input)
    """
    freq_bins, time_frames = spec.shape

    # Frequency masking (horizontal stripes)
    for _ in range(num_freq_masks):
        f = rng.integers(0, freq_mask_param + 1)
        f0 = rng.integers(0, max(1, freq_bins - f))
        spec[f0 : f0 + f, :] = 0

    # Time masking (vertical stripes)
    for _ in range(num_time_masks):
        t = rng.integers(0, time_mask_param + 1)
        t0 = rng.integers(0, max(1, time_frames - t))
        spec[:, t0 : t0 + t] = 0

    return spec


# =============================================================================
# Mixup Augmentation
# =============================================================================


def mixup(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup augmentation to batch.

    Args:
        batch_x: Batch of spectrograms (B, 1, F, T)
        batch_y: Batch of labels (B, num_classes), multi-hot encoded
        alpha: Beta distribution parameter (default: 0.4)
        rng: Random number generator (if None, uses default)

    Returns:
        Mixed spectrograms and labels (same shapes as input)
    """
    if alpha <= 0:
        return batch_x, batch_y

    if rng is None:
        rng = np.random.default_rng()

    batch_size = batch_x.size(0)

    # Sample mixing coefficient from Beta(alpha, alpha)
    lam = rng.beta(alpha, alpha)

    # Random permutation of batch indices
    index = torch.randperm(batch_size, device=batch_x.device)

    # Mix spectrograms and labels
    mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
    mixed_y = lam * batch_y + (1 - lam) * batch_y[index]

    return mixed_x, mixed_y


def mixup_collate_fn(alpha: float = 0.4) -> Callable:
    """Create collate function with Mixup augmentation.

    Use with DataLoader:
        loader = DataLoader(dataset, collate_fn=mixup_collate_fn(alpha=0.4))

    Args:
        alpha: Beta distribution parameter

    Returns:
        Collate function for DataLoader
    """

    def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        # Default collate
        spectrograms = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        # Apply mixup
        spectrograms, labels = mixup(spectrograms, labels, alpha)

        return spectrograms, labels

    return collate_fn
