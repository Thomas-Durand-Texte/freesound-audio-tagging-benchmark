"""Audio data augmentation for training.

This module provides waveform-level and spectrogram-level augmentation functions
for robust audio classification model training.
"""

from collections import deque
from math import log, sqrt

import numpy as np

# =============================================================================
# Reverberation Augmentation (Image Source Method)
# =============================================================================

# Speed of sound in air (m/s)
_SPEED_OF_SOUND = 343.0


def _compute_reverberation_time(
    reflection_coefficient: float, volume: float, surface_area: float
) -> float:
    """Compute T60 reverberation time using Sabine's formula.

    Args:
        reflection_coefficient: Wall reflection coefficient (0-1)
        volume: Room volume in m³
        surface_area: Total room surface area in m²

    Returns:
        T60 reverberation time in seconds
    """
    return -0.16 * volume / (surface_area * log(reflection_coefficient))


class _Point3D:
    """3D point in Cartesian coordinates."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: "_Point3D") -> float:
        """Compute Euclidean distance to another point."""
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


class _Source(_Point3D):
    """Sound source with position and amplitude."""

    __slots__ = "amplitude"

    def __init__(self, x: float, y: float, z: float, amplitude: float):
        super().__init__(x, y, z)
        self.amplitude = amplitude


class _RoomGeometry:
    """Room geometry parameters."""

    __slots__ = ("height", "length", "width")

    def __init__(self, length: float, width: float, height: float):
        self.length = length
        self.width = width
        self.height = height

    @property
    def volume(self) -> float:
        """Compute room volume in m³."""
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        """Compute total surface area in m²."""
        return 2 * (self.length * self.width + self.length * self.height + self.width * self.height)

    def contains_point(self, point: _Point3D) -> bool:
        """Check if a point is inside the room."""
        return (
            0 <= point.x <= self.length
            and 0 <= point.y <= self.width
            and 0 <= point.z <= self.height
        )


def _generate_room_impulse_response(
    sample_rate: int,
    room_size_range: tuple[float, float],
    height_range: tuple[float, float],
    reflection_coeff_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a room impulse response using the image source method.

    The image source method models room reflections by placing virtual "mirror"
    sources behind each wall. Each virtual source contributes to the impulse
    response based on its distance from the microphone and the wall reflection
    coefficient.

    Args:
        sample_rate: Sampling rate in Hz
        room_size_range: (min, max) room dimensions for length and width in meters
        height_range: (min, max) room height in meters
        reflection_coeff_range: (min, max) wall reflection coefficient (0.5-0.95)
        rng: Random number generator

    Returns:
        Impulse response as 1D numpy array (normalized to [-1, 1])
    """
    # 1. Generate random room geometry
    length = rng.uniform(*room_size_range)
    width = rng.uniform(*room_size_range)
    height = rng.uniform(*height_range)
    room = _RoomGeometry(length, width, height)

    # 2. Generate random reflection coefficient
    reflection_coeff = rng.uniform(*reflection_coeff_range)

    # 3. Generate random source position
    margin = 0.1 * room.length
    source_pos = _Point3D(
        rng.uniform(margin, length - margin),
        rng.uniform(margin, width - margin),
        rng.uniform(margin, height - margin),
    )
    source = _Source(source_pos.x, source_pos.y, source_pos.z, amplitude=1.0)

    # 4. Generate random microphone position
    mic_pos = _Point3D(
        rng.uniform(margin, length - margin),
        rng.uniform(margin, width - margin),
        rng.uniform(margin, height - margin),
    )

    # 5. Compute reverberation time (T60)
    rt60 = _compute_reverberation_time(reflection_coeff, room.volume, room.surface_area)

    # 6. Initialize impulse response
    ir_length = int(rt60 * sample_rate + 0.5)
    impulse_response = np.zeros(ir_length, dtype=np.float32)

    # 7. Image source method: process virtual sources
    virtual_sources: deque[_Source] = deque([source])
    max_delay_samples = ir_length

    while virtual_sources:
        current_source = virtual_sources.popleft()

        # Compute distance and delay
        distance = current_source.distance(mic_pos)
        delay_samples = int(distance / _SPEED_OF_SOUND * sample_rate + 0.5)

        # Skip if delay exceeds impulse response length
        if delay_samples >= max_delay_samples:
            continue

        # Add contribution to impulse response (1/r amplitude decay)
        amplitude = current_source.amplitude / distance
        impulse_response[delay_samples] += amplitude

        # Generate 6 mirror sources (reflections from each wall)
        # Only if amplitude is significant enough to matter
        if abs(current_source.amplitude * reflection_coeff) > 1e-6:
            new_amplitude = current_source.amplitude * reflection_coeff
            # Mirror across x=0 wall
            if current_source.x > 0:
                mirror_x_neg = _Source(
                    -current_source.x, current_source.y, current_source.z, new_amplitude
                )
                virtual_sources.append(mirror_x_neg)

            # Mirror across x=length wall
            if current_source.x < room.length:
                mirror_x_pos = _Source(
                    2 * room.length - current_source.x,
                    current_source.y,
                    current_source.z,
                    new_amplitude,
                )
                virtual_sources.append(mirror_x_pos)

            # Mirror across y=0 wall
            if current_source.y > 0:
                mirror_y_neg = _Source(
                    current_source.x, -current_source.y, current_source.z, new_amplitude
                )
                virtual_sources.append(mirror_y_neg)

            # Mirror across y=width wall
            if current_source.y < room.width:
                mirror_y_pos = _Source(
                    current_source.x,
                    2 * room.width - current_source.y,
                    current_source.z,
                    new_amplitude,
                )
                virtual_sources.append(mirror_y_pos)

            # Mirror across z=0 wall (floor)
            if current_source.z > 0:
                mirror_z_neg = _Source(
                    current_source.x, current_source.y, -current_source.z, new_amplitude
                )
                virtual_sources.append(mirror_z_neg)

            # Mirror across z=height wall (ceiling)
            if current_source.z < room.height:
                mirror_z_pos = _Source(
                    current_source.x,
                    current_source.y,
                    2 * room.height - current_source.z,
                    new_amplitude,
                )
                virtual_sources.append(mirror_z_pos)

    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(impulse_response))
    if max_amplitude > 0:
        impulse_response /= max_amplitude

    return impulse_response


def apply_reverb(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    room_size_range: tuple[float, float] = (2.0, 20.0),
    height_range: tuple[float, float] = (2.0, 15.0),
    reflection_coeff_range: tuple[float, float] = (0.5, 0.95),
    dry_wet_mix: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random reverberation to an audio waveform.

    Uses the image source method to generate realistic room impulse responses.
    This augmentation simulates the acoustic properties of various room sizes
    and materials.

    Args:
        waveform: Input audio waveform (1D array)
        sample_rate: Sampling rate in Hz
        room_size_range: (min, max) room dimensions in meters (2-20m recommended)
        height_range: (min, max) room height in meters (2-15m recommended)
        reflection_coeff_range: (min, max) wall reflection coefficient (0.5-0.95)
        dry_wet_mix: Mix ratio (0.0 = dry only, 1.0 = wet only)
        rng: Random number generator (if None, uses default)

    Returns:
        Reverberated waveform (same length as input)

    Note:
        This is computationally expensive due to convolution. Use sparingly
        during training or consider pre-generating impulse responses.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random room impulse response
    ir = _generate_room_impulse_response(
        sample_rate=sample_rate,
        room_size_range=room_size_range,
        height_range=height_range,
        reflection_coeff_range=reflection_coeff_range,
        rng=rng,
    )

    # Convolve waveform with impulse response
    wet_signal = np.convolve(waveform, ir, mode="full")[: len(waveform)]

    # Mix dry and wet signals
    output = (1 - dry_wet_mix) * waveform + dry_wet_mix * wet_signal

    return output.astype(waveform.dtype)
