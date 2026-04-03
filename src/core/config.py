"""Configuration management using dataclasses.

This module provides type-safe configuration management with IDE autocomplete,
validation, and clean dot notation access.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Project metadata and global settings."""

    name: str = "acoustic-event-detection-benchmark"
    seed: int = 42


@dataclass
class DataConfig:
    """Data paths and loading settings."""

    base_dir: Path
    base_folder_name: str = "FSDKaggle2019."
    train_dir: str = "audio_train_curated"
    train_noisy_dir: str = "audio_train_noisy"
    val_dir: str = "meta"
    test_dir: str = "audio_test"
    sample_rate: int = 44100
    clip_duration: float = 5.0
    problematic_files_path: str | None = "configs/problematic_files.csv"

    def __post_init__(self) -> None:
        """Convert base_dir to Path object."""
        self.base_dir = Path(self.base_dir)


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


@dataclass
class SpectrogramConfig:
    """SuperGaussian spectrogram parameters.

    Note: Consider increasing n_bands from 128 to 256 for better frequency resolution.
    This may improve model performance by providing more detailed spectral information.
    """

    f_min: float = 20.0
    f_mid: float = 1000.0
    f_max: float = 8000.0
    n_bands: int = 128  # TODO: Test with 256 for improved frequency resolution
    hop_length: int = 512
    n_fft: int = 2048
    signal_duration: float = 5.0

    # Normalization configuration
    normalization: SpectrogramNormalization = field(default_factory=SpectrogramNormalization)


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    name: str = "efficient_cnn"

    # Encoder configuration
    encoder_channels: list[int] = field(default_factory=lambda: [24, 32, 48, 96])
    encoder_repeats: list[int] = field(default_factory=lambda: [2, 2, 3, 3])
    expansions: list[int] = field(default_factory=lambda: [3, 3, 4, 4])  # Tuned for 300-500k params

    # Architecture details
    activation: str = "Mish"
    norm_type: str = "group"  # 'batch', 'group', 'layer'
    norm_groups: int = 1  # num_groups for GroupNorm (1 = LayerNorm)
    use_se: bool = True  # Squeeze-and-Excitation blocks
    se_reduction: int = 4  # SE reduction ratio
    drop_connect_rate: float = 0.2
    dropout: float = 0.4

    # Weight normalization
    weight_scaling: str | None = "normalization"  # 'normalization' | 'standardization' | None
    weight_norm_interval: int = 1  # Call weight_scaling every N steps

    # Output
    num_classes: int = 80


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""

    # Spectrogram masking
    spec_augment: bool = True
    freq_mask_param: int = 20  # Max frequency bins to mask
    time_mask_param: int = 40  # Max time frames to mask
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # Audio augmentations
    time_stretch: bool = True
    time_stretch_range: tuple[float, float] = (0.8, 1.2)

    pitch_shift: bool = True
    pitch_shift_range: tuple[int, int] = (-2, 2)  # Semitones

    add_noise: bool = True
    noise_std_range: tuple[float, float] = (0.001, 0.01)

    reverberation: bool = False  # Computationally expensive
    reverb_room_size_range: tuple[float, float] = (10.0, 100.0)  # meters

    freq_response_perturbation: bool = True
    freq_response_std: float = 1.0  # dB

    mixup: bool = True
    mixup_alpha: float = 0.4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 64
    epochs: int = 100

    # Optimizer
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Learning rate schedule
    scheduler: str = "cosine"  # 'cosine' | 'plateau' | 'exponential' | 'custom'
    lr_min: float = 1e-6
    warmup_epochs: int = 5

    # Noisy label handling
    use_noisy_data: bool = False
    noisy_sample_weight: float = 0.5  # Weight for noisy samples vs curated
    label_smoothing: float = 0.1  # For noisy samples

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision

    # Validation
    val_split: float = 0.2
    val_interval: int = 1  # Validate every N epochs


@dataclass
class EvaluationConfig:
    """Evaluation parameters."""

    threshold: float = 0.5
    save_predictions: bool = True
    compute_per_class_metrics: bool = True


@dataclass
class OutputConfig:
    """Output paths."""

    model_dir: Path = Path("reports/models")
    metrics_dir: Path = Path("reports/metrics")
    figures_dir: Path = Path("reports/figures")
    tensorboard_dir: Path | None = Path("reports/tensorboard")

    def __post_init__(self) -> None:
        """Convert strings to Path objects and ensure all directories exist."""
        # Convert all string paths to Path objects
        for attr_name in ["model_dir", "metrics_dir", "figures_dir", "tensorboard_dir"]:
            value = getattr(self, attr_name)
            if value is not None and isinstance(value, str):
                setattr(self, attr_name, Path(value))

        # Create directories
        for attr_name in ["model_dir", "metrics_dir", "figures_dir"]:
            path = getattr(self, attr_name)
            if path:
                path.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_dir:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Complete configuration for audio tagging project."""

    project: ProjectConfig
    data: DataConfig
    spectrogram: SpectrogramConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object with all settings
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Handle nested normalization config in spectrogram
        spectrogram_dict = config_dict.get("spectrogram", {})
        if "normalization" in spectrogram_dict:
            normalization_dict = spectrogram_dict.pop("normalization")
            spectrogram_dict["normalization"] = SpectrogramNormalization(**normalization_dict)

        return cls(
            project=ProjectConfig(**config_dict.get("project", {})),
            data=DataConfig(**config_dict["data"]),
            spectrogram=SpectrogramConfig(**spectrogram_dict),
            model=ModelConfig(**config_dict.get("model", {})),
            augmentation=AugmentationConfig(**config_dict.get("augmentation", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            output=OutputConfig(**config_dict.get("output", {})),
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to output YAML file
        """

        # Convert dataclasses to dict recursively
        def dataclass_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            return obj

        config_dict = dataclass_to_dict(self)

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Usage example
if __name__ == "__main__":
    # Load from YAML
    config = Config.from_yaml("configs/baseline.yaml")

    # Access with dot notation (IDE autocomplete works!)
    print(f"Sample rate: {config.data.sample_rate}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Model: {config.model.name}")

    # Type checking works
    # config.data.sample_rate = "44100"  # Linter error: expected int, got str

    # Save to YAML
    config.to_yaml("configs/test_output.yaml")
