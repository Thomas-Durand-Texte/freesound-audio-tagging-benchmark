"""Audio dataset management for Freesound Audio Tagging benchmark."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

DatasetType = Literal["train_curated", "train_noisy", "test"]
VALID_DATASETS: tuple[str, ...] = ("train_curated", "train_noisy", "test")


def validate_dataset_type(dataset_type: str) -> None:
    """Validate dataset type is allowed.

    Args:
        dataset_type: Dataset type to validate

    Raises:
        ValueError: If dataset type is invalid
    """
    if dataset_type not in VALID_DATASETS:
        valid_options = " | ".join(VALID_DATASETS)
        raise ValueError(
            f"Invalid dataset type: {dataset_type}. Must be {valid_options}."
        )


@dataclass
class AudioDatasetConfig:
    """Configuration for audio dataset paths and parameters."""

    base_dir: Path
    base_folder_name: str
    sample_rate: int = 16000
    clip_duration: float = 5.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AudioDatasetConfig":
        """Create config from dictionary."""
        return cls(
            base_dir=Path(config_dict["base_dir"]),
            base_folder_name=config_dict["base_folder_name"],
            sample_rate=config_dict.get("sample_rate", 16000),
            clip_duration=config_dict.get("clip_duration", 5.0),
        )


class MetadataManager:
    """Manages metadata and vocabulary for audio datasets."""

    def __init__(self, config: AudioDatasetConfig) -> None:
        """Initialize metadata manager.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.meta_dir = config.base_dir / (config.base_folder_name + "meta")
        self.vocabulary = self._load_vocabulary()
        self._metadata_cache: dict[str, pd.DataFrame] = {}

    def _load_vocabulary(self) -> pd.DataFrame:
        """Load label vocabulary."""
        return pd.read_csv(self.meta_dir / "vocabulary.csv")

    def load_metadata(self, dataset_type: DatasetType) -> pd.DataFrame:
        """Load metadata for specific dataset split.

        Args:
            dataset_type: Dataset split to load

        Returns:
            DataFrame with fname and labels columns
        """
        validate_dataset_type(dataset_type)

        if dataset_type not in self._metadata_cache:
            metadata_file = self.meta_dir / f"{dataset_type}_post_competition.csv"
            self._metadata_cache[dataset_type] = pd.read_csv(metadata_file)[
                ["fname", "labels"]
            ]

        return self._metadata_cache[dataset_type]


class AudioLoader:
    """Handles audio file loading with efficient path management."""

    def __init__(self, config: AudioDatasetConfig, dataset_type: DatasetType) -> None:
        """Initialize audio loader.

        Args:
            config: Dataset configuration
            dataset_type: Which dataset split to use
        """
        validate_dataset_type(dataset_type)
        self.config = config
        self.dataset_type = dataset_type
        self.audio_dir = self._build_audio_dir(dataset_type)

    def _build_audio_dir(self, dataset_type: DatasetType) -> Path:
        """Build audio directory path for dataset type.

        Args:
            dataset_type: Dataset split

        Returns:
            Path to audio directory
        """
        audio_folder = self.config.base_folder_name + "audio_" + dataset_type
        return self.config.base_dir / audio_folder

    def switch_dataset(self, dataset_type: DatasetType) -> None:
        """Switch to different dataset split.

        Args:
            dataset_type: New dataset split
        """
        validate_dataset_type(dataset_type)
        self.dataset_type = dataset_type
        self.audio_dir = self._build_audio_dir(dataset_type)

    def get_audio_path(self, filename: str) -> Path:
        """Get full path to audio file.

        Args:
            filename: Audio filename

        Returns:
            Full path to audio file
        """
        return self.audio_dir / filename

    def get_audio_info(self, filename: str) -> tuple[int, float]:
        """Get audio file sample rate and duration without loading waveform.

        Args:
            filename: Audio filename

        Returns:
            Tuple of (sample_rate, duration_seconds)
        """
        audio_path = self.get_audio_path(filename)
        info = sf.info(audio_path)
        return info.samplerate, info.duration

    def load_audio(self, filename: str) -> tuple[np.ndarray, int]:
        """Load audio file.

        Args:
            filename: Audio filename

        Returns:
            Tuple of (waveform, sample_rate)
        """
        audio_path = self.get_audio_path(filename)
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        return waveform, sample_rate


class AudioDataset(Dataset):
    """PyTorch Dataset wrapper combining metadata and audio loading."""

    def __init__(
        self,
        config: AudioDatasetConfig,
        dataset_type: DatasetType = "train_curated",
    ) -> None:
        """Initialize dataset.

        Args:
            config: Dataset configuration
            dataset_type: Which dataset split to load
        """
        self.config = config
        self.metadata_manager = MetadataManager(config)
        self.audio_loader = AudioLoader(config, dataset_type)

        self.dataset_type = dataset_type
        self.metadata = self.metadata_manager.load_metadata(dataset_type)

    def switch_dataset(self, dataset_type: DatasetType) -> None:
        """Switch to different dataset split.

        Args:
            dataset_type: New dataset split
        """
        self.dataset_type = dataset_type
        self.metadata = self.metadata_manager.load_metadata(dataset_type)
        self.audio_loader.switch_dataset(dataset_type)

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Get single sample.

        Args:
            index: Sample index

        Returns:
            Dictionary with waveform, labels, filename
        """
        row = self.metadata.iloc[index]
        filename = row["fname"]
        labels = row["labels"]

        waveform, _sample_rate = self.audio_loader.load_audio(filename)
        waveform_tensor = torch.from_numpy(waveform)

        return {
            "waveform": waveform_tensor,
            "labels": labels,
            "filename": filename,
        }


# Legacy function for backward compatibility
def load_metadata(config_data: dict, which: str) -> pd.DataFrame:
    """Load metadata (legacy function).

    Args:
        config_data: Configuration dictionary
        which: Dataset type or 'vocabulary'

    Returns:
        DataFrame with metadata
    """
    path = Path(config_data["base_dir"]) / (config_data["base_folder_name"] + "meta")
    allowed_types = ["train_curated", "train_noisy", "test", "vocabulary"]
    if which not in allowed_types:
        raise ValueError(
            f"Invalid dataset type: {which}. Must be {' | '.join(allowed_types)}."
        )
    if which == "vocabulary":
        return pd.read_csv(path / "vocabulary.csv")
    return pd.read_csv(path / f"{which}_post_competition.csv")[["fname", "labels"]]