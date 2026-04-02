"""Audio dataset management for Freesound Audio Tagging benchmark."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from src.core.config import DataConfig

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
        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be {valid_options}.")


class MetadataManager:
    """Manages metadata and vocabulary for audio datasets using DuckDB."""

    def __init__(self, config: "DataConfig") -> None:
        """Initialize metadata manager with DuckDB backend.

        Args:
            config: Data configuration from src.core.config
        """
        self.config = config
        self.meta_dir = config.base_dir / (config.base_folder_name + "meta")

        # Initialize DuckDB in-memory database
        self.con = duckdb.connect(":memory:")

        # Load vocabulary
        vocabulary_path = self.meta_dir / "vocabulary.csv"
        self.con.execute(f"""
            CREATE TABLE vocabulary AS
            SELECT * FROM read_csv_auto('{vocabulary_path}')
        """)
        self.vocabulary = self.con.execute("SELECT * FROM vocabulary").df()

        # Load all dataset metadata into DuckDB
        for dataset_type in VALID_DATASETS:
            metadata_file = self.meta_dir / f"{dataset_type}_post_competition.csv"
            if metadata_file.exists():
                self.con.execute(f"""
                    CREATE TABLE {dataset_type} AS
                    SELECT fname, labels FROM read_csv_auto('{metadata_file}')
                """)

        # Load problematic files if path provided
        if config.problematic_files_path and config.problematic_files_path.exists():
            self.con.execute(f"""
                CREATE TABLE problematic AS
                SELECT * FROM read_csv_auto('{config.problematic_files_path}')
            """)
        else:
            # Create empty table with schema
            self.con.execute("""
                CREATE TABLE problematic (
                    filename VARCHAR,
                    problem_type VARCHAR,
                    notes VARCHAR
                )
            """)

    def load_metadata(
        self, dataset_type: DatasetType, skip_problematic: bool = False
    ) -> pd.DataFrame:
        """Load metadata for specific dataset split.

        Args:
            dataset_type: Dataset split to load
            skip_problematic: Whether to exclude problematic files

        Returns:
            DataFrame with fname and labels columns
        """
        validate_dataset_type(dataset_type)

        query = f"SELECT fname, labels FROM {dataset_type}"

        if skip_problematic:
            query += " WHERE fname NOT IN (SELECT filename FROM problematic)"

        return self.con.execute(query).df()

    def get_problematic_files(self) -> pd.DataFrame:
        """Get list of all problematic files.

        Returns:
            DataFrame with problematic files information
        """
        return self.con.execute("SELECT * FROM problematic").df()

    def get_label_statistics(
        self, dataset_type: DatasetType, skip_problematic: bool = False
    ) -> pd.DataFrame:
        """Get per-label counts using SQL aggregation.

        Args:
            dataset_type: Dataset split to analyze
            skip_problematic: Whether to exclude problematic files

        Returns:
            DataFrame with label and count columns
        """
        validate_dataset_type(dataset_type)

        # Build WHERE clause for problematic files
        where_clause = ""
        if skip_problematic:
            where_clause = "WHERE fname NOT IN (SELECT filename FROM problematic)"

        query = f"""
            WITH label_split AS (
                SELECT
                    fname,
                    UNNEST(string_split(labels, ',')) as label
                FROM {dataset_type}
                {where_clause}
            )
            SELECT label, COUNT(*) as count
            FROM label_split
            GROUP BY label
            ORDER BY count DESC
        """

        return self.con.execute(query).df()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL query on metadata.

        Args:
            sql: SQL query string

        Returns:
            DataFrame with query results
        """
        return self.con.execute(sql).df()


class AudioLoader:
    """Handles audio file loading with efficient path management."""

    def __init__(self, config: "DataConfig", dataset_type: DatasetType) -> None:
        """Initialize audio loader.

        Args:
            config: Data configuration from src.core.config
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
        config: "DataConfig",
        dataset_type: DatasetType = "train_curated",
    ) -> None:
        """Initialize dataset.

        Args:
            config: Data configuration from src.core.config
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
        raise ValueError(f"Invalid dataset type: {which}. Must be {' | '.join(allowed_types)}.")
    if which == "vocabulary":
        return pd.read_csv(path / "vocabulary.csv")
    return pd.read_csv(path / f"{which}_post_competition.csv")[["fname", "labels"]]
