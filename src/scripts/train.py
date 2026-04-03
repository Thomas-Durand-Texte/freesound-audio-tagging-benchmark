"""Training script for audio tagging models."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.core.device import get_device
from src.core.utils import load_config
from src.data.dataset import AudioDatasetConfig, AudioLoader, MetadataManager
from src.features.signal_tools import SuperGaussianEnvelope
from src.features.spectrogram_optimized import MultiResolutionFilterBank
from src.models.baseline_cnn import create_model_from_config
from src.training.losses import BCEWithLogitsLoss, create_loss_from_config
from src.training.metrics import MetricsTracker


class SpectrogramDataset(Dataset):
    """PyTorch Dataset that returns spectrograms for audio files.

    Args:
        metadata: Metadata DataFrame with fname and labels columns
        audio_loader: AudioLoader instance
        filter_bank: MultiResolutionFilterBank for feature extraction
        vocabulary: List of all possible labels (for multi-label encoding)
        hop_length: Hop length for spectrogram computation
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_loader: AudioLoader,
        filter_bank: MultiResolutionFilterBank,
        vocabulary: list[str],
        hop_length: int = 512,
    ) -> None:
        self.metadata = metadata
        self.audio_loader = audio_loader
        self.filter_bank = filter_bank
        self.vocabulary = vocabulary
        self.label_to_idx = {label: idx for idx, label in enumerate(vocabulary)}
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get spectrogram and multi-label target.

        Returns:
            Tuple of (spectrogram, target)
            - spectrogram: Tensor of shape (1, n_bands, n_frames)
            - target: Multi-hot encoded labels of shape (num_classes,)
        """
        row = self.metadata.iloc[idx]
        filename = row["fname"]
        labels_str = row["labels"]

        # Load audio
        waveform, _sample_rate = self.audio_loader.load_audio(filename)

        # Compute spectrogram
        spectrogram, _, _ = self.filter_bank.compute_spectrogram(
            waveform, hop_length=self.hop_length
        )

        # Convert to torch tensor and add channel dimension
        spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)

        # Multi-hot encode labels
        target = torch.zeros(len(self.vocabulary), dtype=torch.float32)
        labels = [label.strip() for label in labels_str.split(",")]
        for label in labels:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        return spectrogram_tensor, target


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_dataloaders(
    config: dict[str, Any],
    filter_bank: MultiResolutionFilterBank,
    vocabulary: list[str],
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        config: Configuration dictionary
        filter_bank: Pre-initialized filter bank
        vocabulary: List of all possible labels

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset config
    dataset_config = AudioDatasetConfig.from_dict(config["data"])

    # Load metadata
    metadata_manager = MetadataManager(dataset_config)

    # Load curated training data (skip problematic files)
    train_metadata = metadata_manager.load_metadata("train_curated", skip_problematic=True)

    # Split into train and validation
    train_size = int(0.8 * len(train_metadata))
    val_size = len(train_metadata) - train_size

    train_indices, val_indices = random_split(
        range(len(train_metadata)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["project"]["seed"]),
    )

    train_metadata_split = train_metadata.iloc[list(train_indices)].reset_index(drop=True)
    val_metadata_split = train_metadata.iloc[list(val_indices)].reset_index(drop=True)

    # Create audio loaders
    train_audio_loader = AudioLoader(dataset_config, "train_curated")
    val_audio_loader = AudioLoader(dataset_config, "train_curated")

    # Create datasets
    hop_length = config["spectrogram"]["hop_length"]

    train_dataset = SpectrogramDataset(
        train_metadata_split,
        train_audio_loader,
        filter_bank,
        vocabulary,
        hop_length,
    )

    val_dataset = SpectrogramDataset(
        val_metadata_split,
        val_audio_loader,
        filter_bank,
        vocabulary,
        hop_length,
    )

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Not beneficial for MPS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    metrics_tracker = MetricsTracker()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for spectrograms, targets in progress_bar:
        # Move to device
        spectrograms = spectrograms.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        metrics_tracker.update(logits.detach(), targets.detach())

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    metrics = metrics_tracker.compute()
    metrics["loss"] = avg_loss

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> dict[str, float]:
    """Validate the model.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    metrics_tracker = MetricsTracker()

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for spectrograms, targets in progress_bar:
            # Move to device
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(spectrograms)
            loss = criterion(logits, targets)

            # Track metrics
            total_loss += loss.item()
            metrics_tracker.update(logits, targets)

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

    # Compute epoch metrics
    avg_loss = total_loss / len(val_loader)
    metrics = metrics_tracker.compute()
    metrics["loss"] = avg_loss

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Validation metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train audio tagging model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config["project"]["seed"])

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    model_dir = Path(config["output"]["model_dir"])
    metrics_dir = Path(config["output"]["metrics_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    dataset_config = AudioDatasetConfig.from_dict(config["data"])
    metadata_manager = MetadataManager(dataset_config)
    vocabulary = sorted(metadata_manager.vocabulary["label"].tolist())
    num_classes = len(vocabulary)

    print(f"Number of classes: {num_classes}")

    # Update config with num_classes
    if "model" not in config:
        config["model"] = {}
    config["model"]["num_classes"] = num_classes

    # Create filter bank for feature extraction
    print("Initializing filter bank...")
    filter_bank = MultiResolutionFilterBank(
        envelope_class=SuperGaussianEnvelope,
        f_min=config["spectrogram"]["f_min"],
        f_max=config["spectrogram"]["f_max"],
        f_mid=config["spectrogram"].get("f_mid"),
        num_bands=config["spectrogram"]["n_bands"],
        sample_rate=config["data"]["sample_rate"],
        signal_duration=config["spectrogram"]["signal_duration"],
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config, filter_bank, vocabulary)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    model = model.to(device)

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create loss function
    criterion = create_loss_from_config(config)
    if criterion is None:
        criterion = BCEWithLogitsLoss()

    # Create optimizer
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    num_epochs = config["training"]["epochs"]
    best_val_lwlrap = 0.0
    history = {"train": [], "val": []}

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Store history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Print metrics
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, lwlrap: {train_metrics['lwlrap']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, lwlrap: {val_metrics['lwlrap']:.4f}")

        # Save checkpoint if best model
        if val_metrics["lwlrap"] > best_val_lwlrap:
            best_val_lwlrap = val_metrics["lwlrap"]
            checkpoint_path = model_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            print(f"  -> Saved best model (lwlrap: {best_val_lwlrap:.4f})")

        # Save last checkpoint
        checkpoint_path = model_dir / "last_model.pt"
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation lwlrap: {best_val_lwlrap:.4f}")

    # Save training history
    history_path = metrics_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
