"""Evaluation script for trained audio tagging models."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.device import get_device
from src.core.utils import load_config
from src.data.dataset import AudioDatasetConfig, AudioLoader, MetadataManager
from src.features.signal_tools import SuperGaussianEnvelope
from src.features.spectrogram_optimized import MultiResolutionFilterBank
from src.models.baseline_cnn import create_model_from_config
from src.scripts.train import SpectrogramDataset
from src.training.metrics import MetricsTracker


def load_model_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: str,
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model
    model = create_model_from_config(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on test set.

    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Tuple of (metrics, predictions, targets)
    """
    model.eval()
    metrics_tracker = MetricsTracker()

    all_predictions = []
    all_targets = []

    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for spectrograms, targets in progress_bar:
            # Move to device
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(spectrograms)

            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(logits)

            # Track metrics
            metrics_tracker.update(predictions, targets)

            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute metrics
    metrics = metrics_tracker.compute()

    # Concatenate all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    return metrics, all_predictions, all_targets


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    filenames: list[str],
    vocabulary: list[str],
    output_path: Path,
) -> None:
    """Save predictions to CSV file.

    Args:
        predictions: Prediction probabilities of shape (n_samples, n_classes)
        targets: Ground truth targets of shape (n_samples, n_classes)
        filenames: List of filenames
        vocabulary: List of class labels
        output_path: Path to save predictions
    """
    # Create DataFrame with predictions for each class
    data = {"filename": filenames}

    for i, label in enumerate(vocabulary):
        data[f"pred_{label}"] = predictions[:, i]
        data[f"target_{label}"] = targets[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    vocabulary: list[str],
) -> pd.DataFrame:
    """Compute per-class metrics.

    Args:
        predictions: Prediction probabilities of shape (n_samples, n_classes)
        targets: Ground truth targets of shape (n_samples, n_classes)
        vocabulary: List of class labels

    Returns:
        DataFrame with per-class metrics
    """
    from sklearn.metrics import average_precision_score

    metrics_data = []

    for i, label in enumerate(vocabulary):
        class_targets = targets[:, i]
        class_predictions = predictions[:, i]

        # Skip classes with no positive samples
        if class_targets.sum() == 0:
            continue

        # Compute average precision
        ap = average_precision_score(class_targets, class_predictions)

        # Compute support (number of positive samples)
        support = int(class_targets.sum())

        metrics_data.append(
            {
                "label": label,
                "average_precision": ap,
                "support": support,
            }
        )

    df = pd.DataFrame(metrics_data)
    df = df.sort_values("average_precision", ascending=False)

    return df


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate audio tagging model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best_model.pt from config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train_curated", "train_noisy", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    metrics_dir = Path(config["output"]["metrics_dir"])
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

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(config["output"]["model_dir"]) / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")

    # Load model
    model = load_model_checkpoint(checkpoint_path, config, device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create filter bank
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

    # Load test metadata
    print(f"Loading {args.split} metadata...")
    test_metadata = metadata_manager.load_metadata(args.split, skip_problematic=True)
    print(f"Test samples: {len(test_metadata)}")

    # Create test dataset
    audio_loader = AudioLoader(dataset_config, args.split)
    hop_length = config["spectrogram"]["hop_length"]

    test_dataset = SpectrogramDataset(
        test_metadata,
        audio_loader,
        filter_bank,
        vocabulary,
        hop_length,
    )

    # Create test loader
    batch_size = config["training"]["batch_size"]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    print("\nEvaluating model...")
    print("=" * 70)

    metrics, predictions, targets = evaluate_model(model, test_loader, device)

    # Print overall metrics
    print("\nOverall Metrics:")
    print("-" * 70)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:.4f}")

    # Compute per-class metrics
    print("\nComputing per-class metrics...")
    per_class_metrics = compute_per_class_metrics(predictions, targets, vocabulary)

    print("\nTop 10 classes by Average Precision:")
    print(per_class_metrics.head(10).to_string(index=False))

    print("\nBottom 10 classes by Average Precision:")
    print(per_class_metrics.tail(10).to_string(index=False))

    # Save metrics
    metrics_output_path = metrics_dir / f"evaluation_metrics_{args.split}.json"
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_output_path}")

    # Save per-class metrics
    per_class_output_path = metrics_dir / f"per_class_metrics_{args.split}.csv"
    per_class_metrics.to_csv(per_class_output_path, index=False)
    print(f"Per-class metrics saved to: {per_class_output_path}")

    # Save predictions if requested
    if args.save_predictions:
        filenames = test_metadata["fname"].tolist()
        predictions_output_path = metrics_dir / f"predictions_{args.split}.csv"
        save_predictions(predictions, targets, filenames, vocabulary, predictions_output_path)

    print("\n" + "=" * 70)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
