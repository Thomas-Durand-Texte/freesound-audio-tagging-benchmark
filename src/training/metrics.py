"""Evaluation metrics for multi-label audio classification."""

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def lwlrap(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Label-Weighted Label-Ranking Average Precision (lwlrap).

    This is the official metric for the Freesound Audio Tagging 2019 challenge.

    lwlrap is a sample-weighted mean of the per-class average precisions,
    where the weight for each sample is the number of positive labels it has.

    Reference:
        https://www.kaggle.com/competitions/freesound-audio-tagging-2019/overview/evaluation

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        targets: Ground truth binary labels of shape (n_samples, n_classes)

    Returns:
        lwlrap score
    """
    n_samples, n_classes = targets.shape

    # Compute per-sample weights (number of positive labels per sample)
    sample_weights = targets.sum(axis=1)

    # Filter out samples with no positive labels
    mask = sample_weights > 0
    if not mask.any():
        return 0.0

    targets = targets[mask]
    predictions = predictions[mask]
    sample_weights = sample_weights[mask]

    # Compute per-class average precision for each sample
    # This requires computing AP for each (sample, class) pair where target=1
    total_weighted_ap = 0.0
    total_weight = 0.0

    for sample_idx in range(len(targets)):
        sample_targets = targets[sample_idx]
        sample_preds = predictions[sample_idx]

        # Get positive class indices for this sample
        pos_class_indices = np.where(sample_targets == 1)[0]

        if len(pos_class_indices) == 0:
            continue

        # For each positive class, compute precision at its rank
        for pos_class in pos_class_indices:
            # Get prediction score for this positive class
            pos_score = sample_preds[pos_class]

            # Count how many classes have higher or equal scores (including this one)
            # These are the "retrieved" classes at this threshold
            num_retrieved = (sample_preds >= pos_score).sum()

            # Count how many of these retrieved classes are actually positive
            # and have scores >= this positive class score
            retrieved_mask = sample_preds >= pos_score
            num_relevant_retrieved = (sample_targets[retrieved_mask] == 1).sum()

            # Precision at this rank
            precision_at_k = num_relevant_retrieved / num_retrieved

            # Add to weighted sum
            total_weighted_ap += precision_at_k
            total_weight += 1.0

    if total_weight == 0:
        return 0.0

    return total_weighted_ap / total_weight


def mean_average_precision(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Average Precision (mAP) across all classes.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        targets: Ground truth binary labels of shape (n_samples, n_classes)

    Returns:
        mAP score
    """
    # sklearn's average_precision_score with average='macro' gives mAP
    # Handle classes with no positive samples
    aps = []
    for class_idx in range(targets.shape[1]):
        class_targets = targets[:, class_idx]

        # Skip classes with no positive or no negative samples
        if class_targets.sum() == 0 or class_targets.sum() == len(class_targets):
            continue

        ap = average_precision_score(class_targets, predictions[:, class_idx])
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def compute_f1_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute F1, precision, and recall scores.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        targets: Ground truth binary labels of shape (n_samples, n_classes)
        threshold: Decision threshold for converting probabilities to binary predictions

    Returns:
        Dictionary with micro/macro F1, precision, and recall
    """
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)

    # Compute metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets, binary_preds, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, binary_preds, average="macro", zero_division=0
    )

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "precision_macro": float(precision_macro),
        "recall_micro": float(recall_micro),
        "recall_macro": float(recall_macro),
    }


def compute_auc_roc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute macro-averaged AUC-ROC score.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        targets: Ground truth binary labels of shape (n_samples, n_classes)

    Returns:
        Macro-averaged AUC-ROC score
    """
    try:
        return float(roc_auc_score(targets, predictions, average="macro", multi_class="ovr"))
    except ValueError:
        # Handle cases where some classes have no positive samples
        return 0.0


@dataclass
class MetricsTracker:
    """Tracks and computes metrics over batches.

    This class accumulates predictions and targets over multiple batches
    and computes various metrics at the end.

    Attributes:
        predictions: List of prediction arrays
        targets: List of target arrays
        threshold: Decision threshold for binary classification
    """

    threshold: float = 0.5
    predictions: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)

    def update(
        self, predictions: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray
    ) -> None:
        """Add a batch of predictions and targets.

        Args:
            predictions: Batch predictions (logits or probabilities)
            targets: Batch ground truth labels
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Apply sigmoid if values are not in [0, 1] range (i.e., they are logits)
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = 1 / (1 + np.exp(-predictions))  # sigmoid

        self.predictions.append(predictions)
        self.targets.append(targets)

    def compute(self) -> dict[str, float]:
        """Compute all metrics from accumulated predictions and targets.

        Returns:
            Dictionary with all computed metrics
        """
        if not self.predictions:
            return {}

        # Concatenate all batches
        all_predictions = np.vstack(self.predictions)
        all_targets = np.vstack(self.targets)

        # Compute metrics
        metrics = {}

        # lwlrap (official metric)
        metrics["lwlrap"] = lwlrap(all_predictions, all_targets)

        # mAP
        metrics["mAP"] = mean_average_precision(all_predictions, all_targets)

        # F1, precision, recall
        f1_metrics = compute_f1_scores(all_predictions, all_targets, self.threshold)
        metrics.update(f1_metrics)

        # AUC-ROC
        metrics["auc_roc"] = compute_auc_roc(all_predictions, all_targets)

        return metrics

    def reset(self) -> None:
        """Reset accumulated predictions and targets."""
        self.predictions.clear()
        self.targets.clear()

    def __repr__(self) -> str:
        """String representation."""
        n_samples = sum(len(p) for p in self.predictions)
        return f"MetricsTracker(n_samples={n_samples}, threshold={self.threshold})"


if __name__ == "__main__":
    # Test metrics with synthetic data
    np.random.seed(42)

    n_samples = 100
    n_classes = 80

    # Generate synthetic predictions and targets
    predictions = np.random.rand(n_samples, n_classes)
    targets = np.random.randint(0, 2, (n_samples, n_classes)).astype(float)

    # Ensure at least one positive label per sample
    for i in range(n_samples):
        if targets[i].sum() == 0:
            targets[i, np.random.randint(0, n_classes)] = 1

    print("Testing metrics with synthetic data:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Average positive labels per sample: {targets.sum(axis=1).mean():.2f}")
    print("-" * 50)

    # Test individual metrics
    print(f"lwlrap: {lwlrap(predictions, targets):.4f}")
    print(f"mAP: {mean_average_precision(predictions, targets):.4f}")
    print(f"AUC-ROC: {compute_auc_roc(predictions, targets):.4f}")

    f1_metrics = compute_f1_scores(predictions, targets)
    for metric_name, value in f1_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("-" * 50)

    # Test MetricsTracker
    tracker = MetricsTracker(threshold=0.5)

    # Simulate batched updates
    batch_size = 20
    for i in range(0, n_samples, batch_size):
        batch_preds = predictions[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]
        tracker.update(batch_preds, batch_targets)

    print(f"\n{tracker}")
    print("\nAll metrics from tracker:")
    all_metrics = tracker.compute()
    for metric_name, value in all_metrics.items():
        print(f"{metric_name}: {value:.4f}")
