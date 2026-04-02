"""Loss functions for multi-label audio classification."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy loss with logits for multi-label classification.

    Wrapper around PyTorch's BCEWithLogitsLoss for consistency with other losses.

    Args:
        pos_weight: Weight for positive examples (helps with class imbalance)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Model predictions of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch, num_classes)

        Returns:
            Loss value
        """
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard examples. Useful for handling class imbalance.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor in [0, 1] for balancing positive/negative examples
        gamma: Focusing parameter >= 0 (gamma=0 is equivalent to BCE)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model predictions of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch, num_classes)

        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute binary cross-entropy (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute focal term: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine terms
        loss = alpha_t * focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Asymmetric loss applies different focusing parameters to positive and
    negative samples, which is particularly effective for multi-label problems
    with many negative labels per sample.

    Reference:
        Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (2021)
        https://arxiv.org/abs/2009.14119

    Args:
        gamma_pos: Focusing parameter for positive examples (default: 0)
        gamma_neg: Focusing parameter for negative examples (default: 4)
        clip: Probability clipping value to avoid numerical issues (default: 0.05)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            logits: Model predictions of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch, num_classes)

        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Probability clipping for numerical stability
        probs_pos = torch.clamp(probs, min=self.clip)
        probs_neg = torch.clamp(1 - probs, min=self.clip)

        # Asymmetric focusing
        loss_pos = targets * torch.log(probs_pos) * ((1 - probs_pos) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(probs_neg) * (probs**self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss with sample and class weighting.

    Useful for handling both class imbalance and noisy labels by applying
    different weights to samples and classes.

    Args:
        class_weights: Per-class weights of shape (num_classes,)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Model predictions of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch, num_classes)
            sample_weights: Per-sample weights of shape (batch,)

        Returns:
            Loss value
        """
        # Compute BCE loss without reduction
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply class weights
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        # Apply sample weights
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_loss_from_config(config: dict[str, Any]) -> nn.Module:
    """Create loss function from configuration dictionary.

    Args:
        config: Configuration dictionary with loss parameters

    Returns:
        Initialized loss function
    """
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "bce")

    if loss_type == "bce":
        return BCEWithLogitsLoss()
    elif loss_type == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
        )
    elif loss_type == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=loss_config.get("gamma_pos", 0.0),
            gamma_neg=loss_config.get("gamma_neg", 4.0),
            clip=loss_config.get("clip", 0.05),
        )
    elif loss_type == "weighted_bce":
        return WeightedBCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    num_classes = 80

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    print("Testing loss functions:")
    print("-" * 50)

    # BCE Loss
    bce_loss = BCEWithLogitsLoss()
    loss_value = bce_loss(logits, targets)
    print(f"BCE Loss: {loss_value.item():.4f}")

    # Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(logits, targets)
    print(f"Focal Loss: {loss_value.item():.4f}")

    # Asymmetric Loss
    asl_loss = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0)
    loss_value = asl_loss(logits, targets)
    print(f"Asymmetric Loss: {loss_value.item():.4f}")

    # Weighted BCE Loss
    class_weights = torch.ones(num_classes)
    sample_weights = torch.ones(batch_size)
    weighted_bce_loss = WeightedBCELoss(class_weights=class_weights)
    loss_value = weighted_bce_loss(logits, targets, sample_weights=sample_weights)
    print(f"Weighted BCE Loss: {loss_value.item():.4f}")
