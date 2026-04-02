"""Neural network components and building blocks."""

import torch
import torch.nn as nn


class WeightNormalizedConv2d(nn.Conv2d):
    """Conv2d layer with weight normalization support.

    Weight normalization normalizes filter weights to unit L2 norm, improving
    training stability and reducing internal covariate shift. This is a drop-in
    replacement for nn.Conv2d with an additional `weight_scaling()` method.

    Usage:
        conv = WeightNormalizedConv2d(in_channels=64, out_channels=128, kernel_size=3)
        # During training, periodically call:
        conv.weight_scaling()

    Args:
        Same as nn.Conv2d
    """

    def weight_scaling(self) -> None:
        """Normalize filter weights to unit L2 norm.

        Computes L2 norm per output channel (over spatial dims and input channels),
        then divides weights by norm + epsilon for numerical stability.

        The normalization is applied in-place to self.weight. Norm is computed over
        dimensions (1, 2, 3): input_channels, kernel_height, kernel_width.
        """
        with torch.no_grad():
            # Compute L2 norm per filter: sqrt(sum(w^2)) over dims (1, 2, 3)
            # Shape: (out_channels, 1, 1, 1) for proper broadcasting
            norm = torch.sqrt((self.weight**2).sum(dim=(1, 2, 3), keepdim=True) + 1e-5)
            self.weight /= norm
