"""Neural network components and building blocks."""

import torch
import torch.nn as nn


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    SE blocks adaptively recalibrate channel-wise feature responses by learning
    channel interdependencies. They apply global pooling to squeeze spatial info,
    then learn channel-wise excitation weights via a bottleneck FC layer.

    Architecture:
        1. Squeeze: Global average pooling (B, C, H, W) → (B, C, 1, 1)
        2. Excitation: FC → Mish → FC → Sigmoid (with channel reduction)
        3. Scale: Element-wise multiply with input

    Minimal overhead (~1-2% parameters) with +0.5-1.0% accuracy improvement.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 4)

    Example:
        se = SqueezeAndExcitation(channels=64, reduction=4)
        x = torch.randn(8, 64, 32, 32)
        out = se(x)  # Shape: (8, 64, 32, 32)
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize Squeeze-and-Excitation block.

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio (default: 4)
        """
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.mish = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise attention to input features.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Recalibrated features of shape (B, C, H, W)
        """
        # Squeeze: global average pooling
        z = x.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)

        # Excitation: FC → Mish → FC → Sigmoid
        z = self.fc1(z)
        z = self.mish(z)
        z = self.fc2(z)
        scale = self.sigmoid(z)  # (B, C, 1, 1), values in [0, 1]

        # Scale original features
        return x * scale


class DropConnect(nn.Module):
    """DropConnect (stochastic depth) regularization layer.

    DropConnect randomly drops entire channels during training, encouraging diverse
    feature learning and reducing overfitting. Unlike Dropout which drops individual
    activations, DropConnect drops entire feature maps (channels).

    During training, each channel is independently dropped with probability `drop_rate`.
    The remaining channels are scaled by 1/(1-drop_rate) to maintain expected activation
    magnitude. During evaluation, the layer acts as identity (no dropping).

    Args:
        drop_rate: Probability of dropping a channel (0.0 to 1.0). Default: 0.0

    Example:
        drop = DropConnect(drop_rate=0.2)
        x = torch.randn(8, 64, 32, 32)
        out = drop(x)  # During training: random channels dropped
                       # During eval: identical to input
    """

    def __init__(self, drop_rate: float = 0.0) -> None:
        """Initialize DropConnect layer.

        Args:
            drop_rate: Probability of dropping a channel (0.0 to 1.0)
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise dropout during training.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Features with channels randomly dropped during training, shape (B, C, H, W)
        """
        if not self.training or self.drop_rate == 0.0:
            return x

        # Edge case: drop all channels
        if self.drop_rate == 1.0:
            return torch.zeros_like(x)

        # Keep probability
        keep_prob = 1.0 - self.drop_rate

        # Random tensor: (B, C, 1, 1)
        # Dropout along channel dimension, keep spatial dims
        random_tensor = keep_prob + torch.rand(
            (x.shape[0], x.shape[1], 1, 1), dtype=x.dtype, device=x.device
        )

        # Binary mask (0 or 1)
        binary_mask = torch.floor(random_tensor)

        # Apply mask and scale by keep_prob to maintain expected value
        return x * binary_mask / keep_prob


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


class SepConvLayer(nn.Module):
    """Separable convolution layer with expansion, SE block, and DropConnect.

    Efficient building block that combines:
    1. Optional 1x1 expansion conv (if expansion > 1)
    2. Depthwise 3x3 conv (one filter per channel)
    3. GroupNorm + Activation (Mish)
    4. Squeeze-and-Excitation block
    5. Pointwise 1x1 projection
    6. DropConnect regularization

    This provides parameter efficiency (~8x reduction vs standard conv) while
    maintaining model expressiveness through channel expansion and SE attention.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        expansion: Channel expansion factor (default: 4)
        se_reduction: SE block reduction ratio (default: 4)
        drop_connect_rate: DropConnect probability (default: 0.2)
        activation: Activation function name (default: "Mish")
        norm_groups: GroupNorm groups, 1 = LayerNorm (default: 1)

    Example:
        layer = SepConvLayer(in_channels=64, out_channels=128, expansion=4)
        x = torch.randn(8, 64, 32, 32)
        out = layer(x)  # Shape: (8, 128, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        se_reduction: int = 4,
        drop_connect_rate: float = 0.2,
        activation: str = "Mish",
        norm_groups: int = 1,
    ) -> None:
        """Initialize SepConvLayer.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            expansion: Channel expansion factor (default: 4)
            se_reduction: SE reduction ratio (default: 4)
            drop_connect_rate: DropConnect probability (default: 0.2)
            activation: Activation function name (default: "Mish")
            norm_groups: GroupNorm groups (1 = LayerNorm)
        """
        super().__init__()

        expanded_channels = in_channels * expansion

        # 1. Expansion (if expansion > 1)
        if expansion > 1:
            self.expand = WeightNormalizedConv2d(
                in_channels, expanded_channels, kernel_size=1, bias=False
            )
        else:
            self.expand = nn.Identity()
            expanded_channels = in_channels

        # 2. Depthwise
        self.depthwise = WeightNormalizedConv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            padding=1,
            groups=expanded_channels,  # Key: groups = channels
            bias=False,
        )

        # 3. Normalization
        self.norm = nn.GroupNorm(norm_groups, expanded_channels)

        # 4. Activation
        self.activation = nn.Mish(inplace=True) if activation == "Mish" else nn.ReLU(inplace=True)

        # 5. Squeeze-and-Excitation
        self.se = SqueezeAndExcitation(expanded_channels, reduction=se_reduction)

        # 6. Pointwise
        self.project = WeightNormalizedConv2d(
            expanded_channels, out_channels, kernel_size=1, bias=False
        )

        # 7. DropConnect
        self.drop_connect = DropConnect(drop_connect_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through separable convolution layer.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Expansion
        out = self.expand(x)

        # Depthwise + Norm + Activation
        out = self.depthwise(out)
        out = self.norm(out)
        out = self.activation(out)

        # SE block
        out = self.se(out)

        # Pointwise projection
        out = self.project(out)

        # DropConnect
        out = self.drop_connect(out)

        return out
