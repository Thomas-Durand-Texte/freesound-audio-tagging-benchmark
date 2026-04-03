"""Efficient audio CNN for multi-label audio tagging.

DIC-inspired architecture combining separable convolutions, SE blocks,
weight normalization, and advanced regularization techniques.
"""

import torch
import torch.nn as nn

from src.core.config import ModelConfig
from src.models.components import SepConvLayer, WeightNormalizedConv2d


class EfficientAudioCNN(nn.Module):
    """Efficient CNN for audio tagging with 300-500k parameters.

    Architecture inspired by DIC neural networks, combining:
    - Separable convolutions for parameter efficiency
    - Squeeze-and-Excitation blocks for channel attention
    - Weight normalization for training stability
    - DropConnect for regularization
    - Mish activation and GroupNorm throughout

    Args:
        config: ModelConfig dataclass with architecture parameters

    Example:
        config = ModelConfig(
            encoder_channels=[24, 32, 48, 96],
            encoder_repeats=[2, 2, 3, 3],
            expansions=[4, 4, 6, 6],
            num_classes=80
        )
        model = EfficientAudioCNN(config)
        x = torch.randn(4, 1, 128, 431)
        output = model(x)  # (4, 80)
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize EfficientAudioCNN from config dataclass.

        Args:
            config: ModelConfig with encoder_channels, encoder_repeats,
                    expansions, num_classes, dropout, etc.
        """
        super().__init__()

        self.config = config
        channels = config.encoder_channels
        repeats = config.encoder_repeats
        expansions = config.expansions

        # Validate configuration
        assert len(channels) == len(repeats) == len(expansions), (
            f"Length mismatch: channels={len(channels)}, "
            f"repeats={len(repeats)}, expansions={len(expansions)}"
        )

        # Stem: 1x1 conv to first encoder channels
        self.stem = WeightNormalizedConv2d(1, channels[0], kernel_size=1, bias=False)

        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels)):
            # Determine input channels for this block
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            expansion = expansions[i]
            n_layers = repeats[i]

            # Build block with repeated SepConvLayers
            block_layers = []
            for j in range(n_layers):
                # First layer changes channels, rest maintain channels
                layer_in = in_ch if j == 0 else out_ch
                layer = SepConvLayer(
                    in_channels=layer_in,
                    out_channels=out_ch,
                    expansion=expansion,
                    se_reduction=config.se_reduction,
                    drop_connect_rate=config.drop_connect_rate,
                    activation=config.activation,
                    norm_groups=config.norm_groups,
                )
                block_layers.append(layer)

            self.encoder_blocks.append(nn.Sequential(*block_layers))

        # Pooling layers (after blocks 0-2, not after block 3)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(channels[-1], config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: spectrogram → class logits.

        Args:
            x: Input spectrogram of shape (B, 1, H, W)
               where H is frequency bins, W is time frames

        Returns:
            Class logits of shape (B, num_classes)
        """
        # Stem
        x = self.stem(x)

        # Encoder blocks with pooling
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            # Pool after blocks 0, 1, 2 (not after block 3)
            if i < len(self.encoder_blocks) - 1:
                x = self.pool(x)

        # Global pooling
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)

        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def apply_weight_scaling(self) -> None:
        """Apply weight normalization to all WeightNormalizedConv2d layers.

        This should be called periodically during training (e.g., after optimizer step)
        to maintain normalized filter weights for improved training stability.
        """
        for module in self.modules():
            if isinstance(module, WeightNormalizedConv2d):
                module.weight_scaling()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns:
            Total count of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
