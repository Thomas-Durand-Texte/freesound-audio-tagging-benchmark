"""Neural network architectures for audio tagging."""

from typing import Any

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """Convolutional Neural Network for multi-label audio classification.

    Architecture:
        - Multiple Conv2D blocks with BatchNorm, ReLU, and MaxPool
        - Global Average Pooling (handles variable time dimension)
        - Fully connected layer for multi-label classification

    Args:
        num_classes: Number of output classes (default: 80 for FSDKaggle2019)
        channels: List of channel dimensions for conv blocks (default: [16, 32, 64])
        kernel_size: Kernel size for convolutions (default: 3)
        dropout: Dropout rate (default: 0.4)
    """

    def __init__(
        self,
        num_classes: int = 80,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]

        self.num_classes = num_classes
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Build convolutional blocks
        conv_blocks = []
        in_channels = 1  # Single-channel spectrogram input

        for out_channels in channels:
            conv_blocks.append(
                self._make_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Global average pooling (reduces spatial dimensions)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Linear(channels[-1], num_classes)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> nn.Sequential:
        """Create a convolutional block.

        Block structure: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dropout: Dropout rate

        Returns:
            Sequential module containing the conv block
        """
        padding = kernel_size // 2  # Maintain spatial dimensions

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # BatchNorm handles bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram tensor of shape (batch, 1, freq, time)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Convolutional feature extraction
        x = self.conv_blocks(x)  # (batch, channels[-1], freq', time')

        # Global pooling
        x = self.global_pool(x)  # (batch, channels[-1], 1, 1)
        x = x.flatten(1)  # (batch, channels[-1])

        # Classification
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(config: dict[str, Any]) -> AudioCNN:
    """Create model from configuration dictionary.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized AudioCNN model
    """
    model_config = config.get("model", {})

    return AudioCNN(
        num_classes=model_config.get("num_classes", 80),
        channels=model_config.get("channels", [16, 32, 64]),
        kernel_size=model_config.get("kernel_size", 3),
        dropout=model_config.get("dropout", 0.4),
    )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = AudioCNN(num_classes=80, channels=[16, 32, 64])
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Test forward pass with dummy input
    batch_size = 4
    freq_bins = 128
    time_steps = 431  # ~5 seconds at hop_length=512, sr=44100

    dummy_input = torch.randn(batch_size, 1, freq_bins, time_steps)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 80)")
