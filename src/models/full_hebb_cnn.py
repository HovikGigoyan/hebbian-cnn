from __future__ import annotations
from src.models.hebbian_layers import HebbianConv2d
import torch
import torch.nn as nn


class FullHebbCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        input_size: tuple[int, int],
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        hebb_lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.conv1 = HebbianConv2d(
            in_channels,
            conv1_channels,
            kernel_size=kernel_size,
            padding=padding,
            hebb_lr=hebb_lr,
            use_relu_output=True,
        )
        self.conv2 = HebbianConv2d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size,
            padding=padding,
            hebb_lr=hebb_lr,
            use_relu_output=True,
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = self._infer_flatten_dim(in_channels, input_size)
        self.feature_norm = nn.LayerNorm(self.flatten_dim)
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _infer_flatten_dim(self, in_channels: int, input_size: tuple[int, int]) -> int:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, input_size[0], input_size[1])
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            return x.numel()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(x, 1)
        x = self.feature_norm(x)
        x = self.classifier(x)
        return x