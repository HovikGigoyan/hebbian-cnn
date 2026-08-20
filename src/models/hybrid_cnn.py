from __future__ import annotations
from models.hebbian_layers import HebbianConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNN(nn.Module):
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

        self.conv2 = nn.Conv2d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = self._infer_flatten_dim(in_channels, input_size)
        self.feature_norm = nn.LayerNorm(self.flatten_dim)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _infer_flatten_dim(self, in_channels: int, input_size: tuple[int, int]) -> int:
        with torch.no_grad():
            x = torch.zeros(1, in_channels, input_size[0], input_size[1])
            x = self.pool(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(x, 1)
        x = self.feature_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x