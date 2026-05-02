from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        hebb_lr: float = 1e-3,
        normalize_filters: bool = True,
        use_relu_output: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hebb_lr = hebb_lr
        self.normalize_filters = normalize_filters
        self.use_relu_output = use_relu_output

        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.05
        self.weight = nn.Parameter(weight, requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        else:
            self.bias = None

        self._last_input: torch.Tensor | None = None
        self._last_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input = x.detach()

        y = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        if self.use_relu_output:
            y = F.relu(y)

        self._last_output = y.detach()
        return y

    def reset_cache(self) -> None:
        self._last_input = None
        self._last_output = None

    def normalize_weight(self) -> None:
        if not self.normalize_filters:
            return

        with torch.no_grad():
            w = self.weight.view(self.out_channels, -1)
            norms = torch.norm(w, dim=1, keepdim=True).clamp_min(1e-8)
            w = w / norms
            self.weight.copy_(w.view_as(self.weight))

    def hebbian_update(
        self,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> None:

        if x is None:
            x = self._last_input
        if y is None:
            y = self._last_output

        if x is None or y is None:
            raise RuntimeError(
                "Hebbian update requires either explicit x, y or a previous forward pass."
            )

        with torch.no_grad():
            batch_size = x.shape[0]

            patches = F.unfold(
                x,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )

            y_flat = y.view(batch_size, self.out_channels, -1)
            w_flat = self.weight.view(self.out_channels, -1)
            positive = torch.einsum("bol,bpl->op", y_flat, patches)
            positive = positive / (batch_size * patches.shape[-1])

            y2_mean = (y_flat ** 2).mean(dim=(0, 2))
            negative = y2_mean.unsqueeze(1) * w_flat
            delta_w = self.hebb_lr * (positive - negative)
            w_flat = w_flat + delta_w
            self.weight.copy_(w_flat.view_as(self.weight))

            if self.bias is not None:
                bias_delta = self.hebb_lr * y_flat.mean(dim=(0, 2))
                self.bias.add_(bias_delta)

            self.normalize_weight()

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"hebb_lr={self.hebb_lr}, "
            f"normalize_filters={self.normalize_filters}, "
            f"use_relu_output={self.use_relu_output}"
        )