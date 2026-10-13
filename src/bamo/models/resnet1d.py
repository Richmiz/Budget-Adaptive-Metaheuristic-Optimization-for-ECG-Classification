"""1-D residual CNN backbone (protocol §3.1.2).

Conv1D stem -> three residual groups (base widths 32/64/128 scaled by the width multiplier)
-> global average pooling -> dropout -> linear layer with five logits. Groups 2 and 3
downsample by stride 2 in their first block. ReLU and batch normalization are fixed.
"""

from __future__ import annotations

import torch
from torch import nn


def scaled_width(base: int, multiplier: float, divisor: int = 8) -> int:
    return max(divisor, int(round(base * multiplier / divisor)) * divisor)


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, stride: int):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1d(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, base_widths=(32, 64, 128), group_strides=(1, 2, 2),
                 width_multiplier: float = 1.0, blocks_per_group: int = 2, kernel_size: int = 9, dilation: int = 1,
                 stem_stride: int = 2, stem_kernel: int = 15, dropout: float = 0.2, channel_divisor: int = 8):
        super().__init__()
        widths = [scaled_width(w, width_multiplier, channel_divisor) for w in base_widths]
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], stem_kernel, stride=stem_stride, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True))
        layers, c_in = [], widths[0]
        for width, stride in zip(widths, group_strides):
            for b in range(blocks_per_group):
                layers.append(BasicBlock1d(c_in, width, kernel_size, dilation, stride if b == 0 else 1))
                c_in = width
        self.groups = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.groups(self.stem(x))).flatten(1)
        return self.head(self.dropout(x))


def build_model(config: dict, model_cfg: dict, n_classes: int = 5) -> ResNet1d:
    return ResNet1d(
        n_leads=model_cfg.get("n_leads", 12), n_classes=n_classes,
        base_widths=tuple(model_cfg.get("base_widths", (32, 64, 128))),
        group_strides=tuple(model_cfg.get("group_strides", (1, 2, 2))),
        width_multiplier=float(config["width_multiplier"]), blocks_per_group=int(config["blocks_per_group"]),
        kernel_size=int(config["kernel_size"]), dilation=int(config["dilation"]),
        stem_stride=int(config["stem_stride"]), stem_kernel=model_cfg.get("stem_kernel", 15),
        dropout=float(config["dropout"]), channel_divisor=model_cfg.get("channel_divisor", 8))
