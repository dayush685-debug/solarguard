"""Baseline CNN — see PLANNING.md Phase 3 design review for the full architectural
justification. Four conv blocks (16->32->64->128 channels), global average pooling
instead of Flatten+FC, ~98.5K parameters. This is deliberately small: its job is to be
the floor MobileNetV2 and EfficientNet (Phase 4/5) must clear, not to win on its own.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool, halving spatial resolution.

    bias=False on the conv: BatchNorm's own beta term already shifts the output,
    so a separate conv bias would be redundant — standard practice, not a shortcut.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class BaselineCNN(nn.Module):
    """Input: (batch, in_channels, 224, 224). Output: (batch, num_classes) raw logits
    — no softmax applied here; CrossEntropyLoss expects raw logits
    and applies log-softmax internally."""

    def __init__(self, num_classes: int = 6, in_channels: int = 3) -> None:
        super().__init__()
        self.block1 = _conv_block(in_channels, 16)
        self.block2 = _conv_block(16, 32)
        self.block3 = _conv_block(32, 64)
        self.block4 = _conv_block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, 16, 112, 112)
        x = self.block2(x)   # (B, 32, 56, 56)
        x = self.block3(x)   # (B, 64, 28, 28)
        x = self.block4(x)   # (B, 128, 14, 14)

        x = self.pool(x)             # (B, 128, 1, 1)
        x = torch.flatten(x, 1)      # (B, 128)
        x = self.dropout(x)
        return self.classifier(x)    # (B, num_classes) — logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
