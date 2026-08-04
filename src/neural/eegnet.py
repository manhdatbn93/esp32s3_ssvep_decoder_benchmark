"""
EEGNet-style baseline for 12JFPM (Lawhern et al., 2018).

Input:  [B, T, C]  (e.g. [B, 512, 8])
Output: logits [B, num_classes]

Internally reshapes to [B, 1, C, T] for Conv2d operations.
Architecture: temporal conv -> depthwise spatial conv -> separable
temporal conv -> adaptive avg pool -> linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
        kernel_length: int = 64,
        pool1_stride: int = 4,
        pool2_stride: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (embed_dim, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, pool1_stride)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, pool2_stride)),
            nn.Dropout(dropout),
        )
        self.pool_out = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(F2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)  # [B, T, C] -> [B, 1, C, T]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool_out(x).flatten(1)
        return self.fc(x)
