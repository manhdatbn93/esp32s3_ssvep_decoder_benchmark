"""
FBtCNN — Filter-Bank Temporal CNN for SSVEP (Pan et al., 2024).

Architecture: 4 sub-band filter banks -> parallel CNN paths
(spatial conv -> temporal conv -> temporal conv) -> sum fusion
-> temporal conv -> FC.

Input:  [B, T, C]
Output: logits [B, num_classes]

Internally reshaped to [B, 1, C, T] for Conv2d operations.

Note: The filter bank is applied inside the forward pass using
scipy (matching the reference implementation). This makes the
model non-differentiable through the filter stage but ensures
exact parity with the reference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as scipy_signal


class _SamePadConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        pad_len = ((kernel_size[1] // stride[1]) - 1) * stride[1]
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        self.pad = nn.ZeroPad2d((pad_left, pad_right, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class FBtCNN(nn.Module):
    LOW_BANDS = [3, 9, 14, 19]
    HIGH_BANDS = [14, 26, 38, 50]

    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        sampling_rate_hz: float = 256.0,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.sampling_rate_hz = sampling_rate_hz
        self.n_bands = len(self.LOW_BANDS)

        K = seq_len // 5 - 4

        spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(embed_dim, 1), stride=(embed_dim, 1)),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001), nn.ELU(), nn.Dropout(dropout),
        )
        temporal1 = nn.Sequential(
            _SamePadConv2d(16, 16, (1, seq_len), (1, 5)),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001), nn.ELU(), nn.Dropout(dropout),
        )
        temporal2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001), nn.ELU(), nn.Dropout(dropout),
        )
        self.sub_band = nn.Sequential(spatial, temporal1, temporal2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, K), stride=(1, 1)),
            nn.BatchNorm2d(32, momentum=0.99, eps=0.001), nn.ELU(), nn.Dropout(dropout),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def _filter_bank(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply scipy filter bank on CPU, return list of filtered tensors."""
        device = x.device
        x_np = x.detach().cpu().numpy()
        filtered = []
        for lo, hi in zip(self.LOW_BANDS, self.HIGH_BANDS):
            b, a = scipy_signal.butter(
                6, [2 * lo / self.sampling_rate_hz, 2 * hi / self.sampling_rate_hz], btype="bandpass",
            )
            y = scipy_signal.filtfilt(b, a, x_np, axis=-1)
            filtered.append(torch.from_numpy(y.copy()).float().to(device))
        return filtered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)  # [B, T, C] -> [B, 1, C, T]
        bands = self._filter_bank(x)
        fused = sum(self.sub_band(b) for b in bands)
        out = self.final_conv(fused)
        return self.dense(out)
