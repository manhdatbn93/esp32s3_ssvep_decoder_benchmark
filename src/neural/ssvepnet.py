"""
SSVEPNet — Bi-LSTM + CNN hybrid for SSVEP (Pan et al., 2022).

Architecture: spatial Conv2d -> enhanced Conv2d -> Bi-LSTM -> MLP.

Input:  [B, T, C]
Output: logits [B, num_classes]

Internally reshaped to [B, 1, C, T] for Conv2d layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm weight constraint (renormalization)."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, T = x.shape
        x = x.permute(2, 0, 1)  # (T, B, C)
        r_out, _ = self.rnn(x)
        out = r_out.permute(1, 0, 2).reshape(b, -1)  # (B, T*2*hidden)
        return out


class SSVEPNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        F0 = embed_dim * 2
        F1 = embed_dim * 4
        K, S = 10, 2

        self.spatial_block = nn.Sequential(
            _Conv2dWithConstraint(1, F0, kernel_size=(embed_dim, 1), max_norm=1.0),
            nn.BatchNorm2d(F0),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.enhanced_block = nn.Sequential(
            nn.Conv2d(F0, F1, kernel_size=(1, K), stride=(1, S)),
            nn.BatchNorm2d(F1),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.randn(1, 1, embed_dim, seq_len)
            conv_out = self.enhanced_block(self.spatial_block(dummy))
            _, c_out, _, t_out = conv_out.shape

        self.rnn = _BiLSTMBlock(input_size=c_out, hidden_size=c_out)
        rnn_out_dim = t_out * 2 * c_out

        D1 = max(rnn_out_dim // 10, 1)
        D2 = max(D1 // 5, 1)

        self.dense = nn.Sequential(
            nn.Linear(rnn_out_dim, D1),
            nn.PReLU(),
            nn.Linear(D1, D2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)  # [B, T, C] -> [B, 1, C, T]
        x = self.spatial_block(x)
        x = self.enhanced_block(x)
        x = x.squeeze(2)  # [B, F, T'] -> for LSTM
        x = self.rnn(x)
        return self.dense(x)
