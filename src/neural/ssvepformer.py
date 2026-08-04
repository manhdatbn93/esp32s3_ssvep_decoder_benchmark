"""
SSVEPformer — Transformer-based SSVEP decoder (Pan et al., 2024).

Operates on FFT spectrum features (same extraction as CCNN but with
different frequency range: 8-64 Hz, resolution=0.2 Hz).

Input:  [B, T, C]
Output: logits [B, num_classes]

Internally: FFT -> [B, C, Nfc] -> Conv1d patch embedding -> Transformer
-> MLP head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


_FFT_RESOLUTION_HZ: float = 0.2
_FFT_FREQ_LO_HZ: float = 8.0
_FFT_FREQ_HI_HZ: float = 64.0


def _ssvepformer_spectrum(x_btc: torch.Tensor, sampling_rate_hz: float) -> torch.Tensor:
    """Convert [B, T, C] -> [B, C, Nfc] spectrum features for SSVEPformer."""
    _b, t, _c = x_btc.shape
    nfft = round(sampling_rate_hz / _FFT_RESOLUTION_HZ)
    idx_lo = int(round(_FFT_FREQ_LO_HZ / _FFT_RESOLUTION_HZ))
    idx_hi = int(round(_FFT_FREQ_HI_HZ / _FFT_RESOLUTION_HZ))

    x_bct = x_btc.transpose(1, 2)
    spectrum = torch.fft.fft(x_bct, n=nfft, dim=-1) / (t / 2.0)
    real = spectrum[..., idx_lo:idx_hi].real
    imag = spectrum[..., idx_lo:idx_hi].imag
    features = torch.cat([real, imag], dim=-1).float()  # [B, C, 2*n_bins]
    return features


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class _ConvAttention(nn.Module):
    def __init__(self, token_num: int, token_length: int, kernel_length: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernel_length, padding=kernel_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeedForward(nn.Module):
    def __init__(self, token_length: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSVEPformer(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        sampling_rate_hz: float = 256.0,
        depth: int = 1,
        attention_kernel_length: int = 31,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.sampling_rate_hz = sampling_rate_hz

        token_num = embed_dim * 2
        n_bins = int(round(_FFT_FREQ_HI_HZ / _FFT_RESOLUTION_HZ)) - int(round(_FFT_FREQ_LO_HZ / _FFT_RESOLUTION_HZ))
        token_dim = 2 * n_bins  # real + imag

        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(embed_dim, token_num, 1, padding=0, groups=1),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                _PreNorm(token_dim, _ConvAttention(token_num, token_dim, attention_kernel_length, dropout)),
                _PreNorm(token_dim, _FeedForward(token_dim, dropout)),
            ]))
        self.transformer = nn.ModuleList(layers)

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_num, num_classes * 6),
            nn.LayerNorm(num_classes * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes * 6, num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ssvepformer_spectrum(x, self.sampling_rate_hz)  # [B, C, Nfc]
        x = self.to_patch_embedding(x)
        for attn_block, ff_block in self.transformer:
            x = attn_block(x) + x
            x = ff_block(x) + x
        return self.mlp_head(x)
