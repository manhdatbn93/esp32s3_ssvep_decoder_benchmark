"""
C-CNN (frequency-domain CNN) baseline for 12JFPM (Ravi et al., 2020).

Operates on complex spectrum features: time-domain EEG is transformed
via FFT into real + imaginary frequency-bin features before being fed
to two Conv2d layers and a linear classifier.

FFT parameters used by the CCNN spectral frontend:
  resolution = 0.2930 Hz/bin -> NFFT = round(fs / resolution)
  Frequency range: 3-35 Hz -> 110 bins -> 220 features (real || imag)

Input:  [B, T, C]  (e.g. [B, 512, 8])
Output: logits [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FFT_RESOLUTION_HZ: float = 0.2930
_FFT_FREQ_LO_HZ: float = 3.0
_FFT_FREQ_HI_HZ: float = 35.0


def _complex_spectrum_features(
    x_btc: torch.Tensor,
    sampling_rate_hz: float,
) -> torch.Tensor:
    """Convert [B, T, C] time-domain EEG to [B, 1, C, Nfc] spectrum features."""
    _b, t, _c = x_btc.shape
    nfft = round(sampling_rate_hz / _FFT_RESOLUTION_HZ)
    idx_lo = round(_FFT_FREQ_LO_HZ / _FFT_RESOLUTION_HZ)
    idx_hi = round(_FFT_FREQ_HI_HZ / _FFT_RESOLUTION_HZ) + 1

    x_bct = x_btc.transpose(1, 2)
    spectrum = torch.fft.fft(x_bct, n=nfft, dim=-1) / (t / 2.0)
    real = spectrum[..., idx_lo:idx_hi].real
    imag = spectrum[..., idx_lo:idx_hi].imag
    features = torch.cat([real, imag], dim=-1).float()
    return features.unsqueeze(1)


class CCNN(nn.Module):
    NFC: int = 220

    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        sampling_rate_hz: float = 256.0,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.sampling_rate_hz = sampling_rate_hz

        nch = embed_dim
        F = 2 * nch
        D = self.NFC - 10 + 1  # feature-map length after freq conv

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, F, kernel_size=(nch, 1), stride=(nch, 1)),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(F, F, kernel_size=(1, 10), stride=(1, 1)),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F * D, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _complex_spectrum_features(x, self.sampling_rate_hz)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.classifier(x)
