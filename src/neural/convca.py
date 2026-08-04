"""
ConvCA — Convolutional Correlation Analysis network (Pan et al., 2024).

ConvCA is a two-input model: it takes a single trial and a
per-class template (mean of training trials for that class).
The output is a correlation-based score per class.

Because the forward pass needs templates for *all* classes to produce
logits, the model exposes `forward_with_templates(x, templates)`.
The standard `forward(x)` interface requires templates to be
registered via `set_templates()` before inference.

Input:  x [B, T, C], templates [Nf, T, C]
Output: logits [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCA(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_classes: int,
        dropout_sig: float = 0.75,
        dropout_temp: float = 0.15,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self._templates = None

        Nc = embed_dim
        Nt = seq_len
        Nf = num_classes

        self.sig_conv1 = nn.Conv2d(1, 16, kernel_size=(9, Nc), padding="same")
        self.sig_conv2 = nn.Conv2d(16, 1, kernel_size=(1, Nc), padding="same")
        self.sig_conv3 = nn.Conv2d(1, 1, kernel_size=(1, Nc), padding="valid")
        self.sig_dropout = nn.Dropout(dropout_sig)

        self.temp_conv1 = nn.Conv2d(Nc, 40, kernel_size=(9, 1), padding="same")
        self.temp_conv2 = nn.Conv2d(40, 1, kernel_size=(9, 1), padding="same")
        self.temp_dropout = nn.Dropout(dropout_temp)

        self.dense = nn.Linear(Nf, Nf)

    def set_templates(self, templates_ftc: torch.Tensor) -> None:
        """Register per-class templates [Nf, T, C] for use in forward()."""
        self._templates = templates_ftc

    def _encode_signal(self, sig: torch.Tensor) -> torch.Tensor:
        """sig: [B, 1, T, C] -> [B, 1, T, 1]"""
        sig = self.sig_conv1(sig)
        sig = self.sig_conv2(sig)
        sig = self.sig_conv3(sig)
        sig = self.sig_dropout(sig)
        return sig

    def _encode_template(self, temp: torch.Tensor) -> torch.Tensor:
        """temp: [B, C, T, Nf] -> [B, 1, T, Nf]"""
        temp = self.temp_conv1(temp)
        temp = self.temp_conv2(temp)
        temp = self.temp_dropout(temp)
        return temp

    def _correlate(self, sig_enc: torch.Tensor, temp_enc: torch.Tensor) -> torch.Tensor:
        """
        sig_enc:  [B, 1, T, 1]
        temp_enc: [B, 1, T, Nf]
        Returns:  [B, Nf]
        """
        x = sig_enc.squeeze(1)  # [B, T, 1]
        t = temp_enc.squeeze(1)  # [B, T, Nf]

        corr_xt = torch.einsum("btk,btl->blk", x, t)  # [B, Nf, 1]
        corr_xx = torch.einsum("btk,btk->bk", x, x)  # [B, 1]
        corr_tt = torch.einsum("btl,btl->bl", t, t)  # [B, Nf]

        corr = corr_xt.squeeze(-1) / (torch.sqrt(corr_tt) * torch.sqrt(corr_xx) + 1e-8)
        return corr

    def forward_with_templates(self, x_btc: torch.Tensor, templates_ftc: torch.Tensor) -> torch.Tensor:
        """
        x_btc:         [B, T, C]
        templates_ftc:  [Nf, T, C]
        Returns:        [B, Nf] logits
        """
        B = x_btc.shape[0]
        Nf = templates_ftc.shape[0]

        sig = x_btc.unsqueeze(1)  # [B, 1, T, C]

        temp = templates_ftc.permute(2, 1, 0)  # [C, T, Nf]
        temp = temp.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, T, Nf]

        sig_enc = self._encode_signal(sig)
        temp_enc = self._encode_template(temp)
        corr = self._correlate(sig_enc, temp_enc)
        return self.dense(corr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._templates is None:
            raise RuntimeError("Call set_templates() before forward(), or use forward_with_templates().")
        templates = self._templates.to(x.device)
        return self.forward_with_templates(x, templates)
