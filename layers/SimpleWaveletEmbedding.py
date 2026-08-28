"""Stationary Haar-style multi-resolution embedding for iTransformer.

This module is inspired by the representation idea in SimpleTM: expose a
low-frequency component and a high-frequency component before temporal
tokens are inverted into variate tokens.  The transform is fixed and linear;
only the branch gate and embedding are learned.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted


class HaarStyleDecomposition(nn.Module):
    """One-level, length-preserving Haar-style decomposition.

    Pairwise averages and differences are repeated back to the original
    length.  Replicating the final sample for odd lengths avoids dropping
    information and makes the operation usable for every lookback window.
    The returned tensors have shape ``[batch, time, variates]``.
    """

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, variates]")
        length = x.size(1)
        if length == 0:
            raise ValueError("the time dimension must be non-empty")
        if length == 1:
            return x, torch.zeros_like(x)

        even = x[:, 0::2, :]
        odd = x[:, 1::2, :]
        if even.size(1) != odd.size(1):
            odd = torch.cat((odd, even[:, -1:, :]), dim=1)
        low = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        low = low.repeat_interleave(2, dim=1)[:, :length, :]
        detail = detail.repeat_interleave(2, dim=1)[:, :length, :]
        return low, detail


class SimpleWaveletEmbedding(nn.Module):
    """Fuse raw and wavelet views while preserving one token per variate.

    The original inverted embedding is retained for the raw signal and
    timestamp covariates.  A second projection consumes the low/detail pair.
    A data-dependent gate controls the wavelet residual independently for
    every variate, allowing stationary series to rely mostly on the raw
    branch and rapidly changing series to use detail information.
    """

    def __init__(self, seq_len, d_model, embed_type="fixed", freq="h",
                 dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.decomposition = HaarStyleDecomposition()
        self.raw_embedding = DataEmbedding_inverted(
            self.seq_len, d_model, embed_type, freq, dropout
        )
        self.wavelet_embedding = nn.Sequential(
            nn.LayerNorm(2 * self.seq_len),
            nn.Linear(2 * self.seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        # Zero initialization makes the initial gate neutral and avoids a
        # randomly initialized branch overwhelming the pretrained baseline.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.wavelet_gain = nn.Parameter(torch.tensor(0.1))
        self.last_route_weights = None

    def forward(self, x, x_mark=None):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, variates]")
        if x.size(1) != self.seq_len:
            raise ValueError(
                "expected {} time steps, got {}".format(self.seq_len, x.size(1))
            )

        low, detail = self.decomposition(x)
        raw_tokens = self.raw_embedding(x, x_mark)
        wavelet_input = torch.cat((low, detail), dim=1).permute(0, 2, 1)
        wavelet_tokens = self.wavelet_embedding(wavelet_input)

        raw_channel = x.permute(0, 2, 1)
        detail_channel = detail.permute(0, 2, 1)
        stats = torch.stack((
            raw_channel.std(dim=-1, unbiased=False),
            detail_channel.pow(2).mean(dim=-1).sqrt(),
            raw_channel[..., -1] - raw_channel[..., -2] if x.size(1) > 1
            else torch.zeros_like(raw_channel[..., 0]),
        ), dim=-1)
        route = torch.sigmoid(self.router(stats))
        self.last_route_weights = route.detach()
        # DataEmbedding_inverted appends timestamp covariates as extra tokens.
        # They have no observed value stream to decompose, so fuse only the
        # first N variate tokens and preserve covariate tokens unchanged.
        num_variates = x.size(-1)
        variate_tokens = (
            raw_tokens[:, :num_variates, :] +
            self.wavelet_gain * route * wavelet_tokens
        )
        if raw_tokens.size(1) == num_variates:
            return variate_tokens
        return torch.cat((variate_tokens, raw_tokens[:, num_variates:, :]), dim=1)
