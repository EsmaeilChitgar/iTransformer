"""Multi-scale temporal patch tokenization for inverted Transformers.

The original iTransformer projects one complete lookback window into one token
per variate. This module adds locality and multiple receptive fields before
the variate tokens enter the unchanged cross-variate encoder.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_patch_sizes(patch_sizes: Iterable[int] | str) -> Tuple[int, ...]:
    if isinstance(patch_sizes, str):
        patch_sizes = patch_sizes.split(",")
    sizes = tuple(int(size) for size in patch_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("patch_sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("patch_sizes must not contain duplicates")
    return sizes


class _PatchScaleEncoder(nn.Module):
    """Encode and attentively pool patches at one temporal scale."""

    def __init__(self, patch_size: int, d_model: int, dropout: float):
        super().__init__()
        self.patch_size = patch_size
        self.patch_projection = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Shared depthwise mixing adds local patch-to-patch context cheaply.
        self.local_mixer = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        hidden = max(8, d_model // 4)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        # x: [B, C, T]. Right padding retains the most recent observations in
        # the final patch when T is not divisible by the stride.
        _, _, length = x.shape
        if length < self.patch_size:
            pad = self.patch_size - length
        else:
            remainder = (length - self.patch_size) % stride
            pad = (stride - remainder) % stride
        if pad:
            x = F.pad(x, (0, pad))

        patches = x.unfold(dimension=-1, size=self.patch_size, step=stride)
        batch, channels, num_patches, _ = patches.shape
        patches = patches.contiguous().view(batch * channels, num_patches, self.patch_size)
        patch_tokens = self.norm(self.patch_projection(patches))
        mixed = self.local_mixer(patch_tokens.transpose(1, 2)).transpose(1, 2)
        patch_tokens = self.norm(patch_tokens + self.dropout(F.gelu(mixed)))

        scores = self.pool_score(patch_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * patch_tokens, dim=1)
        return pooled.view(batch, channels, -1)


class MultiScalePatchEmbedding(nn.Module):
    """Convert temporal channels into routed multi-scale variate tokens.

    The router is conditioned on each channel's scale representation, so
    different variates in one sample can select different temporal scales.
    Timestamp or other known covariates are concatenated as additional tokens,
    matching ``DataEmbedding_inverted`` and preserving arbitrary-variate use.
    """

    def __init__(
        self,
        patch_sizes: Iterable[int] | str,
        d_model: int,
        dropout: float = 0.1,
        stride_ratio: float = 0.5,
    ):
        super().__init__()
        if not 0 < stride_ratio <= 1:
            raise ValueError("stride_ratio must be in (0, 1]")
        self.patch_sizes = _as_patch_sizes(patch_sizes)
        self.strides = tuple(
            max(1, int(round(size * stride_ratio))) for size in self.patch_sizes
        )
        self.scales = nn.ModuleList(
            _PatchScaleEncoder(size, d_model, dropout)
            for size in self.patch_sizes
        )
        hidden = max(8, d_model // 4)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Detached diagnostics make scale-utilization analysis reproducible
        # without retaining an autograd graph or changing the forward output.
        self.last_route_weights = None

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, channels]")
        if x_mark is not None:
            if x_mark.ndim != 3 or x_mark.shape[:2] != x.shape[:2]:
                raise ValueError("x_mark must have shape [batch, time, covariates]")
            x = torch.cat([x, x_mark], dim=-1)

        channel_first = x.transpose(1, 2)
        scale_tokens = torch.stack(
            [scale(channel_first, stride) for scale, stride in zip(self.scales, self.strides)],
            dim=2,
        )  # [B, C, num_scales, E]
        route_logits = self.router(scale_tokens).squeeze(-1)
        route_weights = torch.softmax(route_logits, dim=2)
        self.last_route_weights = route_weights.detach()
        tokens = torch.sum(route_weights.unsqueeze(-1) * scale_tokens, dim=2)
        return self.output_norm(self.dropout(tokens))
