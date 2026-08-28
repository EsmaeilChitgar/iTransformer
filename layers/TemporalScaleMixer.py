"""Lightweight TimeMixer-inspired temporal decomposition before inversion."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_scales(scales: Iterable[int] | str) -> Tuple[int, ...]:
    if isinstance(scales, str):
        scales = scales.split(",")
    values = tuple(int(scale) for scale in scales)
    if not values or any(scale <= 0 or scale % 2 == 0 for scale in values):
        raise ValueError("temporal scales must be positive odd integers")
    if len(set(values)) != len(values):
        raise ValueError("temporal scales must not contain duplicates")
    return values


class TemporalScaleMixer(nn.Module):
    """Route trend/detail components from several fixed temporal scales.

    Moving averages and residual details are computed per variate with no
    cross-variate mixing. A tiny router uses per-variate statistics to choose
    among the identity, trend, and detail branches. The output retains the
    original ``[batch, time, variates]`` shape for DataEmbedding_inverted.
    """

    def __init__(self, scales: Iterable[int] | str = "3,7,15"):
        super().__init__()
        self.scales = _parse_scales(scales)
        num_branches = 1 + 2 * len(self.scales)
        hidden = max(8, num_branches * 2)
        self.router = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_branches),
        )
        self.last_route_weights = None

    @staticmethod
    def _moving_average(x, kernel_size):
        padding = kernel_size // 2
        x = F.pad(x, (padding, padding), mode="replicate")
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, variates]")
        channel_first = x.transpose(1, 2)
        branches = [channel_first]
        for scale in self.scales:
            trend = self._moving_average(channel_first, scale)
            branches.extend((trend, channel_first - trend))

        branch_stack = torch.stack(branches, dim=2)  # [B, N, branches, L]
        mean = channel_first.mean(dim=-1)
        std = channel_first.std(dim=-1, unbiased=False)
        last_delta = channel_first[..., -1] - channel_first[..., -2]
        statistics = torch.stack((mean, std, last_delta), dim=-1)
        route_weights = torch.softmax(self.router(statistics), dim=-1)
        self.last_route_weights = route_weights.detach()
        mixed = torch.sum(route_weights.unsqueeze(-1) * branch_stack, dim=2)
        return mixed.transpose(1, 2)
