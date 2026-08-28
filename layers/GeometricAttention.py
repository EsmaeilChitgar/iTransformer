"""SimpleTM-inspired geometric self-attention for variate tokens."""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


class GeometricAttention(nn.Module):
    """Dense attention with a dot/wedge geometric similarity term.

    For normalized query/key vectors, the wedge-product magnitude is
    ``sqrt(1 - cosine(query, key)^2)``.  The score rewards aligned vectors
    with ``cosine + weight * (1 - wedge_magnitude)``.  This is computed from
    the identity above, so no explicit exterior-product tensor is created.

    The operation is still dense O(N^2) in the number of variate tokens.  Its
    purpose is representation quality, not a claim of quadratic complexity
    reduction.
    """

    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1,
                 output_attention=False, geometric_weight=0.25):
        super().__init__()
        if geometric_weight < 0:
            raise ValueError("geometric_weight must be non-negative")
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.geometric_weight = float(geometric_weight)
        self.dropout = nn.Dropout(attention_dropout)

    @staticmethod
    def _mask_scores(scores, attn_mask):
        mask = attn_mask.mask if hasattr(attn_mask, "mask") else attn_mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return scores.masked_fill(mask, float("-inf"))

    def forward(self, queries, keys, values, attn_mask=None, tau=None,
                delta=None):
        batch, length, heads, depth = queries.shape
        _, source_length, _, value_depth = values.shape
        if depth != keys.size(-1):
            raise ValueError("query and key dimensions must match")

        query_norm = queries / queries.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        key_norm = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cosine = torch.einsum("blhe,bshe->bhls", query_norm, key_norm)
        wedge = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0) + 1e-6)
        similarity = cosine + self.geometric_weight * (1.0 - wedge)
        scores = similarity * (self.scale or 1.0 / sqrt(depth))

        if attn_mask is not None:
            scores = self._mask_scores(scores, attn_mask)
        elif self.mask_flag:
            raise ValueError("causal geometric attention requires an explicit mask")

        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        output = output.contiguous()
        return output, attention if self.output_attention else None


class GeometricAttentionLayer(nn.Module):
    """Projection wrapper matching the repository's attention interface."""

    def __init__(self, d_model, n_heads, dropout=0.1,
                 output_attention=False, geometric_weight=0.25):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        depth = d_model // n_heads
        self.inner_attention = GeometricAttention(
            mask_flag=False,
            attention_dropout=dropout,
            output_attention=output_attention,
            geometric_weight=geometric_weight,
        )
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = depth

    def forward(self, queries, keys, values, attn_mask=None, tau=None,
                delta=None):
        batch, length, _ = queries.shape
        source_length = keys.size(1)
        query = self.query_projection(queries).view(
            batch, length, self.n_heads, self.head_dim
        )
        key = self.key_projection(keys).view(
            batch, source_length, self.n_heads, self.head_dim
        )
        value = self.value_projection(values).view(
            batch, source_length, self.n_heads, self.head_dim
        )
        output, attention = self.inner_attention(
            query, key, value, attn_mask=attn_mask, tau=tau, delta=delta
        )
        return self.out_projection(output.view(batch, length, -1)), attention
