"""Latent cross-variate attention for high-dimensional iTransformers.

iTransformer makes every variate a token, so dense self-attention has a
quadratic cost in the number of variables. This layer uses a small learned
set of latent tokens as an information bottleneck:

    variates -> latent tokens -> variates

The two cross-attention operations cost O(N*M + M*N) instead of O(N*N), where
N is the number of variates and M is the number of learned latent tokens.
For small N, a full-attention fallback preserves the original inductive bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentVariableAttentionLayer(nn.Module):
    """Drop-in encoder attention with a learned latent variate bottleneck.

    Args:
        d_model: Transformer embedding dimension.
        n_heads: number of attention heads.
        num_latents: M, the number of learned latent tokens.
        dropout: attention and latent residual dropout.
        output_attention: return both latent-to-variate and variate-to-latent
            maps when true.
        full_attention_threshold: use exact full attention when the total
            token count is at most this value. Set to 0 to disable fallback.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_latents: int = 32,
        dropout: float = 0.1,
        output_attention: bool = False,
        full_attention_threshold: int = 64,
        factor: int = 1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if full_attention_threshold < 0:
            raise ValueError("full_attention_threshold cannot be negative")

        self.num_latents = int(num_latents)
        self.output_attention = output_attention
        self.full_attention_threshold = int(full_attention_threshold)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, self.num_latents, d_model) * (d_model ** -0.5)
        )
        # Share projections between the two cross-attention directions. This
        # keeps the parameter count close to the original AttentionLayer,
        # instead of paying for two independent MHA modules.
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latent_norm = nn.LayerNorm(d_model)
        self.latent_dropout = nn.Dropout(dropout)

        self.last_attention_mode = None

    def _project(self, x, projection):
        batch, length, _ = x.shape
        return projection(x).view(batch, length, self.n_heads, self.head_dim)

    def _attention(self, query, key, value, attn_mask=None):
        scores = torch.einsum("blhe,bshe->bhls", query, key)
        scores = scores * (self.head_dim ** -0.5)
        if attn_mask is not None:
            mask = attn_mask.mask if hasattr(attn_mask, "mask") else attn_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("bhls,bshd->blhd", weights, value)
        return output.contiguous(), weights

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch, num_tokens, _ = queries.shape
        query = self._project(queries, self.query_projection)
        if num_tokens <= self.full_attention_threshold:
            self.last_attention_mode = "full"
            key = self._project(keys, self.key_projection)
            value = self._project(values, self.value_projection)
            output, attention = self._attention(query, key, value, attn_mask)
            output = self.out_projection(output.view(batch, num_tokens, -1))
            return output, attention if self.output_attention else None

        latent = self.latent_tokens.expand(batch, -1, -1)
        latent_query = self._project(latent, self.query_projection)
        key = self._project(keys, self.key_projection)
        value = self._project(values, self.value_projection)
        latent_update, latent_attention = self._attention(latent_query, key, value)
        latent_update = latent_update.view(batch, self.num_latents, -1)
        latent = self.latent_norm(latent + self.latent_dropout(latent_update))

        latent_key = self._project(latent, self.key_projection)
        latent_value = self._project(latent, self.value_projection)
        output, variate_attention = self._attention(query, latent_key, latent_value)
        output = self.out_projection(output.view(batch, num_tokens, -1))
        self.last_attention_mode = "latent"
        if self.output_attention:
            return output, (latent_attention, variate_attention)
        return output, None

    def attention_score_count(self, num_variates: int) -> int:
        """Return the pairwise score count used by the active route."""
        if num_variates <= self.full_attention_threshold:
            return num_variates * num_variates
        return 2 * num_variates * self.num_latents
