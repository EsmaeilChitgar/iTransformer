"""Parameter-efficient latent cross-variate attention for iTransformer."""

import torch
import torch.nn as nn


class LatentVariableAttentionLayer(nn.Module):
    """Replace dense variate attention with a learned N-to-M-to-N bottleneck.

    Shared Q/K/V projections are used in both directions so the layer adds
    only ``M * d_model`` latent parameters plus one LayerNorm over the original
    AttentionLayer projection budget.
    """

    def __init__(self, d_model, n_heads, num_latents=32, dropout=0.1,
                 output_attention=False, full_attention_threshold=64, factor=1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if full_attention_threshold < 0:
            raise ValueError("full_attention_threshold cannot be negative")

        self.num_latents = int(num_latents)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.output_attention = output_attention
        self.full_attention_threshold = int(full_attention_threshold)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, self.num_latents, d_model) * (d_model ** -0.5)
        )
        # One shared projection set keeps parameter growth small and makes the
        # two cross-attention directions structurally symmetric.
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
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

    def attention_score_count(self, num_variates):
        """Return the number of pairwise score entries for reporting."""
        if num_variates <= self.full_attention_threshold:
            return num_variates * num_variates
        return 2 * num_variates * self.num_latents
