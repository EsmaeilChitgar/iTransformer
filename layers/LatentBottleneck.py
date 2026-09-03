import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key_value):

        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        output, _ = self.attention(
            q,
            kv,
            kv,
            need_weights=False
        )

        query = query + self.dropout1(output)

        query = query + self.dropout2(
            self.ffn(self.norm2(query))
        )

        return query


class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):

        y = self.norm1(x)

        y, _ = self.attention(
            y,
            y,
            y,
            need_weights=False
        )

        x = x + self.dropout1(y)

        x = x + self.dropout2(
            self.ffn(self.norm2(x))
        )

        return x


class LatentBottleneck(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        num_latents,
        latent_d_ff,
        num_latent_blocks=1,
        dropout=0.1
    ):
        super().__init__()

        self.num_latents = num_latents
        self.d_model = d_model

        self.latent_array = nn.Parameter(
            torch.randn(
                1,
                num_latents,
                d_model
            )
        )

        # N -> K
        self.compress = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=latent_d_ff,
            dropout=dropout
        )

        # K -> K
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=latent_d_ff,
                dropout=dropout
            )
            for _ in range(num_latent_blocks)
        ])

        # N -> K decoder
        #
        # This module is used AFTER the original iTransformer
        # encoder.
        self.reconstruct = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=latent_d_ff,
            dropout=dropout
        )

    def encode(self, x):

        batch_size = x.size(0)

        latent = self.latent_array.expand(
            batch_size,
            -1,
            -1
        )

        # N -> K
        latent = self.compress(
            latent,
            x
        )

        # K -> K
        for block in self.latent_blocks:
            latent = block(latent)

        return latent

    def decode(self, queries, latent):

        # N queries attend to K latent tokens
        return self.reconstruct(
            queries,
            latent
        )