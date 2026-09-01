import torch
import torch.nn as nn


class EiMAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EiMAttention, self).__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        key = torch.randn(1, d_ff, d_model)
        self.key = torch.nn.Parameter(key)
        self.register_parameter(name='key', param=self.key)

        value = torch.randn(1, d_ff, d_model)
        self.value = torch.nn.Parameter(value)
        self.register_parameter(name='value', param=self.value)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        key = self.key.expand(queries.shape[0], -1, -1)
        value = self.value.expand(queries.shape[0], -1, -1)

        out, _ = self.attention(
            queries,
            key,
            value,
            need_weights=False
        )

        return out, None