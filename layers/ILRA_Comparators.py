"""Research reference adaptations of ISAB and non-causal Luna to inverted forecasting.

References
----------
ISAB: https://github.com/juho-lee/set_transformer/blob/master/modules.py
Luna: https://proceedings.neurips.cc/paper/2021/file/14319d9cfc6123106878dc20b94fbaf3-Paper.pdf

Do not describe these as byte-for-byte copies of the authors' forecasting code:
the original studies do NOT supply iTransformer forecasting baselines.
ISAB implementation mirrors the original MAB equations (including sqrt(d_model)).
Luna mirrors equations (3)--(6) and propagates the contextual packed sequence.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SetTransformerMAB(nn.Module):
    """PyTorch-equivalent rewrite of the official Set Transformer MAB.

    The reference code uses q + softmax(q @ k.T / sqrt(dim_V)) @ v,
    followed by an optional pair of LayerNorms and relu(fc_o(...)) residual.
    Dropout is deliberately not added: official MAB has none.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads, ln=False):
        super().__init__()
        if dim_v % num_heads:
            raise ValueError('MAB dim_v must be divisible by num_heads')
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)
        self.ln0 = nn.LayerNorm(dim_v) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_v) if ln else nn.Identity()

    def forward(self, q, k):
        b, nq, _ = q.shape
        nk = k.shape[1]
        h = self.num_heads
        dh = self.dim_v // h
        q = self.fc_q(q)
        kh, vh = self.fc_k(k), self.fc_v(k)
        # Official code concatenates the heads on batch axis. These operations
        # are equivalent in exact arithmetic, with less error-prone reshaping.
        qh = q.reshape(b, nq, h, dh).transpose(1, 2)
        kh = kh.reshape(b, nk, h, dh).transpose(1, 2)
        vh = vh.reshape(b, nk, h, dh).transpose(1, 2)
        a = torch.softmax((qh @ kh.transpose(-2, -1)) / math.sqrt(self.dim_v), dim=-1)
        o = (qh + a @ vh).transpose(1, 2).contiguous().reshape(b, nq, self.dim_v)
        o = self.ln0(o)
        o = o + F.relu(self.fc_o(o))
        return self.ln1(o)


class ISABEncoderLayer(nn.Module):
    """Replace one iTransformer encoder layer with the full reference ISAB.

    Standard variant: both MAB stages see all variate and time tokens.
    Specialized bypass variant: induce only from variates; raw time tokens
    appear directly as uncompressed context for MAB1. This bypass is a
    *forecasting adaptation* and not part of the original ISAB paper.
    """
    def __init__(self, d_model, n_heads, rank, layernorm=False, time_tokens=0, bypass=False):
        super().__init__()
        if rank <= 0 or time_tokens < 0:
            raise ValueError('invalid rank or time_tokens')
        self.bypass = bool(bypass)
        self.time_tokens = int(time_tokens)
        self.inducing_points = nn.Parameter(torch.empty(1, rank, d_model))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab0 = SetTransformerMAB(d_model, d_model, d_model, n_heads, ln=layernorm)
        self.mab1 = SetTransformerMAB(d_model, d_model, d_model, n_heads, ln=layernorm)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if attn_mask is not None:
            raise ValueError('ISAB comparison is non-causal only')
        if self.bypass and self.time_tokens:
            if x.size(1) <= self.time_tokens:
                raise ValueError('time token count exceeds input token count')
            vars_, time = x[:, :-self.time_tokens], x[:, -self.time_tokens:]
            h = self.mab0(self.inducing_points.expand(x.shape[0], -1, -1), vars_)
            result = self.mab1(x, torch.cat((h, time), dim=1))
        else:
            h = self.mab0(self.inducing_points.expand(x.shape[0], -1, -1), x)
            result = self.mab1(x, h)
        return result, None


class LunaEncoderLayer(nn.Module):
    """Luna pack/unpack plus contextual P residual and token FFN.

    The source paper uses two MHA operations per layer and propagates the
    packed sequence to the next layer. We do not claim exact equivalence to
    its highly optimized task-specific fairseq implementation.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu',
                 time_tokens=0, bypass=False):
        super().__init__()
        self.time_tokens = int(time_tokens)
        self.bypass = bool(bypass)
        self.pack = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.unpack = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, p):
        if self.bypass and self.time_tokens:
            if x.size(1) <= self.time_tokens:
                raise ValueError('time token count exceeds input token count')
            context = x[:, :-self.time_tokens]
            time = x[:, -self.time_tokens:]
        else:
            context, time = x, None
        # Paper's Y_P is used by unpack before residual+normalization.
        yp, _ = self.pack(p, context, context, need_weights=False)
        compact = torch.cat((yp, time), dim=1) if time is not None else yp
        yx, _ = self.unpack(x, compact, compact, need_weights=False)
        pa = self.norm_p(p + self.dropout(yp))
        xa = self.norm_x1(x + self.dropout(yx))
        ff = self.ff2(self.dropout(self.activation(self.ff1(xa))))
        x_out = self.norm_x2(xa + self.dropout(ff))
        return x_out, pa


class LunaEncoder(nn.Module):
    """Encoder interface compatible with iTransformer: return (X, attentions)."""
    def __init__(self, d_model, n_heads, d_ff, num_layers, rank,
                 dropout=0.1, activation='gelu', time_tokens=0, bypass=False):
        super().__init__()
        if rank <= 0 or num_layers < 1:
            raise ValueError('invalid rank / num_layers')
        positions = torch.arange(rank, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        initial = torch.zeros(rank, d_model)
        initial[:, 0::2] = torch.sin(positions * div)
        initial[:, 1::2] = torch.cos(positions * div[:initial[:, 1::2].shape[-1]])
        self.initial_p = nn.Parameter(initial.unsqueeze(0))
        self.layers = nn.ModuleList([
            LunaEncoderLayer(d_model, n_heads, d_ff, dropout, activation,
                             time_tokens=time_tokens, bypass=bypass)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if attn_mask is not None:
            raise ValueError('Luna comparison is non-causal only')
        p = self.initial_p.expand(x.size(0), -1, -1)
        for layer in self.layers:
            x, p = layer(x, p)
        return self.norm(x), [None] * len(self.layers)
