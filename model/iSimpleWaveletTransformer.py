"""SimpleTM-inspired wavelet/geometric iTransformer."""

import torch
import torch.nn as nn

from layers.GeometricAttention import GeometricAttentionLayer
from layers.SimpleWaveletEmbedding import SimpleWaveletEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """Forecast with multi-resolution variate tokens and geometric attention.

    The temporal decomposition is linear in the lookback length. Attention
    remains dense over variates, matching the baseline complexity class while
    adding a geometric inductive bias for cross-variate representations.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geometric_weight = float(
            getattr(configs, "wavelet_geometric_weight", 0.25)
        )
        self.enc_embedding = SimpleWaveletEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    GeometricAttentionLayer(
                        configs.d_model,
                        configs.n_heads,
                        dropout=configs.dropout,
                        output_attention=configs.output_attention,
                        geometric_weight=self.geometric_weight,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, num_variates = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        dec_out = dec_out[:, :, :num_variates]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
