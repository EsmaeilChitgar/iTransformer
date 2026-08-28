"""TimeMixer-inspired temporal mixing plus latent iTransformer attention."""

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.Latent_Variable_Attention import LatentVariableAttentionLayer
from layers.TemporalScaleMixer import TemporalScaleMixer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """Decompose temporal dynamics cheaply, then model variate relations.

    The temporal mixer is linear in the lookback length for fixed scales. The
    encoder uses latent cross-variate attention for high-dimensional datasets,
    while retaining dense attention for small token sets.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.num_latents = int(getattr(configs, "num_latents", 32))
        self.full_attention_threshold = int(
            getattr(configs, "latent_full_attention_threshold", 64)
        )
        self.temporal_scales = getattr(configs, "temporal_scales", "3,7,15")

        self.temporal_mixer = TemporalScaleMixer(self.temporal_scales)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LatentVariableAttentionLayer(
                        configs.d_model,
                        configs.n_heads,
                        num_latents=self.num_latents,
                        dropout=configs.dropout,
                        output_attention=configs.output_attention,
                        full_attention_threshold=self.full_attention_threshold,
                        factor=configs.factor,
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
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, num_variates = x_enc.shape
        x_enc = self.temporal_mixer(x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :num_variates]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        return dec_out[:, -self.pred_len :, :]
