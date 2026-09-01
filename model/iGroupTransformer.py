import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.VariateGrouping import VariateGrouping, VariateUngrouping


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        n_groups = configs.n_groups

        self.grouping = VariateGrouping(
            configs.d_model,
            n_groups,
            configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.ungrouping = VariateUngrouping()

        self.local_adapter = nn.Sequential(
            nn.Linear(configs.d_model, configs.group_adapter_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.group_adapter_dim, configs.d_model)
        )

        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len,
            bias=True
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(
                    x_enc,
                    dim=1,
                    keepdim=True,
                    unbiased=False
                ) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        group_tokens, assignment = self.grouping(enc_out)

        group_tokens, attns = self.encoder(
            group_tokens,
            attn_mask=None
        )

        group_context = self.ungrouping(
            group_tokens,
            assignment
        )

        enc_out = enc_out + self.local_adapter(group_context)

        dec_out = self.projector(enc_out).permute(
            0, 2, 1
        )[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]