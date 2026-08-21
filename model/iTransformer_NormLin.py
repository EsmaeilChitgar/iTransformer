import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.NormLin import NormLinMultiHead


class NormLinEncoder(nn.Module):

    def __init__(
        self,
        e_layers,
        d_model,
        d_ff,
        dropout,
        activation,
        token_num,
        n_heads
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            NormLinMultiHead(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                token_num=token_num,
                n_heads=n_heads
            )
            for _ in range(e_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # --------------------------------------------------
        # Same embedding as iTransformer
        # --------------------------------------------------

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # --------------------------------------------------
        # x_mark dimension
        #
        # Standard iTransformer time-feature dimensions
        # --------------------------------------------------

        freq_map = {
            'h': 4,
            't': 5,
            's': 6,
            'm': 1,
            'a': 1,
            'w': 2,
            'd': 3,
            'b': 3
        }

        mark_dim = freq_map.get(
            configs.freq,
            0
        )

        # iTransformer's inverted embedding:
        #
        # [variables] + [x_mark features]
        #
        token_num = configs.enc_in + mark_dim

        self.encoder = NormLinEncoder(
            e_layers=configs.e_layers,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            token_num=token_num,
            n_heads=configs.n_heads
        )

        # Same projector as original iTransformer
        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len,
            bias=True
        )

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec
    ):

        # --------------------------------------------------
        # Same normalization as original iTransformer
        # --------------------------------------------------

        if self.use_norm:

            means = x_enc.mean(
                1,
                keepdim=True
            ).detach()

            x_enc = x_enc - means

            stdev = torch.sqrt(
                torch.var(
                    x_enc,
                    dim=1,
                    keepdim=True,
                    unbiased=False
                ) + 1e-5
            )

            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # --------------------------------------------------
        # IMPORTANT:
        # Keep x_mark.
        #
        # DataEmbedding_inverted will turn:
        #
        # x + x_mark
        #
        # into tokens.
        # --------------------------------------------------

        enc_out = self.enc_embedding(
            x_enc,
            x_mark_enc
        )

        # [B, N(+covariates), D]
        enc_out = self.encoder(
            enc_out
        )

        # Same projector
        dec_out = self.projector(
            enc_out
        ).permute(
            0,
            2,
            1
        )[:, :, :N]

        # --------------------------------------------------
        # Same de-normalization
        # --------------------------------------------------

        if self.use_norm:

            dec_out = (
                dec_out
                * stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(
                    1,
                    self.pred_len,
                    1
                )
            )

            dec_out = (
                dec_out
                + means[:, 0, :]
                .unsqueeze(1)
                .repeat(
                    1,
                    self.pred_len,
                    1
                )
            )

        return dec_out, None

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        dec_out, _ = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        return dec_out[
            :,
            -self.pred_len:,
            :
        ]