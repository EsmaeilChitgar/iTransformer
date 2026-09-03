import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.LatentBottleneck import LatentBottleneck


class Model(nn.Module):

    def __init__(self, configs):

        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # ---------------------------------------------------------
        # Original iTransformer embedding
        # ---------------------------------------------------------

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # ---------------------------------------------------------
        # Latent bottleneck
        # ---------------------------------------------------------

        self.latent_bottleneck = LatentBottleneck(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            num_latents=configs.num_latents,
            latent_d_ff=configs.latent_d_ff,
            num_latent_blocks=configs.num_latent_blocks,
            dropout=configs.dropout
        )

        # ---------------------------------------------------------
        # ORIGINAL iTransformer encoder
        # ---------------------------------------------------------

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
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(
                configs.d_model
            )
        )

        # ---------------------------------------------------------
        # Original projector
        # ---------------------------------------------------------

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

            x_enc /= stdev

        _, _, N = x_enc.shape

        # ---------------------------------------------------------
        # Original embedding
        #
        # [B,L,N] -> [B,N,D]
        # ---------------------------------------------------------

        tokens = self.enc_embedding(
            x_enc,
            x_mark_enc
        )

        # Keep original variate representations as decoder queries.
        #
        # This is important:
        # each original variate keeps its own identity.
        queries = tokens

        # ---------------------------------------------------------
        # Compression
        #
        # [B,N,D] -> [B,K,D]
        # ---------------------------------------------------------

        latent = self.latent_bottleneck.encode(
            tokens
        )

        # ---------------------------------------------------------
        # ORIGINAL iTransformer Encoder
        #
        # [B,K,D] -> [B,K,D]
        # ---------------------------------------------------------

        enc_out, attns = self.encoder(
            latent,
            attn_mask=None
        )

        # ---------------------------------------------------------
        # Reconstruction
        #
        # [B,N,D] queries
        # [B,K,D] latent keys/values
        #
        # -> [B,N,D]
        # ---------------------------------------------------------

        enc_out = self.latent_bottleneck.decode(
            queries,
            enc_out
        )

        # ---------------------------------------------------------
        # Original iTransformer projection
        #
        # [B,N,D]
        # ->
        # [B,N,H]
        # ->
        # [B,H,N]
        # ---------------------------------------------------------

        dec_out = self.projector(
            enc_out
        ).permute(0, 2, 1)

        # Remove possible covariates.
        dec_out = dec_out[:, :, :N]

        # ---------------------------------------------------------
        # De-normalization
        # ---------------------------------------------------------

        if self.use_norm:

            dec_out = dec_out * (
                stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(
                    1,
                    self.pred_len,
                    1
                )
            )

            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(
                    1,
                    self.pred_len,
                    1
                )
            )

        return dec_out, attns

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        dec_out, attns = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        if self.output_attention:

            return (
                dec_out[:, -self.pred_len:, :],
                attns
            )

        return dec_out[:, -self.pred_len:, :]