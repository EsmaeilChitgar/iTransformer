import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prototype_Bottleneck import PrototypeEncoder
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    iPrototype: iTransformer with a Perceiver-style latent prototype
    bottleneck.

    The inverted Transformer treats each variable as a token, which makes
    self-attention cost quadratic in the number of variables D. This model
    keeps the iTransformer embedding, normalization and prediction head, but
    replaces the variate-wise encoder with `PrototypeEncoder`: each block
    compresses the D variable tokens into K << D learnable prototype tokens,
    runs self-attention on the prototypes, and fuses the prototype context
    back into the original variables through a zero-initialized residual
    fusion (identity at the start of training).

    The original `iTransformer` model file is left untouched for fair
    side-by-side comparison; select this model with `--model iPrototype`.

    Paper link (backbone): https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding (identical to iTransformer)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # Number of latent prototypes K. Default to a small fraction of the
        # number of variables when not provided.
        n_prototypes = getattr(configs, "n_prototypes", None)
        if n_prototypes in (None, 0, -1):
            # heuristic: min(enc_in, 64); capped so K << D on high-dim data.
            n_prototypes = max(8, min(getattr(configs, "enc_in", 64), 64))

        # Latent bottleneck encoder (replaces the variate-wise encoder).
        self.encoder = PrototypeEncoder(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_prototypes=n_prototypes,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            share_attn=getattr(configs, "proto_share_attn", False),
        )
        self.n_prototypes = n_prototypes

        # Prediction head (identical to iTransformer)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # B N E -> B N E  (latent prototype bottleneck)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
