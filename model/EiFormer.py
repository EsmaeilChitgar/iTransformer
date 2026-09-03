######1
# import torch
# import torch.nn as nn
#
# from layers.Embed import DataEmbedding_inverted
#
#
# class _BatchNorm2d(nn.Module):
#     def __init__(self):
#         super(_BatchNorm2d, self).__init__()
#         self.bn = nn.BatchNorm2d(1)
#
#     def forward(self, h):
#         h = torch.unsqueeze(h, 1)
#         h = self.bn(h)
#         h = h[:, 0, :, :]
#         return h
#
#
# class _iTrxLayer(nn.Module):
#     def __init__(self, d_ff, d_model, n_heads, dropout):
#         super(_iTrxLayer, self).__init__()
#
#         self.attention = nn.MultiheadAttention(
#             d_model,
#             n_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#
#         key = torch.randn(1, d_ff // 2, d_model)
#         self.key = nn.Parameter(key)
#
#         value = torch.randn(1, d_ff // 2, d_model)
#         self.value = nn.Parameter(value)
#
#     def forward(self, x):
#         key = self.key.expand(x.shape[0], -1, -1)
#         value = self.value.expand(x.shape[0], -1, -1)
#
#         y, _ = self.attention(
#             x,
#             key,
#             value,
#             need_weights=False
#         )
#
#         return y
#
#
# class _EiFormerEncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout):
#         super(_EiFormerEncoderLayer, self).__init__()
#
#         self.att_0 = _iTrxLayer(
#             d_ff,
#             d_model,
#             n_heads,
#             dropout
#         )
#
#         self.att_1 = _iTrxLayer(
#             d_ff,
#             d_model,
#             n_heads,
#             dropout
#         )
#
#         self.lin1 = nn.Linear(
#             d_model,
#             d_model
#         )
#
#         self.lin2 = nn.Linear(
#             d_model,
#             d_model
#         )
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.GELU()
#
#         # Exact RPiT / EiFormer behavior:
#         # first latent-attention key is randomly initialized
#         # and kept frozen.
#         self.att_0.key.requires_grad = False
#
#     def forward(self, x):
#
#         self.att_0.key.requires_grad = False
#
#         h = self.norm1(x)
#
#         h = self.att_0(h)
#
#         h = self.att_1(h)
#
#         h = self.dropout(h)
#
#         x = x + h
#
#         h = self.norm2(x)
#
#         h = self.lin1(h)
#
#         h = self.activation(h)
#
#         h = self.dropout(h)
#
#         h = self.lin2(h)
#
#         h = self.dropout(h)
#
#         y = x + h
#
#         return y
#
#
# class Model(nn.Module):
#
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#
#         # Use the exact embedding already used by official iTransformer.
#         self.enc_embedding = DataEmbedding_inverted(
#             configs.seq_len,
#             configs.d_model,
#             configs.embed,
#             configs.freq,
#             configs.dropout
#         )
#
#         self.encoder_layers = nn.ModuleList([
#             _EiFormerEncoderLayer(
#                 configs.d_model,
#                 configs.n_heads,
#                 configs.d_ff,
#                 configs.dropout
#             )
#             for _ in range(configs.e_layers)
#         ])
#
#         # Same final encoder normalization used by the
#         # official iTransformer encoder.
#         self.encoder_norm = nn.LayerNorm(
#             configs.d_model
#         )
#
#         # Same projector used by the official iTransformer.
#         self.projector = nn.Linear(
#             configs.d_model,
#             configs.pred_len,
#             bias=True
#         )
#
#     def forecast(
#         self,
#         x_enc,
#         x_mark_enc,
#         x_dec,
#         x_mark_dec
#     ):
#
#         if self.use_norm:
#
#             # Same normalization as official iTransformer.
#             means = x_enc.mean(
#                 1,
#                 keepdim=True
#             ).detach()
#
#             x_enc = x_enc - means
#
#             stdev = torch.sqrt(
#                 torch.var(
#                     x_enc,
#                     dim=1,
#                     keepdim=True,
#                     unbiased=False
#                 ) + 1e-5
#             )
#
#             x_enc /= stdev
#
#         _, _, N = x_enc.shape
#
#         # EXACT iTransformer input embedding.
#         #
#         # Input:
#         #   x_enc     = [B, L, N]
#         #   x_mark_enc can contain temporal covariates
#         #
#         # Output:
#         #   [B, N(+covariates), d_model]
#         enc_out = self.enc_embedding(
#             x_enc,
#             x_mark_enc
#         )
#
#         # EiFormer encoder.
#         for layer in self.encoder_layers:
#             enc_out = layer(enc_out)
#
#         enc_out = self.encoder_norm(enc_out)
#
#         # Same projector/output handling as official iTransformer.
#         #
#         # [B, N(+covariates), E]
#         # -> [B, N(+covariates), S]
#         # -> [B, S, N(+covariates)]
#         dec_out = self.projector(enc_out).permute(
#             0,
#             2,
#             1
#         )
#
#         # Remove possible covariate tokens.
#         dec_out = dec_out[
#             :,
#             :,
#             :N
#         ]
#
#         if self.use_norm:
#
#             dec_out = dec_out * (
#                 stdev[:, 0, :]
#                 .unsqueeze(1)
#                 .repeat(
#                     1,
#                     self.pred_len,
#                     1
#                 )
#             )
#
#             dec_out = dec_out + (
#                 means[:, 0, :]
#                 .unsqueeze(1)
#                 .repeat(
#                     1,
#                     self.pred_len,
#                     1
#                 )
#             )
#
#         return dec_out, None
#
#     def forward(
#         self,
#         x_enc,
#         x_mark_enc,
#         x_dec,
#         x_mark_dec,
#         mask=None
#     ):
#
#         dec_out, attns = self.forecast(
#             x_enc,
#             x_mark_enc,
#             x_dec,
#             x_mark_dec
#         )
#
#         if self.output_attention:
#             return (
#                 dec_out[:, -self.pred_len:, :],
#                 attns
#             )
#
#         return dec_out[:, -self.pred_len:, :]
#
#     def get_n_param(self):
#         return sum(
#             torch.numel(param)
#             for param in self.parameters()
#             if param.requires_grad
#         )














#######2
# import torch
# import torch.nn as nn
#
#
# class _BatchNorm2d(nn.Module):
#     def __init__(self):
#         super(_BatchNorm2d, self).__init__()
#         self.bn = nn.BatchNorm2d(1)
#
#     def forward(self, h):
#         h = torch.unsqueeze(h, 1)
#         h = self.bn(h)
#         h = h[:, 0, :, :]
#         return h
#
#
# class _iTrxLayer(nn.Module):
#     def __init__(self, d_ff, d_model, n_heads, dropout):
#         super(_iTrxLayer, self).__init__()
#
#         self.attention = nn.MultiheadAttention(
#             d_model,
#             n_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#
#         key = torch.randn(1, d_ff // 2, d_model)
#         self.key = torch.nn.Parameter(key)
#         self.register_parameter(name='key', param=self.key)
#
#         value = torch.randn(1, d_ff // 2, d_model)
#         self.value = torch.nn.Parameter(value)
#         self.register_parameter(name='value', param=self.value)
#
#     def forward(self, x):
#         key = self.key.expand(x.shape[0], -1, -1)
#         value = self.value.expand(x.shape[0], -1, -1)
#
#         y, _ = self.attention.forward(
#             x,
#             key,
#             value,
#             need_weights=False
#         )
#
#         return y
#
#
# class _EncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, is_ln, dropout):
#         super(_EncoderLayer, self).__init__()
#
#         self.att_0 = _iTrxLayer(
#             d_ff,
#             d_model,
#             n_heads,
#             dropout
#         )
#
#         self.att_1 = _iTrxLayer(
#             d_ff,
#             d_model,
#             n_heads,
#             dropout
#         )
#
#         self.lin1 = nn.Linear(
#             d_model,
#             d_model
#         )
#
#         self.lin2 = nn.Linear(
#             d_model,
#             d_model
#         )
#
#         if is_ln == 1:
#             self.norm1 = nn.LayerNorm(d_model)
#             self.norm2 = nn.LayerNorm(d_model)
#         elif is_ln == 2:
#             self.norm1 = _BatchNorm2d()
#             self.norm2 = _BatchNorm2d()
#
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.GELU()
#
#         self.att_0.key.requires_grad = False
#
#         self.is_ln = is_ln
#
#     def forward(self, x):
#         is_ln = self.is_ln
#
#         self.att_0.key.requires_grad = False
#
#         if is_ln > 0:
#             h = self.norm1(x)
#         else:
#             h = x
#
#         h = self.att_0(h)
#         h = self.att_1(h)
#         h = self.dropout(h)
#
#         x = x + h
#
#         if is_ln > 0:
#             h = self.norm2(x)
#         else:
#             h = x
#
#         h = self.lin1(h)
#         h = self.activation(h)
#         h = self.dropout(h)
#
#         h = self.lin2(h)
#         h = self.dropout(h)
#
#         y = x + h
#
#         return y
#
#
# class Model(nn.Module):
#
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#
#         self.enc_embedding = nn.Linear(
#             configs.seq_len,
#             configs.d_model
#         )
#
#         self.enc_dropout = nn.Dropout(
#             configs.dropout
#         )
#
#         self.encoder_layers = nn.ModuleList([
#             _EncoderLayer(
#                 configs.d_model,
#                 configs.n_heads,
#                 configs.d_ff,
#                 1,
#                 configs.dropout
#             )
#             for _ in range(configs.e_layers)
#         ])
#
#         self.encoder_norm = nn.LayerNorm(
#             configs.d_model
#         )
#
#         self.projector = nn.Linear(
#             configs.d_model,
#             configs.pred_len,
#             bias=True
#         )
#
#     def forecast(
#         self,
#         x_enc,
#         x_mark_enc,
#         x_dec,
#         x_mark_dec
#     ):
#
#         if self.use_norm:
#             means = x_enc.mean(
#                 1,
#                 keepdim=True
#             ).detach()
#
#             x_enc = x_enc - means
#
#             stdev = torch.sqrt(
#                 torch.var(
#                     x_enc,
#                     dim=1,
#                     keepdim=True,
#                     unbiased=False
#                 ) + 1e-5
#             )
#
#             x_enc = x_enc / stdev
#
#         _, _, N = x_enc.shape
#
#         # [B,L,N] -> [B,N,L]
#         enc_out = x_enc.permute(
#             0,
#             2,
#             1
#         )
#
#         enc_out = self.enc_embedding(
#             enc_out
#         )
#
#         enc_out = self.enc_dropout(
#             enc_out
#         )
#
#         for layer in self.encoder_layers:
#             enc_out = layer(enc_out)
#
#         enc_out = self.encoder_norm(
#             enc_out
#         )
#
#         # [B,N,E] -> [B,N,S] -> [B,S,N]
#         dec_out = self.projector(
#             enc_out
#         ).permute(
#             0,
#             2,
#             1
#         )
#
#         dec_out = dec_out[:, :, :N]
#
#         if self.use_norm:
#             dec_out = dec_out * (
#                 stdev[:, 0, :]
#                 .unsqueeze(1)
#                 .repeat(
#                     1,
#                     self.pred_len,
#                     1
#                 )
#             )
#
#             dec_out = dec_out + (
#                 means[:, 0, :]
#                 .unsqueeze(1)
#                 .repeat(
#                     1,
#                     self.pred_len,
#                     1
#                 )
#             )
#
#         return dec_out, None
#
#     def forward(
#         self,
#         x_enc,
#         x_mark_enc,
#         x_dec,
#         x_mark_dec,
#         mask=None
#     ):
#
#         dec_out, attns = self.forecast(
#             x_enc,
#             x_mark_enc,
#             x_dec,
#             x_mark_dec
#         )
#
#         if self.output_attention:
#             return (
#                 dec_out[:, -self.pred_len:, :],
#                 attns
#             )
#
#         return dec_out[:, -self.pred_len:, :]
#
#     def get_n_param(self):
#         n_param = 0
#
#         for param in self.parameters():
#             if param.requires_grad:
#                 n_param += torch.numel(param)
#
#         return n_param











###########3
import torch
import torch.nn as nn


class _iTrxLayer(nn.Module):
    def __init__(self, d_ff, d_model, n_heads, dropout):
        super(_iTrxLayer, self).__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        key = torch.randn(1, d_ff // 2, d_model)
        self.key = nn.Parameter(key)
        self.register_parameter(name='key', param=self.key)

        value = torch.randn(1, d_ff // 2, d_model)
        self.value = nn.Parameter(value)
        self.register_parameter(name='value', param=self.value)

    def forward(self, x):
        key = self.key.expand(x.shape[0], -1, -1)
        value = self.value.expand(x.shape[0], -1, -1)

        y, _ = self.attention(
            x,
            key,
            value,
            need_weights=False
        )

        return y


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(_EncoderLayer, self).__init__()

        self.att_0 = _iTrxLayer(
            d_ff,
            d_model,
            n_heads,
            dropout
        )

        self.att_1 = _iTrxLayer(
            d_ff,
            d_model,
            n_heads,
            dropout
        )

        self.lin1 = nn.Linear(
            d_model,
            d_model
        )

        self.lin2 = nn.Linear(
            d_model,
            d_model
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Exact EiFormer/RPiT behavior.
        self.att_0.key.requires_grad = False

    def forward(self, x):

        self.att_0.key.requires_grad = False

        h = self.norm1(x)

        h = self.att_0(h)
        h = self.att_1(h)

        h = self.dropout(h)

        x = x + h

        h = self.norm2(x)

        h = self.lin1(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.lin2(h)
        h = self.dropout(h)

        y = x + h

        return y


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Same input projection as the EiFormer/RPiT implementation
        # when feat_dim = 0.
        self.enc_embedding = nn.Linear(
            configs.seq_len,
            configs.d_model
        )

        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(
                configs.d_model,
                configs.n_heads,
                configs.d_ff,
                configs.dropout
            )
            for _ in range(configs.e_layers)
        ])

        self.encoder_norm = nn.LayerNorm(
            configs.d_model
        )

        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len
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
                dim=1,
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

        # [B,L,N] -> [B,N,L]
        enc_out = x_enc.permute(
            0,
            2,
            1
        )

        # [B,N,L] -> [B,N,d_model]
        enc_out = self.enc_embedding(
            enc_out
        )

        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        enc_out = self.encoder_norm(
            enc_out
        )

        # [B,N,d_model] -> [B,N,pred_len]
        enc_out = self.projector(
            enc_out
        )

        # [B,N,pred_len] -> [B,pred_len,N]
        dec_out = enc_out.permute(
            0,
            2,
            1
        )

        # Keep only original variates.
        dec_out = dec_out[:, :, :N]

        if self.use_norm:

            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)

            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, None

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
            return dec_out[:, -self.pred_len:, :], attns

        return dec_out[:, -self.pred_len:, :]

    def get_n_param(self):
        return sum(
            torch.numel(param)
            for param in self.parameters()
            if param.requires_grad
        )