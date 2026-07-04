import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Sparse_Variable_Attention import SparseVariableAttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    iTransformer with Sparse Variable Attention.

    This is a natural extension of iTransformer (https://arxiv.org/abs/2310.06625)
    that targets its remaining bottleneck: self-attention is performed over the D
    variates and therefore scales as O(D^2).  When ``configs.sparse_var_attn`` is
    enabled, a lightweight variable-importance estimator + differentiable top-K
    selection is inserted before every self-attention layer so that attention
    complexity drops to O(D * K) (``kv_select``) or O(K^2) (``topk``) with K << D.

    Everything else -- inverted embedding, encoder block structure, FFN,
    LayerNorm, non-stationary normalization, projection head -- is identical to
    iTransformer.  With ``sparse_var_attn`` disabled the model is bit-for-bit the
    original iTransformer, and original iTransformer checkpoints load into the
    shared submodules (the importance estimator is the only freshly-initialized
    component).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Sparse Variable Attention configuration (defaults preserve original
        # iTransformer behavior when the flags are absent).
        self.sparse_var_attn = getattr(configs, 'sparse_var_attn', False)
        self.var_select_reg = getattr(configs, 'var_select_reg', 0.0)

        # Embedding (unchanged)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.class_strategy = configs.class_strategy

        # Encoder: build the attention layer factory according to the flag.
        def make_attention():
            inner = FullAttention(
                False, configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention)
            if self.sparse_var_attn:
                return SparseVariableAttentionLayer(
                    inner, configs.d_model, configs.n_heads,
                    select_k=getattr(configs, 'var_select_k', 0.25),
                    select_mode=getattr(configs, 'var_select_mode', 'kv_select'),
                    estimator=getattr(configs, 'var_select_estimator', 'mlp'),
                    temperature=getattr(configs, 'var_select_temp', 0.1),
                    reg_weight=self.var_select_reg)
            return AttentionLayer(inner, configs.d_model, configs.n_heads)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    make_attention(),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Accumulator for the selection regularization term (consumed by the
        # training loop; 0.0 when sparse attention is disabled).
        self.selection_reg_loss = torch.zeros(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Embedding: B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Sum the selection regularization across encoder layers (0 if disabled).
        if self.sparse_var_attn:
            reg = 0.0
            for layer in self.encoder.attn_layers:
                reg = reg + getattr(layer.attention, 'last_reg', 0.0)
            self.selection_reg_loss = reg

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
