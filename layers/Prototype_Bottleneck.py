import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PrototypeBottleneck(nn.Module):
    """
    Perceiver-style latent bottleneck for inverted (variate-wise) Transformers.

    Replaces full variate-wise self-attention, whose cost grows quadratically
    with the number of variables D, with a compact latent reasoning path:

        Stage A (construction): K learnable latent queries cross-attend to the
            D variable tokens, producing K sample-adaptive prototype tokens
            P in R^{B x K x d}  (K << D).
        Stage B (reasoning):    standard Transformer encoder blocks run on the
            prototypes P (cost is O(K^2) instead of O(D^2)).
        Stage C (return):       the original variable tokens cross-attend back
            to the prototypes and the result is fused via a residual
            connection whose final projection is zero-initialized, so the
            whole module acts as an exact identity mapping at the start of
            training. This preserves variable-specific detail and keeps
            early training stable.

    The number of input/output tokens is unchanged (D in, D out), so the
    downstream projector and training pipeline are untouched.

    Args:
        d_model:      token embedding dimension.
        n_heads:      number of attention heads (shared across the three
                      attention modules).
        n_prototypes: K, the number of latent prototype tokens. Should be
                      much smaller than the number of variables D.
        proto_layers: number of self-attention encoder layers run on the
                      prototypes (Stage B).
        d_ff:         feed-forward width inside the prototype encoder.
        dropout:      dropout rate.
        activation:   'relu' or 'gelu'.
        share_attn:   if True, the Stage-A and Stage-C cross-attention
                      projections are tied (parameter saving); otherwise they
                      are independent.
    """

    def __init__(self, d_model, n_heads, n_prototypes, proto_layers,
                 d_ff=None, dropout=0.1, activation="gelu", share_attn=False):
        super(PrototypeBottleneck, self).__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes

        # Learnable latent queries (sample-adaptive prototypes are produced by
        # cross-attending these fixed queries to the variable tokens).
        self.latent_queries = nn.Parameter(
            torch.empty(1, n_prototypes, d_model))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        def make_attn():
            return AttentionLayer(
                FullAttention(False, factor=1, attention_dropout=dropout,
                              output_attention=False),
                d_model, n_heads)

        # Stage A: latent queries (Q) cross-attend to variable tokens (K, V).
        self.to_prototype = make_attn()

        # Stage B: self-attention over the prototypes.
        self.prototype_encoder = Encoder(
            [
                EncoderLayer(
                    make_attn(),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(proto_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Stage C: variable tokens (Q) cross-attend to prototypes (K, V).
        self.to_variables = make_attn() if not share_attn else self.to_prototype

        # CRITICAL: zero-initialize the final projection immediately before the
        # residual addition. This makes Stage C an exact identity at init
        # (X_out = X_in + 0), preventing early gradient instability and letting
        # the model smoothly learn to integrate the prototype context.
        nn.init.zeros_(self.to_variables.out_projection.weight)
        if self.to_variables.out_projection.bias is not None:
            nn.init.zeros_(self.to_variables.out_projection.bias)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [B, D, d_model] variable tokens (D = number of variates).
        Returns:
            x_out: [B, D, d_model] variable tokens enriched with prototype
                   context (identity at initialization).
            attns: list of attention maps from the prototype encoder.
        """
        B, D, _ = x.shape

        # --- Stage A: build sample-adaptive prototypes ---------------------
        q = self.latent_queries.expand(B, -1, -1)          # [B, K, d]
        proto, _ = self.to_prototype(q, x, x, attn_mask=None)  # [B, K, d]
        proto = q + proto                                   # residual on queries

        # --- Stage B: global reasoning over the prototypes -----------------
        proto, attns = self.prototype_encoder(proto, attn_mask=None)  # [B, K, d]

        # --- Stage C: fuse prototype context back into variables -----------
        # Variables query the prototypes; the (zero-init) projection + residual
        # guarantees identity behavior at the start of training.
        context, _ = self.to_variables(x, proto, proto, attn_mask=None)  # [B, D, d]
        x_out = x + context

        return x_out, attns


class PrototypeEncoder(nn.Module):
    """
    Drop-in replacement for the iTransformer `Encoder` that routes variable
    tokens through a stack of `PrototypeBottleneck` blocks. Each block
    compresses to prototypes, reasons, and returns to D variable tokens, so
    the interface (D tokens in, D tokens out) matches the original encoder
    and the prediction head is unchanged.

    A final LayerNorm mirrors the original Encoder's norm layer.
    """

    def __init__(self, d_model, n_heads, n_prototypes, e_layers,
                 d_ff=None, dropout=0.1, activation="gelu", share_attn=False):
        super(PrototypeEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            PrototypeBottleneck(
                d_model=d_model,
                n_heads=n_heads,
                n_prototypes=n_prototypes,
                proto_layers=1,  # one self-attention layer per block
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                share_attn=share_attn,
            ) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for block in self.blocks:
            x, attn = block(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns
