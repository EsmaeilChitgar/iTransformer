import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.n_heads = configs.n_heads
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy

        # Runtime-only diagnostic mode.
        # This stays OFF during normal training/testing.
        self.diagnostic_active = False
        self.diagnostic_samples = getattr(configs, 'rank_diagnostic_samples', 1)
        self.last_layer_outputs = []
        self.last_attentions = []
        self.rank_ablation_layerwise = getattr(configs, 'rank_ablation_layerwise', False)

        layer_rank_text = getattr(configs, 'rank_ablation_layer_ranks', '')
        self.rank_ablation_layer_ranks = [
            int(x.strip()) for x in layer_rank_text.split(',')
            if x.strip()
        ]

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def set_diagnostic_mode(self, enabled=True):
        """
        Enable/disable runtime diagnostic capture.

        This does not change the mathematical model during normal
        training. Attention matrices are captured only while this
        mode is active.
        """
        self.diagnostic_active = enabled

        for encoder_layer in self.encoder.attn_layers:
            inner_attention = encoder_layer.attention.inner_attention

            if hasattr(inner_attention, 'capture_attention'):
                inner_attention.capture_attention = enabled

    def set_rank_ablation(
            self,
            rank=0,
            layer_ranks=None,
            head_ranks=None
    ):
        """
        Configure attention rank ablation.

        rank=0:
            Always restore original full attention.

        rank>0:
            Global mode:
                same rank for every layer.

            Layer-wise mode:
                one rank per encoder layer.

            Head-wise mode:
                one list of ranks per encoder layer.
                Each list contains one rank per head.

        Examples
        --------
        Full:

            set_rank_ablation(0)

        Global R=8:

            set_rank_ablation(8)

        Layer-wise:

            set_rank_ablation(
                8,
                layer_ranks=[6, 11, 9, 6]
            )

        Head-wise:

            set_rank_ablation(
                8,
                head_ranks=[
                    [8, 7, 10, 9, 8, 6, 9, 7],
                    [7, 8, 11, 9, 8, 7, 9, 7],
                    [6, 8, 12, 10, 7, 8, 10, 7],
                    [7, 7, 10, 9, 8, 7, 10, 7]
                ]
            )
        """

        num_layers = len(
            self.encoder.attn_layers
        )

        # -------------------------------------------------------------
        # Full attention MUST always win.
        # -------------------------------------------------------------

        if rank == 0:

            for layer in self.encoder.attn_layers:
                inner_attention = (
                    layer
                    .attention
                    .inner_attention
                )

                inner_attention.rank_ablation = 0

                inner_attention.head_rank_ablation = None

            return

        # -------------------------------------------------------------
        # HEAD-WISE MODE
        # -------------------------------------------------------------

        if head_ranks is not None:

            head_ranks = [
                list(layer_ranks_for_heads)
                for layer_ranks_for_heads in head_ranks
            ]

            if len(head_ranks) != num_layers:
                raise ValueError(
                    f'Expected {num_layers} layer-wise head-rank '
                    f'lists, got {len(head_ranks)}'
                )

            expected_heads = int(
                self.n_heads
            )

            for layer_idx, ranks_for_layer in enumerate(
                    head_ranks
            ):

                if len(ranks_for_layer) != expected_heads:
                    raise ValueError(
                        f'Layer {layer_idx + 1}: '
                        f'expected {expected_heads} head ranks, '
                        f'got {len(ranks_for_layer)}: '
                        f'{ranks_for_layer}'
                    )

                for r in ranks_for_layer:

                    if int(r) <= 0:
                        raise ValueError(
                            f'Head-wise ranks must be > 0. '
                            f'Layer {layer_idx + 1}: '
                            f'{ranks_for_layer}'
                        )

            # ---------------------------------------------------------
            # Apply
            # ---------------------------------------------------------

            for i, layer in enumerate(
                    self.encoder.attn_layers
            ):
                inner_attention = (
                    layer
                    .attention
                    .inner_attention
                )

                inner_attention.rank_ablation = 0

                inner_attention.head_rank_ablation = [
                    int(r)
                    for r in head_ranks[i]
                ]

            return

        # -------------------------------------------------------------
        # LAYER-WISE MODE
        # -------------------------------------------------------------

        if layer_ranks is not None:

            layer_ranks = list(
                layer_ranks
            )

        # -------------------------------------------------------------
        # GLOBAL MODE
        # -------------------------------------------------------------

        else:

            layer_ranks = [
                rank
                for _ in range(num_layers)
            ]

        # -------------------------------------------------------------
        # Validate number of layers
        # -------------------------------------------------------------

        if len(layer_ranks) != num_layers:
            raise ValueError(
                f'Expected {num_layers} layer ranks, '
                f'got {len(layer_ranks)}: '
                f'{layer_ranks}'
            )

        # -------------------------------------------------------------
        # Apply layer/global ranks
        # -------------------------------------------------------------

        for i, layer in enumerate(
                self.encoder.attn_layers
        ):
            inner_attention = (
                layer
                .attention
                .inner_attention
            )

            inner_attention.rank_ablation = int(
                layer_ranks[i]
            )

            inner_attention.head_rank_ablation = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if not self.diagnostic_active:
            # Keep the original execution path completely unchanged.
            enc_out, attns = self.encoder(enc_out, attn_mask=None)

        else:
            # ---------------------------------------------------------
            # Diagnostic execution
            # ---------------------------------------------------------
            self.last_layer_outputs = []
            self.last_attentions = []

            for encoder_layer in self.encoder.attn_layers:

                enc_out, attn = encoder_layer(
                    enc_out,
                    attn_mask=None
                )

                # Keep only the requested number of samples.
                samples = min(
                    self.diagnostic_samples,
                    enc_out.shape[0]
                )

                self.last_layer_outputs.append(
                    enc_out[:samples].detach().cpu()
                )

                if attn is None:
                    self.last_attentions.append(None)
                else:
                    self.last_attentions.append(
                        attn[:samples].detach().cpu()
                    )

            if self.encoder.norm is not None:
                enc_out = self.encoder.norm(enc_out)

            attns = self.last_attentions

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]