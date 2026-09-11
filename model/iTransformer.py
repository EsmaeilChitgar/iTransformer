import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625

    Original iTransformer architecture with optional
    forward-only profiling.

    Profiling does not intentionally change the model
    computation. It only measures selected forward passes.
    """

    def __init__(self, configs):

        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # ============================================================
        # Profiling configuration
        # ============================================================

        self.enable_profiling = getattr(
            configs,
            'enable_profiling',
            False
        )

        self.profile_skip_steps = getattr(
            configs,
            'profile_skip_steps',
            3
        )

        self.profile_warmup_steps = getattr(
            configs,
            'profile_warmup_steps',
            2
        )

        self.profile_active_steps = getattr(
            configs,
            'profile_active_steps',
            5
        )

        self.profile_save_trace = getattr(
            configs,
            'profile_save_trace',
            True
        )

        self.profile_trace_dir = getattr(
            configs,
            'profile_trace_dir',
            './profiling'
        )

        self.profile_record_shapes = getattr(
            configs,
            'profile_record_shapes',
            True
        )

        self.profile_memory = getattr(
            configs,
            'profile_memory',
            True
        )

        self.profile_with_stack = getattr(
            configs,
            'profile_with_stack',
            False
        )

        self.profile_with_flops = getattr(
            configs,
            'profile_with_flops',
            True
        )

        self.profile_top_ops = getattr(
            configs,
            'profile_top_ops',
            50
        )

        self.profile_print_model_info = getattr(
            configs,
            'profile_print_model_info',
            True
        )

        self.profile_print_shapes = getattr(
            configs,
            'profile_print_shapes',
            True
        )

        # ------------------------------------------------------------
        # Forward call counters
        # ------------------------------------------------------------

        self._forward_call_count = 0
        self._profile_active_count = 0

        # ============================================================
        # Embedding
        # ============================================================

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        self.class_strategy = configs.class_strategy

        # ============================================================
        # Encoder-only architecture
        # ============================================================

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
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(
                configs.d_model
            )
        )

        # ============================================================
        # Profiling metadata
        # ============================================================

        self.encoder.enable_profiling = (
            self.enable_profiling
        )

        for layer_id, encoder_layer in enumerate(
            self.encoder.attn_layers
        ):

            encoder_layer.enable_profiling = (
                self.enable_profiling
            )

            encoder_layer.profile_layer_id = (
                layer_id
            )

            if hasattr(
                encoder_layer,
                'attention'
            ):

                encoder_layer.attention.enable_profiling = (
                    self.enable_profiling
                )

                encoder_layer.attention.profile_layer_id = (
                    layer_id
                )

                if hasattr(
                    encoder_layer.attention,
                    'inner_attention'
                ):

                    encoder_layer.attention.inner_attention.enable_profiling = (
                        self.enable_profiling
                    )

                    encoder_layer.attention.inner_attention.profile_layer_id = (
                        layer_id
                    )

        # ============================================================
        # Projection
        # ============================================================

        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len,
            bias=True
        )

        # ============================================================
        # Profiling directory
        # ============================================================

        if self.enable_profiling:

            os.makedirs(
                self.profile_trace_dir,
                exist_ok=True
            )

            print(
                '[Profiler] Forward-only profiling enabled.'
            )

    # ================================================================
    # Profiling range
    # ================================================================

    def _profile_range(self, name):

        if self._currently_profiling:

            return record_function(name)

        return nullcontext()

    # ================================================================
    # Should this forward be profiled?
    # ================================================================

    def _should_profile_this_forward(self):

        if not self.enable_profiling:
            return False

        # The forward counter starts from 1.
        current_step = self._forward_call_count

        profile_start = (
            self.profile_skip_steps
            + self.profile_warmup_steps
            + 1
        )

        profile_end = (
            self.profile_skip_steps
            + self.profile_warmup_steps
            + self.profile_active_steps
        )

        return (
            profile_start
            <= current_step
            <= profile_end
        )

    # ================================================================
    # Print model information
    # ================================================================

    def _print_profile_input_info(
        self,
        x_enc,
        x_mark_enc
    ):

        print(
            '\n'
            + '=' * 100
        )

        print(
            'iTRANSFORMER FORWARD PROFILE'
        )

        print(
            '=' * 100
        )

        print(
            f'Forward call       : '
            f'{self._forward_call_count}'
        )

        print(
            f'Active profile     : '
            f'{self._profile_active_count}'
        )

        print(
            f'x_enc shape        : '
            f'{tuple(x_enc.shape)}'
        )

        print(
            f'x_enc dtype        : '
            f'{x_enc.dtype}'
        )

        print(
            f'x_enc device       : '
            f'{x_enc.device}'
        )

        if x_mark_enc is not None:

            print(
                f'x_mark_enc shape   : '
                f'{tuple(x_mark_enc.shape)}'
            )

        print(
            f'use_norm           : '
            f'{self.use_norm}'
        )

        print(
            f'encoder layers     : '
            f'{len(self.encoder.attn_layers)}'
        )

        if len(self.encoder.attn_layers) > 0:

            layer0 = (
                self.encoder.attn_layers[0]
            )

            print(
                f'd_model            : '
                f'{layer0.norm1.normalized_shape[0]}'
            )

            if hasattr(layer0, 'conv1'):

                print(
                    f'd_ff               : '
                    f'{layer0.conv1.out_channels}'
                )

            if hasattr(
                layer0,
                'attention'
            ):

                print(
                    f'n_heads            : '
                    f'{layer0.attention.n_heads}'
                )

        print(
            '=' * 100
            + '\n'
        )

    # ================================================================
    # Create a new profiler for ONE forward pass
    # ================================================================

    def _create_forward_profiler(self):

        activities = [
            ProfilerActivity.CPU
        ]

        if torch.cuda.is_available():

            activities.append(
                ProfilerActivity.CUDA
            )

        return profile(
            activities=activities,
            record_shapes=self.profile_record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.profile_with_stack,
            with_flops=self.profile_with_flops
        )

    # ================================================================
    # Save and print profiler results for one forward
    # ================================================================

    def _finish_forward_profile(
        self,
        profiler_instance
    ):

        print(
            '\n'
            + '=' * 100
        )

        print(
            'FORWARD-ONLY PROFILER SUMMARY'
        )

        print(
            '=' * 100
        )

        # ------------------------------------------------------------
        # Prefer CUDA timing when available.
        # ------------------------------------------------------------

        if torch.cuda.is_available():

            sort_key = (
                'self_cuda_time_total'
            )

        else:

            sort_key = (
                'self_cpu_time_total'
            )

        try:

            print(
                profiler_instance.key_averages(
                    group_by_input_shape=(
                        self.profile_record_shapes
                    )
                ).table(
                    sort_by=sort_key,
                    row_limit=self.profile_top_ops
                )
            )

        except Exception as e:

            print(
                '[Profiler] Detailed summary failed:'
            )

            print(
                repr(e)
            )

            try:

                print(
                    profiler_instance.key_averages().table(
                        sort_by='self_cpu_time_total',
                        row_limit=self.profile_top_ops
                    )
                )

            except Exception as e2:

                print(
                    '[Profiler] Fallback summary failed:'
                )

                print(
                    repr(e2)
                )

        # ------------------------------------------------------------
        # Save trace
        # ------------------------------------------------------------

        if self.profile_save_trace:

            trace_path = os.path.join(
                self.profile_trace_dir,
                (
                    'forward_profile_'
                    f'{self._profile_active_count:02d}.json'
                )
            )

            try:

                profiler_instance.export_chrome_trace(
                    trace_path
                )

                print(
                    f'Forward trace saved: '
                    f'{trace_path}'
                )

            except Exception as e:

                print(
                    '[Profiler] Could not save Chrome trace:'
                )

                print(
                    repr(e)
                )

        print(
            '=' * 100
            + '\n'
        )

    # ================================================================
    # Normalization
    # ================================================================

    def _forward_impl(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec
    ):

        # ============================================================
        # Normalization
        # ============================================================

        with self._profile_range(
            'iTransformer::Normalization'
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

        # ============================================================
        # Embedding
        # ============================================================

        with self._profile_range(
            'iTransformer::Embedding'
        ):

            enc_out = self.enc_embedding(
                x_enc,
                x_mark_enc
            )

        if (
            self._currently_profiling
            and self.profile_print_shapes
        ):

            print(
                f'Embedding output  : '
                f'{tuple(enc_out.shape)}'
            )

        # ============================================================
        # Encoder
        # ============================================================

        with self._profile_range(
            'iTransformer::Encoder'
        ):

            enc_out, attns = self.encoder(
                enc_out,
                attn_mask=None
            )

        if (
            self._currently_profiling
            and self.profile_print_shapes
        ):

            print(
                f'Encoder output    : '
                f'{tuple(enc_out.shape)}'
            )

        # ============================================================
        # Projection
        # ============================================================

        with self._profile_range(
            'iTransformer::Projection'
        ):

            dec_out = self.projector(
                enc_out
            ).permute(
                0,
                2,
                1
            )[:, :, :N]

        if (
            self._currently_profiling
            and self.profile_print_shapes
        ):

            print(
                f'Projection output : '
                f'{tuple(dec_out.shape)}'
            )

        # ============================================================
        # De-Normalization
        # ============================================================

        with self._profile_range(
            'iTransformer::DeNormalization'
        ):

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

    # ================================================================
    # Forecast
    # ================================================================

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec
    ):

        # ------------------------------------------------------------
        # Count every forward call
        # ------------------------------------------------------------

        self._forward_call_count += 1

        # ------------------------------------------------------------
        # Decide whether this forward should be profiled
        # ------------------------------------------------------------

        profile_this_forward = (
            self._should_profile_this_forward()
        )

        # ------------------------------------------------------------
        # If not profiling this forward:
        # run the original computation directly.
        # ------------------------------------------------------------

        if not profile_this_forward:

            self._currently_profiling = False

            return self._forward_impl(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

        # ------------------------------------------------------------
        # Active profiling forward
        # ------------------------------------------------------------

        self._profile_active_count += 1

        self._currently_profiling = True

        if (
            self.profile_print_model_info
        ):

            self._print_profile_input_info(
                x_enc,
                x_mark_enc
            )

        profiler_instance = (
            self._create_forward_profiler()
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        # profiler starts immediately before forward computation
        # and stops immediately after forward computation.
        #
        # Therefore:
        #
        #     forward
        #       only
        #
        # is captured.
        #
        # backward() and optimizer.step() happen after this method
        # returns, so they are NOT captured.
        # ------------------------------------------------------------

        profiler_instance.__enter__()

        try:

            dec_out, attns = self._forward_impl(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

        finally:

            self._currently_profiling = False

            profiler_instance.__exit__(
                None,
                None,
                None
            )

        self._finish_forward_profile(
            profiler_instance
        )

        return dec_out, attns

    # ================================================================
    # Forward
    # ================================================================

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

        else:

            return dec_out[:, -self.pred_len:, :]