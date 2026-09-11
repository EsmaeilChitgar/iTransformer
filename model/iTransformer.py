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

    Original iTransformer architecture with optional profiling.

    Profiling only adds measurement/labels.
    It does not intentionally change the model computation.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # ============================================================
        # PROFILING CONFIGURATION
        # ============================================================

        self.enable_profiling = getattr(
            configs,
            'enable_profiling',
            False
        )

        self.profile_skip_steps = getattr(
            configs,
            'profile_skip_steps',
            2
        )

        self.profile_warmup_steps = getattr(
            configs,
            'profile_warmup_steps',
            1
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

        # ============================================================
        # Profiler internal state
        # ============================================================

        self._profiler = None
        self._profiler_step_count = 0
        self._profiler_finished = False

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
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ============================================================
        # Profiling metadata
        #
        # IMPORTANT:
        # self.encoder MUST be created before accessing it.
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

            # AttentionLayer
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

                # FullAttention
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

    # ================================================================
    # Profiling helper
    # ================================================================

    def _profile_range(self, name):

        if self.enable_profiling:
            return record_function(name)

        return nullcontext()

    # ================================================================
    # Start profiler
    # ================================================================

    def _start_profiler(self):

        if not self.enable_profiling:
            return

        if self._profiler is not None:
            return

        if self._profiler_finished:
            return

        os.makedirs(
            self.profile_trace_dir,
            exist_ok=True
        )

        # ------------------------------------------------------------
        # Activities
        # ------------------------------------------------------------

        activities = [
            ProfilerActivity.CPU
        ]

        if torch.cuda.is_available():
            activities.append(
                ProfilerActivity.CUDA
            )

        # ------------------------------------------------------------
        # Schedule
        #
        # wait    -> ignore
        # warmup  -> profiler warms up but does not save results
        # active  -> actual recorded steps
        # ------------------------------------------------------------

        schedule = torch.profiler.schedule(
            wait=self.profile_skip_steps,
            warmup=self.profile_warmup_steps,
            active=self.profile_active_steps,
            repeat=1
        )

        # ------------------------------------------------------------
        # Trace handler
        # ------------------------------------------------------------

        if self.profile_save_trace:

            trace_handler = (
                torch.profiler.tensorboard_trace_handler(
                    self.profile_trace_dir
                )
            )

        else:

            trace_handler = None

        # ------------------------------------------------------------
        # Profiler
        # ------------------------------------------------------------

        self._profiler = profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=self.profile_record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.profile_with_stack,
            with_flops=self.profile_with_flops
        )

        self._profiler.__enter__()

        self._profiler_step_count = 0

        print(
            '\n'
            + '=' * 100
        )

        print(
            'iTRANSFORMER PROFILER STARTED'
        )

        print(
            '=' * 100
        )

        print(
            f'Skip steps   : '
            f'{self.profile_skip_steps}'
        )

        print(
            f'Warmup steps : '
            f'{self.profile_warmup_steps}'
        )

        print(
            f'Active steps : '
            f'{self.profile_active_steps}'
        )

        print(
            f'Trace dir    : '
            f'{self.profile_trace_dir}'
        )

        print(
            '=' * 100
            + '\n'
        )

    # ================================================================
    # Profiler step
    # ================================================================

    def _profiler_step(self):

        if self._profiler is None:
            return

        self._profiler.step()

        self._profiler_step_count += 1

        # Number of calls needed before the scheduled profile is done.
        total_steps = (
            self.profile_skip_steps
            + self.profile_warmup_steps
            + self.profile_active_steps
        )

        if (
            self._profiler_step_count
            >= total_steps
        ):

            self._stop_profiler()

    # ================================================================
    # Stop profiler and print summary
    # ================================================================

    def _stop_profiler(self):

        if self._profiler is None:
            return

        print(
            '\n'
            + '=' * 100
        )

        print(
            'iTRANSFORMER PROFILER SUMMARY'
        )

        print(
            '=' * 100
        )

        try:

            # CUDA time when CUDA is available.
            # Otherwise CPU time.
            if torch.cuda.is_available():

                sort_key = (
                    'self_cuda_time_total'
                )

            else:

                sort_key = (
                    'self_cpu_time_total'
                )

            print(
                self._profiler.key_averages(
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

            print(e)

            try:

                print(
                    self._profiler.key_averages().table(
                        sort_by='self_cpu_time_total',
                        row_limit=self.profile_top_ops
                    )
                )

            except Exception as e2:

                print(
                    '[Profiler] Fallback summary failed:'
                )

                print(e2)

        print(
            '=' * 100
        )

        print(
            'Profiler trace directory: '
            f'{self.profile_trace_dir}'
        )

        print(
            '=' * 100
            + '\n'
        )

        # ------------------------------------------------------------
        # Close profiler
        # ------------------------------------------------------------

        self._profiler.__exit__(
            None,
            None,
            None
        )

        self._profiler = None
        self._profiler_finished = True

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

        # ============================================================
        # Start profiler on first forward
        # ============================================================

        if (
            self.enable_profiling
            and not self._profiler_finished
            and self._profiler is None
        ):

            self._start_profiler()

        # ============================================================
        # Input information
        # ============================================================

        if (
            self.enable_profiling
            and self.profile_print_model_info
            and self._profiler_step_count == 0
        ):

            print(
                '\n'
                + '=' * 100
            )

            print(
                'iTRANSFORMER INPUT'
            )

            print(
                '=' * 100
            )

            print(
                f'x_enc shape      : '
                f'{tuple(x_enc.shape)}'
            )

            print(
                f'x_enc dtype      : '
                f'{x_enc.dtype}'
            )

            print(
                f'x_enc device     : '
                f'{x_enc.device}'
            )

            if x_mark_enc is not None:

                print(
                    f'x_mark_enc shape : '
                    f'{tuple(x_mark_enc.shape)}'
                )

            print(
                f'use_norm         : '
                f'{self.use_norm}'
            )

            print(
                f'encoder layers   : '
                f'{len(self.encoder.attn_layers)}'
            )

            if len(self.encoder.attn_layers) > 0:

                print(
                    f'd_model          : '
                    f'{self.encoder.attn_layers[0].norm1.normalized_shape[0]}'
                )

                if hasattr(
                    self.encoder.attn_layers[0],
                    'conv1'
                ):

                    print(
                        f'FFN d_ff         : '
                        f'{self.encoder.attn_layers[0].conv1.out_channels}'
                    )

                if hasattr(
                    self.encoder.attn_layers[0],
                    'attention'
                ):

                    if hasattr(
                        self.encoder.attn_layers[0].attention,
                        'n_heads'
                    ):

                        print(
                            f'n_heads          : '
                            f'{self.encoder.attn_layers[0].attention.n_heads}'
                        )

            print(
                '=' * 100
                + '\n'
            )

        # ============================================================
        # Normalization
        # ============================================================

        with self._profile_range(
            'iTransformer::Normalization'
        ):

            if self.use_norm:

                # Normalization from Non-stationary Transformer
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
        #
        # B L N -> B N E
        # ============================================================

        with self._profile_range(
            'iTransformer::Embedding'
        ):

            enc_out = self.enc_embedding(
                x_enc,
                x_mark_enc
            )

        if (
            self.enable_profiling
            and self.profile_print_shapes
            and self._profiler_step_count == 0
        ):

            print(
                f'Embedding output : '
                f'{tuple(enc_out.shape)}'
            )

        # ============================================================
        # Encoder
        #
        # B N E -> B N E
        # ============================================================

        with self._profile_range(
            'iTransformer::Encoder'
        ):

            enc_out, attns = self.encoder(
                enc_out,
                attn_mask=None
            )

        if (
            self.enable_profiling
            and self.profile_print_shapes
            and self._profiler_step_count == 0
        ):

            print(
                f'Encoder output   : '
                f'{tuple(enc_out.shape)}'
            )

        # ============================================================
        # Projection
        #
        # B N E -> B N S -> B S N
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
            self.enable_profiling
            and self.profile_print_shapes
            and self._profiler_step_count == 0
        ):

            print(
                f'Projection output: '
                f'{tuple(dec_out.shape)}'
            )

        # ============================================================
        # De-Normalization
        # ============================================================

        with self._profile_range(
            'iTransformer::DeNormalization'
        ):

            if self.use_norm:

                # De-Normalization from Non-stationary Transformer
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

        # ============================================================
        # Advance profiler by one model forward
        # ============================================================

        if (
            self.enable_profiling
            and not self._profiler_finished
        ):

            self._profiler_step()

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