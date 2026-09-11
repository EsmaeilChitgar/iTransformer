import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from torch.profiler import record_function


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()

        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular'
        )

        self.norm = nn.BatchNorm1d(c_in)

        self.activation = nn.ELU()

        self.maxPool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):

        with record_function(
            "ConvLayer::downConv"
        ):
            x = self.downConv(
                x.permute(0, 2, 1)
            )

        with record_function(
            "ConvLayer::BatchNorm"
        ):
            x = self.norm(x)

        with record_function(
            "ConvLayer::Activation"
        ):
            x = self.activation(x)

        with record_function(
            "ConvLayer::MaxPool"
        ):
            x = self.maxPool(x)

        x = x.transpose(1, 2)

        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu"
    ):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = attention

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.activation = (
            F.relu
            if activation == "relu"
            else F.gelu
        )

        # ====================================================
        # Profiling state
        # ====================================================

        self.enable_profiling = False
        self.profile_layer_id = -1

    # ========================================================
    # Profiling helper
    # ========================================================

    def _profile_range(self, name):

        if self.enable_profiling:
            return record_function(name)

        return nullcontext()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        attn_mask=None,
        tau=None,
        delta=None
    ):

        # ====================================================
        # Attention
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::Attention'
        ):

            new_x, attn = self.attention(
                x,
                x,
                x,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta
            )

        # ====================================================
        # Attention residual + dropout
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::AttentionResidual'
        ):

            x = x + self.dropout(
                new_x
            )

        # ====================================================
        # First LayerNorm
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::Norm1'
        ):

            y = x = self.norm1(x)

        # ====================================================
        # FFN Conv1
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFN_Conv1'
        ):

            y = self.conv1(
                y.transpose(-1, 1)
            )

        # ====================================================
        # FFN Activation
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFN_Activation'
        ):

            y = self.activation(y)

        # ====================================================
        # FFN Dropout 1
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFN_Dropout1'
        ):

            y = self.dropout(y)

        # ====================================================
        # FFN Conv2
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFN_Conv2'
        ):

            y = self.conv2(y)

        # ====================================================
        # Transpose + Dropout 2
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFN_Dropout2'
        ):

            y = y.transpose(-1, 1)

            y = self.dropout(y)

        # ====================================================
        # FFN residual
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::FFNResidual'
        ):

            y = x + y

        # ====================================================
        # Second LayerNorm
        # ====================================================

        with self._profile_range(
            f'EncoderLayer[{self.profile_layer_id}]::Norm2'
        ):

            output = self.norm2(y)

        return output, attn


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers=None,
        norm_layer=None
    ):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(
            attn_layers
        )

        self.conv_layers = (
            nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )

        self.norm = norm_layer

        # ====================================================
        # Profiling state
        # ====================================================

        self.enable_profiling = False

    def forward(
        self,
        x,
        attn_mask=None,
        tau=None,
        delta=None
    ):

        # x [B, L, D]

        attns = []

        # ====================================================
        # Encoder with ConvLayer
        # ====================================================

        if self.conv_layers is not None:

            for i, (
                attn_layer,
                conv_layer
            ) in enumerate(
                zip(
                    self.attn_layers,
                    self.conv_layers
                )
            ):

                delta = (
                    delta
                    if i == 0
                    else None
                )

                # ------------------------------------------------
                # Pass profiling metadata
                # ------------------------------------------------

                if hasattr(
                    attn_layer,
                    'enable_profiling'
                ):

                    attn_layer.enable_profiling = (
                        self.enable_profiling
                    )

                if hasattr(
                    attn_layer,
                    'profile_layer_id'
                ):

                    attn_layer.profile_layer_id = i

                if hasattr(
                    attn_layer,
                    'attention'
                ):

                    attn_layer.attention.enable_profiling = (
                        self.enable_profiling
                    )

                    attn_layer.attention.profile_layer_id = i

                    if hasattr(
                        attn_layer.attention,
                        'inner_attention'
                    ):

                        attn_layer.attention.inner_attention.enable_profiling = (
                            self.enable_profiling
                        )

                        attn_layer.attention.inner_attention.profile_layer_id = i

                # ------------------------------------------------
                # Attention
                # ------------------------------------------------

                with record_function(
                    f'Encoder::Layer_{i}'
                ):

                    x, attn = attn_layer(
                        x,
                        attn_mask=attn_mask,
                        tau=tau,
                        delta=delta
                    )

                # ------------------------------------------------
                # ConvLayer
                # ------------------------------------------------

                with record_function(
                    f'Encoder::ConvLayer_{i}'
                ):

                    x = conv_layer(x)

                attns.append(attn)

            # ----------------------------------------------------
            # Final attention layer
            # ----------------------------------------------------

            final_layer_id = (
                len(self.attn_layers) - 1
            )

            final_attn_layer = (
                self.attn_layers[-1]
            )

            if hasattr(
                final_attn_layer,
                'enable_profiling'
            ):

                final_attn_layer.enable_profiling = (
                    self.enable_profiling
                )

            if hasattr(
                final_attn_layer,
                'profile_layer_id'
            ):

                final_attn_layer.profile_layer_id = (
                    final_layer_id
                )

            if hasattr(
                final_attn_layer,
                'attention'
            ):

                final_attn_layer.attention.enable_profiling = (
                    self.enable_profiling
                )

                final_attn_layer.attention.profile_layer_id = (
                    final_layer_id
                )

                if hasattr(
                    final_attn_layer.attention,
                    'inner_attention'
                ):

                    final_attn_layer.attention.inner_attention.enable_profiling = (
                        self.enable_profiling
                    )

                    final_attn_layer.attention.inner_attention.profile_layer_id = (
                        final_layer_id
                    )

            with record_function(
                f'Encoder::Layer_{final_layer_id}'
            ):

                x, attn = self.attn_layers[-1](
                    x,
                    tau=tau,
                    delta=None
                )

            attns.append(attn)

        # ====================================================
        # Standard Encoder
        # ====================================================

        else:

            for i, attn_layer in enumerate(
                self.attn_layers
            ):

                # ------------------------------------------------
                # Pass profiling metadata
                # ------------------------------------------------

                if hasattr(
                    attn_layer,
                    'enable_profiling'
                ):

                    attn_layer.enable_profiling = (
                        self.enable_profiling
                    )

                if hasattr(
                    attn_layer,
                    'profile_layer_id'
                ):

                    attn_layer.profile_layer_id = i

                if hasattr(
                    attn_layer,
                    'attention'
                ):

                    attn_layer.attention.enable_profiling = (
                        self.enable_profiling
                    )

                    attn_layer.attention.profile_layer_id = i

                    if hasattr(
                        attn_layer.attention,
                        'inner_attention'
                    ):

                        attn_layer.attention.inner_attention.enable_profiling = (
                            self.enable_profiling
                        )

                        attn_layer.attention.inner_attention.profile_layer_id = i

                # ------------------------------------------------
                # Encoder layer
                # ------------------------------------------------

                with record_function(
                    f'Encoder::Layer_{i}'
                ):

                    x, attn = attn_layer(
                        x,
                        attn_mask=attn_mask,
                        tau=tau,
                        delta=delta
                    )

                attns.append(attn)

        # ====================================================
        # Final Encoder normalization
        # ====================================================

        if self.norm is not None:

            with record_function(
                'Encoder::FinalNorm'
            ):

                x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu"
    ):
        super(DecoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.norm3 = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.activation = (
            F.relu
            if activation == "relu"
            else F.gelu
        )

    def forward(
        self,
        x,
        cross,
        x_mask=None,
        cross_mask=None,
        tau=None,
        delta=None
    ):

        x = x + self.dropout(
            self.self_attention(
                x,
                x,
                x,
                attn_mask=x_mask,
                tau=tau,
                delta=None
            )[0]
        )

        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x,
                cross,
                cross,
                attn_mask=cross_mask,
                tau=tau,
                delta=delta
            )[0]
        )

        y = x = self.norm2(x)

        y = self.dropout(
            self.activation(
                self.conv1(
                    y.transpose(-1, 1)
                )
            )
        )

        y = self.dropout(
            self.conv2(
                y
            ).transpose(-1, 1)
        )

        return self.norm3(
            x + y
        )


class Decoder(nn.Module):
    def __init__(
        self,
        layers,
        norm_layer=None,
        projection=None
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            layers
        )

        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x,
        cross,
        x_mask=None,
        cross_mask=None,
        tau=None,
        delta=None
    ):

        for layer in self.layers:

            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x