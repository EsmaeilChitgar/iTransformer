import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthoNormLinEncoderLayer(nn.Module):
    """
    OLinear-style encoder layer:
        Cross-Series NormLin
        +
        FFN

    Adapted from OLinear's LinearEncoder_Multihead.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        token_num,
        dropout=0.1,
        activation="relu",
        n_heads=2
    ):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.token_num = token_num
        self.n_heads = n_heads

        head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ===== NormLin: copied from OLinear =====
        self.v_proj = nn.Linear(
            d_model,
            head_dim * head_dim
        )

        self.out_proj = nn.Linear(
            head_dim * head_dim,
            d_model
        )

        self.weight_mat = nn.Parameter(
            torch.randn(
                n_heads,
                token_num,
                token_num
            )
        )

        # ===== FFN: same OLinear / Transformer style =====
        self.conv1 = nn.Conv1d(
            d_model,
            d_ff,
            kernel_size=1
        )

        self.conv2 = nn.Conv1d(
            d_ff,
            d_model,
            kernel_size=1
        )

        self.activation = (
            F.relu
            if activation == "relu"
            else F.gelu
        )

    def forward(self, x):
        # x: [B, N, D]

        B, N, D = x.shape

        if N != self.token_num:
            raise RuntimeError(
                f"NormLin token mismatch: "
                f"expected {self.token_num}, got {N}"
            )

        # --------------------------------------------------
        # NormLin - exactly following OLinear
        # --------------------------------------------------

        values = self.v_proj(x)

        values = values.reshape(
            B,
            N,
            self.n_heads,
            -1
        )

        A = F.softplus(
            self.weight_mat
        )

        A = F.normalize(
            A,
            p=1,
            dim=-1
        )

        A = self.dropout(A)

        new_x = (
            A @ values.transpose(1, 2)
        ).transpose(1, 2)

        new_x = new_x.flatten(
            -2
        )

        new_x = self.out_proj(
            new_x
        )

        x = x + self.dropout(
            new_x
        )

        x = self.norm1(x)

        # --------------------------------------------------
        # FFN
        # --------------------------------------------------

        y = self.dropout(
            self.activation(
                self.conv1(
                    x.transpose(-1, 1)
                )
            )
        )

        y = self.dropout(
            self.conv2(
                y
            ).transpose(-1, 1)
        )

        return self.norm2(
            x + y
        )


class OrthoNormLinEncoder(nn.Module):

    def __init__(
        self,
        e_layers,
        d_model,
        d_ff,
        token_num,
        dropout=0.1,
        activation="relu",
        n_heads=2
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            OrthoNormLinEncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                token_num=token_num,
                dropout=dropout,
                activation=activation,
                n_heads=n_heads
            )
            for _ in range(e_layers)
        ])

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)