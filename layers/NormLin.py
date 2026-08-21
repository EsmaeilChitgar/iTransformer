import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinMultiHead(nn.Module):
    """
    NormLin cross-series learner adapted directly from
    OLinear's LinearEncoder_Multihead.

    Input:
        [B, N, D]

    N = number of tokens / variates
    D = d_model
    """

    def __init__(
        self,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        token_num=None,
        n_heads=2
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff
        self.token_num = token_num
        self.n_heads = n_heads

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ---- OLinear ----
        self.v_proj = nn.Linear(
            d_model,
            head_dim * head_dim
        )

        self.out_proj = nn.Linear(
            head_dim * head_dim,
            d_model
        )

        # Exactly the OLinear multi-head structure
        self.weight_mat = nn.Parameter(
            torch.randn(
                n_heads,
                token_num,
                token_num
            )
        )

        # ---- same FFN structure ----
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

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, N, D]

        B, N, D = x.shape

        if N != self.token_num:
            raise ValueError(
                f"NormLin expected {self.token_num} tokens, "
                f"but got {N}"
            )

        # --------------------------------------------------
        # OLinear
        # --------------------------------------------------

        values = self.v_proj(x)

        # [B, N, H, head_dim]
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

        # OLinear:
        # (A @ values.transpose(1, 2))
        new_x = (
            A @ values.transpose(1, 2)
        ).transpose(1, 2).flatten(-2)

        new_x = self.out_proj(
            new_x
        )

        # --------------------------------------------------
        # Residual + Norm
        # --------------------------------------------------

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