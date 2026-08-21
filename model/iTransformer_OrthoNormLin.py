import os
import numpy as np
import torch
import torch.nn as nn

from layers.OrthoTransNormLin import OrthoNormLinEncoder


class Model(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff

        self.embed_size = getattr(
            configs,
            "embed_size",
            8
        )

        self.n_heads = getattr(
            configs,
            "n_heads",
            2
        )

        self.dropout = configs.dropout

        # ======================================================
        # Load Q
        # ======================================================

        q_path = configs.q_mat_file

        if not os.path.isfile(q_path):
            q_path = os.path.join(
                configs.root_path,
                q_path
            )

        if not os.path.isfile(q_path):
            raise FileNotFoundError(
                f"Q matrix not found: {q_path}"
            )

        q_out_path = configs.q_out_mat_file

        if not os.path.isfile(q_out_path):
            q_out_path = os.path.join(
                configs.root_path,
                q_out_path
            )

        if not os.path.isfile(q_out_path):
            raise FileNotFoundError(
                f"Q_out matrix not found: {q_out_path}"
            )

        Q = np.load(q_path).astype(
            np.float32
        )

        Q_out = np.load(q_out_path).astype(
            np.float32
        )

        if Q.shape != (
            self.seq_len,
            self.seq_len
        ):
            raise ValueError(
                f"Q shape {Q.shape} != "
                f"({self.seq_len}, {self.seq_len})"
            )

        if Q_out.shape != (
            self.pred_len,
            self.pred_len
        ):
            raise ValueError(
                f"Q_out shape {Q_out.shape} != "
                f"({self.pred_len}, {self.pred_len})"
            )

        self.register_buffer(
            "Q_mat",
            torch.from_numpy(Q)
        )

        self.register_buffer(
            "Q_out_mat",
            torch.from_numpy(Q_out)
        )

        # ======================================================
        # OLinear-style token embedding
        # ======================================================

        self.embeddings = nn.Parameter(
            torch.randn(
                1,
                self.embed_size
            )
        )

        # ======================================================
        # Temporal orthogonal domain
        #
        # [B,N,L,D]
        # -> Q
        # -> [B,N,L,D]
        # ======================================================

        self.input_projection = nn.Linear(
            self.seq_len * self.embed_size,
            self.d_model
        )

        # ======================================================
        # NormLin encoder
        # ======================================================

        self.encoder = OrthoNormLinEncoder(
            e_layers=configs.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            token_num=self.enc_in,
            dropout=self.dropout,
            activation=configs.activation,
            n_heads=self.n_heads
        )

        # ======================================================
        # Prediction projection
        # ======================================================

        self.output_projection = nn.Linear(
            self.d_model,
            self.pred_len * self.embed_size
        )

        # Final temporal prediction
        self.fc = nn.Sequential(
            nn.Linear(
                self.pred_len * self.embed_size,
                self.d_ff
            ),
            nn.GELU(),
            nn.Linear(
                self.d_ff,
                self.pred_len
            )
        )

        self.final_dropout = nn.Dropout(
            self.dropout
        )

        # Learnable residuals used in OLinear
        self.delta1 = nn.Parameter(
            torch.zeros(
                1,
                self.enc_in,
                1,
                self.seq_len
            )
        )

        self.delta2 = nn.Parameter(
            torch.zeros(
                1,
                self.enc_in,
                1,
                self.pred_len
            )
        )

    # ==========================================================
    # Token Embedding
    # ==========================================================

    def tokenEmb(self, x):

        # x: [B,L,N]
        #
        # -> [B,N,L,1]
        #
        # -> [B,N,L,D]

        x = x.transpose(
            -1,
            -2
        )

        x = x.unsqueeze(
            -1
        )

        return (
            x * self.embeddings
        )

    # ==========================================================
    # OrthoTrans
    # ==========================================================

    def orthogonal_transform(
        self,
        x
    ):

        # x: [B,N,L,D]

        B, N, L, D = x.shape

        if L != self.seq_len:
            raise ValueError(
                f"Expected sequence length "
                f"{self.seq_len}, got {L}"
            )

        # [B,N,L,D]
        # -> [B,N,D,L]

        x = x.transpose(
            -1,
            -2
        )

        # Q transformation
        #
        # Q: [L,L]
        #
        # result:
        # [B,N,D,L]

        x = torch.einsum(
            "bndl,lv->bndv",
            x,
            self.Q_mat.transpose(
                -1,
                -2
            )
        )

        x = (
            x + self.delta1
        )

        # back to [B,N,L,D]

        x = x.transpose(
            -1,
            -2
        )

        return x

    # ==========================================================
    # Inverse OrthoTrans
    # ==========================================================

    def inverse_orthogonal_transform(
        self,
        x
    ):

        # x:
        # [B,N,H,D]

        B, N, H, D = x.shape

        x = x.transpose(
            -1,
            -2
        )

        # Q_out:
        # [H,H]

        x = torch.einsum(
            "bndh,hv->bndv",
            x,
            self.Q_out_mat
        )

        x = (
            x + self.delta2
        )

        return x.transpose(
            -1,
            -2
        )

    # ==========================================================
    # Forecast
    # ==========================================================

    def forecast(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None
    ):

        # ------------------------------------------------------
        # iTransformer-style normalization
        # ------------------------------------------------------

        means = x_enc.mean(
            dim=1,
            keepdim=True
        ).detach()

        x_enc = (
            x_enc - means
        )

        stdev = torch.sqrt(
            torch.var(
                x_enc,
                dim=1,
                keepdim=True,
                unbiased=False
            ) + 1e-5
        )

        x_enc = (
            x_enc / stdev
        )

        # ------------------------------------------------------
        # Token embedding
        # ------------------------------------------------------

        x = self.tokenEmb(
            x_enc
        )

        # ------------------------------------------------------
        # OrthoTrans
        # ------------------------------------------------------

        x = self.orthogonal_transform(
            x
        )

        B, N, L, D = x.shape

        # ------------------------------------------------------
        # Flatten temporal representation
        #
        # [B,N,L,D]
        # -> [B,N,L*D]
        # ------------------------------------------------------

        x = x.flatten(
            -2
        )

        # ------------------------------------------------------
        # Input projection
        #
        # [B,N,L*D]
        # -> [B,N,d_model]
        # ------------------------------------------------------

        x = self.input_projection(
            x
        )

        # ------------------------------------------------------
        # NormLin
        # ------------------------------------------------------

        x = self.encoder(
            x
        )

        # ------------------------------------------------------
        # Output projection
        #
        # [B,N,d_model]
        # -> [B,N,H*D]
        # ------------------------------------------------------

        x = self.output_projection(
            x
        )

        x = x.reshape(
            B,
            N,
            self.pred_len,
            self.embed_size
        )

        # ------------------------------------------------------
        # Inverse OrthoTrans
        # ------------------------------------------------------

        x = self.inverse_orthogonal_transform(
            x
        )

        # ------------------------------------------------------
        # Final forecast head
        #
        # [B,N,H,D]
        # -> [B,N,H]
        # ------------------------------------------------------

        x = x.reshape(
            B,
            N,
            self.pred_len * self.embed_size
        )

        x = self.fc(
            x
        )

        x = self.final_dropout(
            x
        )

        # [B,N,H] -> [B,H,N]

        x = x.transpose(
            1,
            2
        )

        # ------------------------------------------------------
        # De-normalization
        # ------------------------------------------------------

        x = (
            x
            * stdev[:, 0, :]
            .unsqueeze(1)
        )

        x = (
            x
            + means[:, 0, :]
            .unsqueeze(1)
        )

        return x

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        return self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )