import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # Research controls.  They are inactive in ordinary training and make
        # it possible to run paired oracle diagnostics on a trained model.
        self.oracle_rank = 0
        self.capture_diagnostics = False
        self.last_attention = None
        self.last_values = None
        self.last_output = None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A_raw = torch.softmax(scale * scores, dim=-1)

        max_rank = min(A_raw.shape[-2], A_raw.shape[-1])
        rank = min(int(self.oracle_rank), max_rank)

        if 0 < rank < max_rank:
            # SVD is deliberately an oracle diagnostic.  It still forms the
            # dense attention matrix and therefore cannot measure speedup.
            # Run the factorization in fp32 because CUDA SVD does not support
            # all reduced precision dtypes.
            U, singular_values, Vh = torch.linalg.svd(
                A_raw.float(), full_matrices=False
            )
            U_r = U[..., :rank]
            S_r = singular_values[..., :rank]
            Vh_r = Vh[..., :rank, :]
            compressed_values = torch.einsum(
                "bhrs,bshd->bhrd", Vh_r, values.float()
            )
            V = torch.einsum(
                "bhlr,bhrd->blhd", U_r * S_r.unsqueeze(-2),
                compressed_values
            ).to(values.dtype)
            A_return = None
            if self.output_attention:
                A_return = torch.matmul(
                    U_r * S_r.unsqueeze(-2), Vh_r
                ).to(A_raw.dtype)
        else:
            A = self.dropout(A_raw)
            V = torch.einsum("bhls,bshd->blhd", A, values)
            A_return = A

        if self.capture_diagnostics:
            self.last_attention = A_raw.detach()
            self.last_values = values.detach()
            self.last_output = V.detach()

        if self.output_attention:
            return (V.contiguous(), A_return)
        else:
            return (V.contiguous(), None)


class InducedVariateAttention(nn.Module):
    """Permutation-equivariant cross-variate attention through R latents.

    All original variate queries remain present.  A small learned set of
    inducing queries first summarizes keys and values into ``rank`` latent
    interactions; every original token then reads from those latents.  The
    optional trailing time-feature tokens bypass the bottleneck as exact
    key/value pairs.

    The effective interaction cost is O(N R D), and the parameters do not
    depend on N or on variate order.
    """

    def __init__(self, rank, n_heads, d_head, time_tokens=0,
                 attention_dropout=0.1, output_attention=False,
                 gate_init=1.0):
        super(InducedVariateAttention, self).__init__()
        if rank <= 0:
            raise ValueError('rank must be positive')
        if n_heads <= 0 or d_head <= 0:
            raise ValueError('n_heads and d_head must be positive')
        if time_tokens < 0:
            raise ValueError('time_tokens cannot be negative')

        self.rank = int(rank)
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.time_tokens = int(time_tokens)
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.inducing_queries = nn.Parameter(
            torch.empty(self.n_heads, self.rank, self.d_head)
        )
        nn.init.normal_(
            self.inducing_queries, mean=0.0, std=self.d_head ** -0.5
        )

        # A direct gate permits a stable warm start from a dense checkpoint.
        # gate_init=1 is appropriate for training the architecture from scratch.
        self.output_gate = nn.Parameter(
            torch.full((self.n_heads,), float(gate_init))
        )

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if attn_mask is not None:
            raise ValueError(
                'InducedVariateAttention supports non-causal encoder '
                'attention only'
            )

        B, Nq, H, D = queries.shape
        _, Nk, Hk, Dk = keys.shape
        if Nq != Nk or values.shape[1] != Nk:
            raise ValueError('InducedVariateAttention requires self-attention')
        if H != self.n_heads or Hk != H or D != self.d_head or Dk != D:
            raise ValueError(
                'Projected attention shape does not match configured heads'
            )
        if self.time_tokens >= Nk:
            raise ValueError(
                'time_tokens must be smaller than the total token count'
            )

        Q = queries.permute(0, 2, 1, 3)
        K = keys.permute(0, 2, 1, 3)
        V = values.permute(0, 2, 1, 3)
        variate_count = Nk - self.time_tokens
        K_var = K[:, :, :variate_count, :]
        V_var = V[:, :, :variate_count, :]
        scale = 1.0 / sqrt(D)

        read_scores = torch.einsum(
            'hrd,bhnd->bhrn', self.inducing_queries, K_var
        ) * scale
        read_attention = self.dropout(
            torch.softmax(read_scores, dim=-1)
        )
        latent_keys = torch.einsum(
            'bhrn,bhnd->bhrd', read_attention, K_var
        )
        latent_values = torch.einsum(
            'bhrn,bhnd->bhrd', read_attention, V_var
        )

        if self.time_tokens:
            write_keys = torch.cat(
                [latent_keys, K[:, :, variate_count:, :]], dim=2
            )
            write_values = torch.cat(
                [latent_values, V[:, :, variate_count:, :]], dim=2
            )
        else:
            write_keys = latent_keys
            write_values = latent_values

        write_scores = torch.einsum(
            'bhnd,bhkd->bhnk', Q, write_keys
        ) * scale
        write_attention = self.dropout(
            torch.softmax(write_scores, dim=-1)
        )
        output = torch.einsum(
            'bhnk,bhkd->bhnd', write_attention, write_values
        )
        output = output * self.output_gate.view(1, H, 1, 1)
        output = output.permute(0, 2, 1, 3).contiguous()

        if self.output_attention:
            return output, write_attention.contiguous()
        return output, None


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

