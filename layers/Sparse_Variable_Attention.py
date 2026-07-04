"""
Sparse Variable Attention for iTransformer.

iTransformer treats each *variable* (variate) as one token, so self-attention is
performed over the D variables and costs O(D^2) in both FLOPs and memory.  For
high-dimensional datasets (Electricity: 321, Traffic: 862) this quadratic cost in
the number of variates becomes the dominant bottleneck.

This module introduces a lightweight, sample-dependent **variable importance
estimator** followed by an **adaptive (differentiable) variable selection** step
that runs *immediately before* every self-attention layer.  Only the selected
variable tokens participate in the expensive pairwise attention; the rest bypass
attention through the residual pathway of the encoder block.  Everything else in
iTransformer (inverted embedding, encoder block structure, FFN, normalization,
prediction head, training pipeline) is left untouched.

The module is a drop-in replacement for ``layers.SelfAttention_Family.AttentionLayer``:
its ``forward`` has the same signature and returns ``(out, attn)``.

Three selection modes are provided (``select_mode``):

* ``kv_select`` (default, recommended): every variable token stays a *query*, but
  only the top-K selected tokens act as keys/values.  Complexity drops from
  O(D^2) to O(D * K) with K << D, which is exactly the target complexity.  All
  variables still receive an attention update, so accuracy is well preserved.
* ``topk``: only the top-K tokens participate as queries *and* keys (a K x K
  attention block).  Complexity O(K^2) -- the largest speed-up -- at the cost of
  non-selected tokens receiving no attention update in that layer (they pass
  through the block residual unchanged).
* ``soft``: a fully-differentiable soft top-K mask reweights keys/values without
  any gathering.  This keeps the original O(D^2) cost (no speed-up) and is
  provided as an accuracy reference / upper bound.

Selection is differentiable via a straight-through (ST) estimator: the forward
pass uses a *hard* top-K gather (real FLOPs / memory savings), while the backward
pass routes gradients through a sigmoid relaxation of the top-K mask.  This keeps
gradients stable (bounded) while letting the importance estimator be trained
end-to-end.  A small budget + entropy regularizer (exposed via ``last_reg`` and
added to the training loss by the experiment) shapes the selection distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableImportanceEstimator(nn.Module):
    """Lightweight, variate-agnostic scorer.

    Maps each variable token of shape ``[..., d_model]`` to a scalar importance
    score ``[..., 1]``.  The network is per-token (no mixing across variables),
    so it is independent of the number of variables D and adds negligible
    overhead: O(D * d_model) per layer.

    estimator:
        * ``linear``  : single linear projection d_model -> 1.
        * ``mlp``     : two-layer MLP with ReLU (default).
        * ``se``      : squeeze-excitation style gate.
    """

    def __init__(self, d_model, estimator='mlp'):
        super(VariableImportanceEstimator, self).__init__()
        self.estimator = estimator
        hidden = max(d_model // 4, 8)
        if estimator == 'linear':
            self.net = nn.Linear(d_model, 1)
        elif estimator == 'mlp':
            self.fc1 = nn.Linear(d_model, hidden)
            self.fc2 = nn.Linear(hidden, 1)
        elif estimator == 'se':
            self.fc1 = nn.Linear(d_model, d_model)
            self.fc2 = nn.Linear(d_model, 1)
        else:
            raise ValueError("Unknown estimator type: {}".format(estimator))

    def forward(self, x):
        # x: [B, N, d_model] -> scores [B, N]
        if self.estimator == 'linear':
            return self.net(x).squeeze(-1)
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


def topk_selection(scores, K, temperature):
    """Differentiable (straight-through) top-K selection over the last dim.

    Args:
        scores:     [B, N] importance scores.
        K:          number of tokens to select (int).
        temperature: temperature of the sigmoid relaxation (smaller -> harder).

    Returns:
        m_soft: [B, N] sigmoid relaxation of the top-K mask (for gradients).
        m_hard: [B, N] exact {0, 1} top-K mask (for the forward pass).
        idx:    [B, K] long indices of the selected tokens.
    """
    B, N = scores.shape
    K = min(max(int(K), 1), N)

    # Threshold = smallest score among the top-K (detached, used only for the
    # soft surrogate so that the hard gather remains the true forward pass).
    with torch.no_grad():
        tau = torch.topk(scores, K, dim=-1).values[:, -1]  # [B]

    # Soft surrogate mask: 1 for clearly-selected, 0 for clearly-dropped, ~0.5
    # at the selection boundary.  This is what gradients flow through.
    m_soft = torch.sigmoid((scores - tau.unsqueeze(-1)) / max(temperature, 1e-3))

    # Hard top-K mask + indices (exact K, used in the forward pass).
    idx = torch.topk(scores, K, dim=-1).indices  # [B, K]
    m_hard = torch.zeros_like(scores)
    m_hard.scatter_(1, idx, 1.0)
    return m_soft, m_hard, idx


class SparseVariableAttentionLayer(nn.Module):
    """Drop-in sparse replacement for ``AttentionLayer``.

    Args:
        attention:   inner attention module (e.g. FullAttention).
        d_model:     model dimension.
        n_heads:     number of attention heads.
        d_keys/d_values: per-head dims (default d_model // n_heads).
        select_k:    selection budget. If 0 < select_k <= 1 it is treated as a
                     *ratio* of the number of variables D; if > 1 it is an
                     absolute count.
        select_mode: 'kv_select' | 'topk' | 'soft'.
        estimator:   'linear' | 'mlp' | 'se'.
        temperature: ST sigmoid temperature.
        reg_weight:  unused here (kept for API symmetry); the raw regularization
                     term is stored in ``last_reg`` and weighted by the caller.

    Attributes:
        last_reg:          scalar tensor with the budget+entropy regularization
                           of the most recent forward (0 if no selection).
        last_num_selected: float, the K used in the most recent forward.
        last_keep_ratio:   float, K / N of the most recent forward.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 select_k=0.25, select_mode='kv_select', estimator='mlp',
                 temperature=0.1, reg_weight=0.0):
        super(SparseVariableAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.select_k = select_k
        self.select_mode = select_mode
        self.temperature = temperature

        self.scorer = VariableImportanceEstimator(d_model, estimator=estimator)

        # Reporting / regularization accumulators.
        self.last_reg = torch.zeros(1)
        self.last_num_selected = 0.0
        self.last_keep_ratio = 0.0

    def _k_for(self, n_vars):
        if self.select_k <= 1.0:
            k = int(round(self.select_k * n_vars))
        else:
            k = int(self.select_k)
        return max(1, min(k, n_vars))

    @staticmethod
    def _st_one(m_sel):
        """Straight-through identity-1 gate.

        Forward value is exactly 1.0 (so the gathered tokens are used as-is),
        while the backward pass carries the gradient of ``m_sel`` w.r.t. the
        importance scores, injecting a (boundary-focused, stable) task signal
        into the selection decision.
        """
        return 1.0 + m_sel - m_sel.detach()

    def _regularization(self, m_soft, K, N):
        # Budget: encourage the soft mask to sum to K (keeps the *expected*
        # number of selected variables close to the budget K).
        budget = ((m_soft.sum(dim=-1) - K) / N) ** 2
        # Entropy: encourage a confident (near-binary) selection.
        eps = 1e-7
        ent = -(m_soft * torch.log(m_soft + eps) +
                (1.0 - m_soft) * torch.log(1.0 - m_soft + eps))
        # Normalized binary entropy: 0 when confident, ~1 when uncertain.
        ent = ent.mean()
        return budget.mean() + 0.1 * ent

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, N, _ = queries.shape
        H = self.n_heads
        K = self._k_for(N)

        # 1) Importance estimation from the (pre-projection) variable tokens.
        scores = self.scorer(queries)                       # [B, N]
        m_soft, m_hard, idx = topk_selection(scores, K, self.temperature)

        # 2) Project to multi-head Q, K, V (same as AttentionLayer).
        q = self.query_projection(queries).view(B, N, H, -1)
        k = self.key_projection(keys).view(B, N, H, -1)
        v = self.value_projection(values).view(B, N, H, -1)
        d_v = v.shape[-1]

        if self.select_mode == 'kv_select':
            # Every variable stays a query; only the top-K act as keys/values.
            # Complexity: O(D * K) instead of O(D^2).
            idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, H, d_v)
            k_sel = torch.gather(k, 1, idx_e)
            v_sel = torch.gather(v, 1, idx_e)
            # ST gate so the selection decision receives a task gradient.
            m_sel = torch.gather(m_soft, 1, idx)            # [B, K]
            v_sel = v_sel * self._st_one(m_sel).unsqueeze(-1).unsqueeze(-1)
            out, attn = self.inner_attention(
                q, k_sel, v_sel, attn_mask=None, tau=tau, delta=delta)
            out = out.view(B, N, -1)
            out = self.out_projection(out)

        elif self.select_mode == 'topk':
            # Only the top-K tokens attend among themselves (K x K block).
            # Complexity: O(K^2). Non-selected tokens bypass attention (their
            # output here is 0, so they rely on the block's residual + FFN).
            idx_e = idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, H, d_v)
            q_sel = torch.gather(q, 1, idx_e)
            k_sel = torch.gather(k, 1, idx_e)
            v_sel = torch.gather(v, 1, idx_e)
            m_sel = torch.gather(m_soft, 1, idx)
            v_sel = v_sel * self._st_one(m_sel).unsqueeze(-1).unsqueeze(-1)
            out_sel, attn = self.inner_attention(
                q_sel, k_sel, v_sel, attn_mask=None, tau=tau, delta=delta)
            out_sel = out_sel.view(B, K, -1)
            out_sel = self.out_projection(out_sel)          # [B, K, d_model]
            out = queries.new_zeros(B, N, out_sel.shape[-1])
            out.scatter_(1, idx.unsqueeze(-1).expand(B, K, out_sel.shape[-1]), out_sel)

        elif self.select_mode == 'soft':
            # Fully-differentiable soft mask, no gathering (O(D^2), reference).
            gate = m_soft.unsqueeze(-1).unsqueeze(-1)       # [B, N, 1, 1]
            k = k * gate
            v = v * gate
            out, attn = self.inner_attention(
                q, k, v, attn_mask=attn_mask, tau=tau, delta=delta)
            out = out.view(B, N, -1)
            out = self.out_projection(out)

        else:
            raise ValueError("Unknown select_mode: {}".format(self.select_mode))

        # 3) Regularization (consumed by the experiment during training).
        self.last_reg = self._regularization(m_soft, K, N)
        self.last_num_selected = float(K)
        self.last_keep_ratio = float(K) / float(N)
        return out, attn
