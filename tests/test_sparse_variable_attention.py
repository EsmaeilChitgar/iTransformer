"""Quick smoke test for iSparseTransformer (run where torch is installed).

    python -u tests/test_sparse_variable_attention.py

Builds the model in each selection mode plus the disabled (original) mode and
checks forward/backward shapes and that the selection regularizer is produced.
"""
import argparse
import torch

from model.iSparseTransformer import Model


def make_cfg(**kw):
    c = argparse.Namespace()
    c.seq_len = 96
    c.pred_len = 96
    c.output_attention = False
    c.use_norm = True
    c.d_model = 64
    c.embed = 'fixed'
    c.freq = 'h'
    c.dropout = 0.1
    c.class_strategy = 'projection'
    c.e_layers = 2
    c.d_ff = 512
    c.activation = 'gelu'
    c.n_heads = 2
    c.factor = 1
    # sparse defaults
    c.sparse_var_attn = False
    c.var_select_mode = 'kv_select'
    c.var_select_k = 0.25
    c.var_select_estimator = 'mlp'
    c.var_select_temp = 0.1
    c.var_select_reg = 0.01
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(mode, n_vars):
    cfg = make_cfg(sparse_var_attn=(mode != 'full'), var_select_mode=mode,
                   var_select_k=0.2 if mode != 'full' else 0.25)
    model = Model(cfg)
    B = 4
    x = torch.randn(B, cfg.seq_len, n_vars)
    xm = None
    dec = torch.zeros(B, cfg.seq_len, n_vars)
    out = model(x, xm, dec, None)
    assert out.shape == (B, cfg.pred_len, n_vars), out.shape
    loss = out.float().pow(2).mean()
    loss.backward()
    reg = model.selection_reg_loss
    k = model.encoder.attn_layers[0].attention.last_num_selected if cfg.sparse_var_attn else 0
    ratio = model.encoder.attn_layers[0].attention.last_keep_ratio if cfg.sparse_var_attn else 1.0
    print("mode=%-9s D=%-4d out=%s K=%d keep_ratio=%.3f reg=%.5f"
          % (mode, n_vars, tuple(out.shape), int(k), ratio, float(reg)))
    assert torch.isfinite(loss).all(), "non-finite loss"


if __name__ == '__main__':
    for n in [8, 321, 862]:
        for mode in ['full', 'kv_select', 'topk', 'soft']:
            run(mode, n)
    print("ALL_OK")
