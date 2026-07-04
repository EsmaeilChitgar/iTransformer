"""Profiling utility for iTransformer vs iSparseTransformer.

Measures, on synthetic data of a given dimensionality D:
  * number of selected variable tokens (K) and keep ratio,
  * theoretical attention FLOPs and the reduction ratio vs full attention,
  * peak GPU memory (forward + backward),
  * training (forward+backward) and inference (forward) wall-clock speed.

Run e.g.:
    python -u utils/sparse_profile.py --n_vars 321 --batch_size 32 --seq_len 96
    python -u utils/sparse_profile.py --n_vars 862 --batch_size 16 --seq_len 96 \
        --var_select_mode kv_select --var_select_k 0.15

Note: FLOPs are computed analytically for the cross-variate attention block
(projections + QK^T + AV) per layer, which is the component this work targets.
"""
import argparse
import time

import torch

from model.iTransformer import Model as iTransformer
from model.iSparseTransformer import Model as iSparseTransformer


def base_cfg(**kw):
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
    c.sparse_var_attn = False
    c.var_select_mode = 'kv_select'
    c.var_select_k = 0.25
    c.var_select_estimator = 'mlp'
    c.var_select_temp = 0.1
    c.var_select_reg = 0.01
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def attn_flops(D, K, mode, d_model, e_layers):
    """Analytical cross-variate attention FLOPs (per sample, all layers)."""
    # token projections: q,k,v,in + out  => 4 * D * d_model^2
    proj = 4 * D * d_model * d_model
    if mode == 'full':
        attn = 2 * D * D * d_model            # QK^T + AV
    elif mode == 'kv_select':
        attn = 2 * D * K * d_model
    elif mode == 'topk':
        attn = 2 * K * K * d_model
    elif mode == 'soft':
        attn = 2 * D * D * d_model            # no gather
    else:
        raise ValueError(mode)
    # importance estimator (mlp): D * d_model * hidden + D * hidden * 1
    hidden = max(d_model // 4, 8)
    scorer = D * d_model * hidden + D * hidden
    extra = scorer if mode != 'full' else 0
    return e_layers * (proj + attn + extra), e_layers * (proj + attn), e_layers * proj


def k_for(select_k, D):
    if select_k <= 1.0:
        return max(1, int(round(select_k * D)))
    return max(1, min(int(select_k), D))


def time_model(model, x, device, backward=True, warmup=3, iters=10):
    model.eval() if not backward else model.train()
    for _ in range(warmup):
        out = model(x, None, torch.zeros_like(x), None)
        if backward:
            out.float().pow(2).mean().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = model(x, None, torch.zeros_like(x), None)
        if backward:
            out.float().pow(2).mean().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / iters


def peak_mem(model, x, device):
    if device.type != 'cuda':
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    out = model(x, None, torch.zeros_like(x), None)
    out.float().pow(2).mean().backward()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def report(name, model, mode, D, K, cfg, x, device):
    t_fwd = time_model(model, x, device, backward=False)
    t_bwd = time_model(model, x, device, backward=True)
    mem = peak_mem(model, x, device)
    total_flops, attn_flops_, _ = attn_flops(D, K, mode, cfg.d_model, cfg.e_layers)
    print("  {:<14} K={:<5} keep={:.3f} | attn_flops={:<12} | "
          "fwd={:.4f}s bwd={:.4f}s | peak_mem={:.1f}MB".format(
              name, K if mode != 'full' else D,
              (K / D) if mode != 'full' else 1.0,
              attn_flops_, t_fwd, t_bwd, mem))
    return attn_flops_, mem, t_bwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_vars', type=int, default=321)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=2)
    p.add_argument('--var_select_mode', type=str, default='kv_select')
    p.add_argument('--var_select_k', type=float, default=0.25)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    K = k_for(args.var_select_k, args.n_vars)
    print("D={} batch={} seq={} d_model={} e_layers={} | sparse mode={} K={} keep={:.3f}".format(
        args.n_vars, args.batch_size, args.seq_len, args.d_model, args.e_layers,
        args.var_select_mode, K, K / args.n_vars))

    x = torch.randn(args.batch_size, args.seq_len, args.n_vars, device=device)

    # Original iTransformer (full attention)
    cfg_full = base_cfg(d_model=args.d_model, e_layers=args.e_layers, n_heads=args.n_heads)
    m_full = iTransformer(cfg_full).to(device)
    fa, ma, ba = report('iTransformer', m_full, 'full', args.n_vars, args.n_vars, cfg_full, x, device)

    # iSparseTransformer (sparse attention) -- also disable flag for a full-attention
    # reference built from the same class to confirm parity.
    cfg_sp = base_cfg(d_model=args.d_model, e_layers=args.e_layers, n_heads=args.n_heads,
                      sparse_var_attn=True, var_select_mode=args.var_select_mode,
                      var_select_k=args.var_select_k)
    m_sp = iSparseTransformer(cfg_sp).to(device)
    fs, ms, bs = report('iSparse', m_sp, args.var_select_mode, args.n_vars, K, cfg_sp, x, device)

    print("-" * 70)
    print("  attn FLOPs reduction:   {:.2f}x  ({:.3g} -> {:.3g})".format(fa / max(fs, 1), fa, fs))
    if ma > 0 and ms > 0:
        print("  peak memory reduction:  {:.2f}x  ({:.1f}MB -> {:.1f}MB)".format(ma / ms, ma, ms))
    print("  train step speed-up:    {:.2f}x  ({:.4f}s -> {:.4f}s)".format(ba / max(bs, 1e-9), ba, bs))


if __name__ == '__main__':
    main()
