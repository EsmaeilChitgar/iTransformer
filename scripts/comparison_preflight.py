"""Small CPU smoke test of Dense, ILRA, ISAB and Luna model integration.

No Traffic dataset or GPU required. Run from the iTransformer repo root:
python scripts/comparison_preflight.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.iTransformer import Model


def build_args(method):
    return SimpleNamespace(
        seq_len=12, label_len=6, pred_len=6,
        output_attention=False, use_norm=True,
        embed='timeF', freq='h', dropout=0.0, class_strategy='projection',
        d_model=16, n_heads=4, e_layers=2, d_ff=32,
        factor=1, activation='gelu', induced_attention=False,
        attn_rank=3, attn_time_tokens=4, attn_gate_init=1.0,
        compare_attention=method, comparison_time_tokens=4,
        isab_layernorm=0,
    )


def run(method, time_tokens=True):
    torch.manual_seed(2023)
    cfg = build_args(method)
    if not time_tokens:
        cfg.comparison_time_tokens = 0
        cfg.attn_time_tokens = 0
    model = Model(cfg)
    model.train()
    x = torch.randn(2, 12, 7)
    mark = torch.randn(2, 12, 4) if time_tokens else None
    y = model(x, mark, None, None)
    if isinstance(y, (tuple, list)):
        y = y[0]
    assert y.shape == (2, 6, 7), (method, y.shape)
    assert torch.isfinite(y).all(), method
    loss = y.square().mean()
    loss.backward()
    params = sum(p.numel() for p in model.parameters())
    trainable_grad = sum(int(p.grad is not None and p.grad.abs().sum().item() > 0)
                         for p in model.parameters() if p.requires_grad)
    assert trainable_grad >= 5, (method, trainable_grad)
    print(f'{method:>14s} time={time_tokens} out={list(y.shape)} '
          f'params={params:,} grads={trainable_grad} loss={loss.item():.5f}')
    return params


if __name__ == '__main__':
    for method in ('dense', 'ilra', 'isab', 'luna', 'isab_bypass', 'luna_bypass'):
        run(method)
    for method in ('isab', 'luna', 'ilra'):
        run(method, time_tokens=False)
    print('Preflight successful. This is NOT a full forecasting benchmark.')
