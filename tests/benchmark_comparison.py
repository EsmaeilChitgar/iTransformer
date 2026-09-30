"""Synthetic end-to-end benchmark for comparison models (not accuracy).

Usage from repo root:
python tests/benchmark_comparison.py --device cuda --nvars 862 --rank 8
CUDA timings synchronize; inference batch throughput vs latency must be kept apart.
"""
import sys
import json
import argparse
import time
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.comparison_preflight import build_args
from model.iTransformer import Model


def bench(method, device, nvars, rank, batch, warmup, repeat):
    torch.manual_seed(2023)
    cfg = build_args(method)
    cfg.seq_len, cfg.pred_len = 96, 96
    cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.e_layers = 512, 512, 8, 4
    cfg.attn_rank, cfg.dropout = rank, 0.1
    cfg.comparison_time_tokens, cfg.attn_time_tokens = 4, 4
    model = Model(cfg).to(device).eval()
    x = torch.randn(batch, 96, nvars, device=device)
    mark = torch.randn(batch, 96, 4, device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mark, None, None)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(repeat):
            sync()
            start = time.perf_counter()
            y = model(x, mark, None, None)
            sync()
            elapsed.append(time.perf_counter() - start)
    assert y.shape == (batch, 96, nvars)
    median = statistics.median(elapsed)
    return dict(method=method, rank=rank, batch=batch, nvars=nvars,
                parameters=sum(p.numel() for p in model.parameters()),
                median_ms=median * 1e3, throughput=batch / median,
                peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                    if device.type == 'cuda' else None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--nvars', type=int, default=862)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeat', type=int, default=15)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu --batch 1 --nvars 16 for smoke')
    print(json.dumps([bench(m,device,args.nvars,args.rank,args.batch,
                           args.warmup,args.repeat) for m in
                       ('dense','ilra','isab','luna')], indent=2))


if __name__ == '__main__':
    main()
