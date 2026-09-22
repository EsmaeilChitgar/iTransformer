"""Microbenchmark dense and induced cross-variate attention.

This measures the attention kernel in isolation.  It does not include the
embedding, feed-forward layers, data loading, or checkpoint I/O.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.SelfAttention_Family import FullAttention, InducedVariateAttention


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark(module, queries, keys, values, warmup, iterations):
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            module(queries, keys, values, None)
        synchronize(queries.device)

        elapsed_ms = []
        if queries.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(queries.device)
        for _ in range(iterations):
            synchronize(queries.device)
            started = time.perf_counter()
            output, _ = module(queries, keys, values, None)
            synchronize(queries.device)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    result = {
        'median_ms': statistics.median(elapsed_ms),
        'mean_ms': statistics.mean(elapsed_ms),
        'output_shape': list(output.shape),
    }
    if queries.device.type == 'cuda':
        result['peak_allocated_mib'] = (
            torch.cuda.max_memory_allocated(queries.device) / 2 ** 20
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=866)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--time_tokens', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument(
        '--dtype', choices=['float32', 'float16', 'bfloat16'],
        default='float32'
    )
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        device = torch.device(args.device)

    dtype = getattr(torch, args.dtype)
    if device.type == 'cpu' and dtype == torch.float16:
        raise RuntimeError('float16 benchmark requires CUDA')

    generator = torch.Generator(device=device).manual_seed(2023)
    shape = (args.batch_size, args.tokens, args.heads, args.head_dim)
    queries = torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    keys = torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    values = torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )

    dense = FullAttention(
        mask_flag=False, attention_dropout=0.0
    ).to(device=device, dtype=dtype)
    induced = InducedVariateAttention(
        rank=args.rank,
        n_heads=args.heads,
        d_head=args.head_dim,
        time_tokens=args.time_tokens,
        attention_dropout=0.0,
    ).to(device=device, dtype=dtype)

    dense_result = benchmark(
        dense, queries, keys, values, args.warmup, args.iterations
    )
    induced_result = benchmark(
        induced, queries, keys, values, args.warmup, args.iterations
    )
    report = {
        'device': str(device),
        'dtype': str(queries.dtype),
        'configuration': vars(args),
        'dense': dense_result,
        'induced': induced_result,
        'median_speedup': (
            dense_result['median_ms'] / induced_result['median_ms']
        ),
        'scope': 'attention kernel only',
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
