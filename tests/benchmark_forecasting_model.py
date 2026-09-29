"""Reproducible full-model compute benchmark for iTransformer variants.

The benchmark intentionally uses synthetic inputs: model weights and dataset
values do not change the execution graph, while this removes data-loader and
disk-I/O noise.  It reports batch-1 latency, batched throughput, CUDA memory,
parameter count, and analytical attention MACs.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.iTransformer import Model


CSV_FIELDS = [
    "group", "name", "attention", "rank", "time_tokens", "enc_in", "tokens",
    "seq_len", "pred_len", "e_layers", "dtype", "device",
    "latency_batch_size", "latency_median_ms", "latency_mean_ms",
    "latency_p95_ms", "latency_std_ms", "latency_peak_allocated_mib",
    "throughput_batch_size", "throughput_median_ms", "throughput_mean_ms",
    "samples_per_second", "throughput_peak_allocated_mib",
    "training_step_median_ms", "training_samples_per_second",
    "training_peak_allocated_mib",
    "parameters", "parameter_mib", "attention_macs_per_sample",
    "dense_attention_macs_per_sample", "attention_macs_ratio_vs_dense",
]


def percentile(values, q):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def build_config(args):
    return SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        output_attention=False,
        use_norm=bool(args.use_norm),
        induced_attention=args.attention == "induced",
        attn_rank=args.rank,
        attn_time_tokens=args.time_tokens,
        attn_gate_init=args.gate_init,
        d_model=args.d_model,
        embed="timeF",
        freq="h",
        dropout=args.dropout,
        class_strategy="projection",
        n_heads=args.n_heads,
        factor=1,
        d_ff=args.d_ff,
        activation="gelu",
        e_layers=args.e_layers,
    )


def attention_macs(args):
    tokens = args.enc_in + args.time_features
    head_dim = args.d_model // args.n_heads
    dense_per_layer = 2 * args.n_heads * tokens * tokens * head_dim
    if args.attention == "dense":
        selected_per_layer = dense_per_layer
    else:
        variate_tokens = tokens - args.time_tokens
        if variate_tokens <= 0:
            raise ValueError("time_tokens must be smaller than total tokens")
        selected_per_layer = args.n_heads * head_dim * (
            3 * args.rank * variate_tokens
            + 2 * tokens * (args.rank + args.time_tokens)
        )
    dense = dense_per_layer * args.e_layers
    selected = selected_per_layer * args.e_layers
    return selected, dense


def make_inputs(batch_size, args, device, dtype):
    generator = torch.Generator(device=device).manual_seed(2023 + batch_size)
    x = torch.randn(
        batch_size, args.seq_len, args.enc_in,
        device=device, dtype=dtype, generator=generator,
    )
    x_mark = None
    if args.time_features:
        x_mark = torch.randn(
            batch_size, args.seq_len, args.time_features,
            device=device, dtype=dtype, generator=generator,
        )
    return x, x_mark


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model, inputs, warmup, iterations, device):
    x, x_mark = inputs
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, x_mark, None, None)
        synchronize(device)

        baseline_memory = 0
        if device.type == "cuda":
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        elapsed_ms = []
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for start, end in zip(starts, ends):
                start.record()
                output = model(x, x_mark, None, None)
                end.record()
            synchronize(device)
            elapsed_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            for _ in range(iterations):
                started = time.perf_counter()
                output = model(x, x_mark, None, None)
                elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    result = {
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 0.95),
        "std_ms": statistics.pstdev(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "output_shape": list(output.shape),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
        )
        result["incremental_peak_allocated_mib"] = max(
            0.0,
            (torch.cuda.max_memory_allocated(device) - baseline_memory) / 2 ** 20,
        )
        result["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / 2 ** 20
        )
    return result


def timed_training_step(model, inputs, warmup, iterations, device):
    x, x_mark = inputs
    model.train()

    def step():
        model.zero_grad(set_to_none=True)
        output = model(x, x_mark, None, None)
        loss = output.square().mean()
        loss.backward()
        return output, loss

    for _ in range(warmup):
        output, loss = step()
    synchronize(device)
    model.zero_grad(set_to_none=True)

    baseline_memory = 0
    if device.type == "cuda":
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms = []
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            output, loss = step()
            end.record()
        synchronize(device)
        elapsed_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    else:
        for _ in range(iterations):
            started = time.perf_counter()
            output, loss = step()
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    model.zero_grad(set_to_none=True)
    model.eval()
    result = {
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 0.95),
        "std_ms": statistics.pstdev(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "output_shape": list(output.shape),
        "last_loss": float(loss.detach().cpu()),
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
        )
        result["incremental_peak_allocated_mib"] = max(
            0.0,
            (torch.cuda.max_memory_allocated(device) - baseline_memory) / 2 ** 20,
        )
        result["peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / 2 ** 20
        )
    return result


def load_checkpoint(model, checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def flatten_report(report):
    config = report["configuration"]
    latency = report["latency"]
    throughput = report["throughput"]
    training = report["training_step"]
    return {
        "group": report["group"],
        "name": report["name"],
        "attention": config["attention"],
        "rank": config["rank"] if config["attention"] == "induced" else "",
        "time_tokens": config["time_tokens"] if config["attention"] == "induced" else "",
        "enc_in": config["enc_in"],
        "tokens": config["enc_in"] + config["time_features"],
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
        "e_layers": config["e_layers"],
        "dtype": report["environment"]["dtype"],
        "device": report["environment"]["device"],
        "latency_batch_size": config["latency_batch_size"],
        "latency_median_ms": latency["median_ms"],
        "latency_mean_ms": latency["mean_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_std_ms": latency["std_ms"],
        "latency_peak_allocated_mib": latency.get("peak_allocated_mib", ""),
        "throughput_batch_size": config["throughput_batch_size"],
        "throughput_median_ms": throughput["median_ms"],
        "throughput_mean_ms": throughput["mean_ms"],
        "samples_per_second": report["samples_per_second"],
        "throughput_peak_allocated_mib": throughput.get("peak_allocated_mib", ""),
        "training_step_median_ms": training["median_ms"],
        "training_samples_per_second": report["training_samples_per_second"],
        "training_peak_allocated_mib": training.get("peak_allocated_mib", ""),
        "parameters": report["parameters"],
        "parameter_mib": report["parameter_mib"],
        "attention_macs_per_sample": report["attention_macs_per_sample"],
        "dense_attention_macs_per_sample": report["dense_attention_macs_per_sample"],
        "attention_macs_ratio_vs_dense": report["attention_macs_ratio_vs_dense"],
    }


def write_outputs(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / (report["name"] + ".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / "benchmark_summary.csv"
    row = flatten_report(report)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def run(args):
    if args.d_model % args.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if args.attention == "induced" and args.rank <= 0:
        raise ValueError("rank must be positive for induced attention")
    if args.time_tokens > args.time_features:
        raise ValueError("time_tokens cannot exceed supplied time_features")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 benchmark requires CUDA")

    torch.manual_seed(2023)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2023)
        torch.backends.cudnn.benchmark = False

    model = Model(build_config(args))
    checkpoint = load_checkpoint(model, args.checkpoint)
    model = model.to(device=device, dtype=dtype).eval()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    selected_macs, dense_macs = attention_macs(args)

    latency_inputs = make_inputs(
        args.latency_batch_size, args, device, dtype
    )
    latency = timed_forward(
        model, latency_inputs, args.warmup, args.iterations, device
    )
    del latency_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    throughput_inputs = make_inputs(
        args.throughput_batch_size, args, device, dtype
    )
    throughput = timed_forward(
        model, throughput_inputs, args.warmup, args.iterations, device
    )
    del throughput_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    training_inputs = make_inputs(
        args.throughput_batch_size, args, device, dtype
    )
    training_step = timed_training_step(
        model,
        training_inputs,
        args.training_warmup,
        args.training_iterations,
        device,
    )

    report = {
        "name": args.name,
        "group": infer_group(args.name),
        "scope": "synthetic full-model inference and forward/backward; excludes data loading, optimizer, and checkpoint I/O",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
            ),
            "dtype": str(dtype),
        },
        "configuration": vars(args),
        "checkpoint": checkpoint,
        "parameters": parameters,
        "parameter_mib": parameter_bytes / 2 ** 20,
        "attention_macs_per_sample": selected_macs,
        "dense_attention_macs_per_sample": dense_macs,
        "attention_macs_ratio_vs_dense": selected_macs / dense_macs,
        "latency": latency,
        "throughput": throughput,
        "training_step": training_step,
        "samples_per_second": (
            args.throughput_batch_size * 1000.0 / throughput["mean_ms"]
        ),
        "training_samples_per_second": (
            args.throughput_batch_size * 1000.0 / training_step["mean_ms"]
        ),
    }
    json_path, csv_path = write_outputs(report, Path(args.output_dir))
    report["json_path"] = str(json_path.resolve())
    report["csv_path"] = str(csv_path.resolve())
    return report


def infer_group(name):
    if name.endswith("_dense"):
        return name[:-6]
    pieces = name.rsplit("_r", 1)
    if len(pieces) == 2 and "_tt" in pieces[1]:
        return pieces[0]
    raise ValueError(
        "name must end in _dense or _r<rank>_tt<time_tokens>"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--attention", choices=["dense", "induced"], required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--time_tokens", type=int, default=0)
    parser.add_argument("--gate_init", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--time_features", type=int, choices=[0, 4], required=True)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, required=True)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_norm", type=int, choices=[0, 1], default=1)
    parser.add_argument("--latency_batch_size", type=int, default=1)
    parser.add_argument("--throughput_batch_size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--training_warmup", type=int, default=10)
    parser.add_argument("--training_iterations", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main():
    report = run(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
