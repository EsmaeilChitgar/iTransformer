"""Pair benchmark rows with their dense control and compute paper ratios."""

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "group", "name", "attention", "rank", "time_tokens",
    "latency_median_ms", "latency_speedup_vs_dense",
    "samples_per_second", "throughput_speedup_vs_dense",
    "throughput_peak_allocated_mib", "memory_ratio_vs_dense",
    "training_step_median_ms", "training_speedup_vs_dense",
    "training_peak_allocated_mib", "training_memory_ratio_vs_dense",
    "parameters", "parameter_ratio_vs_dense",
    "attention_macs_ratio_vs_dense",
]


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else None


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return ""
    return numerator / denominator


def compare(rows):
    dense = {
        row["group"]: row for row in rows if row["attention"] == "dense"
    }
    missing = sorted({row["group"] for row in rows} - set(dense))
    if missing:
        raise ValueError("missing dense controls for: " + ", ".join(missing))

    compared = []
    for row in rows:
        control = dense[row["group"]]
        latency = number(row, "latency_median_ms")
        dense_latency = number(control, "latency_median_ms")
        throughput = number(row, "samples_per_second")
        dense_throughput = number(control, "samples_per_second")
        memory = number(row, "throughput_peak_allocated_mib")
        dense_memory = number(control, "throughput_peak_allocated_mib")
        training_time = number(row, "training_step_median_ms")
        dense_training_time = number(control, "training_step_median_ms")
        training_memory = number(row, "training_peak_allocated_mib")
        dense_training_memory = number(control, "training_peak_allocated_mib")
        parameters = number(row, "parameters")
        dense_parameters = number(control, "parameters")
        compared.append({
            "group": row["group"],
            "name": row["name"],
            "attention": row["attention"],
            "rank": row["rank"],
            "time_tokens": row["time_tokens"],
            "latency_median_ms": latency,
            "latency_speedup_vs_dense": ratio(dense_latency, latency),
            "samples_per_second": throughput,
            "throughput_speedup_vs_dense": ratio(throughput, dense_throughput),
            "throughput_peak_allocated_mib": memory if memory is not None else "",
            "memory_ratio_vs_dense": ratio(memory, dense_memory),
            "training_step_median_ms": training_time,
            "training_speedup_vs_dense": ratio(dense_training_time, training_time),
            "training_peak_allocated_mib": training_memory if training_memory is not None else "",
            "training_memory_ratio_vs_dense": ratio(
                training_memory, dense_training_memory
            ),
            "parameters": int(parameters) if parameters is not None else "",
            "parameter_ratio_vs_dense": ratio(parameters, dense_parameters),
            "attention_macs_ratio_vs_dense": number(
                row, "attention_macs_ratio_vs_dense"
            ),
        })
    return compared


def format_value(value, digits=3):
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(rows, path):
    lines = [
        "| Group | Variant | Latency ms | Latency speedup | Samples/s | Throughput speedup | Inference MiB | Training ms | Training speedup | Training MiB | MAC ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        variant = "dense" if row["attention"] == "dense" else f"R{row['rank']}/TT{row['time_tokens']}"
        lines.append(
            "| {group} | {variant} | {latency} | {lat_speed}x | {throughput} | {thr_speed}x | {memory} | {training} | {train_speed}x | {train_memory} | {mac_ratio} |".format(
                group=row["group"],
                variant=variant,
                latency=format_value(row["latency_median_ms"]),
                lat_speed=format_value(row["latency_speedup_vs_dense"]),
                throughput=format_value(row["samples_per_second"], 1),
                thr_speed=format_value(row["throughput_speedup_vs_dense"]),
                memory=format_value(row["throughput_peak_allocated_mib"], 1),
                training=format_value(row["training_step_median_ms"]),
                train_speed=format_value(row["training_speedup_vs_dense"]),
                train_memory=format_value(row["training_peak_allocated_mib"], 1),
                mac_ratio=format_value(row["attention_macs_ratio_vs_dense"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv")
    args = parser.parse_args()
    source = Path(args.summary_csv)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    compared = compare(rows)

    output_csv = source.with_name("benchmark_comparison.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(compared)
    output_md = source.with_name("benchmark_comparison.md")
    write_markdown(compared, output_md)
    print(output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
