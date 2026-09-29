"""CPU smoke tests for the full-model benchmark helper."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.benchmark_forecasting_model import attention_macs, build_config, infer_group
from tests.summarize_paper_benchmark import compare
from model.iTransformer import Model


def arguments(attention):
    return SimpleNamespace(
        attention=attention,
        rank=4,
        time_tokens=2,
        gate_init=1.0,
        seq_len=12,
        pred_len=6,
        enc_in=10,
        time_features=2,
        d_model=16,
        n_heads=4,
        e_layers=2,
        d_ff=16,
        dropout=0.1,
        use_norm=1,
    )


def test_attention_macs_are_reduced():
    args = arguments("induced")
    selected, dense = attention_macs(args)
    assert selected < dense


def test_dense_and_induced_output_shapes_match():
    import torch

    x = torch.randn(2, 12, 10)
    x_mark = torch.randn(2, 12, 2)
    outputs = []
    for attention in ("dense", "induced"):
        args = arguments(attention)
        model = Model(build_config(args)).eval()
        with torch.inference_mode():
            outputs.append(model(x, x_mark, None, None))
    assert outputs[0].shape == outputs[1].shape == (2, 6, 10)


def test_pairing_computes_speedup():
    common = {
        "group": "traffic_96",
        "time_tokens": "",
        "throughput_peak_allocated_mib": "100",
        "parameters": "1000",
        "attention_macs_ratio_vs_dense": "1",
        "training_step_median_ms": "20",
        "training_peak_allocated_mib": "200",
    }
    rows = [
        dict(common, name="traffic_96_dense", attention="dense", rank="",
             latency_median_ms="10", samples_per_second="100"),
        dict(common, name="traffic_96_r8_tt4", attention="induced", rank="8",
             time_tokens="4", latency_median_ms="5", samples_per_second="250",
             throughput_peak_allocated_mib="60", parameters="800",
             training_step_median_ms="10", training_peak_allocated_mib="120",
             attention_macs_ratio_vs_dense="0.1"),
    ]
    result = compare(rows)
    assert infer_group("traffic_96_r8_tt4") == "traffic_96"
    assert result[1]["latency_speedup_vs_dense"] == 2.0
    assert result[1]["throughput_speedup_vs_dense"] == 2.5
    assert result[1]["training_speedup_vs_dense"] == 2.0
