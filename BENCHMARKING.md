# Full-model paper compute benchmark

Run from an activated CUDA environment:

```powershell
.\scripts\paper_benchmark.bat
```

The batch performs no training and reads no datasets. It measures the exact
iTransformer forward graph with synthetic tensors of the same shapes as the
reported experiments. Random values and trained weights have the same runtime
and memory complexity, so excluding data loading gives a cleaner model-level
comparison.

Each configuration records:

- batch-1 median, mean, standard deviation, and p95 latency;
- throughput at the training batch size;
- forward-plus-backward training-step time at the same batch size;
- peak CUDA allocated and reserved memory;
- parameters and parameter storage;
- analytical attention MACs for all encoder layers.

Results are written to a timestamped directory under `benchmark_results/`.
Use `benchmark_comparison.csv` for tables and keep `RUN_INFO.txt`, the raw JSON
files, and `benchmark_summary.csv` as reproducibility evidence.

The default matrix is deliberately small: Traffic 96 supplies the rank scaling
curve, Traffic 720 checks long-horizon projection cost, PEMS03 48 is an
independent multivariate dataset, and ECL 96 exposes the small-token boundary
where low-rank attention may not be faster.

The original VarDrop test path runs its dense `OURS` model on every variate;
without its optional LPRA extension that model is code-equivalent to dense
iTransformer. Consequently the matching dense row is also the VarDrop
inference result. VarDrop training remains different because it samples
variates during training, so its end-to-end training duration should be taken
from the existing VarDrop run logs rather than this synthetic training step.
