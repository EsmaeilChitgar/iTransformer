# Sparse Variable Attention for iTransformer (iSparseTransformer)

This directory adds **iSparseTransformer**, a natural extension of iTransformer
that targets its remaining bottleneck: self-attention is performed over the `D`
variates and therefore scales as **O(D²)** in both FLOPs and memory. For
high-dimensional datasets (Electricity: 321, Traffic: 862) this quadratic cost in
the number of variables dominates.

## Design

iTransformer inverts the input so each *variable* is one token. Everything is
preserved (inverted embedding, encoder blocks, FFN, LayerNorm, non-stationary
normalization, projection head, training pipeline). The only change is **how
variable tokens participate in self-attention**.

Before every self-attention layer we insert a lightweight, sample-dependent
module:

```
variable tokens  ->  Importance Estimator  ->  Adaptive (differentiable) Top-K
Selection  ->  Sparse Variable Attention  ->  residual  ->  FFN  ->  next block
```

1. **Variable Importance Estimator** (`layers/Sparse_Variable_Attention.py`): a
   per-token scorer (`linear` / `mlp` / squeeze-`se`) maps each variable token
   `[B, D, d_model]` to a scalar importance score `[B, D]`. It mixes no
   variables, so it is independent of `D` and costs `O(D·d_model)`.

2. **Adaptive Variable Selection**: a *differentiable straight-through top-K*
   selects exactly `K` variables per sample. The forward pass uses a hard top-K
   gather (real FLOPs/memory savings); the backward pass routes gradients through
   a sigmoid relaxation of the mask, so the scorer is trained end-to-end with
   stable (bounded) gradients. No hand-engineered thresholds. A small budget +
   entropy regularizer (`--var_select_reg`) shapes the selection distribution.

3. **Sparse Variable Attention** — only the selected tokens participate in the
   expensive pairwise attention; the rest bypass via the block residual.

| `--var_select_mode` | attention graph | complexity | notes |
|---|---|---|---|
| `kv_select` (default) | all `D` queries × top-`K` keys | **O(D·K)** | best accuracy/speed trade-off; every variable still gets an attention update |
| `topk` | top-`K` × top-`K` block | **O(K²)** | largest speed-up; non-selected tokens bypass attention in that layer |
| `soft` | soft mask, no gather | O(D²) | fully-differentiable accuracy reference (no speed-up) |

`K` is set by `--var_select_k`: a ratio of `D` if in `(0, 1]` (default `0.2`),
otherwise an absolute count. With `K ≈ √D`, `topk` reaches **O(D)** and
`kv_select` reaches **O(D^1.5)**.

## Configuration

```
--model iSparseTransformer --sparse_var_attn
--var_select_mode {kv_select,topk,soft}      # default kv_select
--var_select_k        FLOAT                  # default 0.2 (ratio of D)
--var_select_estimator {linear,mlp,se}       # default mlp
--var_select_temp     FLOAT                  # default 0.1
--var_select_reg      FLOAT                  # default 0.01
```

With `--sparse_var_attn` omitted, iSparseTransformer is bit-for-bit the original
iTransformer, and **original iTransformer checkpoints load** into all shared
submodules (the importance estimator is the only freshly-initialized component).

## Running

```bash
# Electricity, all horizons, kv_select with K = 0.2*D
ONLY="ECL" bash scripts/sparse_variable_attention/iSparseTransformer.sh

# Traffic, topk mode, K ≈ sqrt(D) ≈ 29
ONLY="Traffic" MODE=topk K=29 bash scripts/sparse_variable_attention/iSparseTransformer.sh

# Compare against the original iTransformer:
bash scripts/multivariate_forecasting/ECL/iTransformer.sh
```

## Profiling / evaluation

```bash
# Analytical attention-FLOPs reduction, peak GPU memory, train/inference speed,
# and the number of selected variables (K) on synthetic data:
python -u utils/sparse_profile.py --n_vars 321 --batch_size 32
python -u utils/sparse_profile.py --n_vars 862 --batch_size 16 --var_select_mode kv_select --var_select_k 0.15
```

Report per dataset:

| metric | how it is obtained |
|---|---|
| Number of selected variables (`K`) / keep ratio | `model.encoder.attn_layers[i].attention.last_num_selected` / `last_keep_ratio` (also printed by the profiler) |
| Attention FLOPs reduction | `utils/sparse_profile.py` (analytical, per layer) |
| GPU memory reduction | `utils/sparse_profile.py` (peak `torch.cuda.max_memory_allocated`) |
| Training / inference speed | `utils/sparse_profile.py` (wall-clock) |
| Forecasting performance (MSE/MAE) | standard `run.py` test output vs the iTransformer baselines |

## Files

- `layers/Sparse_Variable_Attention.py` — reusable module (importance estimator,
  differentiable top-K, sparse attention layer).
- `model/iSparseTransformer.py` — model (iTransformer + the sparse layer, gated
  by `--sparse_var_attn`).
- `experiments/exp_basic.py`, `run.py` — registration + config flags.
- `experiments/exp_long_term_forecasting.py` — adds the selection regularizer to
  the training loss (train only; validation/test use the plain criterion).
- `scripts/sparse_variable_attention/iSparseTransformer.sh` — run scripts.
- `utils/sparse_profile.py`, `tests/test_sparse_variable_attention.py` —
  profiling and smoke test.
