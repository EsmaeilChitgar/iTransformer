# Latent cross-variate iTransformer

The original iTransformer attends directly over all variate tokens. This
branch adds `iLatentTransformer`, which introduces a small learned set of
latent tokens in every encoder layer:

```text
variates --(latent-to-variate attention)--> latent tokens
latent tokens --(variate-to-latent attention)--> variate updates
```

With `N` variates and `M` latent tokens, the dense attention score count
changes from `N^2` to `2*N*M`. For example, `N=321, M=32` gives 90.0% fewer
attention scores; `N=862, M=32` gives 92.6% fewer scores. This is an
attention-memory/FLOP claim, not a guaranteed end-to-end speed claim.

For small datasets, the implementation automatically uses exact full
attention when the token count is at most
`--latent_full_attention_threshold` (default 64). This avoids imposing a
bottleneck on datasets such as Weather with relatively few variates.

## Run

```bash
python tests/test_latent_transformer.py
python -u run.py --is_training 1 --model_id latent_ecl \
  --model iLatentTransformer --data ECL \
  --root_path ./data/electricity/ --data_path electricity.csv \
  --features M --seq_len 96 --pred_len 96 --num_latents 32
```

The minimum publishable evaluation should compare full iTransformer,
iLatentTransformer with `M` in `{16, 32, 64}`, and the existing sparse branch
using identical seeds and training budgets. Report MSE/MAE, parameter count,
peak memory, median GPU latency after warm-up, and the fraction of batches
using the full versus latent route.
