# Multi-Scale Patch Inverted Transformer

`iMultiScalePatchTransformer` is an architectural extension of iTransformer
motivated by PatchTST's local patch representation. Instead of mapping the
entire lookback window directly to one variate token, it:

1. extracts overlapping patches at several temporal scales;
2. mixes neighboring patches with a depthwise temporal operator;
3. pools each scale with learned patch attention; and
4. routes each variate across scales with an input-conditioned gate.

The resulting `[batch, variates, d_model]` tokens use the original iTransformer
encoder and projection head. Timestamp features remain additional tokens, so
the existing multivariate and arbitrary-variate interfaces are preserved.

## Smoke test

```bash
python tests/test_multiscale_patch_transformer.py
```

## Training example

```bash
python -u run.py \
  --is_training 1 --model_id patch_ecl --model iMultiScalePatchTransformer \
  --data ECL --root_path ./data/electricity/ --data_path electricity.csv \
  --features M --seq_len 96 --pred_len 96 \
  --patch_sizes 8,16,32,64 --patch_stride_ratio 0.5 \
  --e_layers 2 --d_model 512 --d_ff 2048 --n_heads 8
```

For a publishable comparison, keep the optimizer, seed, data split, and
encoder settings fixed while reporting: dense iTransformer, one patch scale,
and the routed multi-scale model. Also report patch-routing weights, parameter
count, wall-clock time, and performance by forecast horizon.
