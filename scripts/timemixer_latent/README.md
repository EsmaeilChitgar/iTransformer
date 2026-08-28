# TimeMixer + latent iTransformer

This branch implements the second-ranked hybrid idea: a lightweight
TimeMixer-inspired temporal decomposition followed by latent cross-variate
attention.

The temporal front-end computes identity, trend, and detail components at
several fixed moving-average scales (`3,7,15` by default). A small router
selects a mixture per variate using mean, standard deviation, and latest
change. It remains linear in the lookback length for fixed scales and adds no
large embedding projections.

The resulting tokens are processed by the latent attention bottleneck. With
`N` variates and `M` latent tokens, attention uses `2*N*M` score entries rather
than `N*N`; this is 80.1% fewer for `N=321,M=32` and 92.6% fewer for
`N=862,M=32`. Small token sets use dense attention automatically.

## Run

```bash
python tests/test_timemixer_latent_transformer.py
python -u run.py --is_training 1 --model_id timemixer_latent_ecl \
  --model iTimeMixerLatentTransformer --data ECL \
  --root_path ./data/electricity/ --data_path electricity.csv \
  --features M --seq_len 96 --pred_len 96 \
  --temporal_scales 3,7,15 --num_latents 32
```

Evaluate against dense iTransformer, latent iTransformer, and the hybrid with
the same seeds. Report MSE/MAE, wall-clock latency, peak memory, parameters,
and the selected temporal-scale weights.
