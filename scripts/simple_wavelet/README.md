# SimpleTM-inspired wavelet/geometric iTransformer

`iSimpleWaveletTransformer` is the third-ranked architecture in this project.
It is inspired by the stationary wavelet representation and geometric/wedge
attention used by [SimpleTM](https://github.com/vsingh-group/SimpleTM).

The implementation has three additions to iTransformer:

1. A fixed one-level Haar-style transform produces length-preserving low- and
   high-frequency signals in O(BLN) time.
2. A small input-conditioned gate mixes the wavelet embedding into the normal
   inverted embedding independently for each variate.
3. Geometric attention adds a wedge-product similarity to the standard cosine
   score. The wedge magnitude is computed using
   `sqrt(1 - cosine(query, key)^2)`, avoiding an explicit exterior-product
   tensor.

This is an accuracy/representation experiment, not a complexity-reduction
claim. The attention graph remains dense, O(N^2) in the number of variates,
and the extra temporal branch adds a small O(BLN) cost. Therefore it should
not be advertised as faster than iTransformer without measured benchmarks.

## Run

```bash
python tests/test_simple_wavelet_transformer.py
python -u run.py --is_training 1 --model_id simplewavelet_ecl \
  --model iSimpleWaveletTransformer --data ECL \
  --root_path ./data/electricity/ --data_path electricity.csv \
  --features M --seq_len 96 --pred_len 96 \
  --wavelet_geometric_weight 0.25
```

For an ablation study, compare the same seed and training budget across:

* iTransformer;
* wavelet embedding with `--wavelet_geometric_weight 0`;
* geometric attention only in a controlled fork;
* the complete model.

Report MSE/MAE by horizon, parameter count, peak memory, median warm-up
latency, throughput, and the learned gate distribution. The percentages in
the project report are estimates or analytical counts until these runs are
completed on the target datasets and hardware.
