# iTransformer architecture experiments: consolidated report

## Scope and status

This document records the architectural experiments developed in the
repository branches. None of the accuracy percentages below is a measured
benchmark result unless explicitly marked as measured. They are planning
estimates, useful for selecting experiments but not suitable as paper claims.
Final claims require identical data splits, seeds, optimizer settings,
training budgets, and hardware.

## Priority and source ideas

The ranking used for this work emphasizes simplicity first, then a realistic
chance of improving accuracy or reducing the dominant variate-attention cost.

| Rank | Architecture | Main source idea | Primary goal |
| --- | --- | --- | --- |
| 1 | `iLatentTransformer` | Perceiver-style latent bottleneck applied to iTransformer tokens | Reduce O(N^2) variate attention while retaining global information |
| 2 | `iTimeMixerLatentTransformer` | TimeMixer-style temporal decomposition plus the latent bottleneck | Add trend/detail inductive bias while retaining attention savings |
| 3 | `iSimpleWaveletTransformer` | SimpleTM stationary wavelet representation and geometric/wedge similarity | Improve multi-resolution and cross-variate representations |
| 4 | `iSparseTransformer` | Adaptive top-k variable selection and sparse variable attention | Reduce pairwise attention using data-dependent sparsity |
| 5 | `iMultiScalePatchTransformer` | PatchTST-style local patches and multi-scale patch routing | Improve local pattern modeling; complexity depends on patch settings |

The last two branches predate the three-item priority list and are included
for completeness. The project also contains linear-model experiments, but
they are not Transformer architectural variants and are excluded here.

## Implemented solutions

### 1. Latent cross-variate attention

Branch: `iLatentCrossVariateTransformer`.

Each encoder layer compresses N variate tokens into M learned latent tokens,
then projects latent information back to all variates. The score count is
`2*N*M` rather than `N*N` for the latent route, with exact dense attention
kept for small N through a threshold.

Analytical score-count reductions from the implementation:

* N=321, M=32: 80.1% fewer scores;
* N=862, M=32: 92.6% fewer scores.

Synthetic profiling recorded approximately +5.8% parameters and about 7%
lower attention-layer runtime in the tested setup. These are measurements at
one synthetic configuration, not dataset-wide guarantees. Accuracy
expectation before training: -1% to +3% relative MSE versus dense
iTransformer, with the main risk being information loss when M is too small.

### 2. TimeMixer + latent hybrid

Branch: `iTimeMixerLatentTransformer`.

Before inversion, fixed moving-average scales produce trend/detail branches.
A small per-variate router mixes those branches, then the latent attention
module handles cross-variate interaction. This is a lightweight TimeMixer
inspiration rather than a claim of reproducing the original paper exactly.

The latent attention savings are the same as above. Synthetic timing showed
about 33.1% slower at N=128 because the temporal mixer overhead dominates,
and about 6.1% faster at N=321 where attention is more expensive. The model
added approximately 6.0% parameters in that comparison. Expected accuracy
change: -1% to +5% relative MSE, with higher upside on data containing clear
trend/seasonal changes and no guarantee on every dataset.

### 3. SimpleTM-inspired wavelet/geometric model (this branch)

Branch: `iSimpleWaveletTransformer`.

The new model keeps one token per variate and the original forecast head. Its
front-end computes a fixed one-level Haar-style low/detail decomposition,
embeds the pair, and uses a learned per-variate gate for the wavelet residual.
Its attention score is:

```text
score(q,k) = scale * [cos(q,k) + lambda * (1 - ||q wedge k||)]
||q wedge k|| = sqrt(1 - cos(q,k)^2)
```

where `lambda` is `--wavelet_geometric_weight` (default 0.25). This formula is
implemented through the identity, so it does not allocate a four-dimensional
outer-product tensor. The architecture remains O(N^2) in variate attention;
the wavelet branch is O(BLN). Consequently the honest expectation is a small
time increase, roughly 0% to 15% end-to-end depending on N, L, device, and
kernel efficiency, and no analytical attention reduction. Parameter growth
is expected to be roughly 1% to 5% for common iTransformer settings because
the extra projection consumes 2L inputs; measure it for each configuration.
Accuracy is only a hypothesis: approximately -2% to +5% relative MSE, with
the likely benefit concentrated in series with multi-scale or abrupt local
changes. It may be neutral or worse on smooth datasets. These ranges are not
results and must not be reported as achieved improvements.

## Comparison of expected trade-offs

| Model | Accuracy hypothesis | Complexity/time expectation | Main risk |
| --- | --- | --- | --- |
| Dense iTransformer | Baseline | O(N^2), baseline time | Expensive at many variates |
| Latent | -1% to +3% relative MSE | O(NM) score route; measured ~7% faster in one synthetic setup | Bottleneck loses rare variable interactions |
| TimeMixer + latent | -1% to +5% | O(NM) attention plus O(BLN) mixer; +33.1% at N=128 and -6.1% at N=321 in synthetic tests | Extra front-end overhead and coupled ablations |
| SimpleTM-inspired | -2% to +5% | O(N^2) attention plus O(BLN) wavelet branch; estimated 0% to 15% slower | Geometric score may not help all datasets |
| Sparse | Dataset-dependent | O(NK) or O(K^2) selected route | Hard top-k can miss important variables |
| Multi-scale patch | Dataset-dependent | Extra patch projections and routing; prior prototype was +234% at N=128 | Large overhead can erase accuracy value |

The percentages are deliberately conservative and are not additive. A 5%
accuracy estimate means relative MSE change, not percentage points of an
accuracy classification metric. Timing should be reported as median latency
and throughput on the actual target hardware.

## Reproducible evaluation protocol

For a defensible paper study, run dense iTransformer and each candidate with
the same seed list, splits, batch size, epochs, optimizer, early stopping,
and precision mode. Record parameter count, peak memory, training time per
epoch, total training time, warm-up-excluded inference latency, throughput,
MSE, and MAE at every horizon. For the new model also save gate statistics;
for latent/sparse models save selected-token statistics.

Required ablations for this branch are: raw iTransformer, wavelet embedding
only, geometric attention only, and the complete wavelet/geometric model.
Only after these experiments should the estimates above be replaced with
confidence intervals over repeated seeds.
