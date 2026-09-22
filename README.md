# iTransformer

The repo is the official implementation for the paper: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625). [[Slides]](https://cloud.tsinghua.edu.cn/f/175ff98f7e2d44fbbe8e/), [[Poster]](https://cloud.tsinghua.edu.cn/f/36a2ae6c132d44c0bd8c/), [[Intro (CN)]](https://mp.weixin.qq.com/s/-pvBnA1_NSloNxa6TYXTSg).
.


# Updates

:triangular_flag_on_post: **News** (2024.10) [TimeXer](https://arxiv.org/abs/2402.19072), a Transformer for predicting with exogenous variables, is released. Code is available [here](https://github.com/thuml/TimeXer). 

:triangular_flag_on_post: **News** (2024.05) Many thanks for the great efforts from [lucidrains](https://github.com/lucidrains/iTransformer). A pip package for the usage of iTransformer variants can be simply installed via ```pip install iTransformer```

:triangular_flag_on_post: **News** (2024.04) iTransformer has benn included in [NeuralForecast](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/itransformer.py). Special thanks to the contributor @[Marco](https://github.com/marcopeix)!

:triangular_flag_on_post: **News** (2024.03) Introduction of our work in [Chinese](https://mp.weixin.qq.com/s/-pvBnA1_NSloNxa6TYXTSg) is available.

:triangular_flag_on_post: **News** (2024.02) iTransformer has been accepted as **ICLR 2024 Spotlight**.

:triangular_flag_on_post: **News** (2023.12) iTransformer available in [GluonTS](https://github.com/awslabs/gluonts/pull/3017) with probablistic head and support for static covariates. Notebook is available [here](https://github.com/awslabs/gluonts/blob/dev/examples/iTransformer.ipynb).

:triangular_flag_on_post: **News** (2023.12) We received lots of valuable suggestions. A [revised version](https://arxiv.org/pdf/2310.06625v2.pdf) (**24 Pages**) is now available.

:triangular_flag_on_post: **News** (2023.10) iTransformer has been included in [[Time-Series-Library]](https://github.com/thuml/Time-Series-Library) and achieves state-of-the-art in Lookback-$96$ forecasting.

:triangular_flag_on_post: **News** (2023.10) All the scripts for the experiments in our [paper](https://arxiv.org/pdf/2310.06625.pdf) are available.


## Introduction

🌟 Considering the characteristics of multivariate time series, iTransformer breaks the conventional structure without modifying any Transformer modules. **Inverted Transformer is all you need in MTSF**.

<p align="center">
<img src="./figures/motivation.png"  alt="" align=center />
</p>

🏆 iTransformer achieves the comprehensive state-of-the-art in challenging multivariate forecasting tasks and solves several pain points of Transformer on extensive time series data.

<p align="center">
<img src="./figures/radar.png" height = "360" alt="" align=center />
</p>


## Overall Architecture

iTransformer regards **independent time series as variate tokens** to **capture multivariate correlations by attention** and **utilize layernorm and feed-forward networks to learn series representations**.

<p align="center">
<img src="./figures/architecture.png" alt="" align=center />
</p>

The pseudo-code of iTransformer is as simple as the following:

<p align="center">
<img src="./figures/algorithm.png" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare the performance of Transformer and iTransformer
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Train the model with partial variates, and generalize to the unseen variates
bash ./scripts/variate_generalization/ECL/iTransformer.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/iTransformer.sh

# Utilize FlashAttention for acceleration
bash ./scripts/efficient_attentions/iFlashTransformer.sh
```

## Main Result of Multivariate Forecasting

We evaluate the iTransformer on challenging multivariate forecasting benchmarks (**generally hundreds of variates**). **Comprehensive good performance** (MSE/MAE $\downarrow$) is achieved.



### Online Transaction Load Prediction of Alipay Trading Platform (Avg Results) 

<p align="center">
<img src="./figures/main_results_alipay.png" alt="" align=center />
</p>

## General Performance Boosting on Transformers

By introducing the proposed framework, Transformer and its variants achieve **significant performance improvement**, demonstrating the **generality of the iTransformer approach** and **benefiting from efficient attention mechanisms**.

<p align="center">
<img src="./figures/boosting.png" alt="" align=center />
</p>

## Zero-Shot Generalization on Variates

**Technically, iTransformer is able to forecast with arbitrary numbers of variables**. We train iTransformers on partial variates and forecast unseen variates with good generalizability.

<p align="center">
<img src="./figures/generability.png" alt="" align=center />
</p>

## Model Analysis

Benefiting from inverted Transformer modules: 

- (Left) Inverted Transformers learn **better time series representations** (more similar [CKA](https://github.com/jayroxis/CKA-similarity)) favored by forecasting.
- (Right) The inverted self-attention module learns **interpretable multivariate correlations**.

<p align="center">
<img src="./figures/analysis.png" alt="" align=center />
</p>

## Low-rank cross-variate research path

This fork contains two separate tools for studying low-rank interaction.  They
answer different questions and their results should not be mixed:

1. `--rank_analysis` applies a per-input truncated SVD to the trained dense
   attention matrix.  It is an **oracle diagnostic**: it still constructs the
   full matrix and is not an efficient implementation.
2. `--induced_attention` replaces dense cross-variate attention with a
   trainable read/write bottleneck containing `--attn_rank` latent
   interactions.  All variate queries remain in the model.  Trailing temporal
   covariate tokens can bypass the bottleneck with `--attn_time_tokens`.

For Traffic with hourly time features, the practical starting point is rank 64
and four exact temporal tokens:

```bash
python run.py --is_training 1 --model_id traffic_induced_r64 \
  --model iTransformer --data custom --root_path ./dataset/traffic/ \
  --data_path traffic.csv --features M --seq_len 96 --label_len 48 \
  --pred_len 96 --enc_in 862 --dec_in 862 --c_out 862 --d_model 512 \
  --n_heads 8 --e_layers 4 --d_ff 512 --batch_size 16 \
  --learning_rate 0.001 --induced_attention --attn_rank 64 \
  --attn_time_tokens 4
```

Use `--warm_start_checkpoint <path>` to copy compatible projection,
feed-forward, normalization, embedding, and prediction weights from a dense
checkpoint.  The inducing queries and per-head output gates remain newly
initialized and must be fine-tuned.

Run a deterministic, paired oracle analysis on a dense checkpoint with:

```bash
python run.py --is_training 0 --model_id traffic_rank_analysis \
  --model iTransformer --data custom --root_path ./dataset/traffic/ \
  --data_path traffic.csv --features M --seq_len 96 --label_len 48 \
  --pred_len 96 --enc_in 862 --dec_in 862 --c_out 862 --d_model 512 \
  --n_heads 8 --e_layers 4 --d_ff 512 --batch_size 16 \
  --rank_analysis --checkpoint_path <checkpoint.pth> \
  --rank_analysis_split test --rank_analysis_ranks 8,16,32,64,128
```

The analysis writes per-window paired errors, raw and row-centered spectral
ranks, `AV` approximation errors, and bootstrap confidence intervals.  A zero
`--rank_analysis_max_batches` uses the whole split.  Use a positive value only
for a smoke test.

Measure the two attention kernels on the target hardware with:

```bash
python tests/benchmark_variate_attention.py --device cuda --tokens 866 \
  --batch_size 16 --heads 8 --head_dim 64 --rank 64 --time_tokens 4
```

End-to-end model latency must still be measured separately because embedding,
feed-forward layers, and data transfer limit the attainable model speedup.

## Citation

If you find this repo helpful, please cite our paper. 

```
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- lucidrains (https://github.com/lucidrains/iTransformer)

This work was supported by Ant Group through the CCF-Ant Research Fund and awarded as [Outstanding Projects of CCF Fund](https://mp.weixin.qq.com/s/PDLNbibZD3kqhcUoNejLfA).

## Contact

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Haoran Zhang (z-hr20@mails.tsinghua.edu.cn)
* Tengge Hu (htg21@mails.tsinghua.edu.cn)
