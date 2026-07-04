#!/bin/bash
# iSparseTransformer: iTransformer + Sparse Variable Attention.
#
# This script reproduces the iTransformer multivariate-forecasting runs but
# swaps in iSparseTransformer and enables the adaptive variable-selection
# module. The selection configuration is exposed via environment variables so
# the same script covers Electricity / Traffic / Weather / ETT:
#
#   MODE  : kv_select (default, O(D*K)) | topk (O(K^2)) | soft (O(D^2) reference)
#   K     : selection budget; ratio of D if in (0,1] (default 0.2), else absolute
#   EST   : variable importance estimator -- linear | mlp (default) | se
#   TEMP  : straight-through temperature (default 0.1)
#   REG   : selection regularizer weight (default 0.01)
#   GPU   : CUDA_VISIBLE_DEVICES (default 0)
#   ONLY  : space-separated list of datasets to run (default: all)
#
# Example:
#   MODE=kv_select K=0.15 GPU=0 ONLY="ECL Traffic" bash scripts/sparse_variable_attention/iSparseTransformer.sh

export CUDA_VISIBLE_DEVICES=${GPU:-0}

MODE=${MODE:-kv_select}
K=${K:-0.2}
EST=${EST:-mlp}
TEMP=${TEMP:-0.1}
REG=${REG:-0.01}
ONLY=${ONLY:-"ECL Traffic Weather ETTh1 ETTh2 ETTm1 ETTm2"}

model_name=iSparseTransformer
SPARSE_FLAGS="--sparse_var_attn --var_select_mode ${MODE} --var_select_k ${K} \
  --var_select_estimator ${EST} --var_select_temp ${TEMP} --var_select_reg ${REG}"

run_pred_lens() {
  # $1 root_path  $2 data_path  $3 data  $4 enc_in  $5 e_layers  $6 d_model  $7 d_ff  $8 batch_size  $9 lr  $10 id_prefix
  root=$1; data=$2; dtype=$3; enc=$4; el=$5; dm=$6; dff=$7; bs=$8; lr=$9; idp=${10}
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 \
      --root_path ${root} \
      --data_path ${data} \
      --model_id ${idp}_96_${pl} \
      --model ${model_name} \
      --data ${dtype} \
      --features M \
      --seq_len 96 \
      --pred_len ${pl} \
      --e_layers ${el} \
      --enc_in ${enc} \
      --dec_in ${enc} \
      --c_out ${enc} \
      --des 'Sparse' \
      --d_model ${dm} \
      --d_ff ${dff} \
      --batch_size ${bs} \
      --learning_rate ${lr} \
      --itr 1 \
      ${SPARSE_FLAGS}
  done
}

for ds in ${ONLY}; do
  echo "=================== ${ds} (mode=${MODE} K=${K}) ==================="
  case ${ds} in
    ECL)
      run_pred_lens ./dataset/electricity/ electricity.csv custom 321 3 512 512 16 0.0005 ECL
      ;;
    Traffic)
      run_pred_lens ./dataset/traffic/ traffic.csv custom 862 4 512 512 16 0.001 traffic
      ;;
    Weather)
      run_pred_lens ./dataset/weather/ weather.csv custom 21 3 512 512 32 0.001 weather
      ;;
    ETTh1)
      run_pred_lens ./dataset/ETT-small/ ETTh1.csv ETTh1 7 2 256 256 32 0.0005 ETTh1
      ;;
    ETTh2)
      run_pred_lens ./dataset/ETT-small/ ETTh2.csv ETTh2 7 2 256 256 32 0.0005 ETTh2
      ;;
    ETTm1)
      run_pred_lens ./dataset/ETT-small/ ETTm1.csv ETTm1 7 3 256 256 32 0.0005 ETTm1
      ;;
    ETTm2)
      run_pred_lens ./dataset/ETT-small/ ETTm2.csv ETTm2 7 3 256 256 32 0.0005 ETTm2
      ;;
    *)
      echo "Unknown dataset: ${ds}"
      ;;
  esac
done
