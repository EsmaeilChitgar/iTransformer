@echo off

set "CUDA_VISIBLE_DEVICES=0"

set "MODE=kv_select"
set "K=0.2"
set "EST=mlp"
set "TEMP=0.1"
set "REG=0.01"

set "root_path=./data/electricity/"
set "data_path=electricity.csv"
set "model_id_prefix=electricity"
set "data_type=custom"

set "model_name=iSparseTransformer"
set SPARSE_FLAGS=--sparse_var_attn --var_select_mode %MODE% --var_select_k %K% --var_select_estimator %EST% --var_select_temp %TEMP% --var_select_reg %REG%

set "seq_len=96"
set "pred_len=96"

python -u run.py ^
  --is_training 1 ^
  --root_path %root_path% ^
  --data_path %data_path% ^
  --model_id %model_id_prefix%_%seq_len%_%pred_len% ^
  --model %model_name% ^
  --data %data_type% ^
  --features M ^
  --seq_len %seq_len% ^
  --label_len 48 ^
  --pred_len %pred_len% ^
  --d_model 512 ^
  --n_heads 8 ^
  --train_epochs 10 ^
  --batch_size 16 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --enc_in 321 ^
  --dec_in 321 ^
  --c_out 321 ^
  --des 'Exp' ^
  --learning_rate 0.0005 ^
  --itr 1 ^
  %SPARSE_FLAGS%