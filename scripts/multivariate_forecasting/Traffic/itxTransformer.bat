@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer_Latent

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --model_id traffic_96_96 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 4 ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 862 ^
  --des Exp ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1

pause