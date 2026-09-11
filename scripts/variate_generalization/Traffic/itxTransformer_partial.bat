@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer


model_name=iTransformer


echo.
echo ==========================================
echo Starting experiment: traffic_partial_600
echo ==========================================
python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_partial.csv ^
  --model_id traffic_partial_600 ^
  --model iTransformer ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 600 ^
  --dec_in 600 ^
  --c_out 600 ^
  --des 'Exp' ^
  --exp_name partial_train ^
  --itr 1

pause