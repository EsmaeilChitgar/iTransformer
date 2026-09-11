@echo off
set CUDA_VISIBLE_DEVICES=0

@REM echo.
@REM echo ==========================================
@REM echo Starting experiment: traffic_original_to_OT
@REM echo ==========================================
@REM
@REM python -u run.py ^
@REM   --is_training 1 ^
@REM   --root_path ./dataset/traffic/ ^
@REM   --data_path traffic.csv ^
@REM   --model_id traffic_original_to_OT ^
@REM   --model iTransformer ^
@REM   --data custom ^
@REM   --features MS ^
@REM   --target OT ^
@REM   --seq_len 96 ^
@REM   --label_len 48 ^
@REM   --pred_len 96 ^
@REM   --e_layers 2 ^
@REM   --d_layers 1 ^
@REM   --factor 3 ^
@REM   --enc_in 862 ^
@REM   --dec_in 862 ^
@REM   --c_out 1 ^
@REM   --des 'Exp' ^
@REM   --itr 1



echo.
echo ==========================================
echo Starting experiment: traffic_v4_400
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_400.csv ^
  --model_id traffic_v4_400_to_OT ^
  --model iTransformer ^
  --data custom ^
  --features MS ^
  --target OT ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 401 ^
  --dec_in 401 ^
  --c_out 1 ^
  --des 'Exp' ^
  --itr 1


echo.
echo ==========================================
echo Starting experiment: traffic_v4_500
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_500.csv ^
  --model_id traffic_v4_500_to_OT ^
  --model iTransformer ^
  --data custom ^
  --features MS ^
  --target OT ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 501 ^
  --dec_in 501 ^
  --c_out 1 ^
  --des 'Exp' ^
  --itr 1


echo.
echo ==========================================
echo Starting experiment: traffic_original_to_OT_192
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic.csv ^
  --model_id traffic_original_to_OT_192 ^
  --model iTransformer ^
  --data custom ^
  --features MS ^
  --target OT ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 192 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 1 ^
  --des 'Exp' ^
  --itr 1


echo.
echo ==========================================
echo Starting experiment: traffic_v4_400_to_OT_192
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_400.csv ^
  --model_id traffic_v4_400_to_OT_192 ^
  --model iTransformer ^
  --data custom ^
  --features MS ^
  --target OT ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 192 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 401 ^
  --dec_in 401 ^
  --c_out 1 ^
  --des 'Exp' ^
  --itr 1

pause