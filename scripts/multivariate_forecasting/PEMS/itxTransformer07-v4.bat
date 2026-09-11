@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer

echo.
echo ==========================================
echo Starting experiment: PEMS07_96_12_883
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS07.npz ^
  --model_id PEMS07_96_12_883 ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 883 ^
  --dec_in 883 ^
  --c_out 883 ^
  --des Exp ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEMS07_96_12_883 failed.
    echo Continuing with experiment PEMS07_96_12_883...
) else (
    echo.
    echo Experiment PEMS07_96_12_883 finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: PEMS07_96_12_v4_601
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS07_v4.npz ^
  --model_id PEMS07_96_12_v4_601 ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 601 ^
  --dec_in 601 ^
  --c_out 601 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --learning_rate 0.001 ^
  --itr 1 ^
  --use_norm 0

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEMS07_96_12_v4_601 failed.
    echo Continuing with experiment PEMS07_96_12_v4_601...
) else (
    echo.
    echo Experiment PEMS07_96_12_v4_601 finished successfully.
)



echo.
echo ==========================================
echo Starting experiment: PEMS07_96_12_883_batch32
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS07.npz ^
  --model_id PEMS07_96_12_883_batch32 ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 883 ^
  --dec_in 883 ^
  --c_out 883 ^
  --des Exp ^
  --d_model 512 ^
  --d_ff 512 ^
  --learning_rate 0.001 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEMS07_96_12_883_batch32 failed.
    echo Continuing with experiment PEMS07_96_12_883_batch32...
) else (
    echo.
    echo Experiment PEMS07_96_12_883_batch32 finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: traffic_96_336_v4_400
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_400.csv ^
  --model_id traffic_96_336_v4_400 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 336 ^
  --e_layers 4 ^
  --enc_in 401 ^
  --dec_in 401 ^
  --c_out 401 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment traffic_96_336_v4_400 failed.
    echo Continuing with experiment traffic_96_336_v4_400...
) else (
    echo.
    echo Experiment traffic_96_336_v4_400 finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: traffic_96_720_v4_400
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/traffic/ ^
  --data_path traffic_v4_400.csv ^
  --model_id traffic_96_720_v4_400 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 720 ^
  --e_layers 4 ^
  --enc_in 401 ^
  --dec_in 401 ^
  --c_out 401 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1


  if errorlevel 1 (
    echo.
    echo WARNING: Experiment traffic_96_720_v4_400 failed.
    echo Continuing with experiment traffic_96_720_v4_400...
) else (
    echo.
    echo Experiment traffic_96_720_v4_400 finished successfully.
)


echo.
echo ==========================================
echo Script finished.
echo ==========================================

pause
