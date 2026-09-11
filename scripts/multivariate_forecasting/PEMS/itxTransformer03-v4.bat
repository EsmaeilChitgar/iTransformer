@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer

echo.
echo ==========================================
echo Starting experiment: PEMS03_96_12_358
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS03.npz ^
  --model_id PEMS03_96_12_358 ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 358 ^
  --dec_in 358 ^
  --c_out 358 ^
  --des Exp ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEMS03_96_12_358 failed.
    echo Continuing with experiment PEMS03_96_12_358...
) else (
    echo.
    echo Experiment PEMS03_96_12_358 finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: PEMS03_96_12_v4_251
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS03_v4.npz ^
  --model_id PEMS03_96_12_v4_251 ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 251 ^
  --dec_in 251 ^
  --c_out 251 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --learning_rate 0.001 ^
  --itr 1 ^
  --use_norm 0

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEMS03_96_12_v4_251 failed.
    echo Continuing with experiment PEMS03_96_12_v4_251...
) else (
    echo.
    echo Experiment PEMS03_96_12_v4_251 finished successfully.
)


echo.
echo ==========================================
echo Script finished.
echo ==========================================

pause
