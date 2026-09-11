@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer

echo.
echo ==========================================
echo Starting experiment: ECL_96_96_321
echo ==========================================
  
python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity.csv ^
  --model_id ECL_96_96_321 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 321 ^
  --dec_in 321 ^
  --c_out 321 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.0005 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment ECL_96_96_321 failed.
    echo Continuing with experiment ECL_96_96_321...
) else (
    echo.
    echo Experiment ECL_96_96_321 finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: ECL_96_96_v4_201
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity_v4.csv ^
  --model_id ECL_96_96_v4_201 ^
  --model %model_name% ^
  --data custom ^
  --features M ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 201 ^
  --dec_in 201 ^
  --c_out 201 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.0005 ^
  --itr 1


if errorlevel 1 (
    echo.
    echo WARNING: Experiment ECL_96_96_v4_201 failed.
    echo Continuing with experiment ECL_96_96_v4_201...
) else (
    echo.
    echo Experiment ECL_96_96_v4_201 finished successfully.
)


echo.
echo ==========================================
echo Script finished.
echo ==========================================

pause
