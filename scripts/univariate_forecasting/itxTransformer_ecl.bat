@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer

echo.
echo ==========================================
echo Starting experiment: ECL_original_96_96_to_OT
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity.csv ^
  --model_id ECL_original_96_96_to_OT ^
  --model %model_name% ^
  --data custom ^
  --features MS ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 321 ^
  --dec_in 321 ^
  --c_out 1 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.0005 ^
  --itr 1

if errorlevel 1 (
    echo.
    echo WARNING: Experiment ECL_original_96_96_to_OT failed.
    echo Continuing with experiment ECL_original_96_96_to_OT...
) else (
    echo.
    echo Experiment ECL_original_96_96_to_OT finished successfully.
)


echo.
echo ==========================================
echo Starting experiment: ECL_v4_201_96_96_to_OT
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity_v4.csv ^
  --model_id ECL_v4_201_96_96_to_OT ^
  --model %model_name% ^
  --data custom ^
  --features MS ^
  --seq_len 96 ^
  --pred_len 96 ^
  --e_layers 3 ^
  --enc_in 201 ^
  --dec_in 201 ^
  --c_out 1 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.0005 ^
  --itr 1


if errorlevel 1 (
    echo.
    echo WARNING: Experiment ECL_v4_201_96_96_to_OT failed.
    echo Continuing with experiment ECL_v4_201_96_96_to_OT...
) else (
    echo.
    echo Experiment ECL_v4_201_96_96_to_OT finished successfully.
)


echo.
echo ==========================================
echo Script finished.
echo ==========================================

pause
