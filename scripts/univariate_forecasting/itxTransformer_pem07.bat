@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=iTransformer

:: --------------------------------------------------
:: آزمایش اول
:: --------------------------------------------------
echo.
echo ==========================================
echo Starting experiment: PEM_original_96_12_to_OT
echo Start Time: %date% %time%
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS07.npz ^
  --model_id PEM_original_96_12_to_OT ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 883 ^
  --dec_in 883 ^
  --c_out 1 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --learning_rate 0.001 ^
  --itr 1 ^
  --use_norm 0

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEM_original_96_12_to_OT failed.
    echo End Time: %date% %time%
) else (
    echo.
    echo Experiment PEM_original_96_12_to_OT finished successfully.
    echo End Time: %date% %time%
)


:: --------------------------------------------------
:: آزمایش دوم
:: --------------------------------------------------
echo.
echo ==========================================
echo Starting experiment: PEM07_v4_96_12_601_to_OT
echo Start Time: %date% %time%
echo ==========================================

python -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/PEMS/ ^
  --data_path PEMS07_v4.npz ^
  --model_id PEM07_v4_96_12_601_to_OT ^
  --model %model_name% ^
  --data PEMS ^
  --features M ^
  --seq_len 96 ^
  --pred_len 12 ^
  --e_layers 2 ^
  --enc_in 601 ^
  --dec_in 601 ^
  --c_out 1 ^
  --des 'Exp' ^
  --d_model 512 ^
  --d_ff 512 ^
  --learning_rate 0.001 ^
  --itr 1 ^
  --use_norm 0

if errorlevel 1 (
    echo.
    echo WARNING: Experiment PEM07_v4_96_12_601_to_OT failed.
    echo End Time: %date% %time%
) else (
    echo.
    echo Experiment PEM07_v4_96_12_601_to_OT finished successfully.
    echo End Time: %date% %time%
)


echo.
echo ==========================================
echo Script finished at %date% %time%
echo ==========================================

pause
