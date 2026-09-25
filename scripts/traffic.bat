@echo off
setlocal EnableExtensions
pushd "%~dp0.."
if errorlevel 1 exit /b 1

if not exist ".\dataset\traffic\traffic.csv" (
    echo ERROR: Put traffic.csv in dataset\traffic\traffic.csv
    popd
    exit /b 1
)

rem Activate your Python environment before running this script.
rem Both invocations start from seed 2023 and train from scratch.
call :RUN traffic_96_96_RSL_a0 0
if errorlevel 1 goto :FAILED
call :RUN traffic_96_96_RSL_a05 0.5
if errorlevel 1 goto :FAILED

echo Done. Compare mse/mae in result_long_term_forecast.txt.
popd
exit /b 0

:RUN
echo ==============================================================
echo TRAIN FROM SCRATCH: %~1  alpha=%~2  variance=0.90
echo ==============================================================
python -u run.py ^
  --is_training 1 ^
  --model_id %~1 ^
  --model iTransformer ^
  --data custom ^
  --root_path .\dataset\traffic\ ^
  --data_path traffic.csv ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 862 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 4 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --factor 1 ^
  --dropout 0.1 ^
  --activation gelu ^
  --embed timeF ^
  --use_norm 1 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --patience 3 ^
  --num_workers 1 ^
  --itr 1 ^
  --des rsl ^
  --rsl_alpha %~2 ^
  --rsl_variance_ratio 0.90
exit /b %errorlevel%

:FAILED
echo ERROR: Training stopped. No result claimed for a failed run.
popd
exit /b 1
