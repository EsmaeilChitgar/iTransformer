@echo off
setlocal EnableExtensions

pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

set "DATA_ROOT=./dataset/traffic/"
set "DATA_FILE=traffic.csv"



if not exist "%DATA_ROOT%%DATA_FILE%" (
    echo ERROR: Dataset not found:
    echo %DATA_ROOT%%DATA_FILE%
    pause
    exit /b 1
)

REM ==========================================================
REM ثبت زمان شروع آزمایش
REM ==========================================================

for /f "delims=" %%A in (
    'powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"'
) do set "START_TIME=%%A"

for /f "delims=" %%A in (
    'powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"'
) do set "START_MS=%%A"

echo.
echo ==========================================================
echo آزمایش شروع شد
echo Start time: %START_TIME%
echo ==========================================================
echo Training Induced iTransformer - Traffic 96-to-96
echo Rank: 64
echo Iterations: 1
echo ==========================================================


call "%PYTHON%" -u run.py ^
  --is_training 1 ^
  --model_id traffic_96_96_induced_r64_s2023_64_induced_false ^
  --model iTransformer ^
  --data custom ^
  --root_path "%DATA_ROOT%" ^
  --data_path "%DATA_FILE%" ^
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
  --embed timeF ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --train_epochs 10 ^
  --patience 3 ^
  --num_workers 1 ^
  --itr 1 ^
  --induced_attention false^
  --attn_rank 64 ^
  --attn_time_tokens 4 ^
  --attn_gate_init 1.0 ^
  --checkpoints ./checkpoints/

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo TRAINING FAILED
    echo ==========================================================
    call :SHOW_TIME
    pause
    popd
    exit /b 1
)

echo.
echo ==========================================================
echo TRAINING AND TESTING COMPLETED
echo Send result_long_term_forecast.txt and the new checkpoint.
echo ==========================================================

call :SHOW_TIME

popd
pause
exit /b 0


REM ==========================================================
REM نمایش زمان پایان و مدت اجرای آزمایش
REM ==========================================================

:SHOW_TIME

for /f "delims=" %%A in (
    'powershell -NoProfile -Command "(Get-Date).To-MM-dd HH:mm:ss')"'
) do set "END_TIME "END_TIME=%%A"

for /f "delims=" %%A in (
    'powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"'
) do set "END_MS=%%A"

for /f "delims=" %%A in (
    'powershell -NoProfile -Command "$d=([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - %START_MS%)/1000; $h=[int]($d/3600); $m=[int](($d%%3600)/60); $s=[int]($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"'
) do set "DURATION=%%A"

echo.
echo ==========================================================
echo زمان‌بندی آزمایش
echo ==========================================================
echo Start time : %START_TIME%
echo End time   : %END_TIME%
echo Duration   : %DURATION%
echo ==========================================================

exit /b 0
