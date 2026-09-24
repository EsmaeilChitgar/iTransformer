@echo off
setlocal EnableExtensions
pushd "%~dp0.."
if errorlevel 1 exit /b 1

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"

if not exist "./dataset/electricity/electricity.csv" (
    echo ERROR: ECL dataset not found.
    pause
    exit /b 1
)

for /f %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "START_SECONDS=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "START_TIME=%%A"

echo ==========================================================
echo ECL 96-to-96 - Induced R256 - Training From Scratch
echo rank 256
echo Start: %START_TIME%
echo ==========================================================

call "%PYTHON%" -u run.py ^
  --is_training 1 ^
  --model_id ECL_96_96_induced_rank256_s2024 ^
  --model iTransformer ^
  --data custom ^
  --root_path ./dataset/electricity/ ^
  --data_path electricity.csv ^
  --features M ^
  --target OT ^
  --freq h ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --enc_in 321 ^
  --dec_in 321 ^
  --c_out 321 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 3 ^
  --d_layers 1 ^
  --d_ff 512 ^
  --factor 1 ^
  --dropout 0.1 ^
  --embed timeF ^
  --batch_size 32 ^
  --learning_rate 0.0005 ^
  --train_epochs 10 ^
  --patience 3 ^
  --num_workers 0 ^
  --itr 1 ^
  --des Exp ^
  --induced_attention ^
  --attn_rank 256 ^
  --attn_time_tokens 4 ^
  --attn_gate_init 1.0 ^
  --checkpoints ./checkpoints/

set "RUN_RESULT=%ERRORLEVEL%"

for /f %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "END_SECONDS=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "END_TIME=%%A"

set /a "ELAPSED=END_SECONDS-START_SECONDS"
set /a "HH=ELAPSED/3600"
set /a "MM=(ELAPSED%%3600)/60"
set /a "SS=ELAPSED%%60"

echo ==========================================================
echo End: %END_TIME%
echo Duration: %HH%:%MM%:%SS%
echo Exit code: %RUN_RESULT%
echo ==========================================================

popd
pause
exit /b %RUN_RESULT%