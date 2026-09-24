@echo off
setlocal EnableExtensions

REM Run from repository root
pushd "%~dp0.."
if errorlevel 1 (
    echo ERROR: Cannot enter repository root.
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHON=python"
set "DATA_ROOT=./dataset/weather/"
set "DATA_FILE=weather.csv"

if not exist "%DATA_ROOT%%DATA_FILE%" (
    echo ERROR: Dataset not found:
    echo %DATA_ROOT%%DATA_FILE%
    pause
    popd
    exit /b 1
)

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "START_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "START_SECONDS=%%A"

echo ==========================================================
echo Weather 96-to-720 - Induced Attention R16 - From Scratch
echo Start: %START_TIME%
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --root_path "%DATA_ROOT%" ^
 --data_path "%DATA_FILE%" ^
 --model_id weather_96_720_induced_r16_scratch_seed2024 ^
 --model iTransformer ^
 --data custom ^
 --features M ^
 --seq_len 96 ^
 --pred_len 720 ^
 --e_layers 3 ^
 --enc_in 21 ^
 --dec_in 21 ^
 --c_out 21 ^
 --des Exp ^
 --d_model 512 ^
 --d_ff 512 ^
 --itr 1 ^
 --induced_attention ^
 --attn_rank 16 ^
 --attn_time_tokens 4 ^
 --attn_gate_init 1.0 ^
 --checkpoints ./checkpoints/

set "RUN_EXIT_CODE=%ERRORLEVEL%"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "END_TIME=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "END_SECONDS=%%A"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=%END_SECONDS%-%START_SECONDS%; [TimeSpan]::FromSeconds($d).ToString('hh\:mm\:ss')"') do set "DURATION=%%A"

echo.
echo ==========================================================
echo End: %END_TIME%
echo Duration: %DURATION%
echo Exit code: %RUN_EXIT_CODE%
echo ==========================================================

popd
pause
exit /b %RUN_EXIT_CODE%