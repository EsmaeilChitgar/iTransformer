@echo off
setlocal EnableExtensions

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

set "ATTN_RANK=16"
set "TIME_TOKENS=4"
set "GATE_INIT=1.0"

if not exist "%DATA_ROOT%%DATA_FILE%" (
    echo ERROR: Dataset not found:
    echo %DATA_ROOT%%DATA_FILE%
    popd
    pause
    exit /b 1
)

call "%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available.
    popd
    pause
    exit /b 1
)

REM ==========================================================
REM Total start time
REM ==========================================================

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "TOTAL_START_TIME=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "TOTAL_START_MS=%%A"

echo.
echo ==========================================================
echo Weather Induced iTransformer Research Batch
echo Training from scratch
echo Sequence length: 96
echo Prediction lengths: 720
echo Rank: %ATTN_RANK%
echo Time tokens: %TIME_TOKENS%
echo Batch start: %TOTAL_START_TIME%
echo ==========================================================

REM ==========================================================
REM Run all prediction horizons
REM ==========================================================

@REM call :RUN_EXPERIMENT 96
@REM if errorlevel 1 goto :BATCH_FAILED
@REM
@REM call :RUN_EXPERIMENT 192
@REM if errorlevel 1 goto :BATCH_FAILED

call :RUN_EXPERIMENT 720
if errorlevel 1 goto :BATCH_FAILED

set "BATCH_STATUS=COMPLETED"
set "BATCH_EXIT_CODE=0"
goto :ALL_DONE


:BATCH_FAILED
set "BATCH_STATUS=FAILED"
set "BATCH_EXIT_CODE=1"


:ALL_DONE

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "TOTAL_END_TIME=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "TOTAL_END_MS=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=(%TOTAL_END_MS%-%TOTAL_START_MS%)/1000; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=[math]::Floor($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "TOTAL_DURATION=%%A"

echo.
echo ==========================================================
echo BATCH SUMMARY
echo ==========================================================
echo Status         : %BATCH_STATUS%
echo Batch start    : %TOTAL_START_TIME%
echo Batch end      : %TOTAL_END_TIME%
echo Total duration : %TOTAL_DURATION%
echo Exit code      : %BATCH_EXIT_CODE%
echo ==========================================================

popd
pause
exit /b %BATCH_EXIT_CODE%


REM ==========================================================
REM Run one Weather experiment
REM Argument 1 = prediction length
REM ==========================================================

:RUN_EXPERIMENT

set "PRED_LEN=%~1"
set "EXP_NAME=weather_96_%PRED_LEN%_induced_r%ATTN_RANK%_tt%TIME_TOKENS%_scratch"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_START_TIME=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_START_MS=%%A"

echo.
echo ==========================================================
echo EXPERIMENT START
echo Model ID       : %EXP_NAME%
echo Sequence       : 96
echo Prediction     : %PRED_LEN%
echo Encoder layers : 3
echo Rank           : %ATTN_RANK%
echo Time tokens    : %TIME_TOKENS%
echo Start time     : %EXP_START_TIME%
echo ==========================================================

call "%PYTHON%" -u run.py ^
 --is_training 1 ^
 --root_path "%DATA_ROOT%" ^
 --data_path "%DATA_FILE%" ^
 --model_id "%EXP_NAME%" ^
 --model iTransformer ^
 --data custom ^
 --features M ^
 --seq_len 96 ^
 --pred_len %PRED_LEN% ^
 --e_layers 3 ^
 --enc_in 21 ^
 --dec_in 21 ^
 --c_out 21 ^
 --des Exp ^
 --d_model 512 ^
 --d_ff 512 ^
 --batch_size 32 ^
 --learning_rate 0.0001 ^
 --train_epochs 10 ^
 --patience 3 ^
 --num_workers 0 ^
 --itr 1 ^
 --induced_attention ^
 --attn_rank %ATTN_RANK% ^
 --attn_time_tokens %TIME_TOKENS% ^
 --attn_gate_init %GATE_INIT%

set "RUN_CODE=%ERRORLEVEL%"

for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "EXP_END_TIME=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeMilliseconds()"') do set "EXP_END_MS=%%A"

for /f "delims=" %%A in ('powershell -NoProfile -Command "$d=(%EXP_END_MS%-%EXP_START_MS%)/1000; $h=[math]::Floor($d/3600); $m=[math]::Floor(($d%%3600)/60); $s=[math]::Floor($d%%60); '{0:00}:{1:00}:{2:00}' -f $h,$m,$s"') do set "EXP_DURATION=%%A"

if "%RUN_CODE%"=="0" (
    set "EXP_STATUS=COMPLETED"
) else (
    set "EXP_STATUS=FAILED"
)

echo.
echo ==========================================================
echo EXPERIMENT END
echo Model ID : %EXP_NAME%
echo Status   : %EXP_STATUS%
echo Start    : %EXP_START_TIME%
echo End      : %EXP_END_TIME%
echo Duration : %EXP_DURATION%
echo Exit code: %RUN_CODE%
echo ==========================================================

exit /b %RUN_CODE%